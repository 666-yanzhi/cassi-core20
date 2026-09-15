import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from .mamba_core import MambaConfig, RMSNorm, pscan


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row

    outputs = inputs.new_zeros(bs, nC, row, out_col)
    for i in range(nC):
        s = int(step * i)
        end = min(s + out_col, col)
        width = max(end - s, 0)
        if width > 0:
            outputs[:, i, :, :width] = inputs[:, i, :, s:end]
    return outputs


class MaskGuidedMechanism(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat
        )

    def forward(self, mask_shift):
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)
        return mask_emb


class MaskAwareMambaMixer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        expand_factor=2,
        d_conv=4,
        dt_rank="auto",
        pscan_parallel=True,
        mask_fusion="input_mul",
        spatial_reduction=1,
    ):
        super().__init__()
        valid_fusions = {
            "input_mul",
            "input_add",
            "x_branch",
            "z_branch",
            "delta_bias",
            "output",
            "none",
        }
        if mask_fusion not in valid_fusions:
            raise ValueError(f"Unsupported mask_fusion={mask_fusion!r}. Choose from {sorted(valid_fusions)}")

        self.dim = dim
        self.mask_fusion = mask_fusion
        self.spatial_reduction = max(int(spatial_reduction), 1)
        self.mask_encoder = MaskGuidedMechanism(dim)
        self.seq_norm = RMSNorm(dim)

        config = MambaConfig(
            d_model=dim,
            n_layers=1,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_rank=dt_rank,
            pscan=pscan_parallel,
        )
        self.config = config

        self.in_proj = nn.Linear(dim, 2 * config.d_inner, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
            groups=config.d_inner,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        self.out_proj = nn.Linear(config.d_inner, dim, bias=config.bias)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        dt_init_std = (config.dt_rank ** -0.5) * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        with torch.no_grad():
            dt = torch.exp(
                torch.rand(config.d_inner)
                * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            ).clamp(min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        if mask_fusion in {"x_branch", "z_branch"}:
            self.mask_inner_proj = nn.Linear(dim, config.d_inner, bias=False)
        else:
            self.mask_inner_proj = None

        if mask_fusion == "delta_bias":
            self.mask_delta_proj = nn.Linear(dim, config.d_inner, bias=False)
        else:
            self.mask_delta_proj = None

        if mask_fusion == "input_add":
            self.mask_input_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.mask_input_proj = None

    def _build_mask_attn(self, mask, batch_size, h, w, c):
        if mask is None:
            return None

        mask_attn = self.mask_encoder(mask).permute(0, 2, 3, 1)
        if mask_attn.shape[1] != h or mask_attn.shape[2] != w:
            mask_attn = F.interpolate(
                mask_attn.permute(0, 3, 1, 2),
                size=(h, w),
                mode="nearest",
            ).permute(0, 2, 3, 1)

        if mask_attn.shape[0] == 1 and batch_size > 1:
            mask_attn = mask_attn.expand(batch_size, h, w, mask_attn.shape[-1])
        elif mask_attn.shape[0] != batch_size:
            raise ValueError(
                f"Mask batch mismatch: expected 1 or {batch_size}, got {mask_attn.shape[0]}"
            )

        if mask_attn.shape[-1] != c:
            raise ValueError(
                f"Mask channel mismatch after stage alignment: expected {c}, got {mask_attn.shape[-1]}"
            )

        return mask_attn.reshape(batch_size, h * w, c)

    def forward(self, x_in, mask=None):
        b, h, w, c = x_in.shape
        x_2d = x_in.permute(0, 3, 1, 2).contiguous()

        if self.spatial_reduction > 1:
            x_work = F.avg_pool2d(
                x_2d,
                kernel_size=self.spatial_reduction,
                stride=self.spatial_reduction,
            )
        else:
            x_work = x_2d

        _, _, hr, wr = x_work.shape
        seq = x_work.permute(0, 2, 3, 1).reshape(b, hr * wr, c)
        seq = self.seq_norm(seq)

        mask_seq = self._build_mask_attn(mask, b, hr, wr, c)
        if mask_seq is not None and self.mask_fusion == "input_mul":
            seq = seq * mask_seq
        elif mask_seq is not None and self.mask_fusion == "input_add":
            seq = seq + self.mask_input_proj(mask_seq)

        xz = self.in_proj(seq)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, : hr * wr]
        x_branch = x_branch.transpose(1, 2).contiguous()
        x_branch = F.silu(x_branch)

        if mask_seq is not None and self.mask_fusion == "x_branch":
            x_branch = x_branch * self.mask_inner_proj(mask_seq)

        y = self._ssm(x_branch, mask_seq=mask_seq)
        z = F.silu(z_branch)

        if mask_seq is not None and self.mask_fusion == "z_branch":
            z = z * self.mask_inner_proj(mask_seq)

        out = y * z
        out = self.out_proj(out)

        if mask_seq is not None and self.mask_fusion == "output":
            out = out * mask_seq

        out = out.reshape(b, hr, wr, c).permute(0, 3, 1, 2).contiguous()
        out = out + self.pos_emb(x_work)

        if self.spatial_reduction > 1:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out.permute(0, 2, 3, 1).contiguous()

    def _ssm(self, x_ed, mask_seq=None):
        dtype = x_ed.dtype
        A = -torch.exp(self.A_log.to(dtype))
        D_ = self.D.to(dtype)

        deltaBC = self.x_proj(x_ed)
        delta, B_, C_ = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        dt_w = self.dt_proj.weight.to(dtype)
        dt_b = self.dt_proj.bias.to(dtype)
        delta = dt_w @ delta.transpose(1, 2)
        delta = delta.transpose(1, 2).contiguous()

        if mask_seq is not None and self.mask_fusion == "delta_bias":
            delta = delta + self.mask_delta_proj(mask_seq).to(dtype)

        delta = F.softplus(delta + dt_b)

        if self.config.pscan:
            deltaA = torch.exp(delta.unsqueeze(-1) * A)
            deltaB = delta.unsqueeze(-1) * B_.unsqueeze(2)
            BX = deltaB * x_ed.unsqueeze(-1)
            hs = pscan(deltaA, BX)
            y = (hs @ C_.unsqueeze(-1)).squeeze(-1)
        else:
            bsz, seq_len, ed = delta.shape
            n = self.config.d_state
            h = torch.zeros(bsz, ed, n, device=x_ed.device, dtype=dtype)
            y_list = []
            c_view = C_.unsqueeze(-1)
            for t in range(seq_len):
                delta_t = delta[:, t]
                B_t = B_[:, t]
                x_t = x_ed[:, t]
                deltaA_t = torch.exp(delta_t.unsqueeze(-1) * A)
                deltaB_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
                BX_t = deltaB_t * x_t.unsqueeze(-1)
                h = deltaA_t * h + BX_t
                y_t = (h @ c_view[:, t]).squeeze(-1)
                y_list.append(y_t)
            y = torch.stack(y_list, dim=1)

        y = y + D_ * x_ed
        return y


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSMB(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=2,
        d_state=16,
        expand_factor=2,
        d_conv=4,
        dt_rank="auto",
        pscan_parallel=True,
        mask_fusion="input_mul",
        spatial_reduction=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MaskAwareMambaMixer(
                            dim=dim,
                            d_state=d_state,
                            expand_factor=expand_factor,
                            d_conv=d_conv,
                            dt_rank=dt_rank,
                            pscan_parallel=pscan_parallel,
                            mask_fusion=mask_fusion,
                            spatial_reduction=spatial_reduction,
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, mask):
        x = x.permute(0, 2, 3, 1)
        for mixer, ff in self.blocks:
            x = mixer(x, mask=mask) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2).contiguous()


def validate_stage_reductions(stage, stage_reductions):
    expected = stage * 2 + 1
    if stage_reductions is None:
        if stage == 3:
            return [2, 1, 1, 1, 1, 1, 2]
        return [1] * expected
    if len(stage_reductions) != expected:
        raise ValueError(
            f"stage_reductions must have length {expected} for stage={stage}, got {len(stage_reductions)}."
        )
    return [max(int(v), 1) for v in stage_reductions]
