"""MST-Mamba adapted to the isolated 20-channel Core20 protocol.

The migration retains the legacy mask-aware selective-state-space blocks,
U-Net encoder/decoder, learned 84-channel mask downsampling, and formal default
hyperparameters. The output head is rebuilt for ``reviewed_v3_core20``.

Legacy sources inspected during migration:

* ``MST_Mamba.py`` SHA-256
  ``574e78ae5d316042c9c31209a08d3c14cfe82070523fb3bcbea648ec05e11c77``
* ``MST_Mamba_VegIdx.py`` SHA-256
  ``6260a8cc990771742b4fbb699bbe79f8a62db1bf4bbf7ab2341654b76ce80d29``

The legacy implementation broadcast the first mask across a whole batch. This
port preserves a shared-mask fast path but also handles genuinely per-sample
masks without silently replacing them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..restormer_core20 import (
    DISPERSION_STEP,
    EXPECTED_BANDS,
    MEASUREMENT_WIDTH,
    OUTPUT_CHANNELS,
    PATCH_SIZE,
)
from .blocks import FeedForward, MaskAwareMambaMixer, PreNorm, validate_stage_reductions


class MaskAwareStateSpaceBlock(nn.Module):
    """Project the 84-channel aperture and apply residual Mamba/FFN blocks."""

    def __init__(
        self,
        channels: int,
        *,
        block_count: int,
        d_state: int,
        expand_factor: int,
        d_conv: int,
        dt_rank: str | int,
        pscan_parallel: bool,
        mask_fusion: str,
        spatial_reduction: int,
    ) -> None:
        super().__init__()
        if block_count <= 0:
            raise ValueError("block_count must be positive")
        self.mask_projection = nn.Conv2d(EXPECTED_BANDS, channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    MaskAwareMambaMixer(
                        dim=channels,
                        d_state=d_state,
                        expand_factor=expand_factor,
                        d_conv=d_conv,
                        dt_rank=dt_rank,
                        pscan_parallel=pscan_parallel,
                        mask_fusion=mask_fusion,
                        spatial_reduction=spatial_reduction,
                    ),
                    PreNorm(channels, FeedForward(dim=channels)),
                ]
            )
            for _ in range(block_count)
        )

    def forward(self, value: torch.Tensor, model_mask: torch.Tensor) -> torch.Tensor:
        if model_mask.shape[-2:] != value.shape[-2:]:
            model_mask = F.interpolate(
                model_mask,
                size=value.shape[-2:],
                mode="nearest",
            )
        projected_mask = self.mask_projection(model_mask)
        value = value.permute(0, 2, 3, 1)
        for mixer, feed_forward in self.blocks:
            value = value + mixer(value, mask=projected_mask)
            value = value + feed_forward(value)
        return value.permute(0, 3, 1, 2).contiguous()


class MSTMambaCore20(nn.Module):
    """Mask-conditioned MST-Mamba with an invariant 20-channel output."""

    VALID_MASK_FUSIONS = {
        "input_mul",
        "input_add",
        "x_branch",
        "z_branch",
        "delta_bias",
        "output",
        "none",
    }

    def __init__(
        self,
        *,
        dim: int = 64,
        stage: int = 3,
        num_blocks: Sequence[int] = (2, 2, 2),
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        dt_rank: str | int = "auto",
        pscan_parallel: bool = True,
        mask_fusion: str = "input_mul",
        stage_reductions: Sequence[int] | None = None,
        in_channels: int = EXPECTED_BANDS,
        out_channels: int = OUTPUT_CHANNELS,
    ) -> None:
        super().__init__()
        if in_channels != EXPECTED_BANDS:
            raise ValueError(f"in_channels must be {EXPECTED_BANDS}")
        if out_channels != OUTPUT_CHANNELS:
            raise ValueError(f"out_channels must be {OUTPUT_CHANNELS}")
        if dim <= 0 or stage <= 0:
            raise ValueError("dim and stage must be positive")
        if len(num_blocks) != stage or any(value <= 0 for value in num_blocks):
            raise ValueError("num_blocks must contain one positive value per stage")
        if d_state <= 0 or expand_factor <= 0 or d_conv <= 0:
            raise ValueError("Mamba state, expansion, and convolution sizes must be positive")
        if dt_rank != "auto" and (not isinstance(dt_rank, int) or dt_rank <= 0):
            raise ValueError("dt_rank must be 'auto' or a positive integer")
        if not isinstance(pscan_parallel, bool):
            raise TypeError("pscan_parallel must be boolean")
        if mask_fusion not in self.VALID_MASK_FUSIONS:
            raise ValueError(f"unsupported mask_fusion: {mask_fusion}")

        self.dim = dim
        self.stage = stage
        reductions = validate_stage_reductions(stage, stage_reductions)
        common = {
            "d_state": d_state,
            "expand_factor": expand_factor,
            "d_conv": d_conv,
            "dt_rank": dt_rank,
            "pscan_parallel": pscan_parallel,
            "mask_fusion": mask_fusion,
        }
        self.embedding = nn.Conv2d(EXPECTED_BANDS, dim, 3, padding=1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.encoders = nn.ModuleList()
        channels = dim
        for index in range(stage):
            self.encoders.append(
                nn.ModuleList(
                    [
                        MaskAwareStateSpaceBlock(
                            channels,
                            block_count=num_blocks[index],
                            spatial_reduction=reductions[index],
                            **common,
                        ),
                        nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1, bias=False),
                        nn.Conv2d(
                            EXPECTED_BANDS,
                            EXPECTED_BANDS,
                            4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
            )
            channels *= 2

        self.bottleneck = MaskAwareStateSpaceBlock(
            channels,
            block_count=num_blocks[-1],
            spatial_reduction=reductions[stage],
            **common,
        )

        self.decoders = nn.ModuleList()
        for index in range(stage):
            encoder_index = stage - index - 1
            next_channels = channels // 2
            self.decoders.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(channels, next_channels, 2, stride=2),
                        nn.Conv2d(channels, next_channels, 1, bias=False),
                        MaskAwareStateSpaceBlock(
                            next_channels,
                            block_count=num_blocks[encoder_index],
                            spatial_reduction=reductions[stage + 1 + index],
                            **common,
                        ),
                    ]
                )
            )
            channels = next_channels

        self.mapping = nn.Conv2d(dim, OUTPUT_CHANNELS, 3, padding=1, bias=False)

    @staticmethod
    def initial_x(measurement: torch.Tensor) -> torch.Tensor:
        if measurement.ndim != 3 or tuple(measurement.shape[1:]) != (
            PATCH_SIZE,
            MEASUREMENT_WIDTH,
        ):
            raise ValueError(
                f"measurement must have shape [B,{PATCH_SIZE},{MEASUREMENT_WIDTH}], "
                f"got {tuple(measurement.shape)}"
            )
        return torch.stack(
            [
                measurement[
                    :,
                    :,
                    channel * DISPERSION_STEP : channel * DISPERSION_STEP + PATCH_SIZE,
                ]
                for channel in range(EXPECTED_BANDS)
            ],
            dim=1,
        )

    def forward(
        self,
        measurement: torch.Tensor,
        model_mask: torch.Tensor,
    ) -> torch.Tensor:
        if model_mask.ndim != 4 or tuple(model_mask.shape[1:]) != (
            EXPECTED_BANDS,
            PATCH_SIZE,
            PATCH_SIZE,
        ):
            raise ValueError(
                f"model_mask must have shape [B,84,256,256], got {tuple(model_mask.shape)}"
            )
        if model_mask.shape[0] not in {1, measurement.shape[0]}:
            raise ValueError("model_mask batch must be one or match measurement batch")
        value = self.activation(self.embedding(self.initial_x(measurement)))
        encoder_features: list[torch.Tensor] = []
        encoder_masks: list[torch.Tensor] = []
        mask = model_mask
        for block, downsample, mask_downsample in self.encoders:
            value = block(value, mask)
            encoder_features.append(value)
            encoder_masks.append(mask)
            value = downsample(value)
            mask = mask_downsample(mask)

        value = self.bottleneck(value, mask)
        for index, (upsample, fusion, block) in enumerate(self.decoders):
            value = upsample(value)
            skip = encoder_features[self.stage - index - 1]
            value = fusion(torch.cat([value, skip], dim=1))
            value = block(value, encoder_masks[self.stage - index - 1])
        return self.mapping(value)
