"""Restormer adapted to direct 20-channel vegetation-index prediction.

The model consumes a normalized CASSI measurement ``[B,256,422]`` and the
unshifted repeated aperture ``[B,84,256,256]``. ``initial_x`` only unfolds
measurement windows into aligned features; it is not an HSI reconstruction.

Adapted from the server legacy project's ``Restormer_VegIdx.py``. The exact
source inspected during migration had SHA-256
``d3063679b88084015e6941c3705f40ebf5a42caf1bd96b82d868d4d75a88939b``.
Its encoder, learned mask downsampling, mask-guided MDTA, GDFN, decoder, and
default hyperparameters are retained. The output head is deliberately rebuilt
for the isolated 20-channel ``reviewed_v3_core20`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


EXPECTED_BANDS = 84
OUTPUT_CHANNELS = 20
PATCH_SIZE = 256
DISPERSION_STEP = 2
MEASUREMENT_WIDTH = PATCH_SIZE + (EXPECTED_BANDS - 1) * DISPERSION_STEP


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MaskMDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention with mask-guided value gating."""

    def __init__(self, channels: int, heads: int, *, bias: bool = False) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.mask_projection = nn.Sequential(
            nn.Conv2d(EXPECTED_BANDS, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, value: torch.Tensor, model_mask: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        q, k, v = self.qkv_depthwise(self.qkv(value)).chunk(3, dim=1)
        if model_mask.shape[-2:] != (height, width):
            model_mask = F.interpolate(model_mask, size=(height, width), mode="nearest")
        v = v * self.mask_projection(model_mask)

        q = q.reshape(batch, self.heads, -1, height * width)
        k = k.reshape(batch, self.heads, -1, height * width)
        v = v.reshape(batch, self.heads, -1, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        result = (attention @ v).reshape(batch, channels, height, width)
        return self.output_projection(result)


class GDFN(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion_factor: float,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(channels * expansion_factor)
        if hidden <= 0:
            raise ValueError("GDFN hidden channels must be positive")
        self.input_projection = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.output_projection = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = self.depthwise(self.input_projection(value)).chunk(2, dim=1)
        return self.output_projection(F.gelu(first) * second)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        expansion_factor: float,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = MaskMDTA(channels, heads, bias=bias)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = GDFN(channels, expansion_factor, bias=bias)

    def forward(self, value: torch.Tensor, model_mask: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), model_mask)
        return value + self.feed_forward(self.norm2(value))


class RestormerCore20(nn.Module):
    """Mask-conditioned Restormer with an invariant 20-channel output."""

    def __init__(
        self,
        *,
        dim: int = 48,
        num_heads: Sequence[int] = (1, 2, 4, 8),
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        ffn_expansion_factor: float = 2.66,
        in_channels: int = EXPECTED_BANDS,
        out_channels: int = OUTPUT_CHANNELS,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != EXPECTED_BANDS:
            raise ValueError(f"in_channels must be {EXPECTED_BANDS}")
        if out_channels != OUTPUT_CHANNELS:
            raise ValueError(f"out_channels must be {OUTPUT_CHANNELS}")
        if dim <= 0 or len(num_blocks) == 0 or len(num_heads) != len(num_blocks):
            raise ValueError("dim must be positive and head/block lists must have equal nonzero length")
        if any(blocks <= 0 for blocks in num_blocks):
            raise ValueError("every Restormer stage must contain at least one block")

        self.stage_count = len(num_blocks) - 1
        self.dim = dim
        self.embedding = nn.Conv2d(EXPECTED_BANDS, dim, 3, padding=1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.encoders = nn.ModuleList()
        channels = dim
        for stage in range(self.stage_count):
            blocks = nn.ModuleList(
                TransformerBlock(
                    channels,
                    num_heads[stage],
                    ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks[stage])
            )
            self.encoders.append(
                nn.ModuleList(
                    [
                        blocks,
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

        self.bottleneck = nn.ModuleList(
            TransformerBlock(
                channels,
                num_heads[-1],
                ffn_expansion_factor,
                bias=bias,
            )
            for _ in range(num_blocks[-1])
        )

        self.decoders = nn.ModuleList()
        for decoder_index in range(self.stage_count):
            encoder_index = self.stage_count - decoder_index - 1
            next_channels = channels // 2
            blocks = nn.ModuleList(
                TransformerBlock(
                    next_channels,
                    num_heads[encoder_index],
                    ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks[encoder_index])
            )
            self.decoders.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(channels, next_channels, 2, stride=2),
                        nn.Conv2d(channels, next_channels, 1, bias=False),
                        blocks,
                    ]
                )
            )
            channels = next_channels

        self.mapping = nn.Conv2d(dim, OUTPUT_CHANNELS, 3, padding=1, bias=False)

    @staticmethod
    def initial_x(measurement: torch.Tensor) -> torch.Tensor:
        """Unfold aligned measurement windows; this is not reconstructed HSI."""

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
        if model_mask.shape[0] != measurement.shape[0]:
            raise ValueError("measurement and model_mask batch sizes differ")
        if measurement.dtype != model_mask.dtype:
            raise TypeError("measurement and model_mask dtypes differ")

        features = self.activation(self.embedding(self.initial_x(measurement)))
        encoder_features: list[torch.Tensor] = []
        encoder_masks: list[torch.Tensor] = []
        current_mask = model_mask
        for blocks, downsample, mask_downsample in self.encoders:
            for block in blocks:
                features = block(features, current_mask)
            encoder_features.append(features)
            encoder_masks.append(current_mask)
            features = downsample(features)
            current_mask = mask_downsample(current_mask)

        for block in self.bottleneck:
            features = block(features, current_mask)

        for index, (upsample, fusion, blocks) in enumerate(self.decoders):
            features = upsample(features)
            skip_index = self.stage_count - index - 1
            features = fusion(torch.cat([features, encoder_features[skip_index]], dim=1))
            current_mask = encoder_masks[skip_index]
            for block in blocks:
                features = block(features, current_mask)
        return self.mapping(features)
