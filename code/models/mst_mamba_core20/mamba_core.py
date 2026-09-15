"""
mamba_core.py — Mamba SSM dependencies for MST_Mamba / DHM models.

Provides:
  - MambaConfig: dataclass with all Mamba hyperparameters
  - RMSNorm: Root Mean Square Layer Normalization
  - pscan: re-exported from pscan.py (Blelloch parallel scan)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .pscan import pscan  # Blelloch parallel scan (PScan.apply)


@dataclass
class MambaConfig:
    d_model: int = 28
    n_layers: int = 1
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    dt_rank: object = "auto"      # "auto" or int
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"       # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True            # True: parallel scan, False: sequential

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute in fp32: eps=1e-5 is subnormal in fp16 and underflows to 0,
        # causing x/rms = NaN when activations are near zero.
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_fp32 / rms * self.weight.float()).to(x.dtype)


__all__ = ["MambaConfig", "RMSNorm", "pscan"]
