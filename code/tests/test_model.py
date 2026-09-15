from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from models import RestormerCore20


def tiny_model() -> RestormerCore20:
    return RestormerCore20(
        dim=4,
        num_heads=(1,),
        num_blocks=(1,),
        ffn_expansion_factor=1.5,
    )


def test_initial_x_extracts_aligned_measurement_windows() -> None:
    measurement = torch.zeros(1, 256, 422)
    measurement[0, 3, 2 * 7 + 11] = 5.0

    unfolded = RestormerCore20.initial_x(measurement)

    assert unfolded.shape == (1, 84, 256, 256)
    assert unfolded[0, 7, 3, 11].item() == 5.0


def test_tiny_restormer_has_fixed_output_and_gradients() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    measurement = torch.rand(1, 256, 422)
    model_mask = torch.ones(1, 84, 256, 256)

    prediction = model(measurement, model_mask)
    prediction.mean().backward()

    assert prediction.shape == (1, 20, 256, 256)
    assert torch.isfinite(prediction).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_model_rejects_nonprotocol_channels_and_shapes() -> None:
    with pytest.raises(ValueError, match="out_channels"):
        RestormerCore20(out_channels=32)
    model = tiny_model()
    with pytest.raises(ValueError, match="measurement"):
        model(torch.zeros(1, 256, 421), torch.zeros(1, 84, 256, 256))
    with pytest.raises(ValueError, match="model_mask"):
        model(torch.zeros(1, 256, 422), torch.zeros(1, 84, 255, 256))
