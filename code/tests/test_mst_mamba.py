from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import MSTMambaCore20
from models.mst_mamba_core20.blocks import MaskAwareMambaMixer
from models.mst_mamba_core20.pscan import pscan
import train
import training


def tiny_model() -> MSTMambaCore20:
    return MSTMambaCore20(
        dim=4,
        stage=1,
        num_blocks=(1,),
        d_state=2,
        expand_factor=1,
        d_conv=2,
        stage_reductions=(8, 4, 8),
    )


def test_parallel_scan_matches_direct_recurrence() -> None:
    torch.manual_seed(5)
    coefficients = torch.sigmoid(torch.randn(2, 7, 3, 2))
    inputs = torch.randn(2, 7, 3, 2)
    expected_steps = []
    state = torch.zeros_like(inputs[:, 0])
    for index in range(inputs.shape[1]):
        state = coefficients[:, index] * state + inputs[:, index]
        expected_steps.append(state)
    expected = torch.stack(expected_steps, dim=1)

    actual = pscan(coefficients, inputs)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mst_mamba_core20_forward_gradients_and_alignment() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    measurement = torch.zeros(1, 256, 422)
    measurement[0, 3, 2 * 7 + 11] = 5.0
    unfolded = model.initial_x(measurement)
    assert unfolded.shape == (1, 84, 256, 256)
    assert unfolded[0, 7, 3, 11].item() == 5.0

    prediction = model(measurement, torch.ones(1, 84, 256, 256))
    prediction.square().mean().backward()

    assert prediction.shape == (1, 20, 256, 256)
    assert torch.isfinite(prediction).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_mask_attention_keeps_distinct_batch_masks() -> None:
    mixer = MaskAwareMambaMixer(
        dim=2,
        d_state=2,
        expand_factor=1,
        d_conv=2,
        spatial_reduction=1,
    )
    with torch.no_grad():
        for parameter in mixer.mask_encoder.parameters():
            parameter.fill_(0.1)
    masks = torch.stack(
        [torch.zeros(2, 16, 16), torch.ones(2, 16, 16)],
        dim=0,
    )

    attention = mixer._build_mask_attn(masks, batch_size=2, h=16, w=16, c=2)

    assert attention is not None
    assert attention.shape == (2, 256, 2)
    assert not torch.equal(attention[0], attention[1])


def test_formal_mst_config_is_strict_and_builds_model(tmp_path: Path) -> None:
    path = REPO_ROOT / "configs/formal_mst_mamba_seed42.json"
    config = training.load_resolved_config(path)
    model = train.build_model(config)

    assert config["experiment_id"] == "formal_mst_mamba_core20_seed42_v1"
    assert config["model"]["name"] == "MSTMambaCore20"
    assert config["data"]["batch_size"] == 1
    assert isinstance(model, MSTMambaCore20)

    payload = path.read_text(encoding="utf-8").replace(
        '"out_channels": 20', '"out_channels": 32'
    )
    bad_path = tmp_path / "bad-mst.json"
    bad_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="84 input and 20 output"):
        training.load_resolved_config(bad_path)


def test_mst_mamba_rejects_nonprotocol_shapes() -> None:
    with pytest.raises(ValueError, match="out_channels"):
        MSTMambaCore20(out_channels=32)
    model = tiny_model()
    with pytest.raises(ValueError, match="measurement"):
        model(torch.zeros(1, 256, 421), torch.zeros(1, 84, 256, 256))
    with pytest.raises(ValueError, match="model_mask"):
        model(torch.zeros(1, 256, 422), torch.zeros(1, 84, 255, 256))
