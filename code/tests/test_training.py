from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import pytest
import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import training
from models import RestormerCore20
import scene_macro_runner


def _supervision(target: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    return torch.cat([target, validity.to(target.dtype)], dim=1)


def test_masked_balanced_charbonnier_ignores_invalid_pixels() -> None:
    target = torch.zeros((1, 20, 2, 2), dtype=torch.float32)
    validity = torch.ones_like(target, dtype=torch.bool)
    validity[:, :, 0, 0] = False
    supervision = _supervision(target, validity)
    prediction = torch.zeros_like(target)
    baseline = training.MaskedBalancedCharbonnierLoss(epsilon=1e-6)(
        prediction, supervision
    )

    prediction[:, :, 0, 0] = 1e6
    changed_only_where_invalid = training.MaskedBalancedCharbonnierLoss(epsilon=1e-6)(
        prediction, supervision
    )

    assert torch.equal(baseline, changed_only_where_invalid)


def test_masked_balanced_charbonnier_balances_channels() -> None:
    target = torch.zeros((1, 20, 1, 2), dtype=torch.float32)
    validity = torch.zeros_like(target, dtype=torch.bool)
    validity[:, 0, :, :] = True
    validity[:, 1, :, 0] = True
    prediction = torch.zeros_like(target)
    prediction[:, 0, :, :] = 2.0
    prediction[:, 1, :, 0] = 4.0

    value = training.MaskedBalancedCharbonnierLoss(epsilon=1e-6)(
        prediction, _supervision(target, validity)
    )

    assert value.item() == pytest.approx(3.0, rel=1e-6)


def test_loss_rejects_empty_mask_and_nonfinite_prediction() -> None:
    target = torch.zeros((1, 20, 1, 1), dtype=torch.float32)
    validity = torch.zeros_like(target, dtype=torch.bool)
    loss = training.MaskedBalancedCharbonnierLoss()
    with pytest.raises(ValueError, match="no valid supervision"):
        loss(target, _supervision(target, validity))

    validity[:] = True
    prediction = target.clone()
    prediction[:, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        loss(prediction, _supervision(target, validity))


def test_config_is_strict_and_resolves_paths(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs/local_smoke.json"
    resolved = training.load_resolved_config(config_path)
    assert resolved["data"]["split_id"] == "local_minimum_2_1_0"
    assert resolved["data"]["train_scenes"] == ["hsi_0001", "hsi_0002"]
    assert resolved["data"]["val_scenes"] == ["hsi_0003"]
    assert resolved["data"]["test_scenes"] == []
    assert resolved["data"]["val_batch_size"] == 1
    assert Path(resolved["data"]["manifest"]).is_absolute()
    assert resolved["source_config_sha256"] == training.sha256_file(config_path)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        training.load_resolved_config(bad_path)


def test_training_state_round_trip_and_signature_guard(tmp_path: Path) -> None:
    torch.manual_seed(7)
    random.seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    generator = torch.Generator().manual_seed(9)
    loss = model(torch.ones((1, 3))).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "state.pt"
    expected_parameters = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    training.save_training_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epochs=1,
        history={"train_loss": [float(loss.item())]},
        experiment_signature="signature-a",
        dataloader_generator=generator,
    )

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=2
    )
    restored_generator = torch.Generator()
    state = training.restore_training_state(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_signature="signature-a",
        dataloader_generator=restored_generator,
    )
    assert state["completed_epochs"] == 1
    for key, expected in expected_parameters.items():
        assert torch.equal(restored_model.state_dict()[key], expected)
    assert torch.equal(restored_generator.get_state(), generator.get_state())

    with pytest.raises(ValueError, match="signature mismatch"):
        training.restore_training_state(
            path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            expected_signature="signature-b",
            dataloader_generator=restored_generator,
        )


@pytest.mark.skipif(
    not (
        REPO_ROOT
        / "data/derived/measurements/cassi_forward_v1/local_minimum/manifest.json"
    ).is_file(),
    reason="local minimum experiment data are not present",
)
def test_real_minimum_data_uses_train_only_normalization(tmp_path: Path) -> None:
    manifest_path = (
        REPO_ROOT
        / "data/derived/measurements/cassi_forward_v1/local_minimum/manifest.json"
    )
    normalization_path = tmp_path / "normalization.json"
    payload = training.fit_normalization(
        manifest_path,
        ["hsi_0001", "hsi_0002"],
        normalization_path,
        epsilon=1e-6,
    )

    assert payload["fit_role"] == "train_only"
    assert payload["train_scenes"] == ["hsi_0001", "hsi_0002"]
    assert all("hsi_0003" not in item for item in payload["train_sample_ids"])
    assert payload["measurement"]["count"] == 4 * 256 * 422
    assert len(payload["target"]["count_by_channel"]) == 20

    dataset = training.Core20Dataset(
        manifest_path,
        normalization_path,
        ["hsi_0003"],
    )
    measurement, model_mask, supervision = dataset[0]
    assert measurement.shape == (256, 422)
    assert model_mask.shape == (84, 256, 256)
    assert supervision.shape == (40, 256, 256)
    assert torch.isfinite(measurement).all()
    assert torch.isfinite(supervision).all()
    assert set(torch.unique(supervision[20:]).tolist()) <= {0.0, 1.0}


@pytest.mark.skipif(
    not (
        REPO_ROOT
        / "data/derived/measurements/cassi_forward_v1/local_minimum/manifest.json"
    ).is_file(),
    reason="local minimum experiment data are not present",
)
def test_real_minimum_training_is_deterministic(tmp_path: Path) -> None:
    config = training.load_resolved_config(REPO_ROOT / "configs/local_smoke.json")
    manifest_path = Path(config["data"]["manifest"])
    normalization_path = tmp_path / "determinism-normalization.json"
    training.fit_normalization(
        manifest_path,
        config["data"]["train_scenes"],
        normalization_path,
        epsilon=config["normalization"]["epsilon"],
    )
    RunnerV3, _, _ = training.load_runner_v3(Path(config["runner"]["runner_root"]))
    SceneMacroRunnerV3 = scene_macro_runner.create_scene_macro_runner_class(RunnerV3)

    def run_once(run_number: int) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
        seed = config["runner"]["seed"]
        torch.manual_seed(seed)
        dataset = training.Core20Dataset(
            manifest_path,
            normalization_path,
            config["data"]["train_scenes"],
        )
        loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        model = RestormerCore20(
            dim=config["model"]["dim"],
            num_heads=tuple(config["model"]["num_heads"]),
            num_blocks=tuple(config["model"]["num_blocks"]),
            ffn_expansion_factor=config["model"]["ffn_expansion_factor"],
            bias=config["model"]["bias"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        runner = SceneMacroRunnerV3(
            model,
            optimizer,
            training.MaskedBalancedCharbonnierLoss(config["loss"]["epsilon"]),
            device="cpu",
        )
        runner.fit(
            loader,
            num_epochs=1,
            best_path=tmp_path / f"best-{run_number}.pt",
            grad_clip_norm=config["runner"]["gradient_clip_norm"],
            seed=seed,
        )
        return runner.history, {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    first_history, first_state = run_once(1)
    second_history, second_state = run_once(2)
    assert first_history == second_history
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
