from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import scene_macro_runner
import training


class _IdentityPredictionModel(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, unused_mask: torch.Tensor) -> torch.Tensor:
        del unused_mask
        return prediction


class _MinimalRunnerV3:
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def _to_device(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return batch


class _SceneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        self.samples = [
            {
                "scene": "scene_a",
                "top": 0,
                "left": 0,
                "height": 1,
                "width": 1,
                "sample_id": "scene_a_0",
            }
        ] + [
            {
                "scene": "scene_b",
                "top": row,
                "left": 0,
                "height": 1,
                "width": 1,
                "sample_id": f"scene_b_{row}",
            }
            for row in range(3)
        ]
        self.errors = [0.0, 2.0, 2.0, 2.0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = torch.full((20, 1, 1), self.errors[index], dtype=torch.float32)
        unused_mask = torch.zeros((1,), dtype=torch.float32)
        target = torch.zeros((20, 1, 1), dtype=torch.float32)
        validity = torch.ones((20, 1, 1), dtype=torch.float32)
        return prediction, unused_mask, torch.cat([target, validity], dim=0)


def test_scene_macro_runner_weights_scenes_equally_across_batch_boundary() -> None:
    SceneMacroRunnerV3 = scene_macro_runner.create_scene_macro_runner_class(
        _MinimalRunnerV3
    )
    runner = SceneMacroRunnerV3(
        _IdentityPredictionModel(),
        training.MaskedBalancedCharbonnierLoss(epsilon=1e-6),
    )
    loader = DataLoader(_SceneDataset(), batch_size=2, shuffle=False)

    dev_loss, macro_zmae = runner._eval(loader)

    assert dev_loss == pytest.approx(1.5, rel=1e-6)
    assert macro_zmae == pytest.approx(1.0, rel=1e-6)
    assert macro_zmae != pytest.approx(1.5)


def test_overlap_is_uniformly_fused_before_zmae() -> None:
    accumulator = scene_macro_runner._SceneErrorAccumulator("scene", 1, 1)
    sample = {
        "scene": "scene",
        "top": 0,
        "left": 0,
        "height": 1,
        "width": 1,
        "sample_id": "sample",
    }
    validity = torch.ones((20, 1, 1), dtype=torch.float32)
    target = torch.zeros((20, 1, 1), dtype=torch.float32)
    supervision = torch.cat([target, validity], dim=0)
    accumulator.add(sample, torch.zeros((20, 1, 1)), supervision)
    accumulator.add(sample, torch.full((20, 1, 1), 2.0), supervision)

    assert accumulator.macro_zmae() == pytest.approx(1.0)


def test_overlap_cannot_change_fixed_label_validity() -> None:
    accumulator = scene_macro_runner._SceneErrorAccumulator("scene", 1, 1)
    sample = {
        "scene": "scene",
        "top": 0,
        "left": 0,
        "height": 1,
        "width": 1,
        "sample_id": "sample",
    }
    target = torch.zeros((20, 1, 1), dtype=torch.float32)
    prediction = torch.zeros((20, 1, 1), dtype=torch.float32)
    valid = torch.ones((20, 1, 1), dtype=torch.float32)
    invalid = torch.zeros((20, 1, 1), dtype=torch.float32)
    accumulator.add(sample, prediction, torch.cat([target, valid], dim=0))

    with pytest.raises(ValueError, match="fixed label validity"):
        accumulator.add(sample, prediction, torch.cat([target, invalid], dim=0))
