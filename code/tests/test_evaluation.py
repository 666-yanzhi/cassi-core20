from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluation
import evaluate


def test_index_metrics_identity_and_negative_r2_are_preserved() -> None:
    target = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    mask = np.ones(3, dtype=np.bool_)
    identity = evaluation.compute_index_metrics(target, target, mask)
    assert identity["MAE"] == 0.0
    assert identity["RMSE"] == 0.0
    assert identity["Bias"] == 0.0
    assert identity["R2"] == 1.0

    bad = evaluation.compute_index_metrics(
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32), target, mask
    )
    assert bad["R2"] is not None
    assert bad["R2"] < 0.0


def test_r2_domain_is_explicit() -> None:
    one = evaluation.compute_index_metrics(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.ones(1, dtype=np.bool_),
    )
    assert one["R2"] is None
    assert one["r2_exclusion_reason"] == "fewer_than_two_valid_pixels"

    constant = evaluation.compute_index_metrics(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([2.0, 2.0], dtype=np.float32),
        np.ones(2, dtype=np.bool_),
    )
    assert constant["R2"] is None
    assert constant["r2_exclusion_reason"] == "truth_variance_not_greater_than_1e-12"


def test_nonfinite_prediction_only_fails_inside_fixed_domain() -> None:
    target = np.asarray([0.0, 1.0], dtype=np.float32)
    prediction = np.asarray([np.nan, 1.0], dtype=np.float32)
    outside_mask = np.asarray([False, True], dtype=np.bool_)
    result = evaluation.compute_index_metrics(prediction, target, outside_mask)
    assert result["MAE"] == 0.0

    with pytest.raises(FloatingPointError, match="inside evaluation domain"):
        evaluation.compute_index_metrics(
            prediction, target, np.asarray([True, True], dtype=np.bool_)
        )


def test_scene_macro_weights_scenes_not_pixels() -> None:
    def scene(scene_id: str, mae: float, pixels: int) -> dict[str, object]:
        per_index = []
        for channel, name in enumerate(evaluation.INDEX_NAMES):
            per_index.append(
                {
                    "index": name,
                    "channel": channel,
                    "valid_pixel_count": pixels,
                    "MAE": mae,
                    "RMSE": mae,
                    "Bias": mae,
                    "R2": -mae,
                    "r2_exclusion_reason": None,
                }
            )
        return {
            "scene": scene_id,
            "scene_macro": {"MAE": mae, "RMSE": mae, "Bias": mae, "R2": -mae},
            "per_index": per_index,
        }

    aggregate = evaluation.aggregate_scene_results(
        [scene("small", 1.0, 1), scene("large", 3.0, 1_000_000)]
    )
    assert aggregate["dataset_scene_macro"]["MAE"] == 2.0
    assert aggregate["per_index"][0]["MAE"] == 2.0


def test_scene_evaluation_combines_all_three_masks() -> None:
    target = np.zeros((2, 2, 20), dtype=np.float32)
    prediction = np.ones_like(target)
    label_mask = np.ones_like(target, dtype=np.bool_)
    label_mask[0, 0, 0] = False
    crop = np.asarray([[True, True], [False, True]], dtype=np.bool_)
    nonpadding = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    result = evaluation.evaluate_scene(
        "scene", prediction, target, label_mask, crop, nonpadding
    )
    assert result["coverage_pixel_count"] == 2
    assert result["per_index"][0]["valid_pixel_count"] == 1
    assert result["per_index"][1]["valid_pixel_count"] == 2


def test_formal_test_guard_requires_split_identity_and_exact_counts() -> None:
    config = {
        "data": {
            "split_id": "local_minimum_2_1_0",
            "train_scenes": ["train"],
            "val_scenes": ["validation"],
            "test_scenes": ["test"],
        }
    }
    with pytest.raises(ValueError, match="exact 202/15/35"):
        evaluate.validate_formal_test_split(config, {})

    config["data"] = {
        "split_id": evaluate.FORMAL_SPLIT_ID,
        "train_scenes": [f"train-{index}" for index in range(202)],
        "val_scenes": [f"validation-{index}" for index in range(15)],
        "test_scenes": [f"test-{index}" for index in range(35)],
    }
    with pytest.raises(ValueError, match="manifest formal split identity"):
        evaluate.validate_formal_test_split(config, {})
    evaluate.validate_formal_test_split(
        config,
        {
            "split_id": evaluate.FORMAL_SPLIT_ID,
            "split": {
                "train_scenes": config["data"]["train_scenes"],
                "val_scenes": config["data"]["val_scenes"],
                "test_scenes": config["data"]["test_scenes"],
            },
        },
    )
