"""Scene-level original-space evaluation for the core20 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lable import INDEX_NAMES


METRICS_VERSION = "core20_metrics_v1"
R2_VARIANCE_THRESHOLD = 1e-12
METRIC_NAMES = ("MAE", "RMSE", "Bias", "R2")


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def compute_index_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    variance_threshold: float = R2_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Compute one scene/index item without changing the fixed evaluation mask."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    evaluation_mask = np.asarray(evaluation_mask)
    if prediction.shape != target.shape or prediction.shape != evaluation_mask.shape:
        raise ValueError("prediction, target, and evaluation mask shapes must match")
    if evaluation_mask.dtype != np.bool_:
        raise TypeError("evaluation mask must have bool dtype")
    count = int(evaluation_mask.sum())
    if count == 0:
        return {
            "valid_pixel_count": 0,
            "MAE": None,
            "RMSE": None,
            "Bias": None,
            "R2": None,
            "r2_exclusion_reason": "no_valid_pixels",
        }
    prediction_values = np.asarray(prediction[evaluation_mask], dtype=np.float64)
    target_values = np.asarray(target[evaluation_mask], dtype=np.float64)
    if not np.isfinite(prediction_values).all():
        raise FloatingPointError("prediction contains NaN or Inf inside evaluation domain")
    if not np.isfinite(target_values).all():
        raise FloatingPointError("target contains NaN or Inf inside evaluation domain")
    error = prediction_values - target_values
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    rmse = float(np.sqrt(np.mean(np.square(error), dtype=np.float64)))
    bias = float(np.mean(error, dtype=np.float64))
    r2: float | None = None
    reason: str | None = None
    if count < 2:
        reason = "fewer_than_two_valid_pixels"
    else:
        centered = target_values - float(np.mean(target_values, dtype=np.float64))
        denominator = float(np.sum(np.square(centered), dtype=np.float64))
        if denominator <= variance_threshold:
            reason = "truth_variance_not_greater_than_1e-12"
        else:
            numerator = float(np.sum(np.square(error), dtype=np.float64))
            r2 = float(1.0 - numerator / denominator)
    return {
        "valid_pixel_count": count,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "R2": r2,
        "r2_exclusion_reason": reason,
    }


def evaluate_scene(
    scene_id: str,
    prediction_hwc: np.ndarray,
    target_hwc: np.ndarray,
    label_valid_mask_hwc: np.ndarray,
    crop_coverage_mask_hw: np.ndarray,
    nonpadding_mask_hw: np.ndarray,
    *,
    variance_threshold: float = R2_VARIANCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one stitched scene and average its available indices equally."""

    prediction_hwc = np.asarray(prediction_hwc)
    target_hwc = np.asarray(target_hwc)
    label_valid_mask_hwc = np.asarray(label_valid_mask_hwc)
    crop_coverage_mask_hw = np.asarray(crop_coverage_mask_hw)
    nonpadding_mask_hw = np.asarray(nonpadding_mask_hw)
    if prediction_hwc.shape != target_hwc.shape or prediction_hwc.shape[-1:] != (20,):
        raise ValueError("prediction and target must share HWC [H,W,20] shape")
    if label_valid_mask_hwc.shape != target_hwc.shape:
        raise ValueError("label validity shape mismatch")
    if label_valid_mask_hwc.dtype != np.bool_:
        raise TypeError("label validity must have bool dtype")
    expected_hw = target_hwc.shape[:2]
    for name, mask in (
        ("crop coverage", crop_coverage_mask_hw),
        ("nonpadding", nonpadding_mask_hw),
    ):
        if mask.shape != expected_hw or mask.dtype != np.bool_:
            raise ValueError(f"{name} mask must be bool with shape {expected_hw}")
    fixed_spatial_domain = crop_coverage_mask_hw & nonpadding_mask_hw
    per_index: list[dict[str, Any]] = []
    for channel, index_name in enumerate(INDEX_NAMES):
        evaluation_mask = label_valid_mask_hwc[..., channel] & fixed_spatial_domain
        item = compute_index_metrics(
            prediction_hwc[..., channel],
            target_hwc[..., channel],
            evaluation_mask,
            variance_threshold=variance_threshold,
        )
        item.update({"index": index_name, "channel": channel})
        per_index.append(item)
    scene_macro = {
        metric: _mean_or_none(
            [float(item[metric]) for item in per_index if item[metric] is not None]
        )
        for metric in METRIC_NAMES
    }
    return {
        "scene": scene_id,
        "scene_macro": scene_macro,
        "defined_index_count": {
            metric: sum(item[metric] is not None for item in per_index)
            for metric in METRIC_NAMES
        },
        "coverage_pixel_count": int(fixed_spatial_domain.sum()),
        "per_index": per_index,
    }


def aggregate_scene_results(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate with scene-equal weights, never with pooled pixels."""

    if not scene_results:
        raise ValueError("at least one scene result is required")
    dataset_macro = {
        metric: _mean_or_none(
            [
                float(scene["scene_macro"][metric])
                for scene in scene_results
                if scene["scene_macro"][metric] is not None
            ]
        )
        for metric in METRIC_NAMES
    }
    per_index: list[dict[str, Any]] = []
    for channel, index_name in enumerate(INDEX_NAMES):
        items = [scene["per_index"][channel] for scene in scene_results]
        aggregate: dict[str, Any] = {
            "index": index_name,
            "channel": channel,
            "scene_count": len(items),
            "total_valid_pixel_count": sum(item["valid_pixel_count"] for item in items),
        }
        for metric in METRIC_NAMES:
            values = [float(item[metric]) for item in items if item[metric] is not None]
            aggregate[metric] = _mean_or_none(values)
            aggregate[f"defined_scene_count_{metric}"] = len(values)
        r2_reasons: dict[str, int] = {}
        for item in items:
            reason = item["r2_exclusion_reason"]
            if reason is not None:
                r2_reasons[reason] = r2_reasons.get(reason, 0) + 1
        aggregate["r2_exclusion_reasons"] = r2_reasons
        per_index.append(aggregate)
    return {
        "dataset_scene_macro": dataset_macro,
        "scene_count": len(scene_results),
        "defined_scene_count": {
            metric: sum(scene["scene_macro"][metric] is not None for scene in scene_results)
            for metric in METRIC_NAMES
        },
        "per_index": per_index,
    }


def canonical_metrics_payload(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_scene_results(scene_results)
    return {
        "schema_version": 1,
        "metrics_version": METRICS_VERSION,
        "aggregation": "scene_macro",
        "metric_order": list(METRIC_NAMES),
        **aggregate,
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
