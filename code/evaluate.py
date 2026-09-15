#!/usr/bin/env python3
"""Evaluate any registered Core20 model selected by config.model.name."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

import numpy as np
import torch

import evaluation
import model_registry
import training


FORMAL_SPLIT_ID = "formal_252_split_seed42_202_15_35"


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    return model_registry.build_model(config)


def _scene_samples(manifest: dict[str, Any], scene: str) -> list[dict[str, Any]]:
    samples = [item for item in manifest["samples"] if item["scene"] == scene]
    if not samples:
        raise ValueError(f"manifest contains no samples for scene {scene}")
    return sorted(samples, key=lambda item: (item["top"], item["left"], item["sample_id"]))


def validate_formal_test_split(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    role_counts = (
        len(config["data"]["train_scenes"]),
        len(config["data"]["val_scenes"]),
        len(config["data"]["test_scenes"]),
    )
    if config["data"]["split_id"] != FORMAL_SPLIT_ID or role_counts != (202, 15, 35):
        raise ValueError(
            "test evaluation requires formal_252_split_seed42_202_15_35 "
            "with exact 202/15/35 scene counts"
        )
    if manifest.get("split_id") != FORMAL_SPLIT_ID:
        raise ValueError("measurement manifest formal split identity mismatch")
    manifest_split = manifest.get("split") or {}
    expected_roles = {
        "train_scenes": config["data"]["train_scenes"],
        "val_scenes": config["data"]["val_scenes"],
        "test_scenes": config["data"]["test_scenes"],
    }
    for role, expected in expected_roles.items():
        if manifest_split.get(role) != expected:
            raise ValueError(f"measurement manifest formal {role} differs from config")


def _metadata_paths(manifest_path: Path, samples: list[dict[str, Any]]) -> tuple[Path, Path]:
    label_paths: set[str] = set()
    validity_paths: set[str] = set()
    for sample in samples:
        metadata = evaluation.load_json(manifest_path.parent / sample["metadata"])
        supervision = metadata["supervision"]
        label_paths.add(supervision["indices_path"])
        validity_paths.add(supervision["index_valid_mask_path"])
    if len(label_paths) != 1 or len(validity_paths) != 1:
        raise ValueError("samples from one scene disagree on supervision paths")
    return (
        training._resolve_project_path(label_paths.pop()),
        training._resolve_project_path(validity_paths.pop()),
    )


def _atomic_prediction_memmap(path: Path, shape: tuple[int, int, int]) -> tuple[Path, np.memmap]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    prediction = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    prediction[:] = np.float32(np.nan)
    return temporary, prediction


@torch.no_grad()
def predict_and_evaluate_scene(
    *,
    scene: str,
    samples: list[dict[str, Any]],
    manifest_path: Path,
    model_mask: torch.Tensor,
    model: torch.nn.Module,
    measurement_mean: np.float32,
    measurement_std: np.float32,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    prediction_path: Path,
    coverage_path: Path,
    variance_threshold: float,
) -> dict[str, Any]:
    label_path, validity_path = _metadata_paths(manifest_path, samples)
    target = np.load(label_path, mmap_mode="r", allow_pickle=False)
    label_validity = np.load(validity_path, mmap_mode="r", allow_pickle=False)
    if target.ndim != 3 or target.shape[-1] != 20 or target.dtype != np.float32:
        raise ValueError(f"label contract mismatch for {scene}")
    if label_validity.shape != target.shape or label_validity.dtype != np.bool_:
        raise ValueError(f"label validity contract mismatch for {scene}")
    height, width, _ = target.shape
    temporary_path, prediction = _atomic_prediction_memmap(
        prediction_path, (height, width, 20)
    )
    coverage_count = np.zeros((height, width), dtype=np.uint16)
    try:
        for sample in samples:
            top, left = int(sample["top"]), int(sample["left"])
            patch_height, patch_width = int(sample["height"]), int(sample["width"])
            if patch_height != training.PATCH_SIZE or patch_width != training.PATCH_SIZE:
                raise ValueError(f"unsupported padded patch: {sample['sample_id']}")
            if top < 0 or left < 0 or top + patch_height > height or left + patch_width > width:
                raise ValueError(f"patch lies outside scene: {sample['sample_id']}")
            measurement = np.load(
                manifest_path.parent / sample["measurement"], allow_pickle=False
            )
            normalized = (
                measurement.astype(np.float32, copy=False) - measurement_mean
            ) / measurement_std
            device = next(model.parameters()).device
            input_tensor = (
                torch.from_numpy(np.array(normalized, copy=True))
                .unsqueeze(0)
                .to(device)
            )
            standardized_prediction = model(input_tensor, model_mask).squeeze(0).cpu().numpy()
            if standardized_prediction.shape != (20, patch_height, patch_width):
                raise ValueError(f"prediction shape mismatch: {sample['sample_id']}")
            original_prediction = (
                standardized_prediction.transpose(1, 2, 0) * target_std + target_mean
            ).astype(np.float32, copy=False)
            if not np.isfinite(original_prediction).all():
                raise FloatingPointError(
                    f"prediction contains NaN or Inf: {sample['sample_id']}"
                )
            region = prediction[top : top + patch_height, left : left + patch_width]
            counts = coverage_count[top : top + patch_height, left : left + patch_width]
            unseen = counts == 0
            region[unseen] = original_prediction[unseen]
            if np.any(~unseen):
                old_counts = counts[~unseen].astype(np.float32)[:, None]
                region[~unseen] = (
                    region[~unseen] * old_counts + original_prediction[~unseen]
                ) / (old_counts + np.float32(1.0))
            counts += np.uint16(1)
        prediction.flush()
        del prediction
        os.replace(temporary_path, prediction_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    coverage = coverage_count > 0
    np.save(coverage_path, coverage, allow_pickle=False)
    stitched_prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    result = evaluation.evaluate_scene(
        scene,
        stitched_prediction,
        target,
        label_validity,
        coverage,
        coverage,
        variance_threshold=variance_threshold,
    )
    result["prediction"] = {
        "path": str(prediction_path),
        "sha256": training.sha256_file(prediction_path),
        "shape": list(stitched_prediction.shape),
        "dtype": str(stitched_prediction.dtype),
    }
    result["coverage"] = {
        "path": str(coverage_path),
        "sha256": training.sha256_file(coverage_path),
        "fusion": "uniform_mean",
        "maximum_overlap_count": int(coverage_count.max()),
    }
    result["target"] = {
        "path": str(label_path),
        "sha256": training.sha256_file(label_path),
    }
    result["label_valid_mask"] = {
        "path": str(validity_path),
        "sha256": training.sha256_file(validity_path),
    }
    return result


def run(config_path: Path, split: str, output_dir: Path | None = None) -> dict[str, Any]:
    config = training.load_resolved_config(config_path)
    role_key = {"validation": "val_scenes", "test": "test_scenes"}[split]
    scenes = config["data"][role_key]
    if not scenes:
        raise ValueError(f"configured {split} scene list is empty")
    experiment_dir = Path(config["output"]["experiment_dir"])
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else experiment_dir / "evaluation" / split
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output_dir}")
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(config["data"]["manifest"])
    manifest = evaluation.load_json(manifest_path)
    if split == "test":
        validate_formal_test_split(config, manifest)
    normalization_path = Path(config["data"]["normalization"])
    normalization = evaluation.load_json(normalization_path)
    if normalization["train_scenes"] != config["data"]["train_scenes"]:
        raise ValueError("evaluation normalization is not train-only for this split")
    if normalization["source_manifest_sha256"] != training.sha256_file(manifest_path):
        raise ValueError("evaluation normalization manifest identity mismatch")

    checkpoint_path = experiment_dir / "best_model.pt"
    device = torch.device(config["runner"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA evaluation but CUDA is unavailable")
    model = build_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    model_mask_array = np.load(
        training._resolve_project_path(manifest["model_mask"]["path"]),
        allow_pickle=False,
    )
    model_mask = (
        torch.from_numpy(np.array(model_mask_array, copy=True)).unsqueeze(0).to(device)
    )
    measurement_stats = normalization["measurement"]
    target_stats = normalization["target"]
    target_mean = np.asarray(target_stats["mean_by_channel"], dtype=np.float32)
    target_std = np.asarray(target_stats["std_by_channel"], dtype=np.float32)
    results: list[dict[str, Any]] = []
    for scene in scenes:
        results.append(
            predict_and_evaluate_scene(
                scene=scene,
                samples=_scene_samples(manifest, scene),
                manifest_path=manifest_path,
                model_mask=model_mask,
                model=model,
                measurement_mean=np.float32(measurement_stats["mean"]),
                measurement_std=np.float32(measurement_stats["std"]),
                target_mean=target_mean,
                target_std=target_std,
                prediction_path=predictions_dir / f"{scene}.prediction.npy",
                coverage_path=predictions_dir / f"{scene}.coverage.npy",
                variance_threshold=config["metrics"]["r2_variance_threshold"],
            )
        )

    aggregate = evaluation.canonical_metrics_payload(results)
    per_scene_payload = {
        "schema_version": 1,
        "metrics_version": evaluation.METRICS_VERSION,
        "scenes": results,
    }
    per_index_payload = {
        "schema_version": 1,
        "metrics_version": evaluation.METRICS_VERSION,
        "indices": aggregate.pop("per_index"),
    }
    scope = (
        "formal_test"
        if split == "test"
        else (
            "formal_validation"
            if config["data"]["split_id"] == FORMAL_SPLIT_ID
            else "local_validation_diagnostic"
        )
    )
    summary = {
        **aggregate,
        "scope": scope,
        "split": split,
        "checkpoint_selection_role": split == "validation",
        "generalization_claim_allowed": split == "test",
    }
    protocol = {
        "schema_version": 1,
        "metrics_version": evaluation.METRICS_VERSION,
        "scope": scope,
        "split": split,
        "scene_ids": scenes,
        "split_id": config["data"]["split_id"],
        "evaluation_domain": "label_valid_mask AND crop_coverage AND nonpadding",
        "prediction_layout": "HWC",
        "prediction_dtype": "float32",
        "prediction_space": "original_vegetation_index",
        "overlap_fusion": config["metrics"]["overlap_fusion"],
        "aggregation": config["metrics"]["aggregation"],
        "r2_variance_threshold": config["metrics"]["r2_variance_threshold"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": training.sha256_file(checkpoint_path),
        },
        "model_code": {
            "name": config["model"]["name"],
            "files": {
                label: {
                    "path": str(path),
                    "sha256": training.sha256_file(path),
                }
                for label, path in model_registry.source_files(
                    config["model"]["name"]
                ).items()
            },
        },
        "config": {
            "path": config["source_config"],
            "sha256": config["source_config_sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": training.sha256_file(manifest_path),
        },
        "normalization": {
            "path": str(normalization_path),
            "sha256": training.sha256_file(normalization_path),
            "fit_role": normalization["fit_role"],
            "train_scenes": normalization["train_scenes"],
        },
        "split_manifest": {
            "path": str(experiment_dir / "split_manifest.json"),
            "sha256": training.sha256_file(experiment_dir / "split_manifest.json"),
        },
        "protocol_versions": config["protocol"],
        "evaluation_code": {
            "evaluate_py_sha256": training.sha256_file(Path(__file__)),
            "evaluation_py_sha256": training.sha256_file(Path(evaluation.__file__)),
        },
        "limitations": (
            ["Local validation smoke only; not a held-out formal test result."]
            if split == "validation"
            else []
        ),
    }
    training.atomic_write_json(output_dir / "summary.json", summary)
    training.atomic_write_json(output_dir / "per_scene_metrics.json", per_scene_payload)
    training.atomic_write_json(output_dir / "per_index_metrics.json", per_index_payload)
    training.atomic_write_json(output_dir / "evaluation_protocol.json", protocol)
    return {
        "status": "complete",
        "scope": scope,
        "output_dir": str(output_dir),
        "scene_count": len(results),
        "dataset_scene_macro": summary["dataset_scene_macro"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=training.REPO_ROOT / "config.json")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.split, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, FloatingPointError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
