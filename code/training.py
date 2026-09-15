"""Training data, normalization, loss, configuration, and checkpoint utilities."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_VERSION = "train_global_y_per_index_target_v1"
TRAINING_VERSION = "core20_training_v1"
METRICS_VERSION = "core20_metrics_v1"
LABEL_VERSION = "reviewed_v3_core20"
FORWARD_VERSION = "cassi_forward_v1"
INDEX_CHANNELS = 20
SUPERVISION_CHANNELS = 40
PATCH_SIZE = 256
MEASUREMENT_WIDTH = 422


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_keys_exact(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{label} fields mismatch; missing={missing}, unknown={unknown}")


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Load the sole config source and reject missing, unknown, or inconsistent fields."""

    source_path = Path(path).resolve()
    config = _read_json(source_path)
    top_fields = {
        "schema_version",
        "experiment_id",
        "data",
        "normalization",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "runner",
        "metrics",
        "protocol",
        "output",
    }
    _require_keys_exact(config, top_fields, "config")
    schemas = {
        "data": {
            "split_id",
            "manifest",
            "normalization",
            "train_scenes",
            "val_scenes",
            "test_scenes",
            "patch_size",
            "batch_size",
            "val_batch_size",
            "num_workers",
        },
        "normalization": {"epsilon"},
        "model": {
            "name",
            "dim",
            "num_heads",
            "num_blocks",
            "ffn_expansion_factor",
            "in_channels",
            "out_channels",
            "bias",
        },
        "loss": {"name", "epsilon"},
        "optimizer": {"name", "learning_rate", "weight_decay"},
        "scheduler": {"name", "t_max", "eta_min"},
        "runner": {
            "runner_root",
            "epochs",
            "device",
            "seed",
            "gradient_clip_norm",
            "early_stopping_patience",
            "log_every",
            "run_overfit_probe",
        },
        "metrics": {
            "checkpoint",
            "report",
            "aggregation",
            "r2_variance_threshold",
            "overlap_fusion",
        },
        "protocol": {"label", "forward", "training", "metrics"},
        "output": {"root"},
    }
    for group, fields in schemas.items():
        if not isinstance(config[group], dict):
            raise TypeError(f"config.{group} must be an object")
        _require_keys_exact(config[group], fields, f"config.{group}")

    if config["schema_version"] != 1:
        raise ValueError("config.schema_version must be 1")
    if not isinstance(config["experiment_id"], str) or not config["experiment_id"]:
        raise ValueError("experiment_id must be a nonempty string")
    data = config["data"]
    if not isinstance(data["split_id"], str) or not data["split_id"]:
        raise ValueError("data.split_id must be a nonempty string")
    if data["patch_size"] != PATCH_SIZE:
        raise ValueError(f"data.patch_size must be {PATCH_SIZE}")
    if not isinstance(data["batch_size"], int) or data["batch_size"] <= 0:
        raise ValueError("data.batch_size must be a positive integer")
    if not isinstance(data["val_batch_size"], int) or data["val_batch_size"] <= 0:
        raise ValueError("data.val_batch_size must be a positive integer")
    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("data.num_workers must be a nonnegative integer")
    train_scenes = data["train_scenes"]
    val_scenes = data["val_scenes"]
    test_scenes = data["test_scenes"]
    if not isinstance(train_scenes, list) or not train_scenes:
        raise ValueError("data.train_scenes must be a nonempty list")
    if not isinstance(val_scenes, list) or not val_scenes:
        raise ValueError("data.val_scenes must be a nonempty list")
    if not isinstance(test_scenes, list):
        raise ValueError("data.test_scenes must be a list")
    for role, scenes in (
        ("train", train_scenes),
        ("validation", val_scenes),
        ("test", test_scenes),
    ):
        if any(not isinstance(scene, str) or not scene for scene in scenes):
            raise ValueError(f"data.{role}_scenes must contain nonempty strings")
        if len(set(scenes)) != len(scenes):
            raise ValueError(f"data.{role}_scenes contains duplicate scenes")
    roles = {
        "train": set(train_scenes),
        "validation": set(val_scenes),
        "test": set(test_scenes),
    }
    if any(
        roles[first] & roles[second]
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("train, validation, and test scenes must be pairwise disjoint")
    if data["split_id"] == "formal_252_split_seed42_202_15_35":
        if tuple(len(role) for role in (train_scenes, val_scenes, test_scenes)) != (
            202,
            15,
            35,
        ):
            raise ValueError("formal split must contain exact 202/15/35 scene counts")
        expected_formal_scenes = {f"hsi_{scene_id:04d}" for scene_id in range(1, 253)}
        if set(train_scenes + val_scenes + test_scenes) != expected_formal_scenes:
            raise ValueError("formal split must cover hsi_0001 through hsi_0252 exactly once")

    normalization = config["normalization"]
    if not isinstance(normalization["epsilon"], (int, float)) or normalization["epsilon"] <= 0:
        raise ValueError("normalization.epsilon must be positive")
    model = config["model"]
    expected_model = {"name": "RestormerCore20", "in_channels": 84, "out_channels": 20}
    for key, expected in expected_model.items():
        if model[key] != expected:
            raise ValueError(f"model.{key} must be {expected!r}")
    if not isinstance(model["dim"], int) or model["dim"] <= 0:
        raise ValueError("model.dim must be a positive integer")
    if (
        not isinstance(model["num_heads"], list)
        or not isinstance(model["num_blocks"], list)
        or len(model["num_heads"]) != len(model["num_blocks"])
        or not model["num_heads"]
    ):
        raise ValueError("model num_heads and num_blocks must be equal nonempty lists")
    if model["bias"] not in (True, False):
        raise TypeError("model.bias must be boolean")
    if config["loss"]["name"] != "MaskedBalancedCharbonnier":
        raise ValueError("loss.name must be MaskedBalancedCharbonnier")
    if config["optimizer"]["name"] != "AdamW":
        raise ValueError("optimizer.name must be AdamW")
    if config["scheduler"]["name"] != "CosineAnnealingLR":
        raise ValueError("scheduler.name must be CosineAnnealingLR")
    runner = config["runner"]
    if runner["device"] not in {"cpu", "cuda"}:
        raise ValueError("runner.device must be cpu or cuda")
    for key in ("epochs", "seed", "log_every"):
        if not isinstance(runner[key], int) or runner[key] <= 0:
            raise ValueError(f"runner.{key} must be a positive integer")
    if runner["gradient_clip_norm"] <= 0 or runner["early_stopping_patience"] <= 0:
        raise ValueError("runner clip norm and patience must be positive")
    if runner["run_overfit_probe"] not in (True, False):
        raise TypeError("runner.run_overfit_probe must be boolean")
    metrics = config["metrics"]
    if metrics["checkpoint"] != "macro_zMAE":
        raise ValueError("metrics.checkpoint must be macro_zMAE")
    if metrics["report"] != ["MAE", "RMSE", "Bias", "R2"]:
        raise ValueError("metrics.report must be [MAE, RMSE, Bias, R2]")
    if metrics["aggregation"] != "scene_macro":
        raise ValueError("metrics.aggregation must be scene_macro")
    if metrics["r2_variance_threshold"] != 1e-12:
        raise ValueError("metrics.r2_variance_threshold must be 1e-12")
    if metrics["overlap_fusion"] != "uniform_mean":
        raise ValueError("metrics.overlap_fusion must be uniform_mean")
    expected_protocol = {
        "label": LABEL_VERSION,
        "forward": FORWARD_VERSION,
        "training": TRAINING_VERSION,
        "metrics": METRICS_VERSION,
    }
    if config["protocol"] != expected_protocol:
        raise ValueError(f"config.protocol must equal {expected_protocol}")

    resolved = json.loads(json.dumps(config))
    resolved["source_config"] = str(source_path)
    resolved["source_config_sha256"] = sha256_file(source_path)
    for key in ("manifest", "normalization"):
        resolved["data"][key] = str(_resolve_project_path(data[key]))
    resolved["runner"]["runner_root"] = str(_resolve_project_path(runner["runner_root"]))
    output_root = _resolve_project_path(config["output"]["root"])
    resolved["output"]["root"] = str(output_root)
    resolved["output"]["experiment_dir"] = str(output_root / config["experiment_id"])
    return resolved


def _manifest_samples(manifest: dict[str, Any], scenes: list[str]) -> list[dict[str, Any]]:
    scene_set = set(scenes)
    samples = [sample for sample in manifest.get("samples", []) if sample.get("scene") in scene_set]
    found = {sample["scene"] for sample in samples}
    if found != scene_set:
        raise ValueError(f"manifest scene mismatch; requested={sorted(scene_set)}, found={sorted(found)}")
    return samples


def fit_normalization(
    manifest_path: Path,
    train_scenes: list[str],
    output_path: Path,
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Fit Y scalar and per-index target statistics from training scenes only."""

    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = _read_json(manifest_path)
    samples = _manifest_samples(manifest, train_scenes)
    measurement_sum = 0.0
    measurement_square_sum = 0.0
    measurement_count = 0
    target_sum = np.zeros(INDEX_CHANNELS, dtype=np.float64)
    target_square_sum = np.zeros(INDEX_CHANNELS, dtype=np.float64)
    target_count = np.zeros(INDEX_CHANNELS, dtype=np.int64)
    current_scene: str | None = None
    current_labels: np.ndarray | None = None
    current_validity: np.ndarray | None = None

    for sample in samples:
        measurement = np.load(
            manifest_path.parent / sample["measurement"],
            mmap_mode="r",
            allow_pickle=False,
        )
        if measurement.shape != (PATCH_SIZE, MEASUREMENT_WIDTH) or measurement.dtype != np.float32:
            raise ValueError(f"measurement contract mismatch: {sample['sample_id']}")
        values = np.asarray(measurement, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite measurement: {sample['sample_id']}")
        measurement_sum += float(values.sum())
        measurement_square_sum += float(np.square(values).sum())
        measurement_count += values.size

        metadata = _read_json(manifest_path.parent / sample["metadata"])
        supervision = metadata.get("supervision") or {}
        scene = sample["scene"]
        if scene != current_scene:
            current_labels = np.load(
                _resolve_project_path(supervision["indices_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            current_validity = np.load(
                _resolve_project_path(supervision["index_valid_mask_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            current_scene = scene
        if current_labels is None or current_validity is None:
            raise RuntimeError("normalization label state was not initialized")
        labels, validity = current_labels, current_validity
        top, left = sample["top"], sample["left"]
        label_patch = labels[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
        valid_patch = validity[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
        if label_patch.shape != (PATCH_SIZE, PATCH_SIZE, INDEX_CHANNELS):
            raise ValueError(f"label patch contract mismatch: {sample['sample_id']}")
        for channel in range(INDEX_CHANNELS):
            channel_values = np.asarray(
                label_patch[..., channel][valid_patch[..., channel]],
                dtype=np.float64,
            )
            if channel_values.size:
                if not np.isfinite(channel_values).all():
                    raise ValueError(f"valid label contains non-finite data: {sample['sample_id']}")
                target_sum[channel] += channel_values.sum()
                target_square_sum[channel] += np.square(channel_values).sum()
                target_count[channel] += channel_values.size

    if measurement_count == 0 or np.any(target_count == 0):
        raise ValueError("training samples do not contain enough normalization data")
    measurement_mean = measurement_sum / measurement_count
    measurement_variance = max(
        measurement_square_sum / measurement_count - measurement_mean**2,
        0.0,
    )
    measurement_std = measurement_variance**0.5
    target_mean = target_sum / target_count
    target_variance = np.maximum(target_square_sum / target_count - target_mean**2, 0.0)
    target_std = np.sqrt(target_variance)
    if measurement_std < epsilon or np.any(target_std < epsilon):
        raise ValueError("a fitted normalization standard deviation is below epsilon")

    payload = {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "fit_role": "train_only",
        "train_scenes": list(train_scenes),
        "train_sample_ids": [sample["sample_id"] for sample in samples],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "epsilon": epsilon,
        "ddof": 0,
        "measurement": {
            "count": measurement_count,
            "mean": measurement_mean,
            "std": measurement_std,
        },
        "target": {
            "count_by_channel": target_count.tolist(),
            "mean_by_channel": target_mean.tolist(),
            "std_by_channel": target_std.tolist(),
        },
    }
    atomic_write_json(output_path, payload)
    return payload


class Core20Dataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Return ``measurement, model_mask, concat(safe_zlabel, valid_mask)``."""

    def __init__(
        self,
        manifest_path: Path,
        normalization_path: Path,
        scenes: list[str],
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _read_json(self.manifest_path)
        self.samples = sorted(
            _manifest_samples(self.manifest, scenes),
            key=lambda item: (item["scene"], item["top"], item["left"], item["sample_id"]),
        )
        self.normalization = _read_json(Path(normalization_path).resolve())
        if self.normalization.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError("normalization version mismatch")
        measurement_stats = self.normalization["measurement"]
        target_stats = self.normalization["target"]
        self.measurement_mean = np.float32(measurement_stats["mean"])
        self.measurement_std = np.float32(measurement_stats["std"])
        self.target_mean = np.asarray(target_stats["mean_by_channel"], dtype=np.float32)
        self.target_std = np.asarray(target_stats["std_by_channel"], dtype=np.float32)
        self.model_mask = np.load(
            _resolve_project_path(self.manifest["model_mask"]["path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if self.model_mask.shape != (84, PATCH_SIZE, PATCH_SIZE):
            raise ValueError("model mask contract mismatch")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        measurement = np.load(
            self.manifest_path.parent / sample["measurement"],
            allow_pickle=False,
        )
        metadata = _read_json(self.manifest_path.parent / sample["metadata"])
        supervision_info = metadata["supervision"]
        labels = np.load(
            _resolve_project_path(supervision_info["indices_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        validity = np.load(
            _resolve_project_path(supervision_info["index_valid_mask_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        top, left = sample["top"], sample["left"]
        label_patch = np.asarray(
            labels[top : top + PATCH_SIZE, left : left + PATCH_SIZE],
            dtype=np.float32,
        )
        valid_patch = np.asarray(
            validity[top : top + PATCH_SIZE, left : left + PATCH_SIZE],
            dtype=np.bool_,
        )
        safe = np.where(valid_patch, label_patch, np.float32(0.0))
        standardized = (safe - self.target_mean) / self.target_std
        standardized[~valid_patch] = np.float32(0.0)
        supervision = np.concatenate(
            [standardized.transpose(2, 0, 1), valid_patch.transpose(2, 0, 1)],
            axis=0,
        ).astype(np.float32, copy=False)
        normalized_measurement = (
            measurement.astype(np.float32, copy=False) - self.measurement_mean
        ) / self.measurement_std
        return (
            torch.from_numpy(np.array(normalized_measurement, copy=True)),
            torch.from_numpy(np.array(self.model_mask, copy=True)),
            torch.from_numpy(np.array(supervision, copy=True)),
        )


def split_supervision(supervision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if supervision.ndim != 4 or supervision.shape[1] != SUPERVISION_CHANNELS:
        raise ValueError(f"supervision must have shape [B,40,H,W], got {tuple(supervision.shape)}")
    target = supervision[:, :INDEX_CHANNELS]
    validity = supervision[:, INDEX_CHANNELS:] > 0.5
    return target, validity


class MaskedBalancedCharbonnierLoss(nn.Module):
    """Average pixels within each valid channel, then average valid channels."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("Charbonnier epsilon must be positive")
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, supervision: torch.Tensor) -> torch.Tensor:
        target, validity = split_supervision(supervision)
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape {prediction.shape} != target shape {target.shape}")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("prediction contains NaN or Inf")
        counts = validity.sum(dim=(0, 2, 3))
        valid_channels = counts > 0
        if not bool(valid_channels.any()):
            raise ValueError("batch contains no valid supervision")
        error = torch.sqrt((prediction - target).square() + self.epsilon**2)
        sums = (error * validity).sum(dim=(0, 2, 3))
        channel_losses = sums[valid_channels] / counts[valid_channels]
        return channel_losses.mean()


def balanced_zmae(prediction: torch.Tensor, supervision: torch.Tensor) -> float:
    target, validity = split_supervision(supervision)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("invalid prediction for zMAE")
    counts = validity.sum(dim=(0, 2, 3))
    valid_channels = counts > 0
    if not bool(valid_channels.any()):
        raise ValueError("batch contains no valid zMAE channels")
    sums = ((prediction - target).abs() * validity).sum(dim=(0, 2, 3))
    return float((sums[valid_channels] / counts[valid_channels]).mean().item())


def load_runner_v3(runner_root: Path) -> tuple[type[Any], str, Path]:
    """Directly import the specified learning-repository RunnerV3 and hash it."""

    root = Path(runner_root).resolve()
    runner_file = root / "nndl/runner.py"
    if not runner_file.is_file():
        raise FileNotFoundError(runner_file)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("nndl.runner")
    runner_class = getattr(module, "RunnerV3", None)
    if runner_class is None:
        raise ImportError(f"RunnerV3 is absent from {runner_file}")
    return runner_class, sha256_file(runner_file), runner_file


def save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    completed_epochs: int,
    history: dict[str, Any],
    experiment_signature: str,
    dataloader_generator: torch.Generator,
    best_metric: float | None = None,
    no_improve_epochs: int = 0,
) -> None:
    atomic_torch_save(
        Path(path),
        {
            "schema_version": 1,
            "training_version": TRAINING_VERSION,
            "experiment_signature": experiment_signature,
            "completed_epochs": completed_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_metric": best_metric,
            "no_improve_epochs": no_improve_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_random_state": np.random.get_state(),
            "dataloader_generator_state": dataloader_generator.get_state(),
            "python_random_state": random.getstate(),
        },
    )


def restore_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_signature: str,
    dataloader_generator: torch.Generator,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1 or state.get("training_version") != TRAINING_VERSION:
        raise ValueError("training-state schema or protocol mismatch")
    if state.get("experiment_signature") != expected_signature:
        raise ValueError("training-state experiment signature mismatch")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    numpy_random_state = state.get("numpy_random_state")
    if numpy_random_state is not None:
        np.random.set_state(numpy_random_state)
    dataloader_generator.set_state(state["dataloader_generator_state"])
    random.setstate(state["python_random_state"])
    return state
