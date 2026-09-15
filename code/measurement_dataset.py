#!/usr/bin/env python3
"""Build and verify versioned CASSI measurement samples.

The CLI supports the archived random-mask smoke workflow, the reviewed
three-scene minimum workflow, and the formal 252-scene workflow. Formal mode
requires the frozen historical scene split and writes its identity into the
measurement manifest before training can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np

import cassi_forward


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/raw/HSI"
DEFAULT_VALIDITY_DIR = REPO_ROOT / "data/derived/validity/hsi_reflectance_v1"
DEFAULT_LABEL_DIR = REPO_ROOT / "data/derived/labels/reviewed_v3_core20"
DEFAULT_DEV_MASK = REPO_ROOT / "data/mask/dev_random_seed42.npy"
DEFAULT_DEV_MASK_META = REPO_ROOT / "data/mask/dev_random_seed42.meta.json"
DEFAULT_SMOKE_OUTPUT = (
    REPO_ROOT / "data/derived/measurements/cassi_forward_v1/dev_random_seed42"
)
DEFAULT_FORMAL_MASK = REPO_ROOT / "data/mask/mask.npy"
DEFAULT_FORMAL_MASK_META = REPO_ROOT / "data/mask/mask.meta.json"
DEFAULT_MINIMUM_OUTPUT = (
    REPO_ROOT / "data/derived/measurements/cassi_forward_v1/local_minimum"
)
DEFAULT_FORMAL_SPLIT = (
    REPO_ROOT / "data/manifests/formal_252_split_seed42_202_15_35.json"
)
DEFAULT_FORMAL_OUTPUT = (
    REPO_ROOT
    / "data/derived/measurements/cassi_forward_v1/formal_252_stride256"
)

MASK_ROLE_DEVELOPMENT = "development_smoke_only"
MASK_ROLE_FORMAL = "formal"
SCOPE_LOCAL_MINIMUM = "local_minimum_experiment"
SCOPE_FORMAL = "formal_252_experiment"
FORMAL_SPLIT_ID = "formal_252_split_seed42_202_15_35"
LABEL_VERSION = "reviewed_v3_core20"
VALIDITY_VERSION = "hsi_reflectance_v1"


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def create_development_mask(
    mask_path: Path = DEFAULT_DEV_MASK,
    metadata_path: Path = DEFAULT_DEV_MASK_META,
    *,
    seed: int = 42,
    open_probability: float = 0.5,
) -> dict[str, Any]:
    """Create a deterministic random aperture that can only be used for smoke tests."""

    mask_path = Path(mask_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if mask_path.name == "mask.npy":
        raise ValueError("a development mask must not use the reserved formal name mask.npy")
    if mask_path.exists() or metadata_path.exists():
        raise FileExistsError("development mask artifact already exists")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not np.isfinite(open_probability) or not 0 < open_probability < 1:
        raise ValueError("open_probability must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    mask = (
        rng.random((cassi_forward.PATCH_SIZE, cassi_forward.PATCH_SIZE))
        < open_probability
    ).astype(np.float32)
    open_fraction = cassi_forward.validate_physical_mask(mask)
    _atomic_save_npy(mask_path, mask)
    mask_sha256 = sha256_file(mask_path)
    metadata = {
        "schema_version": 1,
        "mask_id": f"dev_random_seed{seed}",
        "role": MASK_ROLE_DEVELOPMENT,
        "shape": [cassi_forward.PATCH_SIZE, cassi_forward.PATCH_SIZE],
        "dtype": "float32",
        "binary": True,
        "open_fraction": open_fraction,
        "sha256": mask_sha256,
        "path": _recorded_path(mask_path),
        "generator": {
            "name": "numpy.random.default_rng",
            "seed": seed,
            "open_probability": open_probability,
        },
        "warning": "Development smoke test only; never use for formal training data.",
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


def load_mask_artifact(
    mask_path: Path,
    metadata_path: Path,
    *,
    required_role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_path = Path(mask_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    mask = cassi_forward.load_npy_strict(mask_path, mmap_mode=None)
    open_fraction = cassi_forward.validate_physical_mask(mask)
    metadata = _read_json(metadata_path)
    required = {
        "schema_version": 1,
        "role": required_role,
        "shape": [cassi_forward.PATCH_SIZE, cassi_forward.PATCH_SIZE],
        "dtype": "float32",
        "binary": True,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"mask metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    actual_sha256 = sha256_file(mask_path)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError("mask metadata SHA-256 does not match mask.npy")
    if not np.isclose(metadata.get("open_fraction"), open_fraction, rtol=0, atol=1e-12):
        raise ValueError("mask metadata open_fraction does not match mask.npy")
    return mask, metadata


def find_valid_patch_origins(
    hsi_valid_mask: np.ndarray,
    *,
    stride: int,
    max_patches: int | None,
) -> list[tuple[int, int]]:
    validity = np.asarray(hsi_valid_mask)
    if validity.ndim != 2 or validity.dtype != np.bool_:
        raise ValueError("hsi_valid_mask must be an HW bool array")
    if stride <= 0 or (max_patches is not None and max_patches <= 0):
        raise ValueError("stride must be positive and max_patches must be positive or None")
    height, width = validity.shape
    if height < cassi_forward.PATCH_SIZE or width < cassi_forward.PATCH_SIZE:
        raise ValueError("scene is smaller than the fixed 256x256 patch")

    def positions(length: int) -> list[int]:
        result = list(range(0, length - cassi_forward.PATCH_SIZE + 1, stride))
        final = length - cassi_forward.PATCH_SIZE
        if result[-1] != final:
            result.append(final)
        return result

    origins: list[tuple[int, int]] = []
    for top in positions(height):
        for left in positions(width):
            patch = validity[
                top : top + cassi_forward.PATCH_SIZE,
                left : left + cassi_forward.PATCH_SIZE,
            ]
            if bool(patch.all()):
                origins.append((top, left))
                if max_patches is not None and len(origins) == max_patches:
                    return origins
    if not origins:
        raise ValueError("scene has no fully valid 256x256 patch on the deterministic grid")
    return origins


def _discover_sources(input_dir: Path, scenes: list[str] | None) -> list[Path]:
    input_dir = Path(input_dir).resolve()
    if scenes:
        result: list[Path] = []
        for requested in dict.fromkeys(scenes):
            name = Path(requested).name
            if name != requested:
                raise ValueError(f"scene must be a filename or stem, not a path: {requested}")
            candidate = input_dir / (name if name.endswith(".npy") else f"{name}.npy")
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            result.append(candidate)
        return result
    result = sorted(input_dir.glob("hsi_*.npy"))
    if not result:
        raise FileNotFoundError(f"no hsi_*.npy files found in {input_dir}")
    return result


def load_formal_scene_split(path: Path) -> dict[str, Any]:
    """Validate and convert the frozen 202/15/35 integer scene split."""

    path = Path(path).resolve()
    payload = _read_json(path)
    if payload.get("seed") != 42:
        raise ValueError("formal split seed must be 42")
    if payload.get("partition_unit") != "scene":
        raise ValueError("formal split partition_unit must be scene")
    if payload.get("physical_independence_confirmed") is not True:
        raise ValueError("formal split must confirm physical scene independence")
    expected_counts = {"train": 202, "val": 15, "test": 35}
    roles: dict[str, list[str]] = {}
    integer_roles: dict[str, list[int]] = {}
    for role, expected_count in expected_counts.items():
        values = payload.get(role)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(not isinstance(value, int) for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"formal split {role} must contain {expected_count} unique integers")
        integer_roles[role] = values
        roles[role] = [f"hsi_{value:04d}" for value in values]
    role_sets = {role: set(values) for role, values in integer_roles.items()}
    if (
        role_sets["train"] & role_sets["val"]
        or role_sets["train"] & role_sets["test"]
        or role_sets["val"] & role_sets["test"]
    ):
        raise ValueError("formal split roles are not pairwise disjoint")
    if set().union(*role_sets.values()) != set(range(1, 253)):
        raise ValueError("formal split must cover scene IDs 1 through 252 exactly once")
    return {
        "split_id": FORMAL_SPLIT_ID,
        "source_path": _recorded_path(path),
        "source_sha256": sha256_file(path),
        "source_split_hash": payload.get("split_hash"),
        "seed": 42,
        "partition_unit": payload.get("partition_unit"),
        "physical_independence_confirmed": payload.get(
            "physical_independence_confirmed"
        ),
        "train_scenes": roles["train"],
        "val_scenes": roles["val"],
        "test_scenes": roles["test"],
    }


def generate_smoke_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    validity_dir: Path = DEFAULT_VALIDITY_DIR,
    label_dir: Path = DEFAULT_LABEL_DIR,
    mask_path: Path = DEFAULT_DEV_MASK,
    mask_metadata_path: Path = DEFAULT_DEV_MASK_META,
    output_dir: Path = DEFAULT_SMOKE_OUTPUT,
    scenes: list[str] | None = None,
    stride: int = 256,
    max_patches_per_scene: int | None = 2,
    required_mask_role: str = MASK_ROLE_DEVELOPMENT,
    dataset_scope: str = MASK_ROLE_DEVELOPMENT,
    warning: str = "Development smoke data only; never report formal metrics from it.",
    split_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a small deterministic dataset using a development-only aperture."""

    input_dir = Path(input_dir).resolve()
    validity_dir = Path(validity_dir).resolve()
    label_dir = Path(label_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    physical_mask, mask_metadata = load_mask_artifact(
        mask_path,
        mask_metadata_path,
        required_role=required_mask_role,
    )
    sources = _discover_sources(input_dir, scenes)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    samples: list[dict[str, Any]] = []

    try:
        temporary_dir.mkdir(parents=True, exist_ok=False)
        model_mask = cassi_forward.build_model_mask(physical_mask)
        model_mask_path = temporary_dir / "model_mask.npy"
        _atomic_save_npy(model_mask_path, model_mask)
        model_mask_sha256 = sha256_file(model_mask_path)
        _atomic_write_json(
            temporary_dir / "model_mask.meta.json",
            {
                "schema_version": 1,
                "forward_version": cassi_forward.FORWARD_VERSION,
                "policy": cassi_forward.MODEL_MASK_POLICY,
                "layout": "CHW",
                "shape": list(model_mask.shape),
                "dtype": "float32",
                "sha256": model_mask_sha256,
                "physical_mask_sha256": mask_metadata["sha256"],
                "role": required_mask_role,
            },
        )

        for source_path in sources:
            scene = source_path.stem
            validity_path = validity_dir / f"{scene}.hsi_valid_mask.npy"
            indices_path = label_dir / f"{scene}.indices.npy"
            index_mask_path = label_dir / f"{scene}.index_valid_mask.npy"
            label_meta_path = label_dir / f"{scene}.meta.json"
            required_paths = [validity_path, indices_path, index_mask_path, label_meta_path]
            missing = [str(path) for path in required_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"scene {scene} is incomplete: {missing}")

            stored_hsi = cassi_forward.load_npy_strict(source_path)
            hsi_valid_mask = cassi_forward.load_npy_strict(validity_path)
            origins = find_valid_patch_origins(
                hsi_valid_mask,
                stride=stride,
                max_patches=max_patches_per_scene,
            )
            label_metadata = _read_json(label_meta_path)
            source_sha256 = sha256_file(source_path)
            if (label_metadata.get("source") or {}).get("sha256") != source_sha256:
                raise ValueError(f"label source SHA-256 mismatch for {scene}")

            for top, left in origins:
                patch = cassi_forward.extract_unit_reflectance_patch(
                    stored_hsi,
                    hsi_valid_mask,
                    top=top,
                    left=left,
                )
                measurement = cassi_forward.simulate_cassi_measurement(
                    patch,
                    physical_mask,
                )
                sample_id = f"{scene}_y{top:04d}_x{left:04d}"
                measurement_filename = f"{sample_id}.measurement.npy"
                sample_metadata_filename = f"{sample_id}.meta.json"
                measurement_path = temporary_dir / measurement_filename
                _atomic_save_npy(measurement_path, measurement)
                measurement_sha256 = sha256_file(measurement_path)
                sample = {
                    "sample_id": sample_id,
                    "scene": scene,
                    "top": top,
                    "left": left,
                    "height": cassi_forward.PATCH_SIZE,
                    "width": cassi_forward.PATCH_SIZE,
                    "measurement": measurement_filename,
                    "metadata": sample_metadata_filename,
                }
                sample_metadata = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "scope": dataset_scope,
                    "forward_version": cassi_forward.FORWARD_VERSION,
                    "source_hsi": {
                        "path": _recorded_path(source_path),
                        "sha256": source_sha256,
                    },
                    "patch": {
                        "top": top,
                        "left": left,
                        "height": cassi_forward.PATCH_SIZE,
                        "width": cassi_forward.PATCH_SIZE,
                    },
                    "dispersion_step": cassi_forward.DISPERSION_STEP,
                    "measurement_scale": cassi_forward.MEASUREMENT_SCALE,
                    "measurement_scale_semantics": (
                        "historical simulation convention; not calibrated throughput"
                    ),
                    "measurement": {
                        "path": _recorded_path(output_dir / measurement_filename),
                        "shape": list(measurement.shape),
                        "dtype": "float32",
                        "sha256": measurement_sha256,
                    },
                    "physical_mask": {
                        "path": _recorded_path(mask_path),
                        "sha256": mask_metadata["sha256"],
                        "role": required_mask_role,
                    },
                    "model_mask": {
                        "path": _recorded_path(output_dir / "model_mask.npy"),
                        "sha256": model_mask_sha256,
                        "policy": cassi_forward.MODEL_MASK_POLICY,
                    },
                    "supervision": {
                        "label_version": LABEL_VERSION,
                        "indices_path": _recorded_path(indices_path),
                        "index_valid_mask_path": _recorded_path(index_mask_path),
                        "indices_sha256": (label_metadata.get("indices") or {}).get("sha256"),
                        "index_valid_mask_sha256": (
                            label_metadata.get("index_valid_mask") or {}
                        ).get("sha256"),
                    },
                }
                _atomic_write_json(
                    temporary_dir / sample_metadata_filename,
                    sample_metadata,
                )
                samples.append(sample)

        manifest = {
            "schema_version": 1,
            "scope": dataset_scope,
            "forward_version": cassi_forward.FORWARD_VERSION,
            "label_version": LABEL_VERSION,
            "hsi_validity_version": VALIDITY_VERSION,
            "measurement_scale": cassi_forward.MEASUREMENT_SCALE,
            "dispersion_step": cassi_forward.DISPERSION_STEP,
            "model_mask_policy": cassi_forward.MODEL_MASK_POLICY,
            "physical_mask": {
                "path": _recorded_path(mask_path),
                "metadata_path": _recorded_path(mask_metadata_path),
                "sha256": mask_metadata["sha256"],
                "open_fraction": mask_metadata["open_fraction"],
            },
            "model_mask": {
                "path": _recorded_path(output_dir / "model_mask.npy"),
                "sha256": model_mask_sha256,
            },
            "patch_selection": {
                "order": "deterministic_row_major_grid",
                "stride": stride,
                "max_patches_per_scene": max_patches_per_scene,
                "requires_all_hsi_valid": True,
            },
            "sample_count": len(samples),
            "samples": samples,
            "warning": warning,
        }
        if split_identity is not None:
            manifest["split_id"] = split_identity["split_id"]
            manifest["split"] = split_identity
        _atomic_write_json(temporary_dir / "manifest.json", manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, output_dir)
        return {
            "status": "generated",
            "scope": dataset_scope,
            "scene_count": len(sources),
            "sample_count": len(samples),
            "output_dir": str(output_dir),
        }
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def generate_minimum_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    validity_dir: Path = DEFAULT_VALIDITY_DIR,
    label_dir: Path = DEFAULT_LABEL_DIR,
    mask_path: Path = DEFAULT_FORMAL_MASK,
    mask_metadata_path: Path = DEFAULT_FORMAL_MASK_META,
    output_dir: Path = DEFAULT_MINIMUM_OUTPUT,
    scenes: list[str] | None = None,
    stride: int = 256,
    max_patches_per_scene: int = 2,
) -> dict[str, Any]:
    """Generate the three-scene minimum dataset with the formal simulation mask."""

    return generate_smoke_dataset(
        input_dir=input_dir,
        validity_dir=validity_dir,
        label_dir=label_dir,
        mask_path=mask_path,
        mask_metadata_path=mask_metadata_path,
        output_dir=output_dir,
        scenes=scenes,
        stride=stride,
        max_patches_per_scene=max_patches_per_scene,
        required_mask_role=MASK_ROLE_FORMAL,
        dataset_scope=SCOPE_LOCAL_MINIMUM,
        warning=(
            "Three-scene minimum experiment only; the aperture is a formal simulation "
            "baseline but is not hardware calibrated, and these samples cannot support "
            "generalization claims."
        ),
    )


def generate_formal_dataset(
    *,
    split_manifest_path: Path = DEFAULT_FORMAL_SPLIT,
    input_dir: Path = DEFAULT_INPUT_DIR,
    validity_dir: Path = DEFAULT_VALIDITY_DIR,
    label_dir: Path = DEFAULT_LABEL_DIR,
    mask_path: Path = DEFAULT_FORMAL_MASK,
    mask_metadata_path: Path = DEFAULT_FORMAL_MASK_META,
    output_dir: Path = DEFAULT_FORMAL_OUTPUT,
    stride: int = 256,
    max_patches_per_scene: int | None = None,
) -> dict[str, Any]:
    """Generate deterministic measurements after the scene split is frozen."""

    split = load_formal_scene_split(split_manifest_path)
    scenes = sorted(
        split["train_scenes"] + split["val_scenes"] + split["test_scenes"]
    )
    return generate_smoke_dataset(
        input_dir=input_dir,
        validity_dir=validity_dir,
        label_dir=label_dir,
        mask_path=mask_path,
        mask_metadata_path=mask_metadata_path,
        output_dir=output_dir,
        scenes=scenes,
        stride=stride,
        max_patches_per_scene=max_patches_per_scene,
        required_mask_role=MASK_ROLE_FORMAL,
        dataset_scope=SCOPE_FORMAL,
        warning=(
            "Formal 252-scene simulation dataset using a fixed derived aperture. "
            "The aperture is not hardware calibrated; test scenes are final-evaluation only."
        ),
        split_identity=split,
    )


def verify_smoke_dataset(
    output_dir: Path = DEFAULT_SMOKE_OUTPUT,
    *,
    expected_scope: str = MASK_ROLE_DEVELOPMENT,
    required_mask_role: str = MASK_ROLE_DEVELOPMENT,
    mask_path: Path | None = None,
    mask_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute every smoke measurement and verify all recorded identities."""

    output_dir = Path(output_dir).resolve()
    manifest = _read_json(output_dir / "manifest.json")
    expected_manifest = {
        "scope": expected_scope,
        "forward_version": cassi_forward.FORWARD_VERSION,
        "label_version": LABEL_VERSION,
        "hsi_validity_version": VALIDITY_VERSION,
        "measurement_scale": cassi_forward.MEASUREMENT_SCALE,
        "dispersion_step": cassi_forward.DISPERSION_STEP,
        "model_mask_policy": cassi_forward.MODEL_MASK_POLICY,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}")

    physical_info = manifest.get("physical_mask") or {}
    resolved_mask_path = mask_path or (REPO_ROOT / physical_info["path"])
    resolved_mask_metadata_path = mask_metadata_path or (
        REPO_ROOT / physical_info["metadata_path"]
    )
    physical_mask, mask_metadata = load_mask_artifact(
        resolved_mask_path,
        resolved_mask_metadata_path,
        required_role=required_mask_role,
    )
    model_mask_path = output_dir / "model_mask.npy"
    model_mask = cassi_forward.load_npy_strict(model_mask_path, mmap_mode=None)
    expected_model_mask = cassi_forward.build_model_mask(physical_mask)
    if not np.array_equal(model_mask, expected_model_mask):
        raise ValueError("model mask does not match the fixed unshifted-repeat policy")
    if sha256_file(model_mask_path) != (manifest.get("model_mask") or {}).get("sha256"):
        raise ValueError("model mask SHA-256 does not match manifest")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or manifest.get("sample_count") != len(samples):
        raise ValueError("manifest sample_count does not match samples")
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        metadata = _read_json(output_dir / sample["metadata"])
        if metadata.get("sample_id") != sample.get("sample_id"):
            raise ValueError("sample metadata identity mismatch")
        scene = sample["scene"]
        if scene not in source_cache:
            source_cache[scene] = (
                cassi_forward.load_npy_strict(DEFAULT_INPUT_DIR / f"{scene}.npy"),
                cassi_forward.load_npy_strict(
                    DEFAULT_VALIDITY_DIR / f"{scene}.hsi_valid_mask.npy"
                ),
            )
        stored_hsi, hsi_valid_mask = source_cache[scene]
        patch = cassi_forward.extract_unit_reflectance_patch(
            stored_hsi,
            hsi_valid_mask,
            top=sample["top"],
            left=sample["left"],
        )
        expected = cassi_forward.simulate_cassi_measurement(patch, physical_mask)
        measurement_path = output_dir / sample["measurement"]
        actual = cassi_forward.load_npy_strict(measurement_path, mmap_mode=None)
        if not np.array_equal(actual, expected):
            raise ValueError(f"measurement differs from forward recomputation: {sample['sample_id']}")
        if sha256_file(measurement_path) != (metadata.get("measurement") or {}).get("sha256"):
            raise ValueError(f"measurement SHA-256 mismatch: {sample['sample_id']}")
        if (metadata.get("physical_mask") or {}).get("sha256") != mask_metadata["sha256"]:
            raise ValueError(f"physical mask identity mismatch: {sample['sample_id']}")

    return {
        "status": "verified_existing",
        "scope": expected_scope,
        "sample_count": len(samples),
        "model_mask_sha256": sha256_file(model_mask_path),
    }


def verify_minimum_dataset(
    output_dir: Path = DEFAULT_MINIMUM_OUTPUT,
    *,
    mask_path: Path = DEFAULT_FORMAL_MASK,
    mask_metadata_path: Path = DEFAULT_FORMAL_MASK_META,
) -> dict[str, Any]:
    return verify_smoke_dataset(
        output_dir,
        expected_scope=SCOPE_LOCAL_MINIMUM,
        required_mask_role=MASK_ROLE_FORMAL,
        mask_path=mask_path,
        mask_metadata_path=mask_metadata_path,
    )


def verify_formal_dataset(
    output_dir: Path = DEFAULT_FORMAL_OUTPUT,
    *,
    mask_path: Path | None = None,
    mask_metadata_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    manifest = _read_json(output_dir / "manifest.json")
    split = manifest.get("split") or {}
    if manifest.get("split_id") != FORMAL_SPLIT_ID or split.get("split_id") != FORMAL_SPLIT_ID:
        raise ValueError("formal measurement manifest split identity mismatch")
    expected_counts = {"train_scenes": 202, "val_scenes": 15, "test_scenes": 35}
    for key, count in expected_counts.items():
        values = split.get(key)
        if (
            not isinstance(values, list)
            or len(values) != count
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"formal measurement manifest {key} must contain {count} scenes")
    role_sets = {key: set(split[key]) for key in expected_counts}
    if (
        role_sets["train_scenes"] & role_sets["val_scenes"]
        or role_sets["train_scenes"] & role_sets["test_scenes"]
        or role_sets["val_scenes"] & role_sets["test_scenes"]
    ):
        raise ValueError("formal measurement manifest roles overlap")
    manifest_scenes = {sample.get("scene") for sample in manifest.get("samples", [])}
    if set().union(*role_sets.values()) != manifest_scenes:
        raise ValueError("formal measurement manifest samples do not match split scenes")
    result = verify_smoke_dataset(
        output_dir,
        expected_scope=SCOPE_FORMAL,
        required_mask_role=MASK_ROLE_FORMAL,
        mask_path=mask_path,
        mask_metadata_path=mask_metadata_path,
    )
    result["split_id"] = FORMAL_SPLIT_ID
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_mask = subparsers.add_parser("create-dev-mask")
    create_mask.add_argument("--mask-path", type=Path, default=DEFAULT_DEV_MASK)
    create_mask.add_argument("--metadata-path", type=Path, default=DEFAULT_DEV_MASK_META)
    create_mask.add_argument("--seed", type=int, default=42)
    create_mask.add_argument("--open-probability", type=float, default=0.5)

    generate = subparsers.add_parser("generate-smoke")
    generate.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    generate.add_argument("--validity-dir", type=Path, default=DEFAULT_VALIDITY_DIR)
    generate.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    generate.add_argument("--mask-path", type=Path, default=DEFAULT_DEV_MASK)
    generate.add_argument("--mask-metadata-path", type=Path, default=DEFAULT_DEV_MASK_META)
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    generate.add_argument("--scene", action="append")
    generate.add_argument("--stride", type=int, default=256)
    generate.add_argument("--max-patches-per-scene", type=int, default=2)

    verify = subparsers.add_parser("verify-smoke")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_SMOKE_OUTPUT)

    minimum = subparsers.add_parser("generate-minimum")
    minimum.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    minimum.add_argument("--validity-dir", type=Path, default=DEFAULT_VALIDITY_DIR)
    minimum.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    minimum.add_argument("--mask-path", type=Path, default=DEFAULT_FORMAL_MASK)
    minimum.add_argument(
        "--mask-metadata-path", type=Path, default=DEFAULT_FORMAL_MASK_META
    )
    minimum.add_argument("--output-dir", type=Path, default=DEFAULT_MINIMUM_OUTPUT)
    minimum.add_argument("--scene", action="append")
    minimum.add_argument("--stride", type=int, default=256)
    minimum.add_argument("--max-patches-per-scene", type=int, default=2)

    verify_minimum = subparsers.add_parser("verify-minimum")
    verify_minimum.add_argument("--output-dir", type=Path, default=DEFAULT_MINIMUM_OUTPUT)
    verify_minimum.add_argument("--mask-path", type=Path, default=DEFAULT_FORMAL_MASK)
    verify_minimum.add_argument(
        "--mask-metadata-path", type=Path, default=DEFAULT_FORMAL_MASK_META
    )

    formal = subparsers.add_parser("generate-formal")
    formal.add_argument("--split-manifest", type=Path, default=DEFAULT_FORMAL_SPLIT)
    formal.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    formal.add_argument("--validity-dir", type=Path, default=DEFAULT_VALIDITY_DIR)
    formal.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    formal.add_argument("--mask-path", type=Path, default=DEFAULT_FORMAL_MASK)
    formal.add_argument(
        "--mask-metadata-path", type=Path, default=DEFAULT_FORMAL_MASK_META
    )
    formal.add_argument("--output-dir", type=Path, default=DEFAULT_FORMAL_OUTPUT)
    formal.add_argument("--stride", type=int, default=256)
    formal.add_argument(
        "--max-patches-per-scene",
        type=int,
        help="optional positive cap; omit to keep every valid grid patch",
    )

    verify_formal = subparsers.add_parser("verify-formal")
    verify_formal.add_argument("--output-dir", type=Path, default=DEFAULT_FORMAL_OUTPUT)
    verify_formal.add_argument("--mask-path", type=Path, default=DEFAULT_FORMAL_MASK)
    verify_formal.add_argument(
        "--mask-metadata-path", type=Path, default=DEFAULT_FORMAL_MASK_META
    )

    args = parser.parse_args(argv)
    if args.command == "create-dev-mask":
        result = create_development_mask(
            args.mask_path,
            args.metadata_path,
            seed=args.seed,
            open_probability=args.open_probability,
        )
    elif args.command == "generate-smoke":
        result = generate_smoke_dataset(
            input_dir=args.input_dir,
            validity_dir=args.validity_dir,
            label_dir=args.label_dir,
            mask_path=args.mask_path,
            mask_metadata_path=args.mask_metadata_path,
            output_dir=args.output_dir,
            scenes=args.scene,
            stride=args.stride,
            max_patches_per_scene=args.max_patches_per_scene,
        )
    elif args.command == "verify-smoke":
        result = verify_smoke_dataset(args.output_dir)
    elif args.command == "generate-minimum":
        result = generate_minimum_dataset(
            input_dir=args.input_dir,
            validity_dir=args.validity_dir,
            label_dir=args.label_dir,
            mask_path=args.mask_path,
            mask_metadata_path=args.mask_metadata_path,
            output_dir=args.output_dir,
            scenes=args.scene,
            stride=args.stride,
            max_patches_per_scene=args.max_patches_per_scene,
        )
    elif args.command == "verify-minimum":
        result = verify_minimum_dataset(
            args.output_dir,
            mask_path=args.mask_path,
            mask_metadata_path=args.mask_metadata_path,
        )
    elif args.command == "generate-formal":
        result = generate_formal_dataset(
            split_manifest_path=args.split_manifest,
            input_dir=args.input_dir,
            validity_dir=args.validity_dir,
            label_dir=args.label_dir,
            mask_path=args.mask_path,
            mask_metadata_path=args.mask_metadata_path,
            output_dir=args.output_dir,
            stride=args.stride,
            max_patches_per_scene=args.max_patches_per_scene,
        )
    else:
        result = verify_formal_dataset(
            args.output_dir,
            mask_path=args.mask_path,
            mask_metadata_path=args.mask_metadata_path,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
