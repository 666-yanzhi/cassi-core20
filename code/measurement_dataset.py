#!/usr/bin/env python3
"""Build and verify versioned CASSI measurement samples.

The CLI supports both the archived random-mask smoke workflow and the reviewed
three-scene minimum workflow that uses ``data/mask/mask.npy``. The latter is a
formal simulation baseline, but it remains a local behavior gate rather than a
formal 252-scene experiment.
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

MASK_ROLE_DEVELOPMENT = "development_smoke_only"
MASK_ROLE_FORMAL = "formal"
SCOPE_LOCAL_MINIMUM = "local_minimum_experiment"
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
    max_patches: int,
) -> list[tuple[int, int]]:
    validity = np.asarray(hsi_valid_mask)
    if validity.ndim != 2 or validity.dtype != np.bool_:
        raise ValueError("hsi_valid_mask must be an HW bool array")
    if stride <= 0 or max_patches <= 0:
        raise ValueError("stride and max_patches must be positive")
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
                if len(origins) == max_patches:
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
    max_patches_per_scene: int = 2,
    required_mask_role: str = MASK_ROLE_DEVELOPMENT,
    dataset_scope: str = MASK_ROLE_DEVELOPMENT,
    warning: str = "Development smoke data only; never report formal metrics from it.",
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
    else:
        result = verify_minimum_dataset(
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
