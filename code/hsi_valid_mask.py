#!/usr/bin/env python3
"""Generate per-scene HSI validity masks for CASSI patch selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/raw/HSI"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/derived/validity/hsi_reflectance_v1"

VALIDITY_VERSION = "hsi_reflectance_v1"
STORAGE_SCALE = 10000.0
EXPECTED_BANDS = 84
REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 1.0


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the full file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> tuple[int, int, int]:
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _recorded_path(path: Path) -> str:
    """Prefer a portable project-relative path for project-owned artifacts."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _validate_source_array(source: np.ndarray, source_path: Path | None = None) -> None:
    label = str(source_path) if source_path is not None else "source"
    if source.ndim != 3 or source.shape[-1] != EXPECTED_BANDS:
        raise ValueError(f"{label} must have shape (H, W, 84), got {source.shape}")
    if source.dtype != np.float32:
        raise TypeError(f"{label} must have dtype float32, got {source.dtype}")


def calculate_hsi_valid_mask(stored_cube: np.ndarray) -> np.ndarray:
    """Return an HW bool mask where all 84 unit-reflectance bands are valid."""

    cube = np.asarray(stored_cube)
    _validate_source_array(cube)
    reflectance = cube.astype(np.float32, copy=False) / np.float32(STORAGE_SCALE)
    return (
        np.isfinite(reflectance)
        & (reflectance >= np.float32(REFLECTANCE_MIN))
        & (reflectance <= np.float32(REFLECTANCE_MAX))
    ).all(axis=-1)


def artifact_paths(source_path: Path, output_dir: Path) -> tuple[Path, Path]:
    stem = Path(source_path).stem
    output = Path(output_dir)
    return (
        output / f"{stem}.hsi_valid_mask.npy",
        output / f"{stem}.hsi_valid_mask.meta.json",
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_artifact(
    source_path: Path,
    output_dir: Path,
    *,
    tile_rows: int = 64,
) -> dict[str, Any]:
    """Verify structure, metadata, hashes, and values against the source HSI."""

    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    source_path = Path(source_path).resolve()
    mask_path, metadata_path = artifact_paths(source_path, output_dir)
    missing = [str(path) for path in (mask_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete HSI validity artifact; missing: {missing}")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    _validate_source_array(source, source_path)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    expected_shape = source.shape[:2]
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"invalid HSI mask contract: shape={mask.shape}, dtype={mask.dtype}")

    for start in range(0, source.shape[0], tile_rows):
        stop = min(start + tile_rows, source.shape[0])
        expected = calculate_hsi_valid_mask(source[start:stop])
        if not np.array_equal(mask[start:stop], expected):
            raise ValueError(f"HSI validity mask differs from source rows {start}:{stop}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(source_path)
    mask_sha256 = sha256_file(mask_path)
    required_metadata = {
        "schema_version": 1,
        "validity_version": VALIDITY_VERSION,
        "layout": "HW",
        "shape": list(expected_shape),
        "dtype": "bool",
        "source_storage_scale": STORAGE_SCALE,
        "reflectance_valid_range": [REFLECTANCE_MIN, REFLECTANCE_MAX],
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    hash_expectations = {
        "source.sha256": ((metadata.get("source") or {}).get("sha256"), source_sha256),
        "valid_mask.sha256": (
            (metadata.get("valid_mask") or {}).get("sha256"),
            mask_sha256,
        ),
    }
    for key, (recorded, actual) in hash_expectations.items():
        if recorded != actual:
            raise ValueError(f"metadata hash mismatch for {key}: {recorded!r} != {actual!r}")

    valid_count = int(np.count_nonzero(mask))
    invalid_count = int(mask.size - valid_count)
    statistics = metadata.get("statistics") or {}
    if statistics.get("valid_pixel_count") != valid_count:
        raise ValueError("metadata valid_pixel_count does not match mask")
    if statistics.get("invalid_pixel_count") != invalid_count:
        raise ValueError("metadata invalid_pixel_count does not match mask")

    return {
        "scene": source_path.stem,
        "action": "verified_existing",
        "shape": list(expected_shape),
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": invalid_count,
        "source_sha256": source_sha256,
        "valid_mask_sha256": mask_sha256,
        "valid_mask_bytes": mask_path.stat().st_size,
    }


def generate_artifact(
    source_path: Path,
    output_dir: Path,
    *,
    tile_rows: int = 64,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Generate one mask atomically, writing metadata last as completion marker."""

    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.name != VALIDITY_VERSION:
        raise ValueError(f"output directory must be named {VALIDITY_VERSION}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path, metadata_path = artifact_paths(source_path, output_dir)
    existing = [path for path in (mask_path, metadata_path) if path.exists()]
    if existing and not replace_existing:
        raise FileExistsError(
            f"HSI validity artifact already exists: {[str(path) for path in existing]}"
        )

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    _validate_source_array(source, source_path)
    before_fingerprint = _source_fingerprint(source_path)
    source_sha256_before = sha256_file(source_path)
    temporary_mask = output_dir / f".{mask_path.name}.{uuid.uuid4().hex}.tmp"
    started = time.monotonic()
    valid_count = 0

    try:
        mask_memmap = np.lib.format.open_memmap(
            temporary_mask,
            mode="w+",
            dtype=np.bool_,
            shape=source.shape[:2],
            fortran_order=False,
        )
        for start in range(0, source.shape[0], tile_rows):
            stop = min(start + tile_rows, source.shape[0])
            tile_mask = calculate_hsi_valid_mask(source[start:stop])
            mask_memmap[start:stop] = tile_mask
            valid_count += int(np.count_nonzero(tile_mask))
        mask_memmap.flush()
        del mask_memmap

        after_fingerprint = _source_fingerprint(source_path)
        source_sha256_after = sha256_file(source_path)
        if before_fingerprint != after_fingerprint or source_sha256_before != source_sha256_after:
            raise RuntimeError(f"source HSI changed during generation: {source_path}")

        mask_sha256 = sha256_file(temporary_mask)
        total_count = int(source.shape[0] * source.shape[1])
        invalid_count = total_count - valid_count
        elapsed = time.monotonic() - started
        metadata = {
            "schema_version": 1,
            "validity_version": VALIDITY_VERSION,
            "layout": "HW",
            "shape": list(source.shape[:2]),
            "dtype": "bool",
            "source_storage_scale": STORAGE_SCALE,
            "reflectance_valid_range": [REFLECTANCE_MIN, REFLECTANCE_MAX],
            "validity_policy": (
                "all 84 unit-reflectance bands are finite and in [0,1]"
            ),
            "source": {
                "path": _recorded_path(source_path),
                "filename": source_path.name,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "bytes": source_path.stat().st_size,
                "sha256": source_sha256_after,
            },
            "valid_mask": {
                "path": _recorded_path(mask_path),
                "bytes": temporary_mask.stat().st_size,
                "sha256": mask_sha256,
            },
            "generation": {
                "tile_rows": tile_rows,
                "elapsed_seconds": elapsed,
            },
            "statistics": {
                "total_spatial_pixels": total_count,
                "valid_pixel_count": valid_count,
                "invalid_pixel_count": invalid_count,
                "valid_fraction": valid_count / total_count,
            },
        }

        metadata_path.unlink(missing_ok=True)
        os.replace(temporary_mask, mask_path)
        _atomic_write_json(metadata_path, metadata)
        return {
            "scene": source_path.stem,
            "action": "replaced" if existing else "generated",
            "shape": list(source.shape[:2]),
            "elapsed_seconds": elapsed,
            "valid_pixel_count": valid_count,
            "invalid_pixel_count": invalid_count,
            "source_sha256": source_sha256_after,
            "valid_mask_sha256": mask_sha256,
            "valid_mask_bytes": mask_path.stat().st_size,
        }
    finally:
        temporary_mask.unlink(missing_ok=True)


def _discover_sources(input_dir: Path, scenes: list[str] | None) -> list[Path]:
    input_dir = Path(input_dir).resolve()
    if scenes:
        selected: list[Path] = []
        for requested in dict.fromkeys(scenes):
            name = Path(requested).name
            if name != requested:
                raise ValueError(f"scene must be a filename or stem, not a path: {requested}")
            filename = name if name.endswith(".npy") else f"{name}.npy"
            candidate = input_dir / filename
            if not candidate.is_file():
                raise FileNotFoundError(f"scene does not exist: {candidate}")
            selected.append(candidate)
        return selected
    sources = sorted(input_dir.glob("hsi_*.npy"))
    if not sources:
        raise FileNotFoundError(f"no hsi_*.npy files found in {input_dir}")
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scene",
        action="append",
        help="source filename or stem; repeat to select multiple scenes",
    )
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument(
        "--existing",
        choices=("error", "verify", "replace"),
        default="error",
        help="refuse existing output, verify it, or explicitly regenerate it",
    )
    args = parser.parse_args(argv)
    if args.tile_rows <= 0:
        parser.error("--tile-rows must be positive")

    sources = _discover_sources(args.input_dir, args.scene)
    results: list[dict[str, Any]] = []
    for source_path in sources:
        if args.existing == "verify":
            result = verify_artifact(source_path, args.output_dir, tile_rows=args.tile_rows)
        else:
            result = generate_artifact(
                source_path,
                args.output_dir,
                tile_rows=args.tile_rows,
                replace_existing=args.existing == "replace",
            )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "status": "complete",
        "validity_version": VALIDITY_VERSION,
        "mode": args.existing,
        "scene_count": len(results),
        "total_valid_mask_bytes": sum(
            int(result["valid_mask_bytes"]) for result in results
        ),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
