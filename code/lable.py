#!/usr/bin/env python3
"""Generate reviewed_v3_core20 vegetation-index labels from HSI cubes.

The misspelled filename is retained as the compatibility entry point required by
the project protocol. Arrays on disk use HWC layout; invalid index values are
stored as NaN and accompanied by a per-index boolean validity mask.
"""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/derived/labels/reviewed_v3_core20"

FORMULA_VERSION = "reviewed_v3_core20"
STORAGE_SCALE = 10000.0
EXPECTED_BANDS = 84
WAVELENGTH_START_NM = 445
WAVELENGTH_STEP_NM = 5
EPSILON = 1e-6
REFLECTANCE_MIN = 0.0
REFLECTANCE_MAX = 1.0

INDEX_NAMES = (
    "NDVI",
    "EVI",
    "EVI2",
    "DVI",
    "SR",
    "SAVI",
    "OSAVI",
    "MSAVI2",
    "WDRVI",
    "GNDVI",
    "CIgreen",
    "NDRE705",
    "NDRE720",
    "CIrededge",
    "MTCI",
    "MCARI",
    "PRI",
    "PSRI",
    "ARI1",
    "CRI1",
)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file SHA-256 without loading the whole file into memory."""

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


def _band_valid(reflectance: np.ndarray, *bands: int) -> np.ndarray:
    selected = reflectance[..., bands]
    return (
        np.isfinite(selected)
        & (selected >= np.float32(REFLECTANCE_MIN))
        & (selected <= np.float32(REFLECTANCE_MAX))
    ).all(axis=-1)


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    band_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    domain = np.isfinite(denominator) & (np.abs(denominator) >= EPSILON)
    valid = band_mask & domain
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=valid)
    valid &= np.isfinite(result)
    result[~valid] = np.nan
    return result, valid


def _finite_value(
    value: np.ndarray,
    band_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(value, dtype=np.float32)
    valid = band_mask & np.isfinite(result)
    result = np.where(valid, result, np.float32(np.nan)).astype(np.float32, copy=False)
    return result, valid


def calculate_indices(stored_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the fixed 20-channel labels and per-index validity mask."""

    cube = np.asarray(stored_cube)
    _validate_source_array(cube)
    reflectance = cube.astype(np.float32, copy=False) / np.float32(STORAGE_SCALE)

    b5 = reflectance[..., 5]
    b11 = reflectance[..., 11]
    b13 = reflectance[..., 13]
    b17 = reflectance[..., 17]
    b21 = reflectance[..., 21]
    b25 = reflectance[..., 25]
    b45 = reflectance[..., 45]
    b47 = reflectance[..., 47]
    b51 = reflectance[..., 51]
    b52 = reflectance[..., 52]
    b53 = reflectance[..., 53]
    b55 = reflectance[..., 55]
    b61 = reflectance[..., 61]
    b62 = reflectance[..., 62]
    b71 = reflectance[..., 71]

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    def ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
        *bands: int,
    ) -> None:
        value, valid = _safe_ratio(
            numerator.astype(np.float32, copy=False),
            denominator.astype(np.float32, copy=False),
            _band_valid(reflectance, *bands),
        )
        values.append(value)
        masks.append(valid)

    def finite(value: np.ndarray, *bands: int) -> None:
        result, valid = _finite_value(value, _band_valid(reflectance, *bands))
        values.append(result)
        masks.append(valid)

    ratio(b71 - b45, b71 + b45, 71, 45)  # NDVI
    ratio(np.float32(2.5) * (b71 - b45), b71 + 6 * b45 - 7.5 * b5 + 1, 71, 45, 5)
    ratio(np.float32(2.5) * (b71 - b45), b71 + 2.4 * b45 + 1, 71, 45)
    finite(b71 - b45, 71, 45)  # DVI
    ratio(b71, b45, 71, 45)  # SR
    ratio(np.float32(1.5) * (b71 - b45), b71 + b45 + 0.5, 71, 45)
    ratio(np.float32(1.16) * (b71 - b45), b71 + b45 + 0.16, 71, 45)

    # MSAVI2 has a square-root domain in addition to band validity.
    radicand = (2 * b71 + 1) ** 2 - 8 * (b71 - b45)
    msavi_band_mask = _band_valid(reflectance, 71, 45)
    msavi_domain = np.isfinite(radicand) & (radicand >= -EPSILON)
    safe_radicand = np.maximum(radicand, np.float32(0.0))
    with np.errstate(invalid="ignore"):
        msavi = (2 * b71 + 1 - np.sqrt(safe_radicand)) / 2
    msavi_valid = msavi_band_mask & msavi_domain & np.isfinite(msavi)
    msavi = np.where(msavi_valid, msavi, np.float32(np.nan)).astype(np.float32)
    values.append(msavi)
    masks.append(msavi_valid)

    ratio(0.1 * b71 - b45, 0.1 * b71 + b45, 71, 45)  # WDRVI
    ratio(b71 - b21, b71 + b21, 71, 21)  # GNDVI
    ratio(b71, b21, 71, 21)  # CIgreen, subtract one below
    values[-1] = np.where(masks[-1], values[-1] - 1, np.float32(np.nan)).astype(np.float32)
    ratio(b71 - b52, b71 + b52, 71, 52)  # NDRE705
    ratio(b71 - b55, b71 + b55, 71, 55)  # NDRE720

    numerator_mean = reflectance[..., 61:72].mean(axis=-1, dtype=np.float32)
    denominator_mean = reflectance[..., 53:58].mean(axis=-1, dtype=np.float32)
    ratio(numerator_mean, denominator_mean, *range(53, 58), *range(61, 72))
    values[-1] = np.where(masks[-1], values[-1] - 1, np.float32(np.nan)).astype(np.float32)

    ratio(b62 - b53, b53 - b47, 62, 53, 47)  # MTCI
    ratio((b51 - b45) - 0.2 * (b51 - b21), b45, 51, 45, 21)
    values[-1] = np.where(masks[-1], values[-1] * b51, np.float32(np.nan)).astype(np.float32)
    masks[-1] &= np.isfinite(values[-1])
    ratio(b17 - b25, b17 + b25, 17, 25)  # PRI
    ratio(b47 - b11, b61, 47, 11, 61)  # PSRI
    ratio(b51 - b21, b21 * b51, 21, 51)  # ARI1
    ratio(b21 - b13, b13 * b21, 13, 21)  # CRI1

    indices = np.stack(values, axis=-1).astype(np.float32, copy=False)
    valid_mask = np.stack(masks, axis=-1).astype(np.bool_, copy=False)
    if indices.shape[-1] != len(INDEX_NAMES):
        raise AssertionError("internal formula count does not match channel contract")
    indices[~valid_mask] = np.nan
    return indices, valid_mask


def artifact_paths(source_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = Path(source_path).stem
    output = Path(output_dir)
    return (
        output / f"{stem}.indices.npy",
        output / f"{stem}.index_valid_mask.npy",
        output / f"{stem}.meta.json",
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata root must be an object: {path}")
    return payload


def verify_artifact(
    source_path: Path,
    output_dir: Path,
    *,
    tile_rows: int = 64,
) -> dict[str, Any]:
    """Verify array contracts, values, hashes, metadata, and source identity."""

    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    source_path = Path(source_path).resolve()
    indices_path, mask_path, metadata_path = artifact_paths(source_path, output_dir)
    missing = [
        str(path)
        for path in (indices_path, mask_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"incomplete label artifact; missing: {missing}")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    _validate_source_array(source, source_path)
    indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (*source.shape[:2], len(INDEX_NAMES))
    if indices.shape != expected_shape or indices.dtype != np.float32:
        raise ValueError(f"invalid indices contract: shape={indices.shape}, dtype={indices.dtype}")
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"invalid index mask contract: shape={mask.shape}, dtype={mask.dtype}")

    for start in range(0, source.shape[0], tile_rows):
        stop = min(start + tile_rows, source.shape[0])
        expected_indices, expected_mask = calculate_indices(source[start:stop])
        if not np.array_equal(mask[start:stop], expected_mask):
            raise ValueError(f"index validity mask differs from source rows {start}:{stop}")
        if not np.array_equal(indices[start:stop], expected_indices, equal_nan=True):
            raise ValueError(f"indices differ from source rows {start}:{stop}")

    metadata = _load_metadata(metadata_path)
    source_sha256 = sha256_file(source_path)
    indices_sha256 = sha256_file(indices_path)
    mask_sha256 = sha256_file(mask_path)
    required = {
        "schema_version": 1,
        "formula_version": FORMULA_VERSION,
        "layout": "HWC",
        "shape": list(expected_shape),
        "indices_dtype": "float32",
        "index_valid_mask_dtype": "bool",
        "channel_order": list(INDEX_NAMES),
        "source_storage_scale": STORAGE_SCALE,
        "wavelengths_nm": [
            WAVELENGTH_START_NM + WAVELENGTH_STEP_NM * index
            for index in range(EXPECTED_BANDS)
        ],
        "epsilon": EPSILON,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}")

    hashes = {
        "source.sha256": ((metadata.get("source") or {}).get("sha256"), source_sha256),
        "indices.sha256": ((metadata.get("indices") or {}).get("sha256"), indices_sha256),
        "index_valid_mask.sha256": (
            (metadata.get("index_valid_mask") or {}).get("sha256"),
            mask_sha256,
        ),
    }
    for key, (recorded, actual) in hashes.items():
        if recorded != actual:
            raise ValueError(f"metadata hash mismatch for {key}: {recorded!r} != {actual!r}")

    invalid_counts = np.count_nonzero(~mask, axis=(0, 1)).astype(np.int64).tolist()
    if (metadata.get("statistics") or {}).get("invalid_count_by_channel") != invalid_counts:
        raise ValueError("metadata invalid_count_by_channel does not match mask")

    return {
        "scene": source_path.stem,
        "action": "verified_existing",
        "shape": list(expected_shape),
        "source_sha256": source_sha256,
        "indices_sha256": indices_sha256,
        "index_valid_mask_sha256": mask_sha256,
        "artifact_bytes": indices_path.stat().st_size + mask_path.stat().st_size,
    }


def generate_artifact(
    source_path: Path,
    output_dir: Path,
    *,
    tile_rows: int = 64,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Generate one scene atomically, committing metadata last."""

    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.name != FORMULA_VERSION:
        raise ValueError(f"output directory must be named {FORMULA_VERSION}")
    output_dir.mkdir(parents=True, exist_ok=True)
    indices_path, mask_path, metadata_path = artifact_paths(source_path, output_dir)
    existing = [path for path in (indices_path, mask_path, metadata_path) if path.exists()]
    if existing and not replace_existing:
        raise FileExistsError(f"label artifact already exists: {[str(path) for path in existing]}")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    _validate_source_array(source, source_path)
    before_fingerprint = _source_fingerprint(source_path)
    source_sha256_before = sha256_file(source_path)
    token = uuid.uuid4().hex
    temporary_indices = output_dir / f".{indices_path.name}.{token}.tmp"
    temporary_mask = output_dir / f".{mask_path.name}.{token}.tmp"
    started = time.monotonic()
    invalid_counts = np.zeros(len(INDEX_NAMES), dtype=np.int64)

    try:
        indices_memmap = np.lib.format.open_memmap(
            temporary_indices,
            mode="w+",
            dtype=np.float32,
            shape=(*source.shape[:2], len(INDEX_NAMES)),
            fortran_order=False,
        )
        mask_memmap = np.lib.format.open_memmap(
            temporary_mask,
            mode="w+",
            dtype=np.bool_,
            shape=(*source.shape[:2], len(INDEX_NAMES)),
            fortran_order=False,
        )
        for start in range(0, source.shape[0], tile_rows):
            stop = min(start + tile_rows, source.shape[0])
            tile_indices, tile_mask = calculate_indices(source[start:stop])
            indices_memmap[start:stop] = tile_indices
            mask_memmap[start:stop] = tile_mask
            invalid_counts += np.count_nonzero(~tile_mask, axis=(0, 1))
        indices_memmap.flush()
        mask_memmap.flush()
        del indices_memmap, mask_memmap

        after_fingerprint = _source_fingerprint(source_path)
        source_sha256_after = sha256_file(source_path)
        if before_fingerprint != after_fingerprint or source_sha256_before != source_sha256_after:
            raise RuntimeError(f"source HSI changed during generation: {source_path}")

        indices_sha256 = sha256_file(temporary_indices)
        mask_sha256 = sha256_file(temporary_mask)
        elapsed = time.monotonic() - started
        output_shape = [*source.shape[:2], len(INDEX_NAMES)]
        metadata = {
            "schema_version": 1,
            "formula_version": FORMULA_VERSION,
            "layout": "HWC",
            "shape": output_shape,
            "indices_dtype": "float32",
            "index_valid_mask_dtype": "bool",
            "channel_order": list(INDEX_NAMES),
            "source_storage_scale": STORAGE_SCALE,
            "reflectance_valid_range": [REFLECTANCE_MIN, REFLECTANCE_MAX],
            "wavelengths_nm": [
                WAVELENGTH_START_NM + WAVELENGTH_STEP_NM * index
                for index in range(EXPECTED_BANDS)
            ],
            "epsilon": EPSILON,
            "invalid_value": "NaN",
            "source": {
                "path": _recorded_path(source_path),
                "filename": source_path.name,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "bytes": source_path.stat().st_size,
                "sha256": source_sha256_after,
            },
            "indices": {
                "path": _recorded_path(indices_path),
                "bytes": temporary_indices.stat().st_size,
                "sha256": indices_sha256,
            },
            "index_valid_mask": {
                "path": _recorded_path(mask_path),
                "bytes": temporary_mask.stat().st_size,
                "sha256": mask_sha256,
            },
            "generation": {"tile_rows": tile_rows, "elapsed_seconds": elapsed},
            "statistics": {
                "invalid_count_by_channel": invalid_counts.tolist(),
                "invalid_count_by_name": dict(zip(INDEX_NAMES, invalid_counts.tolist())),
            },
            "approximations": {
                "MTCI": [755, 710, 680],
                "PRI": [530, 570],
            },
        }

        metadata_path.unlink(missing_ok=True)
        os.replace(temporary_indices, indices_path)
        os.replace(temporary_mask, mask_path)
        _atomic_write_json(metadata_path, metadata)
        return {
            "scene": source_path.stem,
            "action": "replaced" if existing else "generated",
            "shape": output_shape,
            "elapsed_seconds": elapsed,
            "source_sha256": source_sha256_after,
            "indices_sha256": indices_sha256,
            "index_valid_mask_sha256": mask_sha256,
            "artifact_bytes": indices_path.stat().st_size + mask_path.stat().st_size,
        }
    finally:
        temporary_indices.unlink(missing_ok=True)
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
    parser.add_argument("--scene", action="append", help="filename or stem; repeatable")
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument(
        "--existing",
        choices=("error", "verify", "replace"),
        default="error",
        help="refuse, verify, or explicitly replace existing artifacts",
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

    print(
        json.dumps(
            {
                "status": "complete",
                "formula_version": FORMULA_VERSION,
                "mode": args.existing,
                "scene_count": len(results),
                "total_artifact_bytes": sum(int(item["artifact_bytes"]) for item in results),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
