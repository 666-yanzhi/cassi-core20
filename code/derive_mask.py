#!/usr/bin/env python3
"""Derive and verify the binary project aperture from a legacy mask.

The legacy project mask is a continuous transmission map. The new protocol
requires a binary 256x256 float32 aperture, so this tool preserves the spatial
ranking of the source while opening exactly the highest-transmission half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data/mask/source/legacy_project_mask.npy"
DEFAULT_OUTPUT = REPO_ROOT / "data/mask/mask.npy"
DEFAULT_METADATA = REPO_ROOT / "data/mask/mask.meta.json"

MASK_VERSION = "legacy_top_half_binary_v1"
ROLE = "formal"
USAGE = "formal_simulation_baseline"
PHYSICAL_STATUS = "not_hardware_calibrated"
EXPECTED_SHAPE = (256, 256)


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


def _validate_source(source: np.ndarray) -> None:
    if source.shape != EXPECTED_SHAPE:
        raise ValueError(f"legacy mask must have shape {EXPECTED_SHAPE}, got {source.shape}")
    if source.dtype != np.float32:
        raise TypeError(f"legacy mask must have dtype float32, got {source.dtype}")
    if not np.isfinite(source).all():
        raise ValueError("legacy mask contains NaN or Inf")
    if np.any(source < 0) or np.any(source > 1):
        raise ValueError("legacy mask values must be within [0,1]")


def derive_binary_mask(source: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Open exactly the highest-valued half using deterministic C-order ranking."""

    source = np.asarray(source)
    _validate_source(source)
    flat = source.reshape(-1, order="C")
    open_count = flat.size // 2
    order = np.argsort(flat, kind="stable")
    selected = order[-open_count:]
    binary_flat = np.zeros(flat.shape, dtype=np.float32)
    binary_flat[selected] = np.float32(1.0)
    binary = binary_flat.reshape(source.shape, order="C")

    closed_boundary = float(flat[order[-open_count - 1]])
    open_boundary = float(flat[order[-open_count]])
    boundary_tie_count = int(np.count_nonzero(flat == open_boundary))
    details = {
        "method": "top_half_by_value",
        "rank_order": "ascending_stable",
        "flatten_order": "C",
        "tie_breaker": "later_C_order_positions_open_first",
        "target_open_fraction": 0.5,
        "closed_max_source_value": closed_boundary,
        "open_min_source_value": open_boundary,
        "boundary_tie_count": boundary_tie_count,
    }
    return binary, details


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


def _read_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mask metadata root must be an object")
    return payload


def generate_artifact(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing and not replace_existing:
        raise FileExistsError(f"mask artifact already exists: {[str(path) for path in existing]}")

    source = np.load(source_path, allow_pickle=False)
    binary, transform = derive_binary_mask(source)
    source_sha256 = sha256_file(source_path)
    _atomic_save_npy(output_path, binary)
    output_sha256 = sha256_file(output_path)
    open_count = int(np.count_nonzero(binary))
    metadata = {
        "schema_version": 1,
        "mask_version": MASK_VERSION,
        "mask_id": MASK_VERSION,
        "role": ROLE,
        "usage": USAGE,
        "physical_status": PHYSICAL_STATUS,
        "shape": list(EXPECTED_SHAPE),
        "dtype": "float32",
        "binary": True,
        "allowed_values": [0.0, 1.0],
        "open_value": 1.0,
        "closed_value": 0.0,
        "open_count": open_count,
        "closed_count": int(binary.size - open_count),
        "open_fraction": open_count / binary.size,
        "sha256": output_sha256,
        "path": _recorded_path(output_path),
        "source": {
            "path": _recorded_path(source_path),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "sha256": source_sha256,
            "min": float(source.min()),
            "max": float(source.max()),
            "mean": float(source.mean(dtype=np.float64)),
            "unique_value_count": int(np.unique(source).size),
        },
        "transformation": transform,
        "warning": (
            "Formal simulation baseline derived from a legacy continuous mask; "
            "not a calibrated physical-aperture measurement."
        ),
    }
    _atomic_write_json(metadata_path, metadata)
    return {
        "status": "replaced" if existing else "generated",
        "mask_version": MASK_VERSION,
        "source_sha256": source_sha256,
        "mask_sha256": output_sha256,
        "open_fraction": metadata["open_fraction"],
    }


def verify_artifact(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    for path in (source_path, output_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = np.load(source_path, allow_pickle=False)
    output = np.load(output_path, allow_pickle=False)
    expected, transform = derive_binary_mask(source)
    if output.shape != EXPECTED_SHAPE or output.dtype != np.float32:
        raise ValueError(f"derived mask contract mismatch: {output.shape}, {output.dtype}")
    if not np.array_equal(output, expected):
        raise ValueError("derived mask differs from deterministic source transformation")

    metadata = _read_metadata(metadata_path)
    required = {
        "schema_version": 1,
        "mask_version": MASK_VERSION,
        "mask_id": MASK_VERSION,
        "role": ROLE,
        "usage": USAGE,
        "physical_status": PHYSICAL_STATUS,
        "shape": list(EXPECTED_SHAPE),
        "dtype": "float32",
        "binary": True,
        "open_fraction": 0.5,
        "transformation": transform,
    }
    for key, expected_value in required.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"metadata mismatch for {key}: {metadata.get(key)!r} != {expected_value!r}"
            )
    source_sha256 = sha256_file(source_path)
    output_sha256 = sha256_file(output_path)
    if (metadata.get("source") or {}).get("sha256") != source_sha256:
        raise ValueError("source SHA-256 does not match metadata")
    if metadata.get("sha256") != output_sha256:
        raise ValueError("derived mask SHA-256 does not match metadata")
    return {
        "status": "verified_existing",
        "mask_version": MASK_VERSION,
        "source_sha256": source_sha256,
        "mask_sha256": output_sha256,
        "open_fraction": float(output.mean(dtype=np.float64)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--existing",
        choices=("error", "verify", "replace"),
        default="error",
    )
    args = parser.parse_args(argv)
    if args.existing == "verify":
        result = verify_artifact(args.source, args.output, args.metadata)
    else:
        result = generate_artifact(
            args.source,
            args.output,
            args.metadata,
            replace_existing=args.existing == "replace",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
