#!/usr/bin/env python3
"""Strict CASSI optical-forward primitives for the core20 protocol.

This module intentionally separates the physical measurement simulation from
the neural-network model-mask representation. The latter remains a protocol
decision and must not be inferred from the historical implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


FORWARD_VERSION = "cassi_forward_v1"
STORAGE_SCALE = 10000.0
EXPECTED_BANDS = 84
PATCH_SIZE = 256
DISPERSION_STEP = 2
MEASUREMENT_WIDTH = PATCH_SIZE + DISPERSION_STEP * (EXPECTED_BANDS - 1)
MEASUREMENT_SCALE = 0.9
MODEL_MASK_POLICY = "repeat_unshifted_physical_mask_v1"


def _validate_stored_hsi(cube: np.ndarray) -> None:
    if cube.ndim != 3 or cube.shape[-1] != EXPECTED_BANDS:
        raise ValueError(f"stored HSI must have shape (H, W, 84), got {cube.shape}")
    if cube.dtype != np.float32:
        raise TypeError(f"stored HSI must have dtype float32, got {cube.dtype}")


def extract_unit_reflectance_patch(
    stored_hsi: np.ndarray,
    hsi_valid_mask: np.ndarray,
    *,
    top: int,
    left: int,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Extract an all-valid HWC patch and convert stored values to reflectance.

    The source array is never modified or clipped. A patch containing even one
    invalid spatial pixel is rejected before it can enter the optical forward.
    """

    cube = np.asarray(stored_hsi)
    validity = np.asarray(hsi_valid_mask)
    _validate_stored_hsi(cube)
    if validity.shape != cube.shape[:2] or validity.dtype != np.bool_:
        raise ValueError(
            "hsi_valid_mask must be an HW bool array matching the HSI; "
            f"got shape={validity.shape}, dtype={validity.dtype}"
        )
    if not isinstance(top, int) or not isinstance(left, int):
        raise TypeError("top and left must be integers")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    bottom = top + patch_size
    right = left + patch_size
    if top < 0 or left < 0 or bottom > cube.shape[0] or right > cube.shape[1]:
        raise ValueError(
            f"patch [{top}:{bottom}, {left}:{right}] is outside HSI shape {cube.shape[:2]}"
        )
    patch_validity = validity[top:bottom, left:right]
    if not bool(patch_validity.all()):
        invalid_count = int(np.count_nonzero(~patch_validity))
        raise ValueError(f"patch contains {invalid_count} invalid HSI pixels")
    return (
        cube[top:bottom, left:right].astype(np.float32, copy=False)
        / np.float32(STORAGE_SCALE)
    )


def validate_physical_mask(
    physical_mask: np.ndarray,
    *,
    expected_size: int = PATCH_SIZE,
) -> float:
    """Validate a binary float32 physical aperture and return its open fraction."""

    mask = np.asarray(physical_mask)
    expected_shape = (expected_size, expected_size)
    if mask.shape != expected_shape:
        raise ValueError(f"physical mask must have shape {expected_shape}, got {mask.shape}")
    if mask.dtype != np.float32:
        raise TypeError(f"physical mask must have dtype float32, got {mask.dtype}")
    if not np.isfinite(mask).all():
        raise ValueError("physical mask contains NaN or Inf")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("physical mask must contain only binary 0/1 values")
    return float(mask.mean(dtype=np.float64))


def build_model_mask(physical_mask: np.ndarray) -> np.ndarray:
    """Repeat the unshifted aperture over 84 channels in CHW layout.

    ``initial_x`` in the model extracts one measurement window per spectral
    offset and maps it back to the original patch coordinates. The unshifted
    physical aperture is therefore the aligned condition at every channel.
    """

    validate_physical_mask(physical_mask)
    return np.broadcast_to(
        physical_mask[None, ...],
        (EXPECTED_BANDS, PATCH_SIZE, PATCH_SIZE),
    ).copy()


def simulate_cassi_measurement(
    unit_reflectance_patch: np.ndarray,
    physical_mask: np.ndarray,
    *,
    measurement_scale: float = MEASUREMENT_SCALE,
    dispersion_step: int = DISPERSION_STEP,
) -> np.ndarray:
    """Apply modulation, spectral dispersion, and channel summation.

    For formal data the input contract is fixed to a 256x256x84 unit-reflectance
    patch and a 256x256 binary aperture. The result is a 256x422 float32 array
    when ``dispersion_step=2``. The fixed ``measurement_scale=0.9`` is the
    historical simulation convention, not a claimed calibrated throughput.
    """

    patch = np.asarray(unit_reflectance_patch)
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, EXPECTED_BANDS):
        raise ValueError(
            "unit-reflectance patch must have shape "
            f"({PATCH_SIZE}, {PATCH_SIZE}, {EXPECTED_BANDS}), got {patch.shape}"
        )
    if patch.dtype != np.float32:
        raise TypeError(f"unit-reflectance patch must have dtype float32, got {patch.dtype}")
    if not np.isfinite(patch).all():
        raise ValueError("unit-reflectance patch contains NaN or Inf")
    if np.any(patch < 0) or np.any(patch > 1):
        raise ValueError("unit-reflectance patch contains values outside [0,1]")
    validate_physical_mask(physical_mask)
    if not isinstance(dispersion_step, int) or dispersion_step <= 0:
        raise ValueError("dispersion_step must be a positive integer")
    if not np.isfinite(measurement_scale) or measurement_scale <= 0:
        raise ValueError("measurement_scale must be finite and positive")

    width = PATCH_SIZE + dispersion_step * (EXPECTED_BANDS - 1)
    measurement = np.zeros((PATCH_SIZE, width), dtype=np.float32)
    modulated = patch * physical_mask[..., None]
    channel_scale = np.float32(measurement_scale / EXPECTED_BANDS)
    for channel in range(EXPECTED_BANDS):
        offset = channel * dispersion_step
        measurement[:, offset : offset + PATCH_SIZE] += (
            modulated[..., channel] * channel_scale
        )
    if not np.isfinite(measurement).all():
        raise RuntimeError("CASSI forward produced a non-finite measurement")
    return measurement


def load_npy_strict(path: Path, *, mmap_mode: str | None = "r") -> np.ndarray:
    """Load a non-pickle NPY file through one auditable helper."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return np.load(resolved, mmap_mode=mmap_mode, allow_pickle=False)
