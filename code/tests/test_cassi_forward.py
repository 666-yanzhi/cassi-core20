from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import cassi_forward


def valid_patch(fill: float = 0.0) -> np.ndarray:
    return np.full(
        (
            cassi_forward.PATCH_SIZE,
            cassi_forward.PATCH_SIZE,
            cassi_forward.EXPECTED_BANDS,
        ),
        np.float32(fill),
        dtype=np.float32,
    )


def binary_mask(fill: float = 1.0) -> np.ndarray:
    return np.full(
        (cassi_forward.PATCH_SIZE, cassi_forward.PATCH_SIZE),
        np.float32(fill),
        dtype=np.float32,
    )


def test_zero_input_has_zero_measurement_and_fixed_shape() -> None:
    measurement = cassi_forward.simulate_cassi_measurement(
        valid_patch(), binary_mask(), measurement_scale=0.9
    )

    assert measurement.shape == (
        cassi_forward.PATCH_SIZE,
        cassi_forward.MEASUREMENT_WIDTH,
    )
    assert measurement.dtype == np.float32
    assert np.count_nonzero(measurement) == 0


def test_default_scale_is_fixed_and_model_mask_repeats_without_shift() -> None:
    patch = valid_patch()
    patch[0, 0, 0] = np.float32(1.0)
    physical = binary_mask(fill=0.0)
    physical[0, 0] = np.float32(1.0)
    physical[3, 5] = np.float32(1.0)

    measurement = cassi_forward.simulate_cassi_measurement(patch, physical)
    model_mask = cassi_forward.build_model_mask(physical)

    assert cassi_forward.MEASUREMENT_SCALE == 0.9
    assert cassi_forward.MODEL_MASK_POLICY == "repeat_unshifted_physical_mask_v1"
    assert measurement[0, 0] == pytest.approx(
        cassi_forward.MEASUREMENT_SCALE / cassi_forward.EXPECTED_BANDS
    )
    assert model_mask.shape == (84, 256, 256)
    assert model_mask.dtype == np.float32
    for channel in (0, 1, 42, 83):
        assert np.array_equal(model_mask[channel], physical)


def test_all_one_mask_matches_sparse_golden_values() -> None:
    patch = valid_patch()
    patch[7, 11, 0] = np.float32(0.25)
    patch[7, 11, 1] = np.float32(0.50)
    scale = 0.9

    measurement = cassi_forward.simulate_cassi_measurement(
        patch,
        binary_mask(),
        measurement_scale=scale,
    )

    expected = np.zeros_like(measurement)
    expected[7, 11] = np.float32(0.25 * scale / cassi_forward.EXPECTED_BANDS)
    expected[7, 13] = np.float32(0.50 * scale / cassi_forward.EXPECTED_BANDS)
    np.testing.assert_allclose(measurement, expected, rtol=1e-6, atol=1e-8)


def test_forward_is_deterministic_and_mask_modulates_before_dispersion() -> None:
    rng = np.random.default_rng(42)
    patch = rng.random(
        (
            cassi_forward.PATCH_SIZE,
            cassi_forward.PATCH_SIZE,
            cassi_forward.EXPECTED_BANDS,
        ),
        dtype=np.float32,
    )
    mask = rng.integers(
        0,
        2,
        size=(cassi_forward.PATCH_SIZE, cassi_forward.PATCH_SIZE),
        dtype=np.int8,
    ).astype(np.float32)

    first = cassi_forward.simulate_cassi_measurement(
        patch, mask, measurement_scale=0.9
    )
    second = cassi_forward.simulate_cassi_measurement(
        patch, mask, measurement_scale=0.9
    )
    zero_locations = mask == 0

    assert np.array_equal(first, second)
    only_zero_masked = valid_patch()
    only_zero_masked[~zero_locations] = 0
    blocked = cassi_forward.simulate_cassi_measurement(
        only_zero_masked, mask, measurement_scale=0.9
    )
    assert np.count_nonzero(blocked) == 0


def test_invalid_patch_and_mask_contracts_are_rejected() -> None:
    patch = valid_patch(0.5)
    mask = binary_mask()
    patch[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        cassi_forward.simulate_cassi_measurement(
            patch, mask, measurement_scale=0.9
        )

    patch[0, 0, 0] = np.float32(1.01)
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        cassi_forward.simulate_cassi_measurement(
            patch, mask, measurement_scale=0.9
        )

    patch[0, 0, 0] = np.float32(0.5)
    mask[0, 0] = np.float32(0.5)
    with pytest.raises(ValueError, match="binary"):
        cassi_forward.simulate_cassi_measurement(
            patch, mask, measurement_scale=0.9
        )


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.nan, np.inf])
def test_measurement_scale_must_be_explicitly_valid(bad_scale: float) -> None:
    with pytest.raises(ValueError, match="measurement_scale"):
        cassi_forward.simulate_cassi_measurement(
            valid_patch(), binary_mask(), measurement_scale=bad_scale
        )


def test_extract_patch_requires_every_hsi_pixel_to_be_valid() -> None:
    stored = np.full((4, 5, 84), np.float32(5000), dtype=np.float32)
    validity = np.ones((4, 5), dtype=np.bool_)

    patch = cassi_forward.extract_unit_reflectance_patch(
        stored, validity, top=1, left=2, patch_size=2
    )
    assert patch.shape == (2, 2, 84)
    assert patch.dtype == np.float32
    assert np.all(patch == np.float32(0.5))

    validity[2, 3] = False
    with pytest.raises(ValueError, match="1 invalid HSI pixels"):
        cassi_forward.extract_unit_reflectance_patch(
            stored, validity, top=1, left=2, patch_size=2
        )
    with pytest.raises(ValueError, match="outside HSI"):
        cassi_forward.extract_unit_reflectance_patch(
            stored, np.ones_like(validity), top=3, left=4, patch_size=2
        )


def test_load_npy_strict_disables_pickle(tmp_path: Path) -> None:
    numeric_path = tmp_path / "numeric.npy"
    np.save(numeric_path, np.arange(3, dtype=np.float32), allow_pickle=False)
    loaded = cassi_forward.load_npy_strict(numeric_path)
    assert np.array_equal(loaded, np.arange(3, dtype=np.float32))

    object_path = tmp_path / "object.npy"
    np.save(object_path, np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        cassi_forward.load_npy_strict(object_path, mmap_mode=None)
