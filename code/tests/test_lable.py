from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import lable


def stored_cube(height: int = 2, width: int = 3) -> np.ndarray:
    spectrum = np.linspace(0.12, 0.78, lable.EXPECTED_BANDS, dtype=np.float32)
    cube = np.broadcast_to(spectrum, (height, width, lable.EXPECTED_BANDS)).copy()
    return cube * np.float32(lable.STORAGE_SCALE)


def test_fixed_channel_order_and_golden_formulas() -> None:
    cube = stored_cube(1, 1)
    indices, valid = lable.calculate_indices(cube)
    rho = cube[0, 0] / np.float32(lable.STORAGE_SCALE)

    expected = np.array(
        [
            (rho[71] - rho[45]) / (rho[71] + rho[45]),
            2.5 * (rho[71] - rho[45]) / (rho[71] + 6 * rho[45] - 7.5 * rho[5] + 1),
            2.5 * (rho[71] - rho[45]) / (rho[71] + 2.4 * rho[45] + 1),
            rho[71] - rho[45],
            rho[71] / rho[45],
            1.5 * (rho[71] - rho[45]) / (rho[71] + rho[45] + 0.5),
            1.16 * (rho[71] - rho[45]) / (rho[71] + rho[45] + 0.16),
            (2 * rho[71] + 1 - np.sqrt((2 * rho[71] + 1) ** 2 - 8 * (rho[71] - rho[45]))) / 2,
            (0.1 * rho[71] - rho[45]) / (0.1 * rho[71] + rho[45]),
            (rho[71] - rho[21]) / (rho[71] + rho[21]),
            rho[71] / rho[21] - 1,
            (rho[71] - rho[52]) / (rho[71] + rho[52]),
            (rho[71] - rho[55]) / (rho[71] + rho[55]),
            rho[61:72].mean() / rho[53:58].mean() - 1,
            (rho[62] - rho[53]) / (rho[53] - rho[47]),
            ((rho[51] - rho[45]) - 0.2 * (rho[51] - rho[21])) * rho[51] / rho[45],
            (rho[17] - rho[25]) / (rho[17] + rho[25]),
            (rho[47] - rho[11]) / rho[61],
            1 / rho[21] - 1 / rho[51],
            1 / rho[13] - 1 / rho[21],
        ],
        dtype=np.float32,
    )

    assert indices.shape == (1, 1, 20)
    assert indices.dtype == np.float32
    assert valid.dtype == np.bool_
    assert valid.all()
    np.testing.assert_allclose(indices[0, 0], expected, rtol=2e-5, atol=2e-6)


def test_only_required_bands_affect_each_index() -> None:
    cube = stored_cube(1, 2)
    cube[0, 0, 0] = np.nan  # no formula consumes 445 nm / channel 0
    cube[0, 1, 5] = np.float32(1.2 * lable.STORAGE_SCALE)

    indices, valid = lable.calculate_indices(cube)

    assert valid[0, 0].all()
    assert np.isfinite(indices[0, 0]).all()
    assert not valid[0, 1, lable.INDEX_NAMES.index("EVI")]
    assert np.isnan(indices[0, 1, lable.INDEX_NAMES.index("EVI")])
    unaffected = [name for name in lable.INDEX_NAMES if name != "EVI"]
    assert valid[0, 1, [lable.INDEX_NAMES.index(name) for name in unaffected]].all()


def test_denominator_and_msavi_invalid_input_become_invalid() -> None:
    cube = stored_cube(1, 2)
    cube[0, 0, 45] = 0.0  # invalid SR and MCARI denominators
    cube[0, 1, 71] = np.float32(0.5 * lable.STORAGE_SCALE)
    cube[0, 1, 45] = np.float32(-0.1 * lable.STORAGE_SCALE)

    indices, valid = lable.calculate_indices(cube)

    for name in ("SR", "MCARI"):
        channel = lable.INDEX_NAMES.index(name)
        assert not valid[0, 0, channel]
        assert np.isnan(indices[0, 0, channel])
    msavi_channel = lable.INDEX_NAMES.index("MSAVI2")
    assert not valid[0, 1, msavi_channel]
    assert np.isnan(indices[0, 1, msavi_channel])


@pytest.mark.parametrize("order", ["C", "F"])
def test_generate_and_verify_tiny_artifact(tmp_path: Path, order: str) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    output_dir = tmp_path / lable.FORMULA_VERSION
    cube = np.array(stored_cube(3, 4), order=order)
    cube[2, 3, 45] = np.float32(1.1 * lable.STORAGE_SCALE)
    np.save(source_path, cube, allow_pickle=False)
    source_before = lable.sha256_file(source_path)

    generated = lable.generate_artifact(source_path, output_dir, tile_rows=2)
    verified = lable.verify_artifact(source_path, output_dir, tile_rows=2)
    indices_path, mask_path, metadata_path = lable.artifact_paths(source_path, output_dir)
    indices = np.load(indices_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert generated["action"] == "generated"
    assert verified["action"] == "verified_existing"
    assert indices.shape == (3, 4, 20)
    assert indices.dtype == np.float32
    assert mask.shape == indices.shape
    assert mask.dtype == np.bool_
    assert np.array_equal(np.isnan(indices), ~mask)
    assert metadata["channel_order"] == list(lable.INDEX_NAMES)
    assert metadata["formula_version"] == lable.FORMULA_VERSION
    assert lable.sha256_file(source_path) == source_before
    assert not list(output_dir.glob("*.tmp*"))


def test_existing_policies_and_tamper_detection(tmp_path: Path) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    output_dir = tmp_path / lable.FORMULA_VERSION
    np.save(source_path, stored_cube(1, 2), allow_pickle=False)
    lable.generate_artifact(source_path, output_dir, tile_rows=1)

    with pytest.raises(FileExistsError):
        lable.generate_artifact(source_path, output_dir, tile_rows=1)
    replaced = lable.generate_artifact(
        source_path,
        output_dir,
        tile_rows=1,
        replace_existing=True,
    )
    assert replaced["action"] == "replaced"

    indices_path, _, _ = lable.artifact_paths(source_path, output_dir)
    indices = np.load(indices_path, allow_pickle=False)
    indices[0, 0, 0] += np.float32(0.25)
    np.save(indices_path, indices, allow_pickle=False)
    with pytest.raises(ValueError, match="indices differ from source"):
        lable.verify_artifact(source_path, output_dir, tile_rows=1)


def test_wrong_source_and_unversioned_output_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lable.calculate_indices(np.zeros((2, 84), dtype=np.float32))
    with pytest.raises(TypeError):
        lable.calculate_indices(np.zeros((2, 2, 84), dtype=np.float64))

    source_path = tmp_path / "hsi_0001.npy"
    np.save(source_path, stored_cube(1, 1), allow_pickle=False)
    with pytest.raises(ValueError, match=lable.FORMULA_VERSION):
        lable.generate_artifact(source_path, tmp_path / "labels", tile_rows=1)
