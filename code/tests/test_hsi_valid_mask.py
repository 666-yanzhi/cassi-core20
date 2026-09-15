from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import hsi_valid_mask


def stored_cube(height: int = 2, width: int = 3) -> np.ndarray:
    return np.full(
        (height, width, hsi_valid_mask.EXPECTED_BANDS),
        np.float32(0.5 * hsi_valid_mask.STORAGE_SCALE),
        dtype=np.float32,
    )


def test_all_bands_must_be_valid() -> None:
    cube = stored_cube()
    cube[0, 1, 0] = np.float32(1.1 * hsi_valid_mask.STORAGE_SCALE)
    cube[1, 2, 83] = np.nan

    mask = hsi_valid_mask.calculate_hsi_valid_mask(cube)

    assert mask.shape == (2, 3)
    assert mask.dtype == np.bool_
    assert mask.tolist() == [[True, False, True], [True, True, False]]


@pytest.mark.parametrize(
    "bad_cube,exception",
    [
        (np.zeros((2, 84), dtype=np.float32), ValueError),
        (np.zeros((2, 2, 83), dtype=np.float32), ValueError),
        (np.zeros((2, 2, 84), dtype=np.float64), TypeError),
    ],
)
def test_wrong_input_contract_is_rejected(
    bad_cube: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        hsi_valid_mask.calculate_hsi_valid_mask(bad_cube)


@pytest.mark.parametrize("order", ["C", "F"])
def test_generate_and_verify_tiny_artifact(tmp_path: Path, order: str) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    output_dir = tmp_path / hsi_valid_mask.VALIDITY_VERSION
    cube = np.array(stored_cube(3, 4), order=order)
    cube[2, 3, 17] = np.float32(-1.0)
    np.save(source_path, cube, allow_pickle=False)
    source_before = hsi_valid_mask.sha256_file(source_path)

    generated = hsi_valid_mask.generate_artifact(source_path, output_dir, tile_rows=2)
    verified = hsi_valid_mask.verify_artifact(source_path, output_dir, tile_rows=2)
    mask_path, metadata_path = hsi_valid_mask.artifact_paths(source_path, output_dir)
    mask = np.load(mask_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert generated["action"] == "generated"
    assert verified["action"] == "verified_existing"
    assert mask.shape == (3, 4)
    assert mask.dtype == np.bool_
    assert mask.sum() == 11
    assert not mask[2, 3]
    assert metadata["statistics"]["invalid_pixel_count"] == 1
    assert hsi_valid_mask.sha256_file(source_path) == source_before
    assert not list(output_dir.glob("*.tmp*"))


def test_existing_policies(tmp_path: Path) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    output_dir = tmp_path / hsi_valid_mask.VALIDITY_VERSION
    np.save(source_path, stored_cube(1, 1), allow_pickle=False)
    hsi_valid_mask.generate_artifact(source_path, output_dir, tile_rows=1)

    with pytest.raises(FileExistsError):
        hsi_valid_mask.generate_artifact(source_path, output_dir, tile_rows=1)
    replaced = hsi_valid_mask.generate_artifact(
        source_path,
        output_dir,
        tile_rows=1,
        replace_existing=True,
    )
    assert replaced["action"] == "replaced"


def test_verify_rejects_tampered_mask(tmp_path: Path) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    output_dir = tmp_path / hsi_valid_mask.VALIDITY_VERSION
    np.save(source_path, stored_cube(1, 2), allow_pickle=False)
    hsi_valid_mask.generate_artifact(source_path, output_dir, tile_rows=1)
    mask_path, _ = hsi_valid_mask.artifact_paths(source_path, output_dir)
    mask = np.load(mask_path, allow_pickle=False)
    mask[0, 0] = False
    np.save(mask_path, mask, allow_pickle=False)

    with pytest.raises(ValueError, match="differs from source"):
        hsi_valid_mask.verify_artifact(source_path, output_dir, tile_rows=1)


def test_output_directory_must_be_versioned(tmp_path: Path) -> None:
    source_path = tmp_path / "hsi_0001.npy"
    np.save(source_path, stored_cube(1, 1), allow_pickle=False)

    with pytest.raises(ValueError, match=hsi_valid_mask.VALIDITY_VERSION):
        hsi_valid_mask.generate_artifact(source_path, tmp_path / "validity", tile_rows=1)
