from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import derive_mask


def legacy_source() -> np.ndarray:
    return np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)


def test_derivation_opens_exactly_highest_half() -> None:
    source = legacy_source()
    binary, details = derive_mask.derive_binary_mask(source)

    assert binary.shape == (256, 256)
    assert binary.dtype == np.float32
    assert np.array_equal(np.unique(binary), np.array([0, 1], dtype=np.float32))
    assert int(binary.sum()) == 32768
    assert np.count_nonzero(binary[source < 0.5]) == 0
    assert np.count_nonzero(binary[source > 0.5]) == 32768
    assert details["method"] == "top_half_by_value"


@pytest.mark.parametrize(
    "source,exception",
    [
        (np.zeros((10, 10), dtype=np.float32), ValueError),
        (np.zeros((256, 256), dtype=np.float64), TypeError),
        (np.full((256, 256), np.nan, dtype=np.float32), ValueError),
        (np.full((256, 256), 1.1, dtype=np.float32), ValueError),
    ],
)
def test_invalid_legacy_source_is_rejected(
    source: np.ndarray,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        derive_mask.derive_binary_mask(source)


def test_generate_verify_and_tamper_detection(tmp_path: Path) -> None:
    source_path = tmp_path / "legacy.npy"
    output_path = tmp_path / "mask.npy"
    metadata_path = tmp_path / "mask.meta.json"
    np.save(source_path, legacy_source(), allow_pickle=False)

    generated = derive_mask.generate_artifact(source_path, output_path, metadata_path)
    verified = derive_mask.verify_artifact(source_path, output_path, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert generated["status"] == "generated"
    assert verified["status"] == "verified_existing"
    assert metadata["role"] == "formal"
    assert metadata["usage"] == "formal_simulation_baseline"
    assert metadata["physical_status"] == "not_hardware_calibrated"
    assert metadata["open_fraction"] == 0.5
    with pytest.raises(FileExistsError):
        derive_mask.generate_artifact(source_path, output_path, metadata_path)

    output = np.load(output_path)
    output[0, 0] = 1 - output[0, 0]
    np.save(output_path, output, allow_pickle=False)
    with pytest.raises(ValueError, match="differs"):
        derive_mask.verify_artifact(source_path, output_path, metadata_path)
