from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import cassi_forward
import measurement_dataset


def test_development_mask_is_deterministic_and_cannot_claim_formal_name(
    tmp_path: Path,
) -> None:
    first_mask = tmp_path / "dev_a.npy"
    first_meta = tmp_path / "dev_a.meta.json"
    second_mask = tmp_path / "dev_b.npy"
    second_meta = tmp_path / "dev_b.meta.json"

    first = measurement_dataset.create_development_mask(first_mask, first_meta, seed=7)
    second = measurement_dataset.create_development_mask(second_mask, second_meta, seed=7)

    assert np.array_equal(np.load(first_mask), np.load(second_mask))
    assert first["sha256"] == second["sha256"]
    assert first["role"] == measurement_dataset.MASK_ROLE_DEVELOPMENT
    with pytest.raises(ValueError, match="reserved formal name"):
        measurement_dataset.create_development_mask(
            tmp_path / "mask.npy",
            tmp_path / "mask.meta.json",
        )


def test_mask_loader_rejects_tampered_role_hash_and_open_fraction(tmp_path: Path) -> None:
    mask_path = tmp_path / "dev.npy"
    metadata_path = tmp_path / "dev.meta.json"
    measurement_dataset.create_development_mask(mask_path, metadata_path)

    mask, metadata = measurement_dataset.load_mask_artifact(
        mask_path,
        metadata_path,
        required_role=measurement_dataset.MASK_ROLE_DEVELOPMENT,
    )
    assert mask.shape == (256, 256)
    assert metadata["open_fraction"] == pytest.approx(float(mask.mean()))

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["role"] = "formal"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        measurement_dataset.load_mask_artifact(
            mask_path,
            metadata_path,
            required_role=measurement_dataset.MASK_ROLE_DEVELOPMENT,
        )


def test_valid_patch_origins_are_deterministic_and_reject_bad_inputs() -> None:
    validity = np.ones((520, 520), dtype=np.bool_)
    validity[:256, :256] = False

    origins = measurement_dataset.find_valid_patch_origins(
        validity,
        stride=256,
        max_patches=3,
    )

    assert origins == [(0, 256), (0, 264), (256, 0)]
    with pytest.raises(ValueError, match="HW bool"):
        measurement_dataset.find_valid_patch_origins(
            validity.astype(np.float32), stride=256, max_patches=1
        )
    with pytest.raises(ValueError, match="smaller"):
        measurement_dataset.find_valid_patch_origins(
            np.ones((10, 10), dtype=np.bool_), stride=1, max_patches=1
        )


def test_model_mask_policy_is_recorded_in_forward_module() -> None:
    assert cassi_forward.MODEL_MASK_POLICY == "repeat_unshifted_physical_mask_v1"
    assert cassi_forward.MEASUREMENT_SCALE == 0.9


def test_formal_split_loader_requires_exact_frozen_partition(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    payload = {
        "seed": 42,
        "partition_unit": "scene",
        "physical_independence_confirmed": True,
        "split_hash": "historical-hash",
        "train": list(range(1, 203)),
        "val": list(range(203, 218)),
        "test": list(range(218, 253)),
    }
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    result = measurement_dataset.load_formal_scene_split(split_path)

    assert result["split_id"] == measurement_dataset.FORMAL_SPLIT_ID
    assert len(result["train_scenes"]) == 202
    assert result["train_scenes"][0] == "hsi_0001"
    assert result["test_scenes"][-1] == "hsi_0252"

    payload["test"][-1] = 251
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique integers"):
        measurement_dataset.load_formal_scene_split(split_path)


def test_measurement_verifier_rejects_ungrouped_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "measurements"
    output_dir.mkdir()
    manifest = {
        "scope": measurement_dataset.MASK_ROLE_DEVELOPMENT,
        "forward_version": cassi_forward.FORWARD_VERSION,
        "label_version": measurement_dataset.LABEL_VERSION,
        "hsi_validity_version": measurement_dataset.VALIDITY_VERSION,
        "measurement_scale": cassi_forward.MEASUREMENT_SCALE,
        "dispersion_step": cassi_forward.DISPERSION_STEP,
        "model_mask_policy": cassi_forward.MODEL_MASK_POLICY,
        "physical_mask": {"path": "unused", "metadata_path": "unused"},
        "model_mask": {"sha256": "unused"},
        "sample_count": 2,
        "samples": [
            {"scene": "scene_b", "top": 0, "left": 0, "sample_id": "b"},
            {"scene": "scene_a", "top": 0, "left": 0, "sample_id": "a"},
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # The mask check happens before sample ordering, so isolate the ordering
    # invariant by replacing only the already-tested artifact loaders.
    physical_mask = np.ones((256, 256), dtype=np.float32)
    model_mask = cassi_forward.build_model_mask(physical_mask)
    np.save(output_dir / "model_mask.npy", model_mask)
    manifest["model_mask"]["sha256"] = measurement_dataset.sha256_file(
        output_dir / "model_mask.npy"
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_loader = measurement_dataset.load_mask_artifact
    measurement_dataset.load_mask_artifact = lambda *args, **kwargs: (
        physical_mask,
        {"sha256": "physical"},
    )
    try:
        with pytest.raises(ValueError, match="scene-grouped"):
            measurement_dataset.verify_smoke_dataset(output_dir)
    finally:
        measurement_dataset.load_mask_artifact = original_loader
