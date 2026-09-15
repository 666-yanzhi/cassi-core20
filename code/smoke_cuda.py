#!/usr/bin/env python3
"""Run one real-data CUDA optimization step for the configured training split."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train
import training


def run(config_path: Path) -> dict[str, object]:
    config = training.load_resolved_config(config_path)
    if config["runner"]["device"] != "cuda":
        raise ValueError("the smoke gate requires runner.device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "deterministic CUDA smoke requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )

    seed = config["runner"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    manifest_path = Path(config["data"]["manifest"])
    normalization_path = Path(config["data"]["normalization"])
    dataset = training.Core20Dataset(
        manifest_path,
        normalization_path,
        config["data"]["train_scenes"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    batch = next(iter(loader))
    device = torch.device("cuda")
    measurement, model_mask, supervision = (value.to(device) for value in batch)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = train.build_model(config).to(device)
    optimizer, _ = train.build_optimizer_and_scheduler(model, config)
    loss_fn = training.MaskedBalancedCharbonnierLoss(config["loss"]["epsilon"])
    parameter_name, parameter = next(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    parameter_before = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model(measurement, model_mask)
    loss = loss_fn(prediction, supervision)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["runner"]["gradient_clip_norm"]
    )
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    loss_value = float(loss.item())
    gradient_norm_value = float(gradient_norm.item())
    parameter_changed = not torch.equal(parameter_before, parameter.detach())
    passed = (
        math.isfinite(loss_value)
        and math.isfinite(gradient_norm_value)
        and gradient_norm_value > 0.0
        and parameter_changed
        and tuple(prediction.shape)
        == (len(measurement), 20, config["data"]["patch_size"], config["data"]["patch_size"])
        and bool(torch.isfinite(prediction).all())
    )
    if not passed:
        raise RuntimeError("real-data CUDA smoke acceptance failed")

    return {
        "schema_version": 1,
        "status": "passed",
        "claim": "one real training batch forward, backward, and AdamW step; not convergence",
        "config_sha256": config["source_config_sha256"],
        "manifest_sha256": training.sha256_file(manifest_path),
        "normalization_sha256": training.sha256_file(normalization_path),
        "split_id": config["data"]["split_id"],
        "dataset_role": "train_only",
        "sample_ids": [item["sample_id"] for item in dataset.samples[: len(measurement)]],
        "batch_size": len(measurement),
        "measurement_shape": list(measurement.shape),
        "model_mask_shape": list(model_mask.shape),
        "supervision_shape": list(supervision.shape),
        "prediction_shape": list(prediction.shape),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "loss": loss_value,
        "gradient_norm_before_clip": gradient_norm_value,
        "parameter_checked": parameter_name,
        "parameter_changed": parameter_changed,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "cuda_device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=training.REPO_ROOT / "config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    if args.output is not None:
        output_path = args.output.resolve()
        if output_path.exists():
            raise FileExistsError(output_path)
        training.atomic_write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
