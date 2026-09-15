#!/usr/bin/env python3
"""Train the formal MST-Mamba Core20 comparison model."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import MSTMambaCore20
from models.mst_mamba_core20 import blocks as mst_blocks
from models.mst_mamba_core20 import mamba_core as mst_mamba_core
from models.mst_mamba_core20 import pscan as mst_pscan
import scene_macro_runner
import training


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


MST_MODEL_FIELDS = {
    "name",
    "dim",
    "stage",
    "num_blocks",
    "d_state",
    "expand_factor",
    "d_conv",
    "dt_rank",
    "pscan_parallel",
    "mask_fusion",
    "stage_reductions",
    "in_channels",
    "out_channels",
}


def load_resolved_config(path: Path) -> dict[str, Any]:
    """Reuse the frozen common schema while validating MST-specific fields."""

    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw.get("model")
    if not isinstance(model, dict):
        raise TypeError("config.model must be an object")
    missing = MST_MODEL_FIELDS - set(model)
    unknown = set(model) - MST_MODEL_FIELDS
    if missing or unknown:
        raise ValueError(
            f"config.model fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if model["name"] != "MSTMambaCore20":
        raise ValueError("model.name must be MSTMambaCore20")
    if model["in_channels"] != 84 or model["out_channels"] != 20:
        raise ValueError("MSTMambaCore20 requires 84 input and 20 output channels")
    for field in ("dim", "stage", "d_state", "expand_factor", "d_conv"):
        if not isinstance(model[field], int) or isinstance(model[field], bool) or model[field] <= 0:
            raise ValueError(f"model.{field} must be a positive integer")
    if (
        not isinstance(model["num_blocks"], list)
        or len(model["num_blocks"]) != model["stage"]
        or any(not isinstance(value, int) or value <= 0 for value in model["num_blocks"])
    ):
        raise ValueError("model.num_blocks must contain one positive integer per stage")
    reductions = model["stage_reductions"]
    if (
        not isinstance(reductions, list)
        or len(reductions) != model["stage"] * 2 + 1
        or any(not isinstance(value, int) or value <= 0 for value in reductions)
    ):
        raise ValueError("model.stage_reductions must contain 2*stage+1 positive integers")
    if model["dt_rank"] != "auto" and (
        not isinstance(model["dt_rank"], int) or model["dt_rank"] <= 0
    ):
        raise ValueError("model.dt_rank must be 'auto' or a positive integer")
    if not isinstance(model["pscan_parallel"], bool):
        raise TypeError("model.pscan_parallel must be boolean")
    if model["mask_fusion"] not in MSTMambaCore20.VALID_MASK_FUSIONS:
        raise ValueError("model.mask_fusion is unsupported")

    compatible = json.loads(json.dumps(raw))
    compatible["model"] = {
        "name": "RestormerCore20",
        "dim": 4,
        "num_heads": [1],
        "num_blocks": [1],
        "ffn_expansion_factor": 1.5,
        "in_channels": 84,
        "out_channels": 20,
        "bias": False,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(compatible, handle)
        handle.flush()
        resolved = training.load_resolved_config(Path(handle.name))
    resolved["model"] = json.loads(json.dumps(model))
    resolved["source_config"] = str(path)
    resolved["source_config_sha256"] = training.sha256_file(path)
    return resolved


def build_model(config: dict[str, Any]) -> MSTMambaCore20:
    model = config["model"]
    return MSTMambaCore20(
        dim=model["dim"],
        stage=model["stage"],
        num_blocks=tuple(model["num_blocks"]),
        d_state=model["d_state"],
        expand_factor=model["expand_factor"],
        d_conv=model["d_conv"],
        dt_rank=model["dt_rank"],
        pscan_parallel=model["pscan_parallel"],
        mask_fusion=model["mask_fusion"],
        stage_reductions=tuple(model["stage_reductions"]),
        in_channels=model["in_channels"],
        out_channels=model["out_channels"],
    )


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, Any]:
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_config["t_max"],
        eta_min=scheduler_config["eta_min"],
    )
    return optimizer, scheduler


def run_overfit_probe(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: dict[str, Any],
    *,
    steps: int = 8,
) -> dict[str, Any]:
    """Prove a fresh MST-Mamba can reduce loss on one fixed sample."""

    device = torch.device(config["runner"]["device"])
    measurement, model_mask, supervision = (
        value[:1].to(device) for value in batch
    )
    model = build_model(config).to(device)
    loss_fn = training.MaskedBalancedCharbonnierLoss(config["loss"]["epsilon"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    with torch.no_grad():
        initial_loss = float(loss_fn(model(measurement, model_mask), supervision).item())
    step_losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(measurement, model_mask), supervision)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_losses.append(float(loss.item()))
    with torch.no_grad():
        final_loss = float(loss_fn(model(measurement, model_mask), supervision).item())
    if not final_loss < initial_loss:
        raise RuntimeError(
            f"overfit probe did not reduce loss: initial={initial_loss}, final={final_loss}"
        )
    return {
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_final_loss": final_loss / initial_loss,
        "passed": True,
        "claim": "optimization behavior only; not generalization",
        "step_losses": step_losses,
    }


def run(config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = load_resolved_config(config_path)
    runner_config = config["runner"]
    seed = runner_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if runner_config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but CUDA is unavailable")
    if runner_config["device"] == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "deterministic CUDA training requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 before process start"
            )
        torch.cuda.manual_seed_all(seed)

    experiment_dir = Path(config["output"]["experiment_dir"])
    if experiment_dir.exists() and not resume:
        raise FileExistsError(
            f"experiment directory already exists and will not be overwritten: {experiment_dir}"
        )
    if resume and not experiment_dir.is_dir():
        raise FileNotFoundError(f"resume experiment directory is absent: {experiment_dir}")
    if not resume:
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "predictions").mkdir()
        (experiment_dir / "evaluation").mkdir()
        training.atomic_write_json(experiment_dir / "resolved_config.json", config)
    else:
        recorded_config = json.loads(
            (experiment_dir / "resolved_config.json").read_text(encoding="utf-8")
        )
        if recorded_config != config:
            raise ValueError("resume config differs from the experiment resolved config")

    manifest_path = Path(config["data"]["manifest"])
    normalization_path = Path(config["data"]["normalization"])
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        if normalization.get("train_scenes") != config["data"]["train_scenes"]:
            raise ValueError("existing normalization was not fitted on configured train scenes")
        if normalization.get("source_manifest_sha256") != training.sha256_file(manifest_path):
            raise ValueError("existing normalization manifest identity mismatch")
    else:
        normalization = training.fit_normalization(
            manifest_path,
            config["data"]["train_scenes"],
            normalization_path,
            epsilon=config["normalization"]["epsilon"],
        )
    if not resume:
        training.atomic_write_json(experiment_dir / "normalization.json", normalization)
    elif json.loads(
        (experiment_dir / "normalization.json").read_text(encoding="utf-8")
    ) != normalization:
        raise ValueError("resume normalization differs from the experiment copy")

    split_manifest = {
        "schema_version": 1,
        "scope": (
            "formal_experiment"
            if config["data"]["split_id"] == "formal_252_split_seed42_202_15_35"
            else "local_minimum_experiment"
        ),
        "split_id": config["data"]["split_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": training.sha256_file(manifest_path),
        "train_scenes": config["data"]["train_scenes"],
        "val_scenes": config["data"]["val_scenes"],
        "test_scenes": config["data"]["test_scenes"],
        "warning": (
            "Formal training/validation split; test scenes are final-evaluation only."
            if config["data"]["split_id"] == "formal_252_split_seed42_202_15_35"
            else "Local behavior gate only; no formal test-set result."
        ),
    }
    if not resume:
        training.atomic_write_json(experiment_dir / "split_manifest.json", split_manifest)
    elif json.loads(
        (experiment_dir / "split_manifest.json").read_text(encoding="utf-8")
    ) != split_manifest:
        raise ValueError("resume split manifest differs from the experiment copy")

    train_dataset = training.Core20Dataset(
        manifest_path,
        normalization_path,
        config["data"]["train_scenes"],
    )
    val_dataset = training.Core20Dataset(
        manifest_path,
        normalization_path,
        config["data"]["val_scenes"],
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    if runner_config["run_overfit_probe"]:
        first_batch = next(iter(train_loader))
        overfit_probe = run_overfit_probe(first_batch, config)
    else:
        overfit_probe = {
            "steps": 0,
            "passed": None,
            "claim": "disabled for this run; use a separate smoke gate",
        }

    model = build_model(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    loss_fn = training.MaskedBalancedCharbonnierLoss(config["loss"]["epsilon"])
    RunnerV3, runner_sha256, runner_file = training.load_runner_v3(
        Path(runner_config["runner_root"])
    )
    SceneMacroRunnerV3 = scene_macro_runner.create_scene_macro_runner_class(RunnerV3)
    model_file = Path(sys.modules[MSTMambaCore20.__module__].__file__).resolve()
    training_file = Path(training.__file__).resolve()
    train_entry_file = Path(__file__).resolve()
    code_identity = {
        "model": {"path": str(model_file), "sha256": training.sha256_file(model_file)},
        "model_dependencies": {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": training.sha256_file(Path(module.__file__).resolve()),
            }
            for name, module in (
                ("blocks", mst_blocks),
                ("mamba_core", mst_mamba_core),
                ("pscan", mst_pscan),
            )
        },
        "training": {
            "path": str(training_file),
            "sha256": training.sha256_file(training_file),
        },
        "train_entry": {
            "path": str(train_entry_file),
            "sha256": training.sha256_file(train_entry_file),
        },
        "runner": {"path": str(runner_file), "sha256": runner_sha256},
        "scene_macro_runner": {
            "path": str(Path(scene_macro_runner.__file__).resolve()),
            "sha256": training.sha256_file(Path(scene_macro_runner.__file__).resolve()),
        },
    }
    signature_payload = {
        "config": config,
        "manifest_sha256": training.sha256_file(manifest_path),
        "normalization_sha256": training.sha256_file(normalization_path),
        "code_identity": code_identity,
    }
    experiment_signature = training.canonical_sha256(signature_payload)
    runner = SceneMacroRunnerV3(
        model,
        optimizer,
        loss_fn,
        metric_fn=training.balanced_zmae,
        higher_is_better=False,
        device=runner_config["device"],
    )
    state_path = experiment_dir / "last_training_state.pt"
    start_epoch = 0
    initial_best: float | None = None
    initial_no_improve = 0
    if resume:
        restored = training.restore_training_state(
            state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_signature=experiment_signature,
            dataloader_generator=generator,
        )
        start_epoch = int(restored["completed_epochs"])
        runner.history = restored["history"]
        initial_best = restored.get("best_metric")
        initial_no_improve = int(restored.get("no_improve_epochs", 0))
        if start_epoch >= runner_config["epochs"]:
            raise ValueError("checkpoint already reached the configured epoch limit")

    training_control = {
        "best_metric": initial_best,
        "no_improve_epochs": initial_no_improve,
    }

    def save_epoch_boundary(
        completed_epochs: int,
        best_metric: float,
        no_improve_epochs: int,
    ) -> None:
        training_control["best_metric"] = best_metric
        training_control["no_improve_epochs"] = no_improve_epochs
        training.atomic_write_json(experiment_dir / "history.json", runner.history)
        training.save_training_state(
            state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epochs=completed_epochs,
            history=runner.history,
            experiment_signature=experiment_signature,
            dataloader_generator=generator,
            best_metric=best_metric,
            no_improve_epochs=no_improve_epochs,
        )

    first_parameter_before = next(model.parameters()).detach().clone()
    log_path = experiment_dir / "train.log"
    with log_path.open("a" if resume else "x", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            if resume:
                print(f"resume from completed epoch {start_epoch}", flush=True)
            runner.fit(
                train_loader,
                val_loader,
                num_epochs=runner_config["epochs"],
                log_every=runner_config["log_every"],
                best_path=experiment_dir / "best_model.pt",
                grad_clip_norm=runner_config["gradient_clip_norm"],
                lr_scheduler=scheduler,
                patience=runner_config["early_stopping_patience"],
                seed=None if resume else seed,
                start_epoch=start_epoch,
                initial_best=initial_best,
                initial_no_improve=initial_no_improve,
                epoch_end_callback=save_epoch_boundary,
            )
    parameter_updated = not torch.equal(first_parameter_before, next(model.parameters()).detach())
    if not parameter_updated:
        raise RuntimeError("RunnerV3 completed without updating model parameters")
    training.atomic_write_json(experiment_dir / "history.json", runner.history)
    completed_epochs = len(runner.history["train_loss"])
    best_metric = training_control["best_metric"]
    training.save_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epochs=completed_epochs,
        history=runner.history,
        experiment_signature=experiment_signature,
        dataloader_generator=generator,
        best_metric=best_metric,
        no_improve_epochs=int(training_control["no_improve_epochs"]),
    )

    restore_model = build_model(config).to(runner_config["device"])
    restore_optimizer, restore_scheduler = build_optimizer_and_scheduler(restore_model, config)
    restore_generator = torch.Generator()
    restored = training.restore_training_state(
        state_path,
        model=restore_model,
        optimizer=restore_optimizer,
        scheduler=restore_scheduler,
        expected_signature=experiment_signature,
        dataloader_generator=restore_generator,
    )
    restore_matches = all(
        torch.equal(first, second)
        for first, second in zip(model.state_dict().values(), restore_model.state_dict().values())
    )
    if not restore_matches:
        raise RuntimeError("restored model parameters differ from saved training state")

    metadata = {
        "schema_version": 1,
        "status": (
            "complete_formal_training"
            if config["data"]["split_id"] == "formal_252_split_seed42_202_15_35"
            else "complete_local_smoke"
        ),
        "experiment_signature": experiment_signature,
        "runner": {
            "class": "scene_macro_runner.SceneMacroRunnerV3(nndl.runner.RunnerV3)",
            "path": str(runner_file),
            "sha256": runner_sha256,
        },
        "code_identity": code_identity,
        "completed_epochs": completed_epochs,
        "device": runner_config["device"],
        "resumed": resume,
        "parameter_updated": parameter_updated,
        "overfit_probe": overfit_probe,
        "checkpoint_restore": {
            "passed": restore_matches,
            "restored_completed_epochs": restored["completed_epochs"],
        },
        "best_model_sha256": training.sha256_file(experiment_dir / "best_model.pt"),
        "last_training_state_sha256": training.sha256_file(state_path),
        "limitations": (
            [
                "Formal test-set evaluation has not run yet.",
                "The fixed simulation aperture is not hardware calibrated.",
                "Scene-macro validation is supplied by a project-local RunnerV3 subclass.",
            ]
            if config["data"]["split_id"] == "formal_252_split_seed42_202_15_35"
            else [
                "CPU and six local patches only.",
                "No CUDA evidence.",
                "No formal 202/15/35 split or test-set evaluation.",
                "Scene-macro validation is supplied by a project-local RunnerV3 subclass.",
            ]
        ),
    }
    training.atomic_write_json(experiment_dir / "experiment.meta.json", metadata)
    if config["data"]["split_id"] == "formal_252_split_seed42_202_15_35":
        dev_metrics = runner.history["dev_metric"]
        if not dev_metrics:
            raise RuntimeError("formal training completed without validation metrics")
        best_epoch = min(range(len(dev_metrics)), key=dev_metrics.__getitem__) + 1
        formal_report = {
            "schema_version": 1,
            "status": "formal_training_complete_test_not_run",
            "completion_predicate": {
                "training_process_exited_successfully": True,
                "completed_epochs": completed_epochs,
                "best_checkpoint_exists": (experiment_dir / "best_model.pt").is_file(),
                "last_full_state_exists": state_path.is_file(),
                "checkpoint_restore_passed": restore_matches,
                "history_epoch_count_matches": completed_epochs
                == len(runner.history["train_loss"]),
            },
            "experiment_id": config["experiment_id"],
            "experiment_signature": experiment_signature,
            "split_id": config["data"]["split_id"],
            "completed_epochs": completed_epochs,
            "best_validation": {
                "metric": "macro_zMAE",
                "epoch": best_epoch,
                "value": dev_metrics[best_epoch - 1],
            },
            "artifacts": {
                "log": str(log_path),
                "best_model": str(experiment_dir / "best_model.pt"),
                "best_model_sha256": metadata["best_model_sha256"],
                "last_training_state": str(state_path),
                "last_training_state_sha256": metadata[
                    "last_training_state_sha256"
                ],
                "history": str(experiment_dir / "history.json"),
                "resolved_config": str(experiment_dir / "resolved_config.json"),
                "normalization": str(experiment_dir / "normalization.json"),
            },
            "resume_command": (
                "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "/home/user/anaconda3/envs/cassi/bin/python code/train_mst_mamba.py "
                "--config configs/formal_mst_mamba_seed42.json --resume"
            ),
            "formal_test": {
                "completed": False,
                "policy": "Run exactly once after model, config, normalization, and protocol freeze.",
            },
        }
        if not all(formal_report["completion_predicate"].values()):
            raise RuntimeError("formal training completion predicate failed")
        training.atomic_write_json(
            experiment_dir / "formal_training_report.json", formal_report
        )
    return {
        "status": metadata["status"],
        "experiment_dir": str(experiment_dir),
        "completed_epochs": completed_epochs,
        "overfit_relative_final_loss": overfit_probe.get("relative_final_loss"),
        "checkpoint_restore_passed": restore_matches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=training.REPO_ROOT / "configs/formal_mst_mamba_seed42.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the matching experiment from its epoch-boundary full state",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, resume=args.resume), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
