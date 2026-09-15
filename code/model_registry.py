"""Config-driven registry for all Core20 prediction models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torch import nn

from models.mst_mamba_core20 import MSTMambaCore20
from models.mst_mamba_core20 import blocks as mst_blocks
from models.mst_mamba_core20 import mamba_core as mst_mamba_core
from models.mst_mamba_core20 import model as mst_model
from models.mst_mamba_core20 import pscan as mst_pscan
from models import restormer_core20
from models.restormer_core20 import RestormerCore20


RESTORMER_FIELDS = {
    "name",
    "dim",
    "num_heads",
    "num_blocks",
    "ffn_expansion_factor",
    "in_channels",
    "out_channels",
    "bias",
}

MST_MAMBA_FIELDS = {
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


def _require_exact_fields(model: dict[str, Any], expected: set[str]) -> None:
    actual = set(model)
    if actual != expected:
        raise ValueError(
            "config.model fields mismatch; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _positive_integer(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validate_protocol_channels(model: dict[str, Any]) -> None:
    if model["in_channels"] != 84 or model["out_channels"] != 20:
        raise ValueError(
            f"{model['name']} requires 84 input and 20 output channels"
        )


def _validate_restormer(model: dict[str, Any]) -> None:
    _require_exact_fields(model, RESTORMER_FIELDS)
    _validate_protocol_channels(model)
    _positive_integer(model["dim"], "model.dim")
    heads = model["num_heads"]
    blocks = model["num_blocks"]
    if (
        not isinstance(heads, list)
        or not isinstance(blocks, list)
        or not heads
        or len(heads) != len(blocks)
    ):
        raise ValueError(
            "model.num_heads and model.num_blocks must be equal nonempty lists"
        )
    if any(not isinstance(value, int) or value <= 0 for value in heads):
        raise ValueError("model.num_heads must contain positive integers")
    if any(not isinstance(value, int) or value <= 0 for value in blocks):
        raise ValueError("model.num_blocks must contain positive integers")
    expansion = model["ffn_expansion_factor"]
    if (
        not isinstance(expansion, (int, float))
        or isinstance(expansion, bool)
        or expansion <= 0
    ):
        raise ValueError("model.ffn_expansion_factor must be positive")
    if not isinstance(model["bias"], bool):
        raise TypeError("model.bias must be boolean")


def _validate_mst_mamba(model: dict[str, Any]) -> None:
    _require_exact_fields(model, MST_MAMBA_FIELDS)
    _validate_protocol_channels(model)
    for field in ("dim", "stage", "d_state", "expand_factor", "d_conv"):
        _positive_integer(model[field], f"model.{field}")
    if (
        not isinstance(model["num_blocks"], list)
        or len(model["num_blocks"]) != model["stage"]
        or any(
            not isinstance(value, int) or value <= 0
            for value in model["num_blocks"]
        )
    ):
        raise ValueError(
            "model.num_blocks must contain one positive integer per stage"
        )
    reductions = model["stage_reductions"]
    if (
        not isinstance(reductions, list)
        or len(reductions) != model["stage"] * 2 + 1
        or any(not isinstance(value, int) or value <= 0 for value in reductions)
    ):
        raise ValueError(
            "model.stage_reductions must contain 2*stage+1 positive integers"
        )
    if model["dt_rank"] != "auto" and (
        not isinstance(model["dt_rank"], int) or model["dt_rank"] <= 0
    ):
        raise ValueError("model.dt_rank must be 'auto' or a positive integer")
    if not isinstance(model["pscan_parallel"], bool):
        raise TypeError("model.pscan_parallel must be boolean")
    if model["mask_fusion"] not in MSTMambaCore20.VALID_MASK_FUSIONS:
        raise ValueError("model.mask_fusion is unsupported")


def _build_restormer(model: dict[str, Any]) -> RestormerCore20:
    return RestormerCore20(
        dim=model["dim"],
        num_heads=tuple(model["num_heads"]),
        num_blocks=tuple(model["num_blocks"]),
        ffn_expansion_factor=model["ffn_expansion_factor"],
        in_channels=model["in_channels"],
        out_channels=model["out_channels"],
        bias=model["bias"],
    )


def _build_mst_mamba(model: dict[str, Any]) -> MSTMambaCore20:
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


_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "RestormerCore20": _validate_restormer,
    "MSTMambaCore20": _validate_mst_mamba,
}

_BUILDERS: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "RestormerCore20": _build_restormer,
    "MSTMambaCore20": _build_mst_mamba,
}

_SOURCE_MODULES = {
    "RestormerCore20": {
        "model": restormer_core20,
    },
    "MSTMambaCore20": {
        "model": mst_model,
        "blocks": mst_blocks,
        "mamba_core": mst_mamba_core,
        "pscan": mst_pscan,
    },
}


def supported_model_names() -> tuple[str, ...]:
    return tuple(_BUILDERS)


def validate_model_config(model: dict[str, Any]) -> None:
    if not isinstance(model, dict):
        raise TypeError("config.model must be an object")
    name = model.get("name")
    if name not in _VALIDATORS:
        raise ValueError(
            f"unsupported model.name={name!r}; supported={list(supported_model_names())}"
        )
    _VALIDATORS[name](model)


def build_model(config: dict[str, Any]) -> nn.Module:
    model = config["model"]
    validate_model_config(model)
    return _BUILDERS[model["name"]](model)


def source_files(model_name: str) -> dict[str, Path]:
    if model_name not in _SOURCE_MODULES:
        raise ValueError(
            f"unsupported model.name={model_name!r}; "
            f"supported={list(supported_model_names())}"
        )
    return {
        label: Path(module.__file__).resolve()
        for label, module in _SOURCE_MODULES[model_name].items()
    }
