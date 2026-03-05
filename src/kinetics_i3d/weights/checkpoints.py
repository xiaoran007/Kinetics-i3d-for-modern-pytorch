"""Checkpoint format detection, conversion, and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

CHECKPOINT_FORMATS = ("auto", "canonical", "pytorch_i3d", "kinetics_i3d")


@dataclass
class LoadReport:
    checkpoint_path: str
    source_format: str
    target_format: str
    missing_keys: list[str]
    unexpected_keys: list[str]


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, Mapping):
        state_dict = obj
    else:
        raise TypeError("Unsupported checkpoint object; expected mapping or mapping with 'state_dict'.")

    if not state_dict:
        raise ValueError("Empty state_dict")
    return dict(state_dict)


def detect_checkpoint_format(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(k.startswith("mixed_") or k.startswith("conv3d_0c_1x1") for k in keys):
        return "kinetics_i3d"
    if any(k.startswith("Mixed_") or k.startswith("Conv3d_") or k.startswith("logits.") for k in keys):
        return "pytorch_i3d"
    raise ValueError("Unable to infer checkpoint format from keys")


def _kinetics_to_canonical_key(key: str) -> str:
    mapped = key

    if mapped.startswith("conv3d_0c_1x1."):
        mapped = mapped.replace("conv3d_0c_1x1.", "logits.", 1)
    elif mapped.startswith("conv3d_"):
        head, rest = mapped.split(".", 1)
        mapped = f"{head[0].upper()}{head[1:]}.{rest}"
    elif mapped.startswith("mixed_"):
        head, rest = mapped.split(".", 1)
        mapped = f"{head[0].upper()}{head[1:]}.{rest}"

    mapped = mapped.replace(".branch_0.", ".b0.")
    mapped = mapped.replace(".branch_1.0.", ".b1a.")
    mapped = mapped.replace(".branch_1.1.", ".b1b.")
    mapped = mapped.replace(".branch_2.0.", ".b2a.")
    mapped = mapped.replace(".branch_2.1.", ".b2b.")
    mapped = mapped.replace(".branch_3.1.", ".b3b.")
    mapped = mapped.replace(".batch3d.", ".bn.")
    return mapped


def _canonical_to_kinetics_key(key: str) -> str:
    mapped = key

    if mapped.startswith("logits."):
        mapped = mapped.replace("logits.", "conv3d_0c_1x1.", 1)
    elif mapped.startswith("Conv3d_"):
        head, rest = mapped.split(".", 1)
        mapped = f"{head[0].lower()}{head[1:]}.{rest}"
    elif mapped.startswith("Mixed_"):
        head, rest = mapped.split(".", 1)
        mapped = f"{head[0].lower()}{head[1:]}.{rest}"

    mapped = mapped.replace(".b0.", ".branch_0.")
    mapped = mapped.replace(".b1a.", ".branch_1.0.")
    mapped = mapped.replace(".b1b.", ".branch_1.1.")
    mapped = mapped.replace(".b2a.", ".branch_2.0.")
    mapped = mapped.replace(".b2b.", ".branch_2.1.")
    mapped = mapped.replace(".b3b.", ".branch_3.1.")
    mapped = mapped.replace(".bn.", ".batch3d.")
    return mapped


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    src_format: str,
    dst_format: str = "canonical",
) -> dict[str, torch.Tensor]:
    if src_format not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported src_format={src_format}")
    if dst_format not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported dst_format={dst_format}")

    if src_format == "auto":
        src_format = detect_checkpoint_format(state_dict)
    if dst_format == "auto":
        dst_format = "canonical"

    if src_format == "pytorch_i3d":
        src_format = "canonical"
    if dst_format == "pytorch_i3d":
        dst_format = "canonical"

    if src_format == dst_format:
        return dict(state_dict)

    if src_format == "kinetics_i3d" and dst_format == "canonical":
        return {_kinetics_to_canonical_key(k): v for k, v in state_dict.items()}

    if src_format == "canonical" and dst_format == "kinetics_i3d":
        return {_canonical_to_kinetics_key(k): v for k, v in state_dict.items()}

    raise ValueError(f"Unsupported conversion path: {src_format} -> {dst_format}")


def _resolve_model_for_loading(model: nn.Module) -> tuple[nn.Module, str]:
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        return getattr(model, "backbone"), "canonical"
    return model, "canonical"


def load_pretrained(
    model: nn.Module,
    checkpoint_path: str | Path,
    format: str = "auto",
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> LoadReport:
    if format not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported format={format}; expected one of {CHECKPOINT_FORMATS}")

    checkpoint_path = str(checkpoint_path)
    raw_obj = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _unwrap_state_dict(raw_obj)

    src_format = format if format != "auto" else detect_checkpoint_format(state_dict)
    canonical_state_dict = convert_state_dict(state_dict, src_format=src_format, dst_format="canonical")

    target_model, target_format = _resolve_model_for_loading(model)
    incompatible = target_model.load_state_dict(canonical_state_dict, strict=strict)

    return LoadReport(
        checkpoint_path=checkpoint_path,
        source_format=src_format,
        target_format=target_format,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def convert_checkpoint(
    src_checkpoint: str | Path,
    dst_checkpoint: str | Path,
    src_format: str = "auto",
    dst_format: str = "canonical",
    map_location: str | torch.device = "cpu",
) -> str:
    if src_format not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported src_format={src_format}; expected one of {CHECKPOINT_FORMATS}")
    if dst_format not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported dst_format={dst_format}; expected one of {CHECKPOINT_FORMATS}")

    src_checkpoint = str(src_checkpoint)
    dst_checkpoint = str(dst_checkpoint)

    raw_obj = torch.load(src_checkpoint, map_location=map_location)
    state_dict = _unwrap_state_dict(raw_obj)

    converted = convert_state_dict(state_dict, src_format=src_format, dst_format=dst_format)
    torch.save(converted, dst_checkpoint)
    return dst_checkpoint
