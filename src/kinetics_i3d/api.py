"""Integration-oriented high-level API for modernized I3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .models import I3D, InceptionI3d
from .weights import LoadReport, load_pretrained

FreezeStrategy = Literal["none", "all", "logits", "prefixes"]


@dataclass
class InferenceOutput:
    """Unified inference outputs for both canonical and legacy models."""

    logits_per_frame: torch.Tensor
    clip_logits: torch.Tensor
    clip_probs: torch.Tensor


@dataclass
class FinetuneSetup:
    """Result of preparing model parameters for finetuning."""

    freeze_strategy: FreezeStrategy
    trainable_names: list[str]
    frozen_names: list[str]
    param_groups: list[dict[str, Any]]


def _resolve_modality(modality: str) -> int:
    if modality == "rgb":
        return 3
    if modality == "flow":
        return 2
    raise ValueError(f"Unknown modality={modality}; expected one of ['rgb', 'flow']")


def _resolve_canonical_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        return getattr(model, "backbone")
    return model


def _strip_backbone_prefix(name: str) -> str:
    return name.removeprefix("backbone.")


def build_i3d(
    *,
    num_classes: int = 400,
    modality: str = "rgb",
    legacy: bool = False,
    spatial_squeeze: bool = True,
    final_endpoint: str = "Logits",
    dropout: float = 0.5,
) -> nn.Module:
    """Build an I3D model with a unified constructor.

    Args:
        num_classes: Number of classification classes.
        modality: ``rgb`` or ``flow``.
        legacy: If ``True``, return legacy wrapper ``I3D``.
        spatial_squeeze: Canonical-only setting.
        final_endpoint: Canonical-only endpoint setting.
        dropout: Dropout probability used in the underlying PyTorch module.
    """

    _resolve_modality(modality)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout should be in [0,1], got {dropout}")

    if legacy:
        return I3D(num_classes=num_classes, modality=modality, dropout_prob=dropout)

    in_channels = _resolve_modality(modality)
    return InceptionI3d(
        num_classes=num_classes,
        in_channels=in_channels,
        spatial_squeeze=spatial_squeeze,
        final_endpoint=final_endpoint,
        dropout_keep_prob=dropout,
    )


def load_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    format: str = "auto",
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> LoadReport:
    """Load checkpoint weights into a model with format auto-detection support."""

    return load_pretrained(
        model,
        checkpoint_path=checkpoint_path,
        format=format,
        strict=strict,
        map_location=map_location,
    )


def prepare_finetune(
    model: nn.Module,
    *,
    freeze_strategy: FreezeStrategy = "logits",
    trainable_prefixes: list[str] | tuple[str, ...] | None = None,
) -> FinetuneSetup:
    """Configure trainable parameters for common finetuning strategies.

    Strategies:
        - ``none``: train all parameters.
        - ``all``: freeze all parameters.
        - ``logits``: train only classifier head (``logits.*``).
        - ``prefixes``: train parameters matched by ``trainable_prefixes``.
    """

    if freeze_strategy not in ("none", "all", "logits", "prefixes"):
        raise ValueError(
            f"Unsupported freeze_strategy={freeze_strategy}; expected one of ['none', 'all', 'logits', 'prefixes']"
        )

    normalized_prefixes: tuple[str, ...] = ()
    if freeze_strategy == "prefixes":
        if not trainable_prefixes:
            raise ValueError("trainable_prefixes must be provided when freeze_strategy='prefixes'")
        normalized_prefixes = tuple(_strip_backbone_prefix(p) for p in trainable_prefixes)

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        canonical_name = _strip_backbone_prefix(name)

        if freeze_strategy == "none":
            should_train = True
        elif freeze_strategy == "all":
            should_train = False
        elif freeze_strategy == "logits":
            should_train = canonical_name.startswith("logits.")
        else:
            should_train = canonical_name.startswith(normalized_prefixes)

        param.requires_grad = should_train
        if should_train:
            trainable_names.append(name)
            trainable_params.append(param)
        else:
            frozen_names.append(name)

    param_groups: list[dict[str, Any]] = []
    if trainable_params:
        param_groups.append({"params": trainable_params})

    return FinetuneSetup(
        freeze_strategy=freeze_strategy,
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        param_groups=param_groups,
    )


def forward_infer(model: nn.Module, inputs: torch.Tensor) -> InferenceOutput:
    """Run inference and normalize outputs for downstream pipelines."""

    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), nn.Module):
        logits_per_frame = getattr(model, "backbone")(inputs)
        if logits_per_frame.ndim != 3:
            raise ValueError("Legacy backbone should return logits_per_frame with shape (B, C, T)")
        clip_logits = logits_per_frame.mean(dim=2)
        clip_probs = torch.softmax(clip_logits, dim=1)
        return InferenceOutput(
            logits_per_frame=logits_per_frame,
            clip_logits=clip_logits,
            clip_probs=clip_probs,
        )

    outputs = model(inputs)
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim != 3:
            raise ValueError("Expected canonical model output to have shape (B, C, T)")
        logits_per_frame = outputs
        clip_logits = outputs.mean(dim=2)
        clip_probs = torch.softmax(clip_logits, dim=1)
        return InferenceOutput(
            logits_per_frame=logits_per_frame,
            clip_logits=clip_logits,
            clip_probs=clip_probs,
        )

    if isinstance(outputs, tuple) and len(outputs) == 2:
        probs, logits = outputs
        if logits.ndim != 2:
            raise ValueError("Tuple output logits should have shape (B, C)")
        logits_per_frame = logits.unsqueeze(-1)
        return InferenceOutput(
            logits_per_frame=logits_per_frame,
            clip_logits=logits,
            clip_probs=probs,
        )

    raise TypeError("Unsupported model output type for forward_infer")


def canonical_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Get canonical-format state_dict regardless of wrapper type."""

    canonical_model = _resolve_canonical_model(model)
    return canonical_model.state_dict()


def save_canonical_weights(model: nn.Module, path: str | Path) -> str:
    """Save weights in the default single-file canonical state_dict format."""

    path = str(path)
    torch.save(canonical_state_dict(model), path)
    return path
