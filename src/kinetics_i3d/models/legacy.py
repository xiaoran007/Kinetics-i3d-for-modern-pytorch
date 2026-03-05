"""Legacy-compatible I3D wrapper.

This module keeps the high-level API used in older community implementations,
while delegating model internals to the canonical InceptionI3d.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .inception_i3d import InceptionI3d


class I3D(nn.Module):
    """Legacy API wrapper that returns ``(softmax, logits)``."""

    def __init__(
        self,
        num_classes: int,
        modality: str = "rgb",
        dropout_prob: float = 0.0,
        name: str = "inception",
    ) -> None:
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.modality = modality

        if modality == "rgb":
            in_channels = 3
        elif modality == "flow":
            in_channels = 2
        else:
            raise ValueError(f"{modality} not among known modalities [rgb|flow]")

        self.backbone = InceptionI3d(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_keep_prob=dropout_prob,
        )

    def replace_logits(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.backbone.replace_logits(num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.extract_features(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_per_frame = self.backbone(x)
        logits = logits_per_frame.mean(dim=2)
        probs = torch.softmax(logits, dim=1)
        return probs, logits
