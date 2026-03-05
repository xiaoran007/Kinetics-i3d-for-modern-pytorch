"""Canonical Inception I3D model for PyTorch 2.8."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D max-pooling with TensorFlow-like SAME padding."""

    def _compute_pad(self, dim: int, size: int) -> int:
        stride = _to_3tuple(self.stride)[dim]
        kernel = _to_3tuple(self.kernel_size)[dim]
        if size % stride == 0:
            return max(kernel - stride, 0)
        return max(kernel - (size % stride), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        padding = (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back)
        x = F.pad(x, padding)
        return super().forward(x)


class Unit3D(nn.Module):
    """Basic conv3d block with optional batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation_fn: Callable[[torch.Tensor], torch.Tensor] | None = F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_shape = kernel_shape
        self.stride = stride
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,
            bias=use_bias,
        )

        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

    def _compute_pad(self, dim: int, size: int) -> int:
        stride = self.stride[dim]
        kernel = self.kernel_shape[dim]
        if size % stride == 0:
            return max(kernel - stride, 0)
        return max(kernel - (size % stride), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self._compute_pad(0, t)
        pad_h = self._compute_pad(1, h)
        pad_w = self._compute_pad(2, w)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        padding = (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back)
        x = F.pad(x, padding)
        x = self.conv3d(x)

        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: tuple[int, int, int, int, int, int]) -> None:
        super().__init__()

        self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=(1, 1, 1))
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=(1, 1, 1))
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=(3, 3, 3))
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=(1, 1, 1))
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3))
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture."""

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        final_endpoint: str = "Logits",
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ) -> None:
        super().__init__()
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint

        self.end_points: dict[str, nn.Module] = {}
        self.end_points["Conv3d_1a_7x7"] = Unit3D(in_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2))
        self.end_points["MaxPool3d_2a_3x3"] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.end_points["Conv3d_2b_1x1"] = Unit3D(64, 64, kernel_shape=(1, 1, 1))
        self.end_points["Conv3d_2c_3x3"] = Unit3D(64, 192, kernel_shape=(3, 3, 3))
        self.end_points["MaxPool3d_3a_3x3"] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.end_points["Mixed_3b"] = InceptionModule(192, (64, 96, 128, 16, 32, 32))
        self.end_points["Mixed_3c"] = InceptionModule(256, (128, 128, 192, 32, 96, 64))
        self.end_points["MaxPool3d_4a_3x3"] = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        self.end_points["Mixed_4b"] = InceptionModule(480, (192, 96, 208, 16, 48, 64))
        self.end_points["Mixed_4c"] = InceptionModule(512, (160, 112, 224, 24, 64, 64))
        self.end_points["Mixed_4d"] = InceptionModule(512, (128, 128, 256, 24, 64, 64))
        self.end_points["Mixed_4e"] = InceptionModule(512, (112, 144, 288, 32, 64, 64))
        self.end_points["Mixed_4f"] = InceptionModule(528, (256, 160, 320, 32, 128, 128))
        self.end_points["MaxPool3d_5a_2x2"] = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.end_points["Mixed_5b"] = InceptionModule(832, (256, 160, 320, 32, 128, 128))
        self.end_points["Mixed_5c"] = InceptionModule(832, (384, 192, 384, 48, 128, 128))
        for key, module in self.end_points.items():
            self.add_module(key, module)

        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_shape=(1, 1, 1),
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
        )

    def replace_logits(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.logits = Unit3D(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_shape=(1, 1, 1),
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
        )

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        for endpoint in self.VALID_ENDPOINTS:
            if endpoint == "Logits":
                break
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
            if endpoint == self.final_endpoint:
                return x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_backbone(x)

        if self.final_endpoint != "Logits":
            return x

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self.spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_backbone(x)
        return self.avg_pool(x)
