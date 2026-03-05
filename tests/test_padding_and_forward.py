from __future__ import annotations

import math

import torch

from kinetics_i3d.models import InceptionI3d
from kinetics_i3d.models.inception_i3d import MaxPool3dSamePadding


def test_same_padding_pool_output_is_ceil_division() -> None:
    pool = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)

    for t in (5, 6, 7):
        for h in (31, 32):
            for w in (31, 32):
                x = torch.randn(1, 3, t, h, w)
                y = pool(x)
                assert y.shape[2] == math.ceil(t / 2)
                assert y.shape[3] == math.ceil(h / 2)
                assert y.shape[4] == math.ceil(w / 2)


def test_inception_i3d_forward_shape_and_dtype() -> None:
    model = InceptionI3d(num_classes=400, in_channels=3).eval()
    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        logits = model(x)
        features = model.extract_features(x)

    assert logits.shape == (1, 400, 1)
    assert features.shape == (1, 1024, 1, 1, 1)
    assert logits.dtype == torch.float32
