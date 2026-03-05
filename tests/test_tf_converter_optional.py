from __future__ import annotations

import importlib.util

import pytest
import torch

from kinetics_i3d.models import InceptionI3d
from kinetics_i3d.weights.tf_convert import convert_tf_checkpoint_to_state_dict
from tests.conftest import REPO_ROOT, require_paths


def test_tf_conversion_optional_if_tensorflow_available() -> None:
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow not installed; optional converter test skipped")

    tf_ckpt = REPO_ROOT / "reference/kinetics_i3d_pytorch/model/tf_rgb_imagenet/model.ckpt"
    require_paths(tf_ckpt.with_suffix(".index"), tf_ckpt.with_suffix(".meta"), tf_ckpt.with_suffix(".data-00000-of-00001"))

    state_dict = convert_tf_checkpoint_to_state_dict(tf_ckpt, modality="rgb")

    model = InceptionI3d(400, in_channels=3).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []

    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 400, 1)
