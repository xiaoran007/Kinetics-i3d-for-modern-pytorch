from __future__ import annotations

from pathlib import Path

import torch

from kinetics_i3d.models import InceptionI3d
from kinetics_i3d.weights import convert_checkpoint, detect_checkpoint_format, load_pretrained
from tests.conftest import env_or_default, require_paths


def test_detect_and_load_two_reference_formats(tmp_path: Path) -> None:
    rgb_pt_ckpt = env_or_default("RGB_PYTORCH_CKPT")
    rgb_kinetics_ckpt = env_or_default("RGB_KINETICS_CKPT")
    require_paths(rgb_pt_ckpt, rgb_kinetics_ckpt)

    assert detect_checkpoint_format(torch.load(rgb_pt_ckpt, map_location="cpu")) == "pytorch_i3d"
    assert detect_checkpoint_format(torch.load(rgb_kinetics_ckpt, map_location="cpu")) == "kinetics_i3d"

    model_pt = InceptionI3d(num_classes=400, in_channels=3).eval()
    report_pt = load_pretrained(model_pt, rgb_pt_ckpt, format="auto", strict=True)
    assert report_pt.missing_keys == []
    assert report_pt.unexpected_keys == []

    model_kinetics = InceptionI3d(num_classes=400, in_channels=3).eval()
    report_kinetics = load_pretrained(model_kinetics, rgb_kinetics_ckpt, format="auto", strict=True)
    assert report_kinetics.missing_keys == []
    assert report_kinetics.unexpected_keys == []

    converted_ckpt = tmp_path / "rgb_kinetics_to_canonical.pt"
    convert_checkpoint(rgb_kinetics_ckpt, converted_ckpt, src_format="kinetics_i3d", dst_format="canonical")

    model_converted = InceptionI3d(num_classes=400, in_channels=3).eval()
    report_converted = load_pretrained(model_converted, converted_ckpt, format="canonical", strict=True)
    assert report_converted.missing_keys == []
    assert report_converted.unexpected_keys == []

    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        y_direct = model_kinetics(x)
        y_converted = model_converted(x)

    assert torch.allclose(y_direct, y_converted)
