from __future__ import annotations

import sys

import torch

from kinetics_i3d.models import I3D, InceptionI3d
from kinetics_i3d.weights import load_pretrained
from tests.conftest import REPO_ROOT, env_or_default, require_paths


def test_canonical_logits_match_reference_pytorch_i3d() -> None:
    rgb_pt_ckpt = env_or_default("RGB_PYTORCH_CKPT")
    require_paths(rgb_pt_ckpt)

    ref_path = REPO_ROOT / "reference/pytorch-i3d"
    sys.path.insert(0, str(ref_path))
    from pytorch_i3d import InceptionI3d as RefInceptionI3d  # type: ignore

    ref_model = RefInceptionI3d(400, in_channels=3).eval()
    ref_model.load_state_dict(torch.load(rgb_pt_ckpt, map_location="cpu"))

    model = InceptionI3d(400, in_channels=3).eval()
    load_pretrained(model, rgb_pt_ckpt, format="auto", strict=True)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        y_ref = ref_model(x)
        y_modern = model(x)

    assert torch.allclose(y_ref, y_modern)


def test_legacy_logits_match_reference_kinetics_i3d() -> None:
    rgb_kinetics_ckpt = env_or_default("RGB_KINETICS_CKPT")
    require_paths(rgb_kinetics_ckpt)

    ref_path = REPO_ROOT / "reference/kinetics_i3d_pytorch/src"
    sys.path.insert(0, str(ref_path))
    from i3dpt import I3D as RefLegacyI3D  # type: ignore

    ref_model = RefLegacyI3D(400, modality="rgb").eval()
    ref_model.load_state_dict(torch.load(rgb_kinetics_ckpt, map_location="cpu"))

    model = I3D(400, modality="rgb").eval()
    load_pretrained(model, rgb_kinetics_ckpt, format="auto", strict=True)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        _, logits_ref = ref_model(x)
        _, logits_modern = model(x)

    assert torch.allclose(logits_ref, logits_modern)
