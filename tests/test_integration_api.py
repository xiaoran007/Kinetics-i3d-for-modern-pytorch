from __future__ import annotations

from pathlib import Path

import torch

from kinetics_i3d import (
    InceptionI3d,
    build_i3d,
    canonical_state_dict,
    convert_checkpoint,
    forward_infer,
    load_weights,
    prepare_finetune,
    save_canonical_weights,
)
from tests.conftest import env_or_default, require_paths


def test_build_i3d_supports_canonical_and_legacy() -> None:
    canonical = build_i3d(num_classes=400, modality="rgb", legacy=False)
    assert isinstance(canonical, InceptionI3d)

    flow = build_i3d(num_classes=400, modality="flow", legacy=False)
    assert flow.Conv3d_1a_7x7.conv3d.weight.shape[1] == 2

    legacy = build_i3d(num_classes=400, modality="rgb", legacy=True)
    assert legacy.__class__.__name__ == "I3D"


def test_prepare_finetune_logits_and_prefixes() -> None:
    model = build_i3d(num_classes=10, modality="rgb", legacy=False)

    setup_logits = prepare_finetune(model, freeze_strategy="logits")
    assert setup_logits.trainable_names == ["logits.conv3d.weight", "logits.conv3d.bias"]
    assert len(setup_logits.param_groups) == 1

    setup_prefix = prepare_finetune(
        model,
        freeze_strategy="prefixes",
        trainable_prefixes=["Mixed_5c.", "logits."],
    )
    assert any(name.startswith("Mixed_5c.") for name in setup_prefix.trainable_names)
    assert any(name.startswith("logits.") for name in setup_prefix.trainable_names)


def test_prepare_finetune_backward_smoke_for_logits_only() -> None:
    model = build_i3d(num_classes=8, modality="rgb", legacy=False)
    prepare_finetune(model, freeze_strategy="logits")

    x = torch.randn(1, 3, 16, 224, 224)
    out = forward_infer(model, x)
    loss = out.clip_logits.sum()
    loss.backward()

    assert model.logits.conv3d.weight.grad is not None
    assert model.logits.conv3d.bias.grad is not None
    assert model.Conv3d_1a_7x7.conv3d.weight.grad is None


def test_forward_infer_protocol_for_canonical_and_legacy() -> None:
    x = torch.randn(1, 3, 16, 224, 224)

    canonical = build_i3d(num_classes=400, modality="rgb", legacy=False).eval()
    with torch.no_grad():
        y = forward_infer(canonical, x)
    assert y.logits_per_frame.shape == (1, 400, 1)
    assert y.clip_logits.shape == (1, 400)
    assert y.clip_probs.shape == (1, 400)

    legacy = build_i3d(num_classes=400, modality="rgb", legacy=True).eval()
    with torch.no_grad():
        y_legacy = forward_infer(legacy, x)
    assert y_legacy.logits_per_frame.shape == (1, 400, 1)
    assert y_legacy.clip_logits.shape == (1, 400)
    assert y_legacy.clip_probs.shape == (1, 400)


def test_load_weights_for_three_formats(tmp_path: Path) -> None:
    rgb_pt_ckpt = env_or_default("RGB_PYTORCH_CKPT")
    rgb_kinetics_ckpt = env_or_default("RGB_KINETICS_CKPT")
    require_paths(rgb_pt_ckpt, rgb_kinetics_ckpt)

    canonical_from_kinetics = tmp_path / "from_kinetics_canonical.pt"
    convert_checkpoint(rgb_kinetics_ckpt, canonical_from_kinetics, src_format="kinetics_i3d", dst_format="canonical")

    for ckpt, load_format, expected in (
        (rgb_pt_ckpt, "auto", "pytorch_i3d"),
        (rgb_kinetics_ckpt, "auto", "kinetics_i3d"),
        (canonical_from_kinetics, "canonical", "canonical"),
    ):
        model = build_i3d(num_classes=400, modality="rgb", legacy=False)
        report = load_weights(model, ckpt, format=load_format, strict=True)
        assert report.source_format == expected
        assert report.missing_keys == []
        assert report.unexpected_keys == []


def test_canonical_save_and_reload_smoke(tmp_path: Path) -> None:
    rgb_kinetics_ckpt = env_or_default("RGB_KINETICS_CKPT")
    require_paths(rgb_kinetics_ckpt)

    model = build_i3d(num_classes=400, modality="rgb", legacy=True)
    load_weights(model, rgb_kinetics_ckpt, format="auto", strict=True)

    saved = tmp_path / "canonical_saved.pt"
    out = save_canonical_weights(model, saved)
    assert out == str(saved)
    assert saved.exists()

    sd = torch.load(saved, map_location="cpu")
    assert "Conv3d_1a_7x7.conv3d.weight" in sd
    assert all(not k.startswith("backbone.") for k in sd.keys())

    model2 = build_i3d(num_classes=400, modality="rgb", legacy=False)
    report = load_weights(model2, saved, format="canonical", strict=True)
    assert report.source_format == "canonical"
    assert report.missing_keys == []
    assert report.unexpected_keys == []

    sd_direct = canonical_state_dict(model)
    assert torch.allclose(sd_direct["logits.conv3d.weight"], sd["logits.conv3d.weight"])
