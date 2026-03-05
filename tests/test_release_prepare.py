from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from kinetics_i3d.release import prepare_release_assets
from tests.conftest import env_or_default, require_paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_prepare_release_assets_outputs_and_checks(tmp_path: Path) -> None:
    source_ckpt = env_or_default("RGB_KINETICS_CKPT")
    sample_npy = env_or_default("RGB_SAMPLE_NPY")
    labels = env_or_default("LABEL_MAP")
    require_paths(source_ckpt, sample_npy, labels)

    version = "v0.1.0-beta.1"
    summary = prepare_release_assets(
        version_tag=version,
        source_checkpoint=source_ckpt,
        sample_npy=sample_npy,
        labels_path=labels,
        output_dir=tmp_path,
        top_k=5,
        fail_on_error=True,
    )

    stem = f"i3d_rgb_imagenet_canonical_{version}"
    ckpt_path = tmp_path / f"{stem}.pt"
    sha_path = tmp_path / f"{stem}.sha256"
    md_path = tmp_path / f"{stem}_report.md"
    json_path = tmp_path / f"{stem}_report.json"

    assert ckpt_path.exists()
    assert sha_path.exists()
    assert md_path.exists()
    assert json_path.exists()

    assert summary["all_required_checks_passed"] is True
    assert summary["source_format"] == "kinetics_i3d"
    assert summary["checks"]["random_clip_logits_strict_allclose"] is True
    assert summary["checks"]["sample_clip_logits_strict_allclose"] is True
    assert summary["checks"]["sample_topk_order_match"] is True
    assert summary["checks"]["sample_top1_label_is_expected"] is True
    assert summary["checks"]["canonical_reload_strict_allclose"] is True
    assert summary["checks"]["no_backbone_prefix_in_canonical_keys"] is True

    written_sha = sha_path.read_text().strip().split()[0]
    assert written_sha == _sha256(ckpt_path)

    report_json = json.loads(json_path.read_text())
    assert report_json["artifacts"]["canonical_checkpoint"] == str(ckpt_path)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    assert "Conv3d_1a_7x7.conv3d.weight" in state_dict
    assert all(not k.startswith("backbone.") for k in state_dict)
