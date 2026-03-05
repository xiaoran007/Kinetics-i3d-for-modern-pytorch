"""Utilities to prepare canonical checkpoint assets for GitHub Releases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kinetics_i3d import build_i3d, forward_infer, load_weights, save_canonical_weights

DEFAULT_VERSION_TAG = "v0.1.0-beta.1"
DEFAULT_SOURCE_CHECKPOINT = Path("reference/kinetics_i3d_pytorch/model/model_rgb.pth")
DEFAULT_SAMPLE_NPY = Path("reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy")
DEFAULT_LABELS = Path("reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt")
DEFAULT_OUTPUT_DIR = Path("dist/release")
DEFAULT_TOP_K = 5
EXPECTED_TOP1_LABEL = "playing cricket"


@dataclass
class ReleaseArtifactPaths:
    canonical_checkpoint: Path
    sha256_file: Path
    markdown_report: Path
    json_report: Path


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_bcthw(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 5 and array.shape[1] in (2, 3):
        tensor = torch.from_numpy(array)
    elif array.ndim == 5 and array.shape[-1] in (2, 3):
        tensor = torch.from_numpy(array).permute(0, 4, 1, 2, 3)
    elif array.ndim == 4 and array.shape[-1] in (2, 3):
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
    else:
        raise ValueError(
            "Unsupported sample shape. Expected BxCxTxHxW, BxTxHxWxC, or TxHxWxC with channels in {2,3}."
        )
    return tensor.to(dtype=torch.float32)


def _build_artifact_paths(output_dir: Path, version_tag: str) -> ReleaseArtifactPaths:
    stem = f"i3d_rgb_imagenet_canonical_{version_tag}"
    return ReleaseArtifactPaths(
        canonical_checkpoint=output_dir / f"{stem}.pt",
        sha256_file=output_dir / f"{stem}.sha256",
        markdown_report=output_dir / f"{stem}_report.md",
        json_report=output_dir / f"{stem}_report.json",
    )


def _load_labels(path: Path) -> list[str]:
    labels = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def _format_md_report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    demo_topk = summary["demo_topk"]
    artifacts = summary["artifacts"]

    lines = [
        f"# I3D Release Preparation Report ({summary['version_tag']})",
        "",
        "## Summary",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Source checkpoint: `{summary['source_checkpoint']}`",
        f"- Source format detected: `{summary['source_format']}`",
        f"- Canonical checkpoint: `{artifacts['canonical_checkpoint']}`",
        f"- SHA256: `{summary['canonical_sha256']}`",
        f"- Required gates passed: `{summary['all_required_checks_passed']}`",
        "",
        "## Required Checks",
    ]

    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Parity Metrics",
            f"- `random_clip_logits_max_abs_diff`: `{summary['metrics']['random_clip_logits_max_abs_diff']}`",
            f"- `sample_clip_logits_max_abs_diff`: `{summary['metrics']['sample_clip_logits_max_abs_diff']}`",
            "",
            "## Demo Top-K (Canonical)",
            "",
            "| Rank | Index | Label | Source Prob | Canonical Prob |",
            "|---|---:|---|---:|---:|",
        ]
    )

    for row in demo_topk:
        lines.append(
            f"| {row['rank']} | {row['index']} | {row['label']} | {row['source_prob']:.9f} | {row['canonical_prob']:.9f} |"
        )

    lines.extend(
        [
            "",
            "## License Notice",
            "- Converted checkpoint source: `kinetics_i3d_pytorch`",
            "- Source repository license: `MIT`",
            "- This artifact is redistributed with source attribution; users are responsible for compliance with upstream terms.",
        ]
    )

    return "\n".join(lines) + "\n"


def prepare_release_assets(
    *,
    version_tag: str = DEFAULT_VERSION_TAG,
    source_checkpoint: str | Path = DEFAULT_SOURCE_CHECKPOINT,
    sample_npy: str | Path = DEFAULT_SAMPLE_NPY,
    labels_path: str | Path = DEFAULT_LABELS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = DEFAULT_TOP_K,
    random_seed: int = 20260305,
    fail_on_error: bool = True,
) -> dict[str, Any]:
    source_checkpoint = Path(source_checkpoint)
    sample_npy = Path(sample_npy)
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)

    for required in (source_checkpoint, sample_npy, labels_path):
        if not required.exists():
            raise FileNotFoundError(f"Required input file not found: {required}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _build_artifact_paths(output_dir, version_tag)

    # Load source (legacy-format) checkpoint and convert by saving canonical state_dict.
    model_source = build_i3d(num_classes=400, modality="rgb", legacy=True).eval()
    source_report = load_weights(model_source, source_checkpoint, format="auto", strict=True)
    save_canonical_weights(model_source, artifacts.canonical_checkpoint)

    sha256 = _compute_sha256(artifacts.canonical_checkpoint)
    artifacts.sha256_file.write_text(f"{sha256}  {artifacts.canonical_checkpoint.name}\n")

    # Load canonical checkpoint for parity checks.
    model_canonical = build_i3d(num_classes=400, modality="rgb", legacy=False).eval()
    canonical_report = load_weights(model_canonical, artifacts.canonical_checkpoint, format="canonical", strict=True)

    # Gate 1: strict parity on fixed random input.
    torch.manual_seed(random_seed)
    random_input = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        source_out_random = forward_infer(model_source, random_input)
        canonical_out_random = forward_infer(model_canonical, random_input)

    random_allclose = torch.allclose(
        source_out_random.clip_logits,
        canonical_out_random.clip_logits,
        atol=0.0,
        rtol=0.0,
    )
    random_max_abs_diff = float((source_out_random.clip_logits - canonical_out_random.clip_logits).abs().max().item())

    # Gate 2: sample clip top-k consistency and expected top-1 label.
    labels = _load_labels(labels_path)
    sample_clip = _to_bcthw(np.load(sample_npy))

    with torch.no_grad():
        source_out_sample = forward_infer(model_source, sample_clip)
        canonical_out_sample = forward_infer(model_canonical, sample_clip)

    sample_allclose = torch.allclose(
        source_out_sample.clip_logits,
        canonical_out_sample.clip_logits,
        atol=0.0,
        rtol=0.0,
    )
    sample_max_abs_diff = float((source_out_sample.clip_logits - canonical_out_sample.clip_logits).abs().max().item())

    k = min(top_k, source_out_sample.clip_probs.shape[1])
    source_top_vals, source_top_idx = torch.topk(source_out_sample.clip_probs, k=k, dim=1)
    canonical_top_vals, canonical_top_idx = torch.topk(canonical_out_sample.clip_probs, k=k, dim=1)

    topk_order_match = bool(torch.equal(source_top_idx, canonical_top_idx))
    top1_idx = int(canonical_top_idx[0, 0].item())
    top1_label = labels[top1_idx] if top1_idx < len(labels) else f"class_{top1_idx}"
    top1_expected_match = top1_label == EXPECTED_TOP1_LABEL

    # Gate 3: reload canonical and verify strict parity again.
    model_canonical_reloaded = build_i3d(num_classes=400, modality="rgb", legacy=False).eval()
    canonical_reload_report = load_weights(
        model_canonical_reloaded,
        artifacts.canonical_checkpoint,
        format="canonical",
        strict=True,
    )
    with torch.no_grad():
        canonical_reloaded_out = forward_infer(model_canonical_reloaded, random_input)

    canonical_reload_allclose = torch.allclose(
        canonical_out_random.clip_logits,
        canonical_reloaded_out.clip_logits,
        atol=0.0,
        rtol=0.0,
    )

    raw_state_dict = torch.load(artifacts.canonical_checkpoint, map_location="cpu")
    no_backbone_prefix = all(not k.startswith("backbone.") for k in raw_state_dict)

    checks = {
        "source_strict_load_ok": source_report.missing_keys == [] and source_report.unexpected_keys == [],
        "canonical_strict_load_ok": canonical_report.missing_keys == [] and canonical_report.unexpected_keys == [],
        "canonical_reload_strict_load_ok": canonical_reload_report.missing_keys == []
        and canonical_reload_report.unexpected_keys == [],
        "no_backbone_prefix_in_canonical_keys": no_backbone_prefix,
        "random_clip_logits_strict_allclose": bool(random_allclose),
        "sample_clip_logits_strict_allclose": bool(sample_allclose),
        "sample_topk_order_match": topk_order_match,
        "sample_top1_label_is_expected": top1_expected_match,
        "canonical_reload_strict_allclose": bool(canonical_reload_allclose),
    }

    demo_topk: list[dict[str, Any]] = []
    for rank in range(k):
        idx = int(canonical_top_idx[0, rank].item())
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        demo_topk.append(
            {
                "rank": rank + 1,
                "index": idx,
                "label": label,
                "source_prob": float(source_top_vals[0, rank].item()),
                "canonical_prob": float(canonical_top_vals[0, rank].item()),
            }
        )

    summary = {
        "version_tag": version_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_checkpoint),
        "source_format": source_report.source_format,
        "canonical_sha256": sha256,
        "all_required_checks_passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "random_clip_logits_max_abs_diff": random_max_abs_diff,
            "sample_clip_logits_max_abs_diff": sample_max_abs_diff,
        },
        "demo_topk": demo_topk,
        "artifacts": {
            "canonical_checkpoint": str(artifacts.canonical_checkpoint),
            "sha256_file": str(artifacts.sha256_file),
            "markdown_report": str(artifacts.markdown_report),
            "json_report": str(artifacts.json_report),
        },
        "license_notice": {
            "source_repo": "kinetics_i3d_pytorch",
            "source_license": "MIT",
            "source_checkpoint_name": source_checkpoint.name,
        },
    }

    artifacts.json_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifacts.markdown_report.write_text(_format_md_report(summary))

    if fail_on_error and not summary["all_required_checks_passed"]:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"Release preparation checks failed: {failed}")

    return summary


def main() -> None:
    """Small wrapper for manual execution without CLI module."""

    summary = prepare_release_assets()
    print(json.dumps(summary, indent=2))
