"""Shared test helpers and reference data path conventions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

DEFAULT_PATHS = {
    "RGB_PYTORCH_CKPT": REPO_ROOT / "reference/pytorch-i3d/models/rgb_imagenet.pt",
    "RGB_KINETICS_CKPT": REPO_ROOT / "reference/kinetics_i3d_pytorch/model/model_rgb.pth",
    "RGB_SAMPLE_NPY": REPO_ROOT / "reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy",
    "LABEL_MAP": REPO_ROOT / "reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt",
}


def env_or_default(name: str) -> Path:
    env_name = f"I3D_TEST_{name}"
    if env_name in os.environ and os.environ[env_name].strip():
        return Path(os.environ[env_name]).expanduser().resolve()
    return DEFAULT_PATHS[name]


def require_paths(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"reference test artifacts not found: {missing}")
