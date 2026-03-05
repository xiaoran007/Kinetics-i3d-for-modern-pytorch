from __future__ import annotations

import os
import subprocess
import sys

from tests.conftest import SRC_ROOT, env_or_default, require_paths


def test_demo_infer_cli_smoke() -> None:
    ckpt = env_or_default("RGB_KINETICS_CKPT")
    sample = env_or_default("RGB_SAMPLE_NPY")
    labels = env_or_default("LABEL_MAP")
    require_paths(ckpt, sample, labels)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kinetics_i3d.cli.demo_infer",
            "--weights",
            str(ckpt),
            "--input-npy",
            str(sample),
            "--labels",
            str(labels),
            "--top-k",
            "1",
            "--device",
            "cpu",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Loaded checkpoint format=" in proc.stdout
    assert "playing cricket" in proc.stdout
