# Kinetics I3D for Modern PyTorch

A modern, integration-focused I3D implementation for PyTorch 2.8+, with backward-compatible loading for widely used legacy checkpoints.

## Why This Project

The original I3D ecosystem is fragmented across older TensorFlow and early PyTorch codebases. This project provides a clean, reusable implementation that:

- works on modern PyTorch (`2.8+`)
- keeps compatibility with legacy checkpoint formats
- is easy to plug into existing training or inference pipelines
- standardizes on a single canonical checkpoint format for new artifacts

## Project Goals

- Preserve I3D behavior and checkpoint compatibility.
- Provide pipeline-friendly library APIs (not a heavy framework).
- Make release-ready canonical checkpoints reproducible and verifiable.

## Key Features

- Canonical `InceptionI3d` model for modern PyTorch.
- Legacy-compatible `I3D` wrapper (`forward -> (softmax, logits)`).
- Integration API:
  - `build_i3d(...)`
  - `load_weights(...)`
  - `forward_infer(...)`
  - `prepare_finetune(...)`
  - `save_canonical_weights(...)`
- Checkpoint compatibility:
  - `pytorch-i3d` format (`Mixed_*`, `logits.*`)
  - `kinetics_i3d_pytorch` format (`mixed_*`, `conv3d_0c_1x1.*`)
  - canonical format (`InceptionI3d` state_dict)
- Optional TensorFlow checkpoint conversion utility.
- Release preparation utility for canonical weights + SHA256 + parity reports.

## Installation

Python `>=3.9`

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]   # tests
pip install -e .[tf]    # optional TF conversion
```

## Quick Start

### 1) Build model + load legacy/canonical weights

```python
from kinetics_i3d import build_i3d, load_weights

model = build_i3d(num_classes=400, modality="rgb", legacy=False)
report = load_weights(
    model,
    "reference/kinetics_i3d_pytorch/model/model_rgb.pth",
    format="auto",
    strict=True,
)
print(report.source_format, report.missing_keys, report.unexpected_keys)
```

### 2) Unified inference outputs

```python
import torch
from kinetics_i3d import forward_infer

x = torch.randn(1, 3, 16, 224, 224)
out = forward_infer(model.eval(), x)
print(out.logits_per_frame.shape, out.clip_logits.shape, out.clip_probs.shape)
```

### 3) Prepare finetuning (freeze backbone, train classifier head)

```python
import torch
from kinetics_i3d import prepare_finetune

setup = prepare_finetune(model, freeze_strategy="logits")
optimizer = torch.optim.SGD(setup.param_groups, lr=1e-3, momentum=0.9)
```

### 4) Save canonical single-file weights

```python
from kinetics_i3d import save_canonical_weights

save_canonical_weights(model, "/tmp/i3d_canonical.pt")
```

## Checkpoint Compatibility

| Source format | `load_weights(..., format=...)` | Notes |
|---|---|---|
| `pytorch-i3d` | `auto` | Canonical-equivalent naming |
| `kinetics_i3d_pytorch` | `auto` | Auto key remap to canonical |
| canonical | `canonical` or `auto` | Recommended default for new checkpoints |
| TensorFlow `model.ckpt` | optional converter | Use `kinetics_i3d.cli.convert_tf_ckpt` |

## Release Assets (Current)

Current pre-release:

- Tag: `v0.1.0-beta.1`
- Package version: `0.1.0b1`
- Channel: GitHub pre-release only (no PyPI yet)

Expected release artifacts:

- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.sha256`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1_report.md`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1_report.json`

You can generate these with:

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.prepare_release \
  --version-tag v0.1.0-beta.1 \
  --source-checkpoint reference/kinetics_i3d_pytorch/model/model_rgb.pth \
  --output-dir dist/release
```

## Optional CLI Helpers

Demo inference:

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.demo_infer \
  --weights reference/kinetics_i3d_pytorch/model/model_rgb.pth \
  --input-npy reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy \
  --labels reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt \
  --top-k 5
```

## Testing

```bash
PYTHONPATH=src conda run -n torch python -m pytest -q
```

## Scope / Non-Goals

This repository intentionally does **not** include:

- multi-camera fusion strategy implementations
- dataset-specific full training frameworks
- video decoding/data engineering pipelines

The focus is stable model code and integration APIs.

## Documentation

- Migration guide: `docs/MIGRATION.md`
- Release runbook: `docs/RELEASE.md`

## Acknowledgements

This project builds on prior community efforts:

- [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
- [hassony2/kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)

## License

MIT. See `LICENSE`.
