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

Requires Python `>=3.9`.

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]   # tests
pip install -e .[tf]    # optional TensorFlow checkpoint conversion
```

## Get Pretrained Weights (Recommended)

Use release assets (canonical format).

### 1) Download from GitHub Release

From release tag `v0.1.0-beta.1`, download:

- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.sha256`


### 2) Verify checksum

```bash
shasum -a 256 i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt
cat i3d_rgb_imagenet_canonical_v0.1.0-beta.1.sha256
```

## Integration Guide

### Input contract

Model input is `torch.Tensor` in shape `(B, C, T, H, W)`.

- RGB: `C=3`
- Flow: `C=2`
- Expected preprocessing is project/pipeline specific (sampling, resize/crop, normalization).

### Inference integration

```python
import torch
from kinetics_i3d import build_i3d, load_weights, forward_infer

model = build_i3d(num_classes=400, modality="rgb", legacy=False).eval()
load_weights(
    model,
    checkpoint_path="/path/to/i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt",
    format="canonical",
    strict=True,
)

# clip: (B, C, T, H, W)
clip = torch.randn(1, 3, 16, 224, 224)
out = forward_infer(model, clip)

# pipeline-friendly outputs
logits_per_frame = out.logits_per_frame   # (B, C, T')
clip_logits = out.clip_logits              # (B, C)
clip_probs = out.clip_probs                # (B, C)
```

### Finetuning integration (freeze backbone, train classifier head)

```python
import torch
from kinetics_i3d import build_i3d, load_weights, prepare_finetune

num_classes = 2  # example: fall / non-fall
model = build_i3d(num_classes=400, modality="rgb", legacy=False)
load_weights(model, "/path/to/i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt", format="canonical", strict=True)

# Replace classification head for your task
model.replace_logits(num_classes)

# Freeze backbone, train logits only
setup = prepare_finetune(model, freeze_strategy="logits")
optimizer = torch.optim.SGD(setup.param_groups, lr=1e-3, momentum=0.9)
```

### Save your own canonical checkpoint

```python
from kinetics_i3d import save_canonical_weights

save_canonical_weights(model, "/path/to/my_i3d_checkpoint.pt")
```

## API Summary

- `build_i3d(...)`
- `load_weights(...)`
- `forward_infer(...)`
- `prepare_finetune(...)`
- `save_canonical_weights(...)`
- `convert_checkpoint(...)`

## Legacy Checkpoint Compatibility (Optional)

If you already have older checkpoints, `load_weights(..., format="auto")` supports:

- `pytorch-i3d` naming (`Mixed_*`, `logits.*`)
- `kinetics_i3d_pytorch` naming (`mixed_*`, `conv3d_0c_1x1.*`)
- canonical naming

For migration to canonical files:

```python
from kinetics_i3d import convert_checkpoint

convert_checkpoint(
    src_checkpoint="/path/to/legacy_checkpoint.pt",
    dst_checkpoint="/path/to/canonical_checkpoint.pt",
    src_format="auto",
    dst_format="canonical",
)
```

## Current Release

- Tag: `v0.1.0-beta.1` (pre-release)
- Package version: `0.1.0b1`
- Scope: RGB imagenet canonical checkpoint release
- Channel: GitHub release assets only (no PyPI yet)

## Development and Validation

Run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Release preparation docs and templates:

- `docs/RELEASE.md`
- `docs/releases/v0.1.0-beta.1.md`

## Non-goals

This repository does not provide:

- full dataset/training framework
- video decoding pipeline
- multi-camera fusion implementation

It focuses on stable model code + integration APIs.

## Acknowledgements

This project builds on prior community efforts:

- [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)
- [hassony2/kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch)

## License

MIT. See `LICENSE`.
