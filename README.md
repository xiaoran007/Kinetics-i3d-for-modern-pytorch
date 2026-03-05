# Kinetics I3D for Modern PyTorch

A modernized I3D baseline implementation for **PyTorch 2.8**, with backward-compatible loading for two widely used community checkpoints:

- `pytorch-i3d` format (`Mixed_*`, `logits.*`) from this [repo](https://github.com/piergiaj/pytorch-i3d#)
- `kinetics_i3d_pytorch` format (`mixed_*`, `conv3d_0c_1x1.*`) from this [repo](https://github.com/hassony2/kinetics_i3d_pytorch#)

This repository focuses on model modernization and compatibility, not dataset-specific training pipelines.

## Scope (v1)

- Canonical `InceptionI3d` model for PyTorch 2.8 and newer
- Legacy-compatible `I3D` wrapper (`forward -> (softmax, logits)`)
- Unified checkpoint loader with format auto-detection and conversion
- Numpy clip demo inference CLI
- Optional TensorFlow checkpoint conversion CLI
- Unit/parity/integration tests

## Install

Use your existing environment:

```bash
pip install -e .
```

Dev dependencies:

```bash
pip install -e .[dev]
```

Optional TensorFlow converter dependencies:

```bash
pip install -e .[tf]
```

## Quick Start

### 1) Canonical model + auto checkpoint loading

```python
from kinetics_i3d.models import InceptionI3d
from kinetics_i3d.weights import load_pretrained

model = InceptionI3d(num_classes=400, in_channels=3)
report = load_pretrained(model, "reference/kinetics_i3d_pytorch/model/model_rgb.pth", format="auto")
print(report.source_format, report.missing_keys, report.unexpected_keys)
```

### 2) Legacy-compatible wrapper

```python
from kinetics_i3d.models import I3D
from kinetics_i3d.weights import load_pretrained

model = I3D(num_classes=400, modality="rgb")
load_pretrained(model, "reference/kinetics_i3d_pytorch/model/model_rgb.pth")
```

### 3) Demo inference (numpy clip)

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.demo_infer \
  --weights reference/kinetics_i3d_pytorch/model/model_rgb.pth \
  --input-npy reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy \
  --labels reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt \
  --top-k 5
```

### 4) Convert checkpoint format

```python
from kinetics_i3d.weights import convert_checkpoint

convert_checkpoint(
    "reference/kinetics_i3d_pytorch/model/model_rgb.pth",
    "/tmp/model_rgb_canonical.pt",
    src_format="kinetics_i3d",
    dst_format="canonical",
)
```

### 5) Optional TensorFlow checkpoint conversion

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.convert_tf_ckpt \
  --tf-checkpoint reference/kinetics_i3d_pytorch/model/tf_rgb_imagenet/model.ckpt \
  --dst /tmp/model_rgb_from_tf.pt \
  --modality rgb
```

## Checkpoint Compatibility Matrix

| Source checkpoint format | `load_pretrained(..., format="auto")` | Notes |
|---|---|---|
| `pytorch-i3d` (`Mixed_*`) | Supported | Treated as canonical naming |
| `kinetics_i3d_pytorch` (`mixed_*`) | Supported | Auto key remap to canonical |
| Canonical (`InceptionI3d`) | Supported | Direct load |
| TensorFlow `model.ckpt` | Optional | Use TF converter CLI/API |

## Tests

Run tests in the `torch` conda env:

```bash
PYTHONPATH=src conda run -n torch python -m pytest -q
```

Reference artifact path conventions for integration tests are documented in `tests/README.md`.

## Important Notes

- This repo does **not** commit large pretrained weight files.
- v1 intentionally excludes multi-camera fusion and full training pipeline integration.
- Device policy is script/runtime-level (`--device`), not hardcoded into model definitions.

## Migration Guide

See `docs/MIGRATION.md` for import/API mapping from legacy community repos.
