# Kinetics I3D for Modern PyTorch

A modernized I3D baseline implementation for **PyTorch 2.8**, with backward-compatible loading for two widely used community checkpoints:

- `pytorch-i3d` format (`Mixed_*`, `logits.*`) from this [repo](https://github.com/piergiaj/pytorch-i3d#)
- `kinetics_i3d_pytorch` format (`mixed_*`, `conv3d_0c_1x1.*`) from this [repo](https://github.com/hassony2/kinetics_i3d_pytorch#)

This repository focuses on model modernization and integration-friendly library APIs, not dataset-specific training pipelines.

## Scope (Phase 2)

- Canonical `InceptionI3d` model for PyTorch 2.8 and newer
- Legacy-compatible `I3D` wrapper (`forward -> (softmax, logits)`)
- Integration-oriented API:
  - `build_i3d(...)`
  - `load_weights(...)`
  - `prepare_finetune(...)`
  - `forward_infer(...)`
  - `save_canonical_weights(...)`
- Unified checkpoint loader with format compatibility + conversion
- Single-file canonical weight default (`torch.save(state_dict, ...)`)
- Numpy clip demo CLI (optional helper)
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

## Integration Quick Start

### 1) Build + load

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

### 2) Unified inference output (pipeline-friendly)

```python
import torch
from kinetics_i3d import forward_infer

x = torch.randn(1, 3, 16, 224, 224)
out = forward_infer(model, x)
print(out.logits_per_frame.shape, out.clip_logits.shape, out.clip_probs.shape)
```

### 3) Finetune preparation (freeze backbone, train classifier head)

```python
from kinetics_i3d import prepare_finetune

setup = prepare_finetune(model, freeze_strategy="logits")
optimizer = torch.optim.SGD(setup.param_groups, lr=1e-3, momentum=0.9)
```

### 4) Save canonical single-file weights

```python
from kinetics_i3d import save_canonical_weights

save_canonical_weights(model, "/tmp/i3d_canonical.pt")
```

### 5) Optional checkpoint conversion helper

```python
from kinetics_i3d import convert_checkpoint

convert_checkpoint(
    "reference/kinetics_i3d_pytorch/model/model_rgb.pth",
    "/tmp/model_rgb_canonical.pt",
    src_format="kinetics_i3d",
    dst_format="canonical",
)
```

## Checkpoint Compatibility Matrix

| Source checkpoint format | `load_weights(..., format=...)` | Notes |
|---|---|---|
| `pytorch-i3d` (`Mixed_*`) | Supported (`auto`) | Canonical-equivalent naming |
| `kinetics_i3d_pytorch` (`mixed_*`) | Supported (`auto`) | Auto key remap to canonical |
| Canonical (`InceptionI3d`) | Supported (`canonical` or `auto`) | Recommended default save format |
| TensorFlow `model.ckpt` | Optional | Use TF converter CLI/API |

## Optional CLI helpers

Demo inference (numpy clip):

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.demo_infer \
  --weights reference/kinetics_i3d_pytorch/model/model_rgb.pth \
  --input-npy reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy \
  --labels reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt \
  --top-k 5
```

TensorFlow checkpoint conversion:

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.convert_tf_ckpt \
  --tf-checkpoint reference/kinetics_i3d_pytorch/model/tf_rgb_imagenet/model.ckpt \
  --dst /tmp/model_rgb_from_tf.pt \
  --modality rgb
```

## Tests

Run tests in the `torch` conda env:

```bash
PYTHONPATH=src conda run -n torch python -m pytest -q
```

Reference artifact path conventions for integration tests are documented in `tests/README.md`.

## Important Notes

- This repo does **not** commit large pretrained weight files.
- This repo intentionally excludes multi-camera fusion and full training framework code.
- Device policy is script/runtime-level (`--device`), not hardcoded into model definitions.

## Migration Guide

See `docs/MIGRATION.md` for import/API mapping from legacy community repos.
