# Migration Guide (Legacy I3D -> Modern PyTorch 2.8)

## Import Mapping

- Old (`pytorch-i3d`):
  - `from pytorch_i3d import InceptionI3d`
- New:
  - `from kinetics_i3d.models import InceptionI3d`

- Old (`kinetics_i3d_pytorch`):
  - `from src.i3dpt import I3D`
- New:
  - `from kinetics_i3d.models import I3D`

## Checkpoint Loading Mapping

Old code usually did:

```python
model.load_state_dict(torch.load(path))
```

New recommended pattern:

```python
from kinetics_i3d.weights import load_pretrained

report = load_pretrained(model, path, format="auto", strict=True)
```

This enables automatic compatibility with both legacy naming schemes.

## Legacy Output Semantics

- `InceptionI3d.forward(x)` returns `logits_per_frame` with shape `(B, C, T)`.
- `I3D.forward(x)` returns `(softmax, logits)` with shape `(B, C)` for legacy behavior.

## Conversion API

To convert checkpoints explicitly:

```python
from kinetics_i3d.weights import convert_checkpoint

convert_checkpoint(src_checkpoint, dst_checkpoint, src_format="kinetics_i3d", dst_format="canonical")
```

## CLI Migration

- Legacy demo scripts can be replaced by:
  - `python -m kinetics_i3d.cli.demo_infer ...`
- Optional TF conversion:
  - `python -m kinetics_i3d.cli.convert_tf_ckpt ...`
