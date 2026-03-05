# Migration Guide (Legacy I3D -> Modern PyTorch 2.8)

## Import Mapping

- Old (`pytorch-i3d`):
  - `from pytorch_i3d import InceptionI3d`
- New:
  - `from kinetics_i3d import build_i3d, load_weights, forward_infer`

- Old (`kinetics_i3d_pytorch`):
  - `from src.i3dpt import I3D`
- New:
  - `from kinetics_i3d import build_i3d, load_weights, prepare_finetune`

You can still import low-level classes from `kinetics_i3d.models`, but the
recommended entrypoint for new projects is the package-level API.

## Build + Load Mapping

Old code usually did:

```python
model = InceptionI3d(...)
model.load_state_dict(torch.load(path))
```

New recommended pattern:

```python
from kinetics_i3d import build_i3d, load_weights

model = build_i3d(num_classes=400, modality="rgb", legacy=False)
report = load_weights(model, path, format="auto", strict=True)
```

This enables automatic compatibility with both legacy naming schemes.

## Unified Inference Output

```python
from kinetics_i3d import forward_infer

out = forward_infer(model, inputs)
# out.logits_per_frame: (B, C, T)
# out.clip_logits:      (B, C)
# out.clip_probs:       (B, C)
```

## Legacy Output Semantics

- `InceptionI3d.forward(x)` returns `logits_per_frame` with shape `(B, C, T)`.
- `I3D.forward(x)` returns `(softmax, logits)` with shape `(B, C)` for legacy behavior.

## Finetuning Mapping

```python
from kinetics_i3d import prepare_finetune

setup = prepare_finetune(model, freeze_strategy="logits")
optimizer = torch.optim.SGD(setup.param_groups, lr=1e-3, momentum=0.9)
```

Supported freeze strategies:

- `none`: train all
- `all`: freeze all
- `logits`: train only `logits.*`
- `prefixes`: train only given parameter prefixes

## Canonical Weight Saving

```python
from kinetics_i3d import save_canonical_weights

save_canonical_weights(model, "/tmp/i3d_canonical.pt")
```

This is the recommended single-file weight format for new checkpoints.

## Conversion API

To convert checkpoints explicitly:

```python
from kinetics_i3d.weights import convert_checkpoint

convert_checkpoint(src_checkpoint, dst_checkpoint, src_format="kinetics_i3d", dst_format="canonical")
```

## CLI Migration

CLI modules are still available for convenience, but pipeline integration should
prefer package-level API calls:

- `python -m kinetics_i3d.cli.demo_infer ...`
- `python -m kinetics_i3d.cli.convert_tf_ckpt ...`
