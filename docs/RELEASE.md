# GitHub Release Runbook (No PyPI)

This runbook prepares canonical I3D checkpoint assets for GitHub Release attachments.

## Target

- Tag: `v0.1.0-beta.1`
- Package version: `0.1.0b1`
- Source checkpoint: `reference/kinetics_i3d_pytorch/model/model_rgb.pth`
- Asset policy: release attachments only (no large weights committed to git)

## 1) Prepare Artifacts

Run in the `torch` environment:

```bash
PYTHONPATH=src conda run -n torch python -m kinetics_i3d.cli.prepare_release \
  --version-tag v0.1.0-beta.1 \
  --source-checkpoint reference/kinetics_i3d_pytorch/model/model_rgb.pth \
  --sample-npy reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy \
  --labels-path reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt \
  --output-dir dist/release
```

Expected files in `dist/release/`:

- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.pt`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1.sha256`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1_report.md`
- `i3d_rgb_imagenet_canonical_v0.1.0-beta.1_report.json`

## 2) Validate Test Gates

```bash
PYTHONPATH=src conda run -n torch python -m pytest -q
```

Release gate requires:

- strict load success for source + canonical checkpoints
- strict allclose parity on fixed random input
- strict allclose parity on sample clip
- top-k order match and top-1 label check (`playing cricket`)
- no `backbone.` prefix in canonical checkpoint keys

## 3) Create Tag and Push

```bash
git tag v0.1.0-beta.1
git push origin main
git push origin v0.1.0-beta.1
```

## 4) Create GitHub Pre-release

- Create **Pre-release** for `v0.1.0-beta.1`.
- Attach the 4 files from `dist/release/`.
- Use `docs/releases/v0.1.0-beta.1.md` as release note template.

## 5) Post-release Smoke Validation

In a clean environment:

- Download release checkpoint attachment
- Load with `load_weights(..., format="canonical", strict=True)`
- Run one inference pass and verify outputs are valid
