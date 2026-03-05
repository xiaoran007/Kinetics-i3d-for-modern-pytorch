"""Minimal API usage example for modernized I3D."""

import torch

from kinetics_i3d import build_i3d, forward_infer, load_weights, prepare_finetune


def main() -> None:
    model = build_i3d(num_classes=400, modality="rgb", legacy=False)
    report = load_weights(
        model,
        checkpoint_path="reference/pytorch-i3d/models/rgb_imagenet.pt",
        format="auto",
        strict=True,
    )
    setup = prepare_finetune(model, freeze_strategy="logits")
    x = torch.randn(1, 3, 16, 224, 224)
    out = forward_infer(model.eval(), x)
    print(
        f"loaded={report.checkpoint_path} format={report.source_format} "
        f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}"
    )
    print(
        f"trainable={len(setup.trainable_names)} "
        f"logits_shape={tuple(out.clip_logits.shape)}"
    )


if __name__ == "__main__":
    main()
