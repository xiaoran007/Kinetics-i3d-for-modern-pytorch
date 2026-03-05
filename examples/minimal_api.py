"""Minimal API usage example for modernized I3D."""

from kinetics_i3d.models import InceptionI3d
from kinetics_i3d.weights import load_pretrained


def main() -> None:
    model = InceptionI3d(num_classes=400, in_channels=3)
    report = load_pretrained(
        model,
        checkpoint_path="reference/pytorch-i3d/models/rgb_imagenet.pt",
        format="auto",
        strict=True,
    )
    print(
        f"loaded={report.checkpoint_path} format={report.source_format} "
        f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}"
    )


if __name__ == "__main__":
    main()
