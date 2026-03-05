"""CLI for converting TensorFlow I3D checkpoints into canonical PyTorch checkpoints."""

from __future__ import annotations

import argparse

from kinetics_i3d.weights.tf_convert import convert_tf_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TensorFlow I3D checkpoint to PyTorch state_dict")
    parser.add_argument("--tf-checkpoint", type=str, required=True, help="Path prefix to TensorFlow checkpoint (e.g. model.ckpt)")
    parser.add_argument("--dst", type=str, required=True, help="Output path for converted PyTorch checkpoint")
    parser.add_argument("--modality", type=str, choices=["rgb", "flow"], default="rgb")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = convert_tf_checkpoint(args.tf_checkpoint, args.dst, modality=args.modality)
    print(f"Converted TensorFlow checkpoint saved to: {out}")


if __name__ == "__main__":
    main()
