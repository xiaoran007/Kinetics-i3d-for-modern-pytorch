"""CLI for preparing release-ready canonical checkpoint assets."""

from __future__ import annotations

import argparse
import json

from kinetics_i3d.release import prepare_release_assets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical release artifacts for I3D")
    parser.add_argument("--version-tag", type=str, default="v0.1.0-beta.1")
    parser.add_argument(
        "--source-checkpoint",
        type=str,
        default="reference/kinetics_i3d_pytorch/model/model_rgb.pth",
    )
    parser.add_argument(
        "--sample-npy",
        type=str,
        default="reference/kinetics_i3d_pytorch/data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="reference/kinetics_i3d_pytorch/data/kinetic-samples/label_map.txt",
    )
    parser.add_argument("--output-dir", type=str, default="dist/release")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260305)
    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Write report even if checks fail; exit code remains 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_release_assets(
        version_tag=args.version_tag,
        source_checkpoint=args.source_checkpoint,
        sample_npy=args.sample_npy,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        random_seed=args.random_seed,
        fail_on_error=not args.no_fail_on_error,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
