"""Original-style I3D demo inference for numpy clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kinetics_i3d.models import I3D, InceptionI3d
from kinetics_i3d.weights import load_pretrained


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I3D inference on numpy clips")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input-npy", type=str, required=True, help="Path to numpy clip")
    parser.add_argument("--labels", type=str, default=None, help="Optional label map file (one label per line)")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "canonical", "pytorch_i3d", "kinetics_i3d"])
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "flow"])
    parser.add_argument("--num-classes", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--legacy", action="store_true", help="Use legacy I3D wrapper output semantics")
    parser.add_argument("--device", type=str, default="cpu", help="e.g. cpu, cuda:0, mps")
    return parser.parse_args()


def _load_labels(path: str | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _to_bcthw(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 5 and array.shape[1] in (2, 3):
        # B, C, T, H, W
        tensor = torch.from_numpy(array)
    elif array.ndim == 5 and array.shape[-1] in (2, 3):
        # B, T, H, W, C -> B, C, T, H, W
        tensor = torch.from_numpy(array).permute(0, 4, 1, 2, 3)
    elif array.ndim == 4 and array.shape[-1] in (2, 3):
        # T, H, W, C -> 1, C, T, H, W
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
    else:
        raise ValueError(
            "Unsupported numpy shape. Expected BxCxTxHxW, BxTxHxWxC, or TxHxWxC with channels in {2,3}."
        )

    return tensor.to(dtype=torch.float32)


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.legacy:
        model = I3D(num_classes=args.num_classes, modality=args.modality)
    else:
        in_channels = 3 if args.modality == "rgb" else 2
        model = InceptionI3d(num_classes=args.num_classes, in_channels=in_channels)

    report = load_pretrained(model, checkpoint_path=args.weights, format=args.format, strict=True)
    print(
        f"Loaded checkpoint format={report.source_format}; "
        f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}"
    )
    return model


def main() -> None:
    args = _parse_args()
    labels = _load_labels(args.labels)

    device = torch.device(args.device)
    clip_np = np.load(args.input_npy)
    clip = _to_bcthw(clip_np).to(device)

    model = _build_model(args).to(device).eval()

    with torch.no_grad():
        outputs = model(clip)

    if args.legacy:
        probs, logits = outputs
    else:
        logits = outputs.mean(dim=2)
        probs = torch.softmax(logits, dim=1)

    top_vals, top_idx = torch.topk(probs, k=min(args.top_k, probs.shape[1]), dim=1)

    for rank in range(top_vals.shape[1]):
        cls_idx = int(top_idx[0, rank].item())
        prob = float(top_vals[0, rank].item())
        logit = float(logits[0, cls_idx].item())
        if labels is not None and cls_idx < len(labels):
            label = labels[cls_idx]
        else:
            label = f"class_{cls_idx}"
        print(f"{rank+1:02d}. p={prob:.6e} logit={logit:.6e} {label}")


if __name__ == "__main__":
    main()
