"""Optional TensorFlow checkpoint conversion for I3D.

This module intentionally keeps TensorFlow as an optional dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from kinetics_i3d.models import InceptionI3d


def _get_tf_reader(tf_checkpoint: str):
    try:
        import tensorflow.compat.v1 as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when TF is available
        raise RuntimeError(
            "TensorFlow is not installed. Install optional dependency with: pip install .[tf]"
        ) from exc

    tf.disable_v2_behavior()
    return tf.train.load_checkpoint(tf_checkpoint)


def _load_unit3d_from_tf(
    state_dict: dict[str, torch.Tensor],
    reader,
    pt_prefix: str,
    tf_prefix: str,
    *,
    bias: bool = False,
    bn: bool = True,
) -> None:
    conv_w = reader.get_tensor(f"{tf_prefix}/conv_3d/w")
    conv_w = np.transpose(conv_w, (4, 3, 0, 1, 2))
    state_dict[f"{pt_prefix}.conv3d.weight"] = torch.from_numpy(conv_w)

    if bias:
        conv_b = reader.get_tensor(f"{tf_prefix}/conv_3d/b")
        state_dict[f"{pt_prefix}.conv3d.bias"] = torch.from_numpy(conv_b)

    if bn:
        beta = reader.get_tensor(f"{tf_prefix}/batch_norm/beta")
        moving_mean = reader.get_tensor(f"{tf_prefix}/batch_norm/moving_mean")
        moving_var = reader.get_tensor(f"{tf_prefix}/batch_norm/moving_variance")

        beta_t = torch.from_numpy(beta)
        state_dict[f"{pt_prefix}.bn.weight"] = torch.ones_like(beta_t)
        state_dict[f"{pt_prefix}.bn.bias"] = beta_t
        state_dict[f"{pt_prefix}.bn.running_mean"] = torch.from_numpy(moving_mean)
        state_dict[f"{pt_prefix}.bn.running_var"] = torch.from_numpy(moving_var)


def _mixed_branch_to_tf_path(mixed_name: str, branch_name: str) -> str:
    if branch_name == "b0":
        return f"{mixed_name}/Branch_0/Conv3d_0a_1x1"
    if branch_name == "b1a":
        return f"{mixed_name}/Branch_1/Conv3d_0a_1x1"
    if branch_name == "b1b":
        return f"{mixed_name}/Branch_1/Conv3d_0b_3x3"
    if branch_name == "b2a":
        return f"{mixed_name}/Branch_2/Conv3d_0a_1x1"
    if branch_name == "b2b":
        # Keep typo compatibility from the original conversion scripts.
        if mixed_name == "Mixed_5b":
            return f"{mixed_name}/Branch_2/Conv3d_0a_3x3"
        return f"{mixed_name}/Branch_2/Conv3d_0b_3x3"
    if branch_name == "b3b":
        return f"{mixed_name}/Branch_3/Conv3d_0b_1x1"
    raise ValueError(f"Unsupported branch name: {branch_name}")


def convert_tf_checkpoint_to_state_dict(
    tf_checkpoint: str | Path,
    modality: Literal["rgb", "flow"] = "rgb",
) -> dict[str, torch.Tensor]:
    tf_checkpoint = str(tf_checkpoint)
    reader = _get_tf_reader(tf_checkpoint)

    scope = "RGB" if modality == "rgb" else "Flow"
    tf_root = f"{scope}/inception_i3d"

    state_dict: dict[str, torch.Tensor] = {}

    _load_unit3d_from_tf(state_dict, reader, "Conv3d_1a_7x7", f"{tf_root}/Conv3d_1a_7x7")
    _load_unit3d_from_tf(state_dict, reader, "Conv3d_2b_1x1", f"{tf_root}/Conv3d_2b_1x1")
    _load_unit3d_from_tf(state_dict, reader, "Conv3d_2c_3x3", f"{tf_root}/Conv3d_2c_3x3")

    mixed_blocks = ["Mixed_3b", "Mixed_3c", "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e", "Mixed_4f", "Mixed_5b", "Mixed_5c"]
    branches = ["b0", "b1a", "b1b", "b2a", "b2b", "b3b"]

    for mixed_name in mixed_blocks:
        for branch_name in branches:
            pt_prefix = f"{mixed_name}.{branch_name}"
            tf_prefix = f"{tf_root}/{_mixed_branch_to_tf_path(mixed_name, branch_name)}"
            _load_unit3d_from_tf(state_dict, reader, pt_prefix, tf_prefix)

    _load_unit3d_from_tf(
        state_dict,
        reader,
        "logits",
        f"{tf_root}/Logits/Conv3d_0c_1x1",
        bias=True,
        bn=False,
    )

    # Validate converted state_dict against canonical architecture.
    in_channels = 3 if modality == "rgb" else 2
    model = InceptionI3d(num_classes=400, in_channels=in_channels)
    model.load_state_dict(state_dict, strict=True)

    return state_dict


def convert_tf_checkpoint(
    tf_checkpoint: str | Path,
    dst_checkpoint: str | Path,
    modality: Literal["rgb", "flow"] = "rgb",
) -> str:
    state_dict = convert_tf_checkpoint_to_state_dict(tf_checkpoint=tf_checkpoint, modality=modality)
    dst_checkpoint = str(dst_checkpoint)
    torch.save(state_dict, dst_checkpoint)
    return dst_checkpoint
