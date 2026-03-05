"""Modernized I3D package for PyTorch 2.8."""

from .api import (
    FinetuneSetup,
    InferenceOutput,
    build_i3d,
    canonical_state_dict,
    forward_infer,
    load_weights,
    prepare_finetune,
    save_canonical_weights,
)
from .models import I3D, InceptionI3d
from .weights import LoadReport, convert_checkpoint, convert_state_dict, detect_checkpoint_format, load_pretrained

__all__ = [
    "I3D",
    "InceptionI3d",
    "InferenceOutput",
    "FinetuneSetup",
    "build_i3d",
    "load_weights",
    "prepare_finetune",
    "forward_infer",
    "canonical_state_dict",
    "save_canonical_weights",
    "LoadReport",
    "convert_checkpoint",
    "convert_state_dict",
    "detect_checkpoint_format",
    "load_pretrained",
    "__version__",
]

__version__ = "0.1.0b1"
