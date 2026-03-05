"""Modernized I3D package for PyTorch 2.8."""

from .models import I3D, InceptionI3d
from .weights import LoadReport, convert_checkpoint, convert_state_dict, detect_checkpoint_format, load_pretrained

__all__ = [
    "I3D",
    "InceptionI3d",
    "LoadReport",
    "convert_checkpoint",
    "convert_state_dict",
    "detect_checkpoint_format",
    "load_pretrained",
    "__version__",
]

__version__ = "0.1.0"
