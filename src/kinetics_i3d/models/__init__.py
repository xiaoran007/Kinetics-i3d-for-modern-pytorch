"""Model definitions for modernized I3D."""

from .inception_i3d import InceptionI3d
from .legacy import I3D

__all__ = ["I3D", "InceptionI3d"]
