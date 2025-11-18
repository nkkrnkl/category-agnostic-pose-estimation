"""
Token types for Raster2Seq sequence generation.

Used for both polygon data (legacy) and keypoint pose data (CAPE).
"""

from enum import Enum


class TokenType(Enum):
    """Token types for sequence generation."""
    coord = 0  # Coordinate token (precedes x, y values)
    sep = 1    # Separator token (between polygons/keypoints)
    eos = 2    # End of sequence token
    cls = 3    # Class/semantic token
