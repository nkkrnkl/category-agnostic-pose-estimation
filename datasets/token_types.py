"""
Token types for Raster2Seq sequence generation.
"""
from enum import Enum
class TokenType(Enum):
    """
    Token types for sequence generation.
    """
    coord = 0
    sep = 1
    eos = 2
    cls = 3