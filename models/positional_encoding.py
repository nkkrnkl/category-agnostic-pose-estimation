"""
Positional encoding modules for CAPE.

This module includes:
1. PositionalEncoding1D: Simple 1D positional encoding for sequences
2. SinePositionalEncoding2D: Sinusoidal encoding for 2D coordinates (from CapeX)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    """
    Standard 1D positional encoding for transformer sequences.
    
    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 5000)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinePositionalEncoding2D(nn.Module):
    """
    Sinusoidal positional encoding for 2D coordinates.
    
    Copied from CapeX positional_encoding.py:97-123
    
    This encoding applies sine and cosine functions to normalized (x,y) coordinates,
    providing translation and scale-invariant positional information.
    
    Args:
        num_feats (int): Feature dimension for each axis (total output dim = num_feats * 2)
        temperature (int): Temperature for scaling (default: 10000)
        normalize (bool): Whether coordinates are normalized (default: True)
        scale (float): Scale factor when normalize=True (default: 2*pi)
    
    Example:
        >>> pos_enc = SinePositionalEncoding2D(num_feats=128)
        >>> coords = torch.rand(2, 10, 2)  # [bs, num_pts, 2] in [0, 1]
        >>> pos = pos_enc.forward_coordinates(coords)
        >>> pos.shape
        torch.Size([2, 10, 256])  # 128 * 2
    """
    
    def __init__(self, num_feats, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
    
    def forward_coordinates(self, coord):
        """
        Encode (x,y) coordinates with sine/cosine functions.
        
        Args:
            coord (torch.Tensor): [bs, num_pts, 2] coordinates in [0, 1]
        
        Returns:
            pos (torch.Tensor): [bs, num_pts, num_feats*2] positional embeddings
        
        Notes:
            - x and y coordinates are encoded independently
            - Each coordinate is scaled by self.scale (default: 2*pi)
            - Sine/cosine applied at different frequencies
            - Final output concatenates [pos_y, pos_x]
        """
        x_embed, y_embed = coord[:, :, 0], coord[:, :, 1]  # [bs, num_pts]
        x_embed = x_embed * self.scale
        y_embed = y_embed * self.scale
        
        # Create frequency terms
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=coord.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        # Divide coordinates by frequency terms
        pos_x = x_embed[:, :, None] / dim_t  # [bs, num_pts, num_feats]
        pos_y = y_embed[:, :, None] / dim_t  # [bs, num_pts, num_feats]
        bs, kpt, _ = pos_x.shape
        
        # Apply sine to even indices, cosine to odd indices
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
            dim=3
        ).view(bs, kpt, -1)  # [bs, num_pts, num_feats]
        
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
            dim=3
        ).view(bs, kpt, -1)  # [bs, num_pts, num_feats]
        
        # Concatenate y and x embeddings
        pos = torch.cat((pos_y, pos_x), dim=2)  # [bs, num_pts, num_feats * 2]
        
        return pos
    
    def __repr__(self):
        """String representation of the module."""
        return (f'{self.__class__.__name__}('
                f'num_feats={self.num_feats}, '
                f'temperature={self.temperature}, '
                f'normalize={self.normalize}, '
                f'scale={self.scale})')

