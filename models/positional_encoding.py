"""
Positional encoding modules for CAPE.
"""
import math
import torch
import torch.nn as nn
class PositionalEncoding1D(nn.Module):
    """
    Standard 1D positional encoding for transformer sequences.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, d_model]
        
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class SinePositionalEncoding2D(nn.Module):
    """
    Sinusoidal positional encoding for 2D coordinates.
    
    Args:
        num_feats: Feature dimension for each axis
        temperature: Temperature for scaling
        normalize: Whether coordinates are normalized
        scale: Scale factor when normalize=True
    """
    def __init__(self, num_feats, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
    def forward_coordinates(self, coord):
        """
        Encode coordinates with sine/cosine functions.
        
        Args:
            coord: Coordinates, shape [bs, num_pts, 2]
        
        Returns:
            pos: Positional embeddings, shape [bs, num_pts, num_feats*2]
        """
        x_embed, y_embed = coord[:, :, 0], coord[:, :, 1]
        x_embed = x_embed * self.scale
        y_embed = y_embed * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=coord.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        bs, kpt, _ = pos_x.shape
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
            dim=3
        ).view(bs, kpt, -1)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
            dim=3
        ).view(bs, kpt, -1)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos
    def __repr__(self):
        """
        String representation of the module.
        
        Returns:
            String representation
        """
        return (f'{self.__class__.__name__}('
                f'num_feats={self.num_feats}, '
                f'temperature={self.temperature}, '
                f'normalize={self.normalize}, '
                f'scale={self.scale})')