"""
Geometry-only support encoder for CAPE.

This module implements a support pose graph encoder that uses ONLY coordinates
and skeleton edges (no text), combining:
1. Coordinate MLP embedding (replaces CapeX's text encoder)
2. Sinusoidal positional encoding (from CapeX)
3. Optional GCN pre-encoding (from CapeX)
4. Transformer self-attention

Inspired by CapeX but adapted for geometry-only operation.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from models.graph_utils import adj_from_skeleton, GCNLayer
from models.positional_encoding import SinePositionalEncoding2D, PositionalEncoding1D


class GeometricSupportEncoder(nn.Module):
    """
    Geometry-only support pose graph encoder.
    
    Encodes support keypoints using only coordinates and skeleton structure,
    without any textual descriptions. Combines:
    1. Coordinate embeddings (what are the coordinates?)
    2. 2D spatial positional encoding (where in image space?)
    3. 1D sequence positional encoding (which keypoint in ordering?)
    4. Optional graph convolution (structural relationships)
    5. Transformer self-attention (contextual understanding)
    
    This design follows the PhD student's recommendation that the transformer
    must understand both the spatial position AND the sequential order of keypoints.
    
    Args:
        hidden_dim (int): Feature dimension (default: 256)
        num_encoder_layers (int): Number of transformer encoder layers (default: 3)
        nhead (int): Number of attention heads (default: 8)
        dim_feedforward (int): Feedforward dimension (default: 1024)
        dropout (float): Dropout probability (default: 0.1)
        use_gcn_preenc (bool): Whether to use GCN pre-encoding (default: False)
        num_gcn_layers (int): Number of GCN layers if use_gcn_preenc=True (default: 2)
        activation (str): Activation function ('relu' or 'gelu', default: 'relu')
    
    Input Shapes:
        - support_coords: [bs, num_pts, 2] normalized to [0, 1]
        - support_mask: [bs, num_pts] (True = invalid/invisible keypoint)
        - skeleton_edges: List of length bs, each element is list of [i, j] edge pairs
    
    Output Shape:
        - support_features: [bs, num_pts, hidden_dim]
    
    Example:
        >>> encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
        >>> coords = torch.rand(2, 10, 2)  # [bs, num_pts, 2]
        >>> mask = torch.zeros(2, 10).bool()
        >>> skeleton = [[[0,1], [1,2], [2,3]] for _ in range(2)]
        >>> out = encoder(coords, mask, skeleton)
        >>> out.shape
        torch.Size([2, 10, 256])
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_encoder_layers: int = 3,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 use_gcn_preenc: bool = False,
                 num_gcn_layers: int = 2,
                 activation: str = 'relu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gcn_preenc = use_gcn_preenc
        
        # 1. Coordinate embedding MLP (replaces CapeX's text encoder)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Sinusoidal positional encoding (from CapeX)
        self.pos_encoding = SinePositionalEncoding2D(
            num_feats=hidden_dim // 2,  # Output will be hidden_dim (num_feats * 2)
            temperature=10000,
            normalize=True,
            scale=2 * 3.14159265359  # 2*pi
        )
        
        # 3. 1D sequence positional encoding for keypoint ordering
        # This encodes WHICH keypoint it is (0th, 1st, 2nd, ...) in the sequence,
        # complementing the 2D spatial encoding which encodes WHERE it is (x, y).
        # Following PhD recommendation: "positional encoding for the keypoint sequence"
        self.sequence_pos_encoding = PositionalEncoding1D(
            d_model=hidden_dim,
            max_len=100,  # Max keypoints (MP-100 has ~17 max, 100 provides headroom)
            dropout=0.0   # Don't dropout positional encodings (deterministic)
        )
        
        # 4. Optional GCN pre-encoding layers (from CapeX)
        if use_gcn_preenc:
            self.gcn_layers = nn.ModuleList([
                GCNLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    kernel_size=2,  # Dual-channel: self + neighbors
                    use_bias=True,
                    activation=nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
                    batch_first=True
                )
                for _ in range(num_gcn_layers)
            ])
        else:
            self.gcn_layers = None
        
        # 5. Transformer encoder for self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu' if activation == 'relu' else 'gelu',
            batch_first=True  # Input: [bs, num_pts, hidden_dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self,
                support_coords: torch.Tensor,
                support_mask: torch.Tensor,
                skeleton_edges: List[List[List[int]]]) -> torch.Tensor:
        """
        Encode support pose graph using only geometry.
        
        Pipeline:
        1. Coordinate embedding: MLP(coords) → [bs, N, hidden_dim]
        2. 2D spatial PE: SinePosEnc2D(coords) → [bs, N, hidden_dim]
        3. Combine: coord_emb + spatial_pe
        4. 1D sequence PE: SinePosEnc1D(indices) → add to embeddings
        5. Optional GCN: GCN(embeddings, adjacency)
        6. Transformer: Self-attention with masking
        
        Args:
            support_coords (torch.Tensor): [bs, num_pts, 2] keypoint coordinates in [0, 1]
            support_mask (torch.Tensor): [bs, num_pts] boolean mask (True = invalid)
            skeleton_edges (list): List of edge lists (one per batch element)
                                  Each element is list of [i, j] pairs (0-indexed)
        
        Returns:
            support_features (torch.Tensor): [bs, num_pts, hidden_dim] encoded features
        
        Notes:
            - Coordinates should be normalized to [0, 1] range
            - Skeleton edges should be 0-indexed
            - Masked keypoints will still produce embeddings but won't attend to others
        """
        bs, num_pts, _ = support_coords.shape
        device = support_coords.device
        
        # 1. Coordinate embedding (replaces text encoder)
        coord_emb = self.coord_mlp(support_coords)  # [bs, num_pts, hidden_dim]
        
        # 2. Positional encoding (from CapeX)
        pos_emb = self.pos_encoding.forward_coordinates(support_coords)  # [bs, num_pts, hidden_dim]
        
        # 3. Combine coordinate and spatial positional information
        embeddings = coord_emb + pos_emb  # [bs, num_pts, hidden_dim]
        
        # 4. Add 1D sequence positional encoding
        # This tells the transformer WHICH keypoint it is (0th, 1st, 2nd, ...)
        # complementing the spatial PE which tells it WHERE it is (x, y).
        # Without this, shuffling keypoint order produces identical embeddings.
        embeddings = self.sequence_pos_encoding(embeddings)  # [bs, num_pts, hidden_dim]
        
        # Interpret support_mask coming from dataloader:
        #   - Current convention (episodic_sampler): True = VALID keypoint (visibility > 0)
        #   - PyTorch Transformer expects src_key_padding_mask: True = PAD (to ignore)
        # Convert once here so the rest of this module can use the correct semantics.
        valid_mask = support_mask            # True = valid keypoint
        pad_mask = ~valid_mask               # True = padding / invalid (for Transformer & GCN)
        
        # 5. Optional GCN pre-encoding (from CapeX)
        if self.use_gcn_preenc and self.gcn_layers is not None:
            # Build adjacency matrix from skeleton
            # adj_from_skeleton expects mask=True for INVALID / padded positions
            adj = adj_from_skeleton(num_pts, skeleton_edges, pad_mask, device)
            # adj: [bs, 2, num_pts, num_pts]
            
            # Apply GCN layers sequentially
            for gcn_layer in self.gcn_layers:
                embeddings = gcn_layer(embeddings, adj)  # [bs, num_pts, hidden_dim]
        
        # 6. Transformer self-attention
        # pad_mask: True = positions to ignore (mask out)
        # PyTorch convention: True = ignore

        # DEBUG: Inspect effective keypoints per sample to catch empty supports
        try:
            unmasked_counts = valid_mask.sum(dim=1)
            print("[DEBUG GeometricSupportEncoder] support_coords shape:", tuple(support_coords.shape))
            print("[DEBUG GeometricSupportEncoder] valid_mask shape:", tuple(valid_mask.shape))
            print("[DEBUG GeometricSupportEncoder] unmasked (valid) keypoints per sample:",
                  unmasked_counts.tolist())
        except Exception:
            # Avoid breaking training if debug logging fails for any reason
            pass

        support_features = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=pad_mask
        )  # [bs, num_pts, hidden_dim]
        
        return support_features
    
    def __repr__(self):
        """String representation of the module."""
        gcn_str = f", use_gcn_preenc=True ({len(self.gcn_layers)} layers)" if self.use_gcn_preenc else ""
        return (f'{self.__class__.__name__}('
                f'hidden_dim={self.hidden_dim}, '
                f'spatial_pe=SinePE2D, '
                f'sequence_pe=SinePE1D'
                f'{gcn_str})')

