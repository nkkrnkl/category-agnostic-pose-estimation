"""
Geometry-only support encoder for CAPE.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from models.graph_utils import adj_from_skeleton, GCNLayer
from models.positional_encoding import SinePositionalEncoding2D, PositionalEncoding1D
class GeometricSupportEncoder(nn.Module):
    """
    Geometry-only support pose graph encoder.
    
    Args:
        hidden_dim: Feature dimension
        num_encoder_layers: Number of transformer encoder layers
        nhead: Number of attention heads
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        use_gcn_preenc: Whether to use GCN pre-encoding
        num_gcn_layers: Number of GCN layers if use_gcn_preenc=True
        activation: Activation function ('relu' or 'gelu')
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
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_encoding = SinePositionalEncoding2D(
            num_feats=hidden_dim // 2,
            temperature=10000,
            normalize=True,
            scale=2 * 3.14159265359
        )
        self.sequence_pos_encoding = PositionalEncoding1D(
            d_model=hidden_dim,
            max_len=100,
            dropout=0.0
        )
        if use_gcn_preenc:
            self.gcn_layers = nn.ModuleList([
                GCNLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    kernel_size=2,
                    use_bias=True,
                    activation=nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
                    batch_first=True
                )
                for _ in range(num_gcn_layers)
            ])
        else:
            self.gcn_layers = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu' if activation == 'relu' else 'gelu',
            batch_first=True
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
        
        Args:
            support_coords: Keypoint coordinates, shape [bs, num_pts, 2]
            support_mask: Boolean mask, shape [bs, num_pts] (True = invalid)
            skeleton_edges: List of edge lists, one per batch element
        
        Returns:
            support_features: Encoded features, shape [bs, num_pts, hidden_dim]
        """
        bs, num_pts, _ = support_coords.shape
        device = support_coords.device
        coord_emb = self.coord_mlp(support_coords)
        pos_emb = self.pos_encoding.forward_coordinates(support_coords)
        embeddings = coord_emb + pos_emb
        embeddings = self.sequence_pos_encoding(embeddings)
        if self.use_gcn_preenc and self.gcn_layers is not None:
            adj = adj_from_skeleton(num_pts, skeleton_edges, support_mask, device)
            for gcn_layer in self.gcn_layers:
                embeddings = gcn_layer(embeddings, adj)
        all_masked_per_batch = support_mask.all(dim=1)
        if all_masked_per_batch.any():
            temp_mask = support_mask.clone()
            for b in range(support_mask.shape[0]):
                if all_masked_per_batch[b]:
                    temp_mask[b, 0] = False
            support_features = self.transformer_encoder(
                embeddings,
                src_key_padding_mask=temp_mask
            )
            support_features[all_masked_per_batch] = 0.0
        else:
            support_features = self.transformer_encoder(
                embeddings,
                src_key_padding_mask=support_mask
            )
        return support_features
    def __repr__(self):
        """String representation of the module."""
        gcn_str = f", use_gcn_preenc=True ({len(self.gcn_layers)} layers)" if self.use_gcn_preenc else ""
        return (f'{self.__class__.__name__}('
                f'hidden_dim={self.hidden_dim}, '
                f'spatial_pe=SinePE2D, '
                f'sequence_pe=SinePE1D'
                f'{gcn_str})')