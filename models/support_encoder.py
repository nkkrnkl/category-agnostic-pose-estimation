"""
Support Pose Graph Encoder for CAPE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SupportPoseGraphEncoder(nn.Module):
    """
    Encodes a support pose graph into embeddings using a Transformer encoder.
    """
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, max_keypoints=50):
        """
        Initialize support pose graph encoder.
        
        Args:
            hidden_dim: Embedding dimension
            nheads: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_keypoints: Maximum number of keypoints
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.max_keypoints = max_keypoints
        self.coord_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=hidden_dim
        )
        self.coord_edge_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pos_embedding = PositionalEncoding1D(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()
    def _reset_parameters(self):
        """
        Initialize parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, support_coords, support_mask=None, skeleton_edges=None):
        """
        Forward pass.
        
        Args:
            support_coords: Support coordinates, shape (B, N, 2)
            support_mask: Optional mask, shape (B, N)
            skeleton_edges: Optional list of edge lists
        
        Returns:
            support_embeddings: Support embeddings, shape (B, N, D)
        """
        B, N, _ = support_coords.shape
        device = support_coords.device
        coord_emb = self.coord_embedding(support_coords)
        if skeleton_edges is not None and len(skeleton_edges) > 0:
            adj_matrix = self._build_adjacency_matrix(skeleton_edges, N, device)
            edge_info = self._aggregate_edge_embeddings(adj_matrix, N, device)
            combined = torch.cat([coord_emb, edge_info], dim=-1)
            embeddings = self.coord_edge_proj(combined)
        else:
            embeddings = coord_emb
        embeddings = self.pos_embedding(embeddings)
        if support_mask is not None:
            attn_mask = ~support_mask.bool()
        else:
            attn_mask = None
        support_features = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=attn_mask
        )
        support_features = self.norm(support_features)
        return support_features
    def _build_adjacency_matrix(self, skeleton_edges, N, device):
        """
        Build adjacency matrix from skeleton edges.
        Args:
            skeleton_edges: List of edge lists, one per batch item
            N: Number of keypoints
            device: Device for tensor
        Returns:
            adj_matrix: (B, N, N) binary adjacency matrix
        """
        B = len(skeleton_edges)
        adj_matrix = torch.zeros(B, N, N, dtype=torch.long, device=device)
        for b, edges in enumerate(skeleton_edges):
            if edges is not None and len(edges) > 0:
                for edge in edges:
                    if len(edge) == 2:
                        src, dst = edge
                        src_idx = src - 1 if src > 0 else src
                        dst_idx = dst - 1 if dst > 0 else dst
                        if 0 <= src_idx < N and 0 <= dst_idx < N:
                            adj_matrix[b, src_idx, dst_idx] = 1
                            adj_matrix[b, dst_idx, src_idx] = 1
        return adj_matrix
    def _aggregate_edge_embeddings(self, adj_matrix, N, device):
        """
        Aggregate edge embeddings for each keypoint based on adjacency.
        Args:
            adj_matrix: (B, N, N) adjacency matrix
            N: Number of keypoints
            device: Device
        Returns:
            edge_info: (B, N, D) edge information for each keypoint
        """
        B = adj_matrix.shape[0]
        degree = adj_matrix.sum(dim=2)
        has_connections = (degree > 0).long()
        edge_emb = self.edge_embedding(has_connections)
        degree_scale = degree.float().unsqueeze(-1).clamp(min=1.0) / 10.0
        edge_info = edge_emb * degree_scale
        return edge_info
class PositionalEncoding1D(nn.Module):
    """
    1D Positional Encoding for keypoint sequences
    Uses sinusoidal positional encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, D)
        Returns:
            x + positional encoding: (B, N, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class SupportGraphAggregator(nn.Module):
    """
    Aggregates support graph embeddings into a global pose structure representation.
    Used when we want a single vector representing the entire support pose graph,
    in addition to per-keypoint embeddings.
    """
    def __init__(self, hidden_dim=256, method='attention'):
        """
        Args:
            hidden_dim: Embedding dimension
            method: Aggregation method ('mean', 'max', 'attention')
        """
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        if method == 'attention':
            self.attention_weights = nn.Linear(hidden_dim, 1)
    def forward(self, support_embeddings, support_mask=None):
        """
        Args:
            support_embeddings: (B, N, D) per-keypoint embeddings
            support_mask: (B, N) validity mask (True = valid)
        Returns:
            aggregated: (B, D) global pose structure representation
        """
        if self.method == 'mean':
            if support_mask is not None:
                mask_expanded = support_mask.unsqueeze(-1).float()
                sum_embeddings = (support_embeddings * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1.0)
                aggregated = sum_embeddings / count
            else:
                aggregated = support_embeddings.mean(dim=1)
        elif self.method == 'max':
            if support_mask is not None:
                mask_expanded = support_mask.unsqueeze(-1).float()
                masked_embeddings = support_embeddings * mask_expanded + (1 - mask_expanded) * (-1e4)
                aggregated = masked_embeddings.max(dim=1)[0]
            else:
                aggregated = support_embeddings.max(dim=1)[0]
        elif self.method == 'attention':
            attn_weights = self.attention_weights(support_embeddings)
            if support_mask is not None:
                mask_expanded = support_mask.unsqueeze(-1).float()
                attn_weights = attn_weights.masked_fill(~mask_expanded.bool(), -1e4)
            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (support_embeddings * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
        return aggregated
def build_support_encoder(args):
    """
    Build support pose graph encoder from arguments
    Args:
        args: Arguments containing:
            - hidden_dim: Embedding dimension (default: 256)
            - support_encoder_layers: Number of encoder layers (default: 3)
            - nheads: Number of attention heads (default: 8)
            - dim_feedforward: Feedforward dimension (default: 1024)
            - dropout: Dropout rate (default: 0.1)
    Returns:
        encoder: SupportPoseGraphEncoder module
    """
    return SupportPoseGraphEncoder(
        hidden_dim=getattr(args, 'hidden_dim', 256),
        nheads=getattr(args, 'nheads', 8),
        num_encoder_layers=getattr(args, 'support_encoder_layers', 3),
        dim_feedforward=getattr(args, 'dim_feedforward', 1024),
        dropout=getattr(args, 'dropout', 0.1)
    )
if __name__ == '__main__':
    print("Testing SupportPoseGraphEncoder...")
    encoder = SupportPoseGraphEncoder(hidden_dim=256, num_encoder_layers=3)
    batch_size = 4
    num_keypoints = 20
    support_coords = torch.rand(batch_size, num_keypoints, 2)
    support_mask = torch.ones(batch_size, num_keypoints, dtype=torch.bool)
    support_mask[:, 15:] = False
    support_embeddings = encoder(support_coords, support_mask)
    print(f"Input shape: {support_coords.shape}")
    print(f"Output shape: {support_embeddings.shape}")
    print(f"Expected: ({batch_size}, {num_keypoints}, 256)")
    assert support_embeddings.shape == (batch_size, num_keypoints, 256)
    print("\nTesting SupportGraphAggregator...")
    aggregator = SupportGraphAggregator(hidden_dim=256, method='attention')
    global_embedding = aggregator(support_embeddings, support_mask)
    print(f"Aggregated shape: {global_embedding.shape}")
    print(f"Expected: ({batch_size}, 256)")
    assert global_embedding.shape == (batch_size, 256)
    print("\n✓ All tests passed!")