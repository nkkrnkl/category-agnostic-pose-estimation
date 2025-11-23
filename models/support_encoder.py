"""
Support Pose Graph Encoder for CAPE

This module encodes support pose graphs (keypoint coordinates) into embeddings
that condition the query keypoint prediction.

FIXED: Now uses discrete tokenization with bilinear interpolation like Raster2Seq,
instead of continuous MLP embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphEdgeEncoder(nn.Module):
    """
    Graph Edge Encoder using message passing over skeleton structure.

    Much better than binary (connected/not connected) embedding.
    Encodes:
    - Edge connectivity
    - Relative positions between connected keypoints
    - Graph topology via multi-head attention
    - Aggregates neighbor information
    """

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Edge feature encoder
        # Encodes: (relative_x, relative_y, distance, is_connected)
        self.edge_feature_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph attention for message passing
        # Each keypoint aggregates information from its neighbors
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP for updating node features after message passing
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, coords, skeleton_edges, mask=None):
        """
        Args:
            node_features: (B, N, D) - Initial node embeddings (from coordinates)
            coords: (B, N, 2) - Keypoint coordinates
            skeleton_edges: List of edge lists, one per batch
            mask: (B, N) - Valid keypoint mask

        Returns:
            graph_features: (B, N, D) - Graph-aware node embeddings
        """
        B, N, D = node_features.shape
        device = coords.device

        # Build adjacency matrix and edge features
        adj_matrix, edge_features = self._build_edge_features(
            coords, skeleton_edges, N, device
        )  # adj: (B, N, N), edge_feat: (B, N, N, 4)

        # Encode edge features
        edge_emb = self.edge_feature_encoder(edge_features)  # (B, N, N, D)

        # Graph message passing
        # For each node, aggregate information from neighbors weighted by attention
        graph_features = []
        for b in range(B):
            # Get adjacency for this batch item
            adj_b = adj_matrix[b]  # (N, N)
            node_feat_b = node_features[b:b+1]  # (1, N, D)

            # Graph attention over all nodes, masked by adjacency
            # Only connected nodes (edges) can attend to each other
            attn_mask_b = ~adj_b.bool()  # (N, N) - True = don't attend

            # Multi-head attention
            attn_out, _ = self.graph_attention(
                query=node_feat_b,     # (1, N, D)
                key=node_feat_b,       # (1, N, D)
                value=node_feat_b,     # (1, N, D)
                attn_mask=attn_mask_b  # (N, N)
            )  # (1, N, D)

            graph_features.append(attn_out)

        graph_features = torch.cat(graph_features, dim=0)  # (B, N, D)

        # Update node features with graph information
        combined = torch.cat([node_features, graph_features], dim=-1)  # (B, N, 2D)
        updated = self.node_update(combined)  # (B, N, D)

        # Residual connection + norm
        output = self.norm(updated + node_features)

        return output

    def _build_edge_features(self, coords, skeleton_edges, N, device):
        """
        Build edge features encoding relative positions and connectivity.

        Returns:
            adj_matrix: (B, N, N) - Binary adjacency matrix
            edge_features: (B, N, N, 4) - Edge features for each pair
                [relative_x, relative_y, distance, is_connected]
        """
        B = len(skeleton_edges)

        # Initialize adjacency matrix with self-loops
        # Self-loops ensure nodes can attend to themselves even without skeleton edges
        adj_matrix = torch.zeros(B, N, N, dtype=torch.float32, device=device)
        for b in range(B):
            # Add self-loops (diagonal = 1)
            adj_matrix[b].fill_diagonal_(1.0)

        # Initialize edge features
        edge_features = torch.zeros(B, N, N, 4, device=device)

        for b in range(B):
            coords_b = coords[b]  # (N, 2)

            # Compute pairwise relative positions
            # relative[i, j] = coords[j] - coords[i]
            coords_i = coords_b.unsqueeze(1)  # (N, 1, 2)
            coords_j = coords_b.unsqueeze(0)  # (1, N, 2)
            relative = coords_j - coords_i     # (N, N, 2)

            # Compute distances
            distances = torch.norm(relative, dim=-1, keepdim=True)  # (N, N, 1)

            # Build edge features for ALL pairs (even non-connected)
            # This allows the model to learn from spatial structure
            edge_features[b, :, :, 0:2] = relative  # relative_x, relative_y
            edge_features[b, :, :, 2:3] = distances  # distance

            # Mark connected edges
            if skeleton_edges[b] is not None and len(skeleton_edges[b]) > 0:
                for edge in skeleton_edges[b]:
                    if len(edge) == 2:
                        src, dst = edge
                        # MP-100 uses 1-indexed, convert to 0-indexed
                        src_idx = src - 1 if src > 0 else src
                        dst_idx = dst - 1 if dst > 0 else dst

                        if 0 <= src_idx < N and 0 <= dst_idx < N:
                            adj_matrix[b, src_idx, dst_idx] = 1.0
                            adj_matrix[b, dst_idx, src_idx] = 1.0  # Undirected

                            # Mark as connected in edge features
                            edge_features[b, src_idx, dst_idx, 3] = 1.0
                            edge_features[b, dst_idx, src_idx, 3] = 1.0

        return adj_matrix, edge_features


class SupportPoseGraphEncoder(nn.Module):
    """
    Encodes a support pose graph into embeddings using discrete tokenization.

    Uses the SAME coordinate tokenization as Raster2Seq:
    - Discrete grid of embedding tables (vocab_size bins)
    - Bilinear interpolation for continuous coordinates
    - This ensures support graphs use the same representation as query outputs

    Input:
        - Support keypoints [(x1, y1), (x2, y2), ..., (xN, yN)] normalized to [0, 1]
        - Skeleton edges [[i, j], ...] defining connectivity
    Output: Support embeddings [e1, e2, ..., eN] ∈ R^(N×D)
    """

    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=3,
                 dim_feedforward=1024, dropout=0.1, max_keypoints=50,
                 vocab_size=2000, coord_vocab_embed=None):
        """
        Args:
            hidden_dim: Embedding dimension (D)
            nheads: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_keypoints: Maximum number of keypoints (for adjacency matrix)
            vocab_size: Size of coordinate vocabulary (should match Raster2Seq)
            coord_vocab_embed: Optional shared coordinate embedding table from base model.
                              If None, creates its own table (but should be shared!)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.max_keypoints = max_keypoints
        self.vocab_size = vocab_size

        # Calculate number of bins for x and y coordinates
        # vocab_size = num_bins_x * num_bins_y
        # For simplicity, use square grid: num_bins = sqrt(vocab_size / 2)
        # Divided by 2 because we have separate bins for x and y
        self.num_bins = int((vocab_size / 2) ** 0.5)

        # CRITICAL: Coordinate embedding table for discrete tokenization
        # This should be SHARED with the base model's coordinate vocabulary
        if coord_vocab_embed is not None:
            # Use shared embedding table from base model
            self.coord_vocab_embed = coord_vocab_embed
        else:
            # Create own embedding table (H_b x W_b x D) as per Raster2Seq paper
            # But flatten it for easier indexing: (num_bins * num_bins, D)
            self.coord_vocab_embed = nn.Parameter(
                torch.randn(self.num_bins * self.num_bins, hidden_dim) * 0.02
            )

        # Graph Structure Encoder for skeleton edges
        # Instead of binary embedding, use proper graph message passing
        self.edge_encoder = GraphEdgeEncoder(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )

        # Projection to combine coordinate and edge information
        self.coord_edge_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional encoding for keypoint ordering
        self.pos_embedding = PositionalEncoding1D(hidden_dim, dropout=dropout)

        # Transformer encoder
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

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for name, p in self.named_parameters():
            if 'coord_vocab_embed' in name:
                # Special init for coordinate vocabulary (as per Raster2Seq)
                nn.init.normal_(p, mean=0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _bilinear_coord_embedding(self, coords):
        """
        Convert continuous coordinates to embeddings using bilinear interpolation
        over the learned coordinate vocabulary table.

        This matches the Raster2Seq approach (deformable_transformer_v2.py:947-951)

        Args:
            coords: (B, N, 2) normalized coordinates in [0, 1]

        Returns:
            coord_emb: (B, N, D) coordinate embeddings
        """
        B, N, _ = coords.shape
        device = coords.device

        # Scale coordinates to bin indices [0, num_bins-1]
        # coords are in [0, 1], scale to [0, num_bins-1]
        coords_scaled = coords * (self.num_bins - 1)

        # Separate x and y
        x = coords_scaled[:, :, 0]  # (B, N)
        y = coords_scaled[:, :, 1]  # (B, N)

        # Get surrounding grid points
        x1 = torch.floor(x).long().clamp(0, self.num_bins - 1)  # (B, N)
        y1 = torch.floor(y).long().clamp(0, self.num_bins - 1)  # (B, N)
        x2 = torch.ceil(x).long().clamp(0, self.num_bins - 1)   # (B, N)
        y2 = torch.ceil(y).long().clamp(0, self.num_bins - 1)   # (B, N)

        # Calculate interpolation weights (distances from grid points)
        # delta_x1 = x - x1, delta_x2 = x2 - x (but normalized to [0, 1])
        delta_x1 = (x - x1.float()).clamp(0, 1)  # (B, N)
        delta_y1 = (y - y1.float()).clamp(0, 1)  # (B, N)
        delta_x2 = 1.0 - delta_x1  # (B, N)
        delta_y2 = 1.0 - delta_y1  # (B, N)

        # Get flat indices for the 4 surrounding grid points
        # vocab_embed is (num_bins * num_bins, D)
        # index = y * num_bins + x
        idx11 = y1 * self.num_bins + x1  # (B, N) - top-left
        idx21 = y1 * self.num_bins + x2  # (B, N) - top-right
        idx12 = y2 * self.num_bins + x1  # (B, N) - bottom-left
        idx22 = y2 * self.num_bins + x2  # (B, N) - bottom-right

        # Clamp indices to valid range
        max_idx = self.num_bins * self.num_bins - 1
        idx11 = idx11.clamp(0, max_idx)
        idx21 = idx21.clamp(0, max_idx)
        idx12 = idx12.clamp(0, max_idx)
        idx22 = idx22.clamp(0, max_idx)

        # Get embeddings for the 4 surrounding points
        if isinstance(self.coord_vocab_embed, nn.Parameter):
            # Own embedding table
            e11 = self.coord_vocab_embed[idx11]  # (B, N, D)
            e21 = self.coord_vocab_embed[idx21]  # (B, N, D)
            e12 = self.coord_vocab_embed[idx12]  # (B, N, D)
            e22 = self.coord_vocab_embed[idx22]  # (B, N, D)
        else:
            # Shared embedding table (may be 2D or 3D)
            # Flatten batch and keypoints for indexing
            idx11_flat = idx11.view(-1)  # (B*N,)
            idx21_flat = idx21.view(-1)
            idx12_flat = idx12.view(-1)
            idx22_flat = idx22.view(-1)

            # Index and reshape
            e11 = self.coord_vocab_embed[idx11_flat].view(B, N, -1)  # (B, N, D)
            e21 = self.coord_vocab_embed[idx21_flat].view(B, N, -1)
            e12 = self.coord_vocab_embed[idx12_flat].view(B, N, -1)
            e22 = self.coord_vocab_embed[idx22_flat].view(B, N, -1)

        # Bilinear interpolation
        # out = e11 * (1-dx) * (1-dy) + e21 * dx * (1-dy) + e12 * (1-dx) * dy + e22 * dx * dy
        # which is: e11 * delta_x2 * delta_y2 + e21 * delta_x1 * delta_y2 + ...
        coord_emb = (
            e11 * delta_x2.unsqueeze(-1) * delta_y2.unsqueeze(-1) +
            e21 * delta_x1.unsqueeze(-1) * delta_y2.unsqueeze(-1) +
            e12 * delta_x2.unsqueeze(-1) * delta_y1.unsqueeze(-1) +
            e22 * delta_x1.unsqueeze(-1) * delta_y1.unsqueeze(-1)
        )  # (B, N, D)

        return coord_emb

    def forward(self, support_coords, support_mask=None, skeleton_edges=None):
        """
        Args:
            support_coords: Tensor of shape (B, N, 2) where:
                - B = batch size
                - N = number of keypoints
                - 2 = (x, y) coordinates
                Coordinates should be normalized to [0, 1]

            support_mask: Optional mask of shape (B, N) indicating valid keypoints
                True = valid, False = padding

            skeleton_edges: Optional list of edge lists, one per batch item
                Each edge list contains [src, dst] pairs (0-indexed)
                Example: [[[0, 1], [1, 2], ...], [[0, 1], [1, 2], ...]]
                If None, no skeletal structure is used

        Returns:
            support_embeddings: Tensor of shape (B, N, D) containing
                contextual embeddings for each support keypoint
        """
        B, N, _ = support_coords.shape
        device = support_coords.device

        # 1. Embed coordinates using DISCRETE TOKENIZATION with bilinear interpolation
        # This is the KEY FIX - matching Raster2Seq's coordinate representation
        coord_emb = self._bilinear_coord_embedding(support_coords)  # (B, N, D)

        # 2. Add skeleton edge information via graph message passing
        if skeleton_edges is not None and len(skeleton_edges) > 0:
            # Use graph edge encoder to get structural embeddings
            # This performs message passing over the skeleton graph
            edge_info = self.edge_encoder(
                node_features=coord_emb,
                coords=support_coords,
                skeleton_edges=skeleton_edges,
                mask=support_mask
            )  # (B, N, D)

            # Combine coordinate and graph structure embeddings
            combined = torch.cat([coord_emb, edge_info], dim=-1)  # (B, N, 2D)
            embeddings = self.coord_edge_proj(combined)  # (B, N, D)
        else:
            # No skeleton information - use only coordinates
            embeddings = coord_emb

        # 3. Add positional encoding for keypoint ordering
        embeddings = self.pos_embedding(embeddings)

        # 4. Create attention mask if provided
        if support_mask is not None:
            attn_mask = ~support_mask.bool()
        else:
            attn_mask = None

        # 5. Process through transformer encoder
        support_features = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=attn_mask
        )

        # 6. Normalize
        support_features = self.norm(support_features)

        return support_features



class PositionalEncoding1D(nn.Module):
    """1D Positional Encoding for keypoint sequences"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
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
    """

    def __init__(self, hidden_dim=256, method='attention'):
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
                masked_embeddings = support_embeddings * mask_expanded + (1 - mask_expanded) * (-1e9)
                aggregated = masked_embeddings.max(dim=1)[0]
            else:
                aggregated = support_embeddings.max(dim=1)[0]

        elif self.method == 'attention':
            attn_weights = self.attention_weights(support_embeddings)

            if support_mask is not None:
                mask_expanded = support_mask.unsqueeze(-1).float()
                attn_weights = attn_weights.masked_fill(~mask_expanded.bool(), -1e9)

            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (support_embeddings * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        return aggregated


def build_support_encoder(args, coord_vocab_embed=None):
    """
    Build support pose graph encoder from arguments

    Args:
        args: Arguments
        coord_vocab_embed: Coordinate embedding table from base model (SHOULD BE SHARED)
    """
    return SupportPoseGraphEncoder(
        hidden_dim=getattr(args, 'hidden_dim', 256),
        nheads=getattr(args, 'nheads', 8),
        num_encoder_layers=getattr(args, 'support_encoder_layers', 3),
        dim_feedforward=getattr(args, 'dim_feedforward', 1024),
        dropout=getattr(args, 'dropout', 0.1),
        vocab_size=getattr(args, 'vocab_size', 2000),
        coord_vocab_embed=coord_vocab_embed  # CRITICAL: share with base model
    )
