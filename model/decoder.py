"""
Autoregressive Transformer Decoder for Raster2Seq.
"""
import torch
import torch.nn as nn
import math


class AutoregressiveDecoder(nn.Module):
    """
    Transformer decoder with causal self-attention and cross-attention
    to both query image features and support pose embeddings.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        image_feature_dim=2048,
        max_seq_length=1000
    ):
        """
        Args:
            hidden_dim: decoder hidden dimension
            num_layers: number of decoder layers
            num_heads: number of attention heads
            dropout: dropout rate
            image_feature_dim: dimension of image features from encoder
            max_seq_length: maximum sequence length for positional encoding
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project image features to hidden dimension
        self.image_proj = nn.Linear(image_feature_dim, hidden_dim)

        # Positional encoding for decoder tokens
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt,
        image_features,
        support_embeddings,
        tgt_mask=None,
        support_mask=None
    ):
        """
        Decode autoregressively.

        Args:
            tgt: [B, T, D] - target sequence embeddings
            image_features: [B, 2048, H, W] - query image features
            support_embeddings: [B, N_kpts, D] - support pose embeddings
            tgt_mask: [T, T] - causal mask for target sequence
            support_mask: [B, N_kpts] - mask for support keypoints

        Returns:
            output: [B, T, D] - decoder outputs
            layer_outputs: list of intermediate outputs for auxiliary losses
        """
        B, T, D = tgt.shape
        device = tgt.device

        # Flatten image features to sequence
        # [B, 2048, H, W] -> [B, H*W, 2048]
        _, C, H, W = image_features.shape
        image_seq = image_features.flatten(2).permute(0, 2, 1)  # [B, H*W, 2048]

        # Project to hidden dimension
        image_seq = self.image_proj(image_seq)  # [B, H*W, D]

        # Add positional encoding to target
        tgt = self.pos_encoding(tgt)

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(T).to(device)

        # Store intermediate outputs for auxiliary losses
        layer_outputs = []

        # Pass through decoder layers
        output = tgt
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory_image=image_seq,
                memory_support=support_embeddings,
                tgt_mask=tgt_mask,
                support_mask=support_mask
            )
            layer_outputs.append(output)

        # Final layer norm
        output = self.norm(output)

        return output, layer_outputs

    def _generate_causal_mask(self, size):
        """
        Generate causal mask for autoregressive decoding.

        Args:
            size: sequence length

        Returns:
            mask: [size, size] - causal mask (True = cannot attend)
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


class DecoderLayer(nn.Module):
    """
    Single decoder layer with:
    1. Causal self-attention
    2. Cross-attention to image features
    3. Cross-attention to support embeddings
    4. Feedforward network
    """

    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to image
        self.cross_attn_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Cross-attention to support
        self.cross_attn_support = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tgt,
        memory_image,
        memory_support,
        tgt_mask=None,
        support_mask=None
    ):
        """
        Args:
            tgt: [B, T, D] - target sequence
            memory_image: [B, H*W, D] - image features
            memory_support: [B, N_kpts, D] - support embeddings
            tgt_mask: [T, T] - causal mask
            support_mask: [B, N_kpts] - support keypoint mask

        Returns:
            output: [B, T, D]
        """
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)

        # Cross-attention to image features
        attn_out, _ = self.cross_attn_image(
            query=tgt,
            key=memory_image,
            value=memory_image,
            need_weights=False
        )
        tgt = tgt + self.dropout2(attn_out)
        tgt = self.norm2(tgt)

        # Cross-attention to support embeddings
        # Convert support_mask to key_padding_mask format
        # support_mask: [B, N] with True for valid keypoints
        # key_padding_mask: [B, N] with True for invalid keypoints
        if support_mask is not None:
            key_padding_mask = ~support_mask
        else:
            key_padding_mask = None

        attn_out, _ = self.cross_attn_support(
            query=tgt,
            key=memory_support,
            value=memory_support,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout3(attn_out)
        tgt = self.norm3(tgt)

        # Feedforward network
        ffn_out = self.ffn(tgt)
        tgt = tgt + ffn_out
        tgt = self.norm4(tgt)

        return tgt


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    """

    def __init__(self, hidden_dim, max_len=1000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding.

        Args:
            x: [B, T, D]

        Returns:
            x_with_pe: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
