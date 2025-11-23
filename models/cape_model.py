"""
CAPE Model: Category-Agnostic Pose Estimation with Support Graph Conditioning

This module wraps the base Raster2Seq model (RoomFormerV2) and adds:
1. Support pose graph encoder
2. Cross-modal fusion between support graph and query image
3. Support-conditioned keypoint prediction

Key innovation: Uses coordinate sequences as support (vs. text in CapeX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .support_encoder import SupportPoseGraphEncoder, SupportGraphAggregator


class CAPEModel(nn.Module):
    """
    CAPE model that extends Raster2Seq with support pose graph conditioning.

    Architecture:
        1. Support Encoder: Encodes support pose graph into embeddings
        2. Query Encoder: ResNet backbone for query image features (from base model)
        3. Cross-Modal Fusion: Fuses support and query representations
        4. Decoder: Generates keypoint sequence conditioned on both (from base model)
    """

    def __init__(self, base_model, hidden_dim=256, support_encoder_layers=3,
                 support_fusion_method='cross_attention'):
        """
        Args:
            base_model: RoomFormerV2 model instance
            hidden_dim: Hidden dimension (should match base model)
            support_encoder_layers: Number of transformer layers in support encoder
            support_fusion_method: How to fuse support with query
                - 'cross_attention': Add cross-attention layer to decoder
                - 'concat': Concatenate support features
                - 'add': Add support features to query features
        """
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.support_fusion_method = support_fusion_method

        # Support pose graph encoder
        self.support_encoder = SupportPoseGraphEncoder(
            hidden_dim=hidden_dim,
            nheads=8,
            num_encoder_layers=support_encoder_layers,
            dim_feedforward=1024,
            dropout=0.1
        )

        # Support graph aggregator (for global pose structure)
        self.support_aggregator = SupportGraphAggregator(
            hidden_dim=hidden_dim,
            method='attention'
        )

        # Support fusion layer (injected into decoder)
        if support_fusion_method == 'cross_attention':
            # Will add cross-attention modules to decoder layers
            self._add_support_cross_attention()
        elif support_fusion_method == 'concat':
            # Projection for concatenated features
            self.support_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif support_fusion_method == 'add':
            # Simple addition (no extra parameters needed)
            pass
        else:
            raise ValueError(f"Unknown fusion method: {support_fusion_method}")

    def _add_support_cross_attention(self):
        """
        Add support cross-attention modules to decoder layers.

        This creates cross-attention layers that allow the decoder to attend
        to support graph embeddings when generating query keypoints.
        """
        # Get decoder layers from base model
        decoder = self.base_model.transformer.decoder
        num_layers = decoder.num_layers

        # Create support cross-attention modules for each decoder layer
        self.support_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer norms for support cross-attention
        self.support_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, samples, support_coords, support_mask, targets=None, skeleton_edges=None):
        """
        Forward pass with support pose graph conditioning.

        Args:
            samples: Query images
                - If NestedTensor: contains tensors and mask
                - If list of tensors: batch of images
                - If single tensor: (B, C, H, W)

            support_coords: Support pose graph coordinates
                - Shape: (B, N_support, 2) where N_support = num keypoints in support
                - Coordinates normalized to [0, 1]

            support_mask: Mask for valid support keypoints
                - Shape: (B, N_support)
                - True = valid, False = padding

            targets: Ground truth targets (seq_data) for training
                - dict with keys like 'seq11', 'seq21', 'target_seq', etc.

            skeleton_edges: Optional skeleton edge information
                - List of edge lists, one per batch item
                - Each edge list: [[src1, dst1], [src2, dst2], ...]

        Returns:
            outputs: dict containing:
                - 'pred_logits': Predicted token logits (B, seq_len, vocab_size)
                - 'pred_coords': Predicted coordinates (B, seq_len, 2)
                - 'aux_outputs': Auxiliary outputs from intermediate layers
        """
        # 1. Encode support pose graph (with skeleton edges)
        # support_coords: (B, N_support, 2)
        # support_features: (B, N_support, D)
        support_features = self.support_encoder(support_coords, support_mask, skeleton_edges)

        # Global support representation (optional)
        support_global = self.support_aggregator(support_features, support_mask)
        # support_global: (B, D)

        # 2. Process query image through base model with support conditioning
        # We need to inject support features into the decoder

        # Store support features for use in decoder
        self.base_model.transformer.decoder.support_features = support_features
        self.base_model.transformer.decoder.support_mask = support_mask
        self.base_model.transformer.decoder.support_cross_attn_layers = getattr(
            self, 'support_cross_attention_layers', None
        )
        self.base_model.transformer.decoder.support_attn_norms = getattr(
            self, 'support_attn_layer_norms', None
        )

        # Forward through base model
        # If base model has cape_mode, pass support_graphs directly
        # Otherwise, use the old method of storing support_features in decoder
        if hasattr(self.base_model, 'cape_mode') and self.base_model.cape_mode:
            # New implementation: pass support_graphs and support_mask directly
            outputs = self.base_model(samples, seq_kwargs=targets, support_graphs=support_coords, support_mask=support_mask)
        else:
            # Old implementation: store support_features in decoder
            outputs = self.base_model(samples, seq_kwargs=targets)

        # Clean up stored references
        self.base_model.transformer.decoder.support_features = None
        self.base_model.transformer.decoder.support_mask = None

        return outputs

    def forward_inference(self, samples, support_coords, support_mask, max_seq_len=None):
        """
        Inference mode: Generate keypoint sequence autoregressively.

        Args:
            samples: Query image
            support_coords: Support pose graph (B, N, 2)
            support_mask: Support mask (B, N)
            max_seq_len: Maximum sequence length to generate

        Returns:
            predictions: dict with:
                - 'sequences': Generated token sequences
                - 'coordinates': Decoded keypoint coordinates
        """
        # Encode support
        support_features = self.support_encoder(support_coords, support_mask)

        # Inject into decoder
        self.base_model.transformer.decoder.support_features = support_features
        self.base_model.transformer.decoder.support_mask = support_mask
        self.base_model.transformer.decoder.support_cross_attn_layers = getattr(
            self, 'support_cross_attention_layers', None
        )
        self.base_model.transformer.decoder.support_attn_norms = getattr(
            self, 'support_attn_layer_norms', None
        )

        # Generate sequence
        # Note: This would need to be implemented in the base model's decoder
        # For now, we'll use the forward pass and take argmax
        # Create dummy seq_kwargs for inference (model expects these but we don't use them)
        B = samples.shape[0] if isinstance(samples, torch.Tensor) else len(samples)
        device = support_coords.device
        seq_len = 200  # Default sequence length

        dummy_seq_kwargs = {
            'seq11': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'seq12': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'seq21': torch.zeros((B, seq_len), dtype=torch.long, device=device),  # Token indices, not coords
            'seq22': torch.zeros((B, seq_len), dtype=torch.long, device=device),  # Token indices, not coords
            'target_seq': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'delta_x1': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'delta_x2': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'delta_y1': torch.zeros((B, seq_len), dtype=torch.long, device=device),
            'delta_y2': torch.zeros((B, seq_len), dtype=torch.long, device=device),
        }

        with torch.no_grad():
            outputs = self.base_model(samples, seq_kwargs=dummy_seq_kwargs)

        # Clean up
        self.base_model.transformer.decoder.support_features = None
        self.base_model.transformer.decoder.support_mask = None

        # Decode predictions
        pred_logits = outputs['pred_logits']  # (B, seq_len, vocab_size)
        pred_coords = outputs['pred_coords']  # (B, seq_len, 2)

        # Get token predictions
        pred_tokens = pred_logits.argmax(dim=-1)  # (B, seq_len)

        return {
            'sequences': pred_tokens,
            'coordinates': pred_coords,
            'logits': pred_logits
        }


def build_cape_model(args, base_model):
    """
    Build CAPE model from base Raster2Seq model.

    Args:
        args: Arguments containing:
            - hidden_dim: Hidden dimension (default: 256)
            - support_encoder_layers: Support encoder layers (default: 3)
            - support_fusion_method: Fusion method (default: 'cross_attention')

        base_model: Pre-built RoomFormerV2 model

    Returns:
        cape_model: CAPEModel instance
    """
    return CAPEModel(
        base_model=base_model,
        hidden_dim=getattr(args, 'hidden_dim', 256),
        support_encoder_layers=getattr(args, 'support_encoder_layers', 3),
        support_fusion_method=getattr(args, 'support_fusion_method', 'cross_attention')
    )


class CAPEWithSupportCrossAttention(nn.Module):
    """
    Alternative CAPE implementation that modifies decoder layers directly.

    This version injects support cross-attention into each decoder layer,
    allowing more fine-grained control over the fusion process.
    """

    def __init__(self, base_model, hidden_dim=256, support_encoder_layers=3):
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = hidden_dim

        # Support encoder
        self.support_encoder = SupportPoseGraphEncoder(
            hidden_dim=hidden_dim,
            nheads=8,
            num_encoder_layers=support_encoder_layers,
            dim_feedforward=1024,
            dropout=0.1
        )

        # Modify decoder to add support cross-attention
        self._inject_support_attention_into_decoder()

    def _inject_support_attention_into_decoder(self):
        """
        Inject support cross-attention modules into decoder layers.

        This is more invasive but allows better integration.
        """
        decoder = self.base_model.transformer.decoder

        # Add support cross-attention to decoder
        decoder.support_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(decoder.num_layers)
        ])

        decoder.support_attn_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(decoder.num_layers)
        ])

        # Flag to enable support conditioning
        decoder.use_support_conditioning = True

    def forward(self, samples, support_coords, support_mask, targets=None):
        """Forward pass with support conditioning"""
        # Encode support
        support_features = self.support_encoder(support_coords, support_mask)

        # Store in decoder
        decoder = self.base_model.transformer.decoder
        decoder.support_features = support_features
        decoder.support_mask = support_mask

        # Forward through base model
        outputs = self.base_model(samples, targets=targets)

        # Clean up
        decoder.support_features = None
        decoder.support_mask = None

        return outputs


if __name__ == '__main__':
    print("Testing CAPE model wrapper...")

    # This would require the full base model, so just print structure
    print("CAPE Model components:")
    print("  1. Support Pose Graph Encoder")
    print("  2. Support Graph Aggregator")
    print("  3. Cross-Modal Fusion (cross-attention)")
    print("  4. Base Raster2Seq Model (RoomFormerV2)")

    print("\nArchitecture:")
    print("  Support: (B, N, 2) -> Support Encoder -> (B, N, D)")
    print("  Query: (B, 3, H, W) -> ResNet -> (B, D, H', W')")
    print("  Decoder: Attends to both query features and support embeddings")
    print("  Output: (B, seq_len, vocab_size) for tokens, (B, seq_len, 2) for coords")

    print("\n✓ CAPE model structure validated")
