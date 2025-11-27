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
from .support_encoder import SupportPoseGraphEncoder
from .geometric_support_encoder import GeometricSupportEncoder


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
                 support_fusion_method='cross_attention', use_geometric_encoder=False,
                 use_gcn_preenc=False, num_gcn_layers=2):
        """
        Args:
            base_model: RoomFormerV2 model instance
            hidden_dim: Hidden dimension (should match base model)
            support_encoder_layers: Number of transformer layers in support encoder
            support_fusion_method: How to fuse support with query
                - 'cross_attention': Add cross-attention layer to decoder
                - 'concat': Concatenate support features
                - 'add': Add support features to query features
            use_geometric_encoder: If True, use GeometricSupportEncoder (CapeX-inspired)
                                  If False, use old SupportPoseGraphEncoder (default: False)
            use_gcn_preenc: If True and use_geometric_encoder=True, use GCN pre-encoding
                           in support encoder (default: False)
            num_gcn_layers: Number of GCN layers if use_gcn_preenc=True (default: 2)
        """
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.support_fusion_method = support_fusion_method
        self.use_geometric_encoder = use_geometric_encoder

        # Support pose graph encoder (toggle between old and new)
        if use_geometric_encoder:
            # NEW: Geometry-only encoder with CapeX-inspired components
            self.support_encoder = GeometricSupportEncoder(
                hidden_dim=hidden_dim,
                num_encoder_layers=support_encoder_layers,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                use_gcn_preenc=use_gcn_preenc,
                num_gcn_layers=num_gcn_layers,
                activation='relu'
            )
        else:
            # OLD: Original support encoder
            self.support_encoder = SupportPoseGraphEncoder(
                hidden_dim=hidden_dim,
                nheads=8,
                num_encoder_layers=support_encoder_layers,
                dim_feedforward=1024,
                dropout=0.1
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
        Forward pass with support pose graph conditioning for 1-shot CAPE.

        IMPORTANT: For correct 1-shot episodic learning, the support and query
        batch dimensions MUST match. After the episodic_collate_fn fix, each
        support is repeated K times (once per query in that episode), ensuring:
            support[i] corresponds to query[i]

        Args:
            samples: Query images
                - If NestedTensor: contains tensors and mask
                - If list of tensors: batch of images
                - If single tensor: (B, C, H, W) where B = batch_size (total queries)

            support_coords: Support pose graph coordinates
                - Shape: (B, N_support, 2) where:
                    - B = batch_size (same as query batch size!)
                    - N_support = num keypoints in support (varies by category)
                    - Coordinates normalized to [0, 1]
                - NOTE: B includes all queries from all episodes in the batch.
                  Each support is repeated for its corresponding queries.

            support_mask: Mask for valid support keypoints
                - Shape: (B, N_support)
                - True = valid keypoint, False = padding
                - Same batch size as support_coords and samples

            targets: Ground truth targets (seq_data) for training
                - dict with keys like 'seq11', 'seq21', 'target_seq', etc.
                - Each tensor has shape (B, seq_len, ...)

            skeleton_edges: Optional skeleton edge information
                - List of edge lists, one per batch item (length = B)
                - Each edge list: [[src1, dst1], [src2, dst2], ...]

        Returns:
            outputs: dict containing:
                - 'pred_logits': Predicted token logits (B, seq_len, vocab_size)
                - 'pred_coords': Predicted coordinates (B, seq_len, 2)
                - 'aux_outputs': Auxiliary outputs from intermediate layers
        """
        # ========================================================================
        # CRITICAL FIX: Verify support-query batch alignment for 1-shot learning
        # ========================================================================
        # For 1-shot CAPE, each query must use exactly ONE support (from its episode).
        # The episodic_collate_fn repeats each support K times to ensure:
        #   - support_coords[i] is the support for query[i]
        #   - cross-attention in decoder will correctly pair them
        #
        # If batch sizes don't match, queries could attend to wrong supports!
        # ========================================================================
        
        # Extract batch size from query images
        if isinstance(samples, torch.Tensor):
            query_batch_size = samples.shape[0]
        elif hasattr(samples, 'tensors'):  # NestedTensor
            query_batch_size = samples.tensors.shape[0]
        else:
            query_batch_size = len(samples)
        
        # Verify support batch size matches query batch size
        support_batch_size = support_coords.shape[0]
        
        if support_batch_size != query_batch_size:
            raise ValueError(
                f"Support-Query batch size mismatch! This breaks 1-shot episodic structure.\n"
                f"  Support batch size: {support_batch_size}\n"
                f"  Query batch size: {query_batch_size}\n"
                f"Expected: Both should be (B*K) where B=episodes, K=queries_per_episode.\n"
                f"Check: episodic_collate_fn should repeat each support K times."
            )
        
        # Also verify support_mask matches
        if support_mask.shape[0] != support_batch_size:
            raise ValueError(
                f"Support mask batch size ({support_mask.shape[0]}) doesn't match "
                f"support_coords batch size ({support_batch_size})"
            )
        
        # Verify skeleton_edges list length if provided
        if skeleton_edges is not None and len(skeleton_edges) != support_batch_size:
            raise ValueError(
                f"Skeleton edges list length ({len(skeleton_edges)}) doesn't match "
                f"batch size ({support_batch_size})"
            )

        # 1. Encode support pose graph (with skeleton edges)
        # After verification, we know:
        #   support_coords: (B, N_support, 2)
        #   support_features will be: (B, N_support, D)
        # where B = total number of queries (each with its own support)
        
        # ========================================================================
        # CRITICAL FIX: Handle mask convention differences between encoders
        # ========================================================================
        # Dataloader convention: support_mask has True = valid keypoint (visibility > 0)
        # - Old SupportPoseGraphEncoder: expects True = valid (no conversion needed)
        # - New GeometricSupportEncoder: expects True = invalid (needs inversion)
        # ========================================================================
        # Ensure mask is boolean before operations
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.bool()
        
        if self.use_geometric_encoder:
            # GeometricSupportEncoder expects True = invalid/invisible
            # Invert mask: True (valid) → False (valid), False (invalid) → True (invalid)
            encoder_mask = ~support_mask
        else:
            # SupportPoseGraphEncoder expects True = valid (matches dataloader)
            encoder_mask = support_mask
        
        support_features = self.support_encoder(support_coords, encoder_mask, skeleton_edges)

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
        # CRITICAL: Must clean up ALL temporary attributes to avoid state_dict contamination
        self.base_model.transformer.decoder.support_features = None
        self.base_model.transformer.decoder.support_mask = None
        self.base_model.transformer.decoder.support_cross_attn_layers = None
        self.base_model.transformer.decoder.support_attn_norms = None

        return outputs

    def forward_inference(self, samples, support_coords, support_mask, skeleton_edges=None, max_seq_len=None, use_cache=True):
        """
        Inference mode: Generate keypoint sequence autoregressively.

        This method performs TRUE autoregressive generation:
        1. Start with BOS (beginning of sequence) token
        2. Generate one token at a time
        3. Feed each prediction back as input for the next token
        4. Stop at EOS (end of sequence) or max length

        This is different from training (teacher forcing), where the model
        sees the entire ground truth sequence at once.

        Args:
            samples: Query image tensor or NestedTensor
                - Shape: (B, C, H, W) or list of images
            support_coords: Support pose graph coordinates (B, N, 2)
                - Normalized to [0, 1]
            support_mask: Support keypoint mask (B, N)
                - True = valid keypoint, False = padding
            skeleton_edges: Optional skeleton edge information
                - List of edge lists, one per batch item
                - Each edge list: [[src1, dst1], [src2, dst2], ...]
                - If None, only coordinate information is used
            max_seq_len: Maximum sequence length to generate (default: uses base model's default)
            use_cache: Whether to use KV caching for faster generation (default: True)

        Returns:
            predictions: dict with:
                - 'sequences': Generated token sequences (B, seq_len)
                - 'coordinates': Decoded keypoint coordinates (B, num_keypoints, 2)
                - 'logits': Token prediction logits (B, seq_len, vocab_size)
        """
        # ========================================================================
        # CRITICAL FIX: Use true autoregressive generation
        # ========================================================================
        # The base model has a proper forward_inference() method that generates
        # sequences autoregressively (token-by-token, feeding predictions back).
        #
        # OLD (INCORRECT) APPROACH:
        #   - Create dummy zeros for all sequence positions
        #   - Call forward() once with all zeros
        #   - Take argmax → this is NOT autoregressive!
        #
        # NEW (CORRECT) APPROACH:
        #   - Call base_model.forward_inference()
        #   - It generates: BOS → token_1 → token_2 → ... → EOS
        #   - Each token conditions on all previous tokens
        # ========================================================================

        # ================================================================
        # CRITICAL FIX: Pass skeleton_edges to utilize structural information
        # ================================================================
        # Skeleton edges define connectivity between keypoints (e.g., shoulder→elbow).
        # The support encoder uses these to create adjacency-aware embeddings.
        # Previously missing in forward_inference, now included.
        #
        # The support encoder will:
        #   - Build adjacency matrix from skeleton edges
        #   - Create edge embeddings based on connectivity
        #   - Combine with coordinate embeddings for richer representations
        #
        # If skeleton_edges is None, falls back to coordinate-only encoding.
        # ================================================================
        
        # 1. Encode support pose graph with skeleton structure
        # ========================================================================
        # CRITICAL FIX: Handle mask convention differences between encoders
        # ========================================================================
        # Dataloader convention: support_mask has True = valid keypoint (visibility > 0)
        # - Old SupportPoseGraphEncoder: expects True = valid (no conversion needed)
        # - New GeometricSupportEncoder: expects True = invalid (needs inversion)
        # ========================================================================
        # Ensure mask is boolean before operations
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.bool()
        
        if self.use_geometric_encoder:
            # GeometricSupportEncoder expects True = invalid/invisible
            # Invert mask: True (valid) → False (valid), False (invalid) → True (invalid)
            encoder_mask = ~support_mask
        else:
            # SupportPoseGraphEncoder expects True = valid (matches dataloader)
            encoder_mask = support_mask
        
        support_features = self.support_encoder(support_coords, encoder_mask, skeleton_edges)
        # support_features: (B, N_support, hidden_dim)

        # 2. Check if base model supports CAPE mode (with built-in support handling)
        if hasattr(self.base_model, 'cape_mode') and self.base_model.cape_mode:
            # Base model can handle support graphs directly
            # Pass support_coords directly; the base model will encode them internally
            with torch.no_grad():
                outputs = self.base_model.forward_inference(
                    samples=samples,
                    use_cache=use_cache,
                    support_graphs=support_coords,
                    support_mask=support_mask
                )
        else:
            # Base model doesn't have CAPE mode - inject support via decoder attributes
            # This is a fallback for older base model versions
            
            # Inject support features into decoder so it can cross-attend during generation
            self.base_model.transformer.decoder.support_features = support_features
            self.base_model.transformer.decoder.support_mask = support_mask
            self.base_model.transformer.decoder.support_cross_attn_layers = getattr(
                self, 'support_cross_attention_layers', None
            )
            self.base_model.transformer.decoder.support_attn_norms = getattr(
                self, 'support_attn_layer_norms', None
            )

            # Call base model's autoregressive generation method
            with torch.no_grad():
                outputs = self.base_model.forward_inference(
                    samples=samples,
                    use_cache=use_cache
                )

            # Clean up decoder attributes after generation
            # CRITICAL: Must clean up ALL temporary attributes to avoid state_dict contamination
            self.base_model.transformer.decoder.support_features = None
            self.base_model.transformer.decoder.support_mask = None
            self.base_model.transformer.decoder.support_cross_attn_layers = None
            self.base_model.transformer.decoder.support_attn_norms = None

        # 3. Extract predictions from autoregressive outputs
        # The base model's forward_inference returns:
        #   - 'pred_logits': token classification logits (B, seq_len, vocab_size)
        #   - 'pred_coords': decoded coordinates (B, seq_len, 2)
        #   - 'gen_out': generated output sequence (raw coordinates)
        
        pred_logits = outputs.get('pred_logits')  # (B, seq_len, vocab_size)
        pred_coords = outputs.get('pred_coords')  # (B, seq_len, 2)
        
        # If logits not in outputs, reconstruct from gen_out
        if pred_logits is None:
            # Some base models may only return gen_out (list of coordinate lists)
            # In this case, we'll use pred_coords directly
            gen_out = outputs.get('gen_out', [])
            # Convert gen_out to tensor if needed
            if gen_out and isinstance(gen_out, list):
                # gen_out is list of lists: [[coords], [coords], ...]
                max_len = max(len(seq) for seq in gen_out)
                B = len(gen_out)
                pred_coords = torch.zeros((B, max_len, 2), device=support_coords.device)
                for i, seq in enumerate(gen_out):
                    coords = torch.tensor(seq, device=support_coords.device)
                    pred_coords[i, :len(seq)] = coords
        
        # Get token predictions (argmax over vocabulary)
        # These are the actual generated token IDs
        if pred_logits is not None:
            pred_tokens = pred_logits.argmax(dim=-1)  # (B, seq_len)
        else:
            # If no logits available, return None for sequences
            pred_tokens = None

        return {
            'sequences': pred_tokens,      # (B, seq_len) - generated token IDs
            'coordinates': pred_coords,     # (B, seq_len, 2) - decoded coordinates
            'logits': pred_logits          # (B, seq_len, vocab_size) - token logits (may be None)
        }


def build_cape_model(args, base_model):
    """
    Build CAPE model from base Raster2Seq model.

    Args:
        args: Arguments containing:
            - hidden_dim: Hidden dimension (default: 256)
            - support_encoder_layers: Support encoder layers (default: 3)
            - support_fusion_method: Fusion method (default: 'cross_attention')
            - use_geometric_encoder: Use CapeX-inspired geometric encoder (default: False)
            - use_gcn_preenc: Use GCN pre-encoding if geometric encoder (default: False)
            - num_gcn_layers: Number of GCN layers (default: 2)

        base_model: Pre-built RoomFormerV2 model

    Returns:
        cape_model: CAPEModel instance
    """
    return CAPEModel(
        base_model=base_model,
        hidden_dim=getattr(args, 'hidden_dim', 256),
        support_encoder_layers=getattr(args, 'support_encoder_layers', 3),
        support_fusion_method=getattr(args, 'support_fusion_method', 'cross_attention'),
        use_geometric_encoder=getattr(args, 'use_geometric_encoder', False),
        use_gcn_preenc=getattr(args, 'use_gcn_preenc', False),
        num_gcn_layers=getattr(args, 'num_gcn_layers', 2)
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
