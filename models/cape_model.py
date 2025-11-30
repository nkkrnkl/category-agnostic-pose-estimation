"""
CAPE Model: Category-Agnostic Pose Estimation with Support Graph Conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .support_encoder import SupportPoseGraphEncoder
from .geometric_support_encoder import GeometricSupportEncoder
class CAPEModel(nn.Module):
    """
    CAPE model that extends Raster2Seq with support pose graph conditioning.
    """
    def __init__(self, base_model, hidden_dim=256, support_encoder_layers=3,
                 support_fusion_method='cross_attention', use_geometric_encoder=False,
                 use_gcn_preenc=False, num_gcn_layers=2):
        """
        Initialize CAPE model.
        
        Args:
            base_model: RoomFormerV2 model instance
            hidden_dim: Hidden dimension
            support_encoder_layers: Number of transformer layers in support encoder
            support_fusion_method: How to fuse support with query ('cross_attention', 'concat', 'add')
            use_geometric_encoder: Whether to use GeometricSupportEncoder
            use_gcn_preenc: Whether to use GCN pre-encoding
            num_gcn_layers: Number of GCN layers if use_gcn_preenc=True
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.support_fusion_method = support_fusion_method
        self.use_geometric_encoder = use_geometric_encoder
        if use_geometric_encoder:
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
            self.support_encoder = SupportPoseGraphEncoder(
                hidden_dim=hidden_dim,
                nheads=8,
                num_encoder_layers=support_encoder_layers,
                dim_feedforward=1024,
                dropout=0.1
            )
        if support_fusion_method == 'cross_attention':
            self._add_support_cross_attention()
        elif support_fusion_method == 'concat':
            self.support_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif support_fusion_method == 'add':
            pass
        else:
            raise ValueError(f"Unknown fusion method: {support_fusion_method}")
    def _add_support_cross_attention(self):
        """
        Add support cross-attention modules to decoder layers.
        """
        decoder = self.base_model.transformer.decoder
        num_layers = decoder.num_layers
        self.support_cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.support_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(num_layers)
        ])
    def forward(self, samples, support_coords, support_mask, targets=None, skeleton_edges=None):
        """
        Forward pass with support pose graph conditioning.
        
        Args:
            samples: Query images, shape (B, C, H, W) or NestedTensor
            support_coords: Support pose graph coordinates, shape (B, N_support, 2)
            support_mask: Mask for valid support keypoints, shape (B, N_support)
            targets: Ground truth targets dict for training
            skeleton_edges: Optional skeleton edge information, list of edge lists
        
        Returns:
            outputs: Dict with pred_logits, pred_coords, aux_outputs
        """
        if isinstance(samples, torch.Tensor):
            query_batch_size = samples.shape[0]
        elif hasattr(samples, 'tensors'):
            query_batch_size = samples.tensors.shape[0]
        else:
            query_batch_size = len(samples)
        support_batch_size = support_coords.shape[0]
        if support_batch_size != query_batch_size:
            raise ValueError(
                f"Support-Query batch size mismatch! This breaks 1-shot episodic structure.\n"
                f"  Support batch size: {support_batch_size}\n"
                f"  Query batch size: {query_batch_size}\n"
                f"Expected: Both should be (B*K) where B=episodes, K=queries_per_episode.\n"
                f"Check: episodic_collate_fn should repeat each support K times."
            )
        if support_mask.shape[0] != support_batch_size:
            raise ValueError(
                f"Support mask batch size ({support_mask.shape[0]}) doesn't match "
                f"support_coords batch size ({support_batch_size})"
            )
        if skeleton_edges is not None and len(skeleton_edges) != support_batch_size:
            raise ValueError(
                f"Skeleton edges list length ({len(skeleton_edges)}) doesn't match "
                f"batch size ({support_batch_size})"
            )
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.bool()
        if self.use_geometric_encoder:
            encoder_mask = ~support_mask
        else:
            encoder_mask = support_mask
        support_features = self.support_encoder(support_coords, encoder_mask, skeleton_edges)
        self.base_model.transformer.decoder.support_features = support_features
        self.base_model.transformer.decoder.support_mask = support_mask
        self.base_model.transformer.decoder.support_cross_attn_layers = getattr(
            self, 'support_cross_attention_layers', None
        )
        self.base_model.transformer.decoder.support_attn_norms = getattr(
            self, 'support_attn_layer_norms', None
        )
        if hasattr(self.base_model, 'cape_mode') and self.base_model.cape_mode:
            outputs = self.base_model(samples, seq_kwargs=targets, support_graphs=support_coords, support_mask=support_mask)
        else:
            outputs = self.base_model(samples, seq_kwargs=targets)
        self.base_model.transformer.decoder.support_features = None
        self.base_model.transformer.decoder.support_mask = None
        self.base_model.transformer.decoder.support_cross_attn_layers = None
        self.base_model.transformer.decoder.support_attn_norms = None
        return outputs
    def forward_inference(self, samples, support_coords, support_mask, skeleton_edges=None, max_seq_len=None, use_cache=True):
        """
        Inference mode: Generate keypoint sequence autoregressively.
        
        Args:
            samples: Query image tensor or NestedTensor
            support_coords: Support pose graph coordinates, shape (B, N, 2)
            support_mask: Support keypoint mask, shape (B, N)
            skeleton_edges: Optional skeleton edge information
            max_seq_len: Maximum sequence length to generate
            use_cache: Whether to use KV caching
        
        Returns:
            predictions: Dict with sequences, coordinates, logits
        """
        if support_mask.dtype != torch.bool:
            support_mask = support_mask.bool()
        if self.use_geometric_encoder:
            encoder_mask = ~support_mask
        else:
            encoder_mask = support_mask
        support_features = self.support_encoder(support_coords, encoder_mask, skeleton_edges)
        if hasattr(self.base_model, 'cape_mode') and self.base_model.cape_mode:
            with torch.no_grad():
                outputs = self.base_model.forward_inference(
                    samples=samples,
                    use_cache=use_cache,
                    support_graphs=support_coords,
                    support_mask=support_mask
                )
        else:
            self.base_model.transformer.decoder.support_features = support_features
            self.base_model.transformer.decoder.support_mask = support_mask
            self.base_model.transformer.decoder.support_cross_attn_layers = getattr(
                self, 'support_cross_attention_layers', None
            )
            self.base_model.transformer.decoder.support_attn_norms = getattr(
                self, 'support_attn_layer_norms', None
            )
            with torch.no_grad():
                outputs = self.base_model.forward_inference(
                    samples=samples,
                    use_cache=use_cache
                )
            self.base_model.transformer.decoder.support_features = None
            self.base_model.transformer.decoder.support_mask = None
            self.base_model.transformer.decoder.support_cross_attn_layers = None
            self.base_model.transformer.decoder.support_attn_norms = None
        pred_logits = outputs.get('pred_logits')
        pred_coords = outputs.get('pred_coords')
        if pred_logits is None:
            gen_out = outputs.get('gen_out', [])
            if gen_out and isinstance(gen_out, list):
                max_len = max(len(seq) for seq in gen_out)
                B = len(gen_out)
                pred_coords = torch.zeros((B, max_len, 2), device=support_coords.device)
                for i, seq in enumerate(gen_out):
                    coords = torch.tensor(seq, device=support_coords.device)
                    pred_coords[i, :len(seq)] = coords
        if pred_logits is not None:
            pred_tokens = pred_logits.argmax(dim=-1)
        else:
            pred_tokens = None
        return {
            'sequences': pred_tokens,
            'coordinates': pred_coords,
            'logits': pred_logits
        }
def build_cape_model(args, base_model):
    """
    Build CAPE model from base Raster2Seq model.
    
    Args:
        args: Arguments with model configuration
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
    """
    def __init__(self, base_model, hidden_dim=256, support_encoder_layers=3):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.support_encoder = SupportPoseGraphEncoder(
            hidden_dim=hidden_dim,
            nheads=8,
            num_encoder_layers=support_encoder_layers,
            dim_feedforward=1024,
            dropout=0.1
        )
        self._inject_support_attention_into_decoder()
    def _inject_support_attention_into_decoder(self):
        """
        Inject support cross-attention modules into decoder layers.
        """
        decoder = self.base_model.transformer.decoder
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
        decoder.use_support_conditioning = True
    def forward(self, samples, support_coords, support_mask, targets=None):
        """
        Forward pass with support conditioning.
        
        Args:
            samples: Query images
            support_coords: Support coordinates
            support_mask: Support mask
            targets: Optional targets
        
        Returns:
            outputs: Model outputs
        """
        support_features = self.support_encoder(support_coords, support_mask)
        decoder = self.base_model.transformer.decoder
        decoder.support_features = support_features
        decoder.support_mask = support_mask
        outputs = self.base_model(samples, targets=targets)
        decoder.support_features = None
        decoder.support_mask = None
        return outputs