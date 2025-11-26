"""
CAPE-Specific Loss Functions

This module extends the base Raster2Seq losses (SetCriterion) with
category-agnostic pose estimation specific modifications:
  - Visibility masking for occluded/unlabeled keypoints
  - Keypoint-specific loss computation

Keeping these separate from the base model maintains modularity.
"""

import torch
import torch.nn.functional as F
from .roomformer_v2 import SetCriterion

# Import label smoothing loss (with fallback)
try:
    from .label_smoothing_loss import label_smoothed_nll_loss
except ImportError:
    # Use the fallback from roomformer_v2
    def label_smoothed_nll_loss(logits, target, epsilon=0.0, ignore_index=-100):
        """Fallback label smoothing loss implementation"""
        if logits.dim() == 1:
            raise ValueError(f"logits should be 2D, got shape {logits.shape}")
        
        if target.dim() > 1:
            target = target.view(-1)
        if logits.size(0) != target.size(0):
            raise ValueError(f"logits and target batch size mismatch: {logits.size(0)} vs {target.size(0)}")
        
        if epsilon > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            n_classes = logits.size(-1)
            
            target_one_hot = torch.zeros_like(log_probs)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            target_one_hot = target_one_hot * (1 - epsilon) + epsilon / n_classes
            
            loss = -(target_one_hot * log_probs).sum(dim=-1)
            return loss.mean()
        else:
            return F.cross_entropy(logits, target, reduction='mean')


class CAPESetCriterion(SetCriterion):
    """
    CAPE-specific loss criterion with visibility masking.
    
    Extends the base SetCriterion from Raster2Seq with modifications
    specific to category-agnostic pose estimation:
    
    1. **Visibility Masking**: Only compute loss on visible keypoints
       - Filters out occluded keypoints (visibility == 1)
       - Filters out unlabeled keypoints (visibility == 0)
       - Only trains on visible keypoints (visibility == 2)
    
    2. **Keypoint-Centric**: Adapted for keypoint sequences rather than
       arbitrary polygons (floorplans)
    
    This keeps CAPE-specific logic separate from the base Raster2Seq model.
    """
    
    def __init__(self, num_classes, semantic_classes, matcher, weight_dict, losses,
                 label_smoothing=0., per_token_sem_loss=False, eos_weight=20.0):
        """
        Initialize CAPE criterion.
        
        Args:
            num_classes: Number of token types (coord, sep, eos, cls)
            semantic_classes: Number of semantic classes (for category prediction)
            matcher: Matching module (not used in seq-to-seq CAPE)
            weight_dict: Loss weights {'loss_ce': w1, 'loss_coords': w2, ...}
            losses: List of loss names to compute ['labels', 'polys', ...]
            label_smoothing: Label smoothing factor (0 = no smoothing)
            per_token_sem_loss: Whether to compute semantic loss per token
            eos_weight: Weight multiplier for EOS token to combat class imbalance (default: 20.0)
        """
        super().__init__(
            num_classes=num_classes,
            semantic_classes=semantic_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            label_smoothing=label_smoothing,
            per_token_sem_loss=per_token_sem_loss
        )
        
        # ========================================================================
        # CRITICAL FIX: Class-weighted loss to combat severe class imbalance
        # ========================================================================
        # Problem: EOS token appears ~8-20× less frequently than COORD tokens
        #          (1 EOS per sequence vs. 8-32 COORD tokens)
        # Impact: Model receives weak gradient signal for EOS prediction
        #         Result: Model doesn't learn when to stop generation
        # Solution: Give EOS tokens higher weight in cross-entropy loss
        # ========================================================================
        self.eos_weight = eos_weight
        
        # Create class weights tensor: [COORD, SEP, EOS, CLS]
        # TokenType: coord=0, sep=1, eos=2, cls=3
        self.class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.class_weights[2] = eos_weight  # EOS token gets higher weight
        print(f"✓ CAPE criterion: EOS token weight = {eos_weight}× (to combat class imbalance)")
    
    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss (NLL) for token type prediction.
        
        CAPE-specific: Applies visibility masking to only compute loss
        on visible keypoints, not occluded or unlabeled ones.
        
        Args:
            outputs: Model outputs with 'pred_logits' (B, seq_len, num_classes)
            targets: Ground truth dict with:
                - 'token_labels': Token type labels (B, seq_len)
                - 'visibility_mask': Visibility mask (B, seq_len) [CAPE-specific]
            indices: Matching indices (not used in seq-to-seq setting)
        
        Returns:
            losses: Dict with 'loss_ce' and optionally 'loss_ce_room'
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs = src_logits.shape[0]

        # ========================================================================
        # CAPE-SPECIFIC: Apply visibility masking to loss computation
        # ========================================================================
        # Only compute loss on tokens that are:
        #   1. Not padding (token_labels != -1)
        #   2. Visible (visibility_mask == True)
        # 
        # This prevents the model from being penalized for predictions on
        # occluded or unlabeled keypoints, which improves training quality.
        # ========================================================================

        target_classes = targets['token_labels'].to(src_logits.device)
        
        # Create mask: valid tokens (not padding)
        valid_mask = (target_classes != -1).bool()
        
        # Apply visibility mask if available (CAPE-specific)
        if 'visibility_mask' in targets:
            visibility_mask = targets['visibility_mask'].to(src_logits.device)
            # Combine: must be both valid AND visible
            mask = valid_mask & visibility_mask
        else:
            # Fallback: use only valid mask (backward compatibility)
            mask = valid_mask
        
        # ========================================================================
        # CRITICAL FIX: Use class-weighted cross-entropy for EOS prediction
        # ========================================================================
        # Compute loss with class weights to boost EOS learning
        # ========================================================================
        if self.label_smoothing > 0:
            # Use label smoothing (no class weights supported)
            loss_ce = label_smoothed_nll_loss(src_logits[mask], target_classes[mask],
                                              epsilon=self.label_smoothing)
        else:
            # Use class-weighted cross-entropy
            class_weights_device = self.class_weights.to(src_logits.device)
            loss_ce = F.cross_entropy(
                src_logits[mask],
                target_classes[mask],
                weight=class_weights_device,
                reduction='mean'
            )
        
        losses = {'loss_ce': loss_ce}

        # Semantic prediction (room/door/window for floorplans, category for CAPE)
        if 'pred_room_logits' in outputs:
            room_src_logits = outputs['pred_room_logits']
            
            if not self.per_token_sem_loss:
                mask = (target_classes == 3)  # cls token
                room_target_classes = targets['target_polygon_labels'].to(room_src_logits.device)
                # Only compute loss if there are CLS tokens
                if mask.sum() > 0:
                    loss_ce_room = label_smoothed_nll_loss(
                        room_src_logits[mask], 
                        room_target_classes[room_target_classes != -1], 
                        epsilon=self.label_smoothing
                    )
                else:
                    # No CLS tokens - skip room classification loss
                    loss_ce_room = torch.tensor(0.0, device=src_logits.device)
            else:
                room_target_classes = targets['target_polygon_labels'].to(room_src_logits.device)
                loss_ce_room = label_smoothed_nll_loss(
                    room_src_logits[room_target_classes != -1], 
                    room_target_classes[room_target_classes != -1], 
                    epsilon=self.label_smoothing
                )

            losses = {'loss_ce': loss_ce, 'loss_ce_room': loss_ce_room}

        return losses

    def loss_polys(self, outputs, targets, indices):
        """
        Coordinate regression loss for keypoints.
        
        CAPE-specific: Applies visibility masking to only compute loss
        on visible keypoint coordinates.
        
        Args:
            outputs: Model outputs with 'pred_coords' (B, seq_len, 2)
            targets: Ground truth dict with:
                - 'target_seq': Ground truth coordinates (B, seq_len, 2)
                - 'token_labels': Token type labels (B, seq_len)
                - 'visibility_mask': Visibility mask (B, seq_len) [CAPE-specific]
            indices: Matching indices (not used in seq-to-seq setting)
        
        Returns:
            losses: Dict with 'loss_coords' and optionally 'loss_raster'
        """
        assert 'pred_coords' in outputs
        bs = outputs['pred_coords'].shape[0]
        src_poly = outputs['pred_coords']
        device = src_poly.device
        token_labels = targets['token_labels'].to(device)
        target_polys = targets['target_seq'].to(device)

        # ========================================================================
        # CAPE-SPECIFIC: Apply visibility masking to coordinate loss
        # ========================================================================
        # Only compute loss on coordinates that are:
        #   1. Coordinate tokens (token_labels == 0, not SEP/EOS/padding)
        #   2. Visible (visibility_mask == True)
        #
        # This prevents the model from being penalized for coordinate predictions
        # on occluded or unlabeled keypoints, which improves training quality.
        # ========================================================================
        
        # Create mask: coordinate tokens only
        coord_mask = (token_labels == 0).bool()
        
        # Apply visibility mask if available (CAPE-specific)
        if 'visibility_mask' in targets:
            visibility_mask = targets['visibility_mask'].to(device)
            # Combine: must be both coordinate token AND visible
            mask = coord_mask & visibility_mask
        else:
            # Fallback: use only coord mask (backward compatibility)
            mask = coord_mask

        # Compute L1 loss only on visible coordinate tokens
        loss_coords = F.l1_loss(src_poly[mask], target_polys[mask])

        losses = {}
        losses['loss_coords'] = loss_coords

        # Rasterization loss (optional, typically not used for CAPE)
        if self.weight_dict.get('loss_raster', 0) > 0:
            pred_poly_list = self._extract_polygons(src_poly, token_labels)
            target_poly_list = self._extract_polygons(target_polys, token_labels)
            loss_raster_mask = self.raster_loss(
                pred_poly_list, 
                target_poly_list, 
                [len(x) for x in target_poly_list]
            )
            losses['loss_raster'] = loss_raster_mask

        return losses


def build_cape_criterion(args, num_classes=3):
    """
    Build CAPE-specific criterion with visibility masking.
    
    Args:
        args: Training arguments containing loss weights
        num_classes: Number of token types (default: 3 for coord/sep/eos)
    
    Returns:
        criterion: CAPESetCriterion instance
    """
    matcher = None  # Not used in seq-to-seq CAPE
    
    # Loss weights
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_ce_room': getattr(args, 'room_cls_loss_coef', 0.0),
        'loss_coords': args.coords_loss_coef,
    }
    
    if getattr(args, 'raster_loss_coef', 0) > 0:
        weight_dict['loss_raster'] = args.raster_loss_coef
    
    weight_dict['loss_dir'] = 1
    
    # Encoder losses
    enc_weight_dict = {}
    enc_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    
    # Auxiliary losses (intermediate decoder layers)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    # Loss types to compute
    losses = ['labels', 'polys', 'cardinality']
    
    # Create CAPE criterion
    criterion = CAPESetCriterion(
        num_classes=num_classes,
        semantic_classes=getattr(args, 'semantic_classes', -1),
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
        per_token_sem_loss=getattr(args, 'per_token_sem_loss', False),
        eos_weight=getattr(args, 'eos_weight', 20.0)  # Default 20× weight for EOS
    )
    
    return criterion

