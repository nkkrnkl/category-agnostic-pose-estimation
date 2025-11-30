"""
CAPE-Specific Loss Functions.
"""
import torch
import torch.nn.functional as F
from .roomformer_v2 import SetCriterion
try:
    from .label_smoothing_loss import label_smoothed_nll_loss
except ImportError:
    def label_smoothed_nll_loss(logits, target, epsilon=0.0, ignore_index=-100):
        """
        Fallback label smoothing loss implementation.
        
        Args:
            logits: Model logits
            target: Target labels
            epsilon: Smoothing factor
            ignore_index: Index to ignore
        
        Returns:
            Loss value
        """
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
    """
    def __init__(self, num_classes, semantic_classes, matcher, weight_dict, losses,
                 label_smoothing=0., per_token_sem_loss=False, eos_weight=20.0):
        """
        Initialize CAPE criterion.
        
        Args:
            num_classes: Number of token types
            semantic_classes: Number of semantic classes
            matcher: Matching module
            weight_dict: Loss weights
            losses: List of loss names to compute
            label_smoothing: Label smoothing factor
            per_token_sem_loss: Whether to compute semantic loss per token
            eos_weight: Weight multiplier for EOS token
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
        self.eos_weight = eos_weight
        self.class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.class_weights[2] = eos_weight
        print(f"✓ CAPE criterion: EOS token weight = {eos_weight}× (to combat class imbalance)")
    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss for token type prediction.
        
        Args:
            outputs: Model outputs with pred_logits
            targets: Ground truth dict with token_labels and visibility_mask
            indices: Matching indices
        
        Returns:
            losses: Dict with loss_ce and optionally loss_ce_room
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs = src_logits.shape[0]
        target_classes = targets['token_labels'].to(src_logits.device)
        valid_mask = (target_classes != -1).bool()
        if 'visibility_mask' in targets:
            visibility_mask = targets['visibility_mask'].to(src_logits.device)
            mask = valid_mask & visibility_mask
        else:
            mask = valid_mask
        if self.label_smoothing > 0:
            loss_ce = label_smoothed_nll_loss(src_logits[mask], target_classes[mask],
                                              epsilon=self.label_smoothing)
        else:
            class_weights_device = self.class_weights.to(src_logits.device)
            loss_ce = F.cross_entropy(
                src_logits[mask],
                target_classes[mask],
                weight=class_weights_device,
                reduction='mean'
            )
        losses = {'loss_ce': loss_ce}
        if 'pred_room_logits' in outputs:
            room_src_logits = outputs['pred_room_logits']
            if not self.per_token_sem_loss:
                mask = (target_classes == 3)
                room_target_classes = targets['target_polygon_labels'].to(room_src_logits.device)
                if mask.sum() > 0:
                    loss_ce_room = label_smoothed_nll_loss(
                        room_src_logits[mask], 
                        room_target_classes[room_target_classes != -1], 
                        epsilon=self.label_smoothing
                    )
                else:
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
        
        Args:
            outputs: Model outputs with pred_coords
            targets: Ground truth dict with target_seq, token_labels, visibility_mask
            indices: Matching indices
        
        Returns:
            losses: Dict with loss_coords and optionally loss_raster
        """
        assert 'pred_coords' in outputs
        bs = outputs['pred_coords'].shape[0]
        src_poly = outputs['pred_coords']
        device = src_poly.device
        token_labels = targets['token_labels'].to(device)
        target_polys = targets['target_seq'].to(device)
        coord_mask = (token_labels == 0).bool()
        if 'visibility_mask' in targets:
            visibility_mask = targets['visibility_mask'].to(device)
            mask = coord_mask & visibility_mask
        else:
            mask = coord_mask
        loss_coords = F.l1_loss(src_poly[mask], target_polys[mask])
        losses = {}
        losses['loss_coords'] = loss_coords
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
    matcher = None
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_ce_room': getattr(args, 'room_cls_loss_coef', 0.0),
        'loss_coords': args.coords_loss_coef,
    }
    if getattr(args, 'raster_loss_coef', 0) > 0:
        weight_dict['loss_raster'] = args.raster_loss_coef
    weight_dict['loss_dir'] = 1
    enc_weight_dict = {}
    enc_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'polys', 'cardinality']
    criterion = CAPESetCriterion(
        num_classes=num_classes,
        semantic_classes=getattr(args, 'semantic_classes', -1),
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
        per_token_sem_loss=getattr(args, 'per_token_sem_loss', False),
        eos_weight=getattr(args, 'eos_weight', 20.0)
    )
    return criterion