import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
try:
    from detectron2.structures.instances import Instances
    from detectron2.utils.events import get_event_storage
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    class Instances:
        pass
    def get_event_storage():
        return None
from util.poly_ops import get_all_order_corners
try:
    from diff_ras.polygon import SoftPolygon
except ImportError:
    SoftPolygon = None
try:
    from util.bf_utils import get_union_box, rasterize_instances, POLY_LOSS_REGISTRY
except ImportError:
    get_union_box = None
    rasterize_instances = None
    class MockRegistry:
        def __init__(self):
            self._registry = {}
        def register(self):
            def decorator(cls):
                self._registry[cls.__name__] = cls
                return cls
            return decorator
        def get(self, name):
            return self._registry.get(name)
    POLY_LOSS_REGISTRY = MockRegistry()
def custom_L1_loss(src_polys, target_polys, target_len):
    """
    L1 loss for coordinates regression.
    
    Args:
        src_polys: Predicted polygon coordinates
        target_polys: Target polygon coordinates
        target_len: List of lengths for each polygon
    
    Returns:
        L1 loss value
    """
    total_loss = 0.
    for i in range(target_polys.shape[0]):
        tgt_poly_single = target_polys[i, :target_len[i]]
        all_polys = get_all_order_corners(tgt_poly_single)
        total_loss += torch.cdist(src_polys[i, :target_len[i]].unsqueeze(0), all_polys , p=1).min()
    total_loss = total_loss/target_len.sum()
    return total_loss
class ClippingStrategy(nn.Module):
    def __init__(self, cfg, is_boundary=False):
        super().__init__()
        self.register_buffer("laplacian", torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3))
        self.is_boundary = is_boundary
        self.side_lengths = np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)
    def _extract_target_boundary(self, masks, shape):
        """
        Extract target boundary from masks.
        
        Args:
            masks: Input masks
            shape: Target shape
        
        Returns:
            Boundary targets
        """
        boundary_targets = F.conv2d(masks.unsqueeze(1), self.laplacian, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0
        if boundary_targets.shape[-2:] != shape:
            boundary_targets = F.interpolate(
                boundary_targets, shape, mode='nearest')
        return boundary_targets
    def forward(self, instances, clip_boxes=None, lid=0):
        """
        Forward pass.
        
        Args:
            instances: Input instances
            clip_boxes: Optional clip boxes
            lid: Layer ID
        
        Returns:
            GT masks
        """                
        device = self.laplacian.device
        gt_masks = []
        if clip_boxes is not None:
            clip_boxes = torch.split(clip_boxes, [len(inst) for inst in instances], dim=0)
        for idx, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            if clip_boxes is not None:
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    clip_boxes[idx].detach(), self.side_lengths[lid][0])
            else:
                gt_masks_per_image = instances_per_image.gt_masks.rasterize_no_crop(self.side_length).to(device)
            gt_masks.append(gt_masks_per_image)
        return torch.cat(gt_masks).squeeze(1)
def dice_loss(input, target):
    """
    Compute dice loss.
    
    Args:
        input: Input tensor
        target: Target tensor
    
    Returns:
        Dice loss value
    """
    smooth = 1.
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
def dice_loss_no_reduction(input, target):
    """
    Compute dice loss without reduction.
    
    Args:
        input: Input tensor
        target: Target tensor
    
    Returns:
        Dice loss tensor
    """
    smooth = 1.
    iflat = input.flatten(-2,-1)
    tflat = target.flatten(-2,-1)
    intersection = (iflat * tflat).sum(1)
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum(1) + tflat.sum(1) + smooth))
@POLY_LOSS_REGISTRY.register()
class MaskRasterizationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("rasterize_at", torch.from_numpy(np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)))
        self.inv_smoothness_schedule = (0.1,)
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = ()
        self.inv_smoothness_idx = 0
        self.iter = 0
        self.use_rasterized_gt = True
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.clip_to_proposal = False
        self.predict_in_box_space = True
        if self.clip_to_proposal or not self.use_rasterized_gt:
            self.clipper = ClippingStrategy(cfg=None)
            self.gt_rasterizer = None
        else:
            self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")
        self.offset = 0.5
        self.loss_fn = dice_loss
        self.name = "mask"
    def _create_targets(self, instances, clip_boxes=None, lid=0):
        """
        Create targets from instances.
        
        Args:
            instances: Input instances
            clip_boxes: Optional clip boxes
            lid: Layer ID
        
        Returns:
            Target masks
        """
        if self.clip_to_proposal or not self.use_rasterized_gt:
            targets = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)            
        else:            
            targets = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
        return targets
    def forward(self, preds, targets, target_len, lid=0):
        """
        Forward pass.
        
        Args:
            preds: Predicted polygons
            targets: Target polygons
            target_len: Lengths for each polygon
            lid: Layer ID
        
        Returns:
            Loss value
        """
        resolution = self.rasterize_at[lid]
        target_masks = []
        pred_masks = []
        for i in range(len(targets)):
            tgt_poly_single = targets[i][:target_len[i]].view(-1, 2).unsqueeze(0)
            pred_poly_single = preds[i][:target_len[i]].view(-1, 2).unsqueeze(0)
            tgt_mask = self.gt_rasterizer(tgt_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            tgt_mask = (tgt_mask + 1)/2
            pred_mask = self.pred_rasterizer(pred_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            target_masks.append(tgt_mask)
            pred_masks.append(pred_mask)
        pred_masks = torch.stack(pred_masks)
        target_masks = torch.stack(target_masks)
        return self.loss_fn(pred_masks, target_masks)
class MaskRasterizationCost(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("rasterize_at", torch.from_numpy(np.array([64, 64, 64, 64, 64, 64, 64, 64]).reshape(-1, 2)))
        self.inv_smoothness_schedule = (0.1,)
        self.inv_smoothness = self.inv_smoothness_schedule[0]
        self.inv_smoothness_iter = ()
        self.inv_smoothness_idx = 0
        self.iter = 0
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.use_rasterized_gt = True
        self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")
        self.offset = 0.5
        self.loss_fn = dice_loss_no_reduction
        self.name = "mask"
    def mask_iou(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        ) -> torch.Tensor:
        """
        Compute mask IoU.
        
        Args:
            mask1: First mask tensor, shape NxHxW
            mask2: Second mask tensor, shape MxHxW
        
        Returns:
            IoU tensor, shape NxM
        """
        N, H, W = mask1.shape
        M, H, W = mask2.shape
        mask1 = mask1.view(N, H*W)
        mask2 = mask2.view(M, H*W)
        intersection = torch.matmul(mask1, mask2.t())
        area1 = mask1.sum(dim=1).view(1, -1)
        area2 = mask2.sum(dim=1).view(1, -1)
        union = (area1.t() + area2) - intersection
        ret = torch.where(
            union == 0,
            torch.tensor(0., device=mask1.device),
            intersection / union,
        )
        return ret
    def forward(self, preds, targets, target_len, lid=0):
        """
        Forward pass.
        
        Args:
            preds: Predicted polygons
            targets: Target polygons
            target_len: Lengths for each polygon
            lid: Layer ID
        
        Returns:
            Cost mask tensor
        """
        resolution = self.rasterize_at[lid]
        target_masks = []
        pred_masks = []
        cost_mask = torch.zeros([preds.shape[0], targets.shape[0]], device=preds.device)
        for i in range(targets.shape[0]):
            tgt_poly_single = targets[i, :target_len[i]].view(-1, 2).unsqueeze(0)
            pred_poly_all = preds[:,:target_len[i]].view(preds.shape[0], -1, 2)
            tgt_mask = self.gt_rasterizer(tgt_poly_single * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            pred_masks = self.pred_rasterizer(pred_poly_all * float(resolution[1].item()), resolution[1].item(), resolution[0].item(), 1.0)
            tgt_mask = (tgt_mask + 1)/2
            tgt_masks = tgt_mask.repeat(preds.shape[0], 1, 1)
            cost_mask[:, i] = self.loss_fn(tgt_masks, pred_masks) 
        return cost_mask