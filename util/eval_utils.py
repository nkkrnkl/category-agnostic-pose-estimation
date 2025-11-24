"""
Evaluation utilities for Category-Agnostic Pose Estimation (CAPE).

Implements PCK@bbox (Percentage of Correct Keypoints) metric as used in:
- POMNet
- CAPE  
- MP-100 benchmark
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union


def compute_f1(quant_result_dict, metric_category):
    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec+1e-5)
        quant_result_dict[metric+'_f1'] = f1
    return quant_result_dict


def compute_pck_bbox(
    pred_keypoints: Union[np.ndarray, torch.Tensor],
    gt_keypoints: Union[np.ndarray, torch.Tensor],
    bbox_width: float,
    bbox_height: float,
    visibility: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.2,
    normalize_by: str = 'diagonal'
) -> Tuple[float, int, int]:
    """
    Compute PCK@bbox (Percentage of Correct Keypoints) for a single instance.
    
    PCK measures the percentage of predicted keypoints that fall within a 
    threshold distance from the ground truth, normalized by the bounding box size.
    
    Args:
        pred_keypoints: Predicted keypoints, shape (N, 2) where N = num keypoints
                       Can be in pixel coordinates or normalized [0,1] coordinates
        gt_keypoints: Ground truth keypoints, shape (N, 2)
                     Must be in same coordinate system as pred_keypoints
        bbox_width: Width of the bounding box (used for normalization)
        bbox_height: Height of the bounding box (used for normalization)
        visibility: Visibility flags for each keypoint, shape (N,)
                   - If None, all keypoints are considered visible
                   - Values: 0 = not labeled, 1 or 2 = visible/occluded
                   - Only keypoints with visibility > 0 are evaluated
        threshold: Distance threshold as fraction of bbox size (default: 0.2)
        normalize_by: How to compute bbox_size for normalization:
                     - 'diagonal': sqrt(width^2 + height^2) [standard for PCK@bbox]
                     - 'max': max(width, height)
                     - 'mean': (width + height) / 2
    
    Returns:
        pck: PCK score (0.0 to 1.0) - fraction of correct keypoints
        num_correct: Number of correct keypoints
        num_visible: Number of visible keypoints evaluated
        
    Example:
        >>> pred = np.array([[0.5, 0.3], [0.2, 0.8]])  # normalized coords
        >>> gt = np.array([[0.52, 0.31], [0.19, 0.82]])
        >>> pck, correct, total = compute_pck_bbox(pred, gt, 512, 512, threshold=0.2)
        >>> print(f"PCK@0.2: {pck:.2%} ({correct}/{total} keypoints correct)")
    """
    # Convert to numpy for computation
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.detach().cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.detach().cpu().numpy()
    if visibility is not None and isinstance(visibility, torch.Tensor):
        visibility = visibility.detach().cpu().numpy()
    
    # Ensure correct shapes
    assert pred_keypoints.shape == gt_keypoints.shape, \
        f"Shape mismatch: pred {pred_keypoints.shape} vs gt {gt_keypoints.shape}"
    assert pred_keypoints.shape[1] == 2, \
        f"Expected (N, 2) keypoints, got {pred_keypoints.shape}"
    
    num_keypoints = len(pred_keypoints)
    
    # Handle visibility mask
    if visibility is None:
        # All keypoints are visible
        visible_mask = np.ones(num_keypoints, dtype=bool)
    else:
        # Only evaluate keypoints with visibility > 0
        visible_mask = np.array(visibility) > 0
    
    num_visible = visible_mask.sum()
    
    # Handle edge case: no visible keypoints
    if num_visible == 0:
        return 0.0, 0, 0
    
    # Extract only visible keypoints
    pred_visible = pred_keypoints[visible_mask]
    gt_visible = gt_keypoints[visible_mask]
    
    # Compute Euclidean distance for each keypoint
    # distances shape: (num_visible,)
    distances = np.sqrt(np.sum((pred_visible - gt_visible) ** 2, axis=1))
    
    # Compute bbox size for normalization
    if normalize_by == 'diagonal':
        # Standard: use diagonal of bbox
        bbox_size = np.sqrt(bbox_width ** 2 + bbox_height ** 2)
    elif normalize_by == 'max':
        # Alternative: use max dimension
        bbox_size = max(bbox_width, bbox_height)
    elif normalize_by == 'mean':
        # Alternative: use mean dimension
        bbox_size = (bbox_width + bbox_height) / 2
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")
    
    # Normalize distances by bbox size
    normalized_distances = distances / bbox_size
    
    # Check which keypoints are correct (within threshold)
    correct_mask = normalized_distances < threshold
    num_correct = correct_mask.sum()
    
    # Compute PCK
    pck = num_correct / num_visible
    
    return float(pck), int(num_correct), int(num_visible)


def compute_pck_batch(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    bbox_widths: torch.Tensor,
    bbox_heights: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    threshold: float = 0.2,
    normalize_by: str = 'diagonal'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCK@bbox for a batch of instances.
    
    Args:
        pred_keypoints: Predicted keypoints, shape (B, N, 2)
        gt_keypoints: Ground truth keypoints, shape (B, N, 2)
        bbox_widths: Bbox widths for each instance, shape (B,)
        bbox_heights: Bbox heights for each instance, shape (B,)
        visibility: Visibility flags, shape (B, N) or None
        threshold: Distance threshold (default: 0.2)
        normalize_by: Normalization method (default: 'diagonal')
    
    Returns:
        pck_scores: PCK for each instance, shape (B,)
        num_correct: Number of correct keypoints per instance, shape (B,)
        num_visible: Number of visible keypoints per instance, shape (B,)
    """
    batch_size = pred_keypoints.shape[0]
    
    pck_scores = []
    num_correct_list = []
    num_visible_list = []
    
    for i in range(batch_size):
        vis = visibility[i] if visibility is not None else None
        
        pck, correct, visible = compute_pck_bbox(
            pred_keypoints[i],
            gt_keypoints[i],
            bbox_widths[i].item(),
            bbox_heights[i].item(),
            visibility=vis,
            threshold=threshold,
            normalize_by=normalize_by
        )
        
        pck_scores.append(pck)
        num_correct_list.append(correct)
        num_visible_list.append(visible)
    
    return (
        torch.tensor(pck_scores),
        torch.tensor(num_correct_list),
        torch.tensor(num_visible_list)
    )


class PCKEvaluator:
    """
    Accumulator for PCK metrics across multiple images and categories.
    
    Usage:
        evaluator = PCKEvaluator(threshold=0.2)
        
        for batch in dataloader:
            predictions = model(batch)
            evaluator.add_batch(
                pred_keypoints=predictions,
                gt_keypoints=batch['keypoints'],
                bbox_widths=batch['bbox_widths'],
                bbox_heights=batch['bbox_heights'],
                category_ids=batch['category_ids'],
                visibility=batch['visibility']
            )
        
        results = evaluator.get_results()
        print(f"Overall PCK@0.2: {results['pck_overall']:.2%}")
    """
    
    def __init__(self, threshold: float = 0.2, normalize_by: str = 'diagonal'):
        """
        Args:
            threshold: PCK threshold (default: 0.2 for PCK@0.2)
            normalize_by: Bbox size normalization method
        """
        self.threshold = threshold
        self.normalize_by = normalize_by
        
        # Overall stats
        self.total_correct = 0
        self.total_visible = 0
        
        # Per-category stats
        self.category_correct = {}  # category_id -> num_correct
        self.category_visible = {}  # category_id -> num_visible
        
        # Per-image stats (for detailed analysis)
        self.image_results = []  # List of dicts with per-image results
    
    def add_batch(
        self,
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        bbox_widths: torch.Tensor,
        bbox_heights: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        image_ids: Optional[List] = None
    ):
        """
        Add a batch of predictions to the evaluator.
        
        Args:
            pred_keypoints: Predicted keypoints, shape (B, N, 2)
            gt_keypoints: Ground truth keypoints, shape (B, N, 2)
            bbox_widths: Bbox widths, shape (B,)
            bbox_heights: Bbox heights, shape (B,)
            category_ids: Category IDs, shape (B,) or None
            visibility: Visibility flags, shape (B, N) or None
            image_ids: List of image IDs for tracking (optional)
        """
        batch_size = pred_keypoints.shape[0]
        
        for i in range(batch_size):
            vis = visibility[i] if visibility is not None else None
            cat_id = category_ids[i].item() if category_ids is not None else 0
            img_id = image_ids[i] if image_ids is not None else None
            
            pck, correct, visible = compute_pck_bbox(
                pred_keypoints[i],
                gt_keypoints[i],
                bbox_widths[i].item(),
                bbox_heights[i].item(),
                visibility=vis,
                threshold=self.threshold,
                normalize_by=self.normalize_by
            )
            
            # Update overall stats
            self.total_correct += correct
            self.total_visible += visible
            
            # Update per-category stats
            if cat_id not in self.category_correct:
                self.category_correct[cat_id] = 0
                self.category_visible[cat_id] = 0
            
            self.category_correct[cat_id] += correct
            self.category_visible[cat_id] += visible
            
            # Store per-image result
            self.image_results.append({
                'image_id': img_id,
                'category_id': cat_id,
                'pck': pck,
                'num_correct': correct,
                'num_visible': visible
            })
    
    def get_results(self) -> Dict:
        """
        Get evaluation results.
        
        Returns:
            Dictionary with:
                - pck_overall: Overall PCK score
                - pck_per_category: Dict of category_id -> PCK
                - mean_pck_categories: Mean PCK across categories (macro-average)
                - total_correct: Total correct keypoints
                - total_visible: Total visible keypoints
                - num_categories: Number of categories evaluated
                - num_images: Number of images evaluated
        """
        # Overall PCK
        pck_overall = self.total_correct / self.total_visible if self.total_visible > 0 else 0.0
        
        # Per-category PCK
        pck_per_category = {}
        for cat_id in self.category_correct.keys():
            correct = self.category_correct[cat_id]
            visible = self.category_visible[cat_id]
            pck_per_category[cat_id] = correct / visible if visible > 0 else 0.0
        
        # Mean PCK across categories (macro-average)
        mean_pck_categories = np.mean(list(pck_per_category.values())) if pck_per_category else 0.0
        
        return {
            'pck_overall': pck_overall,
            'pck_per_category': pck_per_category,
            'mean_pck_categories': mean_pck_categories,
            'total_correct': self.total_correct,
            'total_visible': self.total_visible,
            'num_categories': len(pck_per_category),
            'num_images': len(self.image_results),
            'threshold': self.threshold
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_correct = 0
        self.total_visible = 0
        self.category_correct = {}
        self.category_visible = {}
        self.image_results = []