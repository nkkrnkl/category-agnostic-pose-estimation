"""
Evaluation utilities for Category-Agnostic Pose Estimation (CAPE).
Implements PCK@bbox (Percentage of Correct Keypoints) metric as used in:
- POMNet
- CAPE  
- MP-100 benchmark
"""
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
def compute_f1(quant_result_dict, metric_category):
    """
    Compute F1 scores for metrics.
    
    Args:
        quant_result_dict: Dictionary with precision and recall values
        metric_category: List of metric names
    
    Returns:
        Dictionary with added F1 scores
    """
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
    Compute PCK@bbox for a single instance.
    
    Args:
        pred_keypoints: Predicted keypoints, shape (N, 2)
        gt_keypoints: Ground truth keypoints, shape (N, 2)
        bbox_width: Width of the bounding box
        bbox_height: Height of the bounding box
        visibility: Visibility flags for each keypoint, shape (N,)
        threshold: Distance threshold as fraction of bbox size
        normalize_by: Normalization method ('diagonal', 'max', or 'mean')
    
    Returns:
        pck: PCK score (0.0 to 1.0)
        num_correct: Number of correct keypoints
        num_visible: Number of visible keypoints evaluated
    """
    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = pred_keypoints.detach().cpu().numpy()
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.detach().cpu().numpy()
    if visibility is not None and isinstance(visibility, torch.Tensor):
        visibility = visibility.detach().cpu().numpy()
    assert pred_keypoints.shape == gt_keypoints.shape, \
        f"Shape mismatch: pred {pred_keypoints.shape} vs gt {gt_keypoints.shape}"
    assert pred_keypoints.shape[1] == 2, \
        f"Expected (N, 2) keypoints, got {pred_keypoints.shape}"
    num_keypoints = len(pred_keypoints)
    if visibility is None:
        visible_mask = np.ones(num_keypoints, dtype=bool)
    else:
        visibility_array = np.array(visibility)
        assert len(visibility_array) == num_keypoints, \
            f"Visibility length ({len(visibility_array)}) must match keypoints ({num_keypoints})"
        visible_mask = visibility_array > 0
    num_visible = visible_mask.sum()
    if num_visible == 0:
        return 0.0, 0, 0
    pred_visible = pred_keypoints[visible_mask]
    gt_visible = gt_keypoints[visible_mask]
    if np.allclose(pred_visible, gt_visible, atol=1e-6):
        import warnings
        warnings.warn(
            "Predictions are IDENTICAL to ground truth! "
            "This indicates data leakage or a bug in the model. "
            "Check that evaluation uses forward_inference (not teacher forcing).",
            RuntimeWarning
        )
    distances = np.sqrt(np.sum((pred_visible - gt_visible) ** 2, axis=1))
    if normalize_by == 'diagonal':
        bbox_size = np.sqrt(bbox_width ** 2 + bbox_height ** 2)
    elif normalize_by == 'max':
        bbox_size = max(bbox_width, bbox_height)
    elif normalize_by == 'mean':
        bbox_size = (bbox_width + bbox_height) / 2
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")
    normalized_distances = distances / bbox_size
    correct_mask = normalized_distances < threshold
    num_correct = correct_mask.sum()
    pck = num_correct / num_visible
    DEBUG_PCK = os.environ.get('DEBUG_PCK', '0') == '1'
    if DEBUG_PCK:
        print(f"\n[DEBUG_PCK] compute_pck_bbox:")
        print(f"  num_visible: {num_visible}")
        print(f"  bbox_size: {bbox_size:.2f}")
        print(f"  threshold (alpha): {threshold}")
        print(f"  threshold (pixels): {threshold * bbox_size:.2f}")
        print(f"  distances (first 5): {distances[:5]}")
        print(f"  normalized_distances (first 5): {normalized_distances[:5]}")
        print(f"  num_correct: {num_correct} / {num_visible}")
        print(f"  PCK: {pck:.2%}")
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
    """
    def __init__(self, threshold: float = 0.2, normalize_by: str = 'diagonal'):
        """
        Initialize PCK evaluator.
        
        Args:
            threshold: PCK threshold
            normalize_by: Bbox size normalization method
        """
        self.threshold = threshold
        self.normalize_by = normalize_by
        self.total_correct = 0
        self.total_visible = 0
        self.category_correct = {}
        self.category_visible = {}
        self.image_results = []
    def add_batch(
        self,
        pred_keypoints,
        gt_keypoints,
        bbox_widths: torch.Tensor,
        bbox_heights: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
        image_ids: Optional[List] = None
    ):
        """
        Add a batch of predictions to the evaluator.
        
        Args:
            pred_keypoints: Predicted keypoints, Tensor (B, N, 2) or List of Tensors
            gt_keypoints: Ground truth keypoints, same format as pred_keypoints
            bbox_widths: Bbox widths, shape (B,)
            bbox_heights: Bbox heights, shape (B,)
            category_ids: Category IDs, shape (B,) or None
            visibility: Visibility flags, shape (B, N) or None
            image_ids: List of image IDs or None
        """
        if isinstance(pred_keypoints, list):
            batch_size = len(pred_keypoints)
        else:
            batch_size = pred_keypoints.shape[0]
        for i in range(batch_size):
            if isinstance(pred_keypoints, list):
                pred_i = pred_keypoints[i]
                gt_i = gt_keypoints[i]
            else:
                pred_i = pred_keypoints[i]
                gt_i = gt_keypoints[i]
            vis = visibility[i] if visibility is not None else None
            cat_id = category_ids[i].item() if category_ids is not None else 0
            img_id = image_ids[i] if image_ids is not None else None
            pck, correct, visible = compute_pck_bbox(
                pred_i,
                gt_i,
                bbox_widths[i].item(),
                bbox_heights[i].item(),
                visibility=vis,
                threshold=self.threshold,
                normalize_by=self.normalize_by
            )
            self.total_correct += correct
            self.total_visible += visible
            if cat_id not in self.category_correct:
                self.category_correct[cat_id] = 0
                self.category_visible[cat_id] = 0
            self.category_correct[cat_id] += correct
            self.category_visible[cat_id] += visible
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
            Dictionary with pck_overall, pck_per_category, mean_pck_categories,
            total_correct, total_visible, num_categories, num_images, threshold
        """
        pck_overall = self.total_correct / self.total_visible if self.total_visible > 0 else 0.0
        pck_per_category = {}
        for cat_id in self.category_correct.keys():
            correct = self.category_correct[cat_id]
            visible = self.category_visible[cat_id]
            pck_per_category[cat_id] = correct / visible if visible > 0 else 0.0
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
        """
        Reset all statistics.
        """
        self.total_correct = 0
        self.total_visible = 0
        self.category_correct = {}
        self.category_visible = {}
        self.image_results = []