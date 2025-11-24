"""
Training and evaluation engine for CAPE (Category-Agnostic Pose Estimation)

This module handles episodic training where each batch contains:
- Support images + pose graphs
- Query images to predict keypoints on
- Ground truth keypoint sequences

Key differences from standard engine.py:
- Handles support graph conditioning
- Processes episodic batches (support + queries)
- Computes CAPE-specific losses
- Evaluates with PCK@bbox metric
"""

import math
import sys
import torch
import numpy as np
import util.misc as utils
from typing import Iterable, Optional, Dict
from util.eval_utils import PCKEvaluator, compute_pck_bbox


def train_one_epoch_episodic(model: torch.nn.Module, criterion: torch.nn.Module,
                             data_loader: Iterable, optimizer: torch.optim.Optimizer,
                             device: torch.device, epoch: int, max_norm: float = 0,
                             print_freq: int = 10):
    """
    Train one epoch with episodic sampling.

    Each iteration processes one batch of episodes where each episode contains:
    - 1 support example (provides pose graph)
    - K query examples (predict keypoints using support graph)

    Args:
        model: CAPE model
        criterion: Loss function
        data_loader: Episodic dataloader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        max_norm: Gradient clipping max norm
        print_freq: Print frequency

    Returns:
        stats: Dictionary of training statistics
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # Note: class_error meter will be added dynamically if present in loss_dict
    header = f'Epoch: [{epoch}]'

    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move batch to device
        support_images = batch['support_images'].to(device)  # (B, C, H, W)
        support_coords = batch['support_coords'].to(device)  # (B, N, 2)
        support_masks = batch['support_masks'].to(device)    # (B, N)
        query_images = batch['query_images'].to(device)      # (B*Q, C, H, W)

        # Skeleton edges (list, not moved to device - used in adjacency matrix construction)
        support_skeletons = batch.get('support_skeletons', None)

        # Query targets (seq_data)
        query_targets = {}
        for key, value in batch['query_targets'].items():
            query_targets[key] = value.to(device)

        # Forward pass
        # Model expects query images and support conditioning (including skeleton edges)
        outputs = model(
            samples=query_images,
            support_coords=support_coords,
            support_mask=support_masks,
            targets=query_targets,
            skeleton_edges=support_skeletons
        )

        # Compute loss
        loss_dict = criterion(outputs, query_targets)

        # Weight losses
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Reduce losses over all GPUs for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                   for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update metrics
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def extract_keypoints_from_sequence(
    pred_coords: torch.Tensor,
    token_labels: torch.Tensor,
    mask: torch.Tensor,
    max_keypoints: Optional[int] = None
) -> torch.Tensor:
    """
    Extract predicted keypoint coordinates from autoregressive sequence.
    
    The model outputs a sequence with special tokens (<coord>, <sep>, <eos>).
    This function extracts only the coordinate predictions.
    
    Args:
        pred_coords: Predicted coordinates, shape (B, seq_len, 2)
        token_labels: Token type labels, shape (B, seq_len)
                     0 = <coord>, 1 = <sep>, 2 = <eos>
        mask: Valid token mask, shape (B, seq_len)
        max_keypoints: Maximum number of keypoints to extract (or None for all)
    
    Returns:
        keypoints: Extracted keypoints, shape (B, N, 2) where N = num keypoints
                  Padded with zeros if instances have different numbers of keypoints
    """
    from datasets.token_types import TokenType
    
    batch_size = pred_coords.shape[0]
    all_keypoints = []
    
    for i in range(batch_size):
        # Get valid tokens for this instance
        valid_mask = mask[i]
        valid_coords = pred_coords[i][valid_mask]
        valid_labels = token_labels[i][valid_mask]
        
        # Extract only coordinate tokens (TokenType.coord = 0)
        coord_mask = valid_labels == TokenType.coord.value
        kpts = valid_coords[coord_mask]
        
        # Limit to max_keypoints if specified
        if max_keypoints is not None and len(kpts) > max_keypoints:
            kpts = kpts[:max_keypoints]
        
        all_keypoints.append(kpts)
    
    # Pad to same length
    if len(all_keypoints) > 0:
        max_len = max(len(kpts) for kpts in all_keypoints)
        padded_keypoints = []
        
        for kpts in all_keypoints:
            if len(kpts) < max_len:
                padding = torch.zeros(max_len - len(kpts), 2, device=kpts.device)
                kpts = torch.cat([kpts, padding], dim=0)
            padded_keypoints.append(kpts)
        
        return torch.stack(padded_keypoints)
    else:
        return torch.zeros(batch_size, 0, 2, device=pred_coords.device)


@torch.no_grad()
def evaluate_cape(model, criterion, data_loader, device, compute_pck=True, pck_threshold=0.2):
    """
    Evaluate CAPE model on episodic validation set.

    Args:
        model: CAPE model
        criterion: Loss function
        data_loader: Episodic validation dataloader
        device: Device
        compute_pck: Whether to compute PCK metric (default: True)
        pck_threshold: PCK threshold (default: 0.2)

    Returns:
        stats: Dictionary of evaluation statistics including PCK
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    
    # PCK evaluator (if enabled)
    pck_evaluator = PCKEvaluator(threshold=pck_threshold) if compute_pck else None

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Move batch to device
        support_images = batch['support_images'].to(device)
        support_coords = batch['support_coords'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_images = batch['query_images'].to(device)

        # Skeleton edges
        support_skeletons = batch.get('support_skeletons', None)

        query_targets = {}
        for key, value in batch['query_targets'].items():
            query_targets[key] = value.to(device)

        # Forward pass
        outputs = model(
            samples=query_images,
            support_coords=support_coords,
            support_mask=support_masks,
            targets=query_targets,
            skeleton_edges=support_skeletons
        )

        # Compute loss
        loss_dict = criterion(outputs, query_targets)

        # Weight losses
        weight_dict = criterion.weight_dict

        # Reduce losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                   for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}

        # Update metrics
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                           **loss_dict_reduced_scaled,
                           **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        # Compute PCK if enabled
        if compute_pck and pck_evaluator is not None:
            # Extract predicted coordinates
            pred_coords = outputs.get('pred_coords', None)  # (B, seq_len, 2)
            
            if pred_coords is not None:
                # Extract ground truth keypoints from targets
                gt_coords = query_targets.get('target_seq', None)  # (B, seq_len, 2)
                token_labels = query_targets.get('token_labels', None)  # (B, seq_len)
                mask = query_targets.get('mask', None)  # (B, seq_len)
                
                if gt_coords is not None and token_labels is not None:
                    # Extract only coordinate tokens (filter out special tokens)
                    pred_kpts = extract_keypoints_from_sequence(
                        pred_coords, token_labels, mask
                    )
                    gt_kpts = extract_keypoints_from_sequence(
                        gt_coords, token_labels, mask
                    )
                    
                    # Get bbox dimensions from query_metadata
                    # Note: bbox info needs to be passed through from dataloader
                    # For now, assume normalized coordinates and bbox is [512, 512]
                    # TODO: Pass actual bbox dimensions through batch
                    batch_size = pred_kpts.shape[0]
                    bbox_widths = torch.full((batch_size,), 512.0, device=device)
                    bbox_heights = torch.full((batch_size,), 512.0, device=device)
                    
                    # Add to PCK evaluator
                    pck_evaluator.add_batch(
                        pred_keypoints=pred_kpts,
                        gt_keypoints=gt_kpts,
                        bbox_widths=bbox_widths,
                        bbox_heights=bbox_heights,
                        category_ids=batch.get('category_ids', None),
                        visibility=None  # TODO: Pass visibility from metadata
                    )

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Averaged validation stats: {metric_logger}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Add PCK stats if computed
    if compute_pck and pck_evaluator is not None:
        pck_results = pck_evaluator.get_results()
        stats['pck'] = pck_results['pck_overall']
        stats['pck_mean_categories'] = pck_results['mean_pck_categories']
        stats['pck_num_correct'] = pck_results['total_correct']
        stats['pck_num_visible'] = pck_results['total_visible']
        
        print(f"\nPCK@{pck_threshold}: {pck_results['pck_overall']:.2%} "
              f"({pck_results['total_correct']}/{pck_results['total_visible']} keypoints)")

    return stats


@torch.no_grad()
def evaluate_unseen_categories(
    model, 
    data_loader, 
    device, 
    pck_threshold=0.2,
    category_names=None,
    verbose=True
):
    """
    Evaluate CAPE on unseen test categories using PCK@bbox metric.

    This is the KEY evaluation for category-agnostic pose estimation:
    testing generalization to categories never seen during training.

    Args:
        model: CAPE model
        data_loader: Episodic dataloader with test categories (unseen)
        device: Device
        pck_threshold: PCK threshold (default: 0.2 for PCK@0.2)
        category_names: Optional dict mapping category IDs to names
        verbose: Print detailed per-category results

    Returns:
        results: Dictionary with:
            - pck_overall: Overall PCK across all unseen categories
            - pck_mean_categories: Mean PCK across categories (macro-average)
            - pck_per_category: Dict of category_id -> PCK
            - total_correct: Total correct keypoints
            - total_visible: Total visible keypoints
            - num_categories: Number of unseen categories evaluated
    """
    model.eval()

    # PCK evaluator
    pck_evaluator = PCKEvaluator(threshold=pck_threshold)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test (Unseen Categories) - PCK@{pck_threshold}:'

    num_batches = 0
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        num_batches += 1
        
        # Move batch to device
        support_images = batch['support_images'].to(device)
        support_coords = batch['support_coords'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_images = batch['query_images'].to(device)
        support_skeletons = batch.get('support_skeletons', None)

        # Get category IDs and metadata
        category_ids = batch.get('category_ids', None)
        query_metadata = batch.get('query_metadata', None)
        
        # Get ground truth targets
        query_targets = {}
        for key, value in batch['query_targets'].items():
            query_targets[key] = value.to(device)

        # Forward pass (inference mode - NO teacher forcing)
        # Use forward_inference for autoregressive generation
        try:
            if hasattr(model, 'module'):
                predictions = model.module.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks
                )
            else:
                predictions = model.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks
                )
        except AttributeError:
            # Fallback: use regular forward but extract from outputs
            outputs = model(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                targets=query_targets,
                skeleton_edges=support_skeletons
            )
            predictions = {
                'coordinates': outputs.get('pred_coords', None),
                'logits': outputs.get('pred_logits', None)
            }

        # Extract predicted and ground truth keypoints
        pred_coords = predictions.get('coordinates', None)  # (B, seq_len, 2)
        
        if pred_coords is not None:
            # Extract ground truth
            gt_coords = query_targets.get('target_seq', None)  # (B, seq_len, 2)
            token_labels = query_targets.get('token_labels', None)
            mask = query_targets.get('mask', None)
            
            if gt_coords is not None and token_labels is not None:
                # Extract keypoints (filter special tokens)
                pred_kpts = extract_keypoints_from_sequence(
                    pred_coords, token_labels, mask
                )
                gt_kpts = extract_keypoints_from_sequence(
                    gt_coords, token_labels, mask
                )
                
                # Get bbox dimensions for normalization
                # Extract from query_metadata if available
                batch_size = pred_kpts.shape[0]
                
                if query_metadata is not None and len(query_metadata) > 0:
                    # Extract bbox dimensions from metadata
                    bbox_widths = []
                    bbox_heights = []
                    visibility_list = []
                    
                    for meta in query_metadata:
                        bbox_w = meta.get('bbox_width', 512.0)
                        bbox_h = meta.get('bbox_height', 512.0)
                        bbox_widths.append(bbox_w)
                        bbox_heights.append(bbox_h)
                        
                        # Get visibility if available
                        vis = meta.get('visibility', None)
                        if vis is not None:
                            visibility_list.append(vis)
                    
                    bbox_widths = torch.tensor(bbox_widths, device=device)
                    bbox_heights = torch.tensor(bbox_heights, device=device)
                    
                    # Convert visibility to tensor if available
                    if len(visibility_list) > 0:
                        # Pad visibility to match max keypoints
                        max_kpts = max(len(v) for v in visibility_list)
                        visibility_padded = []
                        for vis in visibility_list:
                            if len(vis) < max_kpts:
                                vis = vis + [0] * (max_kpts - len(vis))
                            visibility_padded.append(vis[:max_kpts])
                        visibility = torch.tensor(visibility_padded, device=device)
                    else:
                        visibility = None
                else:
                    # Default: assume 512x512 after resize
                    bbox_widths = torch.full((batch_size,), 512.0, device=device)
                    bbox_heights = torch.full((batch_size,), 512.0, device=device)
                    visibility = None
                
                # Add to PCK evaluator
                pck_evaluator.add_batch(
                    pred_keypoints=pred_kpts,
                    gt_keypoints=gt_kpts,
                    bbox_widths=bbox_widths,
                    bbox_heights=bbox_heights,
                    category_ids=category_ids,
                    visibility=visibility
                )

    # Get final results
    results = pck_evaluator.get_results()
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"UNSEEN CATEGORY EVALUATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Metric: PCK@{pck_threshold}")
    print(f"Number of unseen categories: {results['num_categories']}")
    print(f"Number of images evaluated: {results['num_images']}")
    print(f"\nOverall Results:")
    print(f"  PCK (micro-average): {results['pck_overall']:.2%} "
          f"({results['total_correct']}/{results['total_visible']} keypoints)")
    print(f"  PCK (macro-average): {results['mean_pck_categories']:.2%} "
          f"(mean across {results['num_categories']} categories)")
    
    # Per-category results
    if verbose and results['pck_per_category']:
        print(f"\nPer-Category Results:")
        print(f"{'─' * 80}")
        
        # Sort by category ID
        sorted_categories = sorted(results['pck_per_category'].items())
        
        for cat_id, pck in sorted_categories:
            cat_name = category_names.get(cat_id, f"Category {cat_id}") if category_names else f"Category {cat_id}"
            print(f"  {cat_name:40s}: PCK = {pck:.2%}")
    
    print(f"{'=' * 80}\n")
    
    return results


if __name__ == '__main__':
    print("CAPE training engine")
    print("Functions:")
    print("  - train_one_epoch_episodic: Episodic meta-learning training")
    print("  - evaluate_cape: Validation on training categories")
    print("  - evaluate_unseen_categories: Test on unseen categories (key CAPE evaluation)")
