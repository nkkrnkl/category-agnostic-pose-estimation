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
from tqdm import tqdm


def train_one_epoch_episodic(model: torch.nn.Module, criterion: torch.nn.Module,
                             data_loader: Iterable, optimizer: torch.optim.Optimizer,
                             device: torch.device, epoch: int, max_norm: float = 0,
                             print_freq: int = 10, accumulation_steps: int = 1):
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
        accumulation_steps: Number of mini-batches to accumulate gradients over (default: 1, no accumulation)

    Returns:
        stats: Dictionary of training statistics
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # Note: class_error meter will be added dynamically if present in loss_dict
    header = f'Epoch: [{epoch}]'
    
    # Initialize gradients to zero at start of epoch
    optimizer.zero_grad()

    # Wrap data_loader with tqdm for progress bar
    # mininterval: update at most once per 2 seconds
    # miniters: update at least every 20 iterations
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', leave=True, ncols=100, 
                mininterval=2.0, miniters=20)
    
    for batch_idx, batch in enumerate(pbar):
        # ========================================================================
        # Move batch to device
        # ========================================================================
        # NOTE: After episodic_collate_fn fix #1, support tensors are REPEATED
        # to match query batch size. Each support is repeated K times (once per
        # query in that episode), ensuring support[i] pairs with query[i].
        #
        # Shapes (where B=num_episodes, K=queries_per_episode):
        #   - All tensors have first dimension (B*K) for proper alignment
        # ========================================================================
        support_images = batch['support_images'].to(device)  # (B*K, C, H, W) - repeated!
        support_coords = batch['support_coords'].to(device)  # (B*K, N, 2) - repeated!
        support_masks = batch['support_masks'].to(device)    # (B*K, N) - repeated!
        query_images = batch['query_images'].to(device)      # (B*K, C, H, W)

        # Skeleton edges (list of length B*K, not moved to device)
        # Used in adjacency matrix construction for graph-based encoding
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

        # ========================================================================
        # CRITICAL FIX: Gradient Accumulation for Small Batch Sizes
        # ========================================================================
        # Problem: Small batch sizes (e.g., 2 episodes) lead to noisy gradients
        #
        # Solution: Accumulate gradients over multiple mini-batches before updating
        #   - Divide loss by accumulation_steps (average across mini-batches)
        #   - Call optimizer.step() only every accumulation_steps iterations
        #   - Effective batch size = batch_size * accumulation_steps
        #
        # Example with batch_size=2, accumulation_steps=4:
        #   - Physical batch size: 2 episodes
        #   - Effective batch size: 8 episodes (accumulated gradients)
        #   - Memory: Same as batch_size=2 (no extra memory!)
        #   - Gradient quality: Same as batch_size=8
        # ========================================================================
        
        # Normalize loss by accumulation steps
        # This ensures gradient magnitudes are consistent regardless of accumulation
        normalized_loss = losses / accumulation_steps
        
        # Backward pass (accumulate gradients)
        normalized_loss.backward()
        
        # Only update weights every accumulation_steps iterations
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (applied to accumulated gradients)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # Update weights
            optimizer.step()
            
            # Clear accumulated gradients
            optimizer.zero_grad()

        # Update metrics
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Update tqdm progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'loss_ce': f'{loss_dict_reduced_scaled.get("loss_ce", 0):.4f}',
            'loss_coords': f'{loss_dict_reduced_scaled.get("loss_coords", 0):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    # ========================================================================
    # Handle remaining accumulated gradients (if epoch doesn't end on accumulation boundary)
    # ========================================================================
    # If the total number of batches is not divisible by accumulation_steps,
    # we need to perform a final update with the remaining accumulated gradients.
    # ========================================================================
    total_batches = batch_idx + 1
    if total_batches % accumulation_steps != 0:
        print(f"  → Performing final gradient update with remaining {total_batches % accumulation_steps} accumulated batches")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

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

    # Wrap data_loader with tqdm for progress bar
    # mininterval: update at most once per 2 seconds
    # miniters: update at least every 10 iterations
    pbar = tqdm(data_loader, desc='Validation', leave=True, ncols=100,
                mininterval=2.0, miniters=10)
    
    for batch in pbar:
        # ========================================================================
        # Move batch to device
        # ========================================================================
        # NOTE: After episodic_collate_fn fix #1, all tensors have matching
        # batch size (B*K) where each support is repeated K times to align
        # with its corresponding queries.
        # ========================================================================
        support_images = batch['support_images'].to(device)  # (B*K, C, H, W)
        support_coords = batch['support_coords'].to(device)  # (B*K, N, 2)
        support_masks = batch['support_masks'].to(device)    # (B*K, N)
        query_images = batch['query_images'].to(device)      # (B*K, C, H, W)

        # Skeleton edges (list of length B*K)
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
        val_loss = sum(loss_dict_reduced_scaled.values())
        metric_logger.update(loss=val_loss,
                           **loss_dict_reduced_scaled,
                           **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        # Update tqdm progress bar
        pbar.set_postfix({
            'loss': f'{val_loss:.4f}',
            'loss_ce': f'{loss_dict_reduced_scaled.get("loss_ce", 0):.4f}',
            'loss_coords': f'{loss_dict_reduced_scaled.get("loss_coords", 0):.4f}'
        })
        
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
                    
                    # ================================================================
                    # CRITICAL FIX: Use actual bbox dimensions from query_metadata
                    # ================================================================
                    # PCK@bbox requires normalization by the original bbox diagonal.
                    # Previously used dummy 512x512, now using actual bbox dimensions.
                    # ================================================================
                    
                    batch_size = pred_kpts.shape[0]
                    query_metadata = batch.get('query_metadata', None)
                    
                    if query_metadata is not None and len(query_metadata) > 0:
                        # ================================================================
                        # CRITICAL FIX: Handle variable-length keypoint sequences
                        # ================================================================
                        # Different categories have different numbers of keypoints!
                        # E.g., "beaver_body" has 17, "przewalskihorse_face" has 9
                        # 
                        # The model outputs a fixed sequence length (max keypoints seen)
                        # but we need to trim predictions to match each category's actual
                        # number of keypoints for PCK evaluation.
                        # ================================================================
                        
                        # Extract actual bbox dimensions from metadata
                        bbox_widths = []
                        bbox_heights = []
                        visibility_list = []
                        pred_kpts_trimmed = []
                        gt_kpts_trimmed = []
                        
                        for idx, meta in enumerate(query_metadata):
                            bbox_w = meta.get('bbox_width', 512.0)
                            bbox_h = meta.get('bbox_height', 512.0)
                            bbox_widths.append(bbox_w)
                            bbox_heights.append(bbox_h)
                            
                            vis = meta.get('visibility', [])
                            num_kpts_for_category = len(vis)  # Actual number of keypoints for this category
                            
                            # Trim predictions and ground truth to match this category's keypoint count
                            # This handles variable-length sequences across different MP-100 categories
                            pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])  # (num_kpts, 2)
                            gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])      # (num_kpts, 2)
                            
                            visibility_list.append(vis)
                        
                        # Note: We DON'T stack pred_kpts_trimmed / gt_kpts_trimmed because they may have
                        # different lengths. Instead, we'll pass them as lists to add_batch,
                        # which processes each sample individually anyway.
                        
                        bbox_widths = torch.tensor(bbox_widths, device=device)
                        bbox_heights = torch.tensor(bbox_heights, device=device)
                    else:
                        # Fallback to 512x512 if metadata not available
                        print(f"⚠️  WARNING: query_metadata not available, falling back to 512x512 bbox (batch_size={batch_size})")
                        bbox_widths = torch.full((batch_size,), 512.0, device=device)
                        bbox_heights = torch.full((batch_size,), 512.0, device=device)
                        visibility_list = None
                        pred_kpts_trimmed = pred_kpts  # Use original predictions (no trimming)
                        gt_kpts_trimmed = gt_kpts
                    
                    # Add to PCK evaluator with actual bbox dimensions and visibility
                    # Note: pred_kpts_trimmed and gt_kpts_trimmed are LISTS of tensors
                    # with potentially different lengths (per category)
                    pck_evaluator.add_batch(
                        pred_keypoints=pred_kpts_trimmed,
                        gt_keypoints=gt_kpts_trimmed,
                        bbox_widths=bbox_widths,
                        bbox_heights=bbox_heights,
                        category_ids=batch.get('category_ids', None),
                        visibility=visibility_list  # Now includes actual visibility from metadata
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
    
    # Wrap data_loader with tqdm for progress bar
    # mininterval: update at most once per 2 seconds
    # miniters: update at least every 10 iterations
    pbar = tqdm(data_loader, desc=f'Test (Unseen) PCK@{pck_threshold}', leave=True, ncols=100,
                mininterval=2.0, miniters=10)
    
    for batch in pbar:
        num_batches += 1
        
        # ========================================================================
        # Move batch to device
        # ========================================================================
        # NOTE: All tensors have matching batch size (B*K) after collate_fn fix
        # ========================================================================
        support_images = batch['support_images'].to(device)  # (B*K, C, H, W)
        support_coords = batch['support_coords'].to(device)  # (B*K, N, 2)
        support_masks = batch['support_masks'].to(device)    # (B*K, N)
        query_images = batch['query_images'].to(device)      # (B*K, C, H, W)
        support_skeletons = batch.get('support_skeletons', None)  # List[B*K]

        # Get category IDs and metadata
        category_ids = batch.get('category_ids', None)
        query_metadata = batch.get('query_metadata', None)
        
        # Get ground truth targets
        query_targets = {}
        for key, value in batch['query_targets'].items():
            query_targets[key] = value.to(device)

        # Forward pass (inference mode - NO teacher forcing)
        # Use forward_inference for autoregressive generation
        # ================================================================
        # CRITICAL FIX: Pass skeleton_edges to enable structural encoding
        # ================================================================
        try:
            if hasattr(model, 'module'):
                predictions = model.module.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks,
                    skeleton_edges=support_skeletons  # Now includes skeleton structure!
                )
            else:
                predictions = model.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks,
                    skeleton_edges=support_skeletons  # Now includes skeleton structure!
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
                
                # Update tqdm with current PCK
                current_results = pck_evaluator.get_results()
                pbar.set_postfix({
                    'PCK': f'{current_results["pck_overall"]:.2%}',
                    'correct': current_results['total_correct'],
                    'visible': current_results['total_visible']
                })

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
