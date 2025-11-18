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
"""

import math
import sys
import torch
import util.misc as utils
from typing import Iterable


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


@torch.no_grad()
def evaluate_cape(model, criterion, data_loader, device):
    """
    Evaluate CAPE model on episodic validation set.

    Args:
        model: CAPE model
        criterion: Loss function
        data_loader: Episodic validation dataloader
        device: Device

    Returns:
        stats: Dictionary of evaluation statistics
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # Note: class_error meter will be added dynamically if present in loss_dict
    header = 'Validation:'

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

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Averaged validation stats: {metric_logger}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate_unseen_categories(model, data_loader, device, category_names=None):
    """
    Evaluate CAPE on unseen test categories.

    This is the key evaluation for category-agnostic pose estimation:
    testing generalization to categories never seen during training.

    Args:
        model: CAPE model
        data_loader: Episodic dataloader with test categories
        device: Device
        category_names: Optional mapping of category IDs to names

    Returns:
        results: Dictionary with per-category and overall metrics
    """
    model.eval()

    # Metrics per category
    category_metrics = {}

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test (Unseen Categories):'

    all_predictions = []
    all_ground_truths = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        support_images = batch['support_images'].to(device)
        support_coords = batch['support_coords'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_images = batch['query_images'].to(device)

        # Get category IDs
        category_ids = batch['category_ids']

        # Forward pass (inference mode)
        predictions = model.module.forward_inference(
            samples=query_images,
            support_coords=support_coords,
            support_mask=support_masks
        ) if hasattr(model, 'module') else model.forward_inference(
            samples=query_images,
            support_coords=support_coords,
            support_mask=support_masks
        )

        # Decode predictions and compute metrics
        # This would need to be implemented based on your evaluation protocol
        # For now, just collect predictions

        pred_coords = predictions['coordinates']  # (B*Q, seq_len, 2)

        # Store for later analysis
        for i, cat_id in enumerate(category_ids):
            cat_id = cat_id.item()
            if cat_id not in category_metrics:
                category_metrics[cat_id] = {
                    'predictions': [],
                    'ground_truths': []
                }

            # Would need ground truth here for proper evaluation
            # category_metrics[cat_id]['predictions'].append(pred_coords[i])

    print(f"\nEvaluated on {len(category_metrics)} unseen categories")

    # Compute PCK and other metrics
    # This requires ground truth keypoints and proper PCK computation
    # Placeholder for now

    results = {
        'num_categories': len(category_metrics),
        'category_metrics': category_metrics,
    }

    return results


if __name__ == '__main__':
    print("CAPE training engine")
    print("Functions:")
    print("  - train_one_epoch_episodic: Episodic meta-learning training")
    print("  - evaluate_cape: Validation on training categories")
    print("  - evaluate_unseen_categories: Test on unseen categories (key CAPE evaluation)")
