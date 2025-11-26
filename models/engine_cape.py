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
import os
import torch
import numpy as np
import util.misc as utils
from typing import Iterable, Optional, Dict
from util.eval_utils import PCKEvaluator, compute_pck_bbox
from tqdm import tqdm

# ============================================================================
# DEBUG MODE: Enable detailed logging with environment variable
# ============================================================================
# Set DEBUG_CAPE=1 to enable detailed logging:
#   export DEBUG_CAPE=1
#   python train_cape_episodic.py ...
#
# Logs include:
#   - Episode structure (support vs query indices, category IDs)
#   - Tensor shapes at each stage
#   - Source of decoder input sequences
#   - Causal mask dimensions
#   - Autoregressive loop steps
# ============================================================================
DEBUG_CAPE = os.environ.get('DEBUG_CAPE', '0') == '1'

def debug_log(message, force=False):
    """Print debug message if DEBUG_CAPE is enabled or force=True."""
    if DEBUG_CAPE or force:
        print(f"[DEBUG_CAPE] {message}")


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
    # ncols=None auto-detects terminal width
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', leave=True, ncols=None, 
                mininterval=2.0, miniters=20)
    
    for batch_idx, batch in enumerate(pbar):
        # ========================================================================
        # DEBUG: Log episode structure on first batch of first epoch
        # ========================================================================
        if DEBUG_CAPE and batch_idx == 0 and epoch == 0:
            debug_log("=" * 80)
            debug_log("TRAINING EPISODE STRUCTURE (First Batch)")
            debug_log("=" * 80)
            debug_log(f"Batch contains {len(batch.get('category_ids', []))} total queries")
            if 'category_ids' in batch:
                unique_cats = torch.unique(batch['category_ids']).cpu().numpy()
                debug_log(f"Categories in batch: {unique_cats}")
        
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
        
        # ========================================================================
        # DEBUG: Log tensor shapes and verify query targets come from queries
        # ========================================================================
        if DEBUG_CAPE and batch_idx == 0 and epoch == 0:
            debug_log("\nTensor Shapes:")
            debug_log(f"  support_coords:  {support_coords.shape}")
            debug_log(f"  support_masks:   {support_masks.shape}")
            debug_log(f"  query_images:    {query_images.shape}")
            debug_log(f"  query_targets keys: {list(query_targets.keys())}")
            if 'target_seq' in query_targets:
                debug_log(f"  query_targets['target_seq']: {query_targets['target_seq'].shape}")
            debug_log(f"  skeleton_edges:  List of {len(support_skeletons) if support_skeletons else 0} edge lists")
            
            # Verify targets are different from support
            if 'target_seq' in query_targets:
                # Compare first query target with first support coords
                query_seq = query_targets['target_seq'][0, :support_coords.shape[1], :]
                support_seq = support_coords[0, :, :]
                are_different = not torch.allclose(query_seq, support_seq, atol=1e-4)
                debug_log(f"\n✓ VERIFICATION: Query targets ≠ Support coords: {are_different}")
                if not are_different:
                    debug_log("  ⚠️  WARNING: Query targets match support! This may indicate a bug.")
            debug_log("=" * 80 + "\n")

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
        
        # DEBUG
        DEBUG_EXTRACT = os.environ.get('DEBUG_EXTRACT', '0') == '1'
        if DEBUG_EXTRACT and i < 2:  # Show first 2 samples
            print(f"\n[DEBUG extract_keypoints_from_sequence] Sample {i}:")
            print(f"  pred_coords[i] shape: {pred_coords[i].shape}")
            print(f"  valid_mask shape: {valid_mask.shape}")
            print(f"  valid_mask sum: {valid_mask.sum().item()}")
            print(f"  valid_mask dtype: {valid_mask.dtype}")
        
        try:
            valid_coords = pred_coords[i][valid_mask]
            valid_labels = token_labels[i][valid_mask]
        except Exception as e:
            print(f"\n❌ ERROR on line 'valid_coords = pred_coords[i][valid_mask]':")
            print(f"   Sample index i={i}")
            print(f"   pred_coords[i].shape = {pred_coords[i].shape}")
            print(f"   valid_mask.shape = {valid_mask.shape}")
            print(f"   valid_mask.dtype = {valid_mask.dtype}")
            print(f"   Error: {e}")
            raise
        
        if DEBUG_EXTRACT and i < 2:
            print(f"  After boolean indexing:")
            print(f"    valid_coords shape: {valid_coords.shape}")
            print(f"    valid_labels shape: {valid_labels.shape}")
        
        # Extract only coordinate tokens (TokenType.coord = 0)
        coord_mask = valid_labels == TokenType.coord.value
        kpts = valid_coords[coord_mask]
        
        if DEBUG_EXTRACT and i < 2:
            print(f"  After coord filtering:")
            print(f"    coord_mask shape: {coord_mask.shape}")
            print(f"    coord_mask sum: {coord_mask.sum().item()}")
            print(f"    kpts shape: {kpts.shape}")
        
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
    Evaluate CAPE model on episodic validation set (UNSEEN categories).
    
    CRITICAL: Uses autoregressive inference (forward_inference) to properly
    test the model's ability to predict keypoints without ground truth.
    This is the TRUE test of category-agnostic generalization.

    Args:
        model: CAPE model
        criterion: Loss function (used only if inference produces logits)
        data_loader: Episodic validation dataloader (using split='val' for unseen categories)
        device: Device
        compute_pck: Whether to compute PCK metric (default: True)
        pck_threshold: PCK threshold (default: 0.2)

    Returns:
        stats: Dictionary of evaluation statistics including PCK
    """
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation (Unseen Categories):'
    
    # PCK evaluator (if enabled)
    pck_evaluator = PCKEvaluator(threshold=pck_threshold) if compute_pck else None

    # Wrap data_loader with tqdm for progress bar
    # mininterval: update at most once per 2 seconds
    # miniters: update at least every 10 iterations
    pbar = tqdm(data_loader, desc='Validation (Autoregressive)', leave=True, ncols=None,
                mininterval=2.0, miniters=10)
    
    for batch_idx, batch in enumerate(pbar):
        # ========================================================================
        # DEBUG: Check for data leakage (support == query)
        # ========================================================================
        DEBUG_VAL = os.environ.get('DEBUG_CAPE', '0') == '1'
        if DEBUG_VAL and batch_idx == 0:
            support_meta = batch.get('support_metadata', [])
            query_meta = batch.get('query_metadata', [])
            if support_meta and query_meta:
                print(f"\n🔍 DEBUG BATCH DATA (Batch 0):")
                print(f"  Support image IDs: {[m.get('image_id', 'N/A') for m in support_meta[:3]]}")
                print(f"  Query image IDs: {[m.get('image_id', 'N/A') for m in query_meta[:3]]}")
                
                # Check for same image IDs
                support_ids = set(m.get('image_id', None) for m in support_meta)
                query_ids = set(m.get('image_id', None) for m in query_meta)
                overlap = support_ids & query_ids
                if overlap:
                    print(f"  ⚠️  WARNING: {len(overlap)} images appear in BOTH support and query!")
                    print(f"      Overlapping IDs: {list(overlap)[:5]}")
        # ========================================================================
        
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

        # ========================================================================
        # CRITICAL FIX: Use forward_inference for REAL validation
        # ========================================================================
        # Previously: Used teacher forcing (passed targets to model forward)
        # Problem: PCK@100% because model saw ground truth during inference!
        # Solution: Use autoregressive inference (forward_inference) like test
        # ========================================================================
        
        # ========================================================================
        # DEBUG: Track which inference path is used
        # ========================================================================
        DEBUG_VAL = os.environ.get('DEBUG_CAPE', '0') == '1'
        inference_method_used = None
        
        # ========================================================================
        # CRITICAL: Use ONLY autoregressive inference (NO fallback to teacher forcing)
        # ========================================================================
        # Previous bug: AttributeError was silently caught and fell back to teacher
        # forcing, giving PCK@100%. Now we require forward_inference to exist.
        # If it fails, we raise an error instead of silently cheating.
        # ========================================================================
        
        # Check that forward_inference is available
        if not hasattr(model, 'forward_inference'):
            if hasattr(model, 'module') and hasattr(model.module, 'forward_inference'):
                # DDP model
                model_for_inference = model.module
            else:
                raise RuntimeError(
                    "Model does not have forward_inference method!\n"
                    "Cannot run proper validation without autoregressive inference.\n"
                    "Check that the model was built correctly with a tokenizer."
                )
        else:
            model_for_inference = model
        
        # Run autoregressive inference
        predictions = model_for_inference.forward_inference(
            samples=query_images,
            support_coords=support_coords,
            support_mask=support_masks,
            skeleton_edges=support_skeletons
        )
        
        if DEBUG_VAL and batch_idx == 0:
            print(f"\n🔍 DEBUG VALIDATION (Batch 0):")
            print(f"  ✓ Using: forward_inference (autoregressive)")
            print(f"  Query images shape: {query_images.shape}")
            print(f"  Support coords shape: {support_coords.shape}")
            print(f"  Predictions shape: {predictions.get('coordinates', torch.zeros(1)).shape}")
        
        # ========================================================================
        # CRITICAL FIX: Pad autoregressive predictions to match target length
        # ========================================================================
        # Problem: Autoregressive inference generates variable-length sequences
        #          (e.g., 18 tokens when EOS predicted early), but targets have
        #          fixed length (200 tokens). This causes shape mismatch in loss.
        #
        # Solution: Pad predictions to target length before computing loss.
        #          The visibility mask will ensure padding doesn't affect loss.
        # ========================================================================
        pred_logits = predictions.get('logits', None)
        pred_coords = predictions.get('coordinates', None)
        
        if pred_logits is not None and pred_coords is not None:
            batch_size, pred_seq_len = pred_logits.shape[:2]
            target_seq_len = query_targets['target_seq'].shape[1]
            
            if pred_seq_len < target_seq_len:
                # Pad predictions to match target length
                pad_len = target_seq_len - pred_seq_len
                
                # Pad logits with padding token logits (all zeros except padding class)
                vocab_size = pred_logits.shape[-1]
                pad_logits = torch.zeros(
                    batch_size, pad_len, vocab_size,
                    dtype=pred_logits.dtype,
                    device=pred_logits.device
                )
                # Set padding class to high value (if we know the padding token index)
                # For now, leave as zeros (the mask will exclude these from loss anyway)
                
                pred_logits_padded = torch.cat([pred_logits, pad_logits], dim=1)
                
                # Pad coordinates with zeros
                pad_coords = torch.zeros(
                    batch_size, pad_len, 2,
                    dtype=pred_coords.dtype,
                    device=pred_coords.device
                )
                pred_coords_padded = torch.cat([pred_coords, pad_coords], dim=1)
            else:
                # No padding needed (or predictions longer than targets - shouldn't happen)
                pred_logits_padded = pred_logits[:, :target_seq_len]
                pred_coords_padded = pred_coords[:, :target_seq_len]
        else:
            pred_logits_padded = pred_logits
            pred_coords_padded = pred_coords
        
        # Convert predictions to outputs format for loss computation
        outputs = {
            'pred_coords': pred_coords_padded,
            'pred_logits': pred_logits_padded
        }
        
        # Compute loss if criterion and logits available
        loss_dict = {}
        if criterion is not None and outputs.get('pred_logits') is not None:
            loss_dict = criterion(outputs, query_targets)

        # Weight losses (if loss was computed)
        if loss_dict and criterion is not None:
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
        else:
            # No loss computed (inference only)
            val_loss = 0.0
            loss_dict_reduced_scaled = {}
        
        # Update tqdm progress bar with current metrics
        if compute_pck and pck_evaluator is not None:
            current_results = pck_evaluator.get_results()
            pbar.set_postfix({
                'PCK': f'{current_results["pck_overall"]:.2%}',
                'loss': f'{val_loss:.4f}',
                'correct': current_results['total_correct'],
                'visible': current_results['total_visible']
            })
        else:
            pbar.set_postfix({
                'loss': f'{val_loss:.4f}',
                'loss_ce': f'{loss_dict_reduced_scaled.get("loss_ce", 0):.4f}',
                'loss_coords': f'{loss_dict_reduced_scaled.get("loss_coords", 0):.4f}'
            })
        
        # Compute PCK if enabled
        if compute_pck and pck_evaluator is not None:
            # Extract predicted coordinates from inference
            pred_coords = predictions.get('coordinates', None)  # (B, seq_len, 2)
            
            if pred_coords is not None:
                # Extract ground truth keypoints from targets
                gt_coords = query_targets.get('target_seq', None)  # (B, seq_len, 2)
                token_labels = query_targets.get('token_labels', None)  # (B, seq_len)
                mask = query_targets.get('mask', None)  # (B, seq_len)
                
                if gt_coords is not None and token_labels is not None:
                    # ================================================================
                    # CRITICAL FIX: Extract GT using GT structure, predictions using PREDICTED structure
                    # ================================================================
                    # BUG: Previously used GT token labels for both pred and GT extraction
                    # This assumes model generates EXACTLY the same token sequence as GT
                    # FIX: Use predicted token types from model output for predictions
                    # ================================================================
                    
                    # Extract GT using GT token labels (correct)
                    gt_kpts = extract_keypoints_from_sequence(
                        gt_coords, token_labels, mask
                    )
                    
                    # Extract predictions using PREDICTED token labels (fixed)
                    pred_logits = predictions.get('logits', None)
                    if pred_logits is not None:
                        # Use model's predicted token types
                        from util.sequence_utils import extract_keypoints_from_predictions
                        # ================================================================
                        # CRITICAL FIX: Don't limit max_keypoints during extraction
                        # ================================================================
                        # BUG: max_keypoints=gt_kpts.shape[1] artificially limits extraction
                        #      If model predicts fewer keypoints (e.g., 14 for a 17-kpt category),
                        #      we can't "add" them back during trimming later.
                        # FIX: Extract ALL predicted keypoints, then trim per-category below
                        # ================================================================
                        pred_kpts = extract_keypoints_from_predictions(
                            pred_coords, pred_logits, max_keypoints=None  # Extract all
                        )
                    else:
                        # Fallback if logits not available
                        # Still use GT structure but add WARNING
                        if batch_idx == 0:
                            print("⚠️  WARNING: Using GT token structure for predictions (logits unavailable)")
                            print("   This may cause incorrect PCK if model's token sequence differs from GT")
                        pred_kpts = extract_keypoints_from_sequence(
                            pred_coords, token_labels, mask
                        )
                    
                    # ================================================================
                    # DEBUG: Verify predictions are different from GT
                    # ================================================================
                    DEBUG_PCK = os.environ.get('DEBUG_PCK', '0') == '1'
                    if DEBUG_PCK and batch_idx == 0:
                        print(f"\n[DEBUG_PCK] Batch 0 - Keypoint Extraction:")
                        print(f"  Using predicted token labels: {pred_logits is not None}")
                        print(f"  pred_kpts[0] shape: {pred_kpts[0].shape}")
                        print(f"  gt_kpts[0] shape: {gt_kpts[0].shape}")
                        print(f"  pred_kpts[0, :3]: {pred_kpts[0, :3]}")
                        print(f"  gt_kpts[0, :3]: {gt_kpts[0, :3]}")
                        are_identical = torch.allclose(pred_kpts[0], gt_kpts[0], atol=1e-6)
                        print(f"  Are they identical? {are_identical}")
                        if are_identical:
                            print(f"  ⚠️  CRITICAL: Predictions are IDENTICAL to GT!")
                            print(f"  This indicates data leakage or a bug in the model.")
                    # ================================================================
                    
                    # ================================================================
                    # DEBUG: Check if predictions match support instead of being 
                    # actually predicted (would indicate data leakage or bug)
                    # ================================================================
                    if DEBUG_VAL and batch_idx == 0:
                        print(f"\n🔍 DEBUG PCK COMPUTATION (Batch 0, Sample 0):")
                        print(f"  Predicted coords (first 3): {pred_kpts[0, :3, :].cpu().numpy()}")
                        print(f"  GT coords (first 3): {gt_kpts[0, :3, :].cpu().numpy()}")
                        print(f"  Support coords (first 3): {support_coords[0, :3, :].cpu().numpy()}")
                        
                        # Check if predictions == support (data leakage)
                        pred_vs_support_diff = torch.abs(pred_kpts[0] - support_coords[0][:pred_kpts.shape[1]]).mean().item()
                        pred_vs_gt_diff = torch.abs(pred_kpts[0] - gt_kpts[0]).mean().item()
                        
                        print(f"  Mean diff (pred vs support): {pred_vs_support_diff:.6f}")
                        print(f"  Mean diff (pred vs GT): {pred_vs_gt_diff:.6f}")
                        
                        if pred_vs_support_diff < 0.001:
                            print(f"  ⚠️  WARNING: Predictions == Support (possible data leakage!)")
                        if pred_vs_gt_diff < 0.001:
                            print(f"  ⚠️  WARNING: Predictions == GT (impossible in autoregressive!)")
                    # ================================================================
                    
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
                            
                            # ================================================================
                            # CRITICAL: Handle keypoint count mismatches
                            # ================================================================
                            pred_count = pred_kpts[idx].shape[0]
                            expected_count = num_kpts_for_category
                            
                            if pred_count > expected_count + 10 and batch_idx == 0 and idx == 0:
                                import warnings
                                warnings.warn(
                                    f"⚠️  Model generated {pred_count} keypoints but expected {expected_count}. "
                                    f"Excess: {pred_count - expected_count}. Model likely didn't learn EOS properly."
                                )
                            
                            # ================================================================
                            # CRITICAL FIX: Pad or trim predictions to match GT count
                            # ================================================================
                            # Case 1: Model predicted MORE keypoints than category has
                            #   → Trim to expected count
                            # Case 2: Model predicted FEWER keypoints than category has
                            #   → Pad with zeros (treat as incorrect predictions)
                            # ================================================================
                            if pred_count >= expected_count:
                                # Trim excess predictions
                                pred_kpts_trimmed.append(pred_kpts[idx, :expected_count, :])
                            else:
                                # Pad with zeros if model predicted too few
                                pred_sample = pred_kpts[idx]  # (pred_count, 2)
                                padding = torch.zeros(
                                    expected_count - pred_count, 2,
                                    dtype=pred_sample.dtype,
                                    device=pred_sample.device
                                )
                                pred_padded = torch.cat([pred_sample, padding], dim=0)  # (expected_count, 2)
                                pred_kpts_trimmed.append(pred_padded)
                                
                                if batch_idx == 0 and idx == 0:
                                    import warnings
                                    warnings.warn(
                                        f"⚠️  Model only generated {pred_count}/{expected_count} keypoints. "
                                        f"Padding {expected_count - pred_count} with zeros (will hurt PCK)."
                                    )
                            
                            # Trim GT to match category keypoint count (should already match, but be safe)
                            gt_kpts_trimmed.append(gt_kpts[idx, :expected_count, :])
                            # ================================================================
                            
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
                    
                    # ================================================================
                    # CRITICAL FIX: Scale keypoints to PIXEL space for PCK computation
                    # ================================================================
                    # BUG: Keypoints are in [0,1] normalized space (relative to 512x512 image)
                    #      but bbox dimensions are in PIXELS (original bbox size before resize)
                    # This creates a ~100x scaling error in PCK threshold!
                    # 
                    # FIX: Scale keypoints to pixel space before PCK:
                    #   - Keypoints are in [0,1] relative to 512x512 image
                    #   - Multiply by 512 to get pixel coordinates
                    #   - Then compute PCK using pixel coords and pixel bbox dims
                    # ================================================================
                    pred_kpts_trimmed_pixels = [kpts * 512.0 for kpts in pred_kpts_trimmed]
                    gt_kpts_trimmed_pixels = [kpts * 512.0 for kpts in gt_kpts_trimmed]
                    # ================================================================
                    
                    # Add to PCK evaluator with actual bbox dimensions and visibility
                    # Note: pred_kpts_trimmed and gt_kpts_trimmed are LISTS of tensors
                    # with potentially different lengths (per category)
                    pck_evaluator.add_batch(
                        pred_keypoints=pred_kpts_trimmed_pixels,  # NOW IN PIXELS!
                        gt_keypoints=gt_kpts_trimmed_pixels,      # NOW IN PIXELS!
                        bbox_widths=bbox_widths,
                        bbox_heights=bbox_heights,
                        category_ids=batch.get('category_ids', None),
                        visibility=visibility_list  # Now includes actual visibility from metadata
                    )

    # Gather stats
    metric_logger.synchronize_between_processes()
    
    # Only print loss stats if they were computed
    if len(metric_logger.meters) > 0:
        print(f"Averaged validation stats: {metric_logger}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Add PCK stats if computed
    if compute_pck and pck_evaluator is not None:
        pck_results = pck_evaluator.get_results()
        stats['pck'] = pck_results['pck_overall']
        stats['pck_mean_categories'] = pck_results['mean_pck_categories']
        stats['pck_num_correct'] = pck_results['total_correct']
        stats['pck_num_visible'] = pck_results['total_visible']
        
        print(f"\nPCK@{pck_threshold} (Autoregressive Inference): {pck_results['pck_overall']:.2%} "
              f"({pck_results['total_correct']}/{pck_results['total_visible']} keypoints)")
        print(f"Mean PCK across categories: {pck_results['mean_pck_categories']:.2%}")
    
    # Set default loss if not computed
    if 'loss' not in stats:
        stats['loss'] = 0.0
        stats['loss_ce'] = 0.0
        stats['loss_coords'] = 0.0

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
    pbar = tqdm(data_loader, desc=f'Test (Unseen) PCK@{pck_threshold}', leave=True, ncols=None,
                mininterval=2.0, miniters=10)
    
    for batch_idx, batch in enumerate(pbar):
        num_batches += 1
        
        # ========================================================================
        # DEBUG: Log unseen category evaluation structure on first batch
        # ========================================================================
        if DEBUG_CAPE and batch_idx == 0:
            debug_log("=" * 80)
            debug_log("INFERENCE ON UNSEEN CATEGORIES (First Batch)")
            debug_log("=" * 80)
            if 'category_ids' in batch:
                unique_cats = torch.unique(batch['category_ids']).cpu().numpy()
                debug_log(f"Unseen categories in batch: {unique_cats}")
            debug_log(f"Batch contains {len(batch.get('query_images', []))} queries")
        
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
        
        # ========================================================================
        # DEBUG: Log that query GT is NOT passed to forward_inference
        # ========================================================================
        if DEBUG_CAPE and batch_idx == 0:
            debug_log("\nInference Input Structure:")
            debug_log(f"  query_images:    {query_images.shape}")
            debug_log(f"  support_coords:  {support_coords.shape}")
            debug_log(f"  support_masks:   {support_masks.shape}")
            debug_log(f"  skeleton_edges:  List of {len(support_skeletons) if support_skeletons else 0}")
            debug_log(f"  ✓ Query GT (target_seq) loaded: {query_targets.get('target_seq') is not None}")
            debug_log(f"  ✓ Query GT will be used ONLY for metrics, NOT passed to forward_inference")
            debug_log("=" * 80 + "\n")

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
