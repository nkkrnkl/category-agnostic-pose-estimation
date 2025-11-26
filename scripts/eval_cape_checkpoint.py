#!/usr/bin/env python3
"""
Standalone Evaluation and Visualization Script for CAPE Model

This script:
1. Loads a trained CAPE model from checkpoint
2. Runs evaluation on validation set using existing evaluation logic
3. Computes and logs metrics (PCK@0.2, per-category PCK, etc.)
4. Generates side-by-side visualizations: GT vs Predicted keypoints
5. Saves metrics to JSON and visualizations to output directory

Usage:
    python scripts/eval_cape_checkpoint.py \
        --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
        --split val \
        --num-visualizations 50 \
        --output-dir outputs/cape_eval

The script automatically:
- Detects GPU/CPU and uses appropriate device
- Loads model configuration from checkpoint
- Uses existing evaluation logic from engine_cape.py
- Handles coordinate transformations for proper visualization
"""

import os
# Enable MPS fallback for Apple Silicon
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import build_episodic_dataloader
from models import build_model
from models.cape_model import build_cape_model
from util.eval_utils import PCKEvaluator, compute_pck_bbox
from models.engine_cape import extract_keypoints_from_sequence


def get_args_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate CAPE model checkpoint and generate visualizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to checkpoint (.pth file)')
    
    # Data arguments
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--dataset-root', default=None, type=str,
                       help='Dataset root (if different from checkpoint args)')
    
    # Evaluation arguments
    parser.add_argument('--num-episodes', default=None, type=int,
                       help='Number of episodes to evaluate (None = all)')
    parser.add_argument('--num-queries-per-episode', default=None, type=int,
                       help='Queries per episode (None = use checkpoint default)')
    parser.add_argument('--pck-threshold', default=0.2, type=float,
                       help='PCK threshold (fraction of bbox diagonal)')
    
    # Visualization arguments
    parser.add_argument('--num-visualizations', default=50, type=int,
                       help='Maximum number of examples to visualize')
    parser.add_argument('--draw-skeleton', action='store_true',
                       help='Draw skeleton edges if available')
    parser.add_argument('--save-all-queries', action='store_true',
                       help='Save all queries in episode (default: only first per episode)')
    
    # Output arguments
    parser.add_argument('--output-dir', default='outputs/cape_eval', type=str,
                       help='Directory to save metrics and visualizations')
    
    # Device arguments
    parser.add_argument('--device', default=None, type=str,
                       help='Device to use (cpu/cuda/mps, default: auto-detect)')
    parser.add_argument('--num-workers', default=0, type=int,
                       help='Number of dataloader workers')
    
    return parser


def load_checkpoint_and_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, argparse.Namespace]:
    """
    Load CAPE model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded CAPE model in eval mode
        args: Training arguments from checkpoint
    """
    print("=" * 80)
    print("LOADING MODEL FROM CHECKPOINT")
    print("=" * 80)
    print()
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Loading...")
    
    # Load checkpoint (contains model weights + training args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get training args from checkpoint
    args = checkpoint['args']
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Best PCK: {checkpoint.get('best_pck', 'N/A')}")
    print()
    
    # ========================================================================
    # CRITICAL: Build model WITH tokenizer (fix for PCK@100% bug)
    # ========================================================================
    print("Building dataset to get tokenizer...")
    temp_dataset = build_mp100_cape('train', args)
    tokenizer = temp_dataset.get_tokenizer()
    
    if tokenizer is None:
        print("  ⚠️  Warning: No tokenizer found (poly2seq might be disabled)")
        print("     forward_inference may not work correctly")
    else:
        print(f"  ✓ Tokenizer: vocab_size={len(tokenizer)}, num_bins={tokenizer.num_bins}")
    print()
    
    # Build model
    print("Building model...")
    build_result = build_model(args, train=False, tokenizer=tokenizer)
    if isinstance(build_result, tuple):
        base_model, _ = build_result
    else:
        base_model = build_result
    model = build_cape_model(args, base_model)
    
    # Load weights
    print("Loading model weights...")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if missing_keys:
        print(f"  ⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"     - {key}")
    
    if unexpected_keys:
        print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)}")
        # These are likely from state_dict contamination bug
        contaminated_keys = [k for k in unexpected_keys if 'support_cross_attn_layers' in k or 'support_attn_norms' in k]
        if contaminated_keys:
            print(f"     ({len(contaminated_keys)} contaminated keys from old bug - safe to ignore)")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys[:5]:
                print(f"     - {key}")
            if len(unexpected_keys) > 5:
                print(f"     ... and {len(unexpected_keys) - 5} more")
    
    print()
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Has forward_inference: {hasattr(model, 'forward_inference')}")
    print()
    
    return model, args


def build_dataloader(args: argparse.Namespace, split: str, num_workers: int,
                     num_episodes: int = None, num_queries: int = None) -> DataLoader:
    """
    Build episodic dataloader for evaluation.
    
    Args:
        args: Model/dataset arguments from checkpoint
        split: 'train', 'val', or 'test'
        num_workers: Number of dataloader workers
        num_episodes: Number of episodes per epoch (None = use default)
        num_queries: Number of queries per episode (None = use default from args)
        
    Returns:
        dataloader: Episodic dataloader
    """
    print("=" * 80)
    print(f"BUILDING {split.upper()} DATALOADER")
    print("=" * 80)
    print()
    
    # Build base dataset
    dataset = build_mp100_cape(split, args)
    print(f"✓ Base dataset: {len(dataset)} images")
    
    # Use checkpoint values or override
    if num_queries is None:
        num_queries = getattr(args, 'num_queries_per_episode', 2)
    
    # For validation, use reasonable number of episodes
    if num_episodes is None:
        if split == 'val':
            num_episodes = 100  # Default: 100 episodes for thorough evaluation
        elif split == 'test':
            num_episodes = 200  # More episodes for test
        else:
            num_episodes = 50  # Fewer for train split
    
    # Build episodic dataloader
    category_split_file = Path(args.dataset_root) / args.category_split_file
    
    dataloader = build_episodic_dataloader(
        base_dataset=dataset,
        category_split_file=str(category_split_file),
        split=split,
        batch_size=1,  # Process one episode at a time for visualization
        num_queries_per_episode=num_queries,
        episodes_per_epoch=num_episodes,
        num_workers=num_workers,
        seed=42  # Deterministic for visualization
    )
    
    print(f"✓ Episodic dataloader:")
    print(f"  Episodes per epoch: {num_episodes}")
    print(f"  Queries per episode: {num_queries}")
    print(f"  Total query samples: {num_episodes * num_queries}")
    print()
    
    return dataloader


def run_evaluation(model: nn.Module, dataloader: DataLoader, device: torch.device,
                   pck_threshold: float = 0.2) -> Dict:
    """
    Run evaluation on the dataloader and compute metrics.
    
    Args:
        model: CAPE model in eval mode
        dataloader: Episodic dataloader
        device: Device to run on
        pck_threshold: PCK threshold
        
    Returns:
        metrics: Dictionary with evaluation metrics
        predictions_list: List of prediction dictionaries for visualization
    """
    print("=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)
    print()
    print("ℹ️  Note: The model uses autoregressive inference and generates variable-length")
    print("   sequences based on category keypoint count. Sequences are padded internally")
    print("   for batch processing. This is expected and correct behavior.")
    print()
    
    model.eval()
    pck_evaluator = PCKEvaluator(threshold=pck_threshold)
    
    # Store predictions for visualization
    predictions_list = []
    
    # Track prediction statistics
    pred_stats = {
        'num_keypoints_generated': [],  # How many keypoints model generated
        'num_keypoints_expected': [],   # How many keypoints were in GT
        'sequence_lengths': [],          # Actual sequence lengths generated
    }
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            support_coords = batch['support_coords'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_images = batch['query_images'].to(device)
            support_skeletons = batch.get('support_skeletons', None)
            query_targets = {k: v.to(device) for k, v in batch['query_targets'].items()}
            
            # ====================================================================
            # Run forward inference (autoregressive, no teacher forcing)
            # ====================================================================
            try:
                if hasattr(model, 'module'):
                    predictions = model.module.forward_inference(
                        samples=query_images,
                        support_coords=support_coords,
                        support_mask=support_masks,
                        skeleton_edges=support_skeletons
                    )
                else:
                    predictions = model.forward_inference(
                        samples=query_images,
                        support_coords=support_coords,
                        support_mask=support_masks,
                        skeleton_edges=support_skeletons
                    )
            except Exception as e:
                print(f"\n⚠️  WARNING: forward_inference failed for episode {batch_idx}")
                print(f"   Error: {type(e).__name__}: {e}")
                print(f"   Skipping this episode...")
                continue
            
            # Extract coordinates
            pred_coords = predictions.get('coordinates', None)
            if pred_coords is None:
                print(f"\n⚠️  WARNING: No predictions for episode {batch_idx}, skipping...")
                continue
            
            gt_coords = query_targets['target_seq']
            token_labels = query_targets['token_labels']
            mask = query_targets['mask']
            
            # ====================================================================
            # Pad autoregressive predictions to GT sequence length if needed
            # ====================================================================
            # During autoregressive inference, the model generates variable-length
            # sequences (e.g., 10-40 tokens depending on category) and stops when
            # it predicts EOS. The GT sequences are padded to a fixed length (200).
            # We need to pad predictions to match GT length for loss computation.
            # ====================================================================
            if pred_coords.dim() == 3 and pred_coords.shape[1] < gt_coords.shape[1]:
                # Pad predictions to match GT length
                seq_len = gt_coords.shape[1]
                batch_size = pred_coords.shape[0]
                
                # Create padded tensors
                padded_pred_coords = torch.zeros(batch_size, seq_len, 2, device=pred_coords.device)
                padded_pred_coords[:, :pred_coords.shape[1], :] = pred_coords
                pred_coords = padded_pred_coords
                
                # Also pad pred_logits if available
                pred_logits = predictions.get('logits', None)
                if pred_logits is not None and pred_logits.shape[1] < seq_len:
                    vocab_size = pred_logits.shape[2]
                    padded_pred_logits = torch.zeros(batch_size, seq_len, vocab_size, device=pred_logits.device)
                    padded_pred_logits[:, :pred_logits.shape[1], :] = pred_logits
                    predictions['logits'] = padded_pred_logits
            
            # Extract keypoints from sequences
            try:
                # ================================================================
                # DEBUG: Print shapes before extraction (controlled by DEBUG_EVAL env var)
                # ================================================================
                if batch_idx == 0 and os.environ.get('DEBUG_EVAL', '0') == '1':
                    print(f"\n[DEBUG] Before extraction:")
                    print(f"  gt_coords shape: {gt_coords.shape}")
                    print(f"  token_labels shape: {token_labels.shape}")
                    print(f"  mask shape: {mask.shape}")
                    print(f"  mask dtype: {mask.dtype}")
                    print(f"  mask[0] shape: {mask[0].shape}")
                    print(f"  mask[0] sum (# True values): {mask[0].sum().item()}")
                    print(f"  pred_coords shape: {pred_coords.shape}")
                
                # ================================================================
                # CRITICAL FIX: Extract GT using GT structure, predictions using PREDICTED structure
                # ================================================================
                # BUG: Previously used GT token labels for both pred and GT extraction
                # This assumes model generates EXACTLY the same token sequence as GT
                # FIX: Use predicted token types from model output for predictions
                # ================================================================
                
                # Extract GT using GT token labels (correct)
                gt_kpts = extract_keypoints_from_sequence(gt_coords, token_labels, mask)
                
                # Extract predictions using PREDICTED token labels (fixed)
                pred_logits = predictions.get('logits', None)
                if pred_logits is not None:
                    # Use model's predicted token types
                    from util.sequence_utils import extract_keypoints_from_predictions
                    pred_kpts = extract_keypoints_from_predictions(
                        pred_coords, pred_logits, max_keypoints=gt_kpts.shape[1]
                    )
                else:
                    # Fallback if logits not available
                    # Still use GT structure but add WARNING
                    if batch_idx == 0:
                        print("⚠️  WARNING: Using GT token structure for predictions (logits unavailable)")
                        print("   This may cause incorrect PCK if model's token sequence differs from GT")
                    pred_kpts = extract_keypoints_from_sequence(pred_coords, token_labels, mask)
                
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
                
                # Track prediction statistics
                for idx in range(pred_kpts.shape[0]):
                    pred_stats['num_keypoints_generated'].append(pred_kpts[idx].shape[0])
                    pred_stats['num_keypoints_expected'].append(gt_kpts[idx].shape[0])
                    pred_stats['sequence_lengths'].append(predictions.get('coordinates').shape[1])
                
            except Exception as e:
                print(f"\n⚠️  WARNING: Keypoint extraction failed for episode {batch_idx}")
                print(f"   Error: {e}")
                print(f"   pred_coords shape: {pred_coords.shape}")
                print(f"   gt_coords shape: {gt_coords.shape}")
                print(f"   token_labels shape: {token_labels.shape}")
                print(f"   mask shape: {mask.shape}")
                continue
            
            # Get metadata for PCK computation
            query_metadata = batch.get('query_metadata', None)
            support_metadata = batch.get('support_metadata', None)
            
            if query_metadata is not None and len(query_metadata) > 0:
                # Extract bbox dimensions and visibility
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
                    num_kpts_for_category = len(vis)
                    
                    # Debug logging for keypoint count diagnostic
                    if os.environ.get('DEBUG_KEYPOINT_COUNT', '0') == '1' and idx == 0 and batch_idx == 0:
                        print(f"[DIAG eval_cape] Trimmed keypoints:")
                        print(f"  pred before: {pred_kpts[idx].shape}")
                        print(f"  gt before: {gt_kpts[idx].shape}")
                        print(f"  num_kpts_for_category: {num_kpts_for_category}")
                        print(f"  Will trim to: [{num_kpts_for_category}, 2]")
                    
                    # ================================================================
                    # CRITICAL ASSERTION: Detect keypoint count mismatch before trimming
                    # ================================================================
                    # If predictions significantly exceed expected count, the model
                    # likely didn't learn to predict EOS properly.
                    # ================================================================
                    pred_count = pred_kpts[idx].shape[0]
                    expected_count = num_kpts_for_category
                    excess = pred_count - expected_count
                    
                    if excess > 10 and batch_idx == 0:  # Only warn once
                        import warnings
                        warnings.warn(
                            f"⚠️  Model generated {pred_count} keypoints but expected ~{expected_count}. "
                            f"Excess: {excess} keypoints. This suggests the model didn't learn to "
                            f"predict EOS properly. The model may need retraining with EOS token "
                            f"included in the classification loss."
                        )
                    
                    # Trim to actual number of keypoints for this category
                    pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])
                    gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])
                    visibility_list.append(vis)
                
                bbox_widths = torch.tensor(bbox_widths, device=device)
                bbox_heights = torch.tensor(bbox_heights, device=device)
            else:
                # Fallback: assume 512x512 bbox
                batch_size = pred_kpts.shape[0]
                bbox_widths = torch.full((batch_size,), 512.0, device=device)
                bbox_heights = torch.full((batch_size,), 512.0, device=device)
                visibility_list = None
                pred_kpts_trimmed = pred_kpts
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
            
            # Add to PCK evaluator
            pck_evaluator.add_batch(
                pred_keypoints=pred_kpts_trimmed_pixels,  # NOW IN PIXELS!
                gt_keypoints=gt_kpts_trimmed_pixels,      # NOW IN PIXELS!
                bbox_widths=bbox_widths,
                bbox_heights=bbox_heights,
                category_ids=batch.get('category_ids', None),
                visibility=visibility_list
            )
            
            # Store predictions for visualization
            predictions_list.append({
                'episode_idx': batch_idx,
                'support_images': batch['support_images'].cpu(),
                'support_coords': batch['support_coords'].cpu(),
                'support_masks': batch['support_masks'].cpu(),
                'support_skeletons': support_skeletons,
                'support_metadata': support_metadata,
                'query_images': batch['query_images'].cpu(),
                'pred_keypoints': pred_kpts_trimmed,
                'gt_keypoints': gt_kpts_trimmed,
                'bbox_widths': bbox_widths.cpu(),
                'bbox_heights': bbox_heights.cpu(),
                'visibility': visibility_list,
                'category_ids': batch.get('category_ids', None),
                'query_metadata': query_metadata,
            })
    
    # Get final metrics
    results = pck_evaluator.get_results()
    
    # Add prediction statistics to results
    if len(pred_stats['num_keypoints_generated']) > 0:
        results['pred_stats'] = {
            'avg_keypoints_generated': float(np.mean(pred_stats['num_keypoints_generated'])),
            'avg_keypoints_expected': float(np.mean(pred_stats['num_keypoints_expected'])),
            'avg_sequence_length': float(np.mean(pred_stats['sequence_lengths'])),
            'min_keypoints_generated': int(np.min(pred_stats['num_keypoints_generated'])),
            'max_keypoints_generated': int(np.max(pred_stats['num_keypoints_generated'])),
        }
    
    # Print summary
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Overall PCK@{pck_threshold}: {results['pck_overall']:.4f} ({results['pck_overall']*100:.2f}%)")
    print(f"  Correct keypoints: {results['total_correct']} / {results['total_visible']}")
    print(f"  Mean PCK across categories: {results['mean_pck_categories']:.4f} ({results['mean_pck_categories']*100:.2f}%)")
    print()
    
    # Show prediction statistics
    if 'pred_stats' in results:
        print("Prediction Statistics:")
        print(f"  Avg keypoints generated: {results['pred_stats']['avg_keypoints_generated']:.1f}")
        print(f"  Avg keypoints expected: {results['pred_stats']['avg_keypoints_expected']:.1f}")
        print(f"  Avg sequence length: {results['pred_stats']['avg_sequence_length']:.1f}")
        print(f"  Range: {results['pred_stats']['min_keypoints_generated']}-{results['pred_stats']['max_keypoints_generated']} keypoints")
        print()
        
        # Warning if model is not generating enough keypoints
        if results['pred_stats']['avg_keypoints_generated'] < results['pred_stats']['avg_keypoints_expected'] * 0.5:
            print("⚠️  WARNING: Model is generating significantly fewer keypoints than expected!")
            print("   This suggests the model predicts <eos> too early.")
            print("   Possible causes:")
            print("     - Checkpoint from old (buggy) training code")
            print("     - Undertrained model")
            print("     - Model architecture mismatch")
            print()
    
    # Per-category results
    if 'per_category' in results and len(results['per_category']) > 0:
        print("Per-Category PCK:")
        print(f"  {'Category ID':<15} {'PCK':<10} {'Correct/Total':<15}")
        print(f"  {'-'*15} {'-'*10} {'-'*15}")
        
        # Sort by category ID
        cat_results = sorted(results['per_category'].items(), key=lambda x: x[0])
        for cat_id, cat_metrics in cat_results:
            pck = cat_metrics['pck']
            correct = cat_metrics['correct']
            total = cat_metrics['total']
            print(f"  {cat_id:<15} {pck:<10.4f} {correct}/{total:<15}")
    else:
        print("  (No per-category breakdown available)")
    
    print()
    
    return results, predictions_list


def denormalize_keypoints(kpts_norm: np.ndarray, bbox_width: float, bbox_height: float) -> np.ndarray:
    """
    Convert normalized [0,1] keypoints to pixel coordinates.
    
    Args:
        kpts_norm: Keypoints in normalized [0,1] coordinates, shape (N, 2)
        bbox_width: Width of bounding box
        bbox_height: Height of bounding box
        
    Returns:
        kpts_px: Keypoints in pixel coordinates, shape (N, 2)
    """
    if isinstance(kpts_norm, torch.Tensor):
        kpts_norm = kpts_norm.cpu().numpy()
    
    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = kpts_norm[:, 0] * bbox_width
    kpts_px[:, 1] = kpts_norm[:, 1] * bbox_height
    
    return kpts_px


def draw_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray, 
                            visibility: List[int] = None,
                            skeleton_edges: List[Tuple[int, int]] = None,
                            color: Tuple[int, int, int] = (0, 255, 0),
                            marker: str = 'o',
                            label: str = None) -> np.ndarray:
    """
    Draw keypoints and skeleton on image.
    
    Args:
        image: Image array (H, W, 3)
        keypoints: Keypoints in pixel coordinates, shape (N, 2)
        visibility: Visibility flags (0=not labeled, 1=occluded, 2=visible)
        skeleton_edges: List of (start_idx, end_idx) tuples for skeleton
        color: BGR color for keypoints
        marker: 'o' for circles, 'x' for crosses
        label: Optional label to show on image
        
    Returns:
        image_with_kpts: Image with keypoints drawn
    """
    image = image.copy()
    
    # Draw skeleton edges first (behind keypoints)
    if skeleton_edges is not None and len(skeleton_edges) > 0:
        for start_idx, end_idx in skeleton_edges:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                # Check visibility
                if visibility is not None:
                    if start_idx >= len(visibility) or end_idx >= len(visibility):
                        continue
                    if visibility[start_idx] == 0 or visibility[end_idx] == 0:
                        continue  # Skip if either endpoint not labeled
                
                start_pt = tuple(keypoints[start_idx].astype(int))
                end_pt = tuple(keypoints[end_idx].astype(int))
                
                # Draw edge
                cv2.line(image, start_pt, end_pt, color, thickness=2, lineType=cv2.LINE_AA)
    
    # Draw keypoints
    for idx, (x, y) in enumerate(keypoints):
        # Check visibility
        if visibility is not None and idx < len(visibility):
            if visibility[idx] == 0:
                continue  # Skip not-labeled keypoints
        
        pt = (int(x), int(y))
        
        if marker == 'o':
            # Draw circle
            cv2.circle(image, pt, radius=5, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, pt, radius=5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        elif marker == 'x':
            # Draw X
            size = 8
            cv2.line(image, (pt[0]-size, pt[1]-size), (pt[0]+size, pt[1]+size), 
                    color, thickness=2, lineType=cv2.LINE_AA)
            cv2.line(image, (pt[0]-size, pt[1]+size), (pt[0]+size, pt[1]-size),
                    color, thickness=2, lineType=cv2.LINE_AA)
    
    # Add label if provided
    if label is not None:
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 4, cv2.LINE_AA)  # White outline
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Black text
    
    return image


def create_visualization(pred_dict: Dict, output_path: Path, 
                        draw_skeleton: bool = True,
                        query_idx: int = 0) -> None:
    """
    Create side-by-side visualization: GT vs Predicted.
    
    Args:
        pred_dict: Prediction dictionary with images, keypoints, metadata
        output_path: Path to save visualization
        draw_skeleton: Whether to draw skeleton edges
        query_idx: Which query in the episode to visualize (default: first)
    """
    # Extract data for this query
    support_img = pred_dict['support_images'][0]  # (C, H, W)
    query_img = pred_dict['query_images'][query_idx]  # (C, H, W)
    
    # Convert to numpy (H, W, C) and BGR for cv2
    support_img_np = support_img.permute(1, 2, 0).numpy()
    query_img_np = query_img.permute(1, 2, 0).numpy()
    
    # Convert from [0,1] to [0,255] if needed
    if support_img_np.max() <= 1.0:
        support_img_np = (support_img_np * 255).astype(np.uint8)
        query_img_np = (query_img_np * 255).astype(np.uint8)
    else:
        support_img_np = support_img_np.astype(np.uint8)
        query_img_np = query_img_np.astype(np.uint8)
    
    # Convert RGB to BGR for cv2
    support_img_bgr = cv2.cvtColor(support_img_np, cv2.COLOR_RGB2BGR)
    query_img_bgr = cv2.cvtColor(query_img_np, cv2.COLOR_RGB2BGR)
    
    # ========================================================================
    # CRITICAL SANITY CHECK: Verify image dimensions
    # ========================================================================
    # All images should be 512x512 after preprocessing (crop + resize)
    # If not, something is wrong with the data pipeline
    # ========================================================================
    assert query_img.shape[1] == 512 and query_img.shape[2] == 512, \
        f"Expected 512x512 query images, got {query_img.shape}"
    assert support_img.shape[1] == 512 and support_img.shape[2] == 512, \
        f"Expected 512x512 support images, got {support_img.shape}"
    
    # Get keypoints
    if isinstance(pred_dict['pred_keypoints'], list):
        pred_kpts = pred_dict['pred_keypoints'][query_idx]  # (N, 2)
        gt_kpts = pred_dict['gt_keypoints'][query_idx]  # (N, 2)
    else:
        pred_kpts = pred_dict['pred_keypoints'][query_idx].cpu().numpy()
        gt_kpts = pred_dict['gt_keypoints'][query_idx].cpu().numpy()
    
    # ========================================================================
    # CRITICAL FIX: Get ORIGINAL bbox dimensions for PCK, NOT for visualization!
    # ========================================================================
    # These bbox dimensions are the ORIGINAL bbox size BEFORE resizing to 512x512.
    # They are used ONLY for PCK threshold computation, NOT for denormalizing coords!
    #
    # Coordinate spaces:
    #   - Model outputs: [0, 1] normalized space (relative to 512x512 image)
    #   - For visualization: multiply by 512 to get pixel coords in 512x512 image
    #   - For PCK: compare in [0, 1] space, threshold based on original bbox diagonal
    # ========================================================================
    bbox_w_original = pred_dict['bbox_widths'][query_idx].item()
    bbox_h_original = pred_dict['bbox_heights'][query_idx].item()
    
    # Get visibility
    visibility = None
    if pred_dict['visibility'] is not None and len(pred_dict['visibility']) > query_idx:
        visibility = pred_dict['visibility'][query_idx]
    
    # Get skeleton edges
    skeleton_edges = None
    if draw_skeleton and pred_dict['support_skeletons'] is not None:
        if isinstance(pred_dict['support_skeletons'], list) and len(pred_dict['support_skeletons']) > 0:
            skeleton_edges = pred_dict['support_skeletons'][0]  # Use first support's skeleton
    
    # Get support keypoints for support image
    support_coords_norm = pred_dict['support_coords'][0].cpu().numpy()  # (N, 2)
    support_mask = pred_dict['support_masks'][0].cpu().numpy()  # (N,)
    
    # Filter valid support keypoints
    valid_support_mask = support_mask > 0.5
    support_kpts_valid = support_coords_norm[valid_support_mask]
    
    # Get support bbox dimensions (ORIGINAL, for PCK only - not used here)
    support_bbox_w_original = bbox_w_original  # Assume same size as query
    support_bbox_h_original = bbox_h_original
    if pred_dict['support_metadata'] is not None and len(pred_dict['support_metadata']) > 0:
        support_meta = pred_dict['support_metadata'][0]
        support_bbox_w_original = support_meta.get('bbox_width', 512.0)
        support_bbox_h_original = support_meta.get('bbox_height', 512.0)
    
    # ========================================================================
    # CRITICAL FIX: Denormalize keypoints with 512, NOT original bbox dims!
    # ========================================================================
    # All keypoints are in [0, 1] normalized space relative to 512x512 images.
    # To get pixel coordinates for visualization on 512x512 images, multiply by 512.
    #
    # WRONG: multiply by original bbox dims (e.g., 619x964) → coords outside image!
    # CORRECT: multiply by 512 → coords within [0, 512] range ✓
    #
    # The training pipeline does:
    #   1. Crop to bbox (size: bbox_w × bbox_h)
    #   2. Resize to 512×512
    #   3. Normalize by 512: kpts /= 512
    #
    # So denormalization must be: kpts *= 512 (NOT *= bbox_w!)
    # ========================================================================
    support_kpts_px = denormalize_keypoints(support_kpts_valid, 512.0, 512.0)
    pred_kpts_px = denormalize_keypoints(pred_kpts, 512.0, 512.0)
    gt_kpts_px = denormalize_keypoints(gt_kpts, 512.0, 512.0)
    
    # Debug logging (first visualization only)
    if os.environ.get('DEBUG_VIS', '0') == '1':
        if query_idx == 0:
            print(f"\n[DEBUG_VIS] Coordinate Denormalization Check:")
            print(f"  Image shape: {query_img.shape} (should be [C, 512, 512])")
            print(f"  Original bbox dims: {bbox_w_original:.1f} × {bbox_h_original:.1f}")
            print(f"  GT keypoints (normalized): min={gt_kpts.min():.3f}, max={gt_kpts.max():.3f}")
            print(f"  GT keypoints (pixel): min={gt_kpts_px.min():.1f}, max={gt_kpts_px.max():.1f}")
            if gt_kpts_px.max() > 512 or gt_kpts_px.min() < 0:
                print(f"  ⚠️  WARNING: GT keypoints outside [0, 512] range!")
            else:
                print(f"  ✓ GT keypoints within valid range")
            print(f"  Pred keypoints (pixel): min={pred_kpts_px.min():.1f}, max={pred_kpts_px.max():.1f}")
            if pred_kpts_px.max() > 512 or pred_kpts_px.min() < 0:
                print(f"  ⚠️  WARNING: Predicted keypoints outside [0, 512] range!")
            else:
                print(f"  ✓ Predicted keypoints within valid range")
    
    # Draw on images
    # Support: green circles + skeleton
    support_vis = draw_keypoints_on_image(
        support_img_bgr,
        support_kpts_px,
        visibility=None,  # All support keypoints are visible
        skeleton_edges=skeleton_edges if draw_skeleton else None,
        color=(0, 255, 0),  # Green
        marker='o',
        label='Support (GT)'
    )
    
    # Query: GT with cyan circles + skeleton
    query_gt_vis = draw_keypoints_on_image(
        query_img_bgr.copy(),
        gt_kpts_px,
        visibility=visibility,
        skeleton_edges=skeleton_edges if draw_skeleton else None,
        color=(255, 255, 0),  # Cyan
        marker='o',
        label='Ground Truth'
    )
    
    # Query: Predicted with red X + skeleton
    query_pred_vis = draw_keypoints_on_image(
        query_img_bgr.copy(),
        pred_kpts_px,
        visibility=visibility,
        skeleton_edges=skeleton_edges if draw_skeleton else None,
        color=(0, 0, 255),  # Red
        marker='x',
        label='Predicted'
    )
    
    # Compute PCK for this sample
    try:
        # Ensure keypoints are tensors
        if not isinstance(pred_kpts, torch.Tensor):
            pred_kpts_tensor = torch.from_numpy(pred_kpts)
        else:
            pred_kpts_tensor = pred_kpts
        
        if not isinstance(gt_kpts, torch.Tensor):
            gt_kpts_tensor = torch.from_numpy(gt_kpts)
        else:
            gt_kpts_tensor = gt_kpts
        
        # ====================================================================
        # PCK computation uses ORIGINAL bbox dims for threshold calculation
        # ====================================================================
        # PCK@bbox threshold = alpha * sqrt(bbox_w² + bbox_h²)
        # where bbox_w and bbox_h are the ORIGINAL bbox dimensions.
        # This is correct - keypoints are in [0,1] space, bbox dims scale threshold.
        # ====================================================================
        pck = compute_pck_bbox(
            pred_kpts_tensor, gt_kpts_tensor,
            bbox_w_original, bbox_h_original,  # Use ORIGINAL dims for PCK threshold
            threshold=pred_dict.get('pck_threshold', 0.2),
            visibility_mask=visibility
        )
        pck_text = f"PCK@0.2: {pck:.2%}"
    except Exception as e:
        pck_text = f"PCK: N/A ({type(e).__name__})"
    
    # Add PCK text to predicted image
    cv2.putText(query_pred_vis, pck_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(query_pred_vis, pck_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Create side-by-side visualization: Support | GT | Predicted
    vis_combined = np.hstack([support_vis, query_gt_vis, query_pred_vis])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_combined)


def save_metrics_to_json(metrics: Dict, output_dir: Path, split: str) -> None:
    """Save metrics to JSON file."""
    output_file = output_dir / f'metrics_{split}.json'
    
    # Convert any numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    
    with open(output_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"✓ Metrics saved to: {output_file}")


def main():
    """Main evaluation and visualization pipeline."""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 80)
    print("CAPE MODEL EVALUATION & VISUALIZATION")
    print("=" * 80)
    print()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    print()
    
    # Load model
    model, checkpoint_args = load_checkpoint_and_model(args.checkpoint, device)
    
    # Override dataset root if specified
    if args.dataset_root is not None:
        checkpoint_args.dataset_root = args.dataset_root
    
    # Build dataloader
    dataloader = build_dataloader(
        checkpoint_args,
        args.split,
        args.num_workers,
        num_episodes=args.num_episodes,
        num_queries=args.num_queries_per_episode
    )
    
    # Run evaluation
    metrics, predictions_list = run_evaluation(
        model, dataloader, device,
        pck_threshold=args.pck_threshold
    )
    
    # Add metadata to metrics
    metrics['checkpoint_path'] = str(args.checkpoint)
    metrics['checkpoint_epoch'] = checkpoint_args.__dict__.get('epoch', 'unknown')
    metrics['split'] = args.split
    metrics['pck_threshold'] = args.pck_threshold
    metrics['num_episodes'] = len(predictions_list)
    metrics['num_queries_total'] = len(predictions_list) * (args.num_queries_per_episode or checkpoint_args.num_queries_per_episode)
    
    # Save metrics
    save_metrics_to_json(metrics, output_dir, args.split)
    
    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    num_to_visualize = min(args.num_visualizations, len(predictions_list))
    print(f"Creating {num_to_visualize} visualizations...")
    print()
    
    for idx in range(num_to_visualize):
        pred_dict = predictions_list[idx]
        
        # Store PCK threshold for visualization
        pred_dict['pck_threshold'] = args.pck_threshold
        
        # Determine how many queries to save
        num_queries = len(pred_dict['pred_keypoints']) if isinstance(pred_dict['pred_keypoints'], list) else pred_dict['pred_keypoints'].shape[0]
        queries_to_save = range(num_queries) if args.save_all_queries else [0]
        
        for query_idx in queries_to_save:
            # Get category ID for filename
            cat_id = pred_dict['category_ids'][query_idx] if pred_dict['category_ids'] is not None else 'unknown'
            
            # Get image ID if available
            img_id = 'unknown'
            if pred_dict['query_metadata'] is not None and len(pred_dict['query_metadata']) > query_idx:
                img_id = pred_dict['query_metadata'][query_idx].get('image_id', 'unknown')
            
            # Create output filename
            output_filename = f"vis_{idx:04d}_q{query_idx}_cat{cat_id}_img{img_id}.png"
            output_path = vis_dir / output_filename
            
            # Create visualization
            try:
                create_visualization(
                    pred_dict,
                    output_path,
                    draw_skeleton=args.draw_skeleton,
                    query_idx=query_idx
                )
            except Exception as e:
                print(f"  ⚠️  Failed to create visualization {idx}: {e}")
                continue
    
    print()
    print(f"✓ Visualizations saved to: {vis_dir}")
    print(f"  Total: {num_to_visualize} visualization(s)")
    print()
    
    # Print final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Validation results on split={args.split}:")
    print(f"  Overall PCK@{args.pck_threshold}: {metrics['pck_overall']:.4f} ({metrics['pck_overall']*100:.2f}%)")
    print(f"  Correct: {metrics['total_correct']} / {metrics['total_visible']}")
    print(f"  Mean PCK across categories: {metrics['mean_pck_categories']:.4f}")
    print()
    
    # Top/bottom performing categories
    if 'per_category' in metrics and len(metrics['per_category']) > 0:
        cat_pcks = [(cat_id, cat_metrics['pck']) for cat_id, cat_metrics in metrics['per_category'].items()]
        cat_pcks_sorted = sorted(cat_pcks, key=lambda x: x[1], reverse=True)
        
        print("Top 5 performing categories:")
        for cat_id, pck in cat_pcks_sorted[:5]:
            print(f"  cat_id={cat_id}: {pck:.2%}")
        
        print()
        print("Bottom 5 performing categories:")
        for cat_id, pck in cat_pcks_sorted[-5:]:
            print(f"  cat_id={cat_id}: {pck:.2%}")
        print()
    
    print(f"Visualizations saved to: {vis_dir}")
    print(f"Metrics saved to: {output_dir / f'metrics_{args.split}.json'}")
    print()
    print("Checkpoint: " + str(args.checkpoint))
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

