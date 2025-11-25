#!/usr/bin/env python3
"""
Debug script to understand why validation PCK is 100%.

This script:
1. Loads the latest checkpoint
2. Runs ONE validation batch
3. Prints detailed debug information about:
   - Which inference method is used
   - Support vs query image IDs
   - Predicted vs GT vs Support coordinates
   - PCK computation details
4. Helps identify if there's data leakage or inference issues
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Set debug mode
os.environ['DEBUG_CAPE'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import build_episodic_dataloader
from models.cape_model import build_cape_model
from util.eval_utils import compute_pck_bbox


def load_latest_checkpoint(checkpoint_dir='outputs/cape_run'):
    """Load the most recent checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_e*.pth'), 
                        key=lambda p: p.stat().st_mtime, 
                        reverse=True)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    latest = checkpoints[0]
    print(f"Loading checkpoint: {latest}")
    checkpoint = torch.load(latest, map_location='cpu')
    return checkpoint, latest


def main():
    print("=" * 80)
    print("VALIDATION PCK DEBUG ANALYSIS")
    print("=" * 80)
    print()
    
    # Load checkpoint
    checkpoint, checkpoint_path = load_latest_checkpoint()
    train_args = checkpoint['args']
    
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best PCK: {checkpoint.get('best_pck', 'unknown')}")
    print()
    
    # Build model
    print("Building model...")
    device = torch.device('cpu')  # Use CPU for debugging to avoid memory issues
    model = build_cape_model(train_args, device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"✓ Model loaded on {device}")
    print()
    
    # Check if forward_inference exists
    has_forward_inference = hasattr(model, 'forward_inference')
    print(f"Model has forward_inference: {has_forward_inference}")
    if not has_forward_inference:
        print("  ⚠️  WARNING: forward_inference not available!")
        print("     Validation will use teacher forcing (explains 100% PCK)")
        return
    print()
    
    # Build validation dataset
    print("Building validation dataset...")
    val_dataset = build_mp100_cape('val', train_args)
    print(f"✓ Loaded {len(val_dataset)} validation images")
    print()
    
    # Build validation dataloader (1 batch only)
    print("Building validation dataloader...")
    val_loader = build_episodic_dataloader(
        base_dataset=val_dataset,
        category_split_file=str(Path(train_args.dataset_root) / train_args.category_split_file),
        split='val',
        batch_size=1,
        num_queries_per_episode=train_args.num_queries_per_episode,
        episodes_per_epoch=1,  # Just 1 episode for debugging
        num_workers=0,
        seed=42
    )
    print(f"✓ Created dataloader with {len(val_loader)} episodes")
    print()
    
    # Get one batch
    print("=" * 80)
    print("ANALYZING FIRST VALIDATION BATCH")
    print("=" * 80)
    print()
    
    batch = next(iter(val_loader))
    
    # Extract batch data
    support_images = batch['support_images']
    support_coords = batch['support_coords']
    support_masks = batch['support_masks']
    support_skeletons = batch.get('support_skeletons', None)
    support_metadata = batch.get('support_metadata', [])
    
    query_images = batch['query_images']
    query_targets = batch['query_targets']
    query_metadata = batch.get('query_metadata', [])
    category_ids = batch.get('category_ids', None)
    
    print(f"Batch shapes:")
    print(f"  Support images: {support_images.shape}")
    print(f"  Support coords: {support_coords.shape}")
    print(f"  Query images: {query_images.shape}")
    print(f"  Category IDs: {category_ids}")
    print()
    
    # Check image IDs
    print(f"Image IDs:")
    if support_metadata:
        support_ids = [m.get('image_id', 'N/A') for m in support_metadata]
        print(f"  Support: {support_ids}")
    else:
        print(f"  Support: (metadata not available)")
        support_ids = []
    
    if query_metadata:
        query_ids = [m.get('image_id', 'N/A') for m in query_metadata]
        print(f"  Query: {query_ids}")
    else:
        print(f"  Query: (metadata not available)")
        query_ids = []
    
    # Check for overlap
    if support_ids and query_ids:
        overlap = set(support_ids) & set(query_ids)
        if overlap:
            print(f"\n⚠️  CRITICAL BUG FOUND: Support and query share {len(overlap)} images!")
            print(f"   Overlapping IDs: {list(overlap)}")
            print(f"   This explains 100% PCK - model is predicting keypoints for the")
            print(f"   same image it received as support!")
        else:
            print(f"\n✓ No overlap between support and query image IDs (good)")
    print()
    
    # Run inference
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    print()
    
    with torch.no_grad():
        try:
            predictions = model.forward_inference(
                samples=query_images.to(device),
                support_coords=support_coords.to(device),
                support_mask=support_masks.to(device),
                skeleton_edges=support_skeletons
            )
            print("✓ Used forward_inference (autoregressive)")
        except Exception as e:
            print(f"✗ forward_inference failed: {e}")
            print("  Falling back to teacher forcing...")
            outputs = model(
                samples=query_images.to(device),
                support_coords=support_coords.to(device),
                support_mask=support_masks.to(device),
                targets={k: v.to(device) for k, v in query_targets.items()},
                skeleton_edges=support_skeletons
            )
            predictions = {
                'coordinates': outputs.get('pred_coords', None),
                'logits': outputs.get('pred_logits', None)
            }
            print("  ⚠️  Using teacher forcing (explains 100% PCK)")
    
    print()
    
    # Extract coordinates
    pred_coords = predictions.get('coordinates', None)  # (B*K, seq_len, 2)
    gt_coords = query_targets.get('target_seq', None)  # (B*K, seq_len, 2)
    
    if pred_coords is None or gt_coords is None:
        print("✗ Missing coordinates in predictions or targets")
        return
    
    print(f"Prediction shapes:")
    print(f"  Pred coords: {pred_coords.shape}")
    print(f"  GT coords: {gt_coords.shape}")
    print(f"  Support coords: {support_coords.shape}")
    print()
    
    # Analyze first sample
    print("=" * 80)
    print("DETAILED ANALYSIS OF SAMPLE 0")
    print("=" * 80)
    print()
    
    sample_idx = 0
    pred_sample = pred_coords[sample_idx].cpu().numpy()  # (seq_len, 2)
    gt_sample = gt_coords[sample_idx].cpu().numpy()  # (seq_len, 2)
    support_sample = support_coords[sample_idx].cpu().numpy()  # (max_kpts, 2)
    
    # Get actual number of keypoints for this category
    if query_metadata:
        visibility = query_metadata[sample_idx].get('visibility', [])
        num_kpts = len(visibility)
        bbox_w = query_metadata[sample_idx].get('bbox_width', 512.0)
        bbox_h = query_metadata[sample_idx].get('bbox_height', 512.0)
    else:
        num_kpts = min(len(pred_sample), len(gt_sample), len(support_sample))
        bbox_w = 512.0
        bbox_h = 512.0
        visibility = [2] * num_kpts
    
    print(f"Category info:")
    print(f"  Num keypoints: {num_kpts}")
    print(f"  Bbox: {bbox_w}x{bbox_h}")
    print()
    
    # Trim to actual keypoints
    pred_trimmed = pred_sample[:num_kpts]
    gt_trimmed = gt_sample[:num_kpts]
    support_trimmed = support_sample[:num_kpts]
    
    # Show first 3 keypoints
    print(f"First 3 keypoints (normalized [0,1] coordinates):")
    print(f"  {'Idx':<5} {'Predicted':<20} {'Ground Truth':<20} {'Support':<20} {'Diff(P-GT)':<15} {'Diff(P-S)':<15}")
    print(f"  {'-'*5} {'-'*20} {'-'*20} {'-'*20} {'-'*15} {'-'*15}")
    
    for i in range(min(3, num_kpts)):
        pred_xy = pred_trimmed[i]
        gt_xy = gt_trimmed[i]
        support_xy = support_trimmed[i]
        
        diff_pred_gt = np.linalg.norm(pred_xy - gt_xy)
        diff_pred_support = np.linalg.norm(pred_xy - support_xy)
        
        print(f"  {i:<5} [{pred_xy[0]:.3f}, {pred_xy[1]:.3f}]     "
              f"[{gt_xy[0]:.3f}, {gt_xy[1]:.3f}]       "
              f"[{support_xy[0]:.3f}, {support_xy[1]:.3f}]     "
              f"{diff_pred_gt:<15.4f} {diff_pred_support:<15.4f}")
    
    print()
    
    # Compute overall stats
    diff_pred_gt_all = np.linalg.norm(pred_trimmed - gt_trimmed, axis=1)
    diff_pred_support_all = np.linalg.norm(pred_trimmed - support_trimmed, axis=1)
    
    print(f"Overall coordinate differences (L2 norm):")
    print(f"  Pred vs GT:      mean={diff_pred_gt_all.mean():.4f}, max={diff_pred_gt_all.max():.4f}")
    print(f"  Pred vs Support: mean={diff_pred_support_all.mean():.4f}, max={diff_pred_support_all.max():.4f}")
    print()
    
    # Check if predictions are identical to something
    if diff_pred_gt_all.mean() < 0.001:
        print("🚨 CRITICAL ISSUE: Predictions are IDENTICAL to GT!")
        print("   This means the model is using ground truth (teacher forcing)")
        print("   OR there's data leakage (query GT == support)")
        print()
    elif diff_pred_support_all.mean() < 0.001:
        print("🚨 CRITICAL ISSUE: Predictions are IDENTICAL to Support!")
        print("   This means the model is just copying support keypoints")
        print("   OR there's data leakage (support image == query image)")
        print()
    elif diff_pred_gt_all.mean() < 0.05:
        print("⚠️  WARNING: Predictions are VERY close to GT (mean diff < 0.05)")
        print("   This is suspicious for autoregressive inference on unseen categories")
        print()
    
    # Compute PCK
    print("=" * 80)
    print("PCK COMPUTATION")
    print("=" * 80)
    print()
    
    pck, num_correct, num_visible = compute_pck_bbox(
        pred_keypoints=pred_trimmed,
        gt_keypoints=gt_trimmed,
        bbox_width=bbox_w,
        bbox_height=bbox_h,
        visibility=np.array(visibility),
        threshold=0.2,
        normalize_by='diagonal'
    )
    
    print(f"PCK@0.2: {pck:.2%} ({num_correct}/{num_visible} keypoints)")
    print()
    
    # Show which keypoints are correct vs incorrect
    bbox_diag = np.sqrt(bbox_w**2 + bbox_h**2)
    threshold_pixels = bbox_diag * 0.2
    
    print(f"PCK threshold: {threshold_pixels:.2f} pixels (0.2 * {bbox_diag:.2f})")
    print(f"\nPer-keypoint analysis (first 5):")
    print(f"  {'Idx':<5} {'Vis':<5} {'Dist(pixels)':<15} {'Correct?':<10} {'Coord Error':<20}")
    print(f"  {'-'*5} {'-'*5} {'-'*15} {'-'*10} {'-'*20}")
    
    for i in range(min(5, num_kpts)):
        if visibility[i] > 0:
            # Denormalize to pixels
            pred_px = pred_trimmed[i] * np.array([bbox_w, bbox_h])
            gt_px = gt_trimmed[i] * np.array([bbox_w, bbox_h])
            
            dist_px = np.linalg.norm(pred_px - gt_px)
            is_correct = dist_px < threshold_pixels
            error_str = f"[{pred_px[0]-gt_px[0]:.1f}, {pred_px[1]-gt_px[1]:.1f}]"
            
            print(f"  {i:<5} {visibility[i]:<5} {dist_px:<15.2f} {'✓' if is_correct else '✗':<10} {error_str:<20}")
        else:
            print(f"  {i:<5} {visibility[i]:<5} {'N/A':<15} {'(invisible)':<10} {'-':<20}")
    
    print()
    
    # Summary
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()
    
    if pck >= 0.99:
        print("🚨 PCK is suspiciously high (≥99%)")
        print()
        print("Possible causes:")
        print("  1. Teacher forcing being used (model sees GT during inference)")
        print("  2. Data leakage (support image == query image)")
        print("  3. Support keypoints == Query GT keypoints (annotations issue)")
        print("  4. Model copying support instead of predicting from query image")
        print()
        
        # Check which is the case
        support_eq_gt = diff_pred_support_all.mean() < 0.001 or np.allclose(support_trimmed, gt_trimmed, atol=0.001)
        pred_eq_gt = diff_pred_gt_all.mean() < 0.001
        pred_eq_support = diff_pred_support_all.mean() < 0.001
        
        if pred_eq_gt:
            print("→ Predictions EXACTLY match GT")
            print("  Most likely: Teacher forcing (passing targets to model)")
            print("  Check: Does evaluate_cape pass targets to model.forward()?")
        elif pred_eq_support:
            print("→ Predictions EXACTLY match Support")
            print("  Most likely: Model is copying support, ignoring query image")
            print("  Check: Is query image actually being used in inference?")
        elif support_eq_gt:
            print("→ Support coords EXACTLY match Query GT")
            print("  Most likely: Support image == Query image (data leakage)")
            print("  Check: Are image_ids the same?")
            if support_metadata and query_metadata:
                print(f"     Support ID: {support_metadata[sample_idx].get('image_id', 'N/A')}")
                print(f"     Query ID: {query_metadata[sample_idx].get('image_id', 'N/A')}")
        else:
            print("→ Coordinates are close but not identical")
            print("  This is very suspicious for untrained/early-stage model")
            print("  on UNSEEN categories")
    elif pck > 0.5:
        print("PCK is high but not perfect")
        print("  This could be legitimate if model is well-trained")
        print("  Or could indicate partial data leakage")
    else:
        print("PCK is reasonable for an untrained/early-stage model")
        print("  This is expected behavior for autoregressive inference")
    
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Run this script after each code change to validation logic")
    print("Expected PCK for untrained model on unseen categories: 0-20%")
    print("Expected PCK for well-trained model on unseen categories: 30-60%")
    print("PCK > 90%: Almost certainly indicates a bug")
    print()


if __name__ == '__main__':
    main()

