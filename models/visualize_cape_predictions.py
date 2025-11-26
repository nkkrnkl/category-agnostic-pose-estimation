#!/usr/bin/env python3
"""
Visualize CAPE Pose Predictions on Unseen Test Categories

This script:
1. Loads a trained CAPE model from checkpoint
2. Loads test images from UNSEEN categories (never seen during training)
3. For each category:
   - Uses 1st image as support (template with GT keypoints)
   - Runs autoregressive inference on other images (queries)
4. Visualizes side-by-side:
   - Left: Support image with GT keypoints (green circles + skeleton)
   - Right: Query image with predicted (red X) and GT (cyan circles) keypoints
5. Computes and displays PCK@0.2 for each prediction
6. Saves all visualizations to output directory

Usage:
    python visualize_cape_predictions.py \\
        --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \\
        --device mps \\
        --num_samples 5 \\
        --output_dir visualizations/best_model
"""

# Enable MPS fallback for unsupported operations on Apple Silicon
# This must be set BEFORE importing torch
import os
import sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.roomformer_v2 import build as build_base_model
from models.cape_model import build_cape_model
from datasets.mp100_cape import build_mp100_cape, ImageNotFoundError
from datasets.token_types import TokenType
from datasets.discrete_tokenizer import DiscreteTokenizer


def load_model(checkpoint_path):
    """Load trained CAPE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint (contains model weights + training args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Use saved training args
    train_args = checkpoint['args']
    print(f"  Using training config (epoch {checkpoint.get('epoch', '?')})")

    # Create tokenizer (needed by the base model for sequence generation)
    # vocab_size = num_bins^2, so num_bins = sqrt(vocab_size)
    import numpy as np
    num_bins = int(np.sqrt(train_args.vocab_size))
    tokenizer = DiscreteTokenizer(
        num_bins=num_bins,
        seq_len=train_args.seq_len,
        add_cls=getattr(train_args, 'add_cls_token', False)
    )

    # Build model with saved args
    build_result = build_base_model(train_args, train=False, tokenizer=tokenizer)
    if isinstance(build_result, tuple):
        base_model = build_result[0]
    else:
        base_model = build_result
    model = build_cape_model(train_args, base_model)

    # Load weights (strict=False to handle architecture differences)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, train_args


def decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer):
    """
    Decode predicted token sequence to keypoint coordinates.

    Args:
        pred_tokens: (seq_len,) Token predictions
        pred_coords: (seq_len, 2) Coordinate predictions
        tokenizer: Tokenizer for decoding

    Returns:
        keypoints: List of (x, y) coordinates
    """
    keypoints = []

    for i in range(len(pred_tokens)):
        token = pred_tokens[i].item()

        # Check if this is a coordinate token
        if token == TokenType.coord.value:
            # Coordinates are at the SAME position as the coord token
            if i < len(pred_coords):
                x, y = pred_coords[i]  # pred_coords stores denormalized values
                keypoints.append((x.item(), y.item()))

        # Stop at EOS token
        elif token == TokenType.eos.value:
            break

    return keypoints


def visualize_pose_prediction(support_image, query_image, pred_keypoints, 
                              support_keypoints, gt_keypoints=None,
                              skeleton_edges=None, save_path=None,
                              category_name="Unknown", pck_score=None):
    """
    Visualize predicted pose vs ground truth and support template.

    Args:
        support_image: PIL Image or numpy array (H, W, 3) - support image
        query_image: PIL Image or numpy array (H, W, 3) - query image
        pred_keypoints: List of (x, y) predicted keypoint coordinates (normalized [0,1])
        support_keypoints: List of (x, y) support keypoint coordinates (normalized [0,1])
        gt_keypoints: Optional list of (x, y) ground truth query keypoints (normalized [0,1])
        skeleton_edges: List of [src, dst] edge pairs (1-indexed from MP-100)
        save_path: Path to save visualization
        category_name: Category name for title
        pck_score: Optional PCK score to display
    """
    if isinstance(support_image, Image.Image):
        support_image = np.array(support_image)
    if isinstance(query_image, Image.Image):
        query_image = np.array(query_image)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    support_h, support_w = support_image.shape[:2]
    query_h, query_w = query_image.shape[:2]

    # Left: Support pose (template from another image)
    ax1.imshow(support_image)
    ax1.set_title(f"Support Template (1-Shot)\n{category_name}", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw support keypoints
    if support_keypoints:
        support_kpts_array = np.array(support_keypoints)
        # Denormalize to support image size
        if support_kpts_array.max() <= 1.0:
            support_kpts_array = support_kpts_array.copy()
            support_kpts_array[:, 0] *= support_w
            support_kpts_array[:, 1] *= support_h
        
        ax1.scatter(support_kpts_array[:, 0], support_kpts_array[:, 1],
                   c='lime', s=120, marker='o', edgecolors='black', linewidths=2,
                   label='Support Keypoints', zorder=3)

        # Draw skeleton edges if available
        if skeleton_edges:
            for edge in skeleton_edges:
                if len(edge) == 2:
                    src_idx, dst_idx = edge
                    # Convert 1-indexed to 0-indexed
                    if src_idx > 0:
                        src_idx -= 1
                    if dst_idx > 0:
                        dst_idx -= 1

                    if 0 <= src_idx < len(support_keypoints) and 0 <= dst_idx < len(support_keypoints):
                        x1, y1 = support_kpts_array[src_idx]
                        x2, y2 = support_kpts_array[dst_idx]
                        ax1.plot([x1, x2], [y1, y2], 'lime', linewidth=2, alpha=0.6, zorder=2)

        # Add keypoint numbers
        for idx, (x, y) in enumerate(support_kpts_array):
            ax1.text(x, y - 10, str(idx + 1), color='white', fontsize=8,
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax1.legend(loc='upper right', fontsize=10)

    # Right: Query image with predicted pose (and optionally GT)
    title_text = f"Query Image - Predicted Pose\n{len(pred_keypoints)} keypoints"
    if pck_score is not None:
        title_text += f" | PCK@0.2: {pck_score:.1%}"
    ax2.set_title(title_text, fontsize=14, fontweight='bold')
    ax2.imshow(query_image)
    ax2.axis('off')

    # Draw ground truth keypoints first (if available, in blue)
    if gt_keypoints:
        gt_kpts_array = np.array(gt_keypoints)
        # Denormalize to query image size
        if gt_kpts_array.max() <= 1.0:
            gt_kpts_array = gt_kpts_array.copy()
            gt_kpts_array[:, 0] *= query_w
            gt_kpts_array[:, 1] *= query_h
        
        ax2.scatter(gt_kpts_array[:, 0], gt_kpts_array[:, 1],
                   c='cyan', s=120, marker='o', edgecolors='black', linewidths=2,
                   label='Ground Truth', zorder=2, alpha=0.7)
        
        # Draw GT skeleton
        if skeleton_edges:
            for edge in skeleton_edges:
                if len(edge) == 2:
                    src_idx, dst_idx = edge
                    if src_idx > 0:
                        src_idx -= 1
                    if dst_idx > 0:
                        dst_idx -= 1

                    if 0 <= src_idx < len(gt_keypoints) and 0 <= dst_idx < len(gt_keypoints):
                        x1, y1 = gt_kpts_array[src_idx]
                        x2, y2 = gt_kpts_array[dst_idx]
                        ax2.plot([x1, x2], [y1, y2], 'cyan', linewidth=2, alpha=0.4, zorder=1)

    # Draw predicted keypoints (on top)
    if pred_keypoints:
        pred_kpts_array = np.array(pred_keypoints)
        # Denormalize to query image size
        if pred_kpts_array.max() <= 1.0:
            pred_kpts_array = pred_kpts_array.copy()
            pred_kpts_array[:, 0] *= query_w
            pred_kpts_array[:, 1] *= query_h

        ax2.scatter(pred_kpts_array[:, 0], pred_kpts_array[:, 1],
                   c='red', s=120, marker='x', linewidths=3,
                   label='Predicted', zorder=3)

        # Draw predicted skeleton
        if skeleton_edges:
            for edge in skeleton_edges:
                if len(edge) == 2:
                    src_idx, dst_idx = edge
                    if src_idx > 0:
                        src_idx -= 1
                    if dst_idx > 0:
                        dst_idx -= 1

                    if 0 <= src_idx < len(pred_keypoints) and 0 <= dst_idx < len(pred_keypoints):
                        x1, y1 = pred_kpts_array[src_idx]
                        x2, y2 = pred_kpts_array[dst_idx]
                        ax2.plot([x1, x2], [y1, y2], 'red', linewidth=2, alpha=0.6, zorder=2)

        # Add keypoint numbers
        for idx, (x, y) in enumerate(pred_kpts_array):
            ax2.text(x, y - 10, str(idx + 1), color='white', fontsize=8,
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_from_checkpoint(args):
    """Run visualization on test images using trained checkpoint."""

    print("=" * 80)
    print("CAPE Pose Estimation Visualization")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (uses saved training args from checkpoint)
    model, train_args = load_model(args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)
    
    print(f"✓ Model loaded and moved to {device}")

    # Load dataset (use training args for dataset config)
    print(f"\nLoading MP-100 test dataset...")

    # Update train_args with our dataset_root and split_id
    train_args.dataset_root = args.dataset_root
    train_args.mp100_split = args.split_id

    dataset = build_mp100_cape('test', train_args)
    print(f"✓ Loaded {len(dataset)} test images")

    # Get categories directly from dataset's COCO annotations
    from collections import defaultdict

    # Load actual test categories from annotation file
    ann_file = Path(train_args.dataset_root) / "annotations" / f"mp100_split{args.split_id}_test.json"
    with open(ann_file) as f:
        test_ann_data = json.load(f)
        test_categories_info = {cat['id']: cat['name'] for cat in test_ann_data['categories']}

    test_categories = list(test_categories_info.keys())

    # Filter to specific categories if requested
    if args.categories is not None:
        test_categories = [c for c in test_categories if c in args.categories]
        print(f"\n⚠️  Filtering to {len(test_categories)} requested categories: {args.categories}")

    print(f"\nTest categories to visualize: {len(test_categories)}")
    if len(test_categories) <= 10:
        print(f"Categories: {[test_categories_info.get(c, c) for c in test_categories]}")
    else:
        print(f"Categories (showing first 10): {[test_categories_info.get(c, c) for c in test_categories[:10]]}...")
    print(f"Samples per category: {args.num_samples}")
    print()

    # OPTIMIZATION: Get category info directly from COCO annotations instead of loading all images
    # This avoids slow GCS file access when iterating through thousands of images
    print("Grouping images by category from annotations (fast method)...")
    category_samples = defaultdict(list)
    
    # Use COCO annotations directly to get image->category mapping
    # This is much faster than loading each image from GCS
    coco = dataset.coco
    for idx, img_id in enumerate(dataset.ids):
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue
        
        anns = coco.loadAnns(ann_ids)
        # Get category from first annotation (we use first instance per image)
        if len(anns) > 0 and 'category_id' in anns[0]:
            cat_id = anns[0]['category_id']
            
            # Filter to requested categories
            if args.categories is not None and cat_id not in args.categories:
                continue
            
            category_samples[cat_id].append(idx)
    
    total_images = sum(len(indices) for indices in category_samples.values())
    print(f"✓ Grouped {total_images} images into {len(category_samples)} categories")

    # Visualize samples
    total_visualized = 0

    for cat_id, sample_indices in category_samples.items():
        cat_name = test_categories_info.get(cat_id, f"cat_{cat_id}")
        print(f"\nCategory: {cat_name} (ID: {cat_id})")

        # Take up to num_samples per category
        for sample_idx in sample_indices[:args.num_samples]:
            try:
                data = dataset[sample_idx]
            except ImageNotFoundError:
                print(f"  Skipping sample {sample_idx} (image not found)")
                continue

            # Get support data (first example in category as template)
            try:
                support_idx = sample_indices[0]
                support_data = dataset[support_idx]
            except ImageNotFoundError:
                # Try to find a valid support example
                support_data = None
                for alt_support_idx in sample_indices:
                    try:
                        support_data = dataset[alt_support_idx]
                        support_idx = alt_support_idx
                        break
                    except ImageNotFoundError:
                        continue
                if support_data is None:
                    print(f"  Skipping category {cat_name} (no valid support image)")
                    continue

            # Skip if support and query are the same image
            if support_idx == sample_idx:
                print(f"  Skipping sample {sample_idx} (same as support)")
                continue

            # Prepare inputs
            query_image = data['image']
            support_image = support_data['image']
            
            # Support keypoints and skeleton (ground truth from support image)
            support_coords = support_data['keypoints']  # Normalized [0,1] coordinates
            support_visibility = data.get('visibility', [1] * len(support_coords))
            support_mask = torch.tensor([v > 0 for v in support_visibility], dtype=torch.float32)
            skeleton_edges = support_data.get('skeleton', [])
            
            # Query ground truth (for PCK computation, NOT passed to model)
            query_gt_coords = data['keypoints']  # Normalized [0,1] coordinates
            query_visibility = data.get('visibility', [1] * len(query_gt_coords))

            # Move to device and prepare for model
            if isinstance(query_image, torch.Tensor):
                query_image_tensor = query_image.unsqueeze(0).to(device)
            else:
                query_image_tensor = torch.from_numpy(np.array(query_image)).permute(2, 0, 1).unsqueeze(0).to(device).float()

            support_coords_tensor = torch.tensor(support_coords, dtype=torch.float32).unsqueeze(0).to(device)
            support_mask_tensor = support_mask.unsqueeze(0).to(device)

            # Run inference with skeleton edges (CRITICAL FIX)
            with torch.no_grad():
                predictions = model.forward_inference(
                    samples=query_image_tensor,
                    support_coords=support_coords_tensor,
                    support_mask=support_mask_tensor,
                    skeleton_edges=[skeleton_edges]  # FIXED: Now passing skeleton structure!
                )

            # Decode predictions
            pred_tokens = predictions['sequences'][0].cpu()
            pred_coords = predictions['coordinates'][0].cpu()

            # Extract keypoints from sequence
            tokenizer = dataset.tokenizer
            pred_keypoints = decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer)

            # Compute PCK if we have ground truth
            pck_score = None
            if query_gt_coords and len(pred_keypoints) > 0:
                from util.eval_utils import compute_pck_bbox
                
                # Get bbox dimensions for normalization
                bbox_w = data.get('bbox_width', 512.0)
                bbox_h = data.get('bbox_height', 512.0)
                
                # Trim to actual number of keypoints for this category
                num_kpts = min(len(pred_keypoints), len(query_gt_coords))
                pred_kpts_trimmed = pred_keypoints[:num_kpts]
                gt_kpts_trimmed = query_gt_coords[:num_kpts]
                vis_trimmed = query_visibility[:num_kpts]
                
                # Compute PCK
                pck_score, num_correct, num_visible = compute_pck_bbox(
                    pred_keypoints=np.array(pred_kpts_trimmed),
                    gt_keypoints=np.array(gt_kpts_trimmed),
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    visibility=np.array(vis_trimmed),
                    threshold=0.2,
                    normalize_by='diagonal'
                )

            # Convert images for visualization
            if isinstance(support_data['image'], torch.Tensor):
                vis_support_image = support_data['image'].permute(1, 2, 0).numpy()
                vis_support_image = (vis_support_image * 255).astype(np.uint8)
            else:
                vis_support_image = support_data['image']
            
            if isinstance(data['image'], torch.Tensor):
                vis_query_image = data['image'].permute(1, 2, 0).numpy()
                vis_query_image = (vis_query_image * 255).astype(np.uint8)
            else:
                vis_query_image = data['image']

            # Visualize with both images
            save_path = output_dir / f"{cat_name}_query{sample_idx}_support{support_idx}.png"
            visualize_pose_prediction(
                support_image=vis_support_image,
                query_image=vis_query_image,
                pred_keypoints=pred_keypoints,
                support_keypoints=support_coords,
                gt_keypoints=query_gt_coords,
                skeleton_edges=skeleton_edges,
                save_path=save_path,
                category_name=cat_name,
                pck_score=pck_score
            )

            pck_str = f" | PCK: {pck_score:.2%}" if pck_score is not None else ""
            print(f"  Sample {sample_idx}: {len(pred_keypoints)} keypoints predicted{pck_str}")
            total_visualized += 1

    print()
    print("=" * 80)
    print(f"✓ Visualization Complete!")
    print(f"=" * 80)
    print(f"  Categories visualized: {len(category_samples)}")
    print(f"  Total images: {total_visualized}")
    print(f"  Saved to: {output_dir}")
    print(f"\nVisualization format:")
    print(f"  - Left panel:  Support image with GT keypoints (green circles)")
    print(f"  - Right panel: Query image with predicted (red X) and GT (cyan circles)")
    print(f"  - PCK@0.2 score shown in title for each prediction")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CAPE Pose Predictions on Test Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize using best PCK checkpoint
  python visualize_cape_predictions.py \\
      --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \\
      --device mps \\
      --num_samples 5

  # Visualize specific epoch
  python visualize_cape_predictions.py \\
      --checkpoint outputs/cape_run/checkpoint_e050_*.pth \\
      --dataset_root . \\
      --output_dir visualizations/epoch_50
        """
    )

    # ========================================================================
    # Model Loading (Primary Arguments)
    # ========================================================================
    parser.add_argument('--checkpoint', required=True,
                       help='Path to trained CAPE checkpoint (.pth file). '
                            'Can be any checkpoint: checkpoint_e***, checkpoint_best_pck_*, etc.')
    parser.add_argument('--device', default='cpu', 
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference (default: cpu)')

    # ========================================================================
    # Dataset Configuration
    # ========================================================================
    parser.add_argument('--dataset_root', default='.', 
                       help='Path to MP-100 dataset root directory (default: current directory)')
    parser.add_argument('--split_id', type=int, default=1, 
                       help='MP-100 split ID (1-5). Must match the split used during training.')

    # ========================================================================
    # Visualization Options
    # ========================================================================
    parser.add_argument('--num_samples', type=int, default=3, 
                       help='Number of samples to visualize per category (default: 3)')
    parser.add_argument('--output_dir', default='visualizations', 
                       help='Directory to save visualization images (default: visualizations/)')
    parser.add_argument('--categories', type=int, nargs='+', default=None,
                       help='Specific category IDs to visualize (default: all test categories). '
                            'Example: --categories 40 55 68')

    # ========================================================================
    # Note: Model architecture parameters are loaded from checkpoint
    # ========================================================================
    # The checkpoint contains 'args' with all training hyperparameters.
    # We don't need to specify them here; they're automatically restored.
    # ========================================================================

    args = parser.parse_args()

    visualize_from_checkpoint(args)


if __name__ == '__main__':
    main()
