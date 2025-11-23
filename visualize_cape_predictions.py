#!/usr/bin/env python3
"""
Visualize CAPE Pose Predictions

This script:
1. Loads a trained CAPE model
2. Runs inference on test images
3. Visualizes predicted keypoints overlaid on images
4. Saves visualization results
"""

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

    # Build model with saved args
    build_result = build_base_model(train_args)
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
            # Next two values should be x, y coordinates
            if i + 2 < len(pred_coords):
                x, y = pred_coords[i + 1]  # pred_coords stores denormalized values
                keypoints.append((x.item(), y.item()))

        # Stop at EOS token
        elif token == TokenType.eos.value:
            break

    return keypoints


def visualize_pose_prediction(image, pred_keypoints, support_keypoints,
                              skeleton_edges=None, save_path=None,
                              category_name="Unknown"):
    """
    Visualize predicted pose overlaid on image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        pred_keypoints: List of (x, y) predicted keypoint coordinates
        support_keypoints: List of (x, y) support keypoint coordinates (template)
        skeleton_edges: List of [src, dst] edge pairs (0-indexed)
        save_path: Path to save visualization
        category_name: Category name for title
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    h, w = image.shape[:2]

    # Left: Support pose (template)
    ax1.imshow(image)
    ax1.set_title(f"Support Pose Template\n{category_name}", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw support keypoints
    if support_keypoints:
        support_kpts_array = np.array(support_keypoints)
        ax1.scatter(support_kpts_array[:, 0], support_kpts_array[:, 1],
                   c='lime', s=100, marker='o', edgecolors='black', linewidths=2,
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
                        x1, y1 = support_keypoints[src_idx]
                        x2, y2 = support_keypoints[dst_idx]
                        ax1.plot([x1, x2], [y1, y2], 'lime', linewidth=2, alpha=0.6, zorder=2)

        # Add keypoint numbers
        for idx, (x, y) in enumerate(support_keypoints):
            ax1.text(x, y - 10, str(idx + 1), color='white', fontsize=8,
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax1.legend(loc='upper right')

    # Right: Query image with predicted pose
    ax2.imshow(image)
    ax2.set_title(f"Predicted Pose\n{len(pred_keypoints)} keypoints", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Draw predicted keypoints
    if pred_keypoints:
        pred_kpts_array = np.array(pred_keypoints)
        # Denormalize if normalized
        if pred_kpts_array.max() <= 1.0:
            pred_kpts_array[:, 0] *= w
            pred_kpts_array[:, 1] *= h

        ax2.scatter(pred_kpts_array[:, 0], pred_kpts_array[:, 1],
                   c='red', s=100, marker='x', linewidths=3,
                   label='Predicted Keypoints', zorder=3)

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

                    if 0 <= src_idx < len(pred_keypoints) and 0 <= dst_idx < len(pred_keypoints):
                        x1, y1 = pred_kpts_array[src_idx]
                        x2, y2 = pred_kpts_array[dst_idx]
                        ax2.plot([x1, x2], [y1, y2], 'red', linewidth=2, alpha=0.6, zorder=2)

        # Add keypoint numbers
        for idx, (x, y) in enumerate(pred_kpts_array):
            ax2.text(x, y - 10, str(idx + 1), color='white', fontsize=8,
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax2.legend(loc='upper right')

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
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (uses saved training args from checkpoint)
    model, train_args = load_model(args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)

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

    print(f"\nTest categories: {len(test_categories)}")
    print(f"Categories: {list(test_categories_info.values())[:10]}...")
    print(f"Visualizing {args.num_samples} samples per category...")
    print()

    # Sample and visualize
    category_samples = defaultdict(list)

    # Group samples by category (skip missing images)
    skipped_count = 0
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
            cat_id = data['category_id']
            category_samples[cat_id].append(idx)
        except ImageNotFoundError:
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} missing images during dataset iteration")

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

            # Get support data (first example in category)
            try:
                support_idx = sample_indices[0]
                support_data = dataset[support_idx]
            except ImageNotFoundError:
                # Try to find a valid support example
                support_data = None
                for alt_support_idx in sample_indices:
                    try:
                        support_data = dataset[alt_support_idx]
                        break
                    except ImageNotFoundError:
                        continue
                if support_data is None:
                    print(f"  Skipping category {cat_name} (no valid support image)")
                    continue

            # Prepare inputs
            image = data['image']
            support_coords = support_data['keypoints']  # Ground truth support keypoints
            support_mask = torch.ones(len(support_coords))
            skeleton_edges = support_data.get('skeleton', [])

            # Move to device
            if isinstance(image, torch.Tensor):
                image_tensor = image.unsqueeze(0).to(device)
            else:
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device).float()

            support_coords_tensor = torch.tensor(support_coords, dtype=torch.float32).unsqueeze(0).to(device)
            support_mask_tensor = support_mask.unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                predictions = model.forward_inference(
                    samples=image_tensor,
                    support_coords=support_coords_tensor,
                    support_mask=support_mask_tensor
                )

            # Decode predictions
            pred_tokens = predictions['sequences'][0].cpu()
            pred_coords = predictions['coordinates'][0].cpu()

            # Extract keypoints from sequence
            tokenizer = dataset.tokenizer
            pred_keypoints = decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer)

            # Visualize
            if isinstance(data['image'], torch.Tensor):
                vis_image = data['image'].permute(1, 2, 0).numpy()
                vis_image = (vis_image * 255).astype(np.uint8)
            else:
                vis_image = data['image']

            save_path = output_dir / f"{cat_name}_sample_{sample_idx}.png"
            visualize_pose_prediction(
                image=vis_image,
                pred_keypoints=pred_keypoints,
                support_keypoints=support_coords,
                skeleton_edges=skeleton_edges,
                save_path=save_path,
                category_name=cat_name
            )

            print(f"  Sample {sample_idx}: {len(pred_keypoints)} keypoints predicted")
            total_visualized += 1

    print()
    print("=" * 80)
    print(f"✓ Visualization Complete!")
    print(f"  Total images: {total_visualized}")
    print(f"  Saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Visualize CAPE pose predictions')

    # Model
    parser.add_argument('--checkpoint', required=True, help='Path to trained checkpoint')
    parser.add_argument('--device', default='cpu', help='Device (cpu, cuda, mps)')

    # Dataset
    parser.add_argument('--dataset_root', default='.', help='Path to MP-100 dataset')
    parser.add_argument('--split_file', default='category_splits.json', help='Category split file')
    parser.add_argument('--split_id', type=int, default=1, help='MP-100 split ID (1-5)')

    # Visualization
    parser.add_argument('--num_samples', type=int, default=3, help='Samples per category')
    parser.add_argument('--output_dir', default='visualizations', help='Output directory')

    # Model architecture (must match training)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--support_encoder_layers', type=int, default=3)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_feature_levels', type=int, default=4)
    parser.add_argument('--dec_n_points', type=int, default=4)
    parser.add_argument('--enc_n_points', type=int, default=4)
    parser.add_argument('--num_queries', type=int, default=200)
    parser.add_argument('--aux_loss', action='store_true', default=True)
    parser.add_argument('--with_poly_refine', action='store_true', default=True)

    # Dataset params
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--semantic_classes', type=int, default=70)
    parser.add_argument('--num_polys', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_norm', action='store_true', default=True)
    parser.add_argument('--add_cls_token', action='store_true', default=False)
    parser.add_argument('--dataset_name', default='mp100')

    args = parser.parse_args()

    visualize_from_checkpoint(args)


if __name__ == '__main__':
    main()
