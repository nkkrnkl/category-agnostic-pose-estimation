#!/usr/bin/env python3
"""
Simple CAPE Results Visualization Script

This script visualizes pose estimation results without needing to reload the full model.
It works with saved predictions or by running inference on a few test images.
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import torch
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

from datasets.mp100_cape import build_mp100_cape, ImageNotFoundError


def load_category_info(dataset_root, split_id=1):
    """Load category information from annotations."""
    ann_file = Path(dataset_root) / "annotations" / f"mp100_split{split_id}_test.json"
    with open(ann_file) as f:
        data = json.load(f)
    
    categories = {}
    for cat in data['categories']:
        categories[cat['id']] = {
            'name': cat['name'],
            'keypoints': cat.get('keypoints', []),
            'skeleton': cat.get('skeleton', [])
        }
    
    return categories


def draw_keypoints_and_skeleton(ax, image, keypoints, skeleton, visibility=None, 
                                 title="", color='red', marker='x'):
    """Draw keypoints and skeleton on image."""
    ax.imshow(image)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    if len(keypoints) == 0:
        return
    
    keypoints = np.array(keypoints)
    
    # Filter by visibility if provided
    if visibility is not None:
        visibility = np.array(visibility)
        visible_mask = visibility > 0
        keypoints_to_draw = keypoints[visible_mask]
    else:
        keypoints_to_draw = keypoints
        visible_mask = np.ones(len(keypoints), dtype=bool)
    
    # Draw skeleton edges
    if skeleton and len(skeleton) > 0:
        for edge in skeleton:
            if len(edge) == 2:
                src_idx, dst_idx = edge
                # Convert 1-indexed to 0-indexed
                if src_idx > 0:
                    src_idx -= 1
                if dst_idx > 0:
                    dst_idx -= 1
                
                # Check if both endpoints are visible and within bounds
                if (0 <= src_idx < len(keypoints) and 
                    0 <= dst_idx < len(keypoints) and
                    (visibility is None or (visibility[src_idx] > 0 and visibility[dst_idx] > 0))):
                    x1, y1 = keypoints[src_idx]
                    x2, y2 = keypoints[dst_idx]
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.6, zorder=1)
    
    # Draw keypoints
    if len(keypoints_to_draw) > 0:
        ax.scatter(keypoints_to_draw[:, 0], keypoints_to_draw[:, 1],
                  c=color, s=100, marker=marker, linewidths=3,
                  label='Keypoints', zorder=2)
    
    # Add keypoint numbers
    for idx, (x, y) in enumerate(keypoints):
        if visibility is None or visibility[idx] > 0:
            ax.text(x, y - 10, str(idx + 1), color='white', fontsize=8,
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                   zorder=3)


def visualize_ground_truth_samples(args):
    """Visualize ground truth samples from the dataset."""
    print("=" * 80)
    print("CAPE Ground Truth Visualization")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading test dataset from {args.dataset_root}...")
    
    # Create dummy args for dataset
    class DatasetArgs:
        def __init__(self, dataset_root, split_id):
            self.dataset_root = dataset_root
            self.mp100_split = split_id
            self.semantic_classes = 70
            self.image_norm = False
            self.vocab_size = 2000
            self.seq_len = 200
    
    dataset_args = DatasetArgs(args.dataset_root, args.split_id)
    
    try:
        dataset = build_mp100_cape('test', dataset_args)
        print(f"✓ Loaded {len(dataset)} test images")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Load category information
    categories = load_category_info(args.dataset_root, args.split_id)
    print(f"✓ Loaded {len(categories)} categories")
    
    # Group images by category
    print("\nGrouping images by category...")
    category_images = defaultdict(list)
    
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
            cat_id = data.get('category_id', 0)
            if cat_id in categories:
                category_images[cat_id].append(idx)
        except ImageNotFoundError:
            continue
        except Exception as e:
            continue
    
    print(f"✓ Found images for {len(category_images)} categories")
    
    # Visualize samples from each category
    print(f"\nVisualizing {args.num_samples} samples per category...")
    print()
    
    total_visualized = 0
    
    for cat_id, image_indices in sorted(category_images.items()):
        cat_info = categories[cat_id]
        cat_name = cat_info['name']
        skeleton = cat_info['skeleton']
        
        print(f"Category: {cat_name} (ID: {cat_id})")
        
        # Sample up to num_samples images
        sample_indices = image_indices[:args.num_samples]
        
        for sample_idx in sample_indices:
            try:
                data = dataset[sample_idx]
                
                # Get image and keypoints
                image = data['image']
                if isinstance(image, torch.Tensor):
                    image = image.permute(1, 2, 0).cpu().numpy()
                    image = (image * 255).astype(np.uint8)
                elif not isinstance(image, np.ndarray):
                    image = np.array(image)
                
                keypoints = data.get('keypoints', [])
                visibility = data.get('visibility', None)
                
                # Create visualization
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                draw_keypoints_and_skeleton(
                    ax, image, keypoints, skeleton, visibility,
                    title=f"{cat_name}\nImage ID: {data.get('image_id', 'unknown')}\n"
                          f"{len(keypoints)} keypoints ({sum(v > 0 for v in visibility) if visibility else len(keypoints)} visible)",
                    color='lime',
                    marker='o'
                )
                
                # Save
                save_path = output_dir / f"{cat_name}_idx{sample_idx}_id{data.get('image_id', 'unknown')}.png"
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Saved: {save_path.name}")
                total_visualized += 1
                
            except ImageNotFoundError:
                print(f"  ✗ Skipped sample {sample_idx} (image not found)")
                continue
            except Exception as e:
                print(f"  ✗ Error processing sample {sample_idx}: {e}")
                continue
        
        print()
    
    print("=" * 80)
    print(f"✓ Visualization Complete!")
    print(f"  Total images visualized: {total_visualized}")
    print(f"  Saved to: {output_dir}")
    print("=" * 80)


def visualize_predictions_vs_gt(args):
    """Visualize predictions vs ground truth (requires predictions file)."""
    print("=" * 80)
    print("CAPE Predictions vs Ground Truth Visualization")
    print("=" * 80)
    print()
    
    if not args.predictions_file:
        print("✗ Error: --predictions_file required for this mode")
        print("  Run with --mode gt to visualize ground truth only")
        return
    
    predictions_path = Path(args.predictions_file)
    if not predictions_path.exists():
        print(f"✗ Error: Predictions file not found: {predictions_path}")
        return
    
    # Load predictions
    print(f"Loading predictions from {predictions_path}...")
    with open(predictions_path) as f:
        predictions = json.load(f)
    
    print(f"✓ Loaded {len(predictions)} predictions")
    
    # TODO: Implement side-by-side visualization
    # This would require predictions in a specific format
    print("Note: Predictions visualization not yet implemented")
    print("      Use --mode gt to visualize ground truth annotations")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CAPE pose estimation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize ground truth from test set
  python visualize_results_simple.py --mode gt --dataset_root . --num_samples 3

  # Visualize specific category
  python visualize_results_simple.py --mode gt --dataset_root . --category_filter "horse"

  # Visualize predictions vs ground truth (requires predictions file)
  python visualize_results_simple.py --mode pred --predictions_file predictions.json
        """
    )
    
    # Mode
    parser.add_argument('--mode', choices=['gt', 'pred'], default='gt',
                       help='Visualization mode: gt=ground truth only, pred=predictions vs GT')
    
    # Dataset
    parser.add_argument('--dataset_root', default='.', 
                       help='Path to MP-100 dataset root')
    parser.add_argument('--split_id', type=int, default=1,
                       help='MP-100 split ID (1-5)')
    
    # Sampling
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize per category')
    parser.add_argument('--category_filter', type=str, default=None,
                       help='Only visualize categories containing this string')
    
    # Predictions (for pred mode)
    parser.add_argument('--predictions_file', type=str, default=None,
                       help='Path to predictions JSON file (for --mode pred)')
    
    # Output
    parser.add_argument('--output_dir', default='visualizations/simple',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.mode == 'gt':
        visualize_ground_truth_samples(args)
    elif args.mode == 'pred':
        visualize_predictions_vs_gt(args)


if __name__ == '__main__':
    main()

