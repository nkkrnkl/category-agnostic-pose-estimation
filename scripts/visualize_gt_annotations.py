#!/usr/bin/env python3
"""
Visualize Ground Truth Annotations from MP-100 Dataset.
"""
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from collections import defaultdict
def load_annotations(ann_file):
    """
    Load COCO-format annotation file.
    
    Args:
        ann_file: Path to annotation file
    
    Returns:
        Annotation data dictionary
    """
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {len(data['categories'])}")
    return data
def build_annotation_map(data):
    """
    Build mapping from image_id to annotations and category info.
    
    Args:
        data: Annotation data dictionary
    
    Returns:
        Tuple of (images, categories, img_to_anns)
    """
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    return images, categories, img_to_anns
def parse_keypoints(keypoints):
    """
    Parse keypoint list into (x, y, visibility) tuples.
    
    Args:
        keypoints: List of keypoint values
    
    Returns:
        List of (x, y, visibility) tuples
    """
    kps = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        kps.append((x, y, v))
    return kps
def visualize_annotation(img_path, img_info, annotations, categories, output_path):
    """
    Visualize a single image with its annotations.
    
    Args:
        img_path: Path to image file
        img_info: Image info dictionary
        annotations: List of annotations
        categories: Category dictionary
        output_path: Path to save visualization
    
    Returns:
        bool: Whether visualization was successful
    """
    if not os.path.exists(img_path):
        print(f"  WARNING: Image not found: {img_path}")
        return False
    img = Image.open(img_path)
    num_anns = len(annotations)
    if num_anns == 0:
        return False
    fig, axes = plt.subplots(1, num_anns, figsize=(6 * num_anns, 6))
    if num_anns == 1:
        axes = [axes]
    for idx, (ax, ann) in enumerate(zip(axes, annotations)):
        ax.imshow(img)
        ax.axis('off')
        cat_id = ann['category_id']
        cat_info = categories.get(cat_id, {})
        cat_name = cat_info.get('name', f'cat_{cat_id}')
        skeleton = cat_info.get('skeleton', [])
        kps = parse_keypoints(ann['keypoints'])
        num_kps = ann.get('num_keypoints', len(kps))
        bbox = ann['bbox']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        if skeleton:
            for edge in skeleton:
                kp1_idx, kp2_idx = edge[0] - 1, edge[1] - 1
                if kp1_idx < len(kps) and kp2_idx < len(kps):
                    x1, y1, v1 = kps[kp1_idx]
                    x2, y2, v2 = kps[kp2_idx]
                    if v1 > 0 and v2 > 0:
                        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.6)
        for i, (x, y, v) in enumerate(kps):
            if v == 0:
                continue
            elif v == 1:
                ax.plot(x, y, 'rx', markersize=10, markeredgewidth=2)
            elif v == 2:
                ax.plot(x, y, 'go', markersize=10, markeredgewidth=2)
            ax.text(x + 5, y - 5, str(i), fontsize=8, color='yellow',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        title = f"{cat_name} (cat_id={cat_id})\n"
        title += f"Keypoints: {num_kps}/{len(kps)}\n"
        title += f"Image: {img_info['file_name']}\n"
        title += f"Size: {img_info['width']}x{img_info['height']}"
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True
def visualize_category_summary(img_to_anns, categories, output_path):
    """
    Create a summary visualization showing category distribution.
    
    Args:
        img_to_anns: Mapping from image_id to annotations
        categories: Category dictionary
        output_path: Path to save visualization
    """
    cat_counts = defaultdict(int)
    for anns in img_to_anns.values():
        for ann in anns:
            cat_counts[ann['category_id']] += 1
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    cat_ids = [cat_id for cat_id, _ in sorted_cats]
    counts = [count for _, count in sorted_cats]
    cat_names = [categories.get(cat_id, {}).get('name', f'cat_{cat_id}') for cat_id in cat_ids]
    bars = ax.bar(range(len(cat_ids)), counts, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(cat_ids)))
    ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Annotations', fontsize=12)
    ax.set_title('Category Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               str(count), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Category summary saved to: {output_path}")
def main():
    """
    Main function to visualize ground truth annotations.
    """
    parser = argparse.ArgumentParser(description='Visualize GT annotations from MP-100')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Which split to visualize (train/val/test)')
    parser.add_argument('--data-split', type=str, default='split1',
                       help='Which MP-100 split to use (split1-split5)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--category', type=int, default=None,
                       help='Visualize only this category ID (optional)')
    parser.add_argument('--output-dir', type=str, default='outputs/gt_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for data')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sample selection')
    args = parser.parse_args()
    random.seed(args.random_seed)
    ann_file = os.path.join(args.data_root, 'annotations', 
                           f'mp100_{args.data_split}_{args.split}.json')
    if not os.path.exists(ann_file):
        print(f"ERROR: Annotation file not found: {ann_file}")
        return
    data = load_annotations(ann_file)
    images, categories, img_to_anns = build_annotation_map(data)
    if args.category is not None:
        print(f"\nFiltering for category {args.category}...")
        filtered_img_to_anns = {}
        for img_id, anns in img_to_anns.items():
            cat_anns = [ann for ann in anns if ann['category_id'] == args.category]
            if cat_anns:
                filtered_img_to_anns[img_id] = cat_anns
        img_to_anns = filtered_img_to_anns
        print(f"  Found {len(img_to_anns)} images with category {args.category}")
    output_dir = os.path.join(args.output_dir, f'{args.data_split}_{args.split}')
    if args.category is not None:
        output_dir = os.path.join(output_dir, f'cat_{args.category}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    summary_path = os.path.join(output_dir, 'category_summary.png')
    visualize_category_summary(img_to_anns, categories, summary_path)
    img_ids = list(img_to_anns.keys())
    if len(img_ids) > args.num_samples:
        img_ids = random.sample(img_ids, args.num_samples)
    print(f"\nVisualizing {len(img_ids)} samples...")
    success_count = 0
    for i, img_id in enumerate(img_ids):
        img_info = images[img_id]
        anns = img_to_anns[img_id]
        img_path = os.path.join(args.data_root, img_info['file_name'])
        img_filename = Path(img_info['file_name']).stem
        output_path = os.path.join(output_dir, f'vis_{i:04d}_{img_filename}.png')
        print(f"  [{i+1}/{len(img_ids)}] {img_info['file_name']} ({len(anns)} annotations)")
        success = visualize_annotation(img_path, img_info, anns, categories, output_path)
        if success:
            success_count += 1
    print(f"\n✅ Visualization complete!")
    print(f"  Successfully visualized: {success_count}/{len(img_ids)}")
    print(f"  Output directory: {output_dir}")
    print(f"\nVisualization legend:")
    print("  🟢 Green circles: Labeled and visible keypoints (v=2)")
    print("  ❌ Red X: Labeled but not visible keypoints (v=1)")
    print("  🔵 Cyan dashed box: Bounding box")
    print("  🟢 Green lines: Skeleton edges (if available)")
    print("  🟡 Yellow numbers: Keypoint indices")
    print(f"\nCategory statistics:")
    cat_counts = defaultdict(int)
    for anns in img_to_anns.values():
        for ann in anns:
            cat_counts[ann['category_id']] += 1
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_cats[:10]:
        cat_name = categories.get(cat_id, {}).get('name', f'cat_{cat_id}')
        print(f"  {cat_name} (ID {cat_id}): {count} annotations")
    if len(sorted_cats) > 10:
        print(f"  ... and {len(sorted_cats) - 10} more categories")
if __name__ == '__main__':
    main()