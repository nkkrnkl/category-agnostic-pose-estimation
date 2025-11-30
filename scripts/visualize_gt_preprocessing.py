#!/usr/bin/env python3
"""
Visualize Ground Truth Annotations: Original vs. Preprocessed.
"""
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random
from collections import defaultdict
import albumentations as A
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
def apply_preprocessing(img, keypoints, bbox):
    """
    Apply preprocessing pipeline.
    
    Args:
        img: PIL Image
        keypoints: List of (x, y, v) tuples in original image coordinates
        bbox: [x, y, width, height] in original image coordinates
    
    Returns:
        Tuple of (processed_img, processed_kps)
    """
    img_np = np.array(img)
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    bbox_x, bbox_y, bbox_w, bbox_h = int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)
    img_cropped = img_np[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
    if img_cropped.size == 0 or img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
        print(f"  WARNING: Empty crop with bbox {bbox}")
        return None, None
    kps_relative = []
    kps_xy = []
    for x, y, v in keypoints:
        x_rel = x - bbox_x
        y_rel = y - bbox_y
        kps_relative.append((x_rel, y_rel, v))
        kps_xy.append((x_rel, y_rel))
    transform = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    transformed = transform(image=img_cropped, keypoints=kps_xy)
    img_512 = transformed['image']
    kps_512_xy = transformed['keypoints']
    kps_512 = [(x, y, v) for (x, y), (_, _, v) in zip(kps_512_xy, kps_relative)]
    return img_512, kps_512
def draw_keypoints_and_skeleton(ax, img, keypoints, skeleton, title, show_bbox=False, bbox=None):
    """
    Draw keypoints and skeleton on an axis.
    
    Args:
        ax: Matplotlib axis
        img: Image (PIL or numpy array)
        keypoints: List of (x, y, v) tuples
        skeleton: List of [i, j] edge pairs
        title: Title for the plot
        show_bbox: Whether to draw bounding box
        bbox: [x, y, width, height] bounding box
    """
    ax.imshow(img)
    ax.axis('off')
    if show_bbox and bbox is not None:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--',
            label='Bbox'
        )
        ax.add_patch(rect)
    if skeleton:
        for edge in skeleton:
            kp1_idx, kp2_idx = edge[0] - 1, edge[1] - 1
            if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                x1, y1, v1 = keypoints[kp1_idx]
                x2, y2, v2 = keypoints[kp2_idx]
                if v1 > 0 and v2 > 0:
                    ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.6)
    for i, (x, y, v) in enumerate(keypoints):
        if v == 0:
            continue
        elif v == 1:
            ax.plot(x, y, 'rx', markersize=12, markeredgewidth=3)
        elif v == 2:
            ax.plot(x, y, 'go', markersize=12, markeredgewidth=3)
        ax.text(x + 5, y - 5, str(i), fontsize=10, color='yellow', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    ax.set_title(title, fontsize=12, fontweight='bold')
def visualize_preprocessing(img_path, img_info, ann, categories, output_path):
    """
    Visualize preprocessing: original vs. processed image.
    
    Args:
        img_path: Path to image file
        img_info: Image info dictionary
        ann: Annotation dictionary
        categories: Category dictionary
        output_path: Path to save visualization
    
    Returns:
        bool: Whether visualization was successful
    """
    if not os.path.exists(img_path):
        print(f"  WARNING: Image not found: {img_path}")
        return False
    img = Image.open(img_path).convert('RGB')
    cat_id = ann['category_id']
    cat_info = categories.get(cat_id, {})
    cat_name = cat_info.get('name', f'cat_{cat_id}')
    skeleton = cat_info.get('skeleton', [])
    keypoints_orig = parse_keypoints(ann['keypoints'])
    bbox = ann['bbox']
    num_kps = ann.get('num_keypoints', len(keypoints_orig))
    img_processed, keypoints_processed = apply_preprocessing(img, keypoints_orig, bbox)
    if img_processed is None:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    title_orig = f"ORIGINAL IMAGE\n{cat_name} (ID {cat_id})\n"
    title_orig += f"Size: {img_info['width']}x{img_info['height']}\n"
    title_orig += f"Bbox: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
    draw_keypoints_and_skeleton(
        ax1, img, keypoints_orig, skeleton, title_orig,
        show_bbox=True, bbox=bbox
    )
    title_proc = f"PREPROCESSED (Training Pipeline)\n{cat_name} (ID {cat_id})\n"
    title_proc += f"Size: 512x512 (resized from {int(bbox[2])}x{int(bbox[3])})\n"
    title_proc += f"Keypoints: {num_kps} visible"
    draw_keypoints_and_skeleton(
        ax2, img_processed, keypoints_processed, skeleton, title_proc,
        show_bbox=False, bbox=None
    )
    fig.suptitle(
        f"Ground Truth Preprocessing Verification\nImage: {img_info['file_name']}",
        fontsize=14, fontweight='bold', y=0.98
    )
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10,
               label='Visible (v=2)', markeredgewidth=2),
        Line2D([0], [0], marker='x', color='r', markersize=10,
               label='Not Visible (v=1)', markeredgewidth=2),
        Line2D([0], [0], color='g', linewidth=2, label='Skeleton'),
        patches.Patch(edgecolor='cyan', facecolor='none', linewidth=2,
                     linestyle='--', label='Bbox (original only)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True
def main():
    """
    Main function to visualize preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Visualize GT annotations: Original vs. Preprocessed'
    )
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Which split to visualize')
    parser.add_argument('--data-split', type=str, default='split1',
                       help='Which MP-100 split to use (split1-split5)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--category', type=int, default=None,
                       help='Visualize only this category ID (optional)')
    parser.add_argument('--output-dir', type=str, default='outputs/gt_preprocessing_vis',
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
    img_ids = []
    for img_id, anns in img_to_anns.items():
        if len(anns) == 1:
            img_ids.append(img_id)
    if len(img_ids) < args.num_samples:
        for img_id in img_to_anns.keys():
            if img_id not in img_ids:
                img_ids.append(img_id)
    if len(img_ids) > args.num_samples:
        img_ids = random.sample(img_ids, args.num_samples)
    print(f"\nVisualizing {len(img_ids)} samples...")
    print(f"Legend:")
    print(f"  LEFT panel:  Original unscaled image with bbox (cyan dashed box)")
    print(f"  RIGHT panel: Preprocessed 512x512 image (cropped to bbox + resized)")
    print(f"  🟢 Green circles: Visible keypoints (v=2)")
    print(f"  ❌ Red X: Not visible keypoints (v=1)")
    print(f"  🟢 Green lines: Skeleton edges")
    print()
    success_count = 0
    for i, img_id in enumerate(img_ids):
        img_info = images[img_id]
        anns = img_to_anns[img_id]
        ann = anns[0]
        img_path = os.path.join(args.data_root, img_info['file_name'])
        img_filename = Path(img_info['file_name']).stem
        output_path = os.path.join(output_dir, f'vis_{i:04d}_{img_filename}.png')
        cat_name = categories.get(ann['category_id'], {}).get('name', f"cat_{ann['category_id']}")
        print(f"  [{i+1}/{len(img_ids)}] {img_info['file_name']} ({cat_name})")
        success = visualize_preprocessing(img_path, img_info, ann, categories, output_path)
        if success:
            success_count += 1
    print(f"\n✅ Preprocessing visualization complete!")
    print(f"  Successfully visualized: {success_count}/{len(img_ids)}")
    print(f"  Output directory: {output_dir}")
    print(f"\n📊 What to check:")
    print(f"  ✓ Keypoints are inside bbox in original image (LEFT)")
    print(f"  ✓ Keypoints are correctly positioned in 512x512 image (RIGHT)")
    print(f"  ✓ Skeleton edges connect correct keypoints in both images")
    print(f"  ✓ Visibility markers (green/red) match in both images")
    print(f"  ✓ Bbox cropping doesn't cut off any keypoints")
if __name__ == '__main__':
    main()