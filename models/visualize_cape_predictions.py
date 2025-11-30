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
import os
import sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
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
TARGET_SIZE = 512
def ensure_image_size(image, target_size=TARGET_SIZE):
    """
    Ensure image is exactly target_size x target_size.
    If not, resize it (preserving aspect ratio and padding if needed).
    Args:
        image: numpy array (H, W, 3) or PIL Image
        target_size: Target size (default: 512)
    Returns:
        numpy array (target_size, target_size, 3) in uint8 format
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image
    import cv2
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    if resized.dtype != np.uint8:
        if resized.max() <= 1.0:
            resized = (resized * 255).astype(np.uint8)
        else:
            resized = resized.astype(np.uint8)
    return resized
def load_model(checkpoint_path):
    """Load trained CAPE model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    print(f"  Using training config (epoch {checkpoint.get('epoch', '?')})")
    import numpy as np
    num_bins = int(np.sqrt(train_args.vocab_size))
    tokenizer = DiscreteTokenizer(
        num_bins=num_bins,
        seq_len=train_args.seq_len,
        add_cls=getattr(train_args, 'add_cls_token', False)
    )
    build_result = build_base_model(train_args, train=False, tokenizer=tokenizer)
    if isinstance(build_result, tuple):
        base_model = build_result[0]
    else:
        base_model = build_result
    model = build_cape_model(train_args, base_model)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
    model.eval()
    print(f"✓ Model loaded successfully")
    return model, train_args
def decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer, max_keypoints=None):
    """
    Decode predicted token sequence to keypoint coordinates.
    Args:
        pred_tokens: (seq_len,) Token predictions
        pred_coords: (seq_len, 2) Coordinate predictions (normalized [0,1] relative to 512x512)
        tokenizer: Tokenizer for decoding
        max_keypoints: Optional maximum number of keypoints to extract (stops early if reached)
    Returns:
        keypoints: List of (x, y) coordinates in normalized [0,1] space (relative to 512x512)
    """
    keypoints = []
    for i in range(len(pred_tokens)):
        if max_keypoints is not None and len(keypoints) >= max_keypoints:
            break
        token = pred_tokens[i].item()
        if token == TokenType.coord.value:
            if i < len(pred_coords):
                x, y = pred_coords[i]
                x_val, y_val = x.item(), y.item()
                x_val = max(0.0, min(1.0, x_val))
                y_val = max(0.0, min(1.0, y_val))
                if (np.isfinite(x_val) and np.isfinite(y_val)):
                    keypoints.append((x_val, y_val))
        elif token == TokenType.eos.value:
            break
    return keypoints
def visualize_pose_prediction(support_image, query_image, pred_keypoints, 
                              support_keypoints, gt_keypoints=None,
                              skeleton_edges=None, save_path=None,
                              category_name="Unknown", pck_score=None,
                              debug_coords=False):
    """
    Visualize predicted pose vs ground truth and support template.
    Creates a 3-panel layout: Support (GT), Ground Truth, Predicted.
    CRITICAL: All keypoints are normalized [0,1] relative to 512x512 resized images.
    The images passed in MUST be 512x512 (resized) to match the coordinate frame.
    Args:
        support_image: PIL Image or numpy array (H, W, 3) - MUST be 512x512 resized image
        query_image: PIL Image or numpy array (H, W, 3) - MUST be 512x512 resized image
        pred_keypoints: List of (x, y) predicted keypoint coordinates (normalized [0,1] relative to 512x512)
        support_keypoints: List of (x, y) support keypoint coordinates (normalized [0,1] relative to 512x512)
        gt_keypoints: Optional list of (x, y) ground truth query keypoints (normalized [0,1] relative to 512x512)
        skeleton_edges: List of [src, dst] edge pairs (1-indexed from MP-100)
        save_path: Path to save visualization
        category_name: Category name for title
        pck_score: Optional PCK score to display
        debug_coords: If True, print coordinate comparison for debugging
    """
    if isinstance(support_image, Image.Image):
        support_image = np.array(support_image)
    if isinstance(query_image, Image.Image):
        query_image = np.array(query_image)
    support_h, support_w = support_image.shape[:2]
    query_h, query_w = query_image.shape[:2]
    if support_w != TARGET_SIZE or support_h != TARGET_SIZE:
        import warnings
        warnings.warn(
            f"Support image size ({support_w}x{support_h}) != {TARGET_SIZE}x{TARGET_SIZE}. "
            f"Coordinates are normalized relative to {TARGET_SIZE}x{TARGET_SIZE}. "
            f"Visualization may be misaligned!"
        )
    if query_w != TARGET_SIZE or query_h != TARGET_SIZE:
        import warnings
        warnings.warn(
            f"Query image size ({query_w}x{query_h}) != {TARGET_SIZE}x{TARGET_SIZE}. "
            f"Coordinates are normalized relative to {TARGET_SIZE}x{TARGET_SIZE}. "
            f"Visualization may be misaligned!"
        )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    ax1.imshow(support_image)
    ax1.set_title("Support (GT)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    if support_keypoints:
        support_kpts_array = np.array(support_keypoints)
        if support_kpts_array.max() <= 1.0:
            support_kpts_array = support_kpts_array.copy()
            support_kpts_array[:, 0] *= TARGET_SIZE
            support_kpts_array[:, 1] *= TARGET_SIZE
        else:
            support_kpts_array = support_kpts_array.copy()
            support_kpts_array[:, 0] = np.clip(support_kpts_array[:, 0], 0, support_w - 1)
            support_kpts_array[:, 1] = np.clip(support_kpts_array[:, 1], 0, support_h - 1)
        ax1.scatter(support_kpts_array[:, 0], support_kpts_array[:, 1],
                   c='lime', s=120, marker='o', edgecolors='black', linewidths=2,
                   label='Ground Truth', zorder=3)
        if skeleton_edges:
            for edge in skeleton_edges:
                if len(edge) == 2:
                    src_idx, dst_idx = edge
                    if src_idx > 0:
                        src_idx -= 1
                    if dst_idx > 0:
                        dst_idx -= 1
                    if 0 <= src_idx < len(support_keypoints) and 0 <= dst_idx < len(support_keypoints):
                        x1, y1 = support_kpts_array[src_idx]
                        x2, y2 = support_kpts_array[dst_idx]
                        ax1.plot([x1, x2], [y1, y2], 'lime', linewidth=2, alpha=0.6, zorder=2)
    ax1.legend(loc='upper right', fontsize=10)
    title_text = "Ground Truth"
    if pck_score is not None:
        title_text += f"\nPCK@0.2: {pck_score:.1%}"
    ax2.set_title(title_text, fontsize=14, fontweight='bold')
    ax2.imshow(query_image)
    ax2.axis('off')
    if gt_keypoints:
        gt_kpts_array = np.array(gt_keypoints)
        if gt_kpts_array.max() <= 1.0:
            gt_kpts_array = gt_kpts_array.copy()
            gt_kpts_array[:, 0] *= TARGET_SIZE
            gt_kpts_array[:, 1] *= TARGET_SIZE
        else:
            gt_kpts_array = gt_kpts_array.copy()
            gt_kpts_array[:, 0] = np.clip(gt_kpts_array[:, 0], 0, query_w - 1)
            gt_kpts_array[:, 1] = np.clip(gt_kpts_array[:, 1], 0, query_h - 1)
        ax2.scatter(gt_kpts_array[:, 0], gt_kpts_array[:, 1],
                   c='cyan', s=120, marker='o', edgecolors='black', linewidths=2,
                   label='Ground Truth', zorder=2)
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
                        ax2.plot([x1, x2], [y1, y2], 'cyan', linewidth=2, alpha=0.6, zorder=1)
    ax2.legend(loc='upper right', fontsize=10)
    title_text = "Predicted"
    if pck_score is not None:
        title_text += f"\nPCK@0.2: {pck_score:.1%}"
    else:
        title_text += "\nPCK: N/A"
    ax3.set_title(title_text, fontsize=14, fontweight='bold')
    ax3.imshow(query_image)
    ax3.axis('off')
    if pred_keypoints:
        pred_kpts_array = np.array(pred_keypoints)
        if gt_keypoints is not None:
            num_gt = len(gt_keypoints)
            num_pred = len(pred_kpts_array)
            if num_pred > num_gt:
                if debug_coords:
                    print(f"  ⚠️  Model predicted {num_pred} keypoints but GT has {num_gt} (extra predictions shown)")
        if pred_kpts_array.max() <= 1.0:
            pred_kpts_array = pred_kpts_array.copy()
            pred_kpts_array[:, 0] *= TARGET_SIZE
            pred_kpts_array[:, 1] *= TARGET_SIZE
        else:
            pred_kpts_array = pred_kpts_array.copy()
            pred_kpts_array[:, 0] = np.clip(pred_kpts_array[:, 0], 0, query_w - 1)
            pred_kpts_array[:, 1] = np.clip(pred_kpts_array[:, 1], 0, query_h - 1)
        ax3.scatter(pred_kpts_array[:, 0], pred_kpts_array[:, 1],
                   c='red', s=120, marker='x', linewidths=3,
                   label='Predicted', zorder=3)
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
                        ax3.plot([x1, x2], [y1, y2], 'red', linewidth=2, alpha=0.6, zorder=2)
    ax3.legend(loc='upper right', fontsize=10)
    if debug_coords and gt_keypoints and pred_keypoints:
        if pck_score is not None:
            print(f"\n  PCK@0.2 score: {pck_score:.1%}")
            print(f"  NOTE: This is PCK for THIS SPECIFIC IMAGE, not the validation set average.")
        print(f"\n{'='*80}")
        print(f"COORDINATE COMPARISON DEBUG: {category_name}")
        print(f"{'='*80}")
        gt_array = np.array(gt_keypoints)
        pred_array = np.array(pred_keypoints)
        if gt_array.max() <= 1.0:
            gt_px = gt_array.copy() * TARGET_SIZE
        else:
            gt_px = gt_array.copy()
        if pred_array.max() <= 1.0:
            pred_px = pred_array.copy() * TARGET_SIZE
        else:
            pred_px = pred_array.copy()
        num_kpts = min(len(gt_px), len(pred_px))
        if num_kpts > 0:
            gt_trimmed = gt_px[:num_kpts]
            pred_trimmed = pred_px[:num_kpts]
            abs_diff = np.abs(gt_trimmed - pred_trimmed)
            euclidean_dist = np.sqrt(np.sum(abs_diff ** 2, axis=1))
            num_pred_total = len(pred_keypoints) if pred_keypoints else 0
            num_gt_total = len(gt_keypoints) if gt_keypoints else 0
            if num_pred_total != num_gt_total:
                print(f"  ⚠️  Keypoint count mismatch: {num_pred_total} predicted vs {num_gt_total} GT")
                print(f"     (PCK computed on first {num_kpts} keypoints)")
            print(f"  Number of keypoints (for comparison): {num_kpts}")
            print(f"  GT coordinates (pixels, first 5):")
            for i in range(min(5, num_kpts)):
                print(f"    Kpt {i}: ({gt_trimmed[i,0]:.2f}, {gt_trimmed[i,1]:.2f})")
            print(f"  Pred coordinates (pixels, first 5):")
            for i in range(min(5, num_kpts)):
                print(f"    Kpt {i}: ({pred_trimmed[i,0]:.2f}, {pred_trimmed[i,1]:.2f})")
            print(f"  Absolute differences (pixels, first 5):")
            for i in range(min(5, num_kpts)):
                print(f"    Kpt {i}: dx={abs_diff[i,0]:.2f}, dy={abs_diff[i,1]:.2f}, dist={euclidean_dist[i]:.2f}")
            mean_dist = np.mean(euclidean_dist)
            max_dist = np.max(euclidean_dist)
            print(f"  Mean Euclidean distance: {mean_dist:.2f} pixels")
            print(f"  Max Euclidean distance: {max_dist:.2f} pixels")
            if pck_score is not None:
                threshold_px = 0.2 * np.sqrt(TARGET_SIZE**2 + TARGET_SIZE**2)
                print(f"  PCK@0.2 threshold: {threshold_px:.2f} pixels")
                num_within_threshold = (euclidean_dist < threshold_px).sum()
                print(f"  Keypoints within threshold: {num_within_threshold}/{num_kpts} ({num_within_threshold/num_kpts:.1%})")
            if pck_score is not None and pck_score >= 0.99:
                if mean_dist > 5.0:
                    print(f"  ⚠️  WARNING: PCK={pck_score:.1%} but mean distance={mean_dist:.2f}px!")
                    print(f"     This indicates a visualization/coordinate bug, not a model accuracy issue.")
                else:
                    print(f"  ✓ Coordinates match well (mean dist={mean_dist:.2f}px) - visualization should align")
        print(f"{'='*80}\n")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, train_args = load_model(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_epoch = checkpoint.get('epoch', 'unknown')
    device = torch.device(args.device)
    model = model.to(device)
    print(f"✓ Model loaded and moved to {device}")
    print(f"  Checkpoint epoch: {checkpoint_epoch}")
    if 'val_pck' in checkpoint:
        val_pck = checkpoint.get('val_pck', 0.0)
        print(f"  Checkpoint validation PCK: {val_pck:.2%}")
        print(f"  ⚠️  NOTE: This PCK was computed on the VALIDATION SET (all images).")
        print(f"     Individual images may have different PCK scores.")
        if val_pck < 0.5:
            print(f"  ⚠️  WARNING: Low validation PCK ({val_pck:.2%}) - this checkpoint may not be fully trained!")
    elif 'val_stats' in checkpoint:
        val_stats = checkpoint.get('val_stats', {})
        val_pck = val_stats.get('pck', 0.0)
        if val_pck > 0:
            print(f"  Checkpoint validation PCK: {val_pck:.2%}")
            print(f"  ⚠️  NOTE: This PCK was computed on the VALIDATION SET (all images).")
            print(f"     Individual images may have different PCK scores.")
            if val_pck < 0.5:
                print(f"  ⚠️  WARNING: Low validation PCK ({val_pck:.2%}) - this checkpoint may not be fully trained!")
    print(f"\nLoading MP-100 dataset...")
    train_args.dataset_root = args.dataset_root
    train_args.mp100_split = args.split_id
    if args.single_image_path:
        print(f"\n⚠️  Single image mode: {args.single_image_path}")
        print("   Using this SAME image as both support and query (for overfitting visualization)")
        try:
            dataset = build_mp100_cape('train', train_args)
            print(f"✓ Loaded train dataset: {len(dataset)} images")
        except:
            dataset = build_mp100_cape('test', train_args)
            print(f"✓ Loaded test dataset: {len(dataset)} images")
        single_image_path = Path(args.single_image_path)
        if not single_image_path.is_absolute():
            single_image_path = Path(args.dataset_root) / single_image_path
        found_idx = None
        target_filename = single_image_path.name
        rel_path = str(single_image_path.relative_to(Path(args.dataset_root)))
        coco = dataset.coco
        for img_id in dataset.ids:
            img_info = coco.loadImgs(img_id)[0]
            img_file = img_info['file_name']
            if img_file.endswith(target_filename) or img_file == rel_path or img_file.endswith(rel_path):
                try:
                    idx = dataset.ids.index(img_id)
                    found_idx = idx
                    break
                except ValueError:
                    continue
        if found_idx is None:
            print(f"❌ Image not found in dataset by filename: {args.single_image_path}")
            print("   Trying to load via COCO annotations and mirror dataset preprocessing...")
            if single_image_path.exists():
                from PIL import Image as PILImage
                import json
                ann_file = Path(train_args.dataset_root) / "annotations" / f"mp100_split{args.split_id}_train.json"
                if not ann_file.exists():
                    ann_file = Path(train_args.dataset_root) / "annotations" / f"mp100_split{args.split_id}_test.json"
                if ann_file.exists():
                    with open(ann_file) as f:
                        ann_data = json.load(f)
                    rel_path = str(single_image_path.relative_to(Path(args.dataset_root)))
                    img_info = None
                    for img in ann_data['images']:
                        if img['file_name'] == rel_path or img['file_name'].endswith(single_image_path.name):
                            img_info = img
                            break
                    if img_info:
                        if img_info['id'] in dataset.ids:
                            idx = dataset.ids.index(img_info['id'])
                            print(f"  ✓ Found matching image_id={img_info['id']} in dataset; "
                                  f"using dataset sample at index {idx} for preprocessing.")
                            support_data = dataset[idx]
                            query_data = support_data
                        else:
                            print("  ⚠️  image_id not found in dataset.ids; "
                                  "rebuilding preprocessing manually to match MP100CAPE.")
                            img = PILImage.open(single_image_path).convert('RGB')
                            img_np = np.array(img)
                            anns = [a for a in ann_data['annotations'] if a['image_id'] == img_info['id']]
                            if not anns:
                                print(f"❌ No annotations found for image id {img_info['id']} in COCO file.")
                                return
                            ann = anns[0]
                            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                            bbox_x = max(0, int(bbox_x))
                            bbox_y = max(0, int(bbox_y))
                            bbox_w = max(1, int(bbox_w))
                            bbox_h = max(1, int(bbox_h))
                            img_cropped = img_np[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]
                            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                            num_kpts = keypoints.shape[0]
                            visibility = keypoints[:, 2].tolist()
                            kpts_xy = keypoints[:, :2].copy()
                            kpts_xy[:, 0] -= bbox_x
                            kpts_xy[:, 1] -= bbox_y
                            try:
                                import albumentations as A
                                resize_transform = A.Compose(
                                    [A.Resize(height=TARGET_SIZE, width=TARGET_SIZE)],
                                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
                                )
                                transformed = resize_transform(image=img_cropped, keypoints=kpts_xy.tolist())
                                img_resized = transformed['image']
                                kpts_resized = np.array(transformed['keypoints'], dtype=np.float32)
                            except ImportError:
                                import cv2
                                h0, w0 = img_cropped.shape[:2]
                                img_resized = cv2.resize(
                                    img_cropped, (TARGET_SIZE, TARGET_SIZE),
                                    interpolation=cv2.INTER_CUBIC
                                )
                                scale_x = TARGET_SIZE / max(1.0, w0)
                                scale_y = TARGET_SIZE / max(1.0, h0)
                                kpts_resized = kpts_xy.copy().astype(np.float32)
                                kpts_resized[:, 0] *= scale_x
                                kpts_resized[:, 1] *= scale_y
                            support_data = {
                                'image': img_resized,
                                'keypoints': kpts_resized.tolist(),
                                'visibility': visibility,
                                'image_path': str(single_image_path),
                                'skeleton': ann.get('skeleton', []),
                                'width': TARGET_SIZE,
                                'height': TARGET_SIZE,
                                'bbox_width': TARGET_SIZE,
                                'bbox_height': TARGET_SIZE,
                            }
                            query_data = support_data
                        query_image = query_data['image']
                        support_image = support_data['image']
                        support_coords = support_data['keypoints']
                        support_visibility = support_data.get('visibility', [1] * len(support_coords))
                        support_mask = torch.tensor([v > 0 for v in support_visibility], dtype=torch.float32)
                        skeleton_edges = support_data.get('skeleton', [])
                        query_gt_coords = query_data['keypoints']
                        if isinstance(query_image, PILImage.Image):
                            query_image_np = np.array(query_image)
                        else:
                            query_image_np = np.array(query_image)
                        query_image_tensor = torch.from_numpy(query_image_np).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
                        support_coords_tensor = torch.tensor(support_coords, dtype=torch.float32).unsqueeze(0).to(device)
                        support_mask_tensor = support_mask.unsqueeze(0).to(device)
                        with torch.no_grad():
                            predictions = model.forward_inference(
                                samples=query_image_tensor,
                                support_coords=support_coords_tensor,
                                support_mask=support_mask_tensor,
                                skeleton_edges=[skeleton_edges]
                            )
                        pred_tokens = predictions['sequences'][0].cpu()
                        pred_coords = predictions['coordinates'][0].cpu()
                        num_bins = int(np.sqrt(train_args.vocab_size))
                        from datasets.discrete_tokenizer import DiscreteTokenizer
                        tokenizer = DiscreteTokenizer(
                            num_bins=num_bins,
                            seq_len=train_args.seq_len,
                            add_cls=getattr(train_args, 'add_cls_token', False)
                        )
                        max_kpts = len(query_gt_coords) if query_gt_coords else None
                        pred_keypoints = decode_sequence_to_keypoints(
                            pred_tokens, pred_coords, tokenizer, max_keypoints=max_kpts
                        )
                        pck_score = None
                        if query_gt_coords and len(pred_keypoints) > 0:
                            from util.eval_utils import compute_pck_bbox
                            bbox_w = TARGET_SIZE
                            bbox_h = TARGET_SIZE
                            num_kpts = min(len(pred_keypoints), len(query_gt_coords))
                            pred_kpts_trimmed = np.array(pred_keypoints[:num_kpts])
                            gt_kpts_trimmed = np.array(query_gt_coords[:num_kpts])
                            vis_trimmed = support_visibility[:num_kpts]
                            if pred_kpts_trimmed.max() <= 1.0:
                                pred_kpts_pixels = pred_kpts_trimmed * TARGET_SIZE
                            else:
                                pred_kpts_pixels = pred_kpts_trimmed
                            if gt_kpts_trimmed.max() <= 1.0:
                                gt_kpts_pixels = gt_kpts_trimmed * TARGET_SIZE
                            else:
                                gt_kpts_pixels = gt_kpts_trimmed
                            pck_score, num_correct, num_visible = compute_pck_bbox(
                                pred_keypoints=pred_kpts_pixels,
                                gt_keypoints=gt_kpts_pixels,
                                bbox_width=bbox_w,
                                bbox_height=bbox_h,
                                visibility=np.array(vis_trimmed),
                                threshold=0.2,
                                normalize_by='diagonal'
                            )
                        vis_support_image = ensure_image_size(support_image, TARGET_SIZE)
                        vis_query_image = ensure_image_size(query_image, TARGET_SIZE)
                        save_path = output_dir / f"single_image_{single_image_path.stem}_epoch{checkpoint_epoch}.png"
                        visualize_pose_prediction(
                            support_image=vis_support_image,
                            query_image=vis_query_image,
                            pred_keypoints=pred_keypoints,
                            support_keypoints=support_coords,
                            gt_keypoints=query_gt_coords,
                            skeleton_edges=skeleton_edges,
                            save_path=save_path,
                            category_name=f"Single Image: {single_image_path.name}",
                            pck_score=pck_score,
                            debug_coords=True
                        )
                        print(f"\n✓ Visualization complete!")
                        print(f"  Saved to: {save_path}")
                        if pck_score is not None:
                            print(f"  PCK@0.2: {pck_score:.2%}")
                        return
        if found_idx is not None:
            support_data = dataset[found_idx]
            support_cat_id = support_data.get('category_id')
            print(f"  ⚠️  Single-image mode: Using SAME image as both support and query (overfitting visualization)")
            query_data = support_data
            query_idx = found_idx
            query_image = query_data['image']
            support_image = support_data['image']
            support_coords = support_data['keypoints']
            support_visibility = support_data.get('visibility', [1] * len(support_coords))
            support_mask = torch.tensor([v > 0 for v in support_visibility], dtype=torch.float32)
            skeleton_edges = support_data.get('skeleton', [])
            query_gt_coords = query_data['keypoints']
            if isinstance(query_image, torch.Tensor):
                query_image_tensor = query_image.unsqueeze(0).to(device)
            else:
                query_image_tensor = torch.from_numpy(np.array(query_image)).permute(2, 0, 1).unsqueeze(0).to(device).float()
            support_coords_tensor = torch.tensor(support_coords, dtype=torch.float32).unsqueeze(0).to(device)
            support_mask_tensor = support_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model.forward_inference(
                    samples=query_image_tensor,
                    support_coords=support_coords_tensor,
                    support_mask=support_mask_tensor,
                    skeleton_edges=[skeleton_edges]
                )
            pred_tokens = predictions['sequences'][0].cpu()
            pred_coords = predictions['coordinates'][0].cpu()
            tokenizer = dataset.tokenizer
            max_kpts = len(query_gt_coords) if query_gt_coords else None
            pred_keypoints = decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer, max_keypoints=max_kpts)
            pck_score = None
            if query_gt_coords and len(pred_keypoints) > 0:
                from util.eval_utils import compute_pck_bbox
                bbox_w = query_data.get('width', query_data.get('bbox_width', TARGET_SIZE))
                bbox_h = query_data.get('height', query_data.get('bbox_height', TARGET_SIZE))
                num_kpts = min(len(pred_keypoints), len(query_gt_coords))
                pred_kpts_trimmed = np.array(pred_keypoints[:num_kpts])
                gt_kpts_trimmed = np.array(query_gt_coords[:num_kpts])
                query_visibility = query_data.get('visibility', [1] * len(query_gt_coords))
                vis_trimmed = query_visibility[:num_kpts]
                if pred_kpts_trimmed.max() <= 1.0:
                    pred_kpts_pixels = pred_kpts_trimmed * TARGET_SIZE
                else:
                    pred_kpts_pixels = pred_kpts_trimmed
                if gt_kpts_trimmed.max() <= 1.0:
                    gt_kpts_pixels = gt_kpts_trimmed * TARGET_SIZE
                else:
                    gt_kpts_pixels = gt_kpts_trimmed
                pck_score, num_correct, num_visible = compute_pck_bbox(
                    pred_keypoints=pred_kpts_pixels,
                    gt_keypoints=gt_kpts_pixels,
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    visibility=np.array(vis_trimmed),
                    threshold=0.2,
                    normalize_by='diagonal'
                )
            if isinstance(support_data['image'], torch.Tensor):
                vis_support_image = support_data['image'].permute(1, 2, 0).numpy()
                vis_support_image = (vis_support_image * 255).astype(np.uint8)
            else:
                vis_support_image = np.array(support_data['image'])
            if isinstance(query_data['image'], torch.Tensor):
                vis_query_image = query_data['image'].permute(1, 2, 0).numpy()
                vis_query_image = (vis_query_image * 255).astype(np.uint8)
            else:
                vis_query_image = np.array(query_data['image'])
            vis_support_image = ensure_image_size(vis_support_image, TARGET_SIZE)
            vis_query_image = ensure_image_size(vis_query_image, TARGET_SIZE)
            save_path = output_dir / f"single_image_{single_image_path.stem}_epoch{checkpoint_epoch}.png"
            visualize_pose_prediction(
                support_image=vis_support_image,
                query_image=vis_query_image,
                pred_keypoints=pred_keypoints,
                support_keypoints=support_coords,
                gt_keypoints=query_gt_coords,
                skeleton_edges=skeleton_edges,
                save_path=save_path,
                category_name=f"Single Image: {single_image_path.name}",
                pck_score=pck_score,
                debug_coords=True
            )
            print(f"\n✓ Visualization complete!")
            print(f"  Saved to: {save_path}")
            if pck_score is not None:
                print(f"  PCK@0.2: {pck_score:.2%}")
                if pck_score < 0.5:
                    print(f"\n  ⚠️  WARNING: Low PCK ({pck_score:.2%}) detected!")
                    print(f"     This may indicate:")
                    print(f"     1. The checkpoint is from an early epoch (before overfitting)")
                    print(f"     2. The checkpoint was trained on a different image")
                    print(f"     3. Training failed and you're using an old checkpoint")
                    print(f"     4. Coordinate normalization mismatch")
                    print(f"\n     Expected PCK for overfitted model: ~100%")
                    print(f"     Current checkpoint epoch: {checkpoint_epoch}")
                    print(f"     Check training logs to verify which checkpoint should be used.")
            return
    dataset = build_mp100_cape('test', train_args)
    print(f"✓ Loaded {len(dataset)} test images")
    from collections import defaultdict
    ann_file = Path(train_args.dataset_root) / "annotations" / f"mp100_split{args.split_id}_test.json"
    with open(ann_file) as f:
        test_ann_data = json.load(f)
        test_categories_info = {cat['id']: cat['name'] for cat in test_ann_data['categories']}
    test_categories = list(test_categories_info.keys())
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
    print("Grouping images by category from annotations (fast method)...")
    category_samples = defaultdict(list)
    coco = dataset.coco
    for idx, img_id in enumerate(dataset.ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue
        anns = coco.loadAnns(ann_ids)
        if len(anns) > 0 and 'category_id' in anns[0]:
            cat_id = anns[0]['category_id']
            if args.categories is not None and cat_id not in args.categories:
                continue
            category_samples[cat_id].append(idx)
    total_images = sum(len(indices) for indices in category_samples.values())
    print(f"✓ Grouped {total_images} images into {len(category_samples)} categories")
    total_visualized = 0
    for cat_id, sample_indices in category_samples.items():
        cat_name = test_categories_info.get(cat_id, f"cat_{cat_id}")
        print(f"\nCategory: {cat_name} (ID: {cat_id})")
        try:
            support_idx = sample_indices[0]
            support_data = dataset[support_idx]
        except ImageNotFoundError:
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
        query_samples = sample_indices[1:1+args.num_samples] if len(sample_indices) > 1 else []
        if len(query_samples) == 0:
            print(f"  Skipping category {cat_name} (only 1 image, need at least 2 for support+query)")
            continue
        for sample_idx in query_samples:
            try:
                data = dataset[sample_idx]
            except ImageNotFoundError:
                print(f"  Skipping sample {sample_idx} (image not found)")
                continue
            query_image = data['image']
            support_image = support_data['image']
            support_coords = support_data['keypoints']
            support_visibility = data.get('visibility', [1] * len(support_coords))
            support_mask = torch.tensor([v > 0 for v in support_visibility], dtype=torch.float32)
            skeleton_edges = support_data.get('skeleton', [])
            query_gt_coords = data['keypoints']
            query_visibility = data.get('visibility', [1] * len(query_gt_coords))
            if isinstance(query_image, torch.Tensor):
                query_image_tensor = query_image.unsqueeze(0).to(device)
            else:
                query_image_tensor = torch.from_numpy(np.array(query_image)).permute(2, 0, 1).unsqueeze(0).to(device).float()
            support_coords_tensor = torch.tensor(support_coords, dtype=torch.float32).unsqueeze(0).to(device)
            support_mask_tensor = support_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model.forward_inference(
                    samples=query_image_tensor,
                    support_coords=support_coords_tensor,
                    support_mask=support_mask_tensor,
                    skeleton_edges=[skeleton_edges]
                )
            pred_tokens = predictions['sequences'][0].cpu()
            pred_coords = predictions['coordinates'][0].cpu()
            tokenizer = dataset.tokenizer
            max_kpts = len(query_gt_coords) if query_gt_coords else None
            pred_keypoints = decode_sequence_to_keypoints(pred_tokens, pred_coords, tokenizer, max_keypoints=max_kpts)
            pck_score = None
            if query_gt_coords and len(pred_keypoints) > 0:
                from util.eval_utils import compute_pck_bbox
                bbox_w = data.get('width', data.get('bbox_width', TARGET_SIZE))
                bbox_h = data.get('height', data.get('bbox_height', TARGET_SIZE))
                num_kpts = min(len(pred_keypoints), len(query_gt_coords))
                pred_kpts_trimmed = np.array(pred_keypoints[:num_kpts])
                gt_kpts_trimmed = np.array(query_gt_coords[:num_kpts])
                vis_trimmed = query_visibility[:num_kpts]
                if pred_kpts_trimmed.max() <= 1.0:
                    pred_kpts_pixels = pred_kpts_trimmed * TARGET_SIZE
                else:
                    pred_kpts_pixels = pred_kpts_trimmed
                if gt_kpts_trimmed.max() <= 1.0:
                    gt_kpts_pixels = gt_kpts_trimmed * TARGET_SIZE
                else:
                    gt_kpts_pixels = gt_kpts_trimmed
                pck_score, num_correct, num_visible = compute_pck_bbox(
                    pred_keypoints=pred_kpts_pixels,
                    gt_keypoints=gt_kpts_pixels,
                    bbox_width=bbox_w,
                    bbox_height=bbox_h,
                    visibility=np.array(vis_trimmed),
                    threshold=0.2,
                    normalize_by='diagonal'
                )
            if isinstance(support_data['image'], torch.Tensor):
                vis_support_image = support_data['image'].permute(1, 2, 0).numpy()
                vis_support_image = (vis_support_image * 255).astype(np.uint8)
            else:
                vis_support_image = np.array(support_data['image'])
            if isinstance(data['image'], torch.Tensor):
                vis_query_image = data['image'].permute(1, 2, 0).numpy()
                vis_query_image = (vis_query_image * 255).astype(np.uint8)
            else:
                vis_query_image = np.array(data['image'])
            vis_support_image = ensure_image_size(vis_support_image, TARGET_SIZE)
            vis_query_image = ensure_image_size(vis_query_image, TARGET_SIZE)
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
                pck_score=pck_score,
                debug_coords=(pck_score is not None and pck_score >= 0.99)
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
    print(f"\nVisualization format (3-panel layout):")
    print(f"  - Panel 1 (Support GT):  Support image with GT keypoints (green circles)")
    print(f"  - Panel 2 (Ground Truth): Query image with GT keypoints (cyan circles)")
    print(f"  - Panel 3 (Predicted):    Query image with predicted keypoints (red X marks)")
    print(f"  - PCK@0.2 score shown in title for each prediction")
    print("=" * 80)
def main():
    parser = argparse.ArgumentParser(
        description='Visualize CAPE Pose Predictions on Test Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_cape_predictions.py \\
      --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \\
      --device mps \\
      --num_samples 5
  python visualize_cape_predictions.py \\
      --checkpoint outputs/cape_run/checkpoint_e050_*.pth \\
      --dataset_root . \\
      --output_dir visualizations/epoch_50
        """
    )
    parser.add_argument('--checkpoint', required=True,
                       help='Path to trained CAPE checkpoint (.pth file). '
                            'Can be any checkpoint: checkpoint_e***, checkpoint_best_pck_*, etc.')
    parser.add_argument('--device', default='cpu', 
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference (default: cpu)')
    parser.add_argument('--dataset_root', default='.', 
                       help='Path to MP-100 dataset root directory (default: current directory)')
    parser.add_argument('--split_id', type=int, default=1, 
                       help='MP-100 split ID (1-5). Must match the split used during training.')
    parser.add_argument('--num_samples', type=int, default=3, 
                       help='Number of samples to visualize per category (default: 3)')
    parser.add_argument('--output_dir', default='visualizations', 
                       help='Directory to save visualization images (default: visualizations/)')
    parser.add_argument('--categories', type=int, nargs='+', default=None,
                       help='Specific category IDs to visualize (default: all test categories). '
                            'Example: --categories 40 55 68')
    parser.add_argument('--single_image_path', type=str, default=None,
                       help='Path to a specific image to visualize (for overfitted models). '
                            'If provided, this image will be used as both support and query. '
                            'Example: --single_image_path data/camel_face/camel_133.jpg')
    args = parser.parse_args()
    visualize_from_checkpoint(args)
if __name__ == '__main__':
    main()