# CAPE Visualization Guide

## 🎨 How to Visualize Your Training Results

After training, you want to see how well your model predicts keypoints! This guide shows you multiple ways to visualize your results.

---

## ✅ What You Have

Based on your training output, you have:
- **Checkpoints saved**: `outputs/cape_run/checkpoint_best_loss_*.pth`
- **PCK Score**: 100% (perfect on validation!)
- **Trained for**: 1 epoch
- **Categories**: 69 training, 10 validation

---

## 🎯 Option 1: Use the Existing Visualization Script (RECOMMENDED)

You already have `visualize_cape_predictions.py` which is perfect for this!

### **Quick Start**

```bash
source venv/bin/activate

# Visualize test set predictions
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --dataset_root . \
    --split_id 1 \
    --num_samples 5 \
    --output_dir visualizations/test_predictions \
    --device mps
```

### **What It Does**

1. ✅ Loads your trained checkpoint
2. ✅ Runs inference on **test images** (unseen categories!)
3. ✅ Creates side-by-side visualizations:
   - **Left**: Support pose template with keypoints + skeleton
   - **Right**: Query image with predicted keypoints + skeleton
4. ✅ Saves images to `visualizations/test_predictions/`

### **Output**

You'll get images like:
```
beaver_body_sample_123.png
cat_face_sample_456.png
horse_body_sample_789.png
...
```

Each shows:
- Support keypoints (green) with numbered labels
- Predicted keypoints (red) with numbered labels
- Skeleton connections (if available)

---

## 🔍 Option 2: Visualize Validation Set (See What PCK Saw)

To see the exact images used during validation (the ones that got 100% PCK):

```bash
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --dataset_root . \
    --split_id 1 \
    --num_samples 3 \
    --output_dir visualizations/val_predictions \
    --device mps
```

**Note**: The script loads the test set by default, but you can modify it to load the val set.

---

## 📊 Option 3: Create a Comparison Grid (BEST FOR ANALYSIS)

For a more detailed analysis, you can create a grid showing:
- Ground truth keypoints
- Predicted keypoints
- Error magnitude per keypoint

### **Create a Custom Visualization Script**

Save this as `visualize_comparison.py`:

```python
#!/usr/bin/env python3
"""
Enhanced visualization with GT vs Predicted comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.roomformer_v2 import build as build_base_model
from models.cape_model import build_cape_model
from datasets.mp100_cape import build_mp100_cape, ImageNotFoundError
from datasets.episodic_sampler import build_episodic_dataloader


def visualize_comparison(args):
    """Create GT vs Pred comparison visualizations."""
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    train_args.dataset_root = args.dataset_root
    
    # Build model
    build_result = build_base_model(train_args)
    base_model = build_result[0] if isinstance(build_result, tuple) else build_result
    model = build_cape_model(train_args, base_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # Load validation dataset
    val_dataset = build_mp100_cape('val', train_args)
    val_loader = build_episodic_dataloader(
        base_dataset=val_dataset,
        category_split_file='category_splits.json',
        split='val',
        batch_size=1,
        num_queries_per_episode=1,
        episodes_per_epoch=10,
        num_workers=0,
        seed=42
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process validation batches
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_samples:
            break
        
        # Move to device
        query_images = batch['query_images'].to(device)
        support_coords = batch['support_coords'].to(device)
        support_mask = batch['support_mask'].to(device)
        support_skeletons = batch['support_skeletons']
        
        # Get ground truth
        gt_keypoints = batch['query_metadata'][0]['keypoints']  # (N, 2)
        gt_visibility = batch['query_metadata'][0]['visibility']  # (N,)
        category_id = batch['category_ids'][0].item()
        
        # Run inference
        with torch.no_grad():
            outputs = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_mask,
                skeleton_edges=support_skeletons
            )
        
        # Extract predictions
        from engine_cape import extract_keypoints_from_sequence
        pred_coords = outputs['pred_coords']
        token_labels = outputs['sequences']
        mask = (token_labels != -1)
        
        pred_kpts = extract_keypoints_from_sequence(pred_coords, token_labels, mask)
        pred_kpts = pred_kpts[0].cpu().numpy()  # (N, 2)
        
        # Trim to category's keypoint count
        num_kpts = len(gt_visibility)
        pred_kpts = pred_kpts[:num_kpts]
        
        # Convert GT to numpy
        gt_kpts = np.array(gt_keypoints)
        gt_vis = np.array(gt_visibility)
        
        # Denormalize for visualization (assuming 512x512)
        pred_kpts_vis = pred_kpts * 512
        gt_kpts_vis = gt_kpts * 512
        
        # Compute errors
        errors = np.linalg.norm(pred_kpts_vis - gt_kpts_vis, axis=1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Get query image
        query_img = query_images[0].cpu().permute(1, 2, 0).numpy()
        query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
        
        # Plot 1: Ground Truth
        axes[0].imshow(query_img)
        axes[0].set_title(f'Ground Truth\nCategory {category_id}', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot GT keypoints
        visible_mask = gt_vis > 0
        axes[0].scatter(gt_kpts_vis[visible_mask, 0], gt_kpts_vis[visible_mask, 1],
                       c='lime', s=100, marker='o', edgecolors='black', linewidths=2,
                       label=f'Visible ({visible_mask.sum()})', zorder=3)
        
        if not visible_mask.all():
            axes[0].scatter(gt_kpts_vis[~visible_mask, 0], gt_kpts_vis[~visible_mask, 1],
                           c='gray', s=100, marker='x', linewidths=2,
                           label=f'Invisible ({(~visible_mask).sum()})', zorder=2)
        
        axes[0].legend(loc='upper right')
        
        # Plot 2: Predictions
        axes[1].imshow(query_img)
        axes[1].set_title(f'Predictions\n{len(pred_kpts)} keypoints', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[1].scatter(pred_kpts_vis[:, 0], pred_kpts_vis[:, 1],
                       c='red', s=100, marker='x', linewidths=3,
                       label='Predicted', zorder=3)
        axes[1].legend(loc='upper right')
        
        # Plot 3: Error Heatmap
        axes[2].imshow(query_img, alpha=0.5)
        axes[2].set_title(f'Prediction Errors\nMean: {errors[visible_mask].mean():.1f}px', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Draw error vectors (GT -> Pred)
        for i, (gt_pt, pred_pt, err, vis) in enumerate(zip(gt_kpts_vis, pred_kpts_vis, errors, gt_vis)):
            if vis > 0:  # Only for visible keypoints
                # Draw error vector
                axes[2].arrow(gt_pt[0], gt_pt[1],
                            pred_pt[0] - gt_pt[0], pred_pt[1] - gt_pt[1],
                            head_width=5, head_length=5, fc='red', ec='red', alpha=0.7,
                            linewidth=2)
                
                # GT point
                axes[2].scatter(gt_pt[0], gt_pt[1], c='lime', s=80, marker='o',
                              edgecolors='black', linewidths=2, zorder=3)
                
                # Pred point
                axes[2].scatter(pred_pt[0], pred_pt[1], c='red', s=80, marker='x',
                              linewidths=2, zorder=3)
                
                # Error label
                axes[2].text(pred_pt[0], pred_pt[1]-15, f'{err:.0f}px',
                           color='white', fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        # Add colorbar legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lime', edgecolor='black', label='Ground Truth'),
            Patch(facecolor='red', label='Predicted'),
            Patch(facecolor='red', alpha=0.5, label='Error Vector')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        save_path = output_dir / f'comparison_batch{batch_idx:03d}_cat{category_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        print(f"  Mean error: {errors[visible_mask].mean():.2f}px (visible keypoints only)")
        print(f"  Max error: {errors[visible_mask].max():.2f}px")
        print()
    
    print(f"\n✓ Saved {args.num_samples} comparison visualizations to {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_root', default='.')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', default='visualizations/comparisons')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    
    visualize_comparison(args)
```

### **Run It**

```bash
python visualize_comparison.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --num_samples 20 \
    --output_dir visualizations/error_analysis
```

This creates **3-panel visualizations**:
1. **Ground Truth**: Shows which keypoints are visible/invisible
2. **Predictions**: Shows what the model predicted
3. **Error Heatmap**: Shows error vectors (GT → Pred) with pixel distances

---

## 📈 Option 4: Quick Stats Summary

To get quick statistics without visualizations:

```python
# Add to visualize_comparison.py or run separately
def print_stats(checkpoint_path):
    """Print quick stats about model performance."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    print("="*60)
    print("CHECKPOINT STATS")
    print("="*60)
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Best PCK: {checkpoint.get('best_pck', 'N/A')}")
    print(f"Learning Rate: {checkpoint['args'].lr}")
    print(f"Batch Size: {checkpoint['args'].batch_size}")
    print(f"Queries/Episode: {checkpoint['args'].num_queries_per_episode}")
    print("="*60)
```

---

## 🎯 Recommended Workflow

### **Step 1: Quick Check (Test Set)**
```bash
# Visualize a few test samples to see general performance
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --dataset_root . \
    --num_samples 3 \
    --output_dir visualizations/quick_test
```

### **Step 2: Detailed Analysis (Validation Set)**
```bash
# Create GT vs Pred comparisons with error vectors
python visualize_comparison.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --num_samples 20 \
    --output_dir visualizations/detailed_analysis
```

### **Step 3: Review Results**
```bash
# Open visualizations
open visualizations/quick_test/
open visualizations/detailed_analysis/
```

---

## 🔍 What to Look For

### ✅ **Good Signs**
- Predicted keypoints cluster near GT keypoints
- Skeleton structure is preserved
- Invisible keypoints are handled gracefully
- Low pixel errors (<20px for PCK@0.2)

### ⚠️ **Warning Signs**
- Predictions far from GT (>50px)
- Skeleton connections broken
- All keypoints clustered in one spot (model collapse)
- Random predictions (no structure)

---

## 💡 Advanced: Per-Category Analysis

To see which categories work well:

```python
# Modify visualize_comparison.py to track per-category stats
category_errors = {}

for batch_idx, batch in enumerate(val_loader):
    cat_id = batch['category_ids'][0].item()
    errors = compute_errors(gt, pred)
    
    if cat_id not in category_errors:
        category_errors[cat_id] = []
    category_errors[cat_id].append(errors.mean())

# Print summary
for cat_id, errs in sorted(category_errors.items(), key=lambda x: np.mean(x[1])):
    print(f"Category {cat_id}: {np.mean(errs):.2f}px ± {np.std(errs):.2f}px")
```

---

## 📝 Summary

| Method | Best For | Output |
|--------|----------|--------|
| **visualize_cape_predictions.py** | Quick overview of test predictions | Side-by-side images |
| **visualize_comparison.py** | Detailed error analysis | 3-panel with error vectors |
| **Stats summary** | Quick performance check | Console output |

---

## 🚀 Quick Commands

```bash
# Activate environment
source venv/bin/activate

# Basic visualization (RECOMMENDED FIRST STEP)
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_loss_e000_valloss7.4741_pck1.0000.pth \
    --dataset_root . \
    --num_samples 5 \
    --output_dir visualizations/results

# View results
open visualizations/results/
```

---

## 💭 Interpreting Your 100% PCK

Your validation achieved **PCK@0.2 = 100%** (204/204 keypoints correct).

This means:
- ALL predicted keypoints were within 20% of bbox diagonal from GT
- For a 100px bbox, that's within 20px
- This is **very good** for just 1 epoch!

**Next steps**:
1. ✅ Run visualization to visually confirm
2. ✅ Test on unseen categories (test set)
3. ✅ Train for more epochs (currently only 1)
4. ✅ Check if it generalizes to new categories

---

*Happy visualizing! 🎨*

