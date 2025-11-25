# CAPE Evaluation Script - Quick Start Guide

## Overview

The `scripts/eval_cape_checkpoint.py` script provides standalone evaluation and visualization for CAPE model checkpoints.

---

## Quick Start (30 seconds)

### Step 1: Evaluate a Checkpoint

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck.pth \
    --split val \
    --num-visualizations 50 \
    --output-dir outputs/cape_eval
```

### Step 2: Check Results

```bash
# View metrics
cat outputs/cape_eval/metrics_val.json

# View visualizations
open outputs/cape_eval/visualizations/
```

---

## What It Does

### 1. Loads Model
- Reads checkpoint (.pth file)
- Builds model with same architecture as training
- Loads weights (handles old/new checkpoints)
- Sets to eval mode

### 2. Runs Evaluation
- Uses existing `episodic_sampler` for data loading
- Runs **autoregressive inference** (no teacher forcing)
- Extracts keypoints from predicted sequences
- Computes PCK@0.2 with proper bbox normalization

### 3. Generates Metrics
- **Overall PCK@0.2**: Percentage of correct keypoints
- **Per-category PCK**: Breakdown by animal category
- **Total correct/visible**: Absolute counts

**Saved to:** `{output_dir}/metrics_{split}.json`

### 4. Creates Visualizations
- **3-panel layout**: Support | GT | Predicted
- **Color coding**:
  - Green circles: Support keypoints (template)
  - Cyan circles: Ground truth query keypoints
  - Red X marks: Predicted query keypoints
- **Optional skeleton**: Connect keypoints with edges
- **PCK overlay**: Shows accuracy for each example

**Saved to:** `{output_dir}/visualizations/vis_*.png`

---

## Command Line Arguments

### Required

```bash
--checkpoint PATH_TO_CHECKPOINT.pth
```

### Recommended

```bash
--split val                    # Which split to evaluate
--num-visualizations 50        # How many examples to visualize
--output-dir outputs/eval      # Where to save results
```

### Optional

```bash
--num-episodes 100             # Total episodes (default: 100 for val)
--num-queries-per-episode 2    # Queries per episode (default: from checkpoint)
--pck-threshold 0.2            # PCK threshold (default: 0.2)
--draw-skeleton                # Draw skeleton edges
--save-all-queries             # Save all queries (not just first)
--device mps                   # Force specific device
--num-workers 0                # Dataloader workers
```

---

## Output Structure

```
outputs/cape_eval/
├── metrics_val.json              # Evaluation metrics
└── visualizations/               # Visualization images
    ├── vis_0000_q0_cat12_img1200000000019572.png
    ├── vis_0001_q0_cat35_img3500000000046291.png
    ├── vis_0002_q0_cat95_img9500000000050162.png
    └── ...
```

---

## Understanding the Visualizations

### Visualization Layout

```
┌────────────────┬────────────────┬────────────────┐
│  Support (GT)  │  Ground Truth  │   Predicted    │
│                │                │                │
│  Green circles │  Cyan circles  │  Red X marks   │
│  (template)    │  (ground truth)│  (model output)│
│                │                │  PCK@0.2: 45%  │
└────────────────┴────────────────┴────────────────┘
```

### Color Guide

| Color | Meaning | Panel |
|-------|---------|-------|
| 🟢 Green circles | Support keypoints (template provided to model) | Left |
| 🔵 Cyan circles | Ground truth keypoints (correct answer) | Middle |
| 🔴 Red X marks | Predicted keypoints (model output) | Right |
| Lines | Skeleton edges (connects keypoints) | All |

### What to Look For

**Good Predictions:**
- Red X marks close to cyan circles
- Skeleton structure preserved
- PCK@0.2 > 50%

**Poor Predictions:**
- Red X marks far from cyan circles
- Broken skeleton structure
- PCK@0.2 < 30%

**Model Issues:**
- Only 1-2 red X marks visible → Model predicting `<eos>` too early
- No red X marks → Model not generating any keypoints
- X marks in wrong locations → Model not learning proper patterns

---

## Example Results

### Good Checkpoint (After Fix)

```
================================================================================
EVALUATION RESULTS
================================================================================

Overall PCK@0.2: 0.4521 (45.21%)
  Correct keypoints: 1234 / 2730
  Mean PCK across categories: 0.4521 (45.21%)

Per-Category PCK:
  Category ID     PCK        Correct/Total  
  ---------------  ----------  ---------------
  12               0.6100      153/250        
  35               0.5500      110/200        
  95               0.3800      76/200         
```

**Interpretation:**
- ✅ PCK ~45% is good for few-shot pose estimation
- ✅ Model is generalizing to unseen categories
- ✅ Some categories easier than others (expected)

### Old Checkpoint (Before Fix)

```
⚠️  CRITICAL WARNING: OLD CHECKPOINT DETECTED
   pred_coords shape: torch.Size([2, 1, 2])
   gt_coords shape: torch.Size([2, 200, 2])
   
   This checkpoint was trained WITHOUT a tokenizer and only generates
   1 keypoint(s) before predicting <eos>.

Overall PCK@0.2: 1.0000 (100.00%)  ← INVALID
```

**Interpretation:**
- ❌ PCK@100% is invalid (due to teacher forcing bug)
- ❌ Model only generates 1 keypoint
- ❌ Cannot be properly evaluated
- ⚠️  Must retrain with fixed code

---

## Workflow Examples

### Workflow 1: Evaluate Latest Checkpoint

```bash
# Find latest checkpoint
ls -lt outputs/cape_run/checkpoint_*.pth | head -1

# Evaluate it
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e020_lr1e-04_bs2_acc4_qpe2.pth \
    --split val \
    --num-visualizations 50 \
    --draw-skeleton \
    --output-dir outputs/eval_latest
    
# View results
cat outputs/eval_latest/metrics_val.json
open outputs/eval_latest/visualizations/
```

### Workflow 2: Evaluate on Test Set

```bash
# Evaluate best checkpoint on test set
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck.pth \
    --split test \
    --num-episodes 200 \
    --num-visualizations 100 \
    --draw-skeleton \
    --output-dir outputs/eval_test_final
```

### Workflow 3: Compare Training Progress

```bash
# Evaluate multiple epochs
for epoch in 10 20 30 40 50; do
    python scripts/eval_cape_checkpoint.py \
        --checkpoint outputs/cape_run/checkpoint_e${epoch}_*.pth \
        --split val \
        --num-episodes 50 \
        --num-visualizations 20 \
        --output-dir outputs/eval_e${epoch}
done

# Compare metrics
for epoch in 10 20 30 40 50; do
    echo "Epoch $epoch:"
    jq '.pck_overall' outputs/eval_e${epoch}/metrics_val.json
done
```

---

## Technical Details

### Data Flow

```
1. Load checkpoint
   ↓
2. Build model with tokenizer (CRITICAL!)
   ↓
3. Load weights
   ↓
4. Build episodic dataloader
   ↓
5. For each episode:
   - Run forward_inference (autoregressive)
   - Extract keypoints from sequence
   - Compute PCK with bbox normalization
   - Store predictions for visualization
   ↓
6. Aggregate metrics
   ↓
7. Generate visualizations
   ↓
8. Save JSON + images
```

### Coordinate Systems

The script handles three coordinate systems:

1. **Model Output**: Normalized [0,1] relative to bbox
   - Used internally by model
   - Stored in predictions

2. **Bbox Coordinates**: Pixel coordinates relative to bbox crop
   - Used for PCK computation
   - Multiplied by bbox dimensions

3. **Image Coordinates**: Pixel coordinates for visualization
   - Used for drawing on images
   - Already in correct space (images are bbox crops)

**Transformation:**
```python
# Model output → Pixel coordinates
kpts_px = kpts_norm * [bbox_width, bbox_height]

# For PCK computation (already in correct space)
pck = compute_pck_bbox(pred_norm, gt_norm, bbox_w, bbox_h)

# For visualization (convert to pixels)
kpts_vis = denormalize_keypoints(kpts_norm, bbox_w, bbox_h)
```

### PCK Computation

PCK (Percentage of Correct Keypoints) is computed as:

```python
# For each keypoint:
pred_px = pred_norm * [bbox_width, bbox_height]
gt_px = gt_norm * [bbox_width, bbox_height]

error = ||pred_px - gt_px||₂
threshold = 0.2 * sqrt(bbox_width² + bbox_height²)

correct = (error < threshold)

PCK = sum(correct) / sum(visible)
```

**Notes:**
- Only visible keypoints (visibility > 0) are evaluated
- Threshold is relative to bbox diagonal (not absolute pixels)
- Different bboxes have different thresholds

---

## Comparison with Training Validation

### Training Validation (`engine_cape.py`)
- Runs after each epoch
- Uses **autoregressive inference** (after fix)
- Computes PCK@0.2
- Logged to console and wandb

### Standalone Evaluation (This Script)
- Runs on-demand for any checkpoint
- Uses **same** autoregressive inference
- Computes **same** PCK@0.2
- Saves visualizations and JSON

**Key Difference:** Standalone evaluation saves visualizations and can run on different splits without retraining.

---

## Best Practices

### 1. Always Use Latest Code

```bash
# Pull latest code before evaluating
git pull

# Verify tokenizer fix is in place
python tests/test_tokenizer_fix_simple.py

# Then evaluate
python scripts/eval_cape_checkpoint.py ...
```

### 2. Evaluate on Multiple Splits

```bash
# Validation set (unseen categories, used during training)
python scripts/eval_cape_checkpoint.py --split val ...

# Test set (held-out final evaluation)
python scripts/eval_cape_checkpoint.py --split test ...
```

### 3. Save Visualizations

Always use `--num-visualizations` > 0 to:
- Visually inspect model predictions
- Identify failure cases
- Understand category-specific challenges
- Debug coordinate transformation issues

### 4. Use Appropriate Device

```bash
# On Apple Silicon
--device mps

# On NVIDIA GPU
--device cuda

# On CPU (slower but always works)
--device cpu
```

---

## FAQ

### Q: What split should I evaluate on?

**A:** 
- **val**: Use during development to monitor progress
- **test**: Use for final reported results

### Q: How many episodes should I use?

**A:**
- **Quick check**: 10-20 episodes
- **Thorough**: 100 episodes for val, 200 for test
- **Full**: Use all available (don't set `--num-episodes`)

### Q: Why is my PCK different from training logs?

**A:** Possible reasons:
1. Different number of episodes (random sampling)
2. Different random seed
3. Old checkpoint (teacher forcing bug)
4. Different validation split

### Q: Can I evaluate on training set?

**A:** Yes, but not recommended:
```bash
--split train
```
PCK will be very high (model saw these during training).

### Q: What's a good PCK score?

**A:**
- **Untrained (epoch 1)**: 10-20%
- **Learning (epoch 10-20)**: 30-50%
- **Well-trained (epoch 30+)**: 50-70%
- **Excellent**: 65-75%
- **Suspicious**: >80% (might be overfitting or bug)

### Q: Can I run this during training?

**A:** Yes, but it will slow down training:
```bash
# In another terminal while training is running
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_latest.pth \
    ...
```

---

## Troubleshooting Checklist

If the script fails or gives unexpected results:

- [ ] Check checkpoint path exists
- [ ] Verify dataset_root is correct (or let it use checkpoint args)
- [ ] Try with `--device cpu` (most compatible)
- [ ] Check if model has tokenizer: `python tests/test_tokenizer_fix_simple.py`
- [ ] Use small `--num-episodes 5` for quick testing
- [ ] Check for error messages in console output
- [ ] Verify checkpoint is not from old (buggy) training code

---

## Citation

If you use this evaluation script in your research, please cite the CAPE project.

---

**For more details, see `scripts/README.md`**

