# 📊 CAPE Visualization Guide

## Overview

The visualization system allows you to visually inspect your trained CAPE model's predictions on unseen test categories. This is crucial for understanding:
- **What the model is actually predicting**
- **How well 1-shot conditioning works**
- **Which categories/keypoints are difficult**
- **Whether predictions are anatomically plausible**

---

## 🎯 What Gets Visualized

### Input to Model (1-Shot CAPE)
For each test category:
1. **Support Image** (template): One image with ground truth keypoints
2. **Query Image** (to predict): Different image from same category
3. **Skeleton Structure**: Edge connectivity for the category

### Output Visualization (Side-by-Side)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│  Support Template               │  Query Image - Predicted Pose   │
│  beaver_body                    │  17 keypoints | PCK@0.2: 45.2%  │
│                                 │                                 │
│  [Support Image]                │  [Query Image]                  │
│                                 │                                 │
│  ● ──── ● Green circles         │  ● ──── ● Cyan circles (GT)     │
│  │      │ (support GT)          │  ✗ ──── ✗ Red X's (predicted)  │
│  ● ──── ● Numbered 1,2,3...     │  │      │ Numbered 1,2,3...     │
│                                 │                                 │
└─────────────────────────────────┴─────────────────────────────────┘
```

**Left Panel:** Support image with ground truth keypoints (green)
**Right Panel:** Query image with:
- **Red X's**: Model predictions (autoregressive inference)
- **Cyan circles**: Ground truth (for comparison only)
- **PCK score**: Percentage of correct keypoints

---

## 🚀 Quick Start

### Option 1: Use Convenience Script (Recommended)

```bash
# Visualize best PCK checkpoint (3 samples per category)
./run_visualization.sh

# Visualize with 5 samples per category
./run_visualization.sh "outputs/cape_run/checkpoint_best_pck*.pth" 5

# Visualize specific epoch
./run_visualization.sh outputs/cape_run/checkpoint_e050_*.pth 5
```

### Option 2: Direct Python Command

```bash
# Activate venv first
source activate_venv.sh

# Visualize best model
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_e050_pck0.4567.pth \
    --device mps \
    --num_samples 5 \
    --output_dir visualizations/best_model

# Visualize specific categories only
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \
    --device mps \
    --categories 40 55 68 \
    --num_samples 3
```

---

## 📋 Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--checkpoint` | Path to trained CAPE checkpoint | `outputs/cape_run/checkpoint_best_pck_e050.pth` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cpu` | Device for inference: `cpu`, `cuda`, or `mps` |
| `--dataset_root` | `.` | Path to MP-100 dataset root |
| `--split_id` | `1` | MP-100 split ID (1-5), must match training |
| `--num_samples` | `3` | Number of samples per category |
| `--output_dir` | `visualizations/` | Where to save output images |
| `--categories` | All test cats | Specific category IDs to visualize |

---

## 🔍 How It Works (Technical Details)

### Step 1: Load Trained Model

```python
checkpoint = torch.load(checkpoint_path)
train_args = checkpoint['args']  # Get original training config
model = build_cape_model(train_args, ...)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
```

- Loads checkpoint (e.g., `checkpoint_best_pck_e050.pth`)
- Extracts saved training arguments (architecture, vocab size, etc.)
- Rebuilds exact same model
- Loads trained weights
- Sets to evaluation mode

### Step 2: Load Test Dataset

```python
dataset = build_mp100_cape('test', train_args)
```

- Loads **test split** (20 unseen categories)
- Categories the model has **NEVER seen during training**
- Examples: `przewalskihorse_face`, `beaver_body`, `kangaroo_body`

### Step 3: For Each Category, Sample Episodes

```python
support_data = dataset[support_idx]  # First image (template)
query_data = dataset[query_idx]      # Other images (to predict)
```

**Support (Template):**
- First valid image in category
- Extracts ground truth keypoints
- Extracts skeleton structure
- Used as **1-shot conditioning**

**Query (To Predict):**
- Different images from same category
- Model must predict their keypoints
- Ground truth used only for PCK computation

### Step 4: Run Autoregressive Inference

```python
with torch.no_grad():
    predictions = model.forward_inference(
        samples=query_image,           # Query to predict
        support_coords=support_kpts,   # Support GT (conditioning)
        support_mask=support_mask,     # Which keypoints visible
        skeleton_edges=[skeleton]      # Category structure
    )
```

**Model receives:**
1. Query image (RGB)
2. Support keypoints (normalized [0,1])
3. Support mask (bool tensor)
4. Skeleton edges (connectivity graph)

**Model does:**
1. Starts from BOS token
2. Autoregressively generates keypoint sequence
3. Uses support as reference/template
4. Outputs predicted keypoints

### Step 5: Compute PCK and Visualize

```python
pck_score = compute_pck_bbox(
    pred_keypoints=predictions,
    gt_keypoints=query_gt,
    bbox_width=bbox_w,
    bbox_height=bbox_h,
    visibility=visibility,
    threshold=0.2
)

visualize_pose_prediction(
    support_image=support_img,
    query_image=query_img,
    pred_keypoints=predictions,
    support_keypoints=support_kpts,
    gt_keypoints=query_gt,
    pck_score=pck_score
)
```

Saves visualization showing:
- Support template (left)
- Query predictions vs GT (right)
- PCK score in title

---

## 📈 Interpreting Results

### Good Predictions (High PCK)

```
PCK@0.2: 85%
```

**What you'll see:**
- Red X's (predictions) closely overlap cyan circles (GT)
- Skeleton structure looks anatomically correct
- All major keypoints captured
- Minor errors in fine details (ears, tails, etc.)

**Interpretation:**
- Model learned good representations
- Support conditioning is effective
- Category structure well understood

### Moderate Predictions (Medium PCK)

```
PCK@0.2: 40-60%
```

**What you'll see:**
- Some keypoints correct (head, body)
- Others off (limbs, extremities)
- Skeleton roughly correct but details wrong
- Predictions in plausible locations

**Interpretation:**
- Model captures high-level structure
- Struggles with fine-grained localization
- May need more training or better support

### Poor Predictions (Low PCK)

```
PCK@0.2: <20%
```

**What you'll see:**
- Keypoints in wrong locations
- Skeleton structure broken
- Predictions clustered or nonsensical
- No clear pose pattern

**Interpretation:**
- Model failed to generalize to this category
- Support conditioning not effective
- May indicate:
  - Category too different from training
  - Insufficient training
  - Data quality issues

---

## 🎨 Visual Elements Explained

### Support Panel (Left)

| Element | Color | Meaning |
|---------|-------|---------|
| Green circles (●) | `lime` | Ground truth keypoints from support image |
| Green lines (─) | `lime` | Skeleton edges (connectivity) |
| Numbers (1,2,3...) | White | Keypoint indices (matches skeleton definition) |

**Purpose:** Shows the 1-shot template the model uses for conditioning

### Query Panel (Right)

| Element | Color | Meaning |
|---------|-------|---------|
| Red X's (✗) | `red` | **Model predictions** (autoregressive inference) |
| Cyan circles (●) | `cyan` | Ground truth (for comparison only) |
| Red lines (─) | `red` | Predicted skeleton structure |
| Cyan lines (─) | `cyan` | Ground truth skeleton structure |
| Numbers (1,2,3...) | White | Keypoint indices |

**Purpose:** Compare model predictions against ground truth

---

## 🔧 Troubleshooting

### Issue: No visualizations generated

**Check:**
1. Dataset exists: `data/annotations/mp100_split1_test.json`
2. Images exist: `data/{category_name}/`
3. Checkpoint loads correctly

**Solution:**
```bash
# Check dataset structure
ls data/annotations/
ls data/ | head -20

# Try with specific categories
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \
    --categories 40 55  # Specify known categories
```

### Issue: Images not found errors

**Cause:** Some MP-100 categories have missing images

**Solution:** Script automatically skips missing images and reports count

### Issue: CUDA/MPS errors

**Solution:**
```bash
# Use CPU if GPU issues
python visualize_cape_predictions.py \
    --checkpoint ... \
    --device cpu
```

### Issue: "Checkpoint has unexpected keys"

**Cause:** Old checkpoints with state_dict contamination bug

**Solution:** Script uses `strict=False`, so it still works (just shows warning)

---

## 📊 Expected Output

### Console Output

```
================================================================================
CAPE Pose Estimation Visualization
================================================================================
Checkpoint: outputs/cape_run/checkpoint_best_pck_e050_pck0.4567.pth
Device: mps
Output: visualizations/
================================================================================

Loading model from outputs/cape_run/checkpoint_best_pck_e050_pck0.4567.pth...
  Using training config (epoch 50)
✓ Model loaded successfully
✓ Model loaded and moved to mps

Loading MP-100 test dataset...
✓ Loaded 1234 test images

Test categories to visualize: 20
Categories (showing first 10): ['przewalskihorse_face', 'beaver_body', ...]
Samples per category: 3

Category: beaver_body (ID: 40)
  Sample 123: 17 keypoints predicted | PCK: 0.65
  → Saved: visualizations/beaver_body_query123_support100.png
  Sample 124: 17 keypoints predicted | PCK: 0.71
  → Saved: visualizations/beaver_body_query124_support100.png
  Sample 125: 17 keypoints predicted | PCK: 0.59
  → Saved: visualizations/beaver_body_query125_support100.png

Category: kangaroo_body (ID: 55)
  ...

================================================================================
✓ Visualization Complete!
================================================================================
  Categories visualized: 20
  Total images: 60
  Saved to: visualizations/

Visualization format:
  - Left panel:  Support image with GT keypoints (green circles)
  - Right panel: Query image with predicted (red X) and GT (cyan circles)
  - PCK@0.2 score shown in title for each prediction
================================================================================
```

### File Output

```
visualizations/
├── beaver_body_query123_support100.png
├── beaver_body_query124_support100.png
├── beaver_body_query125_support100.png
├── kangaroo_body_query456_support450.png
├── kangaroo_body_query457_support450.png
├── ...
└── visualization_output.log (console output)
```

---

## 💡 Tips for Analysis

### 1. Compare Across Categories

Look for patterns:
- Which categories have consistently high PCK?
- Which are difficult?
- Do similar animals perform similarly? (e.g., all quadrupeds)

### 2. Examine Failure Cases

For low-PCK predictions:
- Are all keypoints wrong or just a few?
- Are errors systematic? (e.g., left/right flips)
- Is the support image a good template?

### 3. Check Support Quality

- Is the support image representative?
- Are keypoints clearly visible in support?
- Does support pose match query pose orientation?

### 4. Validate Against Ground Truth

- Cyan circles show where keypoints SHOULD be
- Red X's show where model THINKS they are
- Large gaps = model error
- Small gaps = good generalization

---

## 🔄 Workflow Integration

### During Training

**Every N epochs, visualize:**
```bash
# Visualize latest checkpoint
./run_visualization.sh "outputs/cape_run/checkpoint_e050_*.pth" 3

# Compare against best PCK
./run_visualization.sh "outputs/cape_run/checkpoint_best_pck*.pth" 3
```

**What to look for:**
- Are predictions improving over epochs?
- Which categories improve first?
- Are there categories that never improve?

### After Training Completes

```bash
# Final visualization with more samples
./run_visualization.sh "outputs/cape_run/checkpoint_best_pck*.pth" 10

# Generate comprehensive report
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \
    --device mps \
    --num_samples 10 \
    --output_dir visualizations/final_results
```

### For Paper/Presentation

```bash
# Visualize specific "showcase" categories
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \
    --categories 40 55 68 72  # Hand-picked interesting categories \
    --num_samples 5 \
    --output_dir visualizations/paper_figures
```

---

## 🧪 Debugging with Visualizations

### Check If Model is Learning

**Problem:** Training loss decreases but validation PCK stays low

**Debugging:**
1. Visualize predictions at epoch 10, 20, 30, ...
2. Look for gradual improvement in predictions
3. If predictions are random → model not learning
4. If predictions improve → just needs more epochs

### Check Support Conditioning

**Problem:** Model ignores support template

**Debugging:**
1. Check if predicted pose resembles support pose structure
2. If predictions look similar across different supports → model ignoring support
3. If predictions vary with support → conditioning works

### Check Category-Specific Issues

**Problem:** Some categories perform much worse

**Debugging:**
1. Visualize worst-performing categories
2. Check if support images are good quality
3. Check if skeleton definition makes sense
4. May indicate data issues or category too rare

---

## 📝 Output Files Explained

### Filename Format

```
{category_name}_query{query_idx}_support{support_idx}.png
```

**Examples:**
- `beaver_body_query123_support100.png`
  - Category: beaver_body
  - Query image index: 123
  - Support image index: 100

**Why this format:**
- Easy to sort by category
- Easy to identify which images were used
- Can trace back to original dataset indices

### Visualization Log

```
visualization_output.log
```

Contains:
- Console output from visualization run
- Any warnings or errors
- Summary statistics
- Useful for debugging

---

## 🎓 What Your PhD Mentor Will Look For

### 1. Qualitative Validation

**Good signs:**
- Predictions anatomically plausible
- Skeleton structure preserved
- Consistent performance across similar categories
- Clear use of support template

**Red flags:**
- Random predictions
- Broken skeletons
- Model ignoring support
- Copy-pasting support onto query

### 2. Category Generalization

**Evidence of learning:**
- Unseen categories still get reasonable predictions
- Model adapts to different animal types
- Support conditioning guides predictions

**Evidence of overfitting:**
- Good on training-like categories
- Fails on very different categories
- Memorization vs understanding

### 3. Error Analysis

**Systematic errors (fixable):**
- Consistent left/right flips → data augmentation issue
- Consistent scale errors → normalization issue
- Consistent misalignment → preprocessing bug

**Random errors (harder to fix):**
- No clear pattern → model capacity or training time
- Category-specific → data quality or quantity

---

## 🔬 Advanced Usage

### Compare Multiple Checkpoints

```bash
# Visualize evolution of predictions
for epoch in 010 020 030 040 050; do
    python visualize_cape_predictions.py \
        --checkpoint outputs/cape_run/checkpoint_e${epoch}_*.pth \
        --categories 40 55 \
        --num_samples 2 \
        --output_dir visualizations/epoch_${epoch}
done

# Now compare visualizations/epoch_*/beaver_body_*.png across epochs
```

### Focus on Difficult Categories

```bash
# After identifying low-PCK categories from evaluation, visualize them:
python visualize_cape_predictions.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck_*.pth \
    --categories 72 83 91  # Difficult categories \
    --num_samples 10 \
    --output_dir visualizations/difficult_cases
```

### Generate Animation (Optional)

```bash
# After visualizing multiple epochs, create GIF:
cd visualizations/
for cat in beaver_body kangaroo_body; do
    convert -delay 50 -loop 0 \
        epoch_*/${cat}_query*_support*.png \
        ${cat}_training_progress.gif
done
```

---

## 📚 Related Documentation

- **[TRAINING_INFERENCE_IO.md](TRAINING_INFERENCE_IO.md)**: Understand model inputs/outputs
- **[CRITICAL_FIX_VALIDATION_INFERENCE.md](../CRITICAL_FIX_VALIDATION_INFERENCE.md)**: Why validation uses autoregressive inference
- **[VISUALIZATION_QUICK_START.md](../VISUALIZATION_QUICK_START.md)**: Simpler visualization (GT only, no model)

---

## 🐛 Known Issues & Fixes

### Issue: Support and query from same image

**Symptom:** Perfect PCK@100% for some samples

**Cause:** Script checks for this now and skips

**If you see this:** Update to latest `visualize_cape_predictions.py`

### Issue: Missing skeleton edges

**Symptom:** Keypoints shown but no connecting lines

**Cause:** Some categories may have empty skeleton definitions in annotations

**Solution:** This is expected for some MP-100 categories

### Issue: Variable number of keypoints

**Symptom:** Some predictions have more/fewer keypoints than GT

**Cause:** Different categories have different keypoint counts (9 to 39)

**Solution:** Script now trims predictions to match each category's actual count

---

## Summary

**The visualization system lets you:**
- ✅ See what the model actually predicts (not just metrics)
- ✅ Validate that 1-shot conditioning works
- ✅ Identify failure modes and categories
- ✅ Debug training issues visually
- ✅ Generate figures for papers/presentations

**Key insight:**
Looking at visualizations is often more informative than just looking at PCK numbers. A model with 40% PCK might be making plausible predictions with small errors, while another with 40% PCK might be making nonsense predictions that happen to get some keypoints right by chance.
