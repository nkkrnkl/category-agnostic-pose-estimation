# ✅ GT Preprocessing Visualization Complete

**Date:** November 25, 2025  
**Status:** PRODUCTION READY

---

## 🎯 What This Does

**Visualizes ground truth annotations in TWO states:**

1. **LEFT Panel:** Original unscaled image
   - Shows the image exactly as stored in the dataset
   - Displays GT keypoints in original pixel coordinates
   - Shows bounding box (cyan dashed rectangle)

2. **RIGHT Panel:** Preprocessed image (EXACTLY as used in training)
   - Cropped to bounding box
   - Resized to 512×512
   - Keypoints transformed accordingly
   - **This is EXACTLY what the model sees during training**

**Purpose:** Verify that the preprocessing pipeline (crop → resize → normalize) correctly preserves all ground truth annotations.

---

## 🚀 Quick Start

### Basic Usage

```bash
# Visualize 10 validation samples
python scripts/visualize_gt_preprocessing.py --split val --num-samples 10

# Visualize specific category (golden retriever face = 48)
python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 5

# Visualize test set
python scripts/visualize_gt_preprocessing.py --split test --num-samples 10
```

---

## 📊 What You See

### Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│         Ground Truth Preprocessing Verification                 │
│         Image: goldenretriever_face/goldenretriever_18.jpg      │
├──────────────────────────┬──────────────────────────────────────┤
│   ORIGINAL IMAGE         │   PREPROCESSED (Training Pipeline)   │
│   goldenretriever_face   │   goldenretriever_face               │
│   Size: 1844x1224        │   Size: 512x512                      │
│   Bbox: [310, 334, ...]  │   (resized from 619x964)             │
│                          │                                       │
│   ┌─────────────┐        │   ┌──────────────────┐               │
│   │  🔵 Bbox    │        │   │                  │               │
│   │   ╱───╲     │        │   │   ╱────╲         │               │
│   │  🟢 0  │    │        │   │  🟢 0   │         │               │
│   │  🟢 1  │    │        │   │  🟢 1   │         │               │
│   │  🟢 2──🟢 3 │        │   │  🟢 2──🟢 3       │               │
│   │  🟢 4  │    │        │   │  🟢 4   │         │               │
│   │   ╲───╱     │        │   │   ╲────╱         │               │
│   └─────────────┘        │   └──────────────────┘               │
│                          │                                       │
└──────────────────────────┴──────────────────────────────────────┘
Legend: 🟢 Visible (v=2)  ❌ Not Visible (v=1)  🟢 Skeleton  🔵 Bbox
```

### Visualization Elements

**LEFT Panel (Original):**
- Full original image (e.g., 1844×1224)
- 🔵 **Cyan dashed box**: Bounding box
- 🟢 **Green circles**: Visible keypoints (v=2)
- ❌ **Red X**: Not visible keypoints (v=1)
- 🟢 **Green lines**: Skeleton edges
- 🟡 **Yellow numbers**: Keypoint indices (0-based)

**RIGHT Panel (Preprocessed):**
- Cropped and resized to 512×512
- Same keypoint markers (transformed coordinates)
- Same skeleton edges
- No bounding box (entire image is the bbox region)

---

## ✅ Test Results

Already tested successfully:

### General Validation Samples
```bash
python scripts/visualize_gt_preprocessing.py --split val --num-samples 5
```

**Output:**
- ✅ 5/5 samples visualized
- ✅ Categories: weasel, guanaco, retriever, fly, beaver
- ✅ All saved to `outputs/gt_preprocessing_vis/split1_val/`

### Category-Specific (Golden Retriever)
```bash
python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 3
```

**Output:**
- ✅ 3/3 golden retriever faces visualized
- ✅ Saved to `outputs/gt_preprocessing_vis/split1_val/cat_48/`

---

## 🔍 What to Verify

When reviewing the visualizations, check:

### ✓ Bbox Correctness (LEFT panel)
- All keypoints are **inside** the cyan bounding box
- Bbox tightly encloses the object/animal
- No keypoints are cut off at bbox edges

### ✓ Cropping Correctness (RIGHT panel)
- Image shows only the bbox region
- No important parts are cropped out
- Aspect ratio may differ from original (if bbox wasn't square)

### ✓ Keypoint Transformation
- Keypoints in RIGHT panel match relative positions in LEFT panel
- All keypoints that were inside bbox are now in 512×512 image
- Visibility markers (green/red) match in both panels

### ✓ Skeleton Preservation
- Green skeleton lines connect same keypoint pairs in both panels
- No broken or disconnected skeleton edges
- Edge connectivity preserved after transformation

### ✓ Overall Quality
- RIGHT panel is what the model sees during training
- Image quality sufficient for keypoint detection
- No artifacts from resizing

---

## 📁 Output Structure

```
outputs/gt_preprocessing_vis/
├── split1_val/
│   ├── vis_0000_000000050050.png    # Weasel
│   ├── vis_0001_guanaco_103.png     # Guanaco
│   ├── vis_0002_goldenretriever_50.png  # Golden retriever
│   ├── vis_0003_108.png             # Fly
│   └── vis_0004_000000019894.png    # Beaver
│
├── split1_val/cat_48/               # Category-specific
│   ├── vis_0000_goldenretriever_18.png
│   ├── vis_0001_goldenretriever_131.png
│   └── vis_0002_goldenretriever_64.png
│
└── split1_test/                     # Test set
    └── ...
```

---

## 🔧 Command-Line Arguments

```bash
python scripts/visualize_gt_preprocessing.py \
    --split val \                    # train, val, or test
    --data-split split1 \            # split1 through split5
    --num-samples 10 \               # Number of samples
    --category 48 \                  # Optional: specific category
    --output-dir outputs/gt_preprocessing_vis \  # Output dir
    --data-root data \               # Data root directory
    --random-seed 42                 # Random seed
```

---

## 🎓 Use Cases

### 1. Verify Preprocessing Pipeline
**Check that crop + resize doesn't break annotations:**
```bash
python scripts/visualize_gt_preprocessing.py --split val --num-samples 20
```

**What to look for:**
- All keypoints remain visible after cropping
- Skeleton edges are preserved
- 512×512 images look reasonable

### 2. Debug Bbox Issues
**If training has issues, check bbox quality:**
```bash
python scripts/visualize_gt_preprocessing.py --split val --num-samples 50
```

**What to look for:**
- Bboxes aren't too tight (cutting off keypoints)
- Bboxes aren't too loose (lots of background)
- Keypoints are well-centered in bbox

### 3. Category-Specific Analysis
**Understand how different categories are preprocessed:**
```bash
# Animal bodies (17 keypoints, large skeletons)
python scripts/visualize_gt_preprocessing.py --split val --category 91 --num-samples 10

# Animal faces (9 keypoints, no skeleton)
python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 10
```

**What to look for:**
- Face categories: keypoints clustered in facial region
- Body categories: skeleton spanning full body
- Different aspect ratio handling

### 4. Pre-Training Verification
**Before starting a training run:**
```bash
# Check all validation categories
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_preprocessing.py --split val --category $cat_id --num-samples 3
done
```

**What to look for:**
- No systematic issues across categories
- All preprocessing working correctly
- Ready to start training

---

## 📊 Preprocessing Pipeline (Verified)

This script replicates the **EXACT** preprocessing from `datasets/mp100_cape.py`:

### Step 1: Crop to Bounding Box
```python
# Line 332 in mp100_cape.py
img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
```

### Step 2: Make Keypoints Relative to Bbox
```python
# Lines 347-349 in mp100_cape.py
kpts_array[:, 0] -= bbox_x  # x relative to bbox
kpts_array[:, 1] -= bbox_y  # y relative to bbox
```

### Step 3: Resize to 512×512
```python
# Lines 892-893 in mp100_cape.py (validation/test)
A.Compose([
    A.Resize(height=512, width=512)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
```

**Result:** Image is 512×512, keypoints in range [0, 512]×[0, 512]

*(Final normalization to [0,1] happens later in `episodic_sampler.py`)*

---

## 🆚 Comparison with Other Scripts

### vs. `visualize_gt_annotations.py`
- **That script:** Shows only original images with GT
- **This script:** Shows original **AND** preprocessed side-by-side
- **Use this when:** You want to verify preprocessing correctness

### vs. `eval_cape_checkpoint.py`
- **That script:** Shows support, query GT, and model predictions
- **This script:** Shows original vs. preprocessed GT only
- **Use this when:** You want to debug preprocessing (no model needed)

### Together
1. **This script** → Verify GT preprocessing is correct
2. **eval_cape_checkpoint.py** → Evaluate model on preprocessed GT

---

## 🎯 Expected Observations

### Correct Behavior

✅ **Keypoints inside bbox:** All keypoints in LEFT panel are within cyan box  
✅ **Keypoints preserved:** All keypoints visible in LEFT are visible in RIGHT  
✅ **Skeleton intact:** Green lines connect same keypoints in both panels  
✅ **Reasonable crop:** RIGHT panel shows the object clearly  
✅ **No clipping:** No keypoints are cut off during cropping  

### Potential Issues

❌ **Keypoints outside bbox:** Some keypoints in LEFT are outside cyan box  
  → **Problem:** Bad bbox annotation or keypoint annotation  
  → **Impact:** Those keypoints will be lost during cropping

❌ **Tight crop:** Bbox barely contains keypoints  
  → **Problem:** No context around object  
  → **Impact:** Model may struggle to localize keypoints

❌ **Loose crop:** Bbox has lots of background  
  → **Problem:** Object is small in 512×512 image  
  → **Impact:** Keypoints may be too close together after resize

❌ **Aspect ratio distortion:** Object looks squished in RIGHT  
  → **This is NORMAL:** Bbox may not be square, resize makes it square  
  → **Impact:** None (model learns to handle this)

---

## 🧪 Validation Checklist

Before training, verify:

- [ ] Run on 20+ validation samples
- [ ] Check all 10 validation categories
- [ ] Verify keypoints stay within bbox
- [ ] Confirm skeleton edges preserved
- [ ] Check 512×512 images look reasonable
- [ ] No systematic crop/resize issues

**If all checks pass:** ✅ Preprocessing is correct, ready to train!

---

## 📚 Related Documentation

- **Preprocessing Pipeline:** `docs/NORMALIZATION_PIPELINE_EXPLAINED.md`
- **Bbox Cropping:** `docs/BBOX_CROPPING_IMPLEMENTATION.md`
- **Dataset Code:** `datasets/mp100_cape.py` (lines 329-623)
- **Original GT Vis:** `scripts/GT_VISUALIZATION_README.md`
- **Model Evaluation:** `EVAL_SCRIPT_COMPLETE.md`

---

## ⚡ Quick Reference

```bash
# Validation overview
python scripts/visualize_gt_preprocessing.py --split val --num-samples 20

# Specific category
python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 10

# Test set
python scripts/visualize_gt_preprocessing.py --split test --num-samples 20

# All validation categories (3 each)
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_preprocessing.py --split val --category $cat_id --num-samples 3
done
```

---

## ✅ Summary

**Created:** `scripts/visualize_gt_preprocessing.py`

**What it does:**
- Shows original image WITH bounding box (LEFT)
- Shows preprocessed 512×512 image (RIGHT)
- Verifies GT annotations remain valid after preprocessing

**Test results:**
- ✅ 5 general validation samples
- ✅ 3 golden retriever faces (category 48)
- ✅ All saved to `outputs/gt_preprocessing_vis/`

**Use it to:**
- Verify preprocessing pipeline correctness
- Debug bbox/cropping issues
- Understand category-specific preprocessing
- Pre-training data quality check

**Ready to use!** 🎉

---

**Last updated:** November 25, 2025

