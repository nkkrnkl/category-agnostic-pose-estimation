# 🚨 ROOT CAUSE FOUND: PCK Always 100% Bug

**Date:** November 25, 2025  
**Status:** ✅ IDENTIFIED - Implementation in progress

---

## 🎯 Executive Summary

**Found the root cause of PCK being stuck at 100%!**

The bug has **TWO components**:
1. ✅ Using GT token labels to extract predictions (FIXED)
2. 🚨 **COORDINATE SPACE MISMATCH** (NEW - This is the main issue!)

---

## 🔍 The Critical Bug: Coordinate Space Mismatch

### The Problem

**Keypoints are in `[0, 1]` normalized space, but PCK threshold is computed using PIXEL bbox dimensions.**

This creates a massive scaling error that makes the threshold ~100x too large!

### Debug Evidence

From `DEBUG_PCK=1` output:

```
pred_kpts[0, :3]: [[0.1028, 0.0823], [0.2827, 0.1393], [0.7248, 0.1132]]  ← [0,1] space
gt_kpts[0, :3]: [[0.9346, 0.1162], [0.7129, 0.1668], [0.7129, 0.1668]]    ← [0,1] space

distances (first 5): [0.8324, 0.4310, 0.0549, 0.1112, 0.1902]  ← Computed in [0,1] space!
bbox_size: 91.30  ← PIXELS!
normalized_distances: [0.0091, 0.0047, 0.0006, 0.0012, 0.0021]  ← 0.8324 / 91.30 = 0.0091
threshold (alpha): 0.2
num_correct: 9 / 9
PCK: 100.00%  ← ALL correct because 0.0091 < 0.2
```

### The Math

**Point 0:**
- Pred: `[0.1028, 0.0823]`
- GT: `[0.9346, 0.1162]`
- Distance in [0,1] space: `√((0.9346-0.1028)² + (0.1162-0.0823)²) = 0.8324`

**Normalization:**
- Bbox: `44 × 80 pixels`
- Bbox diagonal: `√(44² + 80²) = 91.30 pixels`
- Normalized distance: `0.8324 / 91.30 = 0.0091`  ← 🚨 BUG!
- Threshold: `0.2`
- Result: `0.0091 < 0.2` ✓ "Correct" (WRONG!)

**What SHOULD happen:**
- If keypoints in [0,1] space, bbox should ALSO be in [0,1] space:
  - Bbox in normalized space: `(44/512) × (80/512) = 0.086 × 0.156`
  - Bbox diagonal in normalized space: `√(0.086² + 0.156²) = 0.178`
  - Normalized distance: `0.8324 / 0.178 = 4.68`
  - Result: `4.68 > 0.2` ✗ Incorrect (CORRECT!)

OR:
- Convert keypoints to PIXEL space before computing PCK:
  - Pred_pixels: `[0.1028 * 512, 0.0823 * 512] = [52.6, 42.1]`
  - GT_pixels: `[0.9346 * 512, 0.1162 * 512] = [478.5, 59.5]`
  - Distance: `√((478.5-52.6)² + (59.5-42.1)²) = 426.2 pixels`
  - Normalized: `426.2 / 91.30 = 4.67`
  - Result: `4.67 > 0.2` ✗ Incorrect (CORRECT!)

---

## 📊 Why This Makes PCK = 100%

The mismatch causes the effective threshold to be ~100x larger than intended:

**Intended:**
- Threshold = 0.2 × bbox_diagonal
- If bbox is 91 pixels, threshold = 18.2 pixels

**Actual (buggy):**
- Distance computed in [0,1] space: `~0.8`
- Divided by 91 pixels: `0.8 / 91 = 0.0088`
- Compared to threshold: `0.0088 < 0.2` ✓
- This means predictions can be off by 0.8 in normalized space (i.e., almost the entire image!) and still be considered "correct"!

**Effective threshold:**
- In [0,1] space, threshold is actually: `0.2 × 91 = 18.2` (no normalization)
- But max distance in [0,1] space is: `√(1² + 1²) = 1.414`
- So effectively, threshold is: `18.2 / 1.414 = 12.9` (way larger than the whole image!)

---

## ✅ Fixes Already Implemented

### Fix #1: Use Predicted Token Labels ✅

**What was wrong:**
```python
# BEFORE (WRONG):
pred_kpts = extract_keypoints_from_sequence(pred_coords, token_labels, mask)  # Uses GT labels!
gt_kpts = extract_keypoints_from_sequence(gt_coords, token_labels, mask)
```

**What was fixed:**
```python
# AFTER (CORRECT):
pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)  # Uses predicted labels!
gt_kpts = extract_keypoints_from_sequence(gt_coords, token_labels, mask)
```

**Impact:** This fix ensures we extract keypoints based on what the model actually predicted, not what we expected.

**Status:** ✅ Implemented in `engine_cape.py` and `scripts/eval_cape_checkpoint.py`

---

## 🚨 Fix #2: Coordinate Space Consistency (TO BE IMPLEMENTED)

### Option A: Scale Keypoints to Pixels

**Change:** `engine_cape.py`, `scripts/eval_cape_checkpoint.py`

Before PCK computation:
```python
# After extracting keypoints
pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)  # [0,1] space
gt_kpts = extract_keypoints_from_sequence(gt_coords, token_labels, mask)  # [0,1] space

# NEW: Scale to pixel space (relative to 512x512 image)
pred_kpts_pixels = pred_kpts * 512.0
gt_kpts_pixels = gt_kpts * 512.0

# Then compute PCK using pixel coordinates
pck_evaluator.add_batch(
    pred_keypoints=pred_kpts_pixels,  # Now in pixels
    gt_keypoints=gt_kpts_pixels,      # Now in pixels
    bbox_widths=bbox_widths,          # Already in pixels
    bbox_heights=bbox_heights,        # Already in pixels
    ...
)
```

### Option B: Normalize Bbox to [0,1] Space

**Change:** `util/eval_utils.py` in `compute_pck_bbox`

```python
# After computing bbox_size
if normalize_by == 'diagonal':
    bbox_size_pixels = np.sqrt(bbox_width ** 2 + bbox_height ** 2)
    # NEW: If keypoints are in [0,1] space, normalize bbox too
    bbox_size = bbox_size_pixels / 512.0  # Assuming 512x512 preprocessed images
```

### Recommended: Option A

**Why:**
- More explicit and clear
- Matches how visualization denormalization works
- Easier to verify correctness
- No ambiguity about coordinate spaces

---

## 🧪 Verification Plan

After implementing fix:

1. **Run with DEBUG_PCK=1:**
   ```bash
   DEBUG_PCK=1 python scripts/eval_cape_checkpoint.py \
       --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
       --num-episodes 3 \
       --device cpu
   ```

2. **Expected output:**
   ```
   [DEBUG_PCK] compute_pck_bbox:
     num_visible: 9
     bbox_size: 91.30
     threshold (pixels): 18.26
     distances (first 5): [426.2, 220.8, 28.1, ...]  ← NOW IN PIXELS!
     normalized_distances: [4.67, 2.42, 0.31, ...]   ← Reasonable values
     num_correct: 2 / 9  ← Some correct, some incorrect
     PCK: 22.22%  ← Realistic PCK value!
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/test_pck_pipeline.py -v
   ```
   - All tests should pass
   - PCK for random predictions should be < 50%

4. **Check actual model quality:**
   - Early checkpoint (epoch 1-5): PCK should be LOW (< 30%)
   - Well-trained checkpoint: PCK should be reasonable (40-70%)

---

## 📝 Next Steps

1. ✅ Document root cause (this file)
2. 🔄 Implement Option A (scale keypoints to pixels)
3. 🔄 Update tests to use correct coordinate space
4. 🔄 Run full validation
5. 🔄 Create comprehensive documentation

---

**Status:** ROOT CAUSE IDENTIFIED ✅  
**Fix:** IN PROGRESS 🔄

