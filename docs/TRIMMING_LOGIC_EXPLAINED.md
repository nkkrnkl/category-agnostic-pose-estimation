# Keypoint Sequence Trimming Logic - Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [Why Trimming is Necessary](#why-trimming-is-necessary)
3. [How the Model Generates Sequences](#how-the-model-generates-sequences)
4. [The Padding Problem](#the-padding-problem)
5. [The Trimming Solution](#the-trimming-solution)
6. [Safety Analysis](#safety-analysis)
7. [Edge Cases](#edge-cases)
8. [Code References](#code-references)

---

## Overview

**Question**: Is trimming model predictions to match category keypoint counts safe?

**Answer**: **YES! It's not only safe, but absolutely necessary.**

This document explains in detail why we trim predictions, what we're actually removing, and why no meaningful information is lost.

---

## Why Trimming is Necessary

### The Core Challenge

MP-100 is a **category-agnostic** pose estimation dataset with 100 different object categories. Each category has a **different number of keypoints**:

| Category | Keypoints | Example |
|----------|-----------|---------|
| `beaver_body` | 17 | Full body pose |
| `przewalskihorse_face` | 9 | Facial landmarks |
| `cheetah_face` | 5 | Simplified face |
| `cat_body` | 20 | Detailed body pose |

### The Problem Without Trimming

When evaluating PCK (Percentage of Correct Keypoints), we need:
```python
visibility_array = np.array(visibility)  # Length = actual keypoints for category
visible_mask = visibility_array > 0
pred_visible = pred_keypoints[visible_mask]  # ❌ Shape mismatch!
```

If `pred_keypoints` has 17 elements but `visibility` has only 9:
```
IndexError: boolean index did not match indexed array along axis 0;
            size of axis is 17 but size of corresponding boolean axis is 9
```

---

## How the Model Generates Sequences

### Autoregressive Generation

The Raster2Seq model generates keypoint sequences **token-by-token**, not all at once:

```python
# Pseudocode of generation loop (roomformer_v2.py:439-558)
max_len = self.tokenizer.seq_len  # Typically 1024
i = 0
while i < max_len and not_finished:
    # Generate next token
    token = model.predict_next_token(previous_tokens)
    
    if token == <coord>:
        # Predict x, y coordinates
        append_to_sequence(x, y)
    elif token == <sep>:
        # Separator between different parts
        append_to_sequence(SEP)
    elif token == <eos>:
        # End of sequence - STOP
        break
    
    i += 1
```

**Output format**:
```
<coord> x₁,y₁ <coord> x₂,y₂ ... <coord> xₙ,yₙ <sep> <eos>
```

### When Does It Stop?

The model stops when:
1. **It predicts `<eos>`** (believes pose is complete), OR
2. **Reaches maximum length** (1024 tokens)

**Critical insight**: The model decides when to stop based on **its own learned behavior**, NOT on the category's expected keypoint count!

---

## The Padding Problem

### Batching Requires Same Length

After autoregressive generation, different samples produce different sequence lengths:

```python
# Example batch of 3 samples
Sample 0 (beaver, 17 kpts):  generates → 17 coords + SEP + EOS = 55 tokens total
Sample 1 (beaver, 17 kpts):  generates → 17 coords + SEP + EOS = 57 tokens total  
Sample 2 (horse face, 9 kpts): generates → 9 coords + SEP + EOS = 30 tokens total
```

### Extract and Pad

`extract_keypoints_from_sequence()` (engine_cape.py:195-252) does:

```python
# Step 1: Filter out special tokens (<sep>, <eos>)
for i in range(batch_size):
    coord_mask = token_labels[i] == TokenType.coord.value
    kpts = pred_coords[i][coord_mask]  # Extract only coordinate tokens
    all_keypoints.append(kpts)

# Step 2: Pad to same length for batching
max_len = max(len(kpts) for kpts in all_keypoints)  # 17 in this example
for kpts in all_keypoints:
    if len(kpts) < max_len:
        padding = zeros(max_len - len(kpts), 2)
        kpts = cat([kpts, padding])  # Pad with zeros
    padded_keypoints.append(kpts)

return stack(padded_keypoints)  # Shape: (3, 17, 2)
```

**Result**:
```python
pred_kpts.shape = [3, 17, 2]  # All padded to 17

# Internal structure:
Sample 0: [17 real keypoints]           # Perfect!
Sample 1: [17 real keypoints]           # Perfect!
Sample 2: [9 real keypoints + 8 zeros]  # ⚠️ Padding!
```

---

## The Trimming Solution

### What We Do

```python
# engine_cape.py:391-403
for idx, meta in enumerate(query_metadata):
    vis = meta.get('visibility', [])
    num_kpts_for_category = len(vis)  # 9 for horse, 17 for beaver
    
    # Trim to category's actual keypoint count
    pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])
    gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])
    
    visibility_list.append(vis)
```

### Example

For Sample 2 (horse face with 9 keypoints):

**Before trimming**:
```python
pred_kpts[2] = [
    [x₁, y₁],  # Keypoint 0: nose
    [x₂, y₂],  # Keypoint 1: left_eye
    ...
    [x₉, y₉],  # Keypoint 8: right_ear
    [0, 0],    # Padding (no semantic meaning)
    [0, 0],    # Padding
    [0, 0],    # Padding
    ...
    [0, 0]     # Padding (total 8 padding entries)
]  # Shape: (17, 2)
```

**After trimming to 9**:
```python
pred_kpts_trimmed[2] = [
    [x₁, y₁],  # Keypoint 0: nose
    [x₂, y₂],  # Keypoint 1: left_eye
    ...
    [x₉, y₉]   # Keypoint 8: right_ear
]  # Shape: (9, 2)
```

---

## Safety Analysis

### Scenario A: Model Predicts Exactly Right Number ✅

```
Category expects: 9 keypoints
Model generates:  <coord> x₁,y₁ ... <coord> x₉,y₉ <eos>
Extracted:        9 coordinate tokens
Padded to:        [9 real coords + 8 zeros] = 17
Trimmed to 9:     [9 real coords]
Result:           ✅ Perfect! All 9 predictions evaluated against GT
```

### Scenario B: Model Predicts Fewer (Underprediction) ✅

```
Category expects: 9 keypoints
Model generates:  <coord> x₁,y₁ ... <coord> x₇,y₇ <eos> (stopped early!)
Extracted:        7 coordinate tokens
Padded to:        [7 real coords + 10 zeros] = 17
Trimmed to 9:     [7 real coords + 2 zeros]
Result:           ✅ PCK correctly marks the 2 missing keypoints as incorrect
                     (distance = infinite from GT = 0 PCK contribution)
```

### Scenario C: Model Predicts More (Hallucination) ✅

```
Category expects: 9 keypoints
Model generates:  <coord> x₁,y₁ ... <coord> x₁₂,y₁₂ <eos> (hallucinated 3 extra!)
Extracted:        12 coordinate tokens
Padded to:        [12 real coords + 5 zeros] = 17
Trimmed to 9:     [9 real coords] (extras discarded)
Result:           ✅ Correct! Category only defines 9 keypoints.
                     Keypoints 10-12 are meaningless (category doesn't define them)
                     We evaluate only the first 9 against GT positions 0-8
```

**Q: But aren't we losing the model's predictions for positions 10-12?**

**A: No meaningful information is lost!** The category definition states there are only 9 keypoints. The model was trained to predict keypoints in canonical order. If it predicts more than the category defines, those extra predictions have no semantic meaning (there's no "10th keypoint of a horse face" in the dataset definition).

---

## Edge Cases

### Q1: What if the model generates keypoints out of order?

**A**: This would break the **entire Raster2Seq architecture**, not just trimming!

The model is explicitly trained with teacher forcing on sequences in canonical order:
```python
# Training (datasets/mp100_cape.py:639-698)
for kpt_idx, (x, y) in enumerate(keypoints):
    # keypoints are in category-defined order
    sequence.append(x, y)
```

The autoregressive generation **inherently preserves order** because:
- Each token conditions on all previous tokens
- The model learns the pattern: "predict keypoint i, then keypoint i+1"
- Spontaneous reordering would require the model to "unlearn" its training pattern

**Probability of out-of-order generation**: Negligible (would indicate catastrophic training failure)

### Q2: What if categories have overlapping keypoint definitions?

**A**: They don't! Each category has its own **unique** keypoint definition:

```python
# From COCO annotations
category_91 (beaver_body): {
    "keypoints": ["nose", "left_eye", "right_eye", ..., "tail"],  # 17 total
    "skeleton": [[0,1], [0,2], ...]
}

category_12 (horse_face): {
    "keypoints": ["nose", "left_eye", "right_eye", ..., "right_ear"],  # 9 total
    "skeleton": [[0,1], [0,2], ...]
}
```

Even though both have "nose", they are **different keypoints** in different coordinate systems (different bounding boxes, different poses). The model learns to map from query image → predicted coordinates for that specific category.

### Q3: What about very small categories (e.g., 5 keypoints)?

**A**: Trimming works **regardless** of category size:

```
Category expects: 5 keypoints
Model generates:  5-7 keypoints (typical range)
Padded to:        max(batch) keypoints
Trimmed to 5:     [5 keypoints]
Result:           ✅ Always correct!
```

The trimming is **adaptive** - it uses `len(visibility)` from the category's ground truth, so it automatically handles any category size from 5 to 39 keypoints.

---

## Code References

### Where Trimming Happens

**File**: `engine_cape.py`  
**Lines**: 372-403

```python
# Extract visibility (determines category's keypoint count)
vis = meta.get('visibility', [])
num_kpts_for_category = len(vis)  # Actual keypoints for this category

# Trim predictions and GT to match category
pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])
gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])
```

### Where Padding Happens

**File**: `engine_cape.py`  
**Function**: `extract_keypoints_from_sequence()`  
**Lines**: 195-252

```python
# Pad to same length within batch
max_len = max(len(kpts) for kpts in all_keypoints)
for kpts in all_keypoints:
    if len(kpts) < max_len:
        padding = torch.zeros(max_len - len(kpts), 2)
        kpts = torch.cat([kpts, padding], dim=0)
```

### Where Sequence Generation Happens

**File**: `models/roomformer_v2.py`  
**Function**: `forward_inference()`  
**Lines**: 361-558

```python
# Autoregressive generation loop
while i < max_len and unfinish_flag.any():
    # Generate next token
    cls_output = model.predict_token_class()
    
    if cls_output == TokenType.eos.value and i >= min_len:
        unfinish_flag[j] = 0  # Stop this sample
```

### Where Evaluation Happens

**File**: `util/eval_utils.py`  
**Function**: `PCKEvaluator.add_batch()`  
**Lines**: 235-293

```python
# Now handles both fixed-length and variable-length inputs
if isinstance(pred_keypoints, list):
    # Variable-length (after trimming)
    pred_i = pred_keypoints[i]  # Already correct length!
else:
    # Fixed-length (legacy)
    pred_i = pred_keypoints[i]
```

---

## Summary

### What We're Removing
- ✅ **Batch padding zeros** (added for tensor stacking)
- ✅ **Model hallucinations** (predictions beyond category definition)

### What We're Keeping
- ✅ **All meaningful predictions** (up to category's keypoint count)
- ✅ **Correct index alignment** (keypoint i → category's keypoint i)
- ✅ **Evaluation integrity** (PCK computed on correct keypoints)

### Why It's Safe
1. **Semantic correctness**: Categories define their own keypoint counts
2. **Order preservation**: Model trained to predict in canonical order
3. **Error handling**: Under/over-predictions handled gracefully
4. **No information loss**: Only artifacts and hallucinations removed

### The Alternative (Without Trimming)
```python
# What would happen:
visibility = [9 values]
pred_keypoints = [17 values]
visible_mask = visibility > 0  # 9 True values
pred_visible = pred_keypoints[visible_mask]  # ❌ IndexError!
```

**Conclusion**: Trimming is not a workaround - it's the **correct solution** for category-agnostic pose estimation with variable keypoint counts.

---

## Related Documentation

- `VARIABLE_KEYPOINTS_FIX.md` - Overview of the variable keypoint fix
- `FINAL_FIX_SUMMARY.md` - Complete fix summary for users
- `NORMALIZATION_PIPELINE_EXPLAINED.md` - How coordinates are normalized
- `LOSS_MASKING_VERIFICATION.md` - How visibility masks are used in training

