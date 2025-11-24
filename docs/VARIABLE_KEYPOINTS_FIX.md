# Variable-Length Keypoint Sequences Fix

## 🐛 **The Bug**

During validation, the code crashed with:
```
AssertionError: Visibility length (9) must match keypoints (17)
```

## 🔍 **Root Cause Analysis**

The issue was **NOT a data loading bug** as initially suspected. It was a **fundamental architecture mismatch**:

### The Problem

MP-100 contains categories with **different numbers of keypoints**:
- `beaver_body` (category 91): **17 keypoints**
- `przewalskihorse_face` (category 12): **9 keypoints**
- And many others with varying counts

### Why It Failed

1. **Model Behavior**: The Raster2Seq model outputs a **fixed-length sequence** based on the maximum number of keypoints seen during the current batch/episode.

2. **Batch Composition**: During validation, even though each episode contains queries from a single category, the DataLoader was batching **multiple episodes together** (with `batch_size=2`).

3. **The Crash**: When batch contained:
   - Episode 1: Queries from "beaver_body" (17 keypoints)
   - Episode 2: Queries from "przewalskihorse_face" (9 keypoints)
   
   The collate function would pad all episodes to 17 keypoints, but:
   - Model outputs: `pred_kpts.shape = [4, 17, 2]` (4 queries, 17 keypoints each)
   - Metadata for queries 0-1: `visibility = [17 values]` ✅
   - Metadata for queries 2-3: `visibility = [9 values]` ❌
   
   During PCK evaluation, `compute_pck_bbox` expected 17 elements in `visibility` for query 2, but only got 9!

## ✅ **The Fix**

### 1. Force `batch_size=1` for Validation

**File**: `train_cape_episodic.py`

```python
val_loader = build_episodic_dataloader(
    base_dataset=val_dataset,
    category_split_file=str(category_split_file),
    split='val',
    batch_size=1,  # CRITICAL: Ensure all queries in batch from SAME category
    ...
)
```

**Why**: This ensures each batch contains queries from only ONE category, preventing mixed keypoint counts.

### 2. Trim Predictions to Match Category

**File**: `engine_cape.py`

Instead of passing fixed-length predictions to PCK evaluator, we now:

```python
for idx, meta in enumerate(query_metadata):
    vis = meta.get('visibility', [])
    num_kpts_for_category = len(vis)  # Actual keypoints for this category
    
    # Trim predictions to match category's keypoint count
    pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])
    gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])
    
    visibility_list.append(vis)

# Pass as lists (variable-length) to evaluator
pck_evaluator.add_batch(
    pred_keypoints=pred_kpts_trimmed,  # List of tensors
    gt_keypoints=gt_kpts_trimmed,      # List of tensors
    ...
)
```

**Why**: Even with `batch_size=1`, queries within an episode have the same number of keypoints, but the model may output more. Trimming ensures we only evaluate the relevant keypoints.

### 3. Support Variable-Length Inputs in Evaluator

**File**: `util/eval_utils.py`

Updated `PCKEvaluator.add_batch()` to handle both:
- **Tensor input**: `(B, N, 2)` for fixed-length batches
- **List input**: `[(N_1, 2), (N_2, 2), ...]` for variable-length sequences

```python
def add_batch(self, pred_keypoints, gt_keypoints, ...):
    if isinstance(pred_keypoints, list):
        # Variable-length sequences
        batch_size = len(pred_keypoints)
    else:
        # Fixed-length batch
        batch_size = pred_keypoints.shape[0]
    
    for i in range(batch_size):
        if isinstance(pred_keypoints, list):
            pred_i = pred_keypoints[i]  # Already correct length
            gt_i = gt_keypoints[i]
        else:
            pred_i = pred_keypoints[i]
            gt_i = gt_keypoints[i]
        
        # Compute PCK for this sample
        pck, correct, visible = compute_pck_bbox(pred_i, gt_i, ...)
```

**Why**: This makes the evaluator flexible enough to handle both fixed and variable-length sequences.

### 4. Added Category-Aware Validation

**File**: `datasets/mp100_cape.py`

Added a helper method to get expected keypoint count per category:

```python
def _get_num_keypoints_for_category(self, category_id):
    """Get the expected number of keypoints for a category."""
    try:
        cat_info = self.coco.loadCats(category_id)[0]
        keypoint_names = cat_info.get('keypoints', [])
        return len(keypoint_names) if keypoint_names else None
    except Exception as e:
        return None
```

And updated the final validation in `__getitem__` to check against the category definition:

```python
# Check 2: Both must match the category definition
if expected_num_kpts is not None and num_kpts != expected_num_kpts:
    raise ValueError(
        f"keypoints length ({num_kpts}) != expected for category {category_id} ({expected_num_kpts})"
    )
```

**Why**: This provides an early warning if data loading produces incorrect keypoint counts.

## 📊 **Validation Results**

After the fix, validation completed successfully:

```
Validation: 100%|██| 10/10 [00:16<00:00,  1.70s/it]
PCK@0.2: 100.00% (204/204 keypoints)
```

## 🎓 **Key Lessons**

1. **Category-agnostic** doesn't mean **category-uniform**! Different categories can have vastly different numbers of keypoints.

2. **Episodic sampling** guarantees queries from the same category within an episode, but **batching multiple episodes** can mix categories.

3. **Model output length** is often fixed (or determined by max in batch), so evaluation code must handle trimming/padding appropriately.

4. **Always validate against ground truth schema** (COCO annotations) rather than assuming processed data is correct.

## 🚀 **Impact**

This fix ensures:
- ✅ Training and validation can handle MP-100's diverse categories
- ✅ PCK evaluation is accurate for categories with any number of keypoints
- ✅ No shape mismatches during evaluation
- ✅ Proper category-agnostic pose estimation capability

