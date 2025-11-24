# Query Metadata Fix - Issue #6

## Problem

**Query metadata was collected but not passed through to the model/evaluation!**

The `episodic_collate_fn()` was collecting `query_metadata` from each episode but **not including it in the returned batch dict**. This broke PCK evaluation because critical information was missing:

- `bbox_width` and `bbox_height` - Required for PCK@bbox normalization
- `visibility` - Required for visibility masking in evaluation
- Additional metadata (`image_id`, `height`, `width`) - Useful for debugging

## Impact

### Before Fix ❌

```python
# In episodic_collate_fn():
for episode in batch:
    query_metadata_list.extend(episode['query_metadata'])  # ← Collected...

return {
    'query_images': query_images,
    'query_targets': batched_seq_data,
    # query_metadata NOT INCLUDED!  ← ...but not returned
}
```

**Result**: 
- PCK evaluation had to use dummy bbox dimensions (always 512×512)
- Visibility masking couldn't work in evaluation
- No way to debug which images had issues

### After Fix ✅

```python
# In episodic_collate_fn():
for episode in batch:
    query_metadata_list.extend(episode['query_metadata'])  # ← Collected

return {
    'query_images': query_images,
    'query_targets': batched_seq_data,
    'query_metadata': query_metadata_list,  # ← Now properly returned!
}
```

**Result**:
- PCK uses actual bbox dimensions for correct normalization
- Visibility masking works in evaluation
- Full metadata available for debugging

## Changes Made

### File: `datasets/episodic_sampler.py`

**Line 287**: Added `query_metadata_list` initialization
```python
query_metadata_list = []  # NEW: Collect query metadata
```

**Line 297**: Extract metadata from episodes
```python
query_metadata_list.extend(episode['query_metadata'])  # NEW: Extract query metadata
```

**Lines 371-390**: Added documentation and return value
```python
# ========================================================================
# CRITICAL FIX: Include query_metadata for evaluation
# ========================================================================
# query_metadata contains essential information for PCK evaluation:
#   - bbox_width, bbox_height: Original bbox dimensions for PCK normalization
#   - visibility: Keypoint visibility flags for masking in evaluation
#   - image_id, height, width: Additional metadata for debugging
#
# This was previously collected but not passed through, breaking evaluation.
# Now it's properly included in the batch for use in engine_cape.py
# ========================================================================

return {
    ...
    'query_metadata': query_metadata_list,  # List of B*K metadata dicts
    ...
}
```

## What query_metadata Contains

Each entry in `query_metadata_list` is a dict with:

```python
{
    'image_id': int,           # Image identifier
    'height': int,             # Post-resize height (512)
    'width': int,              # Post-resize width (512)
    'keypoints': list,         # Ground truth keypoints (visible only)
    'num_keypoints': int,      # Number of keypoints for this category
    'bbox': [x, y, w, h],     # Original bbox in full image
    'bbox_width': float,       # Original bbox width (for PCK)
    'bbox_height': float,      # Original bbox height (for PCK)
    'visibility': list,        # Visibility flags for all keypoints
}
```

## How It's Used in Evaluation

### In `engine_cape.py` - `evaluate_unseen_categories()`:

```python
# Extract query metadata from batch
query_metadata = batch.get('query_metadata', None)

if query_metadata is not None and len(query_metadata) > 0:
    # Extract bbox dimensions from metadata
    for meta in query_metadata:
        bbox_w = meta.get('bbox_width', 512.0)
        bbox_h = meta.get('bbox_height', 512.0)
        bbox_widths.append(bbox_w)
        bbox_heights.append(bbox_h)
        
        # Extract visibility for masking
        visibility = meta.get('visibility', [])
        visibility_list.append(visibility)
```

### PCK Computation:

```python
# Compute PCK for each query
pck_score = compute_pck_bbox(
    pred_kpts_filtered,
    gt_kpts_filtered,
    gt_visibility_flags,
    bbox_width,      # ← From query_metadata!
    bbox_height,     # ← From query_metadata!
    threshold=pck_threshold
)
```

## Data Flow

```
EpisodicDataset.__getitem__()
  ↓
  Creates query_metadata list with bbox info
  ↓
  Returns episode dict with 'query_metadata' key
  ↓
episodic_collate_fn()
  ↓
  Extracts query_metadata from each episode
  ↓
  Extends query_metadata_list (B*K entries)
  ↓
  Returns batch dict with 'query_metadata' key  ← FIX: Now included!
  ↓
engine_cape.py - evaluate_unseen_categories()
  ↓
  Gets query_metadata from batch
  ↓
  Uses bbox_width, bbox_height for PCK normalization
  ↓
  Uses visibility for masking
  ↓
PCK evaluation with correct normalization ✓
```

## Benefits

✅ **Correct PCK Evaluation**: Uses actual bbox dimensions, not dummy values

✅ **Visibility Masking**: Can properly mask invisible keypoints in evaluation

✅ **Debugging**: Full metadata available for troubleshooting

✅ **Per-Category Analysis**: Can break down PCK by category with category_id

✅ **Complete Pipeline**: All necessary info flows from dataset → evaluation

## Verification

To verify this fix works:

1. **Check batch contents**: Print `batch.keys()` in training/eval loop
   - Should include `'query_metadata'`

2. **Check metadata structure**: Print `batch['query_metadata'][0]`
   - Should have `bbox_width`, `bbox_height`, `visibility`, etc.

3. **Check PCK computation**: Verify PCK uses actual bbox dimensions
   - Not hardcoded 512×512

4. **Run evaluation**: Ensure no errors in `evaluate_unseen_categories()`

## Summary

This was a **simple but critical fix**:
- **Before**: Data collected but lost in transit
- **After**: Data properly passed through pipeline
- **Impact**: Enables correct PCK evaluation with bbox normalization

The fix required only **3 lines of code** but unlocked proper evaluation functionality! 🎯

