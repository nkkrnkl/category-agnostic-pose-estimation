# Fixes for Issues #10 and #11

## Issue #10: Support Mask Always All-True ✅ FIXED

### Problem

The support mask was being created as **all True** (all keypoints marked as valid), regardless of the actual **visibility** of keypoints in the support image.

**Old code (line 215 in episodic_sampler.py):**
```python
# Create support mask (all valid for now)
support_mask = torch.ones(len(support_coords), dtype=torch.bool)
```

This ignored keypoint visibility, treating all keypoints as valid even if they were:
- Not labeled (visibility = 0)
- Outside the image
- Not annotated

### Why This Is a Problem

**Support masks serve two purposes:**

1. **Attention Masking**: Tell the model which keypoints to attend to
   - If mask[i] = True: "This keypoint is valid, use it"
   - If mask[i] = False: "This keypoint is invalid/padding, ignore it"

2. **Padding Handling**: After batching, support keypoints are padded to `max_support_kpts`
   - Real keypoints should have mask = True
   - Padding positions should have mask = False

**Without visibility-based masking:**
- The model attends to invalid/unlabeled keypoints
- Non-visible keypoints contaminate the support representation
- Leads to noisy support embeddings

### Solution

Modified `datasets/episodic_sampler.py` (lines 210-238) to use visibility information:

```python
# ================================================================
# CRITICAL FIX: Create support mask based on visibility
# ================================================================
# Previously: support_mask was all True (ignoring visibility)
# Now: Use visibility information to mark only visible keypoints as valid
#
# Visibility values (COCO format):
#   0 = not labeled (keypoint outside image or not annotated)
#   1 = labeled but not visible (occluded)
#   2 = labeled and visible
#
# For support mask:
#   - True (valid) if visibility > 0 (labeled, may be occluded)
#   - False (invalid) if visibility == 0 (not labeled)
#
# Note: mp100_cape.py already filters to only visible keypoints
# (visibility > 0), so in practice all should be True. However,
# we still create the mask properly for correctness.
# ================================================================

support_visibility = support_data.get('visibility', [2] * len(support_coords))
support_mask = torch.tensor(
    [v > 0 for v in support_visibility], 
    dtype=torch.bool
)
```

### How It Works

#### Step 1: Extract visibility from support_data
```python
support_visibility = support_data.get('visibility', [2] * len(support_coords))
```
- Gets the visibility flags for each keypoint
- Fallback to all visible (2) if not available

#### Step 2: Create mask based on visibility
```python
support_mask = torch.tensor(
    [v > 0 for v in support_visibility], 
    dtype=torch.bool
)
```
- Creates a boolean tensor
- `True` if visibility > 0 (keypoint is labeled)
- `False` if visibility == 0 (keypoint not labeled)

#### Step 3: Padding preserves mask structure

The collate function already handles padding correctly (lines 318-335):

```python
for coords, mask in zip(support_coords_list, support_masks):
    num_kpts = coords.shape[0]
    if num_kpts < max_support_kpts:
        # Pad coordinates with zeros
        padding = max_support_kpts - num_kpts
        coords = torch.cat([coords, torch.zeros(padding, 2)], dim=0)
        
        # Pad mask with False (mark padding as invalid)
        mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)], dim=0)
```

### Example

**Scenario**: Category has 17 keypoints, but only 15 are visible/labeled

**Before (all True):**
```python
support_mask = [True, True, True, True, True, True, True, True, True, 
                True, True, True, True, True, True, True, True]
# All 17 keypoints marked valid, even if some have visibility=0!
```

**After (visibility-based):**
```python
support_visibility = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]
support_mask = [True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, False, False]
# Only 15 visible keypoints marked valid!
```

**After padding to max_support_kpts=20:**
```python
support_mask = [True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, False, False,
                False, False, False]
# Last 5 are padding (added by collate function)
```

### Impact on Model

✅ **Cleaner Support Embeddings**: Model only attends to valid keypoints

✅ **Better Attention**: Support encoder ignores invalid positions

✅ **Correct Padding**: Both visibility and padding handled properly

✅ **Consistent with Query**: Matches how query visibility is handled

### Note: MP-100 Specific Behavior

In `mp100_cape.py` (lines 268-276), the dataset **already filters** to only visible keypoints:

```python
visible_mask = visibility > 0
visible_kpts = kpts[visible_mask][:, :2]  # only take x, y

if len(visible_kpts) > 0 and 'bbox' in ann:
    keypoints_list.append(kpts[:, :2].tolist())  # ALL keypoints
    visibility_list.append(visibility.tolist())
```

So in practice, **all keypoints in support_coords already have visibility > 0**, making the support_mask all True initially. However:

1. **Correctness**: Properly using visibility is the right approach
2. **Future-proofing**: If dataset behavior changes, code still works
3. **Padding**: The mask is essential for marking padding positions
4. **Consistency**: Matches how other parts of the codebase handle visibility

---

## Issue #11: Category ID Stored Per Episode Not Per Query ✅ ALREADY FIXED

### Problem

Category IDs need to be **stored per query** (B*K entries), not per episode (B entries).

**Why this matters:**
- After episodic collation, all tensors have batch size (B*K)
  - B = number of episodes
  - K = queries per episode
- Category IDs must match this dimension for correct indexing

**Example scenario:**
```
Episode 1: category "cat" (ID=17), 3 queries
Episode 2: category "dog" (ID=18), 3 queries

Batch after collation:
  - query_images: (6, C, H, W)  ← 6 queries total
  - support_coords: (6, N, 2)   ← 6 support instances (repeated)
  - category_ids: ???           ← Should be length 6!
```

### Solution (Already Implemented)

This was **already fixed** in Issue #1 (Support-Query Dimension Mismatch) on line 375 of `episodic_sampler.py`:

**File: `datasets/episodic_sampler.py` (lines 361-377)**

```python
# ========================================================================
# ALREADY FIXED (Issue #11): Category IDs repeated per query
# ========================================================================
# Category IDs must match the (B*K) batch dimension, not just (B).
# Each category ID is repeated K times (once per query in that episode).
#
# Example with B=2 episodes, K=3 queries per episode:
#   Before: category_ids = [cat_A, cat_B]  (length 2)
#   After:  category_ids = [cat_A, cat_A, cat_A, cat_B, cat_B, cat_B]  (length 6)
#
# This ensures category_ids[i] corresponds to query[i] in the batch.
# ========================================================================

# Repeat category_ids to match query dimension
category_ids_tensor = torch.tensor(category_ids, dtype=torch.long)  # (B,)
category_ids_tensor = category_ids_tensor.repeat_interleave(queries_per_episode)  # (B*K,)
```

### How It Works

#### Step 1: Collect category IDs (one per episode)
```python
# In the loop (line 296-297)
category_ids.append(episode['category_id'])
# category_ids is a list of length B (num episodes)
```

#### Step 2: Convert to tensor
```python
category_ids_tensor = torch.tensor(category_ids, dtype=torch.long)  # (B,)
# Shape: (B,) where B = num_episodes
```

#### Step 3: Repeat each category K times
```python
category_ids_tensor = category_ids_tensor.repeat_interleave(queries_per_episode)  # (B*K,)
# repeat_interleave repeats each element K times
# [A, B] with K=3 → [A, A, A, B, B, B]
```

### Example

**Setup:**
- Batch size: B = 2 episodes
- Queries per episode: K = 3
- Categories: Episode 1 = "cat" (17), Episode 2 = "dog" (18)

**Step-by-step transformation:**

```python
# After collecting from episodes
category_ids = [17, 18]  # List, length B=2

# Convert to tensor
category_ids_tensor = torch.tensor([17, 18])  # Shape: (2,)

# Repeat each category K=3 times
category_ids_tensor = category_ids_tensor.repeat_interleave(3)
# Shape: (6,)
# Values: [17, 17, 17, 18, 18, 18]
```

**Final batch alignment:**
```python
Query 0: category_id = 17  (from Episode 1, cat)
Query 1: category_id = 17  (from Episode 1, cat)
Query 2: category_id = 17  (from Episode 1, cat)
Query 3: category_id = 18  (from Episode 2, dog)
Query 4: category_id = 18  (from Episode 2, dog)
Query 5: category_id = 18  (from Episode 2, dog)
```

### Why This Is Correct

1. **Dimension Matching**: category_ids.shape[0] == query_images.shape[0] == (B*K)
2. **Proper Indexing**: category_ids[i] is the category for query[i]
3. **Consistent with Other Tensors**: Matches how support_coords, support_masks are repeated
4. **PCK Evaluation**: Allows per-category PCK aggregation during evaluation

### Verification

You can verify this is working by checking tensor shapes in a batch:

```python
# In engine_cape.py or train loop
batch = next(iter(dataloader))

print(f"Query images shape: {batch['query_images'].shape}")  # (B*K, C, H, W)
print(f"Support coords shape: {batch['support_coords'].shape}")  # (B*K, N, 2)
print(f"Category IDs shape: {batch['category_ids'].shape}")  # (B*K,)

# Verify they all have the same batch size
assert batch['query_images'].shape[0] == batch['category_ids'].shape[0]
```

---

## Summary

### Issue #10: Support Mask Always All-True
- **Status**: ✅ **FIXED**
- **Change**: Use visibility information to create support_mask
- **Impact**: Cleaner support embeddings, proper attention masking
- **Files Modified**: `datasets/episodic_sampler.py` (lines 210-238)

### Issue #11: Category ID Stored Per Episode Not Per Query
- **Status**: ✅ **ALREADY FIXED** (in Issue #1)
- **Location**: `datasets/episodic_sampler.py` (line 375)
- **Implementation**: `repeat_interleave` to expand (B,) → (B*K,)
- **Added**: Clarifying comments to document the fix

### Combined Impact

🎯 **Support Mask**: Now properly reflects keypoint visibility

🎯 **Category IDs**: Correctly aligned with (B*K) batch dimension

🎯 **Attention**: Model attends only to valid keypoints

🎯 **Evaluation**: Per-category PCK aggregation works correctly

🎯 **Consistency**: All batch tensors have matching dimensions

Both issues are now **fully resolved**!

