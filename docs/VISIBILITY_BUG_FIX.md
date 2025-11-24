# Visibility Array Length Mismatch Fix

## 🐛 **Bug Description**

During validation, the code crashed with:

```python
IndexError: boolean index did not match indexed array along axis 0; 
size of axis is 17 but size of corresponding boolean axis is 9
```

**Location**: `util/eval_utils.py:118` in `compute_pck_bbox()`

```python
pred_visible = pred_keypoints[visible_mask]  # ❌ Shape mismatch!
```

---

## 🔍 **Root Cause**

### **The Problem**

In `datasets/mp100_cape.py` line 395:

```python
# OLD (WRONG):
num_visible = int(np.sum(visibility > 0))  # Count visible keypoints (e.g., 9)
record["num_keypoints"] = num_visible      # ❌ Stores VISIBLE count, not TOTAL
```

This caused a semantic inconsistency:
- `record["keypoints"]` had **17 keypoints** (all keypoints, including invisible)
- `record["visibility"]` had **17 visibility flags** (one per keypoint)
- `record["num_keypoints"]` had value **9** (only visible count) ❌

### **How It Broke**

In `datasets/episodic_sampler.py` line 263:

```python
# Fallback when visibility is missing:
'visibility': query_data.get('visibility', [1] * query_data['num_keypoints'])
                                                   # ↑ Uses WRONG value (9)
```

When visibility wasn't passed through for some samples, the fallback created a **length-9 array** instead of **length-17**, causing the mismatch.

### **Evidence from Diagnostics**

```
Sample 1 (WORKS):
  pred_keypoints shape: (17, 2)
  visibility length: 17  ✅
  
Sample 2 (WORKS):
  pred_keypoints shape: (17, 2)
  visibility length: 17  ✅
  
Sample 3 (FAILS):
  pred_keypoints shape: (17, 2)
  visibility length: 9   ❌ FALLBACK USED WITH WRONG LENGTH!
```

---

## ✅ **The Fix**

### **1. Fixed `datasets/mp100_cape.py` (lines 386-401)**

```python
# NEW (CORRECT):
# Store TOTAL number of keypoints (not just visible!)
record["num_keypoints"] = len(kpts_array)  # Total keypoints (e.g., 17)
record["num_visible_keypoints"] = int(np.sum(visibility > 0))  # Visible count (e.g., 9)
```

**Changes:**
- `num_keypoints` now means **total keypoints** (visible + invisible)
- Added `num_visible_keypoints` for statistics/debugging

### **2. Updated `datasets/episodic_sampler.py` (lines 254-264)**

```python
query_metadata.append({
    ...
    'num_keypoints': query_data['num_keypoints'],  # Now TOTAL (17)
    'num_visible_keypoints': query_data.get('num_visible_keypoints', query_data['num_keypoints']),
    ...
    'visibility': query_data.get('visibility', [1] * query_data['num_keypoints'])  # Now correct length!
})
```

**Changes:**
- Added `num_visible_keypoints` to metadata
- Fallback visibility now uses correct length (17 instead of 9)

### **3. Removed Diagnostic Print Statements**

Cleaned up `util/eval_utils.py`:
- Removed verbose debugging output
- Kept core logic intact

---

## 📊 **Impact**

| Aspect | Before (Buggy) | After (Fixed) |
|--------|---------------|---------------|
| `num_keypoints` meaning | Visible count (9) ❌ | Total count (17) ✅ |
| `visibility` fallback length | 9 ❌ | 17 ✅ |
| PCK evaluation | Crashes on some samples ❌ | Works correctly ✅ |

---

## 🧪 **How to Verify**

Run training:

```bash
python train_cape_episodic.py \
  --epochs 1 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run \
  --dataset_root .
```

**Expected behavior:**
- Training completes epoch 1
- Validation runs without `IndexError`
- PCK metrics are computed successfully

---

## 📝 **Summary**

**Root cause**: `num_keypoints` stored visible count instead of total count, breaking fallback logic.

**Fix**: Renamed semantic meaning of `num_keypoints` to mean TOTAL keypoints, added separate `num_visible_keypoints` field.

**Files modified:**
1. `datasets/mp100_cape.py` (lines 386-401)
2. `datasets/episodic_sampler.py` (lines 254-264)
3. `util/eval_utils.py` (removed diagnostics)

**Result**: Visibility arrays now always have correct length matching the total number of keypoints.

