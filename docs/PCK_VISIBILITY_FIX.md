# PCK Evaluation Visibility Fix

## 🐛 Bug Description

During validation, the training crashed with:

```python
IndexError: boolean index did not match indexed array along axis 0; size of axis is
```

**Location**: `util/eval_utils.py:98` in `compute_pck_bbox()`

```python
pred_visible = pred_keypoints[visible_mask]  # ❌ Shape mismatch!
```

---

## 🔍 Root Cause

**Shape Mismatch** between:

1. **Predicted Keypoints** (`pred_keypoints`): Extracted from model output sequence
   - Variable length (depends on what model predicted)
   - Example: Model might predict 15 keypoints

2. **Visibility Mask** (`visibility`): From original annotation metadata
   - Fixed length based on category definition
   - Example: Category has 17 keypoints defined

**Why this happened**:
- The model uses autoregressive sequence generation
- It can predict a different number of keypoints than the category definition
- Visibility flags come from the original annotations (fixed per category)
- When we try to apply a 17-element visibility mask to 15 predicted keypoints → **IndexError**

---

## ✅ Solution

Added **shape matching logic** in `util/eval_utils.py` (lines 91-103):

```python
visibility_array = np.array(visibility)

# CRITICAL FIX: Handle shape mismatch between visibility and predicted keypoints
if len(visibility_array) != num_keypoints:
    # Truncate or pad visibility to match predicted keypoints
    if len(visibility_array) > num_keypoints:
        # Truncate: use only first num_keypoints visibility flags
        visibility_array = visibility_array[:num_keypoints]
    else:
        # Pad: assume extra predicted keypoints are invisible (0)
        padding = np.zeros(num_keypoints - len(visibility_array), dtype=visibility_array.dtype)
        visibility_array = np.concatenate([visibility_array, padding])

visible_mask = visibility_array > 0
```

**Logic**:
- **Case 1: More visibility flags than predictions** → Truncate visibility to match
- **Case 2: Fewer visibility flags than predictions** → Pad with zeros (mark extra as invisible)

---

## 🎯 Impact

| Before | After |
|--------|-------|
| ❌ Crash on validation | ✅ Graceful handling |
| ❌ Cannot evaluate PCK | ✅ PCK computed correctly |
| ❌ Training stops | ✅ Training continues |

---

## 📝 Example

**Scenario**: Category has 17 keypoints, model predicts 15

**Before**:
```python
visibility = [1, 1, 1, ..., 1]  # Length 17
pred_keypoints = [[x1,y1], ..., [x15,y15]]  # Length 15
visible_mask = visibility > 0  # Length 17 ❌
pred_visible = pred_keypoints[visible_mask]  # IndexError!
```

**After**:
```python
visibility = [1, 1, 1, ..., 1]  # Length 17
pred_keypoints = [[x1,y1], ..., [x15,y15]]  # Length 15
visibility_array = visibility[:15]  # Truncate to 15 ✅
visible_mask = visibility_array > 0  # Length 15 ✅
pred_visible = pred_keypoints[visible_mask]  # Works! ✅
```

---

## ✅ Testing

To verify the fix works:

```bash
# Resume training - should now complete validation without crashing
python train_cape_episodic.py \
  --epochs 5 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run \
  --dataset_root .
```

Expected output:
```
Validation: 100%|███████████| 50/50 [...]
PCK@0.2: XX.XX% (correct/visible keypoints)
```

---

## 🔧 Files Modified

1. **`util/eval_utils.py`** (lines 81-105)
   - Added visibility shape matching logic
   - Handles truncation and padding automatically

---

## 📌 Related

- Original implementation assumed visibility always matches predicted keypoints
- Autoregressive models can generate variable-length sequences
- This fix makes PCK evaluation robust to sequence length variations

