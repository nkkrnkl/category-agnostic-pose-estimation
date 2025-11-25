# Quick Visualization Fix Verification

**TL;DR:** Fixed coordinate denormalization bug. Keypoints now appear in correct locations.

---

## ⚡ Quick Test

```bash
# Test with debug logging
DEBUG_VIS=1 python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --num-visualizations 5 \
    --device cpu
```

**Expected output:**
```
[DEBUG_VIS] Coordinate Denormalization Check:
  Image shape: torch.Size([3, 512, 512])
  Original bbox dims: 619.0 × 964.0
  GT keypoints (normalized): min=0.034, max=0.967
  GT keypoints (pixel): min=17.4, max=495.1
  ✓ GT keypoints within valid range     ← Should see this!
  Pred keypoints (pixel): min=45.2, max=478.3
  ✓ Predicted keypoints within valid range  ← And this!
```

---

## ✅ What Was Fixed

**Before:** Keypoints denormalized using original bbox dimensions (e.g., 619×964)  
→ Coords way outside 512×512 image bounds!

**After:** Keypoints denormalized using 512×512  
→ Coords within [0, 512] range ✓

---

## 🔍 Visual Verification

Generate visualizations:
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --num-visualizations 10 \
    --output-dir outputs/cape_eval_FIXED
```

**Check:**
- ✓ GT keypoints (cyan) align with query image features
- ✓ Support keypoints (green) align with support image
- ✓ Predicted keypoints (red) are on/near object
- ✓ NO keypoints outside image bounds

---

## 📊 The Bug

**Root cause:** Used wrong dimensions for denormalization

```python
# BEFORE (WRONG):
bbox_w = 619  # Original bbox width
pred_kpts_px = pred_kpts * bbox_w  # Coords way off!

# AFTER (CORRECT):
pred_kpts_px = pred_kpts * 512.0  # Coords in [0, 512] ✓
```

---

## 📚 Full Documentation

See: **`CRITICAL_VIS_COORD_BUG_FIXED.md`**

---

**Fixed:** Nov 25, 2025  
**Status:** ✅ Working correctly

