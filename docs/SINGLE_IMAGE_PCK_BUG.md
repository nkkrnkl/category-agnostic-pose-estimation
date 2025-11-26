# Single-Image Overfitting PCK Bug

## Problem

When training on a single image and validating on the same image:
- **Validation reports PCK = 1.0 (100%)**
- **But visualization shows keypoints are NOT perfectly aligned** (mean distance 74.45px, max 180.14px)
- **Visualization PCK = 66.7%** (correct, after our fix)

## Root Cause

The validation code in `engine_cape.py` has a **fallback path** that uses **GT token_labels** to extract predictions instead of using the model's predicted token structure.

### The Bug

**Location:** `models/engine_cape.py`, lines 649-657 (fallback path)

**Problem:**
```python
if pred_logits is not None:
    # CORRECT: Use predicted token types
    pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits, ...)
else:
    # BUGGY FALLBACK: Use GT token_labels for predictions!
    pred_kpts = extract_keypoints_from_sequence(pred_coords, token_labels, mask)
```

**Why this is wrong:**
- Using GT `token_labels` assumes the model generates the **exact same token sequence** as GT
- This extracts predictions at GT token positions, not actual predicted positions
- In single-image overfitting, if the model generates a similar (but not identical) sequence, using GT structure can give artificially high PCK
- The coordinates might be slightly different, but if extracted using GT structure, they appear closer to GT

### Why PCK=1.0 but Visualization Shows Misalignment

1. **Validation (buggy):** Uses GT token_labels → extracts at GT positions → coordinates appear close → PCK=1.0
2. **Visualization (correct):** Uses predicted token structure → extracts at actual predicted positions → shows real differences → PCK=66.7%

## The Fix

I've updated the code to:
1. **Try to use predicted sequences** if logits are unavailable
2. **Add strong warnings** when fallback path is used
3. **Add debugging** to identify which path is being used

**Changes:**
- Check for `pred_sequences` (token IDs) as alternative to `pred_logits`
- Warn when using GT token_labels fallback (this gives wrong PCK!)
- Add `DEBUG_PCK_EXTRACTION=1` environment variable to diagnose which path is used

## How to Diagnose

Run validation with debugging enabled:
```bash
export DEBUG_PCK_EXTRACTION=1
export DEBUG_PCK=1
python train_cape_episodic.py ...
```

Look for:
- `"Using predicted logits: True"` → **CORRECT PATH** ✓
- `"Using predicted sequences: True"` → **ACCEPTABLE PATH** ✓
- `"Using GT token_labels (FALLBACK)"` → **BUGGY PATH** ✗

If you see the fallback path, check why `pred_logits` is None.

## Expected Behavior After Fix

- **Validation PCK** should match **visualization PCK** (both ~66.7% in your case)
- **Coordinate differences** should be consistent between validation and visualization
- **Warnings** will appear if fallback path is used

## Why This Matters

In single-image overfitting:
- Model should achieve PCK ≈ 100% if it truly memorized the image
- If validation shows PCK=1.0 but visualization shows misalignment, it indicates:
  1. **Bug in validation** (using wrong extraction method) ← **This is the issue**
  2. **Bug in visualization** (wrong coordinate denormalization) ← **Already fixed**
  3. **Model not actually overfitting** (but validation is wrong, so we can't tell)

## Next Steps

1. **Re-run validation** with debugging enabled to see which path is used
2. **Check if `pred_logits` is None** - if so, investigate why `forward_inference` isn't returning logits
3. **Compare validation PCK with visualization PCK** - they should match after fix

## Related Files

- `models/engine_cape.py` - Validation PCK computation (FIXED)
- `models/visualize_cape_predictions.py` - Visualization PCK computation (FIXED)
- `models/cape_model.py` - Model forward_inference (should return 'logits')

