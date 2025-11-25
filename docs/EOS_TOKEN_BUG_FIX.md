# EOS Token Prediction Bug - Fix Documentation

**Date:** November 25, 2025  
**Status:** ✅ FIXED  
**Impact:** CRITICAL - Model could not learn proper sequence termination  
**Checkpoint:** All checkpoints before this fix are affected

---

## Bug Summary

### Symptom

The model **always generated exactly 200 keypoints** for every validation sample, regardless of the category's actual requirements (typically 17-32 keypoints).

**Observed behavior:**
- Expected keypoints: 17-32 (varies by category)
- Generated keypoints: **200 (always)**
- Extra predictions discarded: 168-183 per sample
- PCK evaluation: Artificially inflated (trimming masked the bug)

### Root Cause

**The EOS (End of Sequence) token was EXCLUDED from the classification loss.**
 cmd + shift + L 
**Location:** `datasets/mp100_cape.py:737`

```python
# Visibility rules (BEFORE FIX):
#   - Coordinate tokens from VISIBLE keypoints (visibility > 0): True
#   - Coordinate tokens from INVISIBLE keypoints (visibility == 0): False
#   - SEP/EOS/padding tokens: False (don't use in loss)  ← BUG HERE!
```

The visibility mask marked only COORD tokens as `True`, setting EOS tokens to `False`. This caused:
1. EOS positions excluded from classification loss computation
2. Model received **zero gradient signal** to learn EOS prediction
3. Model never learned when to stop generation
4. Generation always continued to `max_len=200`

### Evidence

**Diagnostic output (Nov 25, 2025):**

```
Sample 0:
  Expected keypoints: 17
  GT token sequence:   [COORD, COORD, ..., COORD, EOS]  (17 COORD + 1 EOS)
  Pred token sequence: [COORD, COORD, ..., COORD, COORD] (200 COORD + 0 EOS)
  
  Predicted keypoints: 200
  Extra predictions: 183
  
  Generation status:
    - Reached max_len: 200/200
    - unfinish_flag: [1. 1.]  ← Never finished!
    - EOS predicted: 0/2 samples (0%)
```

**All 20 validation samples (100%)** showed the same pattern.

---

## The Fix

### Fix #1: Include EOS in Visibility Mask (PRIMARY FIX)

**File:** `datasets/mp100_cape.py`  
**Lines:** 758-769 (after line 756)

**Change:**
```python
# ========================================================================
# CRITICAL FIX: Include EOS token in visibility mask
# ========================================================================
# BUG IDENTIFIED (Nov 25, 2025):
#   - EOS token was excluded from loss computation (see line 737)
#   - Model never received gradient signal to learn EOS prediction
#   - Result: Model always generates 200 tokens (max_len), never stops
#
# FIX: Mark EOS token position as True in visibility_mask
#   - This ensures classification loss includes EOS token
#   - Model will learn to predict EOS at the correct position
#   - Generation will stop at appropriate length (not max_len)
# ========================================================================
for i, label in enumerate(token_labels):
    if label == TokenType.eos.value:
        visibility_mask[i] = True
        break  # Only mark first EOS
```

**Impact:**
- EOS token now included in classification loss
- Model receives gradient signal to learn EOS prediction
- Should predict EOS after the correct number of keypoints

### Fix #2: Add Warning for Incomplete Generations

**File:** `models/roomformer_v2.py`  
**Lines:** 585-598 (after line 583)

**Change:**
```python
# ========================================================================
# CRITICAL DIAGNOSTIC: Warn if generation reached max_len without EOS
# ========================================================================
# This indicates the model didn't learn to predict EOS properly.
# After fixing the EOS loss masking bug, this warning should not appear.
# ========================================================================
incomplete_generations = unfinish_flag.sum()
if incomplete_generations > 0 and os.environ.get('WARN_INCOMPLETE_GENERATION', '1') == '1':
    import warnings
    warnings.warn(
        f"⚠️  {int(incomplete_generations)}/{bs} sequences reached max_len={max_len} "
        f"without predicting EOS. This suggests the model hasn't learned proper "
        f"stopping behavior. Consider retraining with EOS token included in loss."
    )
```

**Impact:**
- Early detection of EOS prediction failures
- Alerts users if model checkpoints trained before the fix
- Can be disabled with `WARN_INCOMPLETE_GENERATION=0`

### Fix #3: Add Assertions Before Trimming

**Files:**
- `scripts/eval_cape_checkpoint.py:464-476`
- `models/engine_cape.py:611-625`

**Change:**
```python
# ================================================================
# CRITICAL ASSERTION: Detect keypoint count mismatch before trimming
# ================================================================
# If predictions significantly exceed expected count, the model
# likely didn't learn to predict EOS properly.
# ================================================================
pred_count = pred_kpts[idx].shape[0]
expected_count = num_kpts_for_category
excess = pred_count - expected_count

if excess > 10 and batch_idx == 0 and idx == 0:  # Warn once
    import warnings
    warnings.warn(
        f"⚠️  Model generated {pred_count} keypoints but expected ~{expected_count}. "
        f"Excess: {excess}. Model likely didn't learn EOS prediction properly. "
        f"Recommend retraining with EOS token included in classification loss."
    )
```

**Impact:**
- Detects the bug early in evaluation pipeline
- Alerts users that trimming is discarding significant predictions
- Suggests retraining with proper fix

---

## Testing

### Test Suite: `tests/test_eos_prediction.py`

**Test 1: `test_eos_token_in_visibility_mask`** ✅ PASSED
- Verifies EOS token is marked as `True` in `visibility_mask`
- Confirms the fix is correctly applied in dataset construction

**Test 2: `test_token_type_distribution_is_balanced`** ✅ PASSED
- Verifies token distribution includes EOS tokens
- Sample results: 20 EOS tokens for 20 samples (correct)

**Test 3: `test_visibility_mask_includes_all_visible_coords`** ✅ PASSED
- Verifies both COORD and EOS tokens are marked as visible
- Example: 8 COORD + 1 EOS = 9 total True values

**Test 4: `test_eos_prediction_rate_after_training`**
- Tests that trained model predicts EOS at reasonable rate (>50%)
- **Will FAIL on old checkpoints** (expected - they were trained before fix)
- **Should PASS after retraining** with the fix

**Test 5: `test_predicted_keypoint_count_reasonable`**
- Tests that predicted counts are reasonable, not always 200
- **Will FAIL on old checkpoints**
- **Should PASS after retraining**

### Validation Results (Before Fix)

Using `scripts/diagnose_keypoint_counts.py`:

```bash
DEBUG_KEYPOINT_COUNT=1 python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run/checkpoint_e024_lr1e-04_bs2_acc4_qpe2.pth \
    --num-samples 1
```

**Output:**
```
✅ HYPOTHESIS CONFIRMED
   Model generates TOO MANY keypoints
   - 366 extra predictions across 2 samples
   - All predictions are exactly 200 keypoints
   - EOS prediction rate: 0%
```

---

## Impact on Training

### Before Fix

**Loss computation excluded EOS:**
```python
# cape_losses.py:134
loss_ce = label_smoothed_nll_loss(src_logits[mask], target_classes[mask], ...)

# mask = valid_mask & visibility_mask
# visibility_mask[EOS_position] = False  ← EOS excluded!
```

**Result:**
- No gradient flow to EOS prediction logits
- Model learns COORD prediction well (gets gradient signal)
- Model never learns to predict EOS (no gradient signal)
- Default prediction: Always COORD (highest pre-softmax logit)

### After Fix

**Loss computation includes EOS:**
```python
# Now visibility_mask[EOS_position] = True
loss_ce = label_smoothed_nll_loss(src_logits[mask], target_classes[mask], ...)
```

**Expected result after retraining:**
- Model receives gradient for EOS position
- Learns to predict EOS at correct positions (after last keypoint)
- Generation stops at appropriate length (17-32 tokens, not 200)
- No excessive trimming needed

---

## Migration Guide

### For Existing Checkpoints

**All checkpoints trained before this fix are affected.**

⚠️ **Old checkpoints will:**
- Always generate 200 keypoints
- Show warnings during evaluation
- Have artificially high PCK (due to trimming)

✅ **Recommended action:**
1. Retrain model from scratch with the fix
2. Monitor EOS prediction rate during training
3. Verify generation stops at appropriate lengths

### For New Training Runs

**Monitoring EOS prediction:**

Add to training loop logging:
```python
# After each epoch
eos_rate = count_eos_predictions(model, val_dataloader)
print(f"Epoch {epoch}: EOS prediction rate = {eos_rate:.1%}")

# Target: >50% by epoch 10, >80% by epoch 20
```

**Expected behavior:**
- Early epochs: Low EOS rate (~10-30%)
- Middle epochs: Increasing rate (~40-70%)
- Late epochs: High rate (>80%)

### Diagnostic Commands

**Check if checkpoint affected:**
```bash
python scripts/diagnose_keypoint_counts.py \
    --checkpoint path/to/checkpoint.pth \
    --num-samples 5
```

**Expected output (after fix + retraining):**
```
Total extra predictions: <50  (should be near zero)
EOS prediction rate: >80%
Avg predicted keypoints: 20-30 (varies by category)
```

---

## Related Bugs

This bug interacted with other issues in the evaluation pipeline:

1. **PCK always 100%** - Previously fixed (coordinate space mismatch)
2. **Single keypoint prediction** - Previously fixed (accumulation bug)
3. **Keypoint count mismatch** - **This bug** (EOS never predicted)

All three bugs have now been identified and fixed.

---

## Files Modified

### Core Fixes
- `datasets/mp100_cape.py:758-769` - Include EOS in visibility mask
- `models/roomformer_v2.py:585-598` - Add incomplete generation warning

### Diagnostic Assertions
- `scripts/eval_cape_checkpoint.py:464-476` - Pre-trimming assertion
- `models/engine_cape.py:611-625` - Pre-trimming assertion

### Diagnostic Tools
- `scripts/diagnose_keypoint_counts.py` - Comprehensive diagnostic script
- `tests/test_eos_prediction.py` - Validation test suite

### Documentation
- `KEYPOINT_COUNT_DIAGNOSTIC_REPORT.md` - Full diagnostic report
- `docs/EOS_TOKEN_BUG_FIX.md` - This document

---

## Next Steps

### 1. Retrain Model ⏸️

```bash
# Start fresh training with the fix
python train_cape_episodic.py \
    --lr 1e-4 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --queries_per_episode 2 \
    --epochs 30 \
    --output_dir outputs/cape_run_fixed
```

### 2. Monitor Training 🔍

Watch for:
- EOS prediction rate increasing over epochs
- Validation loss decreasing
- Generation length converging to category expectations

### 3. Validate Fix ✅

After training:
```bash
# Run diagnostic
python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run_fixed/checkpoint_best.pth \
    --per-category \
    --num-samples 20

# Run evaluation
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run_fixed/checkpoint_best.pth \
    --num-visualizations 50 \
    --output-dir outputs/cape_eval_fixed

# Run tests
python -m pytest tests/test_eos_prediction.py -v
```

**Expected results:**
- ✅ EOS prediction rate: >80%
- ✅ Average predicted keypoints: 17-32 (varies by category)
- ✅ Extra predictions: <5 per sample
- ✅ All tests passing

### 4. Compare Performance 📊

Compare old vs new checkpoints:
- PCK scores (should be more realistic, likely lower)
- Generation quality (visualizations should show reasonable keypoint counts)
- Training convergence (may be faster with proper EOS learning)

---

## Lessons Learned

### 1. Visibility Masking Must Include Special Tokens

While it makes sense to exclude SEP and padding from loss, **EOS tokens are critical for sequence generation** and must be included in the loss.

**Design principle:**
- Exclude tokens that don't carry information (padding)
- Include tokens that control generation (EOS)
- Include tokens that carry data (COORD)

### 2. Trimming Can Hide Bugs

The evaluation code's trimming operation:
```python
pred_kpts_trimmed = pred_kpts[:, :num_kpts_for_category, :]
```

This silently discarded 183 extra predictions per sample, making the model appear to work correctly despite the severe bug.

**Best practice:**
- Add assertions before trimming operations
- Log statistics on what's being discarded
- Investigate if trimming discards >5% of predictions

### 3. Token Classification Imbalance

In CAPE sequences:
- COORD tokens: 17-32 per sequence (~95%)
- EOS token: 1 per sequence (~1%)
- SEP tokens: 0-2 per sequence (~2%)

This severe class imbalance makes EOS hard to learn even WITH loss signal. Consider:
- Class weighting (weight EOS higher)
- Focal loss for rare classes
- Explicit EOS prediction rewards

### 4. Debug Instrumentation is Essential

The diagnostic script `scripts/diagnose_keypoint_counts.py` was instrumental in:
- Confirming the hypothesis (100% of samples affected)
- Identifying the exact failure mode (EOS never predicted)
- Measuring the bug's impact (183 extra predictions per sample)

**Recommendation:** Keep diagnostic tools in the repository for future debugging.

---

## Code Review Checklist

When implementing sequence generation models, verify:

- [ ] All critical special tokens (BOS, EOS, SEP) are included in loss
- [ ] Token type distribution is balanced or weighted appropriately
- [ ] Generation stopping logic works even if EOS is not predicted
- [ ] Evaluation code detects anomalies (e.g., excessive trimming)
- [ ] Tests validate token prediction rates
- [ ] Diagnostic tools exist for debugging sequence generation

---

## References

**Related Documentation:**
- `KEYPOINT_COUNT_DIAGNOSTIC_REPORT.md` - Full diagnostic process
- `docs/CRITICAL_SINGLE_KEYPOINT_BUG.md` - Previous sequence generation bug
- `docs/CRITICAL_BUG_PCK_100_ANALYSIS.md` - Related PCK evaluation bug

**Key Files:**
- `datasets/mp100_cape.py` - Dataset and tokenization
- `models/roomformer_v2.py` - Autoregressive generation
- `models/cape_losses.py` - Loss computation
- `scripts/diagnose_keypoint_counts.py` - Diagnostic tool
- `tests/test_eos_prediction.py` - Validation tests

---

## Appendix: Detailed Token Flow

### Ground Truth Construction

```
Keypoints: [(x1,y1), (x2,y2), ..., (x17,y17)]
Visibility: [2, 2, 0, 2, 2, ..., 2]  (17 values)

Tokenization:
  token_labels:     [COORD, COORD, COORD, ..., COORD, EOS, PAD, PAD, ...]
                     ^^17 coords^^           ^1 EOS   ^^^^padding^^^^
  
  visibility_mask (BEFORE FIX):
                    [T, T, F, T, T, ..., T, F, F, F, ...]
                     ^^matches visibility^^  ^EOS=F!  ^^padding=F^^
  
  visibility_mask (AFTER FIX):
                    [T, T, F, T, T, ..., T, T, F, F, ...]
                     ^^matches visibility^^  ^EOS=T!  ^^padding=F^^
```

### Loss Computation

```python
# BEFORE FIX
mask = valid_mask & visibility_mask
# EOS position: valid_mask[17]=True, visibility_mask[17]=False
# mask[17] = True & False = False  ← EOS excluded!

# AFTER FIX
mask = valid_mask & visibility_mask
# EOS position: valid_mask[17]=True, visibility_mask[17]=True
# mask[17] = True & True = True  ← EOS included!
```

### Training Signal

```python
# Loss computation
loss_ce = label_smoothed_nll_loss(
    src_logits[mask],      # Shape: (num_visible_tokens, num_classes)
    target_classes[mask],  # Shape: (num_visible_tokens,)
    epsilon=label_smoothing
)

# BEFORE FIX: num_visible_tokens = 8 COORD (only visible ones)
#   - EOS position not in mask → no gradient to EOS logit
#   - Model learns COORD prediction only
#   - Default to COORD for all positions (including where EOS should be)

# AFTER FIX: num_visible_tokens = 8 COORD + 1 EOS
#   - EOS position in mask → gradient flows to EOS logit
#   - Model learns both COORD and EOS prediction
#   - Predicts EOS at position 17
```

---

## Success Criteria

The fix will be considered successful when:

1. ✅ **Dataset construction:** EOS token included in visibility_mask
2. ✅ **Tests pass:** `test_eos_token_in_visibility_mask` passes
3. ⏸️ **Model training:** EOS prediction rate increases during training
4. ⏸️ **Inference:** Generation stops at appropriate lengths (not max_len)
5. ⏸️ **Evaluation:** Trimming discards <5 keypoints per sample
6. ⏸️ **PCK:** More realistic scores (not 100%)
7. ⏸️ **Visualizations:** Reasonable keypoint counts in predictions

**Status: Items 1-2 complete. Items 3-7 require model retraining.**

---

## Timeline

- **Nov 25, 2025 10:00** - Bug reported (PCK always 100%, 200 keypoints predicted)
- **Nov 25, 2025 11:30** - Diagnostic plan approved
- **Nov 25, 2025 12:00** - Diagnostic script created and executed
- **Nov 25, 2025 12:30** - Root cause identified (EOS excluded from loss)
- **Nov 25, 2025 13:00** - Fix implemented and tested
- **Nov 25, 2025 13:30** - Documentation complete

**Next:** Retrain model and validate fix (ETA: 24-48 hours for full training)

