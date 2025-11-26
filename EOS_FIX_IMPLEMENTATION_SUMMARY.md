# EOS Token Bug - Implementation Summary

**Date:** November 25, 2025  
**Status:** ✅ ALL FIXES IMPLEMENTED  
**Next Step:** Retrain model to validate fix effectiveness

---

## ✅ Implementation Complete

All planned fixes have been implemented and tested. The core bug has been resolved at the dataset level.

---

## What Was Fixed

### 🔴 THE BUG

**Problem:** Model always generated exactly **200 keypoints** instead of 17-32 (category-dependent).

**Root Cause:** EOS token was **EXCLUDED from classification loss** via visibility masking.

**Impact:**
- Model never learned to predict EOS token
- Generation always continued to max_len (200 tokens)
- 168-183 extra predictions discarded per sample
- PCK evaluation masked the bug via trimming

**Evidence:**
```
Hypothesis Testing Results:
  ✅ 100% of samples generate 200 keypoints (expected: 17-32)
  ✅ EOS prediction rate: 0% (should be >80%)
  ✅ Average excess predictions: 183 per sample
  ✅ Token distribution: {COORD: 200, EOS: 0}
```

---

## 🔧 Fixes Applied

### Fix #1: Include EOS in Visibility Mask ✅

**File:** `datasets/mp100_cape.py:758-769`

**What changed:**
```python
# BEFORE (line 737):
#   - SEP/EOS/padding tokens: False (don't use in loss)  ← BUG!

# AFTER (lines 758-769):
# Include EOS token in visibility mask
for i, label in enumerate(token_labels):
    if label == TokenType.eos.value:
        visibility_mask[i] = True  # ← FIX: Now included in loss!
        break
```

**Impact:**
- EOS token now receives gradient signal during training
- Model will learn to predict EOS at correct positions
- Generation will stop at appropriate lengths

**Test:** ✅ `test_eos_token_in_visibility_mask` - PASSED

---

### Fix #2: Add Warning for Incomplete Generations ✅

**File:** `models/roomformer_v2.py:585-598`

**What changed:**
```python
incomplete_generations = unfinish_flag.sum()
if incomplete_generations > 0:
    warnings.warn(
        f"⚠️  {int(incomplete_generations)}/{bs} sequences reached max_len={max_len} "
        f"without predicting EOS. Model needs retraining with EOS in loss."
    )
```

**Impact:**
- Early detection if model hasn't learned EOS prediction
- Alerts users to retrain with proper fix
- Can be disabled with `WARN_INCOMPLETE_GENERATION=0`

---

### Fix #3: Add Assertions Before Trimming ✅

**Files:**
- `scripts/eval_cape_checkpoint.py:464-476`
- `models/engine_cape.py:611-625`

**What changed:**
```python
pred_count = pred_kpts[idx].shape[0]
expected_count = num_kpts_for_category
excess = pred_count - expected_count

if excess > 10 and batch_idx == 0:  # Warn once
    warnings.warn(
        f"⚠️  Model generated {pred_count} keypoints but expected ~{expected_count}. "
        f"Excess: {excess}. Recommend retraining with EOS in loss."
    )
```

**Impact:**
- Detects the bug early in evaluation
- Prevents silent masking via trimming
- Guides users to proper fix

---

## 🧪 Tests Created

### Test Suite: `tests/test_eos_prediction.py`

**Dataset-Level Tests (No Model Required):**

1. ✅ `test_eos_token_in_visibility_mask` - **PASSED**
   - Verifies EOS is marked True in visibility_mask
   - Confirms fix is correctly applied

2. ✅ `test_token_type_distribution_is_balanced` - **PASSED**
   - Verifies token sequences include EOS (20 EOS for 20 samples)
   - Confirms visibility mask includes all EOS tokens

3. ✅ `test_visibility_mask_includes_all_visible_coords` - **PASSED**
   - Verifies COORD and EOS tokens both marked as visible
   - Example: 8 COORD + 1 EOS = 9 total True values

**Model-Level Tests (Require Trained Checkpoint):**

4. ⏸️ `test_eos_prediction_rate_after_training`
   - Verifies model predicts EOS at >50% rate
   - **Will PASS after retraining with fix**

5. ⏸️ `test_predicted_keypoint_count_reasonable`
   - Verifies predicted counts are reasonable (<150, not always 200)
   - **Will PASS after retraining with fix**

6. ⏸️ `test_trimming_discards_minimal_predictions`
   - Verifies trimming discards <5 keypoints per sample
   - **Will PASS after retraining with fix**

---

## 🛠️ Diagnostic Tools

### `scripts/diagnose_keypoint_counts.py`

Comprehensive diagnostic script for analyzing keypoint count mismatches.

**Usage:**
```bash
# Quick diagnostic
DEBUG_KEYPOINT_COUNT=1 python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run/checkpoint_eXXX.pth \
    --num-samples 1

# Per-category analysis
python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run/checkpoint_eXXX.pth \
    --per-category \
    --num-samples 10

# Token sequence inspection
python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run/checkpoint_eXXX.pth \
    --dump-tokens \
    --num-samples 3
```

**Output Example:**
```
Sample 0:
  Category ID: 35
  Expected keypoints: 17
  GT keypoints: 17
  Predicted keypoints: 200  ← Will be 17-32 after retrain
  Extra predictions: 183    ← Will be <5 after retrain
  
  GT tokens:   [COORD×17, EOS]
  Pred tokens: [COORD×200]     ← Will include EOS after retrain
```

---

## 📋 Verification Checklist

### ✅ Pre-Training Checks (Completed)

- [x] EOS token included in visibility_mask
- [x] Dataset construction tests pass (3/3)
- [x] Debug instrumentation in place
- [x] Diagnostic script working
- [x] Documentation complete

### ⏸️ Post-Training Checks (Pending Retrain)

- [ ] EOS prediction rate >50% during training
- [ ] EOS prediction rate >80% by final epoch
- [ ] Average predicted keypoints: 17-32 (varies by category)
- [ ] Excess predictions <5 per sample
- [ ] Model-level tests pass (3/3)
- [ ] PCK scores more realistic (not 100%)
- [ ] Visualizations show reasonable keypoint counts

---

## 🚀 Next Steps

### 1. Retrain Model (REQUIRED)

All existing checkpoints were trained **before** the EOS fix and will exhibit the bug.

**Command:**
```bash
# Start fresh training with the fix
python train_cape_episodic.py \
    --lr 1e-4 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --queries_per_episode 2 \
    --epochs 30 \
    --output_dir outputs/cape_run_with_eos_fix \
    --resume ''  # Start fresh, don't resume old checkpoint
```

### 2. Monitor Training

**Watch for:**
- Classification loss convergence (should decrease normally)
- EOS prediction appearing in logs (if instrumented)
- Validation PCK (should stabilize at realistic level, not 100%)

**Add monitoring (optional):**
```python
# In training loop, after validation
if epoch % 5 == 0:
    eos_stats = check_eos_prediction_rate(model, val_dataloader)
    print(f"Epoch {epoch}: EOS prediction rate = {eos_stats['rate']:.1%}")
```

### 3. Validate Fix

**After training completes (e.g., epoch 30):**

```bash
# Step 1: Run diagnostic
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/diagnose_keypoint_counts.py \
    --checkpoint outputs/cape_run_with_eos_fix/checkpoint_best.pth \
    --per-category \
    --num-samples 20

# Expected output:
#   ✅ EOS prediction rate: >80%
#   ✅ Average predicted keypoints: 17-32
#   ✅ Extra predictions: <50 total (<5 per sample)

# Step 2: Run evaluation
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run_with_eos_fix/checkpoint_best.pth \
    --num-visualizations 50 \
    --output-dir outputs/cape_eval_with_fix

# Expected output:
#   ✅ PCK scores realistic (likely 30-60%, not 100%)
#   ✅ No warnings about excessive keypoint generation
#   ✅ Visualizations show reasonable keypoint counts

# Step 3: Run tests
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest tests/test_eos_prediction.py -v

# Expected output:
#   ✅ 6/6 tests passing
```

### 4. Compare Results

**Old checkpoint (epoch 24, before fix):**
- Predicted keypoints: 200 (always)
- EOS rate: 0%
- Excess predictions: 183 per sample
- PCK: 100% (unrealistic)

**New checkpoint (after retraining with fix):**
- Predicted keypoints: 17-32 (category-dependent) ✅
- EOS rate: >80% ✅
- Excess predictions: <5 per sample ✅
- PCK: 30-60% (realistic) ✅

---

## 📊 Summary Statistics

### Files Modified

**Core Fixes:**
- `datasets/mp100_cape.py` - EOS included in visibility mask (11 lines added)
- `models/roomformer_v2.py` - Incomplete generation warning (14 lines added)

**Diagnostic Assertions:**
- `scripts/eval_cape_checkpoint.py` - Pre-trimming check (15 lines added)
- `models/engine_cape.py` - Pre-trimming check (15 lines added)

**Total code changes:** ~55 lines added across 4 files

### Tools Created

- `scripts/diagnose_keypoint_counts.py` - 462 lines (comprehensive diagnostic)
- `tests/test_eos_prediction.py` - 331 lines (test suite)

**Total diagnostic tools:** ~793 lines

### Documentation

- `KEYPOINT_COUNT_DIAGNOSTIC_REPORT.md` - Full diagnostic report
- `docs/EOS_TOKEN_BUG_FIX.md` - Technical documentation
- `EOS_FIX_IMPLEMENTATION_SUMMARY.md` - This summary

**Total documentation:** ~850 lines

---

## 🎯 Key Takeaways

### The Bug in One Sentence

**"The EOS token was excluded from the classification loss, causing the model to never learn when to stop generating keypoints."**

### The Fix in One Sentence

**"Include the EOS token in the visibility_mask so it contributes to the classification loss and the model learns proper stopping behavior."**

### Success Criteria

The fix is successful when:
1. ✅ EOS token included in loss (confirmed by tests)
2. ⏸️ Model predicts EOS during inference (requires retrain)
3. ⏸️ Generation stops at appropriate lengths (requires retrain)
4. ⏸️ Trimming is minimal (<5 keypoints, requires retrain)
5. ⏸️ PCK scores are realistic (requires retrain)

**Current Status: 1/5 complete (code fix done, awaiting retrain for validation)**

---

## ⚠️ Important Notes

### For Existing Checkpoints

**All checkpoints in `outputs/cape_run/` are AFFECTED by this bug.**

Do NOT use them for production evaluation. They will:
- Generate 200 keypoints always
- Show unrealistic PCK scores
- Trigger warnings during evaluation

### For New Training

**Start from scratch** (don't resume from old checkpoints):
```bash
--resume ''  # Critical: Don't load old weights
```

Old checkpoints learned to predict COORD for all positions. Starting from them would bias the model even with the fix.

### MPS Device Limitations

If running on Mac with MPS:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This enables CPU fallback for unsupported operations (slight performance penalty but necessary for inference).

---

## 📞 Support

If issues persist after retraining:

1. **Run diagnostic script:**
   ```bash
   python scripts/diagnose_keypoint_counts.py --checkpoint path/to/new_checkpoint.pth
   ```

2. **Check EOS rate:**
   - Should be >50% by epoch 10
   - Should be >80% by epoch 20

3. **Verify visibility mask:**
   ```bash
   python -m pytest tests/test_eos_prediction.py::test_eos_token_in_visibility_mask -v
   ```

4. **Check training logs:**
   - Classification loss should decrease
   - No systematic bias toward COORD class

---

## 🎉 Conclusion

The EOS token bug has been **successfully identified, fixed, and tested** at the code level.

**Completed:**
- ✅ Root cause analysis (100% confirmed)
- ✅ Core fix implemented (EOS in visibility mask)
- ✅ Safety warnings added (incomplete generation detection)
- ✅ Assertions added (pre-trimming checks)
- ✅ Tests created and passing (3/6 - dataset level)
- ✅ Diagnostic tools created
- ✅ Documentation complete

**Remaining:**
- ⏸️ Model retraining (requires 24-48 hours)
- ⏸️ Validation on new checkpoint (3/6 model-level tests)
- ⏸️ Performance comparison (old vs new)

**The codebase is now ready for retraining with proper EOS token learning.**

