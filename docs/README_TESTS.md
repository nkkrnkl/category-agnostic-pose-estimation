# Test Suite for CAPE - Category-Agnostic Pose Estimation

## Overview

This directory contains comprehensive tests to verify the correctness of the training and validation pipeline, with a focus on debugging the PCK@100% issue.

---

## Test Files

### 1. Core System Tests

#### `test_checkpoint_system_comprehensive.py`
**Purpose:** Comprehensive checkpoint system testing  
**Coverage:**
- Checkpoint saving and loading
- Optimizer state preservation
- RNG state preservation
- Multi-epoch training simulation

**Status:** ✅ PASSING (with known state_dict contamination caveat)

**Run:**
```bash
python tests/test_checkpoint_system_comprehensive.py
```

#### `test_checkpoint_loading.py`
**Purpose:** Focused test for checkpoint loading edge cases  
**Coverage:**
- Loading with missing keys
- Loading with extra keys (state_dict contamination)
- Optimizer state validation

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_checkpoint_loading.py
```

---

### 2. PCK@100% Diagnostic Tests

#### `test_tokenizer_fix_simple.py` ⭐ **MOST IMPORTANT**
**Purpose:** Verify the critical tokenizer fix  
**Coverage:**
- Model is built WITH tokenizer
- `base_model.tokenizer` is not None
- `forward_inference` exists and can be called

**What it tests:**
This test verifies that the model is built correctly with a tokenizer, which is the CRITICAL FIX for the PCK@100% bug. Without a tokenizer, `forward_inference()` crashes and validation falls back to teacher forcing, giving artificially high PCK@100%.

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_tokenizer_fix_simple.py
```

**Expected output:**
```
Step 1: Build dataset and get tokenizer...
  ✓ Tokenizer: <DiscreteTokenizerV2>

Step 2: Build model WITH tokenizer...
  ✓ base_model.tokenizer: <DiscreteTokenizerV2>
  ✓ Model created
  ✓ Has forward_inference: True

✅ TOKENIZER FIX VERIFIED!
```

#### `test_pck_100_diagnosis.py`
**Purpose:** Comprehensive data leakage check  
**Coverage:**
- EpisodicSampler sampling without replacement
- No image ID overlap between support and query
- Predictions differ from GT and support coordinates

**What it tests:**
Confirms that the validation pipeline doesn't have data leakage (support and query are different images) and that predictions are not identical to ground truth or support.

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_pck_100_diagnosis.py
```

**Expected output:**
```
Episode 0:
  Support IDs: [1200000000019508, ...]
  Query IDs:   [1200000000019572, ...]
  ✓ No image ID overlap
  
  Coordinate differences:
    Pred vs GT:      0.364896  # NOT identical!
    Pred vs Support: 0.284650  # NOT identical!

✓ NO ISSUES FOUND in validation pipeline!
```

#### `test_validation_pck_debug.py`
**Purpose:** Comprehensive unit tests for validation components  
**Coverage:**
- EpisodicSampler logic
- EpisodicDataset data loading
- episodic_collate_fn batch construction
- metadata propagation
- Model forward_inference

**Status:** ⚠️ Some tests incomplete (due to missing model args)

**Run:**
```bash
python tests/test_validation_pck_debug.py
```

#### `test_evaluate_cape_function.py`
**Purpose:** Test the `evaluate_cape` function directly  
**Coverage:**
- Mock model testing
- Dummy batch processing
- PCK computation flow

**Status:** ⚠️ Incomplete (requires extensive mock setup)

**Run:**
```bash
python tests/test_evaluate_cape_function.py
```

#### `test_pck_with_real_model.py`
**Purpose:** Full end-to-end PCK computation with real checkpoint  
**Coverage:**
- Load real checkpoint
- Run inference on validation set
- Compute actual PCK
- Detailed per-keypoint analysis

**Status:** ⚠️ Incomplete (shape mismatch issues with forward_inference output)

**Run:**
```bash
python tests/test_pck_with_real_model.py
```

---

### 3. Critical Single-Keypoint Bug Regression Tests 🚨

These tests were added after fixing the **CRITICAL** bug where `forward_inference` only returned 1 keypoint instead of the full sequence.

#### `test_forward_inference_full_sequence.py` ⭐
**Purpose:** Verify `forward_inference` returns full sequence, not single token  
**Coverage:**
- Output shape is `(B, seq_len, 2)` not `(B, 1, 2)`
- `gen_out` length matches `pred_coords` sequence length
- Both classification and regression outputs are full sequences

**What it tests:**
This is a **regression test** to prevent the critical bug where `cls_output` and `reg_output` were overwritten in each loop iteration, causing only the last token to be returned.

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_forward_inference_full_sequence.py
```

**Expected output:**
```
Check 1: Sequence length > 1
  Actual: 200
  ✅ PASS: Multiple tokens returned

Check 2: gen_out length matches pred_coords
  gen_out[0] length: 200
  pred_coords seq length: 200
  ✅ PASS: Lengths match

✅ ALL CRITICAL CHECKS PASSED
```

#### `test_no_single_token_collapse.py`
**Purpose:** Test on real validation data to ensure no single-token collapse  
**Coverage:**
- Load real checkpoint
- Run inference on multiple validation episodes
- Verify all episodes generate seq_len > 1

**What it tests:**
End-to-end test with real data to confirm the fix works in production scenarios, not just unit tests.

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_no_single_token_collapse.py
```

**Expected output:**
```
Testing with real checkpoint and validation data...

Episode 0: sequence length = 200  ✅
Episode 1: sequence length = 200  ✅

✓ All episodes generated full sequences (seq_len > 1)
✅ TEST PASSED: No single-token collapse detected
```

#### `test_pck_computation_no_error.py`
**Purpose:** Verify PCK computation works without TypeError  
**Coverage:**
- Single instance PCK computation
- Batch PCK computation with `PCKEvaluator`
- Visibility mask handling

**What it tests:**
Previously, the single-keypoint bug caused `TypeError` during PCK computation due to shape mismatches. This test confirms the fix allows PCK to compute successfully.

**Status:** ✅ PASSING

**Run:**
```bash
python tests/test_pck_computation_no_error.py
```

**Expected output:**
```
Test 1: compute_pck_bbox() with single instance...
  PCK@0.2: 0.50
  ✓ No TypeError

Test 2: PCKEvaluator with batch...
  Overall PCK: 0.50
  ✓ No errors

✅ ALL PCK TESTS PASSED
```

---

### 4. Debugging Scripts

#### `debug_validation_pck.py`
**Purpose:** Early diagnostic script for validation analysis  
**Usage:** Analyze validation behavior during training

**Run:**
```bash
python debug_validation_pck.py
```

---

## Test Execution Order

When debugging PCK issues or verifying the fixes, run tests in this order:

1. **`test_forward_inference_full_sequence.py`** - ⭐ Verify single-keypoint bug is fixed
2. **`test_tokenizer_fix_simple.py`** - Verify tokenizer fix is in place
3. **`test_pck_computation_no_error.py`** - Verify PCK computation works
4. **`test_pck_100_diagnosis.py`** - Check for data leakage and prediction quality
5. **`test_checkpoint_system_comprehensive.py`** - Verify checkpoint system works
6. Run actual training and monitor PCK

---

## Key Findings from Testing

### Bug 1: Single-Keypoint Output (CRITICAL!) 🚨
- **Symptom:** Model predicts only 1 keypoint, `PCK: N/A (TypeError)`, `IndexError` in keypoint extraction
- **Cause:** `forward_inference` only returned last token's output instead of accumulating full sequence
- **Root Cause:** `cls_output` and `reg_output` were overwritten in each autoregressive loop iteration
- **Fix:** Accumulate outputs in lists (`output_cls_list`, `output_reg_list`) and concatenate after loop
- **Test:** `test_forward_inference_full_sequence.py`, `test_no_single_token_collapse.py`, `test_pck_computation_no_error.py`
- **File:** `models/roomformer_v2.py` lines ~443-585
- **Status:** ✅ FIXED (Nov 25, 2025)

### Bug 2: Missing Tokenizer (PCK@100%)
- **Symptom:** PCK stuck at 100% during validation
- **Cause:** Model built without tokenizer → forward_inference crashes → falls back to teacher forcing
- **Fix:** Build datasets first, get tokenizer, pass to build_model()
- **Test:** `test_tokenizer_fix_simple.py`
- **Status:** ✅ FIXED

### Bug 3: State Dict Contamination
- **Symptom:** `RuntimeError: Unexpected key(s) in state_dict` when loading checkpoint
- **Cause:** Temporary module attributes assigned during forward() not cleaned up
- **Fix:** Added cleanup in `models/cape_model.py`
- **Test:** `test_checkpoint_loading.py`
- **Status:** ✅ FIXED

### Bug 4: Silent Fallback to Teacher Forcing
- **Symptom:** Validation gives PCK@100% with no error message
- **Cause:** `except AttributeError` catching tokenizer errors silently
- **Fix:** Removed silent fallback, require forward_inference to exist
- **Test:** `test_pck_100_diagnosis.py`
- **Status:** ✅ FIXED

---

## Running All Tests

```bash
# Quick verification (run in order)
python tests/test_forward_inference_full_sequence.py && \
python tests/test_tokenizer_fix_simple.py && \
python tests/test_pck_computation_no_error.py && \
python tests/test_pck_100_diagnosis.py && \
python tests/test_checkpoint_loading.py

# If all pass:
echo "✅ All critical tests passed!"
echo "Ready to start training with valid validation metrics"
```

### Minimal Test Suite (Fastest)
```bash
# Just the critical regression tests
python tests/test_forward_inference_full_sequence.py && \
python tests/test_tokenizer_fix_simple.py

# If both pass:
echo "✅ Critical fixes verified!"
```

---

## Test Maintenance

### Adding New Tests

When adding new tests:
1. Place them in the `tests/` directory
2. Name them descriptively: `test_<feature>_<aspect>.py`
3. Include clear docstrings explaining purpose
4. Add entry to this README

### Test Data

Tests use:
- **Real data:** MP-100 CAPE validation set (1703 images, 10 categories)
- **Real checkpoint:** `outputs/cape_run/checkpoint_e007_lr1e-04_bs2_acc4_qpe2.pth`
- **Minimal config:** Small batch sizes and episode counts for fast testing

---

## Known Issues

### Issue 1: Old Checkpoints Incompatible
Old checkpoints (epoch 1-7) were trained without a tokenizer and cannot be properly validated with the new code. They will generate very few keypoints during forward_inference.

**Solution:** Retrain from scratch

### Issue 2: State Dict Contamination in Old Checkpoints
Old checkpoints contain extra keys due to state_dict contamination bug. Loading them will show warnings about unexpected keys.

**Solution:** Load with `strict=False` (already implemented)

### Issue 3: Forward Inference Shape Issues
Some tests have shape mismatch issues when processing forward_inference output. The model's forward_inference may return different tensor shapes than expected.

**Status:** Under investigation

---

## Documentation

Related documentation:
- `../docs/CRITICAL_SINGLE_KEYPOINT_BUG.md` - 🚨 **CRITICAL**: Single-keypoint output bug (Nov 25, 2025)
- `../CRITICAL_BUG_PCK_100_ANALYSIS.md` - Detailed PCK@100% bug analysis
- `../FIX_SUMMARY_PCK_100.md` - Fix implementation summary
- `../CRITICAL_FIX_VALIDATION_INFERENCE.md` - Validation fix documentation
- `../docs/INDEX.md` - Full documentation index

---

## Contact & Debugging

If tests fail or PCK is still 100%:

1. Run `test_tokenizer_fix_simple.py` - if this fails, the core fix is broken
2. Check if `build_model()` is called with `tokenizer=...`
3. Check if `forward_inference` exists on the model
4. Enable debug mode: `DEBUG_CAPE=1` during training
5. Check training logs for "Using: forward_inference" message

---

Last updated: 2025-11-25 (Critical Single-Keypoint Bug Fixed ✅)

