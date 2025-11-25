# PCK@100% Bug - Fix Summary

## Problem

The validation PCK was stuck at 100% because validation was using **teacher forcing** instead of **autoregressive inference**.

## Root Cause

1. Model was built WITHOUT a tokenizer
2. `forward_inference()` crashed with `AttributeError: 'NoneType' object has no attribute 'bos'`
3. Exception was silently caught, validation fell back to `model.forward(targets=...)` (teacher forcing)
4. Teacher forcing lets the model see ground truth → 100% PCK

## The Fix

### File 1: `train_cape_episodic.py`

**Changed lines 222-244:**

```python
# ========================================================================
# CRITICAL FIX: Build model with tokenizer for forward_inference
# ========================================================================
# Without tokenizer, forward_inference() crashes with AttributeError
# when accessing self.tokenizer.bos, causing silent fallback to teacher
# forcing in validation, which gives artificially high PCK@100%.
# ========================================================================

# Build datasets first to get tokenizer
from datasets.mp100_cape import build_mp100_cape

train_dataset = build_mp100_cape('train', args)
val_dataset = build_mp100_cape('val', args)

# Get tokenizer from dataset
tokenizer = train_dataset.get_tokenizer()
print(f"Tokenizer: {tokenizer}")
print(f"  vocab_size: {len(tokenizer) if tokenizer else 'N/A'}")
print(f"  num_bins: {tokenizer.num_bins if tokenizer else 'N/A'}")
print()

# Build base model (RoomFormerV2) WITH tokenizer
print("Building base Raster2Seq model...")
base_model, _ = build_model(args, tokenizer=tokenizer)  # Pass tokenizer!
```

**Removed duplicate dataset building at lines 270-274**

### File 2: `engine_cape.py`

**Changed lines 405-433:**

```python
# ========================================================================
# CRITICAL: Use ONLY autoregressive inference (NO fallback to teacher forcing)
# ========================================================================
# Previous bug: AttributeError was silently caught and fell back to teacher
# forcing, giving PCK@100%. Now we require forward_inference to exist.
# If it fails, we raise an error instead of silently cheating.
# ========================================================================

# Check that forward_inference is available
if not hasattr(model, 'forward_inference'):
    if hasattr(model, 'module') and hasattr(model.module, 'forward_inference'):
        # DDP model
        model_for_inference = model.module
    else:
        raise RuntimeError(
            "Model does not have forward_inference method!\n"
            "Cannot run proper validation without autoregressive inference.\n"
            "Check that the model was built correctly with a tokenizer."
        )
else:
    model_for_inference = model

# Run autoregressive inference
predictions = model_for_inference.forward_inference(
    samples=query_images,
    support_coords=support_coords,
    support_mask=support_masks,
    skeleton_edges=support_skeletons
)
```

**Removed the silent `except AttributeError` fallback to teacher forcing**

## Verification

### Test 1: Tokenizer Present ✅

```bash
$ python tests/test_tokenizer_fix_simple.py

Step 1: Build dataset and get tokenizer...
  ✓ Tokenizer: <DiscreteTokenizerV2>
    Vocab size: 1940
    Num bins: 44
    Has BOS: True
    Has EOS: True

Step 2: Build model WITH tokenizer...
  ✓ base_model.tokenizer: <DiscreteTokenizerV2>
  ✓ Model created
  ✓ Has forward_inference: True

✅ TOKENIZER FIX VERIFIED!
```

### Test 2: No Data Leakage ✅

```bash
$ python tests/test_pck_100_diagnosis.py

Episode 0:
  Support IDs: [1200000000019508, 1200000000019508]
  Query IDs:   [1200000000019572, 1200000000019564]
  ✓ No image ID overlap
  
  Coordinate differences:
    Pred vs GT:      0.364896  # NOT identical!
    Pred vs Support: 0.284650  # NOT identical!
  ✓ Predictions are different from GT and Support
```

## Impact

### Before Fix
- Validation used teacher forcing (cheating)
- PCK@100% on ALL epochs (1-7)
- **All validation scores are INVALID**
- Early stopping never triggered
- No way to know if model is actually learning

### After Fix
- Validation uses autoregressive inference (correct)
- PCK will reflect TRUE performance
- Valid early stopping
- Can properly evaluate generalization

## Important Notes

### Old Checkpoints Cannot Be Fixed

The checkpoint at epoch 7 was trained without a tokenizer. Even though we can load it with the new code, it will:
- Generate very few keypoints (predicts `<eos>` immediately)
- Give poor PCK scores
- **Cannot be properly validated**

**Recommendation:** Retrain from scratch with the fixed code.

### Expected PCK After Fix

For a newly trained model:
- **Epoch 1-5:** PCK ~5-20% (random baseline on unseen categories)
- **Epoch 10-20:** PCK ~30-50% (learning to generalize)
- **Epoch 30+:** PCK ~50-70% (well-trained few-shot model)

**If you see PCK@100% again, it means the bug is back!**

## Files Modified

1. `train_cape_episodic.py` - Build model with tokenizer
2. `engine_cape.py` - Remove silent fallback to teacher forcing
3. `models/cape_model.py` - Fix state_dict contamination (separate bug)

## Files Created (Test Suite)

1. `tests/test_tokenizer_fix_simple.py` - Verify tokenizer fix
2. `tests/test_pck_100_diagnosis.py` - Data leakage check
3. `tests/test_validation_pck_debug.py` - Comprehensive validation tests
4. `tests/test_pck_with_real_model.py` - Full PCK computation test
5. `CRITICAL_BUG_PCK_100_ANALYSIS.md` - Detailed analysis document

## Next Steps

1. **Retrain from scratch** with the fixed code
2. Monitor first validation epoch - should see PCK ~10-20%
3. Watch PCK improve gradually over epochs
4. Enable debug mode with `DEBUG_CAPE=1` to see validation details

## How to Verify Training

```bash
# Start fresh training
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 50 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --output_dir outputs/cape_run_fixed \
    2>&1 | tee train_fixed.log

# Enable debug mode (optional)
DEBUG_CAPE=1 python train_cape_episodic.py ...
```

**Expected first validation:**
```
Epoch 1:
  Train loss: ...
  Val PCK@0.2: 12.3%  ← Should be LOW, not 100%!
  
🔍 DEBUG VALIDATION (Batch 0):
  ✓ Using: forward_inference (autoregressive)
  Pred vs GT diff: 0.4521  ← Should be > 0.01
```

## Confidence Level

**100% confident this is the root cause:**
- ✅ Identified exact error: `'NoneType' object has no attribute 'bos'`
- ✅ Traced execution path from training → validation → fallback
- ✅ Verified fix eliminates the crash
- ✅ Comprehensive test suite confirms behavior
- ✅ Predictions now differ from GT (not cheating)

The fix is correct. The model just needs to be retrained.

