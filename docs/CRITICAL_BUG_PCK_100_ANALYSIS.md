# CRITICAL BUG ANALYSIS: PCK@100% in Validation

## Executive Summary

**Root Cause Found:** The model was trained WITHOUT a tokenizer, causing `forward_inference()` to crash during validation. This crash was **silently caught** and validation fell back to teacher forcing, giving artificially high PCK@100%.

**Impact:** 
- ALL validation PCK scores up to epoch 7 are INVALID (teacher forcing was used)
- The model has NEVER been properly validated with autoregressive inference
- We don't actually know the model's true performance on unseen categories

**Status:** FIXED for future training, but existing checkpoints cannot be properly validated

---

## Detailed Analysis

### 1. The Bug Chain

```
1. train_cape_episodic.py builds model WITHOUT tokenizer
   ↓
2. Model's self.tokenizer = None
   ↓
3. During validation, evaluate_cape() calls model.forward_inference()
   ↓
4. forward_inference() tries to access self.tokenizer.bos
   ↓
5. AttributeError: 'NoneType' object has no attribute 'bos'
   ↓
6. Exception is caught by "except AttributeError" in evaluate_cape()
   ↓
7. Falls back to model.forward() with targets (TEACHER FORCING)
   ↓
8. Model sees ground truth during validation → PCK@100%
```

### 2. Evidence

**Test 1: Data Leakage Check**
```
✓ PASS: Episodic sampler samples without replacement
✓ PASS: EpisodicDataset returns unique images per episode
✓ PASS: Collate function preserves metadata and no overlap
```
**Result:** No data leakage - support and query are always different images

**Test 2: Forward Inference Crash**
```
Episode 0:
Support IDs: [1200000000019508, 1200000000019508]
Query IDs:   [1200000000019572, 1200000000019564]
  ✓ No image ID overlap

AttributeError: 'NoneType' object has no attribute 'bos'
  File "models/roomformer_v2.py", line 339, in _prepare_sequences
    prev_output_token_11 = [[self.tokenizer.bos] for _ in range(b)]
```
**Result:** forward_inference crashes because tokenizer is None

**Test 3: Fallback Behavior**
```python
# In evaluate_cape():
try:
    predictions = model.forward_inference(...)  # CRASHES
except AttributeError as e:
    # Silently falls back to teacher forcing
    outputs = model(targets=query_targets, ...)  # CHEATING
```
**Result:** Silent fallback to teacher forcing

### 3. Why Tokenizer Was Missing

**Original Code:**
```python
# train_cape_episodic.py (line 224)
base_model, _ = build_model(args)  # No tokenizer parameter!
```

**build_model signature:**
```python
# models/__init__.py
def build_model(args, train=True, tokenizer=None):
    return build_v2(args, train, tokenizer=tokenizer, cape_mode=is_cape)
```

**Problem:** Tokenizer defaults to `None`, and we weren't passing it

### 4. Impact on Training History

| Epoch | PCK Reported | Actual PCK | Method Used |
|-------|-------------|------------|-------------|
| 1-7   | 100.00%     | UNKNOWN    | Teacher Forcing (cheating) |
| 8+    | TBD         | TBD        | Will use forward_inference ✓ |

**ALL validation scores before epoch 8 are INVALID**

---

## The Fix

### Fix 1: Build Model WITH Tokenizer

**File:** `train_cape_episodic.py`

```python
# NEW: Build datasets BEFORE model to get tokenizer
train_dataset = build_mp100_cape('train', args)
val_dataset = build_mp100_cape('val', args)

# Get tokenizer from dataset
tokenizer = train_dataset.get_tokenizer()

# Build base model WITH tokenizer (CRITICAL!)
base_model, _ = build_model(args, tokenizer=tokenizer)
model = build_cape_model(args, base_model)
```

### Fix 2: Remove Silent Fallback

**File:** `engine_cape.py`

**Before:**
```python
try:
    predictions = model.forward_inference(...)
except AttributeError as e:
    # SILENTLY falls back to teacher forcing
    outputs = model(targets=query_targets, ...)
```

**After:**
```python
# NO fallback - require forward_inference to exist
if not hasattr(model, 'forward_inference'):
    raise RuntimeError("Model missing forward_inference!")

predictions = model.forward_inference(...)  # Will crash if broken
```

### Fix 3: Added Debug Logging

```python
DEBUG_VAL = os.environ.get('DEBUG_CAPE', '0') == '1'
if DEBUG_VAL and batch_idx == 0:
    print(f"✓ Using: forward_inference (autoregressive)")
    print(f"  Predictions differ from GT: {diff_pred_gt:.4f}")
```

---

## Testing Results

### Test 1: Forward Inference Now Works
```bash
$ python tests/test_pck_100_diagnosis.py

Episode 0:
  ✓ No image ID overlap
  Coordinate differences:
    Pred vs GT:      0.364896  # NOT identical
    Pred vs Support: 0.284650  # NOT identical
  ✓ Predictions are different from GT and Support
```

### Test 2: Old Checkpoints Cannot Be Fixed
```
pred_coords: torch.Size([2, 1, 2])  # Only 1 keypoint generated!
```

**Problem:** The old checkpoint was trained without tokenizer, so:
- Model never learned to generate full sequences
- It predicts `<eos>` on first step
- Cannot be used with forward_inference properly

**Solution:** Must retrain from scratch with tokenizer

---

## Action Items

### IMMEDIATE (Required)
1. ✅ Fix `train_cape_episodic.py` to build model with tokenizer
2. ✅ Fix `engine_cape.py` to not silently fall back to teacher forcing
3. ✅ Add debug logging to track inference method used
4. ⚠️  **RETRAIN MODEL from scratch** - old checkpoint unusable for validation

### VALIDATION (To Verify Fix)
1. Start new training run with fixed code
2. Check first validation epoch:
   - PCK should be LOW (0-30%) for untrained model
   - Should see "Using: forward_inference" in debug output
3. Monitor PCK improvement over epochs
4. Expected final PCK: 30-60% (not 100%)

### TESTING
1. ✅ Created comprehensive test suite:
   - `tests/test_pck_100_diagnosis.py` - data leakage check
   - `tests/test_pck_with_real_model.py` - full PCK computation
   - `tests/test_validation_pck_debug.py` - comprehensive validation
   - `tests/test_evaluate_cape_function.py` - function-level testing

---

## How to Verify the Fix

### Step 1: Start Fresh Training

```bash
# Remove old checkpoints to avoid confusion
rm -rf outputs/cape_run_old && mv outputs/cape_run outputs/cape_run_old

# Start new training with fixed code
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 20 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 2 \
    --episodes_per_epoch 500 \
    --output_dir outputs/cape_run_new \
    2>&1 | tee train_new.log
```

### Step 2: Check First Validation
```
Expected output at epoch 1:
  PCK@0.2: 5-20%  (random performance on unseen categories)
  
NOT:
  PCK@0.2: 100%   (indicates teacher forcing bug is back)
```

### Step 3: Enable Debug Mode
```bash
DEBUG_CAPE=1 python train_cape_episodic.py ...
```

Should show:
```
🔍 DEBUG VALIDATION (Batch 0):
  ✓ Using: forward_inference (autoregressive)
  Pred vs GT diff: 0.XXXX  (should be > 0.01)
```

---

## Why This Bug Was Hard to Find

1. **Silent Failure:** Exception was caught without logging
2. **Plausible PCK:** 100% on "validation" seemed high but not impossible
3. **No Error Messages:** No indication that fallback occurred
4. **Teacher Forcing Works:** The fallback code ran without errors

## Lessons Learned

1. **Never silently catch exceptions** in critical code paths
2. **Log which inference method is used** during validation
3. **Test with random models** to establish baselines
4. **Separate validation from training** data clearly

---

## Files Modified

1. `train_cape_episodic.py` - Build model with tokenizer
2. `engine_cape.py` - Remove silent fallback, add debug logging
3. `datasets/episodic_sampler.py` - Add support_metadata for debugging
4. Created comprehensive test suite in `tests/`

---

## Current Status

✅ Bug identified and root cause understood  
✅ Training code fixed for future runs  
✅ Validation code fixed to require forward_inference  
✅ Debug logging added  
✅ Comprehensive tests created  
⚠️  Old checkpoints (epoch 1-7) CANNOT be properly validated  
⚠️  Must RETRAIN from scratch to get valid PCK scores  

---

## Expected Behavior After Fix

### Epoch 1 (Untrained)
- PCK@0.2: 5-20%
- Method: forward_inference ✓
- Performance: Random baseline

### Epoch 10-20 (Trained)
- PCK@0.2: 30-60% (expected for few-shot learning)
- Method: forward_inference ✓
- Performance: True generalization

### What PCK@100% Would Mean Now
If PCK reaches 100% after the fix, it would indicate:
- Model is EXTREMELY well-trained (legitimate, but rare)
- Or a different bug (not teacher forcing)

But early epochs should definitely be LOW (<30%).

---

## Recommendation

**RETRAIN THE MODEL** from epoch 0 with the fixed code to get valid validation metrics and proper early stopping.

The current checkpoint at epoch 7 was never properly validated, so we don't know:
- Is it actually learning?
- Is it generalizing to unseen categories?
- Should we stop training (early stopping)?

