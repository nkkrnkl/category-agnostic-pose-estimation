# Keypoint Count Mismatch Diagnostic Report

**Date:** November 25, 2025  
**Checkpoint:** `checkpoint_e024_lr1e-04_bs2_acc4_qpe2.pth` (Epoch 24, PCK: 100%)  
**Hypothesis:** The model generates more predicted keypoints than ground truth keypoints

---

## Executive Summary

### ✅ **HYPOTHESIS CONFIRMED**

The model generates **significantly more keypoints** than expected, consistently producing **200 keypoints** regardless of the actual category requirements.

**Key Finding:** The model **NEVER predicts the EOS (End of Sequence) token**, causing generation to continue until the maximum sequence length (200) is reached.

---

## Diagnostic Results

### Test 1: Single Sample Analysis

**Configuration:**
- Samples analyzed: 2
- Debug mode: Enabled
- Device: MPS (with CPU fallback)

**Results:**
```
Sample 0:
  Expected keypoints: 17
  GT keypoints: 17
  Predicted keypoints: 200
  Extra predictions: 183
  
  GT token sequence:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2]  ← 17 COORD + 1 EOS
  Pred token sequence: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ← 18 COORD + 0 EOS
  
  Mismatch: ✅ CONFIRMED
```

**Debug Output:**
```
[DIAG roomformer_v2] Generation finished at step 200/200
  unfinish_flag: [1. 1.]              ← Both instances didn't finish!
  gen_out[0] length: 200
  output_cls_list length: 200
  output_reg_list length: 200

[DIAG sequence_utils] Extracted 200 keypoints from predicted sequence
  Total tokens: 200
  COORD tokens: 200                   ← ALL tokens are COORD!
  max_keypoints param: None
```

### Test 2: Per-Category Analysis

**Configuration:**
- Samples analyzed: 20
- Per-category breakdown: Enabled

**Results:**
```
Total samples: 20
Mismatches: 20/20 (100.0%)
Total extra predictions: 3632
Average predictions per sample: 200.0 (constant)
Average GT keypoints: 23.0 (varies by category)

Category N/A:  ← Note: metadata not correctly propagated
  Samples: 20
  Mismatches: 20 (100.0%)
  Expected keypoints: 17 (but varies: min=17, max=32)
  GT keypoints: min=17, max=32, avg=23.0
  Pred keypoints: min=200, max=200, avg=200.0  ← ALWAYS 200!
```

### Test 3: Token Sequence Inspection

**GT vs Predicted Token Types:**

| Token Position | GT Token | Pred Token | Token Name |
|----------------|----------|------------|------------|
| 0-16 | 0 | 0 | COORD |
| 17 | 2 | 0 | EOS vs COORD |
| 18-199 | (padding) | 0 | - vs COORD |

**Key Observation:** The model predicts COORD for position 17 where GT has EOS, and continues predicting COORD for all remaining positions.

---

## Root Cause Analysis

### Primary Issue: EOS Token Never Predicted

The model's token classification head is **failing to predict the EOS token** at the correct position (or at all). This causes:

1. **Uncontrolled Generation:** The model doesn't know when to stop
2. **Max Length Reached:** Generation continues to `max_len=200`
3. **Invalid Keypoints:** Extra 183 predictions are meaningless
4. **Trimming Masks Issue:** Evaluation code trims to expected count, hiding the bug

### Contributing Factors

#### 1. Token Classification Head Issue
**Location:** `models/roomformer_v2.py` - classification output layer

**Evidence:**
- GT distribution: {COORD: 17, EOS: 1} - balanced
- Pred distribution: {COORD: 200, EOS: 0} - completely biased toward COORD
- Model predicts `cls_j == TokenType.coord.value` for ALL tokens

**Possible Causes:**
- Classification loss not properly weighted (COORD vs EOS class imbalance)
- EOS token never receives strong gradient signal during training
- Softmax/argmax always favors COORD class

#### 2. EOS Stopping Logic
**Location:** `models/roomformer_v2.py:551-562`

```python
# Current logic
if cls_j == TokenType.eos.value:
    unfinish_flag[j] = 0
```

**Observation:** Logic is correct, but `cls_j == TokenType.eos.value` is **never True** because the model never predicts EOS.

#### 3. Minimum Length Constraint
**Location:** `models/roomformer_v2.py:510`

```python
if cls_j == TokenType.coord.value or (cls_j == TokenType.eos.value and i < min_len):
```

**Note:** `min_len` might be preventing early EOS, but this doesn't explain why EOS is never predicted even after `min_len`.

#### 4. Training Signal Issue
**Location:** Training loop (teacher forcing)

**Hypothesis:** During training, the model may not receive sufficient signal to learn when to predict EOS because:
- Loss masking might exclude EOS positions
- Class imbalance (many COORD, few EOS per sequence)
- Teacher forcing provides correct input regardless of model's EOS prediction

---

## Impact on Evaluation

### PCK Computation
The trimming operation masks the underlying issue:

```python
# scripts/eval_cape_checkpoint.py:464
pred_kpts_trimmed.append(pred_kpts[idx, :num_kpts_for_category, :])
gt_kpts_trimmed.append(gt_kpts[idx, :num_kpts_for_category, :])
```

**Effect:**
- PCK only evaluates first `num_kpts_for_category` predictions
- Extra 183 predictions are silently discarded
- Model appears to work correctly
- **Actual PCK = 100%** is suspicious (likely still due to previous bugs)

### Shape Mismatches
Without trimming, would cause:
- `pred_kpts.shape = (B, 200, 2)`
- `gt_kpts.shape = (B, 17, 2)`
- Broadcasting errors or assertion failures

---

## Evidence Summary

### Hypothesis Confirmation Criteria

| Criterion | Expected if TRUE | Observed | Status |
|-----------|------------------|----------|--------|
| `pred_kpts.shape[1] > gt_kpts.shape[1]` | Majority of samples | 20/20 (100%) | ✅ CONFIRMED |
| `pred_kpts.shape[1] > num_kpts_for_category` | Consistently | Always (200 > 17-32) | ✅ CONFIRMED |
| `COORD_tokens_pred > COORD_tokens_gt` | True | 200 > 17 | ✅ CONFIRMED |
| Trimming discards predictions | True | 183 per sample | ✅ CONFIRMED |

### Additional Evidence

- **EOS prediction rate:** 0/20 samples (0%)
- **unfinish_flag at end:** `[1. 1.]` (both instances incomplete)
- **Generation steps:** 200/200 (always max)
- **Token diversity:** Only COORD tokens (no EOS, SEP, or CLS in predictions)

---

## Remediation Plan

### Option A: Fix Token Classification Training

**Target:** Loss function and class balancing

**Changes:**
1. Add class weights to classification loss (emphasize EOS)
2. Ensure EOS tokens in GT sequences are not masked in loss
3. Monitor EOS prediction accuracy during training

**Files to modify:**
- `train_cape_episodic.py` - loss computation
- `datasets/mp100_cape.py` - loss masking logic

### Option B: Fix EOS Stopping Logic

**Target:** Force EOS at correct position

**Changes:**
1. Add explicit EOS insertion after expected keypoint count
2. Penalize predictions beyond expected length
3. Use category-aware `max_len` instead of fixed 200

**Files to modify:**
- `models/roomformer_v2.py` - forward_inference logic

### Option C: Hybrid Approach (RECOMMENDED)

Combine both fixes:

1. **Training time:**
   - Fix classification loss weighting
   - Add EOS prediction monitoring
   - Ensure loss includes EOS tokens

2. **Inference time:**
   - Add category-aware max_len calculation
   - Add safety check: if no EOS by expected count, force stop
   - Log warning when generation exceeds expected length

3. **Evaluation time:**
   - Add assertion before trimming (detect bug early)
   - Log statistics on prediction lengths vs expected

**Priority Files:**
1. `train_cape_episodic.py` (lines ~200-300) - loss computation
2. `models/roomformer_v2.py` (lines ~434, 551-562) - EOS handling
3. `datasets/mp100_cape.py` (lines ~710) - sequence construction
4. `scripts/eval_cape_checkpoint.py` (lines ~464) - add assertion

---

## Validation Tests

### Test 1: EOS Prediction Rate
```python
def test_eos_prediction_rate():
    """Model should predict EOS for most sequences."""
    model, dataloader = setup_validation()
    eos_count = 0
    total_count = 0
    
    for batch in dataloader:
        predictions = model.forward_inference(batch)
        token_types = predictions['logits'].argmax(-1)
        
        for i in range(token_types.shape[0]):
            has_eos = (token_types[i] == TokenType.eos.value).any()
            if has_eos:
                eos_count += 1
            total_count += 1
    
    eos_rate = eos_count / total_count
    assert eos_rate > 0.8, f"EOS prediction rate too low: {eos_rate:.2%}"
```

### Test 2: Sequence Length Consistency
```python
def test_sequence_length_matches_category():
    """Predicted sequence should not exceed expected keypoint count + margin."""
    model, dataloader = setup_validation()
    
    for batch in dataloader:
        pred_kpts, gt_kpts, metadata = run_inference(batch)
        
        for i, meta in enumerate(metadata):
            num_expected = meta['num_keypoints']
            num_predicted = pred_kpts[i].shape[0]
            
            # Allow small margin for EOS + SEP tokens
            margin = 5
            assert num_predicted <= num_expected + margin, \
                f"Too many predictions: {num_predicted} > {num_expected + margin}"
```

### Test 3: Trimming is No-Op
```python
def test_trimming_is_noop():
    """After fix, trimming should not discard predictions."""
    model, dataloader = setup_validation()
    
    for batch in dataloader:
        pred_kpts, _, metadata = run_inference(batch)
        
        for i, meta in enumerate(metadata):
            num_expected = meta['num_keypoints']
            num_predicted = pred_kpts[i].shape[0]
            
            discarded = max(0, num_predicted - num_expected)
            assert discarded == 0, \
                f"Trimming discards {discarded} predictions"
```

---

## Next Steps

1. ⏸️ **DO NOT APPLY FIXES YET** - awaiting user approval of remediation plan
2. ⏸️ Implement chosen remediation option (A, B, or C)
3. ⏸️ Add validation tests
4. ⏸️ Retrain model with fixes
5. ⏸️ Verify EOS prediction rate > 80%
6. ⏸️ Re-run full evaluation
7. ⏸️ Confirm PCK is computed correctly

---

## Files Involved

### Diagnostic Tools (Created)
- `scripts/diagnose_keypoint_counts.py` - Main diagnostic script
- `KEYPOINT_COUNT_DIAGNOSTIC_REPORT.md` - This report

### Debug Instrumentation (Already present)
- `models/roomformer_v2.py:578-583` - Generation loop debugging
- `util/sequence_utils.py:58-62` - Keypoint extraction debugging
- `scripts/eval_cape_checkpoint.py:456-461` - Trimming debugging

### Files Requiring Fixes
- `train_cape_episodic.py` - Loss computation and weighting
- `models/roomformer_v2.py` - EOS stopping logic
- `datasets/mp100_cape.py` - Sequence construction and masking
- `scripts/eval_cape_checkpoint.py` - Add pre-trimming assertion

---

## Conclusion

**The hypothesis is CONFIRMED with 100% confidence.**

The model generates exactly 200 keypoints for every sample, regardless of category requirements (17-32 keypoints expected). This is caused by the model **never predicting the EOS token**, leading to uncontrolled generation until `max_len` is reached.

The evaluation pipeline's trimming operation masks this severe bug, making it appear that the model works correctly. The artificially high PCK score (100%) was likely due to a combination of this bug and previously identified issues (coordinate space mismatches, identical pred/GT due to data leakage, etc.).

**Immediate action required:** Implement remediation plan (Option C recommended) before further training or evaluation.

