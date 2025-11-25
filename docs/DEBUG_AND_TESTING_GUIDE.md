# CAPE Debug and Testing Guide

**Purpose**: Guide for using debug mode and validation tests  
**Audience**: Developers and researchers working with CAPE  
**Status**: Ready to use

---

## Quick Start

### Run Tests (Recommended First Step)

```bash
# Navigate to project root
cd /path/to/category-agnostic-pose-estimation

# Activate virtual environment
source venv/bin/activate

# Run validation tests
python tests/test_training_inference_structure.py
```

**Expected output**: All 6 tests pass ✅

### Enable Debug Mode

```bash
# Enable debug logging
export DEBUG_CAPE=1

# Run training
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 1 \
    --batch_size 2 \
    --num_queries_per_episode 2 \
    --output_dir ./outputs/debug_test
```

**What you'll see**:
- Episode structure on first batch
- Tensor shapes
- Verification that query targets ≠ support coords
- Inference input structure (if running evaluation)

---

## Debug Mode Details

### What DEBUG_CAPE Logs

**During Training** (first batch of first epoch):

```
[DEBUG_CAPE] ================================================================================
[DEBUG_CAPE] TRAINING EPISODE STRUCTURE (First Batch)
[DEBUG_CAPE] ================================================================================
[DEBUG_CAPE] Batch contains 4 total queries
[DEBUG_CAPE] Categories in batch: [15 27 38 52]

[DEBUG_CAPE] Tensor Shapes:
[DEBUG_CAPE]   support_coords:  torch.Size([4, 17, 2])
[DEBUG_CAPE]   support_masks:   torch.Size([4, 17])
[DEBUG_CAPE]   query_images:    torch.Size([4, 3, 512, 512])
[DEBUG_CAPE]   query_targets keys: ['seq11', 'seq12', 'target_seq', ...]
[DEBUG_CAPE]   query_targets['target_seq']: torch.Size([4, 200, 2])
[DEBUG_CAPE]   skeleton_edges:  List of 4 edge lists

[DEBUG_CAPE] ✓ VERIFICATION: Query targets ≠ Support coords: True
[DEBUG_CAPE] ================================================================================
```

**During Inference** (first batch):

```
[DEBUG_CAPE] ================================================================================
[DEBUG_CAPE] INFERENCE ON UNSEEN CATEGORIES (First Batch)
[DEBUG_CAPE] ================================================================================
[DEBUG_CAPE] Unseen categories in batch: [82 85 90 97]
[DEBUG_CAPE] Batch contains 4 queries

[DEBUG_CAPE] Inference Input Structure:
[DEBUG_CAPE]   query_images:    torch.Size([4, 3, 512, 512])
[DEBUG_CAPE]   support_coords:  torch.Size([4, 15, 2])
[DEBUG_CAPE]   support_masks:   torch.Size([4, 15])
[DEBUG_CAPE]   skeleton_edges:  List of 4
[DEBUG_CAPE]   ✓ Query GT (target_seq) loaded: True
[DEBUG_CAPE]   ✓ Query GT will be used ONLY for metrics, NOT passed to forward_inference
[DEBUG_CAPE] ================================================================================
```

### What to Look For

**✅ Good Signs**:
- `Query targets ≠ Support coords: True`
- Query batch size == Support batch size
- Token sequence shapes match expected lengths
- Unseen categories different from training categories

**⚠️ Warning Signs**:
- `Query targets ≠ Support coords: False` (indicates potential bug)
- Batch size mismatches
- Missing keys in query_targets
- Empty skeleton edge lists (may reduce performance)

### Disable Debug Mode

```bash
unset DEBUG_CAPE
# or
export DEBUG_CAPE=0
```

---

## Test Suite Details

### Test 1: Episode Query Targets Source

**What it tests**:
- Verifies query targets come from query images, not support
- Checks that support keypoints ≠ query keypoints

**Why it matters**:
- **CRITICAL**: If targets come from support, model learns wrong objective
- Ensures 1-shot learning paradigm is correct

**Expected outcome**:
```
✅ TEST 1 PASSED: Query targets come from query images
```

### Test 2: Support-Query Batch Alignment

**What it tests**:
- Verifies support[i] corresponds to query[i] after collation
- Checks that support is repeated K times (once per query)

**Why it matters**:
- **CRITICAL**: Misaligned support breaks 1-shot episodic structure
- Each query must use correct support from its episode

**Expected outcome**:
```
✅ TEST 2 PASSED: Support-query alignment correct
```

### Test 3: Causal Mask Structure

**What it tests**:
- Verifies causal mask has upper triangular structure
- Checks that future positions are masked with -inf

**Why it matters**:
- **CRITICAL**: Without causal mask, model can cheat by seeing future
- Ensures model learns proper autoregressive distribution

**Expected outcome**:
```
Causal mask for seq_len=5:
[[  0. -inf -inf -inf -inf]
 [  0.   0. -inf -inf -inf]
 [  0.   0.   0. -inf -inf]
 [  0.   0.   0.   0. -inf]
 [  0.   0.   0.   0.   0.]]

✅ TEST 3 PASSED: Causal mask correct
```

### Test 4: Forward Inference Signature

**What it tests**:
- Verifies `forward_inference` does NOT have `targets` parameter
- Checks that query GT is not passed during inference

**Why it matters**:
- **CRITICAL**: Passing targets during inference is cheating
- Query GT should only be used for metric computation

**Expected outcome**:
```
forward_inference parameters: ['self', 'samples', 'support_coords', 'support_mask', 'skeleton_edges', ...]
  ✓ 'samples' (query images) present
  ✓ 'support_coords' present
  ✓ 'support_mask' present
  ✗ 'targets' NOT present (correct!)

✅ TEST 4 PASSED: forward_inference signature correct
```

### Test 5: Mock Inference (No Query GT)

**What it tests**:
- Conceptual verification that inference works without query GT
- Confirms query GT is loaded separately for metrics

**Why it matters**:
- Validates overall inference architecture
- Ensures testing paradigm is correct

**Expected outcome**:
```
✅ TEST 5 PASSED: Inference concept verified
```

### Test 6: Support Encoding Path

**What it tests**:
- Verifies support goes through SupportPoseGraphEncoder
- Confirms support is not used as decoder input sequence

**Why it matters**:
- **CRITICAL**: Support should be conditioning, not target
- Ensures support is used correctly via cross-attention

**Expected outcome**:
```
Support encoding flow:
  1. support_coords → SupportPoseGraphEncoder
  2. → support_features (B, N, hidden_dim)
  3. → Injected into decoder for cross-attention
  4. Decoder cross-attends to support_features
  5. Decoder input (tgt) comes from query targets (NOT support)

✅ TEST 6 PASSED: Support encoding verified
```

---

## Running Individual Tests

If you want to run specific test classes:

```python
# Run only training structure tests
python -m unittest tests.test_training_inference_structure.TestTrainingStructure

# Run only inference tests
python -m unittest tests.test_training_inference_structure.TestInferenceStructure

# Run only support conditioning tests
python -m unittest tests.test_training_inference_structure.TestSupportConditioning
```

---

## Troubleshooting

### Tests Fail to Load Dataset

**Error**:
```
⚠️  Warning: Could not load full dataset: [error]
   Tests will use mock data instead.
```

**Cause**: Dataset not available or path incorrect

**Solution**:
- Ensure you're in the project root directory
- Check that `data/annotations/mp100_split1_train.json` exists
- Verify `category_splits.json` is present

**Impact**: Some tests will skip or use mock data, but core structure tests still run

### "Query targets ≠ Support coords: False"

**Error**: Debug log shows targets match support

**Cause**: Potential bug in episode construction or collation

**Solution**:
1. Check `datasets/episodic_sampler.py` line 261-264
2. Verify query targets come from `query_data['seq_data']`
3. Report issue with debug log output

**Impact**: 🚨 CRITICAL - training will be incorrect

### Causal Mask Wrong Structure

**Error**: Upper triangle not -inf

**Cause**: Mask creation function broken

**Solution**:
1. Check `models/deformable_transformer_v2.py` line 166-174
2. Verify `torch.triu(..., diagonal=1)` is used
3. Ensure `.masked_fill(mask == 1, float('-inf'))` is applied

**Impact**: 🚨 CRITICAL - model can cheat during training

---

## Extending Tests

### Add New Test

1. **Create test method** in appropriate class:

```python
def test_my_new_check(self):
    """Test description."""
    print("\n" + "-" * 80)
    print("TEST X: My New Check")
    print("-" * 80)
    
    # Your test logic
    self.assertTrue(condition, "Error message")
    
    print("\n✅ TEST X PASSED: Description")
```

2. **Run tests** to verify:

```bash
python tests/test_training_inference_structure.py
```

### Add New Debug Log

1. **Add to appropriate location** in `engine_cape.py`:

```python
if DEBUG_CAPE and <condition>:
    debug_log("Your debug message")
    debug_log(f"  Variable: {value}")
```

2. **Test with**:

```bash
export DEBUG_CAPE=1
python train_cape_episodic.py ...
```

---

## Best Practices

### For Development

1. **Always run tests** after making changes to:
   - Episode sampling logic
   - Model architecture
   - Forward pass logic
   - Collation functions

2. **Enable DEBUG_CAPE** when:
   - First time running training
   - Debugging unexpected behavior
   - Verifying changes worked correctly
   - Training on new dataset split

3. **Disable DEBUG_CAPE** when:
   - Running long training jobs (reduces log spam)
   - Debug output confirmed correct
   - Batch size > 10 (too much output)

### For Debugging Issues

1. **Start with tests**: Run `test_training_inference_structure.py`
2. **Enable debug mode**: Export `DEBUG_CAPE=1`
3. **Run one epoch**: `--epochs 1` to see logs quickly
4. **Check logs**: Verify tensor shapes and source verification
5. **Compare to expected**: Use this guide as reference

---

## Expected Behavior Summary

| Stage | Query GT | Support | Usage |
|-------|----------|---------|-------|
| **Training Forward** | ✅ Passed as `targets` | ✅ Passed as `support_coords` | Teacher forcing |
| **Training Loss** | ✅ Used as ground truth | ❌ Not used in loss | Compare predictions to query GT |
| **Inference Forward** | ❌ NOT passed | ✅ Passed as `support_coords` | Autoregressive generation |
| **Inference Metrics** | ✅ Loaded separately | ❌ Not used in metrics | Compare predictions to query GT |

**Key Invariant**: Support is **NEVER** used as the target sequence for the decoder.

---

## Quick Checklist

Before starting training:

- [ ] Run tests: `python tests/test_training_inference_structure.py`
- [ ] All 6 tests pass
- [ ] Enable debug: `export DEBUG_CAPE=1`
- [ ] Run 1 epoch to verify logs
- [ ] Check "Query targets ≠ Support coords: True"
- [ ] Disable debug for long training: `unset DEBUG_CAPE`

Before evaluating on test set:

- [ ] Enable debug: `export DEBUG_CAPE=1`
- [ ] Run evaluation once
- [ ] Check inference logs show correct structure
- [ ] Verify "Query GT will be used ONLY for metrics"
- [ ] Disable debug for full evaluation

---

**Status**: ✅ Ready to use  
**Tested**: November 25, 2025  
**Maintained by**: CAPE Development Team

