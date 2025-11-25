# Quick Start - After PCK@100% Fix

## TL;DR

**Bug:** Model had no tokenizer → validation crashed → fell back to teacher forcing → PCK@100%  
**Fix:** Build model with tokenizer → validation uses autoregressive inference → real PCK  
**Action:** Retrain from scratch (old checkpoint cannot be fixed)

---

## Verification (30 seconds)

```bash
# 1. Verify fix is in place
python tests/test_tokenizer_fix_simple.py

# Expected output:
# ✅ TOKENIZER FIX VERIFIED!

# 2. Verify no data leakage
python tests/test_pck_100_diagnosis.py

# Expected output:
# ✓ NO ISSUES FOUND in validation pipeline!
```

If both pass → Ready to retrain!

---

## Start New Training (1 command)

```bash
# Archive old (invalid) run
mv outputs/cape_run outputs/cape_run_OLD_INVALID

# Start fresh training with FIXED code
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 50 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 2 \
    --episodes_per_epoch 500 \
    --early_stopping_patience 10 \
    --output_dir outputs/cape_run_fixed \
    2>&1 | tee outputs/train_fixed.log
```

---

## What to Watch For

### ✅ GOOD (Fix Working)

```
Building model...
  Tokenizer: <DiscreteTokenizerV2>  ← Should see this!
    vocab_size: 1940
    num_bins: 44

Epoch 1:
  Train loss: 3.8542
  Val PCK@0.2: 0.1523 (15.23%)  ← LOW score (expected for untrained model)

Epoch 10:
  Val PCK@0.2: 0.3245 (32.45%)  ← Improving

Epoch 20:
  Val PCK@0.2: 0.4521 (45.21%)  ← Still improving
```

### ❌ BAD (Bug is Back)

```
Epoch 1:
  Val PCK@0.2: 1.0000 (100.00%)  ← BUG IS BACK!

🚨 STOP TRAINING IMMEDIATELY
```

If this happens:
```bash
# Enable debug mode
DEBUG_CAPE=1 python train_cape_episodic.py ...

# Should show:
# ✓ Using: forward_inference (autoregressive)
# 
# If you see:
# ⚠️  Using: FALLBACK (teacher forcing)
# → The fix didn't work
```

---

## Expected PCK Progression

| Epoch Range | Expected PCK | What It Means |
|-------------|--------------|---------------|
| 1-5 | 10-25% | Random baseline (untrained) |
| 10-20 | 30-45% | Learning to generalize |
| 30-40 | 45-60% | Well-trained few-shot model |
| 50+ | 50-70% | Mature model (excellent) |

**If PCK@100% at ANY epoch → Bug is back!**

---

## Debug Mode (Optional)

```bash
DEBUG_CAPE=1 python train_cape_episodic.py ...
```

**What you'll see:**
```
🔍 DEBUG VALIDATION (Batch 0):
  ✓ Using: forward_inference (autoregressive)
  Query images shape: torch.Size([2, 3, 512, 512])
  Support coords shape: torch.Size([2, 25, 2])
  Predictions shape: torch.Size([2, 25, 2])
```

---

## Files Changed

### Core Fixes (2 files)
1. `train_cape_episodic.py` - Build model with tokenizer
2. `engine_cape.py` - No silent fallback to teacher forcing

### Tests Created (6 files in `tests/`)
1. `test_tokenizer_fix_simple.py` ⭐
2. `test_pck_100_diagnosis.py`
3. `test_checkpoint_loading.py`
4. `test_checkpoint_system_comprehensive.py`
5. `test_validation_pck_debug.py`
6. `test_evaluate_cape_function.py`

### Documentation (4 files)
1. `FINAL_REPORT_PCK_BUG.md` - Complete analysis
2. `CRITICAL_BUG_PCK_100_ANALYSIS.md` - Technical details
3. `FIX_SUMMARY_PCK_100.md` - Quick reference
4. `QUICK_START_AFTER_FIX.md` - This file

---

## FAQ

### Q: Can I resume from the epoch 7 checkpoint?
**A:** No. It was trained without a tokenizer and cannot generate proper sequences. You must retrain from scratch.

### Q: Why was PCK@100% before?
**A:** Validation was using teacher forcing (showing the model the correct answer), not autoregressive inference.

### Q: How do I know the fix is working?
**A:** 
1. Run `python tests/test_tokenizer_fix_simple.py` → should pass
2. Start training → epoch 1 PCK should be ~10-20%, NOT 100%
3. Enable `DEBUG_CAPE=1` → should see "Using: forward_inference"

### Q: What if PCK is still 100%?
**A:** 
1. Run `python tests/test_pck_100_diagnosis.py`
2. Enable `DEBUG_CAPE=1` and check which inference method is used
3. Check if tokenizer is printed during model building

### Q: What PCK should I expect?
**A:** 
- Epoch 1: ~10-20% (random)
- Epoch 20: ~35-50% (learning)
- Epoch 50: ~50-70% (mature)

Anything above 70% is exceptional for few-shot pose estimation!

---

## One-Liner Verification

```bash
python tests/test_tokenizer_fix_simple.py && echo "✅ Ready to train!" || echo "❌ Fix broken!"
```

---

## Ready to Train?

If you've run the verification test and it passes, you're ready!

**Start here:**
```bash
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 50 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --output_dir outputs/cape_run_fixed
```

**Watch the first validation:**
- Epoch 1 PCK ~15% → ✅ Fix working!
- Epoch 1 PCK ~100% → ❌ Bug is back!

---

Good luck with training! 🚀

