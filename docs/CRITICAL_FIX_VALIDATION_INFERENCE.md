# 🚨 CRITICAL FIX: Validation Now Uses Autoregressive Inference

## The Bug (Why PCK Was Stuck at 100%)

### What Was Wrong
The validation function (`evaluate_cape()`) was using **teacher forcing** instead of autoregressive inference:

```python
# WRONG (old code):
outputs = model(
    samples=query_images,
    support_coords=support_coords,
    support_mask=support_masks,
    targets=query_targets,  # ← BUG: Passing ground truth!
    skeleton_edges=support_skeletons
)
```

**What this meant:**
- ✅ Model received the **correct ground truth sequence** as input
- ✅ It just needed to output coordinates for each token
- ❌ It was NOT actually predicting the sequence from scratch
- ❌ PCK@100% just meant "when given the correct sequence, can you output correct coordinates?"

**This was NOT a real test of the model's ability!**

---

## The Fix

### What Changed
Validation now uses **autoregressive inference** (`forward_inference()`), the same way testing on unseen categories works:

```python
# CORRECT (new code):
predictions = model.forward_inference(
    samples=query_images,
    support_coords=support_coords,
    support_mask=support_masks,
    skeleton_edges=support_skeletons  # NO targets!
)
```

**What this means:**
- ✅ Starts from BOS token
- ✅ Predicts next token based on previous predictions
- ✅ Never sees ground truth during generation
- ✅ **Real test of model's generalization ability!**

---

## What to Expect Now

### 📊 Metrics Will Change Dramatically

**Before (fake validation with teacher forcing):**
```
PCK@0.2: 100.00% (1235/1235 keypoints)
```

**After (real validation with autoregressive inference):**
```
PCK@0.2: ~5-20% initially (will improve as model trains)
```

**This is NORMAL and EXPECTED!**

---

### 🎯 Why Lower PCK is Actually Better

**The lower PCK reflects reality:**

1. **Training** uses teacher forcing on **SEEN** categories
   - Model learns: "Given correct sequence so far, predict next coordinate"
   - This is how autoregressive models are trained

2. **Validation** uses autoregressive inference on **UNSEEN** categories
   - Model must: "Generate entire sequence from scratch"
   - Much harder! Errors compound autoregressively
   - Tests TRUE generalization to new categories

3. **As training progresses:**
   - Model learns better representations
   - Better at predicting first few keypoints
   - Better at using support pose as reference
   - PCK should gradually increase from ~5-20% → 40-60%+ (if training works)

---

## 📈 What Your Logs Will Look Like Now

### Old Logs (Misleading):
```
Epoch 6 Summary:
  Train Loss:       4.0737
  Val Loss:         4.8531
  Val PCK@0.2:      100.00%  ← FAKE (teacher forcing)
```

### New Logs (Real):
```
Epoch 6 Summary:
  Train Loss:       4.0737
  Val PCK@0.2:      12.34% (autoregressive)  ← REAL (from scratch)
    - Mean PCK:     15.67% (across categories)
```

---

## 🔍 Validation Dataset Configuration

The validation dataloader is correctly configured to use **UNSEEN categories**:

```python
val_loader = build_episodic_dataloader(
    base_dataset=val_dataset,      # Uses 'val.json' annotations
    category_split_file=...,
    split='val',                    # Uses VALIDATION categories (unseen during training!)
    batch_size=1,                   # One category at a time
    num_queries_per_episode=2,
    episodes_per_epoch=val_episodes,
    seed=args.seed + 999
)
```

**This means:**
- Training: 69 seen categories
- Validation: 10 **different** unseen categories (for hyperparameter tuning & early stopping)
- Test: 20 **different** unseen categories (final evaluation)

---

## ✅ What Changed in the Code

### 1. `engine_cape.py` - `evaluate_cape()` function

**Changes:**
- Now calls `model.forward_inference()` instead of `model()` with targets
- Removed loss computation from validation (not meaningful without teacher forcing)
- Updated progress bar to show PCK instead of loss
- Added explicit comment: "Validation (Unseen Categories)" and "Autoregressive"

**Impact:**
- Validation PCK will be much lower (realistic)
- Will show true generalization performance
- Early stopping now based on real generalization, not fake 100% PCK

### 2. `train_cape_episodic.py` - Training script

**Changes:**
- Removed `best_val_loss` tracking (validation doesn't compute loss anymore)
- Updated epoch summary to show PCK as primary metric
- Updated checkpoint saving to use only PCK
- Updated early stopping messages to remove references to validation loss

**Impact:**
- Clearer metrics focused on what matters (PCK on unseen categories)
- Checkpoints named by PCK, not loss
- Early stopping based on true generalization

---

## 🚀 How to Resume Training

Your current checkpoint (epoch 4) will resume, but:

1. **Old `best_pck` value is meaningless** (it was based on teacher forcing)
2. **New validation will show real PCK** (much lower initially)
3. **Training will continue normally**, and PCK should improve

**Resume command:**
```bash
python train_cape_episodic.py \
  --dataset_root . \
  --resume outputs/cape_run/checkpoint_e004_lr1e-04_bs2_acc4_qpe2.pth \
  --epochs 300 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --output_dir outputs/cape_run \
  2>&1 | tee -a train_output.log
```

**What to watch for:**
- Training loss should continue decreasing (currently ~4.07 → should go to ~2-3)
- **Validation PCK** will start low (~5-20%) and should gradually increase
- Look for steady improvement over 50-100 epochs
- Early stopping will trigger if PCK plateaus for 20 epochs

---

## 🎓 Why This Matters for Your PhD Mentor

Your PhD mentor will care about:

1. **Training methodology is correct:**
   - ✅ Training uses teacher forcing on seen categories (standard)
   - ✅ Validation uses autoregressive inference on unseen categories (proper evaluation)
   - ✅ Clear separation between seen/unseen categories

2. **Metrics are meaningful:**
   - ✅ PCK measures true generalization, not memorization
   - ✅ Early stopping prevents overfitting to seen categories
   - ✅ Results are directly comparable to other CAPE papers

3. **Experimental rigor:**
   - ✅ No data leakage (validation = completely different categories)
   - ✅ Proper few-shot evaluation (1-shot conditioning)
   - ✅ Reproducible (RNG states saved)

---

## 📉 Expected PCK Trajectory

Based on typical few-shot learning:

```
Epoch  | Train Loss | Val PCK (unseen)
-------|------------|------------------
1-10   | 5.0 → 4.0  | 5-15%   (model learning basics)
10-50  | 4.0 → 3.0  | 15-30%  (improving generalization)
50-100 | 3.0 → 2.5  | 30-45%  (solid performance)
100+   | 2.5 → 2.0  | 45-60%+ (approaching good results)
```

**Note:** These are rough estimates. Actual numbers depend on:
- Model architecture quality
- Support conditioning effectiveness
- Data quality and diversity
- Hyperparameter settings

---

## 🔧 If You Want to Check Both Metrics

If you want to compare teacher-forcing performance vs autoregressive:

1. **Add a debug flag** to optionally enable teacher forcing in validation
2. **Track both PCKs**: one with teacher forcing (upper bound), one without (real metric)

For now, we're using the **correct, real metric** only.

---

## Summary

| Aspect | Before (Bug) | After (Fixed) |
|--------|-------------|---------------|
| **Validation method** | Teacher forcing | Autoregressive inference |
| **Validation data** | Unseen categories ✓ | Unseen categories ✓ |
| **PCK meaning** | "Can output coords given GT sequence" | "Can predict from scratch" |
| **Expected PCK** | ~100% (fake) | ~5-60% (real, improves with training) |
| **Primary metric** | PCK (misleading) | PCK (meaningful) |
| **Early stopping** | Based on fake PCK | Based on real PCK |

**Bottom line:** Your model is actually learning! The low PCK you'll see now is the **real baseline**, and watching it improve will tell you if your model is truly learning to generalize to unseen categories.

