# AI Agent Context Handoff - CAPE Project
## Category-Agnostic Pose Estimation (CAPE)

**Last Updated:** November 27, 2025
**Status:** Active development, training in progress

---

## 🎯 PROJECT OVERVIEW

CAPE is a **category-agnostic pose estimation** model that can predict keypoints for ANY object category (animals, vehicles, furniture, etc.) without category-specific training. It uses:

- **Episodic meta-learning**: Each episode has 1 support image + N query images from the same category
- **Support conditioning**: Model sees support keypoints to understand the pose structure
- **Autoregressive generation**: Predicts keypoint sequence token-by-token with EOS stopping
- **MP-100 dataset**: 100 categories split into 5 folds for cross-validation

---

## 📁 KEY FILES & ARCHITECTURE

### Core Model Files
| File | Purpose |
|------|---------|
| `models/roomformer_v2.py` | Base transformer model with decoder |
| `models/cape_model.py` | CAPE wrapper with support conditioning |
| `models/geometric_support_encoder.py` | **NEW** CapeX-style geometric encoder (preferred) |
| `models/train_cape_episodic.py` | Main training script |
| `models/engine_cape.py` | Training loop, evaluation, loss computation |
| `models/cape_losses.py` | Loss criterion with EOS weighting |

### Data Pipeline
| File | Purpose |
|------|---------|
| `datasets/mp100_cape.py` | MP-100 dataset loader with tokenization |
| `datasets/episodic_sampler.py` | Episodic batch construction |
| `datasets/discrete_tokenizer.py` | Coordinate → token conversion |

### Evaluation
| File | Purpose |
|------|---------|
| `scripts/eval_cape_checkpoint.py` | Standalone evaluation + visualization |
| `util/eval_utils.py` | PCK computation |

---

## 🐛 CRITICAL BUGS FIXED (REMEMBER THESE!)

### Bug 1: EOS Token Never Learned
**Problem:** Model generated 200 tokens (max_len) instead of stopping at correct keypoint count.
**Root Cause:** EOS token was EXCLUDED from loss computation in `mp100_cape.py` line ~758.
**Fix:** Include EOS in `visibility_mask`:
```python
# OLD (wrong): visibility_mask[eos_idx] = 0
# NEW (correct): visibility_mask[eos_idx] = 1  # Include EOS in loss
```
**Also added:** `--eos_weight 20.0` class-weighted cross-entropy to combat 20:1 class imbalance.

### Bug 2: CUDA Nested Tensor Crash (Colab Only)
**Problem:** `RuntimeError: to_padded_tensor: at least one constituent tensor should have non-zero numel`
**Root Cause:** TWO issues:
1. **Mask convention inverted** in `episodic_sampler.py`: Was `v > 0` (True=visible), should be `v == 0` (True=ignore)
2. **No safety check** for all-masked batches in `geometric_support_encoder.py`

**Fix 1** (`datasets/episodic_sampler.py` ~line 269):
```python
# OLD (wrong): support_mask = torch.tensor([v > 0 for v in support_visibility], ...)
# NEW (correct): support_mask = torch.tensor([v == 0 for v in support_visibility], ...)
```

**Fix 2** (`models/geometric_support_encoder.py` ~line 197-220):
```python
# CRITICAL SAFETY CHECK: Handle all-masked batches
all_masked_per_batch = support_mask.all(dim=1)
if all_masked_per_batch.any():
    temp_mask = support_mask.clone()
    for b in range(support_mask.shape[0]):
        if all_masked_per_batch[b]:
            temp_mask[b, 0] = False  # Unmask first keypoint
    support_features = self.transformer_encoder(embeddings, src_key_padding_mask=temp_mask)
    support_features[all_masked_per_batch] = 0.0
else:
    support_features = self.transformer_encoder(embeddings, src_key_padding_mask=support_mask)
```

**Why MPS (Mac) works but CUDA crashes:** MPS doesn't use nested tensor optimization, silently processes wrong data. CUDA optimizes with nested tensors and crashes on empty batches.

### Bug 3: Validation Loss Shape Mismatch
**Problem:** `IndexError` during validation because autoregressive predictions have variable length but targets are padded to 200.
**Fix:** Pad predictions to target length before loss computation (`engine_cape.py` ~line 447-498).

### Bug 4: Train Loss >> Val Loss (Misleading)
**Problem:** Training loss was 3x higher than validation loss, looked like bug.
**Cause:** Training uses auxiliary losses from ALL 6 decoder layers. Validation only uses final layer.
**Fix:** Display both "Train Loss (all layers)" and "Train Loss (final layer)" for fair comparison.

---

## 🔧 IMPORTANT CLI ARGUMENTS

### Training (`models/train_cape_episodic.py`)
```bash
--use_geometric_encoder      # Use CapeX-style encoder (recommended)
--use_gcn_preenc             # Enable GCN pre-encoding
--eos_weight 20.0            # EOS token class weight (combat imbalance)
--val_episodes_per_epoch 200 # Validation episodes
--fixed_val_episodes         # Reuse same val episodes (stable curves)
--val_seed 42                # Seed for fixed validation
--lr 1e-4                    # Learning rate
--lr_drop 130,140            # Comma-separated LR drop epochs
--resume PATH                # Resume from checkpoint
--mp100_split 1              # Which MP-100 split (1-5)
```

### Evaluation (`scripts/eval_cape_checkpoint.py`)
```bash
--checkpoint PATH            # Checkpoint to evaluate
--num-episodes 50            # Number of evaluation episodes
--num-visualizations 10      # Number of images to visualize
--output-dir PATH            # Where to save results
--eval_seed 123              # For reproducibility
```

---

## 📊 CURRENT TRAINING STATUS

### Best Results (Split 1, ~85 epochs)
- **Best PCK@0.2:** ~64% (on 10 unseen validation categories)
- **Training stable** after EOS fixes
- **Overfitting observed:** Val/Train loss ratio ~3x after 80+ epochs

### Known Issues in Progress
1. **Validation oscillation:** 20-30% PCK swings between epochs due to small val set
   - **Mitigation:** Use `--fixed_val_episodes` and increase `--val_episodes_per_epoch`
2. **Early stopping patience exhausted** around epoch 85

---

## 📝 MP-100 DATASET STRUCTURE

### Splits (5-fold cross-validation)
```
data/annotations/
├── mp100_split1_train.json  (69 train categories)
├── mp100_split1_val.json    (10 val categories)
├── mp100_split1_test.json   (20 test categories)
├── mp100_split2_train.json
... (repeat for splits 2-5)
```

### Category Splits File
```
category_splits.json  # Defines which categories are train/val/test for each split
```

### K-Fold Scripts
```
scripts/run_kfold_cross_validation.sh   # Orchestrates 5-fold training
scripts/aggregate_kfold_results.py      # Aggregates results across folds
```

---

## 🖥️ COLAB NOTEBOOK

**File:** `train_mp100_cape_colab_(1).ipynb` or `train_mp100_cape_colab_pavlos.ipynb`

### Cell Order (After Fixes)
1. Check GPU
2. Clone repo from GitHub
3. Git pull
4. **CRITICAL: Apply CUDA fixes** (Cell 5-6) ← PATCHES THE CODE
5. Install requirements
6. Mount GCS bucket (data)
7. Mount Google Drive (checkpoints)
8. Run training

### The Patching Cell (Cell 6)
Automatically applies both CUDA fixes if not present in cloned code. Idempotent - safe to run multiple times.

---

## ⚠️ COMMON ERRORS & SOLUTIONS

| Error | Cause | Solution |
|-------|-------|----------|
| `to_padded_tensor: non-zero numel` | All keypoints masked on CUDA | Apply safety check in geometric_support_encoder.py |
| `visible=0` in progress bar | Early EOS prediction (normal in epoch 1) | Wait - model learns to predict correctly |
| PCK stuck at 100% | Teacher forcing used during eval | Use `forward_inference()` not `forward()` |
| Shape mismatch in PCK | Pred/GT keypoint count differs | Trim or pad predictions to GT length |
| `_pickle.UnpicklingError` | Spaces in checkpoint path | Quote paths: `"path with spaces/file.pth"` |

---

## 🔑 ARCHITECTURAL DECISIONS

### Why Geometric Encoder?
- Uses **coordinate MLP + 2D spatial PE + 1D sequence PE**
- Combines spatial "where" (x,y) with ordering "which keypoint"
- More robust than old SupportPoseGraphEncoder

### Why EOS Weight 20x?
- COORD tokens appear 17-32x per sequence
- EOS appears 1x per sequence
- Without weighting, gradient signal drowns EOS learning

### Why Auxiliary Losses?
- Deep supervision: losses from all 6 decoder layers
- Improves gradient flow to early layers
- Training uses all; validation only uses final layer

---

## 📈 EXPECTED TRAINING BEHAVIOR

### Epoch 1
- `visible=0` may appear on first batch (model predicts EOS immediately)
- PCK ~15-20% by end of epoch
- This is NORMAL with EOS weight

### Epochs 2-10
- PCK climbs rapidly to 30-50%
- Model calibrates EOS prediction timing
- May see "Model only generated X/Y keypoints" warnings

### Epochs 10-50
- PCK stabilizes around 50-60%
- Overfitting starts (val loss increases while train loss decreases)

### Epochs 50+
- Diminishing returns
- Consider lower LR or early stopping

---

## 🔗 RELATED DOCUMENTS

- `docs/validation_stability.md` - Validation sampling changes
- `docs/EOS_TOKEN_BUG_FIX.md` - EOS bug details
- `CAPE_PHD_SPEC_COMPLIANCE_AUDIT.md` - Architecture audit
- `K_FOLD_USAGE_GUIDE.md` - Cross-validation instructions

---

## 📞 QUICK COMMANDS

### Start Fresh Training (Local)
```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder --use_gcn_preenc \
    --epochs 300 --batch_size 2 --accumulation_steps 4 \
    --val_episodes_per_epoch 200 --fixed_val_episodes --val_seed 42 \
    --output_dir outputs/my_run \
    2>&1 | tee training_log.txt
```

### Resume Training
```bash
python models/train_cape_episodic.py \
    --resume "outputs/my_run/checkpoint_e050.pth" \
    --use_geometric_encoder --use_gcn_preenc \
    [other args...] \
    2>&1 | tee training_log_resume.txt
```

### Evaluate Checkpoint
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/my_run/checkpoint_best_pck.pth \
    --num-episodes 50 --num-visualizations 10 \
    --output-dir outputs/eval_results
```

---

## 🎯 NEXT STEPS (If Continuing)

1. **If Colab still crashes:** Verify Cell 6 patches were applied (check for "✅ All CUDA fixes applied")
2. **If PCK oscillates:** Increase `--val_episodes_per_epoch` to 400
3. **If overfitting:** Reduce LR with `--lr 5e-5 --lr_drop 50,80`
4. **For final results:** Run all 5 splits with `scripts/run_kfold_cross_validation.sh`

---

*This document contains critical context for continuing CAPE development. The mask convention bug and EOS token bug were particularly subtle and took significant debugging to identify.*

