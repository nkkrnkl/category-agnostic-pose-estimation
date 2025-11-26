# K-Fold Cross-Validation Analysis for MP-100 CAPE Training

**Date:** November 26, 2025  
**Status:** Analysis Complete - NO CODE MODIFICATIONS MADE

---

## Executive Summary

**Direct Answer: NO, our current procedure is NOT truly k-fold cross-validation.**

**Dataset Design:** ✅ MP-100 is **designed for 5-fold cross-validation**  
**Current Implementation:** ❌ We only train on **1 split per run** (no automatic k-fold)  
**What's Needed:** Manual orchestration to run 5 separate experiments

---

## 1. Current Split Usage in Training

### 1.1 How `mp100_split` is Defined

**File:** `models/train_cape_episodic.py`  
**Line:** 159

```python
parser.add_argument('--mp100_split', default=1, type=int, choices=[1, 2, 3, 4, 5])
```

**Default Value:** 1  
**No loop or iteration mechanism in the code**

### 1.2 How `mp100_split` is Used

**File:** `datasets/mp100_cape.py`  
**Function:** `build_mp100_cape()`  
**Line:** 844

```python
split_num = getattr(args, 'mp100_split', 1)

# Paths
ann_file = Path(args.dataset_root).resolve() / "annotations" / f"mp100_split{split_num}_{image_set}.json"
```

**Process:**
1. Extract `mp100_split` from args (defaults to 1)
2. Load annotation file: `data/annotations/mp100_split{N}_{train/val/test}.json`
3. Build dataset from that **single split only**

### 1.3 Training Script Execution

**Finding:** No loop over splits anywhere in the codebase

**Verified locations:**
- ❌ `models/train_cape_episodic.py` - no `for split in [1,2,3,4,5]` loop
- ❌ `START_CAPE_TRAINING.sh` - no loop, uses default mp100_split=1
- ❌ No `cross_validation.py` or similar orchestration script exists

**Conclusion:** Each training run uses **exactly 1 split** from start to finish.

---

## 2. Is Our Current Procedure Truly K-Fold?

### **Direct Answer: NO**

### 2.1 What We Currently Do

**Single-Split Training:**
```
Run 1: python train_cape_episodic.py --mp100_split=1 --epochs=300
  → Trains on split 1's 70 categories
  → Validates on split 1's 10 categories
  → Never sees splits 2-5
```

**This is ONE fold, not k-fold.**

### 2.2 What True K-Fold Would Require

**5-Fold Training (what MP-100 is designed for):**
```bash
# Fold 1
python train_cape_episodic.py --mp100_split=1 --epochs=300
→ Save results: pck_split1.json

# Fold 2
python train_cape_episodic.py --mp100_split=2 --epochs=300
→ Save results: pck_split2.json

# Fold 3
python train_cape_episodic.py --mp100_split=3 --epochs=300
→ Save results: pck_split3.json

# Fold 4
python train_cape_episodic.py --mp100_split=4 --epochs=300
→ Save results: pck_split4.json

# Fold 5
python train_cape_episodic.py --mp100_split=5 --epochs=300
→ Save results: pck_split5.json

# Aggregate
python aggregate_results.py pck_split*.json
→ Report: Mean PCK = (pck1 + pck2 + pck3 + pck4 + pck5) / 5
```

**Status:** ❌ This orchestration does NOT exist in the current codebase

### 2.3 Why Current Implementation is NOT K-Fold

| Aspect | K-Fold Requirement | Current Implementation | Match? |
|--------|-------------------|------------------------|--------|
| Multiple folds | Run on all 5 splits | Run on 1 split only | ❌ |
| Automatic iteration | Loop over splits | No loop mechanism | ❌ |
| Result aggregation | Average metrics across folds | Single fold result | ❌ |
| Category coverage | All cats in train eventually | Only 70 cats ever trained | ❌ |

**Verdict:** Current implementation is **single-fold training**, not k-fold.

---

## 3. Intended K-Fold Protocol vs. Current Implementation

### 3.1 What the Dataset Design Implies

**MP-100 was explicitly designed for k-fold cross-validation:**

**Evidence from Category Distribution Analysis:**
```
✅ TRUE K-FOLD: All 100 categories appear in different roles across splits

Category 1:  Train in splits [1, 3]     | Val in [2, 4]  | Test in [5]
Category 2:  Train in splits [2, 3, 4]  | Val in [5]     | Test in [1]
Category 3:  Train in splits [2, 3, 4, 5] | Val in []    | Test in [1]
...
```

**Key Properties:**
1. ✅ Each split has **disjoint** train/val/test sets (no overlap within a split)
2. ✅ **All 100 categories** appear in train role in at least one split
3. ✅ **All 100 categories** appear in test role in at least one split
4. ✅ Categories rotate through train/val/test roles across splits

**This is textbook k-fold design for category-level cross-validation.**

### 3.2 How Our Code Diverges

**Intended Protocol (MP-100 authors' design):**
```
For each split in [1, 2, 3, 4, 5]:
    Train model on split's training categories
    Validate on split's validation categories
    Test on split's test categories
    Record test PCK

Report: Mean PCK ± std across 5 folds
```

**Current Implementation:**
```
Select split = 1 (hardcoded default)
Train model on split 1's training categories
Validate on split 1's validation categories
(Optional: test on split 1's test categories)

Report: PCK on split 1 only
```

**Gap:** We're using **1/5 of the intended evaluation protocol**.

### 3.3 Implications for Publication/Research

**If following standard meta-learning protocol:**
- ❌ Current results are **not directly comparable** to papers that report 5-fold averaged results
- ❌ Performance estimate has **higher variance** (single fold vs averaged)
- ❌ May be **biased** by the particular train/val/test split in fold 1

**What published papers on MP-100 typically report:**
- Mean PCK across all 5 folds
- Standard deviation across folds
- Sometimes per-fold breakdown

---

## 4. Implications for Coverage of Training Categories

### 4.1 Single Run (Current Implementation)

**For `mp100_split=1` (your current training):**

```
Training Categories:   70 (out of 100 total)
Validation Categories: 10 (unseen during training)
Test Categories:       20 (held-out for final eval)

Categories model NEVER sees: 30 (the 10 val + 20 test)
```

**Example categories the model never trains on (split 1):**
- Category 6: hamster_body (in val)
- Category 22: guanaco_face (in val)
- Category 35: gorilla_body (in val)
- Category 2: horse_body (in test)
- Category 3: dog_body (in test)
- ... (25 more categories)

**This is BY DESIGN** - for testing generalization to unseen categories.

### 4.2 Across All 5 Splits (If We Ran K-Fold)

**If we trained on all 5 splits separately:**

```
Categories trained on (union across all folds): 100 ✓
Categories validated on (union): 42
Categories tested on (union): 100 ✓
```

**Coverage breakdown:**

| Category ID | Split 1 Role | Split 2 Role | Split 3 Role | Split 4 Role | Split 5 Role |
|-------------|--------------|--------------|--------------|--------------|--------------|
| 1 | Train | Val | Train | Val | Test |
| 2 | Test | Train | Train | Train | Val |
| 3 | Test | Train | Train | Train | Train |
| ... | ... | ... | ... | ... | ... |

**Key insight:** Over all 5 folds:
- ✅ Every category gets trained on in at least one fold
- ✅ Every category gets tested on in at least one fold
- ✅ More robust estimate of generalization performance

### 4.3 Per-Fold Category Exposure

**For each individual fold (e.g., split 1):**

```
Episodic Training:
  - Samples episodes from 70 training categories only
  - Each episode: 1 support + 2 queries from SAME category
  - Over 500 episodes/epoch × 300 epochs = 150,000 episodes total
  - All 70 training categories seen many times

Episodic Validation:
  - Samples episodes from 10 UNSEEN validation categories
  - Tests few-shot generalization to categories model never trained on
  - This is the core meta-learning evaluation

Episodic Test:
  - Samples episodes from 20 UNSEEN test categories
  - Final held-out evaluation on completely unseen categories
```

**This is consistent across all 5 splits** - each split evaluates on its own unseen categories.

---

## 5. How This Interacts With Episodic Training

### 5.1 For a Single Split (Current Behavior)

**Training Phase:**
```python
# EpisodicSampler for split 1
categories = category_splits['train']  # 70 categories
for episode in range(episodes_per_epoch):
    cat = random.choice(categories)  # Sample from 70 training cats
    support = sample_image(cat)
    queries = sample_images(cat, num=2)
    # Model learns to predict keypoints for queries given support
```

**Validation Phase:**
```python
# EpisodicSampler for split 1 validation
categories = category_splits['val']  # 10 UNSEEN categories
for episode in range(val_episodes):
    cat = random.choice(categories)  # Sample from 10 val cats
    support = sample_image(cat)
    queries = sample_images(cat, num=2)
    # Test generalization to NEVER-BEFORE-SEEN categories
```

**This is correct meta-learning protocol** for a single fold.

### 5.2 In a Full K-Fold Setting

**If we ran all 5 folds, here's what would happen:**

**Fold 1 (split 1):**
```
Train on 70 categories → Learn pose estimation from support
Validate on 10 unseen categories → Test generalization
Test on 20 unseen categories → Final held-out performance

Category 1: In TRAIN → model learns it
Category 2: In TEST → model tested on it (unseen)
```

**Fold 2 (split 2):**
```
Train on 70 DIFFERENT categories → Learn pose estimation
Validate on 10 DIFFERENT unseen → Test generalization
Test on 20 DIFFERENT unseen → Final performance

Category 1: In VAL → model validated on it (unseen)
Category 2: In TRAIN → model learns it
```

**... Folds 3, 4, 5 continue rotating ...**

**After all 5 folds:**
- Category 1: Trained on in folds [1, 3], validated on in [2, 4], tested on in [5]
- Category 2: Trained on in folds [2, 3, 4], validated on in [5], tested on in [1]

**Aggregated Result:**
```
Mean Test PCK = (test_pck_split1 + test_pck_split2 + ... + test_pck_split5) / 5
Std Test PCK = std(test_pck across splits)
```

**This gives:**
- ✅ More robust performance estimate
- ✅ Confidence interval (via std)
- ✅ Every category evaluated in unseen context at least once

### 5.3 Episodic Meta-Learning Guarantee

**Key property that holds for BOTH single-fold and k-fold:**

```
For any episode during validation/test:
  - Support category = Query category (SAME category)
  - Model must generalize from 1 support example
  - Category is UNSEEN during training (zero-shot generalization)
```

**This is maintained whether you run 1 fold or 5 folds.**

**Difference in k-fold:**
- Single fold: Tests generalization to 10 val + 20 test categories
- K-fold: Tests generalization to ALL 100 categories (across different folds)

---

## 6. Suggested Strategy (Conceptual - NO CODE CHANGES YET)

### 6.1 Manual K-Fold Orchestration (Simplest)

**Approach:** Run training manually 5 times

**Steps:**
```bash
# Fold 1
python models/train_cape_episodic.py \
    --mp100_split=1 \
    --output_dir=outputs/kfold/split1 \
    --epochs=300

# Fold 2
python models/train_cape_episodic.py \
    --mp100_split=2 \
    --output_dir=outputs/kfold/split2 \
    --epochs=300

# ... Folds 3, 4, 5 ...

# Aggregate (manual or script)
# Collect PCK from each fold's checkpoint
# Compute mean and std
```

**Pros:**
- No code changes needed
- Full control over each fold

**Cons:**
- Manual process
- Time: 5× training time (e.g., 5 days if 1 day per fold)
- Need to manually aggregate results

### 6.2 Bash Script for Automated K-Fold

**File:** `scripts/run_kfold_cross_validation.sh` (to be created)

**Pseudo-code:**
```bash
#!/bin/bash

for SPLIT in 1 2 3 4 5; do
    echo "Training on split $SPLIT..."
    
    python models/train_cape_episodic.py \
        --mp100_split=$SPLIT \
        --output_dir=outputs/kfold/split${SPLIT} \
        --epochs=300 \
        --batch_size=2 \
        --num_queries_per_episode=2 \
        --episodes_per_epoch=500
    
    # Evaluate on test set
    python scripts/eval_cape_checkpoint.py \
        --checkpoint=outputs/kfold/split${SPLIT}/checkpoint_best.pth \
        --split=test \
        --mp100_split=$SPLIT \
        --output_dir=outputs/kfold/split${SPLIT}/test_eval
done

# Aggregate results
python scripts/aggregate_kfold_results.py \
    --input_dirs outputs/kfold/split*/test_eval \
    --output_file outputs/kfold/aggregated_results.json
```

**Pros:**
- Automated - run once and wait
- Consistent configuration across folds
- Automatic aggregation

**Cons:**
- Still takes 5× training time
- Need to create aggregation script

### 6.3 Result Aggregation Strategy

**What to aggregate:**

**Per-fold results to collect:**
```json
{
  "split": 1,
  "test_pck": 0.385,
  "test_pck_mean_categories": 0.402,
  "val_pck": 0.371,
  "num_test_categories": 20,
  "num_test_samples": 1933
}
```

**Aggregated report:**
```json
{
  "method": "CAPE (5-fold cross-validation)",
  "test_pck_mean": 0.388,
  "test_pck_std": 0.012,
  "test_pck_per_fold": [0.385, 0.391, 0.387, 0.392, 0.383],
  "val_pck_mean": 0.375,
  "val_pck_std": 0.009,
  "total_folds": 5
}
```

**This is what you'd report in a paper.**

### 6.4 Sanity Checks to Add

**Before running k-fold:**

1. **Verify no category overlap within each split:**
   ```python
   assert len(train_cats & val_cats) == 0
   assert len(train_cats & test_cats) == 0
   assert len(val_cats & test_cats) == 0
   ```

2. **Verify all splits have same structure:**
   ```python
   for split in [1, 2, 3, 4, 5]:
       assert len(train_cats) == 70
       assert len(val_cats) == 10
       assert len(test_cats) == 20
   ```

3. **Verify k-fold coverage:**
   ```python
   # Union of all training categories across folds should be 100
   all_train = set()
   for split in [1, 2, 3, 4, 5]:
       all_train |= get_train_categories(split)
   assert len(all_train) == 100
   ```

4. **Log which categories are trained/validated/tested per fold:**
   ```python
   for split in [1, 2, 3, 4, 5]:
       train_cats = get_train_categories(split)
       print(f"Split {split}: Training on {train_cats}")
   ```

### 6.5 Minimal Changes Needed for K-Fold

**If you want to implement k-fold properly:**

**1. Create batch script** (`scripts/run_kfold_cross_validation.sh`)
   - Loop over splits 1-5
   - Call training script with `--mp100_split=$SPLIT`
   - Save results with split-specific output dirs

**2. Create aggregation script** (`scripts/aggregate_kfold_results.py`)
   - Load results from all 5 output dirs
   - Compute mean and std of PCK
   - Generate summary report

**3. Update category_splits.json** (optional but recommended)
   - Currently only valid for split 1
   - Either:
     - Create 5 separate files: `category_splits_split{1-5}.json`
     - Or: Make current file work for all splits (if categories are always 70/10/20)

**4. Document the protocol**
   - Add README explaining k-fold process
   - Provide example commands

**No changes needed to core training code** - it already supports `--mp100_split` argument.

---

## 7. Summary of Findings

### 7.1 Dataset Design

**MP-100 Structure:**
- ✅ Designed for 5-fold cross-validation
- ✅ All 100 categories rotate through train/val/test roles
- ✅ Each split has disjoint train/val/test (70/10/20)
- ✅ Standard meta-learning evaluation protocol

### 7.2 Current Implementation

**What We Do:**
- ❌ Train on 1 split only (default: split 1)
- ❌ No automatic k-fold mechanism
- ❌ Report results from single fold only
- ✅ Episodic training/validation works correctly for single fold

**Gap:** Missing orchestration layer for k-fold

### 7.3 Implications

**For your current training:**
- ✓ Valid meta-learning experiment on split 1
- ✓ Tests generalization to 10 unseen val categories
- ✗ Results not directly comparable to k-fold papers
- ✗ Higher variance than averaged results

**To match standard protocol:**
- Need to run 5 separate experiments (one per split)
- Aggregate results across folds
- Report mean ± std

---

## 8. Recommendations

### For Your Project (Learning/Development)

**Current approach is fine:**
- ✅ One split sufficient for development
- ✅ Validates model can generalize to unseen categories
- ✅ Faster iteration (1× time instead of 5×)

**Use k-fold if:**
- Publishing results (needed for comparison)
- Need confidence intervals
- Want most robust performance estimate

### For Publication/Research

**Must implement k-fold:**
- Run on all 5 splits
- Report mean ± std of PCK
- Include per-fold breakdown in appendix

**Estimated effort:**
- Code changes: 1-2 hours (bash script + aggregation)
- Compute time: 5× your current training time
- Worth it if publishing on MP-100 benchmark

---

## 9. Conclusion

**Direct Answer to Your Question:**

> **"Does our current code really do k-fold training?"**

**NO.** Our current code runs **single-fold training** on split 1 (default).

**The MP-100 dataset is designed for k-fold cross-validation**, but our implementation doesn't automatically perform it. To achieve true k-fold, you must manually run 5 separate experiments (one per split) and aggregate the results.

**This is not a bug** - it's a design choice for flexibility. The code correctly supports all 5 splits via the `--mp100_split` argument, but doesn't force you to run all of them.

---

**Report Status:** Analysis Complete  
**Next Steps:** User decides whether to implement k-fold orchestration  
**No code modifications made** per instructions

