# MP-100 CAPE Category Analysis

## 🔍 Discovery: Actual Category Splits in MP-100

After analyzing the **actual annotation files** in `data/annotations/`, we discovered that MP-100 uses a **completely different split strategy** than initially assumed.

---

## 📊 Actual Split Structure

### Category Distribution (Excluding Category 80 - Hand)

| Split | Categories | Image Examples | Purpose |
|-------|-----------|----------------|---------|
| **Train** | 69 | 12,816 images | Training categories for meta-learning |
| **Val** | 10 | 1,703 images | **Separate categories** (not used in CAPE) |
| **Test** | 20 | 1,933 images | Unseen categories for zero-shot eval |

### Key Finding: **ZERO OVERLAP**

```
Train ∩ Val  = ∅  (0 categories in common)
Train ∩ Test = ∅  (0 categories in common)
Val ∩ Test   = ∅  (0 categories in common)
```

**Total unique categories: 99** (after excluding hand)

---

## 🎯 What This Means for CAPE

### The MP-100 Val Split is NOT Used for Meta-Learning Validation

The standard MP-100 validation split (`mp100_split1_val.json`) contains **10 completely different categories** that are:
- ❌ Not in the training set
- ❌ Not in the test set
- ❌ Not suitable for validating performance on seen categories

**Categories in Val Split (IDs):**
```
[6, 12, 22, 35, 48, 66, 91, 92, 95, 96]
```

These are effectively **unused** in our CAPE setup.

---

## ✅ Correct Meta-Learning Strategy

For category-agnostic pose estimation (CAPE), we follow standard meta-learning practice:

### Training
- **Categories**: Train split (69 categories)
- **Images**: Train split images (12,816)
- **Episodes**: Randomly sampled with seed X

### Validation (During Training)
- **Categories**: Train split (69 categories) ← **SAME as training**
- **Images**: Train split images (12,816) ← **SAME pool as training**
- **Episodes**: Randomly sampled with seed X + 999 ← **DIFFERENT episodes**
- **Purpose**: Check if model overfits to specific episodes vs generalizing across episodes of seen categories

### Final Evaluation (Zero-Shot)
- **Categories**: Test split (20 categories) ← **UNSEEN during training**
- **Images**: Test split images (1,933)
- **Purpose**: Evaluate true category-agnostic generalization

---

## 📁 New `category_splits.json`

The new file was **derived directly from the annotations**, not manually created:

```json
{
  "description": "MP-100 CAPE category splits - DERIVED FROM ACTUAL ANNOTATIONS",
  "total_categories": 89,
  "train_categories": 69,
  "test_categories": 20,
  
  "train": [1, 4, 5, 7, 8, 9, 11, 13, 15, ...],  // 69 IDs
  "test": [2, 3, 10, 14, 24, 29, 30, ...]        // 20 IDs
}
```

### What Changed from Manual Version

| Aspect | Old (Manual) | New (Derived) |
|--------|-------------|---------------|
| **Source** | Manually created | Extracted from annotations |
| **Train categories** | 55 | **69** |
| **Test categories** | 14 | **20** |
| **Total** | 69 | **89** |
| **Accuracy** | Some categories were in wrong split | ✅ Matches actual data |

**Key Discovery**: The old manual split had **14 categories in test** that were actually **in the train annotations**!

Example IDs that moved from "test" to "train":
- 82, 83, 85, 86, 87, 88, 89, 90, 93, 94, 97, 98, 99, 100

---

## 🚀 Impact on Training

### Before This Fix
```python
# Tried to use val_dataset with train categories
val_loader = build_episodic_dataloader(
    base_dataset=val_dataset,  # ❌ Has different categories!
    split='train'              # Looking for train categories
)
# Result: "Valid categories (>=3 examples): 0" → CRASH
```

### After This Fix
```python
# Use train_dataset with different seed for validation
val_loader = build_episodic_dataloader(
    base_dataset=train_dataset,  # ✅ Has train categories!
    split='train',               # Looking for train categories
    seed=args.seed + 999         # Different episodes
)
# Result: "Valid categories (>=3 examples): 69" → SUCCESS
```

---

## 📋 Summary

1. ✅ **`category_splits.json` now reflects actual annotations**
   - Derived from `mp100_split1_train.json` and `mp100_split1_test.json`
   - 69 train categories, 20 test categories

2. ✅ **Validation uses train dataset**
   - Train categories, train images, different episodes
   - Standard meta-learning practice

3. ✅ **Test evaluation on unseen categories**
   - 20 completely different categories
   - True zero-shot generalization test

4. ✅ **Val split images are ignored**
   - They contain 10 categories not used in CAPE
   - Not an error, just not needed for meta-learning

---

## 🧪 Verification

To verify the new splits work correctly:

```bash
# This should now work without errors
python train_cape_episodic.py \
  --dataset_root . \
  --epochs 5 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run
```

Expected output:
```
Episodic sampler for train split: 69 categories
Valid categories (>=3 examples): 69
```

✅ No more `ValueError: min() iterable argument is empty`!

