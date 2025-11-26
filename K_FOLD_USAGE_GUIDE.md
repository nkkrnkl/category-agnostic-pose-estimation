# K-Fold Cross-Validation Usage Guide

Complete guide for running 5-fold cross-validation with CAPE on MP-100.

---

## Quick Start

### Run Full K-Fold Cross-Validation

```bash
# Full training + evaluation on all 5 splits (takes ~5× normal training time)
./scripts/run_kfold_cross_validation.sh --epochs 300
```

**This will:**
1. Train on all 5 MP-100 splits sequentially (split1 → split2 → split3 → split4 → split5)
2. Evaluate each trained model on its test set
3. Aggregate results and compute mean ± std PCK

**Estimated time:** ~5 days if 1 epoch = 1 min (adjust based on your hardware)

---

## Basic Usage

### 1. Full K-Fold (Training + Evaluation)

```bash
./scripts/run_kfold_cross_validation.sh \
    --epochs 300 \
    --batch_size 2 \
    --episodes 500 \
    --output_dir outputs/my_kfold_run
```

### 2. Quick K-Fold (Fewer Epochs for Testing)

```bash
# Test the k-fold pipeline with only 10 epochs per fold
./scripts/run_kfold_cross_validation.sh \
    --epochs 10 \
    --episodes 100 \
    --output_dir outputs/kfold_test
```

### 3. Resume From Specific Split

```bash
# If split 1-2 are done, resume from split 3
./scripts/run_kfold_cross_validation.sh \
    --epochs 300 \
    --output_dir outputs/my_kfold_run \
    --resume_from 3
```

### 4. Evaluation Only (Already Trained)

```bash
# If you already trained all 5 splits, just run evaluation + aggregation
./scripts/run_kfold_cross_validation.sh \
    --output_dir outputs/my_existing_kfold \
    --eval_only
```

---

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of epochs per fold | 300 |
| `--batch_size N` | Batch size | 2 |
| `--episodes N` | Episodes per epoch | 500 |
| `--output_dir DIR` | Base output directory | `outputs/kfold_<timestamp>` |
| `--resume_from N` | Resume from split N (1-5) | 1 |
| `--eval_only` | Skip training, only evaluate | false |
| `--device DEVICE` | Device (mps/cuda/cpu) | mps |

---

## Output Structure

After running k-fold, your output directory will look like:

```
outputs/kfold_20251126_120000/
├── kfold_config.txt              # Configuration used
├── kfold_summary.json            # Aggregated results (machine-readable)
├── kfold_report.txt              # Aggregated results (human-readable)
│
├── split1/                       # Fold 1
│   ├── checkpoint_best.pth
│   ├── checkpoint_e300_*.pth
│   ├── training.log
│   ├── test_eval/
│   │   ├── metrics.json
│   │   └── visualizations/
│   ├── val_eval/
│   │   ├── metrics.json
│   │   └── visualizations/
│   └── ...
│
├── split2/                       # Fold 2
│   └── ...
│
├── split3/                       # Fold 3
│   └── ...
│
├── split4/                       # Fold 4
│   └── ...
│
└── split5/                       # Fold 5
    └── ...
```

---

## Understanding Results

### Aggregated Summary (JSON)

`kfold_summary.json` contains:

```json
{
  "method": "CAPE - 5-Fold Cross-Validation on MP-100",
  "num_folds": 5,
  "test": {
    "pck_overall": {
      "mean": 0.3845,      # Mean PCK across 5 folds
      "std": 0.0124,       # Standard deviation
      "min": 0.3701,       # Worst fold
      "max": 0.3989,       # Best fold
      "count": 5
    },
    "pck_mean_categories": { ... }
  },
  "per_fold_results": [
    {"split": 1, "test_pck_overall": 0.3845, ...},
    {"split": 2, "test_pck_overall": 0.3912, ...},
    ...
  ]
}
```

### Human-Readable Report

`kfold_report.txt` contains formatted results:

```
================================================================================
K-FOLD CROSS-VALIDATION RESULTS
================================================================================

Method: CAPE - 5-Fold Cross-Validation on MP-100
Number of folds: 5

--------------------------------------------------------------------------------
TEST SET RESULTS
--------------------------------------------------------------------------------

PCK@0.2 (Overall):
  Mean:  0.3845 (38.45%)
  Std:   0.0124
  Min:   0.3701 (37.01%)
  Max:   0.3989 (39.89%)

Per-fold breakdown:
  Split 1: PCK=0.3845 (38.45%)
  Split 2: PCK=0.3912 (39.12%)
  ...

================================================================================

REPORTING GUIDELINES:

For publication/benchmark comparison, report:
  Test PCK@0.2: 38.45% ± 1.24%

LaTeX format:
  $38.45 \pm 1.24$

================================================================================
```

---

## Workflow Examples

### Example 1: Full K-Fold for Publication

```bash
# Step 1: Run full k-fold cross-validation
./scripts/run_kfold_cross_validation.sh \
    --epochs 300 \
    --output_dir outputs/cape_kfold_final

# Wait for completion (several days)

# Step 2: Check results
cat outputs/cape_kfold_final/kfold_report.txt

# Step 3: Use in paper
# Report: "Test PCK@0.2: 38.45% ± 1.24%"
```

### Example 2: Quick Validation Run

```bash
# Test the k-fold pipeline with minimal epochs
./scripts/run_kfold_cross_validation.sh \
    --epochs 5 \
    --episodes 50 \
    --output_dir outputs/kfold_quick_test

# Check if everything works
cat outputs/kfold_quick_test/kfold_report.txt
```

### Example 3: Incremental K-Fold (Spread Over Multiple Days)

```bash
# Day 1: Run splits 1-2
./scripts/run_kfold_cross_validation.sh \
    --epochs 300 \
    --output_dir outputs/my_kfold \
    --resume_from 1

# Manually stop after split 2 completes

# Day 2: Resume from split 3
./scripts/run_kfold_cross_validation.sh \
    --epochs 300 \
    --output_dir outputs/my_kfold \
    --resume_from 3

# Continue until all 5 splits are done

# Final step: Aggregate (if not done automatically)
python scripts/aggregate_kfold_results.py \
    --input_base outputs/my_kfold \
    --output_file outputs/my_kfold/kfold_summary.json
```

### Example 4: Re-evaluate Existing Checkpoints

```bash
# You already trained 5 models manually, just run evaluation
./scripts/run_kfold_cross_validation.sh \
    --output_dir outputs/my_manual_splits \
    --eval_only

# This will:
# - Find checkpoint_best.pth in each split{1-5}/
# - Run evaluation on test sets
# - Aggregate results
```

---

## Manual K-Fold (Alternative Approach)

If you prefer manual control, you can run each split separately:

### Step 1: Train Each Split

```bash
# Split 1
python models/train_cape_episodic.py \
    --mp100_split=1 \
    --output_dir=outputs/manual_kfold/split1 \
    --epochs=300

# Split 2
python models/train_cape_episodic.py \
    --mp100_split=2 \
    --output_dir=outputs/manual_kfold/split2 \
    --epochs=300

# ... Splits 3, 4, 5
```

### Step 2: Evaluate Each Split

```bash
for SPLIT in {1..5}; do
    python scripts/eval_cape_checkpoint.py \
        --checkpoint outputs/manual_kfold/split${SPLIT}/checkpoint_best.pth \
        --split test \
        --num-episodes 100 \
        --output-dir outputs/manual_kfold/split${SPLIT}/test_eval
done
```

### Step 3: Aggregate Results

```bash
python scripts/aggregate_kfold_results.py \
    --input_base outputs/manual_kfold \
    --output_file outputs/manual_kfold/kfold_summary.json
```

---

## Troubleshooting

### Issue: Split N Failed

**Solution:** Use `--resume_from` to skip completed splits

```bash
./scripts/run_kfold_cross_validation.sh \
    --resume_from 3 \
    --output_dir outputs/my_kfold
```

### Issue: Out of Memory

**Solution:** Reduce batch size or episodes

```bash
./scripts/run_kfold_cross_validation.sh \
    --batch_size 1 \
    --episodes 250
```

### Issue: Want to Stop Mid-Run

**Solution:** Press Ctrl+C, then resume later

```bash
# After stopping
./scripts/run_kfold_cross_validation.sh \
    --output_dir outputs/my_kfold \
    --resume_from 3  # Resume from next uncompleted split
```

### Issue: Missing Metrics

**Solution:** Check that evaluation ran successfully

```bash
# Look for metrics.json in each split
ls outputs/my_kfold/split*/test_eval/metrics.json

# If missing, re-run evaluation only
./scripts/run_kfold_cross_validation.sh \
    --output_dir outputs/my_kfold \
    --eval_only
```

---

## Performance Tips

### Speed Up K-Fold

1. **Use GPU if available:**
   ```bash
   ./scripts/run_kfold_cross_validation.sh --device cuda
   ```

2. **Reduce epochs for initial testing:**
   ```bash
   ./scripts/run_kfold_cross_validation.sh --epochs 50
   ```

3. **Reduce episodes per epoch:**
   ```bash
   ./scripts/run_kfold_cross_validation.sh --episodes 250
   ```

4. **Run splits in parallel (if you have multiple GPUs):**
   ```bash
   # Terminal 1
   python models/train_cape_episodic.py --mp100_split=1 --device cuda:0 &
   
   # Terminal 2
   python models/train_cape_episodic.py --mp100_split=2 --device cuda:1 &
   
   # ... etc
   ```

### Estimate Time

**Formula:**
```
Total time = (time per epoch) × (epochs) × 5 splits
```

**Example:**
- Time per epoch: 1 minute
- Epochs: 300
- Total: 1 min × 300 × 5 = 1500 min = 25 hours

---

## Verification

### Check K-Fold Is Working Correctly

After completion, verify:

```bash
# 1. All splits have results
ls outputs/my_kfold/split{1..5}/test_eval/metrics.json

# 2. Results look reasonable
cat outputs/my_kfold/kfold_report.txt

# 3. Standard deviation is not too high
# (If std > 10%, might indicate unstable training)

# 4. All 5 folds contributed
python scripts/aggregate_kfold_results.py \
    --input_base outputs/my_kfold \
    --output_file outputs/my_kfold/verify.json

# Check "count": 5 in verify.json
```

---

## FAQ

**Q: Do I need different `category_splits.json` for each split?**

A: No! The current `category_splits.json` works for all splits. The `--mp100_split` argument controls which annotation file is loaded, which has the actual category partition.

**Q: Can I run k-fold on a subset of splits (e.g., only 3 folds)?**

A: Yes, but it won't be true 5-fold. You can manually run specific splits and aggregate only those.

**Q: How do I compare my results to published papers?**

A: Most papers on MP-100 report 5-fold averaged results. Use the mean ± std from `kfold_report.txt`.

**Q: What if I only care about one split?**

A: That's fine for development! Just run:
```bash
python models/train_cape_episodic.py --mp100_split=1
```

**Q: Can I use different hyperparameters for each split?**

A: Yes, but then it's not true cross-validation. Edit the bash script if you need custom configs per split.

---

## Next Steps

1. **Run quick test:** `--epochs 5` to verify setup
2. **Run full k-fold:** `--epochs 300` for publication-quality results
3. **Analyze results:** Check `kfold_report.txt`
4. **Report metrics:** Use mean ± std in your paper

---

For questions or issues, see:
- `K_FOLD_CROSS_VALIDATION_ANALYSIS.md` - Full technical analysis
- `K_FOLD_QUICK_ANSWER.md` - Quick reference

