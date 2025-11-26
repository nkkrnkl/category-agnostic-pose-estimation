# K-Fold Cross-Validation Quick Reference

One-page cheat sheet for running k-fold cross-validation with CAPE on MP-100.

---

## 🚀 Quick Commands

```bash
# Test setup (10 minutes)
./scripts/test_kfold_setup.sh

# Full k-fold (2-5 days)
./scripts/run_kfold_cross_validation.sh --epochs 300

# Quick run (1-2 hours)
./scripts/run_kfold_cross_validation.sh --epochs 10

# Resume from split 3
./scripts/run_kfold_cross_validation.sh --resume_from 3

# Evaluation only (skip training)
./scripts/run_kfold_cross_validation.sh --eval_only

# View results
cat outputs/kfold_*/kfold_report.txt
```

---

## 📊 What You Get

**After k-fold completion:**
- Mean PCK across 5 folds
- Standard deviation
- Per-fold breakdown
- Publication-ready format

**Example output:**
```
Test PCK@0.2: 38.45% ± 1.24%

Per-fold:
  Split 1: 38.45%
  Split 2: 39.12%
  Split 3: 37.01%
  Split 4: 39.89%
  Split 5: 37.78%
```

---

## 🎯 Common Options

| Option | Example | Purpose |
|--------|---------|---------|
| `--epochs N` | `--epochs 300` | Set epochs per fold |
| `--batch_size N` | `--batch_size 2` | Set batch size |
| `--episodes N` | `--episodes 500` | Episodes per epoch |
| `--output_dir DIR` | `--output_dir outputs/my_kfold` | Set output location |
| `--resume_from N` | `--resume_from 3` | Skip splits 1-2 |
| `--eval_only` | `--eval_only` | Skip training |

---

## ⏱️ Time Estimates

| Epochs | Time per Fold | Total (5 folds) |
|--------|---------------|-----------------|
| 1 | 2 min | 10 min |
| 10 | 20 min | 1.7 hrs |
| 50 | 100 min | 8.3 hrs |
| 100 | 200 min | 16.7 hrs |
| 300 | 600 min | 50 hrs (2 days) |

*Based on Apple Silicon MPS. Adjust for your hardware.*

---

## 📁 Output Structure

```
outputs/kfold_TIMESTAMP/
├── kfold_summary.json     # Machine-readable results
├── kfold_report.txt       # Human-readable results
├── split1/                # Fold 1
│   ├── checkpoint_best.pth
│   ├── test_eval/
│   └── val_eval/
├── split2/                # Fold 2
├── split3/                # Fold 3
├── split4/                # Fold 4
└── split5/                # Fold 5
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Split N failed | `--resume_from N` |
| Out of memory | `--batch_size 1 --episodes 250` |
| Want to stop | Ctrl+C, resume later with `--resume_from` |
| Missing metrics | `--eval_only` to re-run evaluation |

---

## ✅ Verification

```bash
# Check all splits completed
ls outputs/my_kfold/split{1..5}/test_eval/metrics.json

# Verify aggregation worked
cat outputs/my_kfold/kfold_report.txt | grep "Mean:"

# Check fold count
cat outputs/my_kfold/kfold_summary.json | grep "num_folds"
# Should show: "num_folds": 5
```

---

## 📝 For Publication

Report k-fold results as:

**Text format:**
```
Test PCK@0.2: 38.45% ± 1.24%
```

**LaTeX format:**
```latex
$38.45 \pm 1.24$
```

**Get from:**
```bash
cat outputs/my_kfold/kfold_report.txt
# Look for "REPORTING GUIDELINES" section
```

---

## 🆘 Need Help?

1. **Full documentation:** `K_FOLD_USAGE_GUIDE.md`
2. **Implementation details:** `K_FOLD_IMPLEMENTATION_SUMMARY.md`
3. **Technical analysis:** `K_FOLD_CROSS_VALIDATION_ANALYSIS.md`

---

## 💡 Pro Tips

**Speed up testing:**
```bash
./scripts/run_kfold_cross_validation.sh --epochs 5 --episodes 50
```

**Run splits in parallel (if you have multiple GPUs):**
```bash
# Terminal 1
python models/train_cape_episodic.py --mp100_split=1 --device cuda:0 &

# Terminal 2
python models/train_cape_episodic.py --mp100_split=2 --device cuda:1 &
```

**Check progress:**
```bash
tail -f outputs/kfold_*/split*/training.log
```

---

**Quick Start:** `./scripts/test_kfold_setup.sh` ← Start here!

