# Quick Reference - CAPE Evaluation & Training

## Evaluate a Checkpoint (One Command)

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck.pth \
    --split val \
    --num-visualizations 50 \
    --draw-skeleton \
    --output-dir outputs/cape_eval
```

**Outputs:**
- `outputs/cape_eval/metrics_val.json` - PCK metrics
- `outputs/cape_eval/visualizations/` - GT vs Predicted images

---

## Verify Training Code is Fixed

```bash
python tests/test_tokenizer_fix_simple.py
```

**Expected:** `✅ TOKENIZER FIX VERIFIED!`

---

## Start Training (Fixed Code)

```bash
python train_cape_episodic.py \
    --dataset_root . \
    --epochs 50 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --output_dir outputs/cape_run_fixed
```

**Watch for:** Epoch 1 PCK ~10-20% (NOT 100%)

---

## Common Commands

### Quick Evaluation (5 episodes)
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint PATH.pth \
    --num-episodes 5 \
    --num-visualizations 3
```

### Full Validation (100 episodes)
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint PATH.pth \
    --split val \
    --num-episodes 100 \
    --num-visualizations 50
```

### Test Set (200 episodes)
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint PATH.pth \
    --split test \
    --num-episodes 200 \
    --num-visualizations 100
```

### With Debug Mode
```bash
DEBUG_CAPE=1 python train_cape_episodic.py ...
```

---

## Expected PCK Scores

| Epoch | Expected PCK | Status |
|-------|--------------|--------|
| 1-5 | 10-25% | Learning |
| 10-20 | 30-45% | Improving |
| 30+ | 45-65% | Well-trained |
| 100% | BUG ALERT | Teacher forcing! |

---

## Files to Read

### For Evaluation Script
- `EVALUATION_SCRIPT_GUIDE.md` - Quick start
- `scripts/README.md` - Full documentation

### For PCK Bug
- `QUICK_START_AFTER_FIX.md` - What to do
- `FINAL_REPORT_PCK_BUG.md` - Full analysis

### For Testing
- `tests/README.md` - Test guide

---

## Troubleshooting

### PCK is 100%
```bash
# 1. Verify fix
python tests/test_tokenizer_fix_simple.py

# 2. Check if using old checkpoint
# Look for warning: "OLD CHECKPOINT DETECTED"

# 3. Retrain with fixed code
```

### Evaluation Script Fails
```bash
# 1. Check checkpoint exists
ls -lh outputs/cape_run/*.pth

# 2. Try with CPU
--device cpu

# 3. Try fewer episodes
--num-episodes 5
```

### No Visualizations
```bash
# Check console for errors
python scripts/eval_cape_checkpoint.py ... 2>&1 | tee eval.log

# Try more episodes
--num-episodes 20
```

---

**Quick Help:** See `COMPLETE_WORK_SUMMARY.md` for everything

