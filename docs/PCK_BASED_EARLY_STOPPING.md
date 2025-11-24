# PCK-Based Early Stopping Implementation

## Summary

Changed early stopping criterion from **validation loss** to **PCK (Percentage of Correct Keypoints)** for better alignment with pose estimation objectives.

---

## Motivation

### The Problem with Loss-Based Early Stopping

**Scenario**: Model training with loss-based early stopping
```
Epoch 100: loss=0.10, PCK=0.75  ← Best loss
Epoch 101: loss=0.11, PCK=0.76  ← Loss worse, but PCK improved!
Epoch 102: loss=0.11, PCK=0.77  ← Loss still worse, PCK improving
...
Epoch 120: loss=0.11, PCK=0.85  ← PCK keeps improving!

→ Early stopping triggers at epoch 120 (20 epochs without loss improvement)
→ Training stops even though PCK is climbing ❌
→ Never reach potential best PCK of 0.90+ at epoch 150
```

### Why PCK is Better for Pose Estimation

1. **Direct Measurement**: PCK directly measures keypoint accuracy (what we care about)
2. **Task-Aligned**: Loss is a proxy; PCK is the actual evaluation metric
3. **Prevents Premature Stopping**: Won't stop if pose accuracy is still improving
4. **Better Final Model**: Maximizes the metric used for evaluation

---

## Changes Made

### 1. `train_cape_episodic.py`

**Lines 478-560**: Updated best model tracking and early stopping logic

**Key Changes**:
```python
# BEFORE: Early stopping based on loss
if val_loss < best_val_loss:
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

# AFTER: Early stopping based on PCK
pck_improved = False
if val_pck > best_pck:
    pck_improved = True
    epochs_without_improvement = 0  # Reset when PCK improves
else:
    epochs_without_improvement += 1  # Only increment if PCK doesn't improve
```

**Console Output Updated**:
```python
# BEFORE
print(f"  → No improvement in val_loss for {epochs_without_improvement} epoch(s)")
print(f"     Best loss: {best_val_loss:.4f} | Current: {val_loss:.4f}")
print(f"     Best PCK:  {best_pck:.4f} | Current: {val_pck:.4f}")

# AFTER
print(f"  → No improvement in PCK for {epochs_without_improvement} epoch(s)")
print(f"     Best PCK:  {best_pck:.4f} | Current: {val_pck:.4f}")
print(f"     Best loss: {best_val_loss:.4f} | Current: {val_loss:.4f}")
```

**Early Stopping Message**:
```python
# BEFORE
print(f"No improvement in validation loss for {args.early_stopping_patience} epochs.")
print(f"Best validation loss: {best_val_loss:.4f}")

# AFTER
print(f"No improvement in PCK for {args.early_stopping_patience} epochs.")
print(f"Best PCK: {best_pck:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")
```

**Help Text**:
```python
# BEFORE
help='Stop training if validation loss does not improve for N epochs'

# AFTER
help='Stop training if PCK does not improve for N epochs'
```

---

### 2. `README.md`

**Early Stopping Section**: Updated to explain PCK-based early stopping

**Added Explanation**:
```markdown
**Why PCK instead of loss?**
- PCK directly measures pose estimation accuracy (keypoint correctness)
- Validation loss is a proxy metric that can diverge from PCK
- Training can stop early on loss while PCK is still improving
- For pose estimation, we care about keypoint accuracy, not loss magnitude
```

**Best Model Selection**: Updated recommendations
```markdown
**For pose estimation (recommended)**: Use `checkpoint_best_pck_*.pth`
- Highest PCK = best keypoint accuracy
- Used as early stopping criterion
- What we actually optimize for
```

---

### 3. `CHECKPOINT_FIXES_SUMMARY.md`

Updated to document PCK-based early stopping as part of the checkpoint system improvements.

---

## Impact

### Before (Loss-Based)
- ❌ Could stop training while PCK still improving
- ❌ Might miss best pose accuracy
- ⚠️ Loss and PCK can diverge
- ✅ More stable (loss less noisy)

### After (PCK-Based)
- ✅ Stops when pose accuracy plateaus
- ✅ Maximizes keypoint accuracy
- ✅ Aligned with evaluation metric
- ⚠️ PCK can be noisy on small validation sets (acceptable trade-off)

---

## Example Training Run

### Scenario: PCK Improves While Loss Plateaus

```bash
Epoch 95/300
  ✓ Saved BEST LOSS model (val_loss: 0.0987, PCK: 0.7543)

Epoch 100/300
  ✓ Saved BEST PCK model (PCK: 0.7651, val_loss: 0.1012)
  → No improvement in PCK for 0 epoch(s)
     Best PCK:  0.7651 | Current: 0.7651
     Best loss: 0.0987 | Current: 0.1012

Epoch 105/300
  ✓ Saved BEST PCK model (PCK: 0.7823, val_loss: 0.1034)
  → No improvement in PCK for 0 epoch(s)
     Best PCK:  0.7823 | Current: 0.7823
     Best loss: 0.0987 | Current: 0.1034

# Loss stays around 0.10, but PCK keeps climbing to 0.85
# Training continues because PCK is improving!

Epoch 142/300
  → No improvement in PCK for 20 epoch(s)
     Best PCK:  0.8523 | Current: 0.8401
     Best loss: 0.0987 | Current: 0.1045

================================================================================
Early stopping triggered!
No improvement in PCK for 20 epochs.
Best PCK: 0.8523 (epoch 122)
Best validation loss: 0.0987
================================================================================
```

**Result**: 
- ✅ Reached PCK=0.8523 (much better than 0.7543 at best-loss epoch)
- ✅ Saved both best-loss and best-PCK models
- ✅ Can use best-PCK model for deployment (highest accuracy)

---

## Backward Compatibility

✅ **Fully backward compatible**
- All existing checkpoint files work
- Resume logic unchanged
- Only the early stopping criterion changed
- Both best-loss and best-PCK models still saved

---

## Validation

### Manual Testing
1. Run training for a few epochs
2. Observe console output shows "No improvement in PCK"
3. Verify early stopping message mentions PCK
4. Check both checkpoint_best_loss_*.pth and checkpoint_best_pck_*.pth are saved

### Expected Behavior
```bash
python train_cape_episodic.py \
    --epochs 300 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/test_pck_early_stop

# Should see:
#   → No improvement in PCK for X epoch(s)
#   Early stopping triggered when PCK plateaus for 20 epochs
#   Both best-loss and best-PCK checkpoints saved
```

---

## Future Considerations

### Optional: Make Early Stopping Metric Configurable

If needed in the future, could add:
```python
parser.add_argument('--early_stopping_metric', 
                    default='pck', 
                    choices=['loss', 'pck', 'both'])
```

**Why not implemented now**:
- PCK is the right metric for pose estimation
- Keeps code simpler
- Can add if other use cases emerge

---

## Files Modified

1. ✅ `train_cape_episodic.py` - Early stopping logic
2. ✅ `README.md` - Documentation updated
3. ✅ `CHECKPOINT_FIXES_SUMMARY.md` - Summary updated
4. ✅ `PCK_BASED_EARLY_STOPPING.md` - This document

---

## Conclusion

Early stopping is now based on **PCK (pose accuracy)** instead of validation loss, making it better aligned with the pose estimation objective. This prevents premature stopping when keypoint accuracy is still improving.

**Recommendation**: Use default `--early_stopping_patience 20` for most training runs. The system will automatically stop when PCK plateaus for 20 epochs.

---

*Implemented: 2024*
*Rationale: Better alignment with pose estimation objectives*

