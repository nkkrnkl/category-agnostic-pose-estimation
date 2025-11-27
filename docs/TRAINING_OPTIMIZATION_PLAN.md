# CAPE Training Optimization Plan

## Executive Summary

This document outlines specific optimizations to maximize PCK accuracy for the **Geometry-Only CAPE** model using `GeometricSupportEncoder`. Based on a comprehensive audit of the training pipeline, we identify improvements ranked by expected impact.

**Current Performance Baseline (from logs):**
- Val PCK@0.2: ~29-30% (plateaus around epoch 5-6)
- Train loss continues decreasing while val PCK stagnates → classic plateau problem

---

## 1. Strategy for Loss Plateaus (CRITICAL)

### The Problem

Looking at the training logs, we observe:
```
Epoch 4: Val PCK = 29.83%
Epoch 5: Val PCK = 29.79% (plateau)
Epoch 6: Val PCK = 26.29% (regression)
```

The current `MultiStepLR` scheduler only decays the learning rate at fixed epochs (200, 250), which:
1. **Does not adapt** to loss plateaus
2. **Cannot escape local minima** - once stuck, the model stays stuck
3. **Wastes training time** - continues at same LR even when not learning

### Recommended Solution: CosineAnnealingWarmRestarts (SGDR)

**How it addresses the user's request to "pop out of local minima":**

```
LR
 │     ┌─────┐            ┌───────────┐
 │    /       \          /             \
 │   /         \        /               \
 │  /           \      /                 \
 │ /             \____/                   \____
 └──────────────────────────────────────────────► Epochs
        T_0        2×T_0          Restart
```

- **Warm Restarts**: Every `T_0` epochs, the LR "restarts" to a high value
- **Escape Mechanism**: The LR spike provides gradient energy to jump out of local minima
- **Cosine Decay**: Smooth decay prevents oscillation between restarts
- **Adaptive**: If stuck at epoch 5, the restart at epoch `T_0` reinjects momentum

### Implementation

```python
# In train_cape_episodic.py, replace:
# lr_drop_epochs = [int(x) for x in args.lr_drop.split(',')]
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop_epochs)

# With:
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,          # Restart every 20 epochs (tune based on convergence speed)
    T_mult=2,        # Double period after each restart (20, 40, 80, ...)
    eta_min=1e-6     # Minimum LR at each cycle trough
)

# Add warmup for first 5 epochs (prevents early divergence)
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1,  # Start at 10% of base LR
    total_iters=5      # 5 epoch warmup
)
main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])
```

### Arguments to Add

```python
parser.add_argument('--scheduler', default='cosine_warmrestarts', 
                    choices=['multistep', 'cosine', 'cosine_warmrestarts', 'onecycle'],
                    help='Learning rate scheduler')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='Number of warmup epochs')
parser.add_argument('--T_0', default=20, type=int,
                    help='CosineAnnealingWarmRestarts: period of first restart')
parser.add_argument('--T_mult', default=2, type=int,
                    help='CosineAnnealingWarmRestarts: period multiplier after restart')
parser.add_argument('--eta_min', default=1e-6, type=float,
                    help='Minimum learning rate')
```

---

## 2. Ranked Improvements Table

| Priority | Feature | Current Value | Recommended Value | Expected Impact | Effort |
|----------|---------|---------------|-------------------|-----------------|--------|
| 🔴 **HIGH** | LR Scheduler | `MultiStepLR` @ [200,250] | `CosineAnnealingWarmRestarts` | +5-10% PCK | Medium |
| 🔴 **HIGH** | Weight Decay | `1e-4` | `1e-2` | +2-5% PCK (reduce overfitting) | Low |
| 🔴 **HIGH** | Geometric Augmentation | Disabled | Enable ShiftScaleRotate | +5-8% PCK | Medium |
| 🟡 **MEDIUM** | LR Warmup | None | 5 epochs linear warmup | +1-3% PCK (stability) | Low |
| 🟡 **MEDIUM** | Label Smoothing | `0.0` | `0.1` | +1-2% PCK | Low |
| 🟡 **MEDIUM** | Gradient Clipping | `0.1` | `1.0` | Faster convergence | Low |
| 🟢 **LOW** | Backbone LR Ratio | `1:10` | `1:100` (optional) | Minor impact | Low |
| 🟢 **LOW** | Dropout Tuning | `0.1` everywhere | `0.2` in encoder, `0.1` decoder | Minor impact | Low |

---

## 3. Detailed Recommendations

### 3.1 Weight Decay (HIGH IMPACT)

**Current State:**
```python
parser.add_argument('--weight_decay', default=1e-4, type=float)  # Too low!
```

**Problem:**
- AdamW with `weight_decay=1e-4` provides weak regularization
- Standard for Transformers is `1e-2` (10× higher)
- From logs: Train loss continues decreasing while val PCK stagnates → **overfitting**

**Recommendation:**
```python
parser.add_argument('--weight_decay', default=1e-2, type=float)  # Standard for AdamW
```

**References:**
- BERT: `weight_decay=0.01`
- DETR: `weight_decay=1e-4` (but different architecture)
- ViT: `weight_decay=0.01-0.1`

---

### 3.2 Geometric Augmentation (HIGH IMPACT)

**Current State:**
```python
# In datasets/mp100_cape.py build_mp100_cape():
transforms = A.Compose([
    A.ColorJitter(...),     # Appearance only
    A.GaussNoise(...),      # Appearance only
    A.GaussianBlur(...),    # Appearance only
    A.Resize(512, 512),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
```

**Problem:**
- Model sees images only at original orientation
- `GeometricSupportEncoder` must be robust to rotation/scale
- The codebase comment "WHY NO GEOMETRIC AUGMENTATION" is **outdated** - Albumentations **does** correctly transform keypoints when `keypoint_params` is set

**Recommendation: Add Safe Geometric Augmentations**

```python
# In datasets/mp100_cape.py build_mp100_cape():
if image_set == 'train':
    import albumentations as A
    transforms = A.Compose([
        # ============================================================
        # GEOMETRIC AUGMENTATION (coordinates auto-updated by Albumentations)
        # ============================================================
        A.ShiftScaleRotate(
            shift_limit=0.1,       # ±10% shift
            scale_limit=0.15,      # ±15% scale
            rotate_limit=30,       # ±30° rotation
            border_mode=0,         # cv2.BORDER_CONSTANT (pad with zeros)
            value=0,               # Black padding
            p=0.7                  # Apply 70% of the time
        ),
        
        # Horizontal flip (most pose datasets are symmetric)
        A.HorizontalFlip(p=0.5),
        
        # ============================================================
        # APPEARANCE AUGMENTATION (existing)
        # ============================================================
        A.ColorJitter(
            brightness=0.3,        # Increase from 0.2 to 0.3
            contrast=0.3,
            saturation=0.3,
            hue=0.1,               # Increase from 0.05 to 0.1
            p=0.6
        ),
        
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # Final resize
        A.Resize(height=512, width=512),
        
    ], keypoint_params=A.KeypointParams(
        format='xy', 
        remove_invisible=False,
        label_fields=[]  # No labels to track
    ))
```

**Critical Note on Keypoint Dropping:**
The code already handles Albumentations dropping keypoints:
```python
if num_keypoints_after != num_keypoints_before:
    raise ImageNotFoundError(f"Albumentations dropped keypoints! ...")
```
With `remove_invisible=False`, keypoints outside image bounds are kept (marked invisible). The retry logic in `EpisodicDataset` will skip such samples.

---

### 3.3 Label Smoothing (MEDIUM IMPACT)

**Current State:**
```python
parser.add_argument('--label_smoothing', default=0.0, type=float)  # Disabled
```

**Problem:**
- Hard targets encourage overconfident predictions
- Model may memorize training data rather than generalize

**Recommendation:**
```python
parser.add_argument('--label_smoothing', default=0.1, type=float)  # Enable
```

The loss criterion already supports this:
```python
# In cape_losses.py
class CAPESetCriterion(SetCriterion):
    def __init__(self, ..., label_smoothing=0., ...):
        ...
```

---

### 3.4 Gradient Clipping (MEDIUM IMPACT)

**Current State:**
```python
parser.add_argument('--clip_max_norm', default=0.1, type=float)  # Very conservative
```

**Analysis:**
- `0.1` is standard for DETR/Transformer object detection
- For sequence prediction (like CAPE), slightly higher values can speed convergence
- From logs: Loss is stable, no gradient explosion signs

**Recommendation:**
```python
parser.add_argument('--clip_max_norm', default=0.5, type=float)  # Less aggressive
```

Try `1.0` if training remains stable. Monitor for NaN losses.

---

### 3.5 Dropout Tuning (LOW IMPACT)

**Current State:**
- All layers use `dropout=0.1`
- Support encoder: `dropout=0.1`
- Decoder layers: `dropout=0.1`

**Observation:**
The encoder processes support keypoints (small sequence, ~10-30 tokens), while decoder processes image features. Different dropout rates may help.

**Optional Tuning:**
```python
# In geometric_support_encoder.py
class GeometricSupportEncoder(nn.Module):
    def __init__(self, ..., dropout=0.15):  # Slightly higher for small sequences
        ...

# Keep decoder dropout at 0.1 (larger sequences)
```

---

## 4. Complete Code Changes

### 4.1 Scheduler Implementation (`train_cape_episodic.py`)

```python
# Add imports at top
from torch.optim.lr_scheduler import (
    MultiStepLR, 
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    LinearLR,
    SequentialLR
)

# Add arguments
parser.add_argument('--scheduler', default='cosine_warmrestarts',
                    choices=['multistep', 'cosine_warmrestarts', 'onecycle'],
                    help='LR scheduler type')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='Warmup epochs (0 to disable)')
parser.add_argument('--T_0', default=20, type=int,
                    help='Initial restart period for CosineAnnealingWarmRestarts')
parser.add_argument('--T_mult', default=2, type=int,
                    help='Period multiplier for CosineAnnealingWarmRestarts')
parser.add_argument('--eta_min', default=1e-6, type=float,
                    help='Minimum LR for cosine schedulers')

# Replace scheduler creation (around line 490)
def build_scheduler(optimizer, args, steps_per_epoch=None):
    """Build learning rate scheduler with optional warmup."""
    
    if args.scheduler == 'multistep':
        lr_drop_epochs = [int(x) for x in args.lr_drop.split(',')]
        main_scheduler = MultiStepLR(optimizer, lr_drop_epochs)
        
    elif args.scheduler == 'cosine_warmrestarts':
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min
        )
        
    elif args.scheduler == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        main_scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,  # Peak at 10x base LR
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,        # 10% warmup
            anneal_strategy='cos'
        )
        return main_scheduler  # OneCycleLR has built-in warmup
    
    # Add warmup wrapper if requested
    if args.warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs
        )
        return SequentialLR(
            optimizer,
            [warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    
    return main_scheduler

# Usage:
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))
```

### 4.2 Augmentation Update (`datasets/mp100_cape.py`)

```python
def build_mp100_cape(image_set, args):
    """Build MP-100 CAPE dataset with geometric augmentation."""
    
    # ... existing path setup ...
    
    if image_set == 'train':
        import albumentations as A
        
        transforms = A.Compose([
            # ========================================================
            # GEOMETRIC AUGMENTATION
            # Albumentations automatically updates keypoint coordinates
            # ========================================================
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,  # BORDER_CONSTANT
                value=0,        # Black padding
                p=0.7
            ),
            
            A.HorizontalFlip(p=0.5),
            
            # ========================================================
            # APPEARANCE AUGMENTATION
            # ========================================================
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.6
            ),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # Final resize
            A.Resize(height=512, width=512),
            
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))
    else:
        # Validation/Test: deterministic resize only
        import albumentations as A
        transforms = A.Compose([
            A.Resize(height=512, width=512)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    # ... rest of function ...
```

---

## 5. Recommended Training Command

```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.01 \
    --scheduler cosine_warmrestarts \
    --warmup_epochs 5 \
    --T_0 20 \
    --T_mult 2 \
    --eta_min 1e-6 \
    --clip_max_norm 0.5 \
    --label_smoothing 0.1 \
    --fixed_val_episodes \
    --early_stopping_patience 50 \
    --dataset_root . \
    --output_dir outputs/optimized_run
```

---

## 6. Monitoring Checklist

After implementing changes, monitor:

| Metric | Baseline | Target | Action if Not Met |
|--------|----------|--------|-------------------|
| Val PCK@0.2 | 30% | 45%+ | Check augmentation is working |
| Train/Val Loss Gap | High (overfitting) | Small | Increase weight_decay, add dropout |
| Gradient Norm | Stable | Stable | Reduce LR or increase clip_max_norm |
| LR Restart Effect | N/A | PCK jump after restart | Tune T_0, T_mult |

---

## 7. Summary

The most impactful changes are:

1. **CosineAnnealingWarmRestarts** - Actively escapes loss plateaus with LR restarts
2. **Weight Decay 1e-2** - Proper regularization for AdamW/Transformers
3. **Geometric Augmentation** - Critical for rotation/scale invariance

Implement these in order of priority. Expected combined improvement: **+10-20% PCK**.

