# Category-Agnostic Pose Estimation (CAPE) with Raster2Seq

This project implements a 1-shot category-agnostic pose estimation model using the Raster2Seq framework, trained and evaluated on the MP-100 dataset.

## Overview

**Task**: Given a query image and a single support example from the same category, predict the keypoint coordinates in the query image.

**Key Innovation**: Unlike CapeX which uses text descriptions, we use coordinate sequences as support data, adapting the Raster2Seq autoregressive sequence-to-sequence model for pose estimation.

---

## CAPE 1-Shot & Raster2Seq Integration – Recent Changes

This section documents critical fixes made to ensure correctness of the 1-shot CAPE setup.

### CRITICAL FIX #1: Keypoint-Edge Index Correspondence

#### What Was Wrong

**Problem**: The original implementation filtered out invisible keypoints (visibility == 0) before storing them:

```python
# OLD (INCORRECT)
visible_mask = visibility > 0
visible_kpts = kpts_array[visible_mask].tolist()
record["keypoints"] = visible_kpts  # Only visible keypoints!
```

**Why This Broke Training**:

1. **Index Misalignment**: Skeleton edges reference keypoint indices (e.g., `[[0,1], [1,2], [2,3]]`)
2. **Filtering Changes Indices**: If keypoint #1 is invisible and removed:
   ```
   Original:  [kpt_0, kpt_1(invisible), kpt_2, kpt_3]
   After:     [kpt_0, kpt_2, kpt_3]  ← Renumbered to [0, 1, 2]!
   Skeleton:  [[0,1], [1,2], [2,3]]  ← Still references original indices!
   ```
   Now edge `[0,1]` connects `kpt_0 → kpt_2` instead of `kpt_0 → kpt_1` (WRONG!)

3. **Impact**: 
   - Skeleton edges connected WRONG keypoints
   - Adjacency matrix built with incorrect structure
   - Model learned spurious correlations
   - PCK evaluation meaningless (predictions in wrong order)

#### The Fix

**Solution**: Keep ALL keypoints (including invisible ones) and use visibility as a **MASK**, not for filtering:

```python
# NEW (CORRECT)
record["keypoints"] = kpts_array.tolist()  # ALL keypoints!
record["visibility"] = visibility.tolist()  # Visibility as metadata
```

**How Invisibility Is Handled Now**:
- **Dataset**: Stores all keypoints with visibility flags
- **Tokenization**: Creates visibility_mask aligned with token sequence
- **Loss Computation** (`models/cape_losses.py`): Masks out invisible keypoints using `visibility_mask`
- **PCK Evaluation** (`util/eval_utils.py`): Ignores invisible keypoints using visibility flags
- **Support Encoder**: Receives all keypoints with proper skeleton edge alignment

**Files Changed**:
- `datasets/mp100_cape.py`: Lines 335-360 (removed filtering, added comprehensive comments)
- `datasets/mp100_cape.py`: Lines 522-630 (updated `_tokenize_keypoints` to use visibility as mask)

---

### CRITICAL FIX #2: Sequence Logic for Bilinear Interpolation

#### What Was Wrong

**Problem**: A previous "fix" (incorrectly labeled as #18) removed `seq21` and `seq22` from the dataset, thinking they were duplicates. They were NOT duplicates!

**Why This Broke Training**:

1. **Bilinear Interpolation Requires 4 Sequences**: Raster2Seq embeds coordinates using bilinear interpolation over a quantized grid. For each coordinate `(x, y)`, it needs 4 grid corners:
   ```
   (floor_x, floor_y)  → seq11, delta_x1, delta_y1
   (ceil_x,  floor_y)  → seq21, (1-delta_x1), delta_y1
   (floor_x, ceil_y)   → seq12, delta_x1, (1-delta_y1)
   (ceil_x,  ceil_y)   → seq22, (1-delta_x1), (1-delta_y1)
   ```

2. **Model Expected 4, Dataset Provided 2**: 
   - Dataset (after bad fix): Only `seq11`, `seq12`
   - Model decoder (`models/deformable_transformer_v2.py:1020-1023`): Expected all 4!
   ```python
   output = self._seq_embed(seq11=seq_kwargs['seq11'], seq12=seq_kwargs['seq12'], 
                            seq21=seq_kwargs['seq21'], seq22=seq_kwargs['seq22'], ...)
   ```
   → KeyError or incorrect embeddings!

3. **Impact**:
   - Training would crash or use wrong embeddings
   - Bilinear interpolation broken
   - Coordinate embeddings incorrect

#### The Fix

**Solution**: Restored all 4 sequences and all 4 deltas:

```python
# NEW (CORRECT) - dataset returns all 4 sequences
return {
    'seq11': seq11,  # (floor_x, floor_y)
    'seq21': seq21,  # (ceil_x, floor_y)
    'seq12': seq12,  # (floor_x, ceil_y)
    'seq22': seq22,  # (ceil_x, ceil_y)
    'delta_x1': delta_x1,  # x - floor_x
    'delta_x2': delta_x2,  # ceil_x - x (= 1 - delta_x1)
    'delta_y1': delta_y1,  # y - floor_y
    'delta_y2': delta_y2,  # ceil_y - y (= 1 - delta_y1)
    ...
}
```

**Why They're Not Duplicates**:
- For coordinate `(64.5, 32.7)` normalized to `(0.126, 0.064)`:
  - `seq11`: index at `(floor(0.126), floor(0.064))` = `(0, 0)`
  - `seq21`: index at `(ceil(0.126), floor(0.064))` = `(1, 0)` ← DIFFERENT!
  - `seq12`: index at `(floor(0.126), ceil(0.064))` = `(0, 1)` ← DIFFERENT!
  - `seq22`: index at `(ceil(0.126), ceil(0.064))` = `(1, 1)` ← DIFFERENT!

**Files Changed**:
- `datasets/mp100_cape.py`: Lines 572-600 (restored seq21, seq22, added comments)
- `datasets/mp100_cape.py`: Lines 669-682 (return dict includes all 4 sequences + deltas)

---

## Running the Code

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure MP-100 data is in ./data/mp100/
# Annotations in ./annotations/
```

### Training

```bash
python train_cape_episodic.py \
    --dataset_name mp100 \
    --data_root ./data/mp100 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 3 \
    --episodes_per_epoch 1000 \
    --epochs 300 \
    --lr 1e-4 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_1shot
```

**Key Arguments**:
- `--batch_size`: Number of episodes per batch (physical batch size)
- `--accumulation_steps`: Gradient accumulation steps (effective batch = batch_size × accumulation_steps)
- `--num_queries_per_episode`: K queries per support in each episode (1-shot setup)
- `--episodes_per_epoch`: Total episodes per epoch
- `--early_stopping_patience`: Stop if no improvement for N epochs (0 to disable)

### Evaluation

```bash
# Evaluate on unseen categories
python evaluate_unseen.py \
    --checkpoint ./outputs/cape_1shot/checkpoint_best_*.pth \
    --dataset_name mp100 \
    --data_root ./data/mp100 \
    --pck_threshold 0.2
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_critical_fix_1_index_correspondence.py -v
pytest tests/test_critical_fix_2_sequence_logic.py -v
```

**What Tests Cover**:
- **CRITICAL FIX #1 Tests**: Verify keypoint-edge index alignment, visibility masking, padding handling
- **CRITICAL FIX #2 Tests**: Verify all 4 sequences produced, bilinear interpolation weights correct, model compatibility
- **Checkpoint System Tests**: Verify checkpoint save/resume, RNG restoration, best-model tracking

---

## Checkpointing & Resume System

### Overview

The training script (`train_cape_episodic.py`) implements a robust checkpointing system that:
- ✅ Saves full training state every epoch
- ✅ Tracks **BOTH** best-validation-loss AND best-PCK models
- ✅ Supports safe resume with full state restoration
- ✅ Preserves RNG states for reproducible training
- ✅ Implements early stopping to prevent overfitting

### What Gets Saved in Checkpoints

Every checkpoint contains:

```python
{
    # Model & optimizer state
    'model': model.state_dict(),           # Model weights
    'optimizer': optimizer.state_dict(),   # Optimizer state (momentum, etc.)
    'lr_scheduler': lr_scheduler.state_dict(),  # Learning rate schedule
    'epoch': current_epoch,                # Current epoch number
    'args': training_args,                 # Full training configuration
    
    # Training metrics
    'train_stats': {...},                  # Training loss, metrics
    'val_stats': {...},                    # Validation loss, PCK, metrics
    
    # CRITICAL: Best-model tracking (for safe resume)
    'best_val_loss': best_val_loss,        # Best validation loss so far
    'best_pck': best_pck,                  # Best PCK metric so far
    'epochs_without_improvement': N,       # Early stopping counter
    
    # CRITICAL: RNG states (for reproducibility)
    'rng_state': torch.get_rng_state(),    # PyTorch RNG
    'cuda_rng_state': torch.cuda.get_rng_state_all(),  # CUDA RNG (if GPU)
    'np_rng_state': np.random.get_state(), # NumPy RNG
    'py_rng_state': random.getstate(),     # Python RNG
}
```

### Checkpoint Types

The system saves **THREE** types of checkpoints:

#### 1. Regular Epoch Checkpoints
**Filename**: `checkpoint_e{epoch:03d}_lr{lr}_bs{batch_size}_acc{acc_steps}_qpe{queries_per_ep}.pth`

**Example**: `checkpoint_e050_lr1e-4_bs2_acc4_qpe2.pth`

**Saved**: Every epoch

**Purpose**: Resume training from any point

#### 2. Best Validation Loss Checkpoint
**Filename**: `checkpoint_best_loss_e{epoch:03d}_valloss{loss:.4f}_pck{pck:.4f}.pth`

**Example**: `checkpoint_best_loss_e042_valloss0.0987_pck0.7654.pth`

**Saved**: When validation loss improves (lowest so far)

**Purpose**: Early stopping criterion, best generalization

**Use for**: Most training scenarios (lowest loss often means best model)

#### 3. Best PCK Checkpoint
**Filename**: `checkpoint_best_pck_e{epoch:03d}_pck{pck:.4f}_valloss{loss:.4f}.pth`

**Example**: `checkpoint_best_pck_e067_pck0.8123_valloss0.1234.pth`

**Saved**: When PCK metric improves (highest so far)

**Purpose**: Best pose estimation performance

**Use for**: Final evaluation, deployment (highest accuracy)

### Resuming Training

To resume training from a checkpoint:

```bash
python train_cape_episodic.py \
    --resume ./outputs/cape_1shot/checkpoint_e050_lr1e-4_bs2_acc4_qpe2.pth \
    [... other args ...]
```

**What Gets Restored**:
- ✅ **Model weights** (exact state from checkpoint epoch)
- ✅ **Optimizer state** (momentum buffers, learning rate)
- ✅ **LR scheduler** (continues LR schedule from checkpoint epoch)
- ✅ **Epoch number** (training continues from `checkpoint['epoch'] + 1`)
- ✅ **Best model tracking** (preserves `best_val_loss`, `best_pck`)
- ✅ **Early stopping counter** (preserves `epochs_without_improvement`)
- ✅ **RNG states** (reproducible data sampling and initialization)

**CRITICAL**: The resume logic restores `best_val_loss` and `best_pck` from the checkpoint to prevent incorrectly overwriting the best model with a worse one. This was a critical bug fix.

### Example Training Session

```bash
# Start training
python train_cape_episodic.py \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_run1

# Training runs for 100 epochs, then crashes or is interrupted
# Checkpoints saved:
#   - checkpoint_e100_lr1e-4_bs2_acc4_qpe2.pth (latest)
#   - checkpoint_best_loss_e067_valloss0.0987_pck0.7654.pth (best loss at epoch 67)
#   - checkpoint_best_pck_e089_pck0.8123_valloss0.1234.pth (best PCK at epoch 89)

# Resume from epoch 100
python train_cape_episodic.py \
    --resume ./outputs/cape_run1/checkpoint_e100_lr1e-4_bs2_acc4_qpe2.pth \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_run1

# Training continues from epoch 101
# Best-model tracking correctly uses best_val_loss=0.0987 from checkpoint
# Will NOT overwrite best checkpoint unless validation loss < 0.0987
```

### Early Stopping

The system tracks `epochs_without_improvement` (how many epochs since last improvement in **PCK**).

**Behavior** (PCK-based for pose estimation):
- If `val_pck > best_pck`: Reset counter to 0, save new best-PCK checkpoint
- If `val_pck <= best_pck`: Increment counter
- If `counter >= patience`: Stop training (saves compute, prevents overfitting)

**Why PCK instead of loss?**
- PCK directly measures pose estimation accuracy (keypoint correctness)
- Validation loss is a proxy metric that can diverge from PCK
- Training can stop early on loss while PCK is still improving
- For pose estimation, we care about keypoint accuracy, not loss magnitude

**Configure**:
```bash
--early_stopping_patience 20  # Stop if PCK doesn't improve for 20 epochs
--early_stopping_patience 0   # Disable early stopping (train all epochs)
```

**Output**:
```
Epoch 142/300
  → No improvement in PCK for 20 epoch(s)
     Best PCK:  0.8123 | Current: 0.7890
     Best loss: 0.0987 | Current: 0.1234

================================================================================
Early stopping triggered!
No improvement in PCK for 20 epochs.
Best PCK: 0.8123 (epoch 122)
Best validation loss: 0.0987
================================================================================
Early stopping saved 158 epochs of compute time!
```

### Best Model Selection Strategy

**For pose estimation (recommended)**: Use `checkpoint_best_pck_*.pth`
- Highest PCK = best keypoint accuracy
- Used as early stopping criterion
- What we actually optimize for

**For general ML tasks**: Use `checkpoint_best_loss_*.pth`
- Lowest validation loss
- Better generalization in some cases
- More stable than PCK on small validation sets

**For deployment/evaluation**: Compare both!
```bash
# Evaluate best-loss model
python evaluate_cape.py --checkpoint checkpoint_best_loss_*.pth

# Evaluate best-PCK model
python evaluate_cape.py --checkpoint checkpoint_best_pck_*.pth

# Pick whichever performs better on held-out test set
```

**Why track both?**
- Loss and PCK can diverge (model minimizes loss but doesn't maximize keypoint accuracy)
- Best-loss model may have slightly worse PCK, but better generalization
- Best-PCK model may overfit but have highest pose accuracy
- Having both gives you options

### Reproducibility

RNG states are saved in every checkpoint, ensuring:
- ✅ Same data augmentation
- ✅ Same episode sampling
- ✅ Same dropout patterns
- ✅ Same initialization (if resuming mid-epoch, though not common)

**Important**: To fully reproduce training from scratch:
1. Use same `--seed` argument
2. Use same data ordering (same dataset version)
3. Same hardware/software (GPU model, PyTorch version)

Resume is guaranteed to produce identical results to uninterrupted training.

### Disk Space Management

**Checkpoint size**: ~500MB per checkpoint (model + optimizer + metadata)

**Storage for 300 epochs**:
- Regular checkpoints: 300 × 500MB = ~150GB
- Best-loss checkpoints: N × 500MB (one per improvement)
- Best-PCK checkpoints: M × 500MB (one per improvement)

**Recommendation**: After training, delete old regular checkpoints, keep only:
- Last checkpoint (for potential resume)
- All best-loss checkpoints (track improvement history)
- All best-PCK checkpoints (track improvement history)

```bash
# Clean up old regular checkpoints (keep last 5)
cd ./outputs/cape_run1
ls -t checkpoint_e*.pth | tail -n +6 | xargs rm

# Keep all best-* checkpoints for analysis
```

### Automated Tests

The checkpoint system is thoroughly tested (`tests/test_checkpoint_system.py`):

**Test 1**: Checkpoint Contains Expected Fields
- Verifies all required fields present (model, optimizer, RNG states, best metrics)

**Test 2**: Resume Restores Full State
- Verifies model weights, optimizer state, LR scheduler restored correctly
- Verifies RNG states restored for reproducibility
- Verifies best metrics restored to prevent checkpoint overwrite bug

**Test 3**: PCK-Based Saving
- Verifies best-PCK checkpoint saved when PCK improves
- Verifies best-loss and best-PCK tracked independently

**Test 4**: Best Checkpoint Not Overwritten After Resume
- **CRITICAL**: Verifies resuming doesn't overwrite best checkpoint with worse model
- Simulates resume scenario, ensures `best_val_loss` restored correctly

**Run tests**:
```bash
pytest tests/test_checkpoint_system.py -v

# Or standalone (without pytest)
python3 tests/test_checkpoint_system.py
```

---

## Data Augmentation Strategy

### Appearance-Only Augmentation

To improve model robustness without modifying keypoint annotations, we apply **appearance-only augmentations** during training.

#### ✅ Applied Augmentations (Safe - No Geometric Changes)

| Augmentation | Parameters | Probability | Purpose |
|--------------|-----------|-------------|---------|
| **Color Jitter** | brightness=±20%<br>contrast=±20%<br>saturation=±20%<br>hue=±5% | 60% | Robustness to lighting conditions |
| **Gaussian Noise** | variance=5-25<br>mean=0 | 40% | Robustness to sensor noise |
| **Gaussian Blur** | kernel=3×3 or 5×5<br>sigma=auto | 20% | Robustness to focus variations |

#### ❌ Explicitly Avoided (Would Require Annotation Updates)

We **DO NOT** use any geometric augmentations:

- ❌ **Random Crop**: Would change keypoint coordinates
- ❌ **Horizontal/Vertical Flip**: Would swap left/right keypoints
- ❌ **Rotation/Affine/Perspective**: Would change keypoint positions
- ❌ **Scale/Resize** (beyond deterministic 512×512): Would change coordinates
- ❌ **Random Erasing/Cutout**: Would change visibility flags

#### Why Appearance-Only?

**Problem**: Geometric augmentations (flip, rotation, crop) change pixel positions, which requires updating keypoint coordinates accordingly. Implementing this correctly is complex and error-prone.

**Solution**: Use only appearance transforms that modify pixel values but preserve spatial relationships. This guarantees:

1. ✅ **Keypoint annotations remain BITWISE IDENTICAL** (no modification)
2. ✅ **Skeleton connectivity preserved** (indices unchanged)
3. ✅ **No risk of annotation-coordinate mismatch**
4. ✅ **Still improves generalization** through photometric variation

#### Training vs Validation/Test

```python
# Training: Appearance augmentation enabled
transforms = [ColorJitter, GaussNoise, GaussianBlur, Resize(512)]

# Validation/Test: Only deterministic resize
transforms = [Resize(512)]
```

#### Verification Tests

Run tests to verify augmentation correctness:

```bash
# Run augmentation tests
pytest tests/test_appearance_augmentation.py -v
```

**Tests verify:**
- ✅ Keypoints remain unchanged (bitwise identical) across augmented samples
- ✅ Images DO change (augmentation is active)
- ✅ Training uses augmentation, validation/test do not
- ✅ Bounding boxes and image shapes remain unchanged

---

## Architecture

### Data Flow (1-Shot Episodic Training)

```
Episode:
  Support: (image, keypoints, skeleton, category)
  Queries: [(image, keypoints), ...] from same category

    ↓
    
Dataset (mp100_cape.py):
  1. Crop image to bbox → resize to 512×512
  2. Adjust keypoints to bbox coords → scale to 512×512
  3. Keep ALL keypoints (CRITICAL FIX #1)
  4. Tokenize to sequences (seq11, seq21, seq12, seq22) (CRITICAL FIX #2)
  5. Create visibility_mask from visibility flags

    ↓

Episodic Sampler:
  1. Sample 1 support + K queries per episode
  2. Repeat support K times to match query batch size (B×K)
  3. Normalize coordinates to [0,1]
  4. Stack into batch

    ↓

Model (CAPEModel):
  1. Encode support pose graph with skeleton edges (SupportPoseGraphEncoder)
  2. Encode query image (ResNet-50)
  3. Autoregressively decode keypoint sequence with:
     - Cross-attention to query image features
     - Cross-attention to support pose embeddings
  4. Predict token types + coordinates

    ↓

Loss (CAPESetCriterion):
  - Token classification loss (NLL) with visibility masking
  - Coordinate regression loss (L1) with visibility masking
  
    ↓

Evaluation (PCK@0.2):
  - Extract predicted keypoints from sequence
  - Compute distance to ground truth (in pixels)
  - Normalize by original bbox diagonal
  - Threshold at α=0.2
  - Mask out invisible keypoints using visibility
```

### Key Components

**Support Pose Graph Encoder** (`models/support_encoder.py`):
- Embeds support coordinates (x, y) → hidden_dim
- Builds adjacency matrix from skeleton edges
- Aggregates edge embeddings (connectivity info)
- Combines coordinate + edge features
- Processes through transformer encoder → contextual embeddings

**CAPE Model Wrapper** (`models/cape_model.py`):
- Wraps base Raster2Seq model (RoomFormerV2)
- Injects support embeddings into decoder
- Verifies support-query batch alignment (1-shot structure)

**Base Raster2Seq** (`models/roomformer_v2.py`):
- Image encoder (ResNet-50 + FPN)
- Autoregressive decoder (Deformable Transformer)
- Bilinear coordinate embedding (uses all 4 sequences)
- Causal masking for autoregression

**Loss Functions** (`models/cape_losses.py`):
- `CAPESetCriterion`: Extends base `SetCriterion`
- Overrides `loss_labels()` and `loss_polys()` with visibility masking
- Ensures loss only computed on visible keypoints

**PCK Evaluation** (`util/eval_utils.py`):
- `compute_pck_bbox()`: Single-instance PCK with bbox normalization
- `PCKEvaluator`: Accumulates and reports overall/per-category/mean PCK
- Handles visibility masking and bbox diagonal normalization

---

## Project Structure

```
category-agnostic-pose-estimation/
├── datasets/
│   ├── mp100_cape.py          # MP-100 dataset loader (CRITICAL FIX #1 & #2)
│   ├── episodic_sampler.py    # 1-shot episodic sampling
│   ├── tokenizer.py           # Discrete tokenizer (bilinear)
│   └── token_types.py         # Token type definitions
├── models/
│   ├── cape_model.py          # CAPE wrapper with support conditioning
│   ├── cape_losses.py         # CAPE-specific losses with visibility masking
│   ├── support_encoder.py     # Support pose graph encoder (with skeleton edges)
│   ├── roomformer_v2.py       # Base Raster2Seq model
│   └── deformable_transformer_v2.py  # Decoder (uses 4 sequences)
├── util/
│   └── eval_utils.py          # PCK evaluation utilities
├── engine_cape.py             # Training and evaluation loops
├── train_cape_episodic.py     # Main training script
├── tests/
│   ├── test_critical_fix_1_index_correspondence.py
│   └── test_critical_fix_2_sequence_logic.py
├── papers/                    # Project proposals and references
├── data/                      # MP-100 dataset (images)
├── annotations/               # COCO-format annotations
└── README.md                  # This file
```

---

## Metrics

**Primary Metric**: PCK@0.2 (Percentage of Correct Keypoints)
- Distance threshold: α = 0.2 × bbox_diagonal
- Bbox diagonal: `sqrt(bbox_width² + bbox_height²)` (original bbox, not 512×512)
- Computed per instance, averaged per category, then averaged overall
- Ignores invisible keypoints (visibility == 0)

**Reported Metrics**:
- Overall PCK (mean across all instances)
- Per-category PCK (breakdown by object category)
- Mean category PCK (average per-category PCK, for category-level analysis)

---

## Known Limitations & Future Work

### Current Limitations

1. **Multi-Instance Images**: Currently uses only the first instance per image. If an image has multiple objects (e.g., 2 people), only the first is used.
   - **Impact**: ~10% of MP-100 images have 2+ instances, so we're using ~90% of potential data.
   - **Mitigation**: MP-100 mostly has single-instance images, so impact is minimal for this dataset.
   - **See**: `MULTI_INSTANCE_LIMITATION.md` for details and upgrade path.

2. **Fixed Input Size**: Images resized to 512×512 regardless of aspect ratio.
   - **Impact**: May distort non-square objects.
   - **Mitigation**: Bbox cropping removes much background, so distortion is reduced.

3. **No Category-Specific Ordering**: Keypoint sequences follow COCO annotation order, not a canonical semantic order.
   - **Impact**: Same keypoint may appear at different positions in different images.
   - **Mitigation**: Positional encoding and attention help model learn correspondences.

### Optional Improvements (Not Yet Implemented)

See "Optional Improvements" section below for full list.

---

## References

- **Raster2Seq**: Yuanwen Yue, "Raster2Seq: Deep Learning for Floorplan Reconstruction"
- **MP-100**: Multi-Category Pose estimation dataset (100 categories)
- **CapeX**: Text-based category-agnostic pose estimation (uses text descriptions instead of coordinates)
- **POMNet**: Pose-guided Matching Network for few-shot pose estimation

---

## Contact

For questions or issues, please refer to the project repository or contact the development team.

---

## Changelog

### 2025-11-24: CRITICAL FIX #1 & #2

- **CRITICAL FIX #1**: Preserved all keypoints to maintain skeleton edge alignment
  - Removed visibility-based filtering that broke index correspondence
  - Use visibility as mask in loss/eval instead
  - Files: `datasets/mp100_cape.py`

- **CRITICAL FIX #2**: Restored all 4 sequences for bilinear interpolation
  - Re-added seq21, seq22, delta_x2, delta_y2 (they're NOT duplicates!)
  - Ensures model can perform correct bilinear coordinate embedding
  - Files: `datasets/mp100_cape.py`

- **Tests Added**:
  - `tests/test_critical_fix_1_index_correspondence.py`: Verify keypoint-edge alignment
  - `tests/test_critical_fix_2_sequence_logic.py`: Verify bilinear interpolation sequences

### Previous Changes

- Implemented 1-shot episodic training
- Added support pose graph encoder with skeleton edges
- Implemented gradient accumulation, data augmentation, early stopping
- Added PCK@bbox evaluation
- Fixed support-query dimension alignment
- Fixed validation dataset leakage
- Implemented visibility masking in loss
- Separated CAPE-specific losses from base model

