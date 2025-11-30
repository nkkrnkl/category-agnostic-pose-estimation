# Category-Agnostic Pose Estimation (CAPE)

This repository implements **Category-Agnostic Pose Estimation (CAPE)** using a geometric encoder for few-shot pose estimation on the MP-100 dataset. The model learns to predict 2D keypoints on unseen object categories using only a few support examples.

## Table of Contents

- [How to Run](#how-to-run) **Start here for execution instructions**
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
- [Overview](#overview)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [File Descriptions](#file-descriptions)

---

## How to Run

### Training

**Basic training command:**
```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --dataset_root . \
    --output_dir outputs/cape_training \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --mp100_split 1
```

**Key files for training:**
- **Main script**: `models/train_cape_episodic.py` - Run this to start training
- **Configuration**: `category_splits.json` - Defines train/val/test categories
- **Output**: Checkpoints saved to `--output_dir` (default: `outputs/cape_training/`)

**Quick test (1 epoch):**
```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --dataset_root . \
    --output_dir outputs/test_run \
    --epochs 1 \
    --mp100_split 1 \
    --batch_size 10 \
    --episodes_per_epoch 500
```

### Evaluation

**1-shot evaluation:**
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_training/checkpoint_best_pck_*.pth \
    --split test \
    --num-support-per-episode 1 \
    --dataset-root . \
    --output-dir outputs/eval_1shot
```

**5-shot evaluation:**
```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_training/checkpoint_best_pck_*.pth \
    --split test \
    --num-support-per-episode 5 \
    --dataset-root . \
    --output-dir outputs/eval_5shot
```

**Key files for evaluation:**
- **Main script**: `scripts/eval_cape_checkpoint.py` - Run this to evaluate a checkpoint
- **Output**: Metrics saved to `--output-dir` as `metrics_test.json` and visualizations in `visualizations/` folder

---

## Dependencies

### Installation

Install all required dependencies:

```bash
pip install -r requirements_cape.txt
```

### Required Libraries and Versions

All dependencies with specific versions are listed in `requirements_cape.txt`. Key libraries include:

| Library | Minimum Version | Purpose |
|---------|----------------|---------|
| `torch` | >= 2.0.0 | Deep learning framework |
| `torchvision` | >= 0.15.0 | Image processing utilities |
| `numpy` | >= 1.24.0 | Numerical computations |
| `scipy` | >= 1.10.0 | Scientific computing |
| `pycocotools` | >= 2.0.6 | COCO dataset format support |
| `opencv-python` | >= 4.7.0 | Image I/O and processing |
| `albumentations` | >= 1.3.0 | Data augmentation |
| `matplotlib` | >= 3.7.0 | Visualization |
| `tqdm` | >= 4.65.0 | Progress bars |

**Note**: For detectron2, install separately based on your CUDA version:
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

See `requirements_cape.txt` for the complete list with exact version specifications.

---

## Code Structure

### Main Entry Points

| File | Purpose | How to Run |
|------|---------|------------|
| `models/train_cape_episodic.py` | **Training script** | `python models/train_cape_episodic.py [args]` |
| `scripts/eval_cape_checkpoint.py` | **Evaluation script** | `python scripts/eval_cape_checkpoint.py --checkpoint [path] [args]` |
| `train_mp100_cape_cola.ipynb` | **Colab notebook** | Open in Google Colab and run cells sequentially |

### Core Components

| Directory/File | Purpose |
|----------------|---------|
| `models/train_cape_episodic.py` | Main training script - run this to train the model |
| `models/cape_model.py` | CAPE model wrapper combining base model + support encoder |
| `models/geometric_support_encoder.py` | Geometric encoder for support keypoints |
| `models/engine_cape.py` | Training and evaluation engine |
| `scripts/eval_cape_checkpoint.py` | Evaluation script - run this to evaluate a trained checkpoint |
| `datasets/mp100_cape.py` | MP-100 dataset loader |
| `datasets/episodic_sampler.py` | Episodic sampling for 1-shot/5-shot learning |
| `util/eval_utils.py` | PCK evaluation utilities |
| `category_splits.json` | Train/val/test category splits configuration |

---

## Overview

**CAPE** is a few-shot learning approach for pose estimation that:

- **Learns from support examples**: Uses 1 or 5 support images with ground truth keypoints
- **Generalizes to unseen categories**: Evaluates on categories never seen during training
- **Uses geometric encoder**: Encodes support keypoints using only coordinates and skeleton structure (no text)
- **Autoregressive prediction**: Generates keypoint sequences token-by-token

### Key Features

- **Geometric Support Encoder**: Encodes support pose graphs using coordinates + skeleton
- **1-shot and 5-shot**: Supports both 1-shot and 5-shot evaluation
- **Episodic meta-learning**: Trains on episodes with support + query pairs
- **PCK@0.2 evaluation**: Standard pose estimation metric

---

---

## Key Components

### 1. **Geometric Support Encoder** (`models/geometric_support_encoder.py`)
- Encodes support keypoints using **only coordinates and skeleton** (no text)
- Combines coordinate embeddings, positional encoding, optional GCN, and transformer attention
- Used when `--use_geometric_encoder` flag is enabled

### 2. **Episodic Sampler** (`datasets/episodic_sampler.py`)
- Creates episodes with support + query pairs
- Handles 1-shot (1 support) and 5-shot (5 supports, mean-pooled) sampling
- Ensures support and query are from same category but different images

### 3. **PCK Evaluator** (`util/eval_utils.py`)
- Computes PCK@0.2 (Percentage of Correct Keypoints)
- Reports both overall PCK and mean PCK across categories
- Handles visibility masking and bbox normalization

---

## Installation

### Dataset Setup

1. Download MP-100 dataset
2. Organize data in `data/` directory:
   ```
   data/
   ├── images/           # All category images
   └── annotations/       # JSON annotation files (mp100_split1_*.json)
   ```
3. Ensure `category_splits.json` is in the project root

---

## Quick Start

### 1. Train on Single Image (Debug/Overfitting Test)

Test that the model can learn by training on a single image:

```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --debug_single_image_path "bison_body/000000001120.jpg" \
    --epochs 50 \
    --output_dir outputs/single_image_debug \
    --dataset_root . \
    --batch_size 1 \
    --num_queries_per_episode 1
```

**Expected**: Training loss should drop to near-zero within ~10-20 epochs (perfect overfitting).

### 2. Full Training 

Train the model with 1-shot support:

```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --num_support_per_episode 1 \
    --epochs 300 \
    --batch_size 64 \
    --accumulation_steps 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --output_dir outputs/cape_1shot \
    --dataset_root . \
    --category_split_file category_splits.json
```


### 4. Evaluate (1-shot)

Evaluate a trained model with 1-shot:

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_1shot/checkpoint_best_pck.pth \
    --split val \
    --num-support-per-episode 1 \
    --output-dir outputs/eval_1shot \
    --dataset-root .
```

### 5. Evaluate (5-shot)

Evaluate a trained model with 5-shot:

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_5shot/checkpoint_best_pck.pth \
    --split val \
    --num-support-per-episode 5 \
    --output-dir outputs/eval_5shot \
    --dataset-root .
```

---

## Training

### Training Process

1. **Episodic Sampling**: Each episode contains:
   - **Support**: 1 or 5 images with ground truth keypoints
   - **Queries**: 2 query images (by default) from the same category

2. **Support Encoding**: 
   - Geometric encoder processes support keypoints (coordinates + skeleton)
   - For 5-shot: Support keypoints are mean-pooled before encoding

3. **Query Processing**:
   - Query images pass through ResNet backbone
   - Support features are fused with query features via cross-attention

4. **Sequence Generation**:
   - **Training**: Teacher forcing (uses GT sequences)
   - **Validation**: Autoregressive generation (predicts token-by-token)

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_geometric_encoder` | False | Enable geometric encoder (required) |
| `--num_support_per_episode` | 1 | Number of support images (1 or 5) |
| `--num_queries_per_episode` | 2 | Number of query images per episode |
| `--batch_size` | 64 | Episodes per batch |
| `--accumulation_steps` | 4 | Gradient accumulation (effective batch = 64×4) |
| `--epochs` | 300 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--lr_backbone` | 1e-5 | Backbone learning rate |
| `--early_stopping_patience` | 20 | Stop if PCK doesn't improve for N epochs |

### Single Image Training

For debugging/overfitting tests:

```bash
# By category ID
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --debug_single_image 40 \
    --epochs 50

# By exact image path
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --debug_single_image_path "category_name/image.jpg" \
    --epochs 50
```

**Note**: Single image mode uses the same image as both support and query (self-supervised).

---

## Evaluation

### Evaluation Metrics

- **PCK@0.2**: Percentage of keypoints within 20% of bbox diagonal
- **Overall PCK**: Micro-average (weighted by keypoint count)
- **Mean PCK**: Macro-average (equal weight per category)

### Evaluation Process

1. Load trained checkpoint
2. Run inference on validation/test set
3. Use **autoregressive generation** (no teacher forcing)
4. Compute PCK for each prediction
5. Aggregate metrics across categories

### Evaluation Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to trained model checkpoint |
| `--split` | Dataset split (`val` or `test`) |
| `--num-support-per-episode` | 1 or 5 (must match training) |
| `--num-visualizations` | Number of visualizations to generate |
| `--output-dir` | Directory to save results |

### Output Files

- `metrics.json`: Overall PCK, mean PCK, per-category PCK
- `visualizations/`: Side-by-side GT vs Predicted images
- Console output: Detailed PCK statistics

---

## Configuration

### Category Splits (`category_splits.json`)

Defines which categories are in train/val/test sets:

```json
{
  "train_categories": [1, 4, 5, ...],  // 69 categories
  "val_categories": [2, 3, 6, ...],     // 10 categories
  "test_categories": [7, 8, 9, ...]     // 20 categories
}
```

**Important**: Categories in val/test are **never seen during training** (true few-shot evaluation).

### Geometric Encoder Options

- `--use_geometric_encoder`: Enable geometric encoder (required)
- `--use_gcn_preenc`: Use GCN pre-encoding (optional)
- `--num_gcn_layers`: Number of GCN layers (if GCN enabled)

### Support Fusion Methods

- `cross_attention`: Cross-attention between support and query (default)
- `concat`: Concatenate support features
- `add`: Add support features to query features

---

## File Descriptions

### Core Training/Evaluation

| File | Purpose |
|------|---------|
| `models/train_cape_episodic.py` | Main training script |
| `scripts/eval_cape_checkpoint.py` | Evaluation script with visualization |
| `models/engine_cape.py` | Training/evaluation engine |

### Model Architecture

| File | Purpose |
|------|---------|
| `models/cape_model.py` | CAPE model wrapper (combines base + support encoder) |
| `models/geometric_support_encoder.py` | **Geometric encoder** (coordinates + skeleton) |
| `models/roomformer_v2.py` | Base Raster2Seq model |
| `models/backbone.py` | ResNet-50 backbone (ImageNet pretrained) |
| `models/graph_utils.py` | Graph utilities (GCN layers, adjacency matrices) |
| `models/positional_encoding.py` | Positional encoding for geometric encoder |

### Data Loading

| File | Purpose |
|------|---------|
| `datasets/mp100_cape.py` | MP-100 dataset loader |
| `datasets/episodic_sampler.py` | **Episodic sampling** (1-shot/5-shot) |
| `datasets/discrete_tokenizer.py` | Sequence tokenizer |
| `datasets/mp100_splits.py` | Category split utilities |

### Evaluation

| File | Purpose |
|------|---------|
| `util/eval_utils.py` | **PCK evaluation** (overall + mean PCK) |
| `util/misc.py` | Miscellaneous utilities |

### Configuration

| File | Purpose |
|------|---------|
| `category_splits.json` | **Train/val/test category splits** |

---

## Understanding the Code Flow

### Training Flow

```
1. Load dataset → datasets/mp100_cape.py
2. Create episodes → datasets/episodic_sampler.py
   - Sample support images (1 or 5)
   - Sample query images (2)
3. Forward pass → models/cape_model.py
   - Encode support → models/geometric_support_encoder.py
   - Encode query → models/backbone.py → models/roomformer_v2.py
   - Fuse support + query → cross-attention
   - Decode sequence → autoregressive generation
4. Compute loss → models/cape_losses.py
5. Backward pass → optimizer step
```

### Evaluation Flow

```
1. Load checkpoint → scripts/eval_cape_checkpoint.py
2. Create episodes → datasets/episodic_sampler.py
3. Autoregressive inference → models/cape_model.py
   - No teacher forcing (uses previous predictions)
4. Extract keypoints → models/engine_cape.py
5. Compute PCK → util/eval_utils.py
6. Generate visualizations → scripts/eval_cape_checkpoint.py
```

---

## Expected Results

### Single Image Training
- **Training loss**: Should drop to ~0 within 10-20 epochs
- **PCK**: Should reach ~100% (perfect overfitting)

### Full Training (1-shot)
- **Validation PCK@0.2**: ~30-50% (varies by category)
- **Mean PCK**: Typically higher than overall PCK

### Full Training (5-shot)
- **Validation PCK@0.2**: ~40-60% (better than 1-shot)
- **Mean PCK**: Higher than 1-shot

**Note**: Results depend on:
- Category difficulty
- Number of training epochs
- Hyperparameters
- Random seed

---

## Troubleshooting

### Common Issues

1. **"Category split file not found"**
   - Ensure `category_splits.json` is in project root
   - Or specify path with `--category_split_file`

2. **"CUDA out of memory"**
   - Reduce `--batch_size`
   - Increase `--accumulation_steps` to maintain effective batch size

3. **"PCK is 100% during validation"**
   - This indicates data leakage or using teacher forcing
   - Ensure evaluation uses `forward_inference()` (autoregressive)

4. **"No checkpoints found"**
   - Check `--output_dir` path
   - Ensure training completed successfully

---

## Notes

### 1-shot vs 5-shot

- **1-shot**: Uses 1 support image → encodes single pose graph
- **5-shot**: Uses 5 support images → mean-pools keypoints → encodes aggregated pose graph
- **Same model**: Can evaluate same checkpoint with different support counts

### Geometric Encoder

- **Input**: Support keypoints (coordinates) + skeleton edges
- **Output**: Encoded support features
- **No text**: Unlike CapeX, uses only geometric information

### Autoregressive Generation

- **Training**: Teacher forcing (uses GT sequences) → faster convergence
- **Evaluation**: Autoregressive (uses previous predictions) → realistic performance
- **Error compounding**: Errors accumulate in sequence generation

---

## References

- **MP-100 Dataset**: Few-shot pose estimation benchmark
- **Raster2Seq**: Base sequence-to-sequence framework
- **CapeX**: Inspiration for geometric encoder design

---

## Contact

For questions or issues, please refer to the project documentation or contact the maintainers.

---

**Last Updated**: 2025

