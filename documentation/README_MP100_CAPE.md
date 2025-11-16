# MP-100 Category-Agnostic Pose Estimation (CAPE) with Raster2Seq

This project adapts the Raster2Seq framework for Category-Agnostic Pose Estimation on the MP-100 dataset, following the project proposal.

## Overview

**Goal**: Extend the Raster2Seq autoregressive framework to perform category-agnostic pose estimation using only:
1. A query image (of any object category)
2. A pose graph represented as a **keypoint sequence** (2D coordinates)

**Key Distinction from CapeX**: Instead of using textual descriptions of keypoints, we directly utilize 2D coordinate sequences as support data.

## Dataset

- **Name**: MP-100 (subset with 49 categories)
- **Format**: COCO-style annotations with keypoints
- **Splits**: 5-fold cross-validation (splits 1-5)
- **Location**:
  - Images: `theodoros/data/`
  - Annotations: `annotations/mp100_split{1-5}_{train,val,test}.json`

### Dataset Statistics

From the category coverage report:
- **Total categories**: 100 (original MP-100)
- **Available categories**: 49 (with images)
- **Missing categories**: 51 (no images)
- **Total images**: 40,528 across all splits
- **Total annotations**: 45,897 keypoint annotations

Categories with good coverage (≥10 images):
- Furniture: chair (966), table (965), sofa (964), bed (962)
- Birds: Gull (965), Sparrow (965), Warbler (965), Wren (962)
- Animals: Various body keypoints (600-900 images each)

## Implementation

### Files Created

1. **`datasets/mp100_cape.py`**: MP-100 dataset loader for CAPE
   - Loads images and keypoint annotations in COCO format
   - Converts keypoints to sequences for autoregressive generation
   - Uses discrete tokenizer for coordinate quantization

2. **`train_mp100_cape.py`**: Training script for MP-100 CAPE
   - Configured for pose estimation task
   - Supports 5-fold cross-validation
   - Includes WandB logging support

3. **`test_mp100_loading.py`**: Test script to verify dataset loading

### Key Adaptations from Raster2Seq

| Component | Original (Floorplan) | Adapted (CAPE) |
|-----------|---------------------|----------------|
| **Input** | Rasterized floorplan image | Query image (RGB, any object) |
| **Output** | Polygon sequences (room boundaries) | Keypoint sequences (pose) |
| **Support Data** | None or template | Pose graph (keypoint coordinates) |
| **Sequence Format** | `[x1,y1], [x2,y2], ... <SEP> [x1,y1], ...` | `[x1,y1], [x2,y2], ... <EOS>` |
| **Semantic Classes** | Room types (16-19) | Object categories (49) |
| **Max Sequence Length** | 800 (20 rooms × 40 corners) | 200 (100 keypoints × 2 coords) |

## Setup Instructions

### 1. Install Dependencies

```bash
cd "/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project"

# Activate virtual environment
source venv/bin/activate

# Install PyTorch (choose appropriate version for your system)
pip install torch torchvision

# Install other requirements
pip install -r theodoros/requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install pycocotools pillow opencv-python numpy scipy
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install wandb  # Optional, for logging
pip install albumentations  # For image transforms
```

### 2. Verify Dataset Structure

Ensure your directory structure looks like this:

```
Project/
├── theodoros/
│   ├── data/                    # Images organized by category
│   │   ├── cat_body/
│   │   ├── dog_body/
│   │   ├── horse_body/
│   │   └── ...
│   ├── annotations/             # COCO format annotations
│   │   ├── mp100_split1_train.json
│   │   ├── mp100_split1_val.json
│   │   ├── mp100_split1_test.json
│   │   └── ...
│   ├── datasets/
│   │   ├── mp100_cape.py
│   │   └── ...
│   ├── models/
│   ├── train_mp100_cape.py
│   └── test_mp100_loading.py
└── annotations/                 # Cleaned annotations (symlink or copy)
    └── mp100_split*.json
```

### 3. Test Dataset Loading

```bash
cd theodoros
python test_mp100_loading.py
```

Expected output:
```
================================================================================
Testing MP-100 CAPE Dataset Loading
================================================================================

Dataset root: /path/to/Project/theodoros
Split: 1

--------------------------------------------------------------------------------
Loading TRAIN dataset...
--------------------------------------------------------------------------------
✓ Train dataset loaded: XXXX samples
✓ Sample loaded successfully
  - Image shape: torch.Size([3, 256, 256])
  - Num keypoints: XX
  - Category ID: X
```

### 4. Train the Model

#### Basic Training (Debug Mode)

```bash
cd theodoros
python train_mp100_cape.py \
    --debug \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 1 \
    --epochs 10 \
    --output_dir output/debug
```

#### Full Training (Single Split)

```bash
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 4 \
    --epochs 300 \
    --lr 1e-4 \
    --lr_drop 200,250 \
    --image_norm \
    --use_anchor \
    --dec_layer_type v5 \
    --output_dir output/split1 \
    --job_name mp100_split1 \
    --num_workers 4
```

#### With WandB Logging

```bash
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 4 \
    --image_norm \
    --use_wandb \
    --wandb_project "MP100-CAPE" \
    --output_dir output/split1_wandb
```

#### 5-Fold Cross-Validation

Train on all 5 splits:

```bash
for split in 1 2 3 4 5; do
    python train_mp100_cape.py \
        --dataset_root . \
        --mp100_split $split \
        --batch_size 4 \
        --epochs 300 \
        --image_norm \
        --output_dir output/split${split} \
        --job_name mp100_split${split}
done
```

## Model Architecture

The model uses the anchor-based autoregressive decoder from Raster2Seq:

```
Query Image → CNN Encoder (ResNet50)
                    ↓
              Feature Maps (multi-scale)
                    ↓
          Deformable Transformer Encoder
                    ↓
              Image Features (memory)

Pose Graph → Discrete Tokenizer → Keypoint Embeddings
            (coordinate quantization)      ↓
                               Autoregressive Decoder
                            (with learnable anchors)
                                     ↓
                          Predicted Keypoints

Components:
- Learnable Anchors: Guide attention to informative regions
- Deformable Attention: Focus on sparse keypoint locations
- Masked Self-Attention: Causal mask for autoregressive generation
- FeatFusion: Concatenate image features in self-attention
```

## Training Configuration

### Recommended Hyperparameters

```python
# Model
--backbone resnet50
--hidden_dim 256
--nheads 8
--enc_layers 6
--dec_layers 6
--dec_layer_type v5  # or v6 for pooled image features
--num_feature_levels 4

# Sequence
--seq_len 200         # Max keypoints × 2
--vocab_size 2000     # Discrete coordinate bins
--num_queries 200

# Training
--batch_size 4
--lr 1e-4
--lr_backbone 1e-5
--epochs 300
--lr_drop 200,250

# Loss weights
--coords_loss_coef 5.0
--cls_loss_coef 2.0
--room_cls_loss_coef 0.5  # Category classification

# Features
--image_norm          # Normalize images with ImageNet stats
--use_anchor          # Use learnable anchors
--with_poly_refine    # Iterative coordinate refinement
--aux_loss            # Multi-layer supervision
```

## Evaluation

To evaluate a trained model:

```bash
# TODO: Create eval_mp100_cape.py script
python eval_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --resume output/split1/checkpoint.pth \
    --eval
```

### Metrics

Following the project proposal, compare against ≥3 CAPE baselines:
- **Quantitative**: PCK (Percentage of Correct Keypoints), mAP
- **Qualitative**: Visualization of predicted keypoints

## Expected Results

Based on the project proposal, you should:

1. **Train** the model on MP-100 using 5-fold cross-validation
2. **Compare** quantitatively against CAPE baselines (CapeX, GraphCape, etc.)
3. **Demonstrate** effectiveness through qualitative visualizations
4. **Analyze** how the model handles:
   - Novel object categories (zero-shot)
   - Varying numbers of keypoints
   - Different pose complexities

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--batch_size` or `--hidden_dim`
2. **Slow Training**: Increase `--num_workers` or use smaller `--backbone`
3. **Poor Convergence**: Try different `--dec_layer_type` or adjust learning rates
4. **Dataset Not Found**: Check paths in `--dataset_root`

### Debug Mode

Always test with `--debug` first to overfit a single sample:

```bash
python train_mp100_cape.py --debug --epochs 100
```

This should achieve near-perfect keypoint prediction on the single sample.

## Next Steps

1. **Create evaluation script** (`eval_mp100_cape.py`)
2. **Implement visualization** for predicted keypoints
3. **Add baseline comparisons** (CapeX, GraphCape, EdgeCape)
4. **Fine-tune hyperparameters** based on validation performance
5. **Write final report** with quantitative/qualitative results

## References

- Raster2Seq Paper: [Link to paper]
- CapeX Paper: [Link to arXiv:2406.00384]
- MP-100 Dataset: Pose for Everything (ECCV 2022)
- Project Proposal: `papers/ProjectProposal_5854619_5752994_5854229.pdf`

## Contact

For questions about the implementation, refer to the project proposal or contact the course instructor.
