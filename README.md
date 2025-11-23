# Raster2Seq for Category-Agnostic Pose Estimation

Implementation of the Raster2Seq autoregressive framework for category-agnostic pose estimation (CAPE) on the MP-100 dataset.

## Project Structure

```
project_root/
├── dataset.py                 # MP-100 COCO-format dataset loader
├── episodic_sampler.py        # Episodic meta-learning sampler
├── train.py                   # Training script
├── eval.py                    # Evaluation script (1-shot)
├── model/
│   ├── __init__.py
│   ├── encoders.py           # ResNet-50 query + Transformer support encoder
│   ├── decoder.py            # Autoregressive Transformer decoder
│   ├── heads.py              # Token classification + coordinate regression heads
│   └── model.py              # Complete Raster2Seq model
├── utils/
│   ├── __init__.py
│   ├── geometry.py           # Coordinate transformations and PCK
│   ├── masking.py            # Keypoint masking utilities
│   └── logging.py            # Training metrics logging
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   └── split_1.json          # Category splits
├── annotations/              # MP-100 annotation files
├── data/                     # MP-100 images
├── checkpoints/              # Saved models
└── logs/                     # Training logs
```

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow pyyaml tqdm
```

## Dataset

The MP-100 dataset should be organized as:
- `annotations/mp100_split{1-5}_{train,val,test}.json` - COCO-format annotations
- `data/` - Image directory with subdirectories per category

## Training

### Quick Test (5 epochs):
```bash
python train.py --config configs/default.yaml --preliminary
```

### Full Training (300 epochs):
```bash
python train.py --config configs/default.yaml
```

### Training Options:
- `--config`: Path to config file (default: configs/default.yaml)
- `--preliminary`: Run quick 5-epoch test
- `--resume`: Path to checkpoint to resume from

### Configuration

Edit `configs/default.yaml` to modify:
- Batch size (default: 4)
- Learning rate (default: 1e-4)
- Number of epochs (default: 300)
- Model architecture (hidden_dim, num_layers, etc.)
- Data augmentation settings

## Evaluation

Evaluate a trained model on the test set:

```bash
python eval.py --checkpoint checkpoints/raster2seq_cape_split1_final.pth --split test --output results.json
```

### Evaluation Options:
- `--checkpoint`: Path to model checkpoint (required)
- `--config`: Path to config file (default: configs/default.yaml)
- `--split`: Dataset split to evaluate on (train/val/test)
- `--output`: Output JSON file for results

### Evaluation Metrics

The evaluation computes:
- **Mean PCK@0.2**: Average PCK across all instances
- **Mean Category PCK@0.2**: Average PCK across categories (treats each category equally)
- **Per-category PCK**: Individual PCK scores for each category

## Model Architecture

### 1. Query Image Encoder
- ResNet-50 pretrained on ImageNet
- Outputs: [B, 2048, 16, 16] feature maps

### 2. Support Pose Encoder
- Linear embedding of 2D coordinates
- Learned keypoint ID embeddings
- 3-layer Transformer encoder (256-dim, 8 heads)

### 3. Autoregressive Decoder
- 6-layer Transformer decoder (256-dim, 8 heads)
- Causal self-attention
- Cross-attention to image features
- Cross-attention to support embeddings

### 4. Prediction Heads
- Token classification head: predicts `<coord>`, `<sep>`, `<eos>`
- Coordinate regression head: predicts (x, y) in [0, 1]²

## Training Details

### Episodic Meta-Learning
- Each episode contains 1 support + 1 query from the same category
- Support provides keypoint structure
- Query is predicted autoregressively

### Loss Function
```
L_total = L_token + λ * L_coord
```
- `L_token`: Cross-entropy for special tokens
- `L_coord`: L1 loss for coordinates (masked by visibility)
- λ = 5.0

### Data Augmentation
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- Random crop (consistent for image + keypoints)

### Optimizer
- AdamW with lr=1e-4, weight_decay=1e-4
- Cosine learning rate schedule
- Gradient clipping (max_norm=1.0)

## Inference (1-Shot)

The model performs 1-shot pose estimation:

**Input:**
- Query image (novel instance)
- Support coordinates (from another instance of the same category)
- Category skeleton (edge list)

**Output:**
- Predicted keypoints for the query image

**Process:**
1. Encode query image with ResNet-50
2. Encode support coordinates with Transformer
3. Autoregressively generate keypoints:
   - Predict `<coord>` token
   - Predict (x, y) coordinates
   - Predict `<sep>` separator
   - Repeat until `<eos>`

## Key Features

✓ **Pure PyTorch** - No Lightning or external frameworks
✓ **Modular Design** - Clean separation of components
✓ **Variable Keypoints** - Handles categories with different #keypoints via padding
✓ **Visibility Masking** - Properly handles occluded/unlabeled keypoints
✓ **Teacher Forcing** - Used during training for stability
✓ **Autoregressive Generation** - No teacher forcing during inference
✓ **PCK Evaluation** - Standard CAPE evaluation metric

## Expected Results

After training:
- Training loss should decrease steadily
- Coordinate loss typically converges faster than token loss
- PCK@0.2 on seen categories: ~60-70% (depends on hyperparameters)
- PCK@0.2 on unseen categories: ~40-50% (generalization)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in configs/default.yaml
- Reduce `image_size` (e.g., 256 instead of 512)
- Reduce `hidden_dim` or `num_decoder_layers`

### Slow Training
- Reduce `num_episodes_per_epoch`
- Use more workers: set `num_workers` > 0 (may require fixing dataloader)
- Use mixed precision training (add to train.py)

### Poor Performance
- Ensure data augmentation is enabled
- Check that teacher forcing is ON during training
- Verify loss masking is working correctly
- Try longer training (300 epochs)
- Adjust `coord_loss_weight`

## Citation

This implementation is based on the Raster2Seq framework adapted for category-agnostic pose estimation.

```bibtex
@article{mp100,
  title={MP-100: A More Diverse Animal Pose Dataset},
  journal={arXiv preprint},
  year={2022}
}
```

## License

This code is for educational purposes. Please refer to the MP-100 dataset license for data usage terms.
