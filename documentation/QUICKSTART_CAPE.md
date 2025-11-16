# Quick Start Guide: MP-100 CAPE Training

## Prerequisites

Ensure you have completed the data cleaning step and have:
- ✅ 49 categories with images (from original 100)
- ✅ Cleaned annotations at `annotations/mp100_split{1-5}_{train,val,test}.json`
- ✅ Images at `theodoros/data/<category_name>/*.jpg`

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Navigate to project
cd "/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project"

# Activate virtual environment
source venv/bin/activate

# Install PyTorch (choose based on your system)
# For Mac M1/M2:
pip install torch torchvision

# For CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
cd theodoros
pip install -r requirements_cape.txt

# Install detectron2 (adjust for your CUDA version)
# For CPU/Mac:
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# For CUDA 11.8:
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision {torchvision.__version__}')"
python -c "from pycocotools.coco import COCO; print('pycocotools OK')"
```

### 3. Test Dataset Loading

```bash
# Make sure you're in the theodoros directory
cd theodoros

# Run the test script
python test_mp100_loading.py
```

**Expected Output:**
```
================================================================================
Testing MP-100 CAPE Dataset Loading
================================================================================

Dataset root: /Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/theodoros
Split: 1

--------------------------------------------------------------------------------
Loading TRAIN dataset...
--------------------------------------------------------------------------------
loading annotations into memory...
Done (t=X.XXs)
creating index...
index created!
Loaded MP-100 train dataset: XXXX images
✓ Train dataset loaded: XXXX samples

Testing single sample loading...
✓ Sample loaded successfully
  - Image shape: torch.Size([3, 256, 256])
  - Num keypoints: XX
  - Category ID: X
  - Image ID: XXXXX
  - First 3 keypoints: [[x1, y1], [x2, y2], [x3, y3]]

--------------------------------------------------------------------------------
Loading VAL dataset...
--------------------------------------------------------------------------------
✓ Val dataset loaded: XXX samples
...
================================================================================
✓ All tests passed!
================================================================================
```

### 4. Quick Debug Training (Overfit Single Sample)

This is the fastest way to verify everything works:

```bash
python train_mp100_cape.py \
    --debug \
    --dataset_root . \
    --mp100_split 1 \
    --epochs 50 \
    --output_dir output/debug_test
```

**What to expect:**
- Should take ~5-10 minutes
- Loss should drop significantly (→ near 0)
- Indicates model can learn the mapping

### 5. Small-Scale Training (Quick Experiment)

```bash
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 2 \
    --epochs 20 \
    --lr 1e-4 \
    --output_dir output/quick_experiment \
    --job_name quick_test
```

**What to expect:**
- Takes ~30-60 minutes (depending on hardware)
- You'll see training/validation loss
- Checkpoints saved in `output/quick_experiment/`

### 6. Full Training (Single Split)

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
    --output_dir output/split1_full \
    --job_name mp100_split1_v5 \
    --num_workers 4
```

**What to expect:**
- Takes several hours to days
- Best results with GPU
- Checkpoints every 20 epochs

### 7. Training with Logging (WandB)

```bash
# First time: wandb login
wandb login

# Then train with logging
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 4 \
    --epochs 100 \
    --image_norm \
    --use_wandb \
    --wandb_project "MP100-CAPE-Project" \
    --output_dir output/split1_logged \
    --job_name split1_experiment
```

## Common Commands

### Resume from Checkpoint

```bash
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 1 \
    --batch_size 4 \
    --resume output/split1_full/checkpoint.pth \
    --output_dir output/split1_resumed
```

### Train on Different Split

```bash
# Train on split 2 instead of split 1
python train_mp100_cape.py \
    --dataset_root . \
    --mp100_split 2 \
    --batch_size 4 \
    --epochs 300 \
    --output_dir output/split2
```

### Train All 5 Splits (Cross-Validation)

```bash
#!/bin/bash
# Save as train_all_splits.sh

for split in 1 2 3 4 5; do
    echo "Training split $split..."
    python train_mp100_cape.py \
        --dataset_root . \
        --mp100_split $split \
        --batch_size 4 \
        --epochs 300 \
        --lr 1e-4 \
        --lr_drop 200,250 \
        --image_norm \
        --use_anchor \
        --output_dir output/split${split} \
        --job_name mp100_split${split}
done
```

## Troubleshooting

### Error: "No module named 'torch'"

```bash
# Make sure venv is activated
source ../venv/bin/activate

# Re-install torch
pip install torch torchvision
```

### Error: "FileNotFoundError: Annotation file not found"

Check that annotations exist:
```bash
ls -la ../annotations/mp100_split1_train.json
```

If missing, the annotations should be in the parent `annotations/` directory.

### Error: "CUDA out of memory"

Reduce batch size:
```bash
python train_mp100_cape.py \
    --batch_size 1 \  # or 2
    --dataset_root . \
    --mp100_split 1
```

### Error: "No module named 'detectron2'"

```bash
# For Mac/CPU:
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# For CUDA 11.8:
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### Training is very slow

1. **Reduce image resolution** (edit mp100_cape.py, change Resize to (128, 128))
2. **Use fewer workers**: `--num_workers 0`
3. **Smaller model**: `--hidden_dim 128 --enc_layers 3 --dec_layers 3`

## Monitoring Training

### Check Loss

```bash
# View log file
tail -f output/split1_full/quick_test/log.txt
```

### Check Checkpoints

```bash
ls -lh output/split1_full/quick_test/*.pth
```

## Next Steps After Training

1. **Evaluate model** (create eval script)
2. **Visualize predictions** (create visualization script)
3. **Compare with baselines** (implement baseline models)
4. **Write report** with results

## Tips for Best Results

1. **Start small**: Always test with `--debug` first
2. **Monitor overfitting**: Check train vs val loss
3. **Try different architectures**: `--dec_layer_type v1` through `v6`
4. **Use image normalization**: `--image_norm` often helps
5. **Enable anchors**: `--use_anchor` improves convergence
6. **Tune learning rate**: Start with `1e-4`, adjust if needed

## Hardware Recommendations

- **Minimum**: CPU with 16GB RAM (slow, ~days for full training)
- **Recommended**: GPU with ≥8GB VRAM (hours for full training)
- **Optimal**: GPU with ≥16GB VRAM + multi-GPU (fastest)

## Success Criteria

Your training is working if:
- ✅ Debug mode achieves near-zero loss on single sample
- ✅ Training loss decreases steadily
- ✅ Validation loss follows training loss (with some gap)
- ✅ Checkpoints are being saved
- ✅ No CUDA OOM or other errors

## Getting Help

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review `README_MP100_CAPE.md` for detailed explanations
3. Check the error message carefully
4. Verify all paths and file existence
5. Try `--debug` mode first

Good luck with your CAPE implementation! 🚀
