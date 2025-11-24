# Virtual Environment Setup Guide

## ✅ Setup Complete!

Your Python virtual environment is successfully configured with all dependencies for CAPE training.

---

## 📦 **What Was Installed**

### **Python Version**
- Python 3.13.7

### **Core Dependencies**
- **PyTorch**: 2.9.1 (with MPS support for Apple Silicon)
- **TorchVision**: 0.24.1
- **NumPy**: 2.2.6
- **SciPy**: 1.16.3

### **Image Processing**
- **OpenCV**: 4.12.0.88
- **Albumentations**: 2.0.8 (data augmentation)
- **Pillow**: 12.0.0

### **Dataset & Annotations**
- **pycocotools**: 2.0.10 (MP-100 COCO format)

### **Visualization**
- **Matplotlib**: 3.10.7
- **Seaborn**: 0.13.2

### **Model & Training**
- **timm**: 1.0.22 (model components)
- **einops**: 0.8.1 (tensor operations)
- **tqdm**: 4.67.1 (progress bars)

### **Development & Testing**
- **pytest**: 9.0.1 (testing framework)
- **black**: 25.11.0 (code formatting)

---

## 🚀 **How to Use**

### **Option 1: Using the Activation Script** (Recommended)

```bash
cd "/Users/pavlosrousoglou/Desktop/Cornell/Deep Learning/category-agnostic-pose-estimation"
source activate_venv.sh
```

This will:
- Activate the virtual environment
- Show installed Python/PyTorch versions
- Display available commands

### **Option 2: Manual Activation**

```bash
cd "/Users/pavlosrousoglou/Desktop/Cornell/Deep Learning/category-agnostic-pose-estimation"
source venv/bin/activate
```

To deactivate:
```bash
deactivate
```

---

## ✅ **Verification**

### **Check Installation**
```bash
source venv/bin/activate
python -c "import torch; import torchvision; import numpy; import cv2; import albumentations; print('✅ All dependencies OK')"
```

### **Run Tests**
```bash
source venv/bin/activate
python tests/test_checkpoint_system.py
```

Expected output:
```
================================================================================
CHECKPOINT SYSTEM TESTS
================================================================================

[Test 1] Checkpoint Contains Expected Fields
✓ All required checkpoint fields present and correct

[Test 2] Resume Restores Full State
✓ Model, optimizer, and scheduler restored correctly
✓ RNG states restored correctly for reproducibility
✓ Best metrics restored correctly (prevents checkpoint overwrite bug)

[Test 3] PCK-Based Best Model Saving
✓ Best PCK checkpoint saved correctly when PCK improves
✓ Best-loss and best-PCK tracked independently

[Test 4] Best Checkpoint Not Overwritten After Resume
✓ Best checkpoint preserved after resume (bug fixed!)
✓ Resume allows saving better checkpoints (as expected)

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

---

## 🖥️ **Device Configuration**

Your system is configured to use:
- **Primary Device**: **MPS** (Metal Performance Shaders - Apple Silicon GPU)
- **Fallback**: CPU

**Note**: The training script automatically enables MPS fallback for compatibility:
```python
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Some operations may fall back to CPU for compatibility with deformable attention layers.

---

## 🏋️ **Training Commands**

### **Quick Test (5 epochs)**
```bash
source venv/bin/activate

python train_cape_episodic.py \
    --epochs 5 \
    --batch_size 1 \
    --accumulation_steps 2 \
    --num_queries_per_episode 1 \
    --episodes_per_epoch 100 \
    --early_stopping_patience 0 \
    --output_dir ./outputs/test_run
```

### **Full Training (300 epochs)**
```bash
source venv/bin/activate

python train_cape_episodic.py \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 2 \
    --episodes_per_epoch 1000 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_production
```

### **Resume Training**
```bash
source venv/bin/activate

python train_cape_episodic.py \
    --resume ./outputs/cape_production/checkpoint_e050_*.pth \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 2 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_production
```

---

## 📁 **File Structure**

```
category-agnostic-pose-estimation/
├── venv/                          # Virtual environment (DO NOT COMMIT)
├── activate_venv.sh               # Convenience activation script
├── train_cape_episodic.py         # Main training script
├── engine_cape.py                 # Training/evaluation engine
├── datasets/
│   ├── mp100_cape.py              # MP-100 dataset loader
│   └── episodic_sampler.py        # Episodic sampling
├── models/
│   ├── cape_model.py              # CAPE wrapper model
│   ├── cape_losses.py             # CAPE-specific losses
│   ├── support_encoder.py         # Support pose graph encoder
│   └── roomformer_v2.py           # Base Raster2Seq model
├── tests/
│   └── test_checkpoint_system.py  # Checkpoint system tests
├── data/                          # MP-100 images
├── annotations/                   # MP-100 annotations
└── outputs/                       # Training outputs (checkpoints)
```

---

## 🔧 **Troubleshooting**

### **Issue: Import errors**
```bash
# Make sure venv is activated
source venv/bin/activate

# Verify Python path
which python
# Should show: .../venv/bin/python
```

### **Issue: PyTorch not using MPS**
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If False, PyTorch will use CPU (slower but functional).

### **Issue: Out of memory**
Reduce batch size and/or queries per episode:
```bash
--batch_size 1 \
--num_queries_per_episode 1
```

---

## 📝 **Additional Setup (Optional)**

### **Install Jupyter (for notebooks)**
```bash
source venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name cape --display-name "CAPE (Python 3.13)"
```

### **Install wandb (for logging)**
```bash
source venv/bin/activate
pip install wandb
wandb login  # Enter your API key
```

Then use `--use_wandb` flag when training:
```bash
python train_cape_episodic.py --use_wandb --wandb_project "CAPE-MP100" ...
```

---

## ⚠️ **Important Notes**

1. **Always activate venv before running code**:
   ```bash
   source venv/bin/activate
   ```

2. **DO NOT commit venv/ to git**:
   - Already in `.gitignore`
   - Recreate venv on different machines

3. **PyTorch 2.9+ compatibility**:
   - Checkpoints use `weights_only=False` for compatibility
   - This is safe for trusted checkpoints (your own training)

4. **MPS (Apple Silicon) limitations**:
   - Some operations may fall back to CPU
   - Training is functional but may be slower than CUDA

---

## 📚 **Next Steps**

1. ✅ Virtual environment setup complete
2. ✅ All dependencies installed
3. ✅ Tests passing

**You're ready to start training!**

```bash
# Start with a quick test
source venv/bin/activate
python train_cape_episodic.py \
    --epochs 5 \
    --batch_size 1 \
    --num_queries_per_episode 1 \
    --episodes_per_epoch 50 \
    --output_dir ./outputs/quick_test

# If successful, run full training
python train_cape_episodic.py \
    --epochs 300 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_queries_per_episode 2 \
    --early_stopping_patience 20 \
    --output_dir ./outputs/cape_final
```

---

*Setup completed: 2024*
*Python: 3.13.7*
*PyTorch: 2.9.1*
*Device: MPS (Apple Silicon)*

