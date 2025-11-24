# MPS (Apple Silicon) Training Fix

## Issue

When training on Apple Silicon (M1/M2/M3/M4), PyTorch's MPS backend doesn't yet implement all operations. Specifically, `grid_sampler_2d_backward` (used in the rasterization loss for bilinear interpolation) causes this error:

```
NotImplementedError: The operator 'aten::grid_sampler_2d_backward' is not currently implemented for the MPS device.
```

## Solution

The training script now **automatically enables MPS fallback** to CPU for unsupported operations.

### What Was Changed

**File: `train_cape_episodic.py`**

Added at the top of the file (before importing torch):
```python
# Enable MPS fallback for operations not yet implemented on Apple Silicon
# This must be set BEFORE importing torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
```

**Why this works:**
- Most operations run on MPS (GPU) for speed
- Unsupported operations (like `grid_sampler_2d_backward`) fall back to CPU
- Training continues without errors
- Performance impact is minimal (only affects the rasterization loss backward pass)

## Running Training

Simply run the training command as before:

```bash
# Activate virtual environment
source activate_venv.sh

# Run training
python train_cape_episodic.py \
  --dataset_root . \
  --epochs 5 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run
```

**Expected output:**
```
Note: Using MPS (Apple Silicon GPU)
      MPS fallback enabled for unsupported ops (e.g., grid_sampler)
```

Training will now run successfully on your M4 Mac! 🚀

## Performance Notes

- **MPS operations**: Run on GPU (fast)
- **Fallback operations**: Run on CPU (slower)
- **Overall impact**: Minimal - most of the model runs on GPU

The rasterization loss (which uses `grid_sampler`) is only one component of the total loss, so the CPU fallback has minimal impact on overall training speed.

## Alternative: CPU-Only Training

If you prefer to run entirely on CPU (for testing or debugging):

```bash
python train_cape_episodic.py \
  --dataset_root . \
  --device cpu \
  ...
```

This will be slower but avoids any MPS compatibility issues.

