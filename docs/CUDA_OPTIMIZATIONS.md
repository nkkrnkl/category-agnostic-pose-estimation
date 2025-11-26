# CUDA Optimizations for CAPE Training

This document describes the CUDA optimizations implemented to accelerate training on GPU.

## 🚀 Optimizations Implemented

### 1. **Mixed Precision Training (AMP)** - ~2x Speedup
- **What**: Uses FP16 for forward pass, FP32 for loss computation and gradients
- **Benefit**: ~2x faster training on modern GPUs (V100, A100, RTX 30xx+)
- **Memory**: Reduces GPU memory usage by ~50%, allowing larger batch sizes
- **Status**: Enabled by default (`--use_amp` flag, default: True)
- **Implementation**: Uses `torch.cuda.amp.autocast()` and `GradScaler`

### 2. **cuDNN Benchmark Mode** - Faster Convolutions
- **What**: cuDNN selects fastest convolution algorithms for your input sizes
- **Benefit**: 10-30% faster convolutions (especially in ResNet backbone)
- **Trade-off**: First iteration is slower (algorithm selection), subsequent iterations are faster
- **Status**: Enabled by default (`--cudnn_benchmark` flag, default: True)
- **Note**: Only effective when input sizes are consistent (which they are in our case)

### 3. **Non-Blocking Data Transfers** - Overlapped CPU/GPU Work
- **What**: Data transfers to GPU happen asynchronously while GPU processes previous batch
- **Benefit**: Hides data loading latency, improves GPU utilization
- **Status**: Automatically enabled for CUDA devices
- **Implementation**: Uses `tensor.to(device, non_blocking=True)`

### 4. **Enhanced Device Detection**
- **What**: Improved CUDA detection with detailed GPU information
- **Features**:
  - GPU name and memory
  - CUDA version
  - cuDNN version
  - Automatic fallback to CPU if CUDA unavailable

## 📊 Expected Performance Improvements

| Optimization | Speedup | Memory Savings |
|-------------|---------|----------------|
| Mixed Precision (AMP) | ~2x | ~50% |
| cuDNN Benchmark | 10-30% | None |
| Non-blocking transfers | 5-15% | None |
| **Combined** | **~2.5-3x** | **~50%** |

*Note: Actual speedup depends on GPU model and workload*

## 🔧 Usage

### Default (All Optimizations Enabled)
```bash
./START_CAPE_TRAINING.sh
```

The script automatically:
- Detects CUDA
- Enables AMP (mixed precision)
- Enables cuDNN benchmark
- Uses non-blocking transfers

### Manual Control
```bash
python -m models.train_cape_episodic \
    --use_amp \              # Enable mixed precision (default: True)
    --cudnn_benchmark \      # Enable cuDNN benchmark (default: True)
    --cudnn_deterministic \  # Use deterministic algorithms (slower, reproducible)
    --device cuda:0
```

### Disable AMP (if you encounter issues)
```bash
python -m models.train_cape_episodic \
    --no-use_amp \           # Disable mixed precision
    --device cuda:0
```

## ⚠️ Important Notes

### Mixed Precision Training
- **Gradient Scaling**: Automatically handled by `GradScaler`
- **Gradient Clipping**: Properly unscales gradients before clipping
- **NaN Handling**: GradScaler handles underflow/overflow automatically
- **Compatibility**: Works with all PyTorch operations (some ops run in FP32 automatically)

### cuDNN Benchmark
- **First Iteration**: Slower (algorithm selection)
- **Subsequent Iterations**: Faster (uses cached algorithm)
- **Input Size Changes**: If input sizes vary, benchmark mode may be slower
- **Our Case**: Input sizes are fixed (256x256), so benchmark mode is beneficial

### Non-Blocking Transfers
- **Requires**: `pin_memory=True` in DataLoader (already enabled)
- **Benefit**: Most noticeable with slow data loading (many workers, complex transforms)
- **Safety**: Automatically disabled for non-CUDA devices

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
1. **Reduce batch size**: `--batch_size 1`
2. **Disable AMP**: `--no-use_amp` (uses more memory but may help)
3. **Reduce accumulation steps**: `--accumulation_steps 2`

### NaN Losses
1. **Check if AMP is causing underflow**: Try `--no-use_amp`
2. **Reduce learning rate**: `--lr 5e-5`
3. **Increase gradient clipping**: `--clip_max_norm 1.0`

### Slow Training
1. **Verify CUDA is being used**: Check logs for "Using device: cuda:0"
2. **Check GPU utilization**: `nvidia-smi` should show high GPU usage
3. **Increase num_workers**: `--num_workers 4` (if CPU has cores available)
4. **Verify AMP is enabled**: Check logs for "Mixed precision training (AMP) enabled"

## 📈 Monitoring GPU Usage

While training, monitor GPU usage:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected:
- **GPU Utilization**: 80-100% during training
- **Memory Usage**: 50-80% of GPU memory (with AMP)
- **Power**: High power draw indicates GPU is working hard

## ✅ Verification

To verify CUDA optimizations are working:

1. **Check training logs** for:
   ```
   ✓ Mixed precision training (AMP) enabled
   ✓ cuDNN benchmark mode enabled
   ✓ CUDA Memory: XX.XX GB
   ```

2. **Compare training speed**:
   - With AMP: ~2x faster than without
   - With cuDNN benchmark: 10-30% faster than without
   - Combined: ~2.5-3x faster than baseline

3. **Monitor GPU**:
   - `nvidia-smi` should show high utilization
   - Memory usage should be reasonable (not 100%)

## 🔗 References

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [cuDNN Benchmark Mode](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.benchmark)
- [Non-blocking Transfers](https://pytorch.org/docs/stable/notes/cuda.html#use-pin-memory)

---

**Last Updated**: 2025-01-XX
**Status**: ✅ All optimizations implemented and tested

