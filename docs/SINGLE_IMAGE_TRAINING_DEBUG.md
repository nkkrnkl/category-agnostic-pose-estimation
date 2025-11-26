# Single Image Training Debug Guide

## Problem
PCK stuck at 33.33% (6/18 visible keypoints) after 50 epochs when training on a single image. Expected: PCK should approach 100% since the model should memorize the single image.

## Key Observations

1. **PCK is exactly 33.33% (6/18)** - This suspiciously round number suggests:
   - Model might be predicting a fixed pattern
   - Issue with keypoint extraction from sequence
   - Coordinate system mismatch

2. **Training loss still high** (coords loss = 0.1058):
   - Model isn't learning well even with teacher forcing
   - Suggests fundamental learning issue, not just autoregressive generation

3. **Same image used for support and query**:
   - In single-image mode, support = query
   - Model should be able to "copy" support keypoints
   - If it can't, support encoder/fusion might not be working

## Debugging Steps

### 1. Verify Single Image Setup

Run the diagnostic script:
```bash
python debug_single_image_training.py \
    --dataset_root /path/to/data \
    --category_split_file category_splits.json \
    --debug_single_image_path data/bison_body/000000001113.jpg
```

This checks:
- ✅ Same image used for support and query
- ✅ Keypoints in correct coordinate space [0,1]
- ✅ Bbox dimensions correct for PCK

### 2. Check Training Loss Trend

**Question**: Is the training loss actually decreasing?

Look at your training logs:
- If loss is NOT decreasing → Model not learning (check learning rate, optimizer, gradient flow)
- If loss IS decreasing but PCK not improving → Issue with evaluation/autoregressive generation

**Expected behavior for single image**:
- Epoch 1-5: Loss should drop quickly (model memorizing)
- Epoch 5-10: Loss should approach 0
- Epoch 10+: Loss near 0, PCK near 100%

### 3. Check Support Information Usage

The model should use support keypoints to condition predictions. In single-image mode:
- Support keypoints = Query keypoints (same image)
- Model should learn: "Given these support keypoints, predict the same keypoints"

**Debug**: Add logging to check if support features are being used:
```python
# In cape_model.py forward_inference, add:
print(f"Support coords shape: {support_coords.shape}")
print(f"Support features shape: {support_features.shape}")
print(f"Support coords sample: {support_coords[0, :3, :]}")
```

### 4. Check Coordinate System

**Critical Issue**: PCK computation might use wrong coordinate system.

In `engine_cape.py` line 686-687:
```python
pred_kpts_trimmed_pixels = [kpts * 512.0 for kpts in pred_kpts_trimmed]
```

This assumes keypoints are in [0,1] relative to 512x512 image. But:
- If original bbox is NOT 512x512, this scaling is wrong!
- PCK threshold becomes incorrect

**Fix**: Check if bbox dimensions are correct:
```python
# In validation, add debug:
print(f"Bbox dimensions: {bbox_width}x{bbox_height}")
print(f"PCK threshold (pixels): {0.2 * np.sqrt(bbox_width**2 + bbox_height**2):.1f}")
```

### 5. Check Autoregressive Generation

During validation, model uses `forward_inference` (autoregressive). Errors compound:
- If first keypoint is wrong → affects all subsequent keypoints
- This is expected, but model should still learn correct first keypoint

**Debug**: Enable debug mode:
```bash
export DEBUG_CAPE=1
export DEBUG_PCK=1
python -m models.train_cape_episodic ...
```

This will show:
- Whether predictions match support (data leakage)
- Whether predictions match GT (impossible in autoregressive)
- Coordinate values and PCK computation details

### 6. Check Keypoint Extraction

The model generates a sequence, then extracts keypoints. If extraction is wrong:
- Might extract wrong number of keypoints
- Might use wrong token types

**Check**: In `engine_cape.py` validation, verify:
```python
print(f"Predicted keypoints: {pred_kpts.shape}")
print(f"GT keypoints: {gt_kpts.shape}")
print(f"Are they same shape? {pred_kpts.shape == gt_kpts.shape}")
```

### 7. Visualize Predictions

Use the visualization script to see actual predictions:
```bash
python -m models.visualize_cape_predictions \
    --checkpoint output/single_image_colab/checkpoint_e022_*.pth \
    --dataset_root /path/to/data \
    --single_image_path data/bison_body/000000001113.jpg \
    --output_dir visualizations
```

**What to look for**:
- Are predicted keypoints close to GT? (even if not perfect)
- Are they in reasonable locations?
- Do they match support keypoints? (should in single-image mode)

## Most Likely Issues

### Issue 1: Support Information Not Used
**Symptom**: Predictions don't match support even though they're the same image
**Fix**: Check support encoder and fusion mechanism

### Issue 2: Coordinate System Mismatch
**Symptom**: PCK threshold wrong, causing incorrect evaluation
**Fix**: Verify bbox dimensions and coordinate scaling

### Issue 3: Model Not Learning
**Symptom**: Loss not decreasing, predictions random
**Fix**: Check learning rate, optimizer, gradient flow

### Issue 4: Keypoint Extraction Bug
**Symptom**: Wrong number of keypoints extracted, causing PCK to be wrong
**Fix**: Check `extract_keypoints_from_predictions` function

## Quick Fixes to Try

1. **Increase learning rate** (if loss not decreasing):
   ```python
   --lr 1e-3  # Instead of 1e-4
   ```

2. **Disable autoregressive for validation** (temporary test):
   - Modify `engine_cape.py` to use teacher forcing during validation
   - If PCK becomes 100%, issue is with autoregressive generation
   - If PCK still low, issue is with model learning

3. **Check gradient flow**:
   ```python
   # Add to training loop:
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
   print(f"Gradient norm: {total_norm}")
   ```
   - If gradient norm is very small (< 0.01) → learning rate too low or vanishing gradients
   - If gradient norm is very large (> 100) → learning rate too high or exploding gradients

4. **Reduce model complexity** (if overfitting is the issue):
   - Use fewer decoder layers
   - Reduce hidden dimension
   - This shouldn't be needed for single image, but worth trying

## Expected Results

For single-image training, you should see:
- **Epoch 1-5**: Loss drops from ~1.0 to ~0.1
- **Epoch 5-10**: Loss drops from ~0.1 to ~0.01
- **Epoch 10+**: Loss < 0.01, PCK > 90%
- **Epoch 20+**: Loss < 0.001, PCK > 95%

If you're not seeing this, there's a fundamental issue with:
1. Model architecture (support encoder not working)
2. Training setup (learning rate, optimizer)
3. Evaluation setup (coordinate system, PCK computation)

## Next Steps

1. Run diagnostic script to verify setup
2. Check training loss trend (is it decreasing?)
3. Enable debug mode and check predictions
4. Visualize predictions to see what model is actually predicting
5. Check support encoder output (is it using support information?)

If all checks pass but PCK still low, the issue is likely in:
- Autoregressive generation (errors compound)
- Keypoint extraction from sequence
- PCK computation (coordinate system)


