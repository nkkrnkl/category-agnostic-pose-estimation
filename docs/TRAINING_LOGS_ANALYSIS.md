# Training Logs Analysis - Single Image Training

## Summary

**Training completed:** 50 epochs on single image `camel_face/camel_26.jpg`  
**Best PCK:** 55.56% (10/18 visible keypoints) at **Epoch 15**  
**Final PCK:** 44.44% (8/18 visible keypoints) at Epoch 50  
**Issue:** PCK peaked at epoch 15, then degraded and fluctuated

## Key Findings

### 1. Training Loss IS Decreasing ✅

The model **IS learning** - training loss decreased significantly:

| Epoch | Total Loss | Coords Loss | Class Loss |
|-------|------------|-------------|------------|
| 1     | 17.95      | 1.65        | 1.91       |
| 10    | 6.34       | 0.90        | 0.00       |
| 15    | 2.62       | 0.32        | 0.00       |
| 20    | 1.15       | 0.11        | 0.00       |
| 30    | 0.69       | 0.09        | 0.00       |
| 40    | 0.39       | 0.05        | 0.00       |
| 50    | 0.30       | 0.045       | 0.00       |

**Conclusion:** Model is learning to minimize training loss with teacher forcing.

### 2. PCK Performance Trend ❌

PCK performance shows a concerning pattern:

| Epoch Range | PCK Range | Notes |
|-------------|-----------|-------|
| 1-2         | 0%        | Initial learning phase |
| 3-9         | 11.11%    | Stuck at 2/18 keypoints |
| 10          | 44.44%    | **Jump to 8/18** |
| 15          | **55.56%** | **BEST: 10/18 keypoints** |
| 16-50       | 11-44%    | **Degraded and fluctuates** |

**Critical Issue:** 
- Best PCK achieved at epoch 15 (55.56%)
- After epoch 15, PCK **degraded** and fluctuates between 11-44%
- Never exceeds the epoch 15 best
- Final epoch (50) has 44.44% (8/18), worse than epoch 15

### 3. Loss vs PCK Discrepancy ⚠️

**The Problem:**
- Training loss continues to decrease (0.30 at epoch 50)
- Coords loss continues to decrease (0.045 at epoch 50)
- **BUT** PCK does NOT improve (worse than epoch 15)

**This indicates:**
1. Model learns well with **teacher forcing** (training)
2. Model fails with **autoregressive generation** (validation)
3. There's a **mismatch** between training and validation modes

### 4. Autoregressive Generation Issues

The validation uses autoregressive inference (`forward_inference`), which:
- Generates keypoints one at a time
- Errors compound (if first keypoint is wrong, all subsequent are affected)
- Is harder than teacher forcing (where model sees GT during training)

**Evidence:**
- Training loss (teacher forcing) → 0.045 coords loss
- Validation loss (autoregressive) → 0.18-0.21 coords loss (4-5x higher!)
- PCK stuck at ~33-44% despite low training loss

## Root Cause Analysis

### Hypothesis 1: Overfitting to Teacher Forcing
**Evidence:**
- Training loss very low (0.045 coords loss)
- Validation loss much higher (0.18-0.21 coords loss)
- Model learned to predict well when seeing GT, but fails when generating autoregressively

**Solution:** Use scheduled sampling or curriculum learning to gradually transition from teacher forcing to autoregressive generation during training.

### Hypothesis 2: Autoregressive Generation Bug
**Evidence:**
- PCK peaked at epoch 15, then degraded
- Fluctuates randomly (11%, 22%, 33%, 44%)
- Suggests autoregressive generation might have instability

**Solution:** Check `forward_inference` implementation for bugs, especially:
- Initial token generation
- Causal masking
- Support feature injection

### Hypothesis 3: Support Information Not Used Effectively
**Evidence:**
- In single-image mode, support = query (same image)
- Model should be able to "copy" support keypoints
- But PCK only 55.56% at best (should be ~100%)

**Solution:** Verify support encoder and fusion mechanism:
- Are support features actually being used?
- Is cross-attention working correctly?
- Are support keypoints being properly encoded?

### Hypothesis 4: Coordinate System Mismatch
**Evidence:**
- PCK computation might use wrong coordinate system
- Keypoints scaled by 512, but bbox might be different size
- Could cause incorrect PCK threshold

**Solution:** Verify PCK computation uses correct bbox dimensions.

## Recommendations

### Immediate Actions

1. **Visualize Predictions**
   ```bash
   python -m models.visualize_cape_predictions \
       --checkpoint output/single_image_colab/checkpoint_best_pck_*.pth \
       --single_image_path data/camel_face/camel_26.jpg
   ```
   Check if predictions are reasonable or random.

2. **Compare Epoch 15 vs Epoch 50**
   - Load checkpoint from epoch 15 (best PCK)
   - Load checkpoint from epoch 50 (final)
   - Compare predictions side-by-side
   - Why did epoch 15 perform better?

3. **Enable Debug Mode**
   ```bash
   export DEBUG_CAPE=1
   export DEBUG_PCK=1
   # Re-run validation to see detailed logs
   ```

4. **Check Autoregressive Generation**
   - Verify `forward_inference` is working correctly
   - Check if support features are being used
   - Verify initial token generation

### Long-term Fixes

1. **Scheduled Sampling**
   - Gradually transition from teacher forcing to autoregressive during training
   - Start with 100% teacher forcing, end with 100% autoregressive
   - This bridges the gap between training and validation

2. **Autoregressive Training**
   - Use autoregressive generation during training (not just validation)
   - More expensive but should improve validation performance

3. **Support Encoder Debugging**
   - Verify support features are actually conditioning predictions
   - Check cross-attention weights
   - Ensure support information flows to decoder

4. **PCK Computation Fix**
   - Verify coordinate system is correct
   - Check bbox dimensions match original (not preprocessed)
   - Ensure keypoint scaling is correct

## Expected vs Actual

### Expected (Single Image Overfitting):
- Epoch 1-5: Loss drops quickly (17 → 1)
- Epoch 5-10: Loss approaches 0 (1 → 0.1)
- Epoch 10+: Loss < 0.01, PCK > 90%

### Actual:
- ✅ Loss drops (17 → 0.3)
- ❌ PCK stuck at 33-55% (should be >90%)
- ❌ PCK peaked at epoch 15, then degraded
- ❌ Never reaches expected overfitting performance

## Conclusion

The model **IS learning** (loss decreasing), but there's a **fundamental mismatch** between:
- **Training mode:** Teacher forcing (sees GT) → Low loss
- **Validation mode:** Autoregressive (generates) → Low PCK

The best PCK (55.56%) was achieved at epoch 15, then performance degraded. This suggests:
1. Model overfitted to teacher forcing
2. Autoregressive generation has issues
3. Support information might not be used effectively

**Next steps:** Visualize predictions, compare epoch 15 vs 50, and debug autoregressive generation.


