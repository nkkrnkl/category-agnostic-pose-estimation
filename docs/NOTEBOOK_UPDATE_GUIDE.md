# Notebook Update Guide for Visualization Fixes

## ✅ No Changes Required!

The visualization script (`models/visualize_cape_predictions.py`) has been updated with all fixes and will automatically:
- ✅ Use correct denormalization (TARGET_SIZE = 512)
- ✅ Ensure images are 512×512 before visualization
- ✅ Enable coordinate debugging for single-image mode
- ✅ Print coordinate comparisons when PCK ≥ 0.99

**Your existing notebook cell will work as-is!**

---

## Optional: Add Verification

If you want to verify the fixes are active, you can add this check before running visualization:

```python
# Optional: Verify visualization fixes are active
import importlib
import models.visualize_cape_predictions as viz_module

# Reload module to ensure latest code (useful if you modified files)
importlib.reload(viz_module)

# Check that TARGET_SIZE constant exists (confirms fixes are present)
if hasattr(viz_module, 'TARGET_SIZE'):
    print(f"✅ Visualization fixes active (TARGET_SIZE = {viz_module.TARGET_SIZE})")
else:
    print("⚠️  TARGET_SIZE not found - may be using old code")
```

---

## What to Expect

When you run visualization with the fixed code, you should see:

### 1. **Coordinate Debugging Output** (for single-image mode):
```
================================================================================
COORDINATE COMPARISON DEBUG: Single Image: image.jpg
================================================================================
  Number of keypoints: 10
  GT coordinates (pixels, first 5):
    Kpt 0: (256.00, 128.00)
    ...
  Pred coordinates (pixels, first 5):
    Kpt 0: (256.00, 128.00)
    ...
  Absolute differences (pixels, first 5):
    Kpt 0: dx=0.00, dy=0.00, dist=0.00
    ...
  Mean Euclidean distance: 0.05 pixels
  Max Euclidean distance: 0.15 pixels
  ✓ Coordinates match well (mean dist=0.05px) - visualization should align
================================================================================
```

### 2. **Perfect Alignment** (when PCK = 1.0):
- GT and predicted keypoints should overlap **pixel-perfectly**
- Mean distance should be < 1 pixel
- Visualization should show keypoints in correct locations

### 3. **Warnings** (if images aren't 512×512):
```
⚠️  Support image size (256x256) != 512x512. Coordinates are normalized relative to 512x512. Visualization may be misaligned!
```
*(Note: Images are automatically resized, so this warning is just informational)*

---

## Your Current Notebook Cell

Your existing cell is already correct:

```python
cmd = [
    sys.executable, "-m", "models.visualize_cape_predictions",
    "--checkpoint", CHECKPOINT,
    "--dataset_root", PROJECT_ROOT,
    "--device", "cuda",
    "--single_image_path", SINGLE_IMAGE_PATH,
    "--output_dir", VISUALIZATION_DIR,
]
```

**No changes needed!** The script will automatically:
- Use fixed denormalization (TARGET_SIZE = 512)
- Ensure images are 512×512
- Enable coordinate debugging
- Show perfect alignment when PCK = 1.0

---

## Troubleshooting

### If keypoints still don't align:

1. **Check coordinate debugging output** - it will show the actual pixel differences
2. **Verify PCK score** - if PCK < 1.0, some misalignment is expected
3. **Check image size** - should be 512×512 (automatically handled)
4. **Verify checkpoint** - make sure you're using the correct checkpoint

### If you see warnings about image size:

- Images are automatically resized to 512×512, so this is just informational
- The visualization will still work correctly

---

## Summary

✅ **No code changes needed in your notebook**
✅ **Fixes are automatic and transparent**
✅ **Coordinate debugging enabled by default for single-image mode**
✅ **Perfect alignment expected when PCK = 1.0**

Just run your existing cell and you should see improved visualizations with coordinate debugging output!

