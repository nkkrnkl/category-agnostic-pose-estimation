# Image Validation Fix

## Issue

During training (around iteration 80), encountered a cv2.error from OpenCV:

```
cv2.error: OpenCV(4.12.0) ... error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

**Root Cause**: The dataloader encountered an image with invalid dimensions (empty or corrupted image), causing Albumentations' Resize transform to fail when it tried to resize an empty array.

---

## Solution

Added comprehensive validation at multiple stages of image loading to detect and skip corrupted/invalid images early:

### 1. **Post-Load Validation**

**Location**: `datasets/mp100_cape.py` (after loading image)

```python
# Check if image loaded correctly
if img is None or img.size == 0:
    raise ImageNotFoundError(...)

if len(img.shape) < 2:
    raise ImageNotFoundError(...)

# Validate dimensions are positive
if orig_h <= 0 or orig_w <= 0:
    raise ImageNotFoundError(...)
```

**Why**: Detects if `PIL.Image.open()` failed to load a valid image.

### 2. **Post-Crop Validation**

**Location**: `datasets/mp100_cape.py` (after bbox cropping)

```python
# After: img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

# Validate cropped image has valid dimensions
if img_cropped.size == 0 or img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
    raise ImageNotFoundError(
        f"Image {img_id} produced empty crop with bbox [{bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}]. "
        f"Original image size: {orig_w}x{orig_h}"
    )
```

**Why**: Detects if bbox coordinates are invalid (e.g., outside image bounds), resulting in an empty crop.

### 3. **Transform Error Handling**

**Location**: `datasets/mp100_cape.py` (around transform application)

```python
try:
    img, record = self._apply_transforms(img, record)
except Exception as e:
    # If transforms fail (e.g., resize on corrupted image), skip this sample
    raise ImageNotFoundError(
        f"Image {img_id} ({record.get('file_name', 'unknown')}) failed during transforms: {str(e)}"
    ) from e
```

**Why**: Catches any remaining edge cases (e.g., corrupted image data) that pass initial validation but fail during transforms.

---

## How It Works

1. **Detection**: When an invalid image is encountered, `ImageNotFoundError` is raised with detailed diagnostics
2. **Retry Logic**: The episodic sampler has built-in retry logic (see `datasets/episodic_sampler.py`)
3. **Automatic Skip**: Bad images are automatically skipped, and a new image is sampled
4. **Training Continues**: No manual intervention needed - training continues with valid images only

---

## Benefits

- ✅ **Robustness**: Training won't crash on corrupted images
- ✅ **Diagnostics**: Clear error messages identify which image/bbox caused the problem
- ✅ **Data Quality**: Only valid images are used for training
- ✅ **No Manual Cleanup**: Dataset automatically filters out bad samples

---

## What to Do If This Happens Frequently

If you see many image validation errors, you should:

1. **Check the error messages** - they will tell you which images are problematic
2. **Investigate those images**:
   ```bash
   # Example: Check if image file exists and is valid
   ls -lh data/path/to/problematic_image.jpg
   file data/path/to/problematic_image.jpg
   ```
3. **Options**:
   - Remove corrupted images from the dataset
   - Fix the annotation file to remove references to missing images
   - Re-download the dataset if many images are corrupted

---

## Expected Behavior

**Normal Operation** (rare corrupted images):
```
...
Epoch: [0]  [ 80/500]  ...
Note: Image 8800000000049056 has 2 instances, using first only
[Skips bad image automatically]
Epoch: [0]  [ 90/500]  ...
```

Training continues smoothly with automatic skipping of bad images.

**Problem Scenario** (many corrupted images):
```
WARNING: Skipped image 12345 (empty crop)
WARNING: Skipped image 67890 (failed to load)
WARNING: Skipped image 11111 (invalid dimensions)
```

If you see many warnings, investigate the dataset quality.

---

## Technical Details

### Validation Checkpoints

| Stage | Check | Error Message |
|-------|-------|---------------|
| **Load** | Image loaded? | "failed to load or is empty" |
| **Load** | Valid shape? | "has invalid shape" |
| **Load** | Positive dimensions? | "has invalid dimensions: WxH" |
| **Crop** | Non-empty crop? | "produced empty crop with bbox" |
| **Transform** | Transforms succeeded? | "failed during transforms" |

### Exception Flow

```
Image Loading
    ↓
Validation Checks ─→ [FAIL] → ImageNotFoundError
    ↓                              ↓
[PASS]                      Episodic Sampler
    ↓                              ↓
Bbox Cropping                Retry Logic
    ↓                              ↓
Crop Validation ───→ [FAIL] → Selects New Image
    ↓                              ↓
[PASS]                      Training Continues
    ↓
Transforms
    ↓
Transform Error ───→ [FAIL] → (same retry)
    ↓
[PASS]
    ↓
Training
```

---

## Files Modified

1. ✅ `datasets/mp100_cape.py` - Added 3 validation checkpoints
2. ✅ `IMAGE_VALIDATION_FIX.md` - This documentation

---

## Testing

To test the fix, training should now:
- ✅ Continue past iteration 80 (where it previously crashed)
- ✅ Automatically skip any corrupted images
- ✅ Complete full epochs without crashing on bad data

Run training normally:
```bash
python train_cape_episodic.py \
  --dataset_root . \
  --epochs 5 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run
```

Training should now be **robust to corrupted images**! 🛡️

