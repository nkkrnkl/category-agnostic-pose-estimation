# Keypoint Normalization Pipeline for CAPE

## Overview

This document explains how keypoints are normalized in the CAPE dataset pipeline, addressing the apparent "double normalization" concern. The normalization is **mathematically correct** and achieves bbox-relative normalization despite going through multiple steps.

## Complete Pipeline

### **Step 1: Crop Image to Bounding Box** (`mp100_cape.py`, lines 234-241)

**Goal**: Focus on the object of interest, removing background clutter.

```python
# Extract bbox [x, y, width, height] from annotation
bbox_x, bbox_y, bbox_w, bbox_h = bbox

# Crop image
img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

# Make keypoints relative to bbox top-left corner
kpts_array[:, 0] -= bbox_x  # x relative to bbox
kpts_array[:, 1] -= bbox_y  # y relative to bbox
```

**Result:**
- Image dimensions: `(bbox_h, bbox_w)` pixels
- Keypoint range: `[0, bbox_w] × [0, bbox_h]` pixels
- Keypoints are now in **bbox-relative pixel coordinates**

**Example:**
```
Original image: 1024 × 768
Bbox: [200, 100, 300, 400]  # x, y, w, h
Keypoint at (250, 150) in original image

After cropping:
- Cropped image: 400 × 300
- Keypoint: (250-200, 150-100) = (50, 50)
```

---

### **Step 2: Resize to 512×512** (`mp100_cape.py`, lines 330-358)

**Goal**: Standardize all images to same size for batching.

```python
# Resize cropped image to 512×512
transformed = self._transforms(image=img)  # Resize transform
img = transformed['image']

# Scale keypoints proportionally
scale_x = 512 / bbox_w
scale_y = 512 / bbox_h

scaled_keypoints = []
for kpt in record["keypoints"]:
    x, y = kpt
    scaled_keypoints.append([x * scale_x, y * scale_y])
```

**Result:**
- Image dimensions: `(512, 512)` pixels
- Keypoint range: `[0, 512] × [0, 512]` pixels
- `record["height"]` = 512
- `record["width"]` = 512

**Example (continuing from above):**
```
Before resize:
- Image: 400 × 300
- Keypoint: (50, 50)

After resize to 512×512:
- scale_x = 512 / 300 = 1.707
- scale_y = 512 / 400 = 1.28
- Keypoint: (50 × 1.707, 50 × 1.28) = (85.35, 64.0)
```

---

### **Step 3: Normalize to [0, 1]** (`episodic_sampler.py`, lines 193-209)

**Goal**: Normalize keypoints to [0, 1] range for model input.

```python
# Load keypoints (currently in range [0, 512])
support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)

# Get dimensions (both are 512 after resize)
h, w = support_data['height'], support_data['width']  # 512, 512

# Normalize to [0, 1]
support_coords[:, 0] /= w  # Divide by 512
support_coords[:, 1] /= h  # Divide by 512
```

**Result:**
- Keypoint range: `[0, 1] × [0, 1]` (normalized)

**Example (continuing from above):**
```
Before normalization:
- Keypoint: (85.35, 64.0) in 512×512 image

After normalization:
- Keypoint: (85.35/512, 64.0/512) = (0.167, 0.125)
```

---

## Mathematical Equivalence

### Question: Is this the same as directly normalizing by bbox dimensions?

**Answer: YES!** ✅

Let's prove this mathematically:

```
Original keypoint in bbox coordinates: (x_bbox, y_bbox)
Bbox dimensions: (bbox_w, bbox_h)

Step-by-step transformation:
1. After resize:   x_resized = x_bbox × (512 / bbox_w)
2. After normalize: x_final = x_resized / 512
                           = (x_bbox × 512 / bbox_w) / 512
                           = x_bbox / bbox_w

Final result: (x_bbox / bbox_w, y_bbox / bbox_h)
```

**This is exactly the same as directly normalizing by bbox dimensions!**

### Concrete Example

Let's verify with real numbers:

```
Bbox dimensions: 300 × 400
Keypoint in bbox coords: (50, 50)

Method 1 (our pipeline):
- Resize: (50 × 512/300, 50 × 512/400) = (85.33, 64.0)
- Normalize: (85.33/512, 64.0/512) = (0.1667, 0.125)

Method 2 (direct bbox normalization):
- Normalize: (50/300, 50/400) = (0.1667, 0.125)

✓ IDENTICAL RESULTS
```

---

## Why This Pipeline Design?

### Advantages

1. **Simplicity**: Each step has a clear, single purpose
   - Step 1: Crop → focus on object
   - Step 2: Resize → standardize size
   - Step 3: Normalize → scale to [0, 1]

2. **Modularity**: Easy to modify individual steps
   - Can change resize resolution (e.g., 256×256 instead of 512×512)
   - Can add augmentations between steps
   - Can experiment with different normalization schemes

3. **Standard Practice**: Matches common computer vision pipelines
   - Crop → Resize → Normalize is a standard pattern
   - Compatible with existing data augmentation libraries

4. **Numerical Stability**: Avoids very small or very large intermediate values
   - Working in 512×512 space is more stable than tiny [0, 1] space
   - Integer pixel ops in resize are well-optimized

### Trade-offs

- **Slightly More Complex**: Requires understanding 3 steps vs. 1 direct normalization
- **Documentation Burden**: Need to explain the pipeline (hence this doc!)
- **Minor Precision Loss**: Float operations in resize + normalize (negligible in practice)

---

## Storage in record dict

After the complete pipeline:

```python
record = {
    'keypoints': [...],          # In range [0, 512] (Step 2 output)
    'height': 512,               # Image height after resize
    'width': 512,                # Image width after resize
    'bbox': [x, y, w, h],       # Original bbox in full image
    'bbox_width': bbox_w,        # Original bbox width (for PCK)
    'bbox_height': bbox_h,       # Original bbox height (for PCK)
    'visibility': [...]          # Visibility flags
}
```

**Important Notes:**
- `keypoints` are stored in **pixel coordinates** [0, 512], not normalized
- `height` and `width` are **post-resize** dimensions (512, 512), not original bbox
- `bbox_width` and `bbox_height` store **original bbox dimensions** for PCK evaluation
- Final normalization to [0, 1] happens in `episodic_sampler.py`

---

## Verification

### How to verify normalization is correct:

1. **Check mathematical equivalence** (proven above) ✅
2. **Verify coordinate ranges** at each step:
   - After Step 1: [0, bbox_dim]
   - After Step 2: [0, 512]
   - After Step 3: [0, 1]
3. **Test with known coordinates**:
   - Bbox corner (0, 0) should → (0, 0) after normalization
   - Bbox corner (bbox_w, bbox_h) should → (1, 1) after normalization
4. **Visual inspection**: Overlay predicted keypoints on images to confirm correct positions

---

## Summary

✅ **The normalization pipeline is mathematically correct**

✅ **Final result is bbox-relative normalization in [0, 1]**

✅ **The "double normalization" concern was just unclear documentation**

✅ **Now fully documented with clear comments at each step**

The pipeline achieves **scale and translation invariance** by normalizing keypoints relative to their bounding box dimensions, which is essential for category-agnostic pose estimation where objects vary greatly in size and position.

