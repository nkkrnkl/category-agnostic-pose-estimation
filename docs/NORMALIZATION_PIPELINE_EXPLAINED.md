# Normalization Pipeline Explanation

## ❓ **Your Question**: "I thought we normalize using bbox size?"

**Short Answer**: We DO normalize using bbox size - but it happens in **3 steps**, and you're looking at the **final step** which divides by the **resized image dimensions** (512×512), not the original bbox dimensions.

---

## 🔄 **The Complete 3-Step Normalization Pipeline**

### **Step 1: Crop to Bounding Box** (`mp100_cape.py:341-360`)

**What happens**:
```python
# Original image: 1920 × 1080
# Bbox: [100, 200, 300, 400]  (x=100, y=200, width=300, height=400)

img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
# Cropped image: 300 × 400

# Make keypoints relative to bbox top-left corner
kpts_array[:, 0] -= bbox_x  # Subtract 100 from all x coords
kpts_array[:, 1] -= bbox_y  # Subtract 200 from all y coords
```

**Before Step 1**:
- Image: 1920 × 1080 (full original)
- Keypoint at pixel (150, 250) in full image

**After Step 1**:
- Image: 300 × 400 (cropped to bbox)
- Keypoint at (50, 50) in cropped bbox space
  - `x = 150 - 100 = 50`
  - `y = 250 - 200 = 50`

**Result**: Keypoints are now **relative to the bbox** ✅

---

### **Step 2: Resize to 512×512** (`mp100_cape.py:529-560`)

**What happens**:
```python
# Apply Albumentations transform (includes A.Resize(512, 512))
transformed = self._transforms(image=img, keypoints=keypoints_list)
img = transformed['image']  # Now 512 × 512

# Scale keypoints proportionally
scale_x = 512 / 300  # = 1.7067
scale_y = 512 / 400  # = 1.28

scaled_keypoints = []
for x, y in keypoints:
    scaled_keypoints.append([x * scale_x, y * scale_y])
```

**Before Step 2**:
- Image: 300 × 400 (bbox size)
- Keypoint at (50, 50) in bbox space

**After Step 2**:
- Image: 512 × 512 (resized)
- Keypoint at (85.3, 64.0) in resized space
  - `x = 50 × (512/300) = 85.3`
  - `y = 50 × (512/400) = 64.0`

**Result**: Keypoints are **scaled proportionally** with the image ✅

**Updated record**:
```python
record["height"] = 512
record["width"] = 512
record["keypoints"] = [[85.3, 64.0], ...]
```

---

### **Step 3: Normalize to [0, 1]** (`mp100_cape.py:601-604`)

**What happens**:
```python
# Normalize keypoints to [0, 1]
# height = 512, width = 512 (from Step 2)
normalized_kpts = []
for x, y in keypoints:
    normalized_kpts.append([x / width, y / height])
    # [85.3 / 512, 64.0 / 512] = [0.167, 0.125]
```

**Before Step 3**:
- Image: 512 × 512
- Keypoint at (85.3, 64.0) in pixel space

**After Step 3**:
- Keypoint at (0.167, 0.125) in normalized space [0, 1]

**Result**: Keypoints are in **[0, 1] range** for tokenization ✅

---

## 🎯 **Why This Works**

### **Mathematical Equivalence**

The 3-step process is **mathematically equivalent** to directly normalizing by the original bbox size:

```python
# Original keypoint in full image: (150, 250)
# Bbox: [100, 200, 300, 400]

# Method 1 (Our 3-step pipeline):
x_bbox_relative = 150 - 100 = 50           # Step 1: Make bbox-relative
x_resized = 50 × (512 / 300) = 85.3        # Step 2: Scale to 512×512
x_normalized = 85.3 / 512 = 0.167          # Step 3: Normalize to [0, 1]

# Method 2 (Direct bbox normalization):
x_normalized = (150 - 100) / 300 = 0.167   # Same result!
```

### **Why We Do It This Way**

1. **Image processing needs pixel coordinates**:
   - Albumentations needs pixel-space coordinates to apply transforms
   - Can't work with normalized [0, 1] values

2. **Consistent pipeline**:
   - Crop → Resize → Normalize is the standard CV pipeline
   - Same flow for training, validation, and inference

3. **Augmentation compatibility**:
   - Step 2 can include augmentations (ColorJitter, GaussNoise, etc.)
   - Augmentations work on pixel-space images

---

## 📊 **The Key Insight**

When you see this code:

```python
# Line 601-604
normalized_kpts.append([x / width, y / height])
```

**It looks like** we're dividing by full image size, but:
- `width = 512` (NOT the original image width!)
- `height = 512` (NOT the original image height!)
- These are the **post-resize dimensions** from `record["width"]` and `record["height"]`

**So we ARE normalizing by bbox size**, just in 3 steps:
1. Shift to bbox-relative coordinates
2. Scale to 512×512
3. Normalize to [0, 1]

Which is **exactly equivalent** to:
```python
# Direct bbox normalization
normalized_x = (x - bbox_x) / bbox_w
normalized_y = (y - bbox_y) / bbox_h
```

---

## ✅ **Summary**

| Step | Input | Operation | Output | Purpose |
|------|-------|-----------|--------|---------|
| **1. Crop** | Full image (1920×1080) | Crop to bbox | Bbox image (300×400) | Focus on object |
| | Keypoint (150, 250) | `kpt -= bbox_offset` | Keypoint (50, 50) | Bbox-relative coords |
| **2. Resize** | Bbox image (300×400) | Resize to 512×512 | Standard image (512×512) | Standard size |
| | Keypoint (50, 50) | `kpt × scale` | Keypoint (85.3, 64.0) | Proportional scaling |
| **3. Normalize** | Keypoint (85.3, 64.0) | `kpt / 512` | Keypoint (0.167, 0.125) | [0, 1] range |

**Final result**: Keypoint is **normalized by the original bbox size**, just through a multi-step process.

---

## 🔍 **Why It Might Be Confusing**

The variable names in Step 3 are `width` and `height`, which sound like they refer to the **original image**.

But they actually refer to the **current image dimensions** (512×512 after resize):

```python
# This is happening AFTER _apply_transforms()
record["height"] = 512  # Updated by resize
record["width"] = 512   # Updated by resize

# So when _tokenize_keypoints() is called:
height = record["height"]  # = 512 (not original image height!)
width = record["width"]    # = 512 (not original image width!)
```

---

## 💡 **To Make It Clearer (Optional Refactor)**

If you want to make the code more explicit, you could rename variables:

```python
# In _tokenize_keypoints():
resized_height = height  # Make it clear this is the resized dimension
resized_width = width

# Normalize keypoints to [0, 1]
normalized_kpts = []
for x, y in keypoints:
    normalized_kpts.append([x / resized_width, y / resized_height])
```

But this is **purely cosmetic** - the math is already correct!

---

## ✅ **Conclusion**

**YES**, we ARE normalizing by bbox size. The fact that we divide by 512 in Step 3 doesn't change this - it's just the final step in a 3-step process that ultimately normalizes coordinates relative to the **original bounding box dimensions**.

The pipeline is:
1. ✅ Crop to bbox → bbox-relative coords
2. ✅ Resize to 512×512 → scaled coords
3. ✅ Divide by 512 → normalized coords [0, 1]

Which is equivalent to:
```python
normalized_x = (x_in_full_image - bbox_x) / bbox_w
normalized_y = (y_in_full_image - bbox_y) / bbox_h
```

