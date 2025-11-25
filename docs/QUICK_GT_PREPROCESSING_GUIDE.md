# Quick GT Preprocessing Verification Guide

**TL;DR:** Verify that bbox cropping + resizing to 512×512 preserves all GT annotations correctly.

---

## ⚡ Quick Commands

```bash
# Visualize 10 validation samples (original vs. preprocessed)
python scripts/visualize_gt_preprocessing.py --split val --num-samples 10

# Check specific category (golden retriever = 48)
python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 5

# Verify test set
python scripts/visualize_gt_preprocessing.py --split test --num-samples 10
```

---

## 🎨 What You Get

### Side-by-Side Panels

**LEFT:** Original unscaled image
- Full-size image (e.g., 1844×1224)
- 🔵 Cyan dashed box = bounding box
- 🟢 Green circles = visible keypoints
- ❌ Red X = not visible keypoints

**RIGHT:** Preprocessed image (EXACTLY as model sees it)
- Cropped to bbox + resized to 512×512
- Same keypoints (transformed coordinates)
- **This is what the model trains on!**

---

## ✅ What to Check

When viewing visualizations, verify:

- ✓ All keypoints are **inside** the bbox (LEFT panel)
- ✓ Keypoints are correctly positioned in 512×512 image (RIGHT panel)
- ✓ Skeleton edges match in both panels
- ✓ No keypoints cut off during cropping
- ✓ 512×512 images look reasonable

---

## 📁 Output Location

```
outputs/gt_preprocessing_vis/split1_val/
  - vis_0000_<filename>.png    # Side-by-side visualizations
  - vis_0001_<filename>.png
  - ...
```

---

## 🎯 Use Cases

### Before Training
```bash
# Verify preprocessing for all val categories
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_preprocessing.py --split val --category $cat_id --num-samples 3
done
```

### Debug Bbox Issues
```bash
# If training looks weird, check bbox quality
python scripts/visualize_gt_preprocessing.py --split val --num-samples 20
```

### Specific Category Analysis
```bash
# Category IDs (validation):
# 6=hamster, 12=horse, 22=guanaco, 35=gorilla, 48=retriever
# 66=fly, 91=beaver, 92=macaque, 95=weasel, 96=penguin

python scripts/visualize_gt_preprocessing.py --split val --category 48 --num-samples 10
```

---

## 🔄 Preprocessing Steps (Verified)

This script shows the **EXACT** preprocessing pipeline:

1. **Crop to bbox:** Extract bbox region from original image
2. **Relative coords:** Make keypoints relative to bbox top-left
3. **Resize to 512×512:** Use Albumentations (transforms keypoints too)

**Result:** 512×512 image with keypoints in [0, 512]×[0, 512]

---

## ✅ Test Results

Already tested:
- ✅ 5 validation samples (mixed categories)
- ✅ 3 golden retriever faces (category 48)
- ✅ All visualizations saved correctly
- ✅ Preprocessing verified working

**Ready to use!** 🚀

---

## 📚 Full Documentation

See: **`GT_PREPROCESSING_VIS_COMPLETE.md`**

---

**Created:** Nov 25, 2025

