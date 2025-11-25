# Quick GT Visualization Guide

**TL;DR:** Visualize ground truth annotations from validation/test sets.

---

## ⚡ Quick Commands

### Basic Usage
```bash
# 20 validation samples
python scripts/visualize_gt_annotations.py --split val --num-samples 20

# 30 test samples
python scripts/visualize_gt_annotations.py --split test --num-samples 30

# Specific category (golden retriever face = 48)
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 10
```

---

## 📊 What You Get

### Output Files
```
outputs/gt_visualizations/
├── split1_val/
│   ├── category_summary.png    ← Bar chart
│   └── vis_XXXX_<filename>.png ← Visualizations
```

### Each Visualization Shows
- 🟢 Green circles: Visible keypoints (v=2)
- ❌ Red X: Not visible keypoints (v=1)
- 🔵 Cyan box: Bounding box
- 🟢 Green lines: Skeleton edges
- 🟡 Yellow numbers: Keypoint indices

---

## 🎯 Common Tasks

### Explore Validation Data
```bash
python scripts/visualize_gt_annotations.py --split val --num-samples 20
# Output: outputs/gt_visualizations/split1_val/
```

### Check Specific Category
```bash
# Category IDs (validation):
# 6=hamster, 12=horse, 22=guanaco, 35=gorilla, 48=retriever
# 66=fly, 91=beaver, 92=macaque, 95=weasel, 96=penguin

python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 10
# Output: outputs/gt_visualizations/split1_val/cat_48/
```

### Inspect Test Set
```bash
python scripts/visualize_gt_annotations.py --split test --num-samples 30
# Output: outputs/gt_visualizations/split1_test/
```

### All Validation Categories (5 each)
```bash
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_annotations.py --split val --category $cat_id --num-samples 5
done
```

---

## 📚 Full Documentation

See: `scripts/GT_VISUALIZATION_README.md`

---

## ✅ Test Results

Already tested successfully:
- ✅ 5 validation samples visualized
- ✅ 5 golden retriever faces (category 48) visualized
- ✅ Category summary charts generated
- ✅ All outputs saved correctly

**Ready to use!** 🚀

---

**Created:** Nov 25, 2025

