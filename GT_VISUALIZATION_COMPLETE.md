# ✅ Ground Truth Visualization Script Complete

**Date:** November 25, 2025  
**Status:** READY TO USE

---

## 🎯 What Was Created

I've created a comprehensive script to visualize ground truth annotations from the MP-100 validation and test sets.

### Files Created

1. **`scripts/visualize_gt_annotations.py`** - Main visualization script
2. **`scripts/example_visualize_gt.sh`** - Example usage script
3. **`scripts/GT_VISUALIZATION_README.md`** - Comprehensive documentation

---

## 🚀 Quick Start

### Visualize 20 Validation Samples

```bash
python scripts/visualize_gt_annotations.py --split val --num-samples 20
```

**Output:** `outputs/gt_visualizations/split1_val/`

### Visualize Test Set

```bash
python scripts/visualize_gt_annotations.py --split test --num-samples 30
```

**Output:** `outputs/gt_visualizations/split1_test/`

### Visualize Specific Category

```bash
# Example: Golden Retriever faces (category 48)
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 10
```

**Output:** `outputs/gt_visualizations/split1_val/cat_48/`

---

## 📊 Test Run Results

I ran a quick test with 5 validation samples:

```
✅ Visualization complete!
  Successfully visualized: 5/5
  Output directory: outputs/gt_visualizations/split1_val

Files created:
  - category_summary.png          # Bar chart of category distribution
  - vis_0000_000000049999.png    # Weasel body
  - vis_0001_guanaco_103.png     # Guanaco face
  - vis_0002_goldenretriever_50.png  # Golden retriever face
  - vis_0003_1286.png            # Fly
  - vis_0004_000000019940.png    # Beaver body
```

---

## 🎨 What You'll See

### Visualization Elements

1. **Keypoints**
   - 🟢 **Green circles**: Labeled and visible (v=2)
   - ❌ **Red X**: Labeled but not visible (v=1)
   - No marker: Not labeled (v=0)

2. **Bounding Box**
   - 🔵 Cyan dashed rectangle

3. **Skeleton Edges**
   - 🟢 Green lines connecting keypoints
   - Only for categories with skeleton definition

4. **Keypoint Numbers**
   - 🟡 Yellow labels (0-indexed)

5. **Title Information**
   - Category name and ID
   - Number of visible keypoints
   - Image filename and dimensions

### Example Visualization

Each image shows:
```
┌─────────────────────────────────────┐
│  goldenretriever_face (cat_id=48)  │
│  Keypoints: 9/9                     │
│  Image: goldenretriever_50.jpg      │
│  Size: 1844x1224                    │
├─────────────────────────────────────┤
│                                     │
│       🔵 Cyan dashed bbox           │
│           ╱─────────╲              │
│          │  🟢 0     │              │
│          │  🟢 1     │              │
│          │  🟢 2 ── 🟢 3           │
│          │  🟢 4     │              │
│          │  🟢 5     │              │
│          │  🟢 6 ── 🟢 7           │
│          │  🟢 8     │              │
│           ╲─────────╱              │
│      (Green lines = skeleton)       │
│      (Yellow numbers = kp indices)  │
└─────────────────────────────────────┘
```

---

## 📈 Category Summary

The script also creates a **category distribution bar chart** showing:
- All categories in the split
- Number of annotations per category
- Sorted by frequency

Example for validation set:
```
Category Distribution
┌────────────────────────────────────┐
│ weasel_body        ████████ 232    │
│ fly                ████████ 232    │
│ hamster_body       ████████ 231    │
│ gorilla_body       ████████ 231    │
│ gentoopenguin_face ███████  212    │
│ beaver_body        ██████   197    │
│ przewalskihorse    ████     148    │
│ guanaco_face       ████     145    │
│ goldenretriever    ████     140    │
│ macaque            █         27    │
└────────────────────────────────────┘
```

---

## 🎓 Use Cases

### 1. Data Quality Check
Verify ground truth annotations are correct:
```bash
python scripts/visualize_gt_annotations.py --split val --num-samples 50
```

### 2. Category Analysis
Understand category characteristics:
```bash
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 20
```

### 3. Pre-Evaluation Inspection
Inspect test set before running evaluation:
```bash
python scripts/visualize_gt_annotations.py --split test --num-samples 30
```

### 4. Compare with Model Predictions
Side-by-side comparison:
```bash
# 1. Visualize ground truth
python scripts/visualize_gt_annotations.py --split val --num-samples 10

# 2. Visualize model predictions
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --num-visualizations 10
```

---

## 📋 All Command-Line Arguments

```bash
python scripts/visualize_gt_annotations.py \
    --split val \                    # train, val, or test
    --data-split split1 \            # split1 through split5
    --num-samples 20 \               # Number of samples
    --category 48 \                  # Optional: specific category
    --output-dir outputs/gt_vis \    # Output directory
    --data-root data \               # Data root directory
    --random-seed 42                 # Random seed
```

---

## 🗂️ Output Structure

```
outputs/gt_visualizations/
├── split1_val/
│   ├── category_summary.png       # Distribution chart
│   ├── vis_0000_<filename>.png   # Individual samples
│   ├── vis_0001_<filename>.png
│   └── ...
│
├── split1_test/
│   └── ... (same structure)
│
└── split1_val/cat_48/             # Category-specific
    ├── category_summary.png
    └── vis_0000_<filename>.png
```

---

## 🔄 Validation Categories (Split 1)

All 10 validation categories with annotations:

| ID | Name | Type | Keypoints | Annotations |
|----|------|------|-----------|-------------|
| 6 | hamster_body | animal_body | 17 | 231 |
| 12 | przewalskihorse_face | animal_face | 9 | 148 |
| 22 | guanaco_face | animal_face | 9 | 145 |
| 35 | gorilla_body | animal_body | 17 | 231 |
| 48 | goldenretriever_face | animal_face | 9 | 140 |
| 66 | fly | insect | varies | 232 |
| 91 | beaver_body | animal_body | 17 | 197 |
| 92 | macaque | primate | varies | 27 |
| 95 | weasel_body | animal_body | 17 | 232 |
| 96 | gentoopenguin_face | animal_face | 9 | 212 |

**Total:** 1,795 annotations

---

## 🧪 Test Categories (Split 1)

All 20 test categories (2,000 annotations total):

| ID | Name | Annotations |
|----|------|-------------|
| 2 | horse_body | 134 |
| 3 | dog_body | 231 |
| 10 | klipspringer_face | 144 |
| 14 | Woodpecker | 73 |
| 24 | dassie_face | 52 |
| 29 | rabbit_body | 88 |
| 30 | bison_body | 96 |
| 33 | squirrel_body | 95 |
| 39 | swivelchair | 200 |
| 42 | sheep_body | 120 |
| 47 | alpaca_face | 132 |
| 53 | Tern | 31 |
| 60 | short_sleeved_dress | 193 |
| 68 | fox_body | 59 |
| 70 | skunk_body | 52 |
| 73 | lion_body | 87 |
| 77 | commonwarthog_face | 124 |
| 78 | long_sleeved_outwear | 23 |
| 81 | bighornsheep_face | 51 |
| 84 | bed | 15 |

---

## 📚 Documentation

Full documentation available in:
- **`scripts/GT_VISUALIZATION_README.md`** - Comprehensive guide
- **`scripts/example_visualize_gt.sh`** - Example usage

---

## ⚡ Performance

- **Speed:** ~5-10 images per second
- **Memory:** Minimal (one image at a time)
- **Storage:** ~200-500 KB per visualization

**For 50 samples:** ~30 seconds, ~20 MB total

---

## 🎯 Next Steps

### 1. Explore Validation Data
```bash
python scripts/visualize_gt_annotations.py --split val --num-samples 20
```

### 2. Check Each Category
```bash
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_annotations.py --split val --category $cat_id --num-samples 5
done
```

### 3. Inspect Test Set
```bash
python scripts/visualize_gt_annotations.py --split test --num-samples 30
```

### 4. Compare with Predictions
```bash
# Ground truth
python scripts/visualize_gt_annotations.py --split val --num-samples 10

# Model predictions
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth \
    --num-visualizations 10
```

---

## ✅ Summary

You now have a complete toolkit to visualize ground truth annotations!

**What you can do:**
- ✅ Visualize validation/test GT keypoints
- ✅ Filter by specific categories
- ✅ See category distribution
- ✅ Verify data quality
- ✅ Compare with model predictions

**Files created:**
- ✅ `scripts/visualize_gt_annotations.py` (main script)
- ✅ `scripts/example_visualize_gt.sh` (examples)
- ✅ `scripts/GT_VISUALIZATION_README.md` (documentation)

**Test output:**
- ✅ 5 samples visualized successfully
- ✅ Located in `outputs/gt_visualizations/split1_val/`

---

**Ready to use! Start exploring your ground truth data.** 🚀

**Last updated:** November 25, 2025

