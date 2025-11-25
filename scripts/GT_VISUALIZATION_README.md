# Ground Truth Annotation Visualization

This script visualizes ground truth keypoint annotations from the MP-100 dataset's validation and test sets.

## Features

✅ **Visualize GT keypoints** on original images  
✅ **Show bounding boxes** around instances  
✅ **Draw skeleton edges** (if available)  
✅ **Color-coded keypoints** by visibility  
✅ **Category distribution** summary chart  
✅ **Filter by category** for focused analysis  
✅ **Support for all 5 MP-100 splits**  

---

## Quick Start

### Basic Usage

```bash
# Visualize 20 random validation samples
python scripts/visualize_gt_annotations.py --split val --num-samples 20

# Visualize test set
python scripts/visualize_gt_annotations.py --split test --num-samples 30

# Visualize specific category (e.g., goldenretriever_face = 48)
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 10
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--split` | str | `val` | Which split to visualize (`train`, `val`, or `test`) |
| `--data-split` | str | `split1` | Which MP-100 split to use (`split1` through `split5`) |
| `--num-samples` | int | `20` | Number of samples to visualize |
| `--category` | int | `None` | Visualize only this category ID (optional) |
| `--output-dir` | str | `outputs/gt_visualizations` | Output directory for visualizations |
| `--data-root` | str | `data` | Root directory for data |
| `--random-seed` | int | `42` | Random seed for sample selection |

---

## Examples

### Example 1: Validation Set Overview

```bash
python scripts/visualize_gt_annotations.py \
    --split val \
    --num-samples 20 \
    --output-dir outputs/gt_visualizations
```

**Output:**
- 20 random validation images with GT keypoints
- Category distribution summary chart
- Saved to: `outputs/gt_visualizations/split1_val/`

### Example 2: Test Set Samples

```bash
python scripts/visualize_gt_annotations.py \
    --split test \
    --num-samples 30
```

**Output:**
- 30 random test images
- Saved to: `outputs/gt_visualizations/split1_test/`

### Example 3: Specific Category (Golden Retriever Face)

```bash
python scripts/visualize_gt_annotations.py \
    --split val \
    --category 48 \
    --num-samples 10
```

**Output:**
- Only images with category 48 (goldenretriever_face)
- Saved to: `outputs/gt_visualizations/split1_val/cat_48/`

### Example 4: All Validation Categories

```bash
# Visualize each validation category separately
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_annotations.py \
        --split val \
        --category $cat_id \
        --num-samples 5
done
```

Or use the provided example script:

```bash
bash scripts/example_visualize_gt.sh
```

---

## Validation Categories (Split 1)

| Category ID | Name | Type | # Keypoints | # Annotations |
|-------------|------|------|-------------|---------------|
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

**Total validation annotations:** 1,795

---

## Test Categories (Split 1)

| Category ID | Name | # Annotations |
|-------------|------|---------------|
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

**Total test annotations:** 2,000

---

## Visualization Legend

### Keypoint Colors

- 🟢 **Green circles**: Labeled and visible keypoints (`visibility = 2`)
- ❌ **Red X**: Labeled but not visible keypoints (`visibility = 1`)
- (No marker): Not labeled keypoints (`visibility = 0`)

### Other Elements

- 🔵 **Cyan dashed box**: Bounding box around instance
- 🟢 **Green lines**: Skeleton edges connecting keypoints
- 🟡 **Yellow numbers**: Keypoint indices (0-indexed)

---

## Output Files

### Directory Structure

```
outputs/gt_visualizations/
├── split1_val/
│   ├── category_summary.png       # Bar chart of category distribution
│   ├── vis_0000_<filename>.png   # Individual visualizations
│   ├── vis_0001_<filename>.png
│   └── ...
│
├── split1_test/
│   └── ...
│
└── split1_val/cat_48/             # Category-specific folder
    ├── category_summary.png
    └── vis_0000_<filename>.png
```

### File Naming

- `category_summary.png`: Category distribution bar chart
- `vis_XXXX_<original_filename>.png`: Individual visualization
  - `XXXX`: 4-digit index (0000, 0001, ...)
  - `<original_filename>`: Original image filename (without extension)

---

## Use Cases

### 1. Data Quality Check
Verify that ground truth annotations are correct and properly formatted.

```bash
python scripts/visualize_gt_annotations.py --split val --num-samples 50
```

### 2. Category Analysis
Understand how different categories look and what keypoints they have.

```bash
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 20
```

### 3. Test Set Inspection
Visualize test set samples before evaluation.

```bash
python scripts/visualize_gt_annotations.py --split test --num-samples 30
```

### 4. Skeleton Verification
Check if skeleton edges are defined correctly for different categories.

```bash
# Categories with skeletons (body types)
for cat_id in 6 35 91 95; do
    python scripts/visualize_gt_annotations.py --split val --category $cat_id --num-samples 3
done
```

---

## Technical Details

### COCO Format

The MP-100 dataset uses COCO keypoint format:

```json
{
  "keypoints": [x1, y1, v1, x2, y2, v2, ...],
  "num_keypoints": 9,
  "bbox": [x, y, width, height],
  "category_id": 48,
  "image_id": 4800000000012684
}
```

### Keypoint Format

- **x, y**: Pixel coordinates in original image
- **v**: Visibility flag
  - `0` = not labeled (excluded from loss/PCK)
  - `1` = labeled but not visible (included in loss/PCK)
  - `2` = labeled and visible (included in loss/PCK)

### Skeleton Edges

- Defined in `categories` field
- Format: `[[kp1_idx, kp2_idx], ...]`
- Indices are 1-based (COCO convention)
- Only drawn if both keypoints have `v > 0`

---

## Comparison with Model Predictions

To compare ground truth with model predictions, use:

```bash
# Visualize GT
python scripts/visualize_gt_annotations.py --split val --num-samples 10

# Visualize model predictions
python scripts/eval_cape_checkpoint.py \
    --checkpoint <path> \
    --num-visualizations 10
```

This allows side-by-side comparison of:
- **GT visualizations** (this script): What the model should predict
- **Prediction visualizations** (eval script): What the model actually predicts

---

## Troubleshooting

### Issue: "Image not found"

**Cause:** Image file missing from `data/` directory  
**Solution:** Verify that images are in `data/<category>/<image_name>`

### Issue: "Annotation file not found"

**Cause:** Incorrect data split or path  
**Solution:** Check that `data/annotations/mp100_split1_val.json` exists

### Issue: Empty visualizations

**Cause:** Category filter too restrictive  
**Solution:** Use `--category None` or increase `--num-samples`

---

## Performance Notes

- **Fast:** Processes ~5-10 images per second
- **Memory:** Minimal (loads one image at a time)
- **Storage:** Each visualization ~200-500 KB

**Estimate for 50 samples:** ~30 seconds, ~20 MB

---

## Related Scripts

- **`scripts/eval_cape_checkpoint.py`**: Evaluate model and visualize predictions
- **`visualize_cape_predictions.py`**: Legacy prediction visualization
- **`scripts/example_visualize_gt.sh`**: Example usage script

---

## Quick Reference

```bash
# Validation overview (20 samples)
python scripts/visualize_gt_annotations.py --split val --num-samples 20

# Test overview (30 samples)
python scripts/visualize_gt_annotations.py --split test --num-samples 30

# Specific category (e.g., 48 = goldenretriever_face)
python scripts/visualize_gt_annotations.py --split val --category 48 --num-samples 10

# All validation categories (5 samples each)
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    python scripts/visualize_gt_annotations.py --split val --category $cat_id --num-samples 5
done
```

---

**Created:** November 25, 2025  
**Purpose:** Ground truth annotation visualization for MP-100 dataset  
**Status:** ✅ Production ready

