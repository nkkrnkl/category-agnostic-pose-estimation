# CAPE Visualization Quick Start

## 🎨 Simple Ground Truth Visualization (Recommended)

The easiest way to visualize your dataset without loading the model:

```bash
chmod +x run_simple_visualization.sh
./run_simple_visualization.sh
```

This will:
- ✅ Load the test dataset
- ✅ Visualize 3 samples from each category
- ✅ Show keypoints and skeleton connections
- ✅ Color-code visibility (green = visible, numbered)
- ✅ Save to `visualizations/ground_truth/`

**No model loading required!** This is perfect for:
- Checking your dataset
- Understanding keypoint annotations
- Verifying skeleton connections
- Exploring different categories

---

## 🎯 Command Line Options

### Basic Usage

```bash
python visualize_results_simple.py --mode gt --dataset_root . --num_samples 5
```

### Options

- `--mode gt`: Visualize ground truth (no model needed)
- `--dataset_root .`: Path to your dataset
- `--num_samples 3`: How many images per category
- `--split_id 1`: Which MP-100 split to use (1-5)
- `--output_dir visualizations/custom`: Where to save images

### Examples

**Visualize more samples:**
```bash
python visualize_results_simple.py \
    --mode gt \
    --dataset_root . \
    --num_samples 10 \
    --output_dir visualizations/many_samples
```

**Filter by category:**
```bash
python visualize_results_simple.py \
    --mode gt \
    --dataset_root . \
    --category_filter "horse" \
    --num_samples 5
```

---

## 📁 Output Format

Each visualization shows:
- **Image**: Original test image
- **Keypoints**: Green circles with numbers
- **Skeleton**: Green lines connecting keypoints
- **Title**: Category name, image ID, keypoint counts
- **Filename**: `{category_name}_idx{dataset_idx}_id{image_id}.png`

Example output:
```
visualizations/ground_truth/
  ├── horse_body_idx123_id200000000026715.png
  ├── horse_body_idx456_id200000000026792.png
  ├── fox_body_idx789_id200000000027301.png
  └── ...
```

---

## 🔧 Troubleshooting

### Error: "Image not found"
- Some images might be missing from your dataset
- The script automatically skips missing images
- Check `annotation_cleanup_report.txt` for details

### Error: "No images for category X"
- Category might not be in the test split
- Check `category_splits.json` to see which categories are in test

### Want to visualize training data?
Change the dataset mode in the script:
```python
dataset = build_mp100_cape('train', dataset_args)  # Instead of 'test'
```

---

## 🚀 Next Steps

After visualizing ground truth:
1. ✅ Verify your annotations look correct
2. ✅ Check skeleton connections make sense
3. ✅ Train your model with `python train_cape_episodic.py ...`
4. ✅ Visualize predictions vs ground truth (coming soon)

---

## 📝 Notes

- This script **does NOT** require loading the trained model
- It only visualizes the dataset annotations
- Perfect for quick data exploration
- Much faster and simpler than the full visualization pipeline
- To visualize model predictions, you'll need the original `visualize_cape_predictions.py` (once we fix the tokenizer issue)

---

**Status**: ✅ READY TO USE

Just run `./run_simple_visualization.sh` and check the output!

