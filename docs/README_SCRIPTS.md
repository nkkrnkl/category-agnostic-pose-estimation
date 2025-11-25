# Scripts - CAPE Model Utilities

This folder contains standalone scripts for evaluating and analyzing CAPE models.

---

## `eval_cape_checkpoint.py` - Standalone Evaluation & Visualization

### Purpose

Comprehensive evaluation script that:
1. Loads a trained CAPE model from checkpoint
2. Runs evaluation on validation/test set using existing evaluation logic
3. Computes metrics (PCK@0.2, per-category PCK)
4. Generates side-by-side visualizations (Ground Truth vs Predicted)
5. Saves results to JSON and visualization images

### Usage

#### Basic Usage

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e020_best_pck.pth \
    --split val \
    --num-visualizations 50 \
    --output-dir outputs/cape_eval
```

#### Full Command with All Options

```bash
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e020_best_pck.pth \
    --split val \
    --num-episodes 100 \
    --num-queries-per-episode 2 \
    --pck-threshold 0.2 \
    --num-visualizations 50 \
    --draw-skeleton \
    --save-all-queries \
    --output-dir outputs/cape_eval \
    --device mps \
    --num-workers 0
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **Required** | Path to checkpoint (.pth file) |
| `--split` | str | `val` | Which split to evaluate (`train`, `val`, or `test`) |
| `--dataset-root` | str | `None` | Override dataset root (uses checkpoint args by default) |
| `--num-episodes` | int | Auto | Number of episodes to evaluate (val=100, test=200, train=50) |
| `--num-queries-per-episode` | int | Auto | Queries per episode (uses checkpoint default) |
| `--pck-threshold` | float | `0.2` | PCK threshold (fraction of bbox diagonal) |
| `--num-visualizations` | int | `50` | Maximum number of visualizations to generate |
| `--draw-skeleton` | flag | `False` | Draw skeleton edges if available |
| `--save-all-queries` | flag | `False` | Save all queries in episode (default: only first) |
| `--output-dir` | str | `outputs/cape_eval` | Directory to save outputs |
| `--device` | str | Auto | Device to use (`cpu`, `cuda`, `mps`) |
| `--num-workers` | int | `0` | Number of dataloader workers |

### Outputs

#### 1. Metrics JSON (`metrics_{split}.json`)

```json
{
  "pck_overall": 0.4521,
  "pck_per_category": {
    "12": 0.61,
    "35": 0.55,
    "95": 0.38,
    ...
  },
  "mean_pck_categories": 0.4521,
  "total_correct": 1234,
  "total_visible": 2730,
  "num_categories": 10,
  "num_images": 100,
  "threshold": 0.2,
  "checkpoint_path": "outputs/cape_run/checkpoint_e020.pth",
  "split": "val",
  "num_episodes": 100,
  "num_queries_total": 200
}
```

#### 2. Visualizations (`visualizations/vis_*.png`)

Each visualization shows three panels side-by-side:

**Left Panel - Support (GT):**
- Shows the support image
- Green circles: ground truth keypoints
- Skeleton edges (if `--draw-skeleton` enabled)
- Label: "Support (GT)"

**Middle Panel - Ground Truth:**
- Shows the query image
- Cyan circles: ground truth keypoints
- Skeleton edges (if `--draw-skeleton` enabled)
- Label: "Ground Truth"

**Right Panel - Predicted:**
- Shows the query image
- Red X marks: predicted keypoints
- Skeleton edges (if `--draw-skeleton` enabled)
- PCK score for this example
- Label: "Predicted"

**Filename format:**
```
vis_{episode_idx:04d}_q{query_idx}_cat{category_id}_img{image_id}.png
```

Example: `vis_0042_q0_cat12_img1200000000019572.png`

### Features

#### Automatic Device Detection

The script automatically detects and uses the best available device:
1. CUDA GPU (if available)
2. Apple Silicon MPS (if available)
3. CPU (fallback)

You can override with `--device cpu/cuda/mps`.

#### Reuses Existing Code

The script leverages existing project components:
- `engine_cape.py::extract_keypoints_from_sequence` - Keypoint extraction
- `util/eval_utils.py::PCKEvaluator` - PCK computation
- `datasets/episodic_sampler.py::build_episodic_dataloader` - Data loading
- `models/` - Model building and loading

This ensures consistency with training evaluation.

#### Handles Old Checkpoints

If you use a checkpoint trained with the old (buggy) code, the script will:
- Detect that predictions are shorter than expected
- Display a warning about the checkpoint quality
- Pad predictions for visualization (but warn that metrics are invalid)

```
⚠️  CRITICAL WARNING: OLD CHECKPOINT DETECTED
   This checkpoint was trained WITHOUT a tokenizer and only generates
   1 keypoint(s) before predicting <eos>.
   
   Recommendation: Use a checkpoint from the FIXED training code
```

#### Coordinate Handling

The script properly handles coordinate transformations:
- Model outputs: Normalized [0,1] relative to bbox
- Visualization: Pixel coordinates [0, image_size]
- PCK computation: Uses bbox dimensions from metadata

All coordinate systems are correctly aligned.

### Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================

Overall PCK@0.2: 0.4521 (45.21%)
  Correct keypoints: 1234 / 2730
  Mean PCK across categories: 0.4521 (45.21%)

Per-Category PCK:
  Category ID     PCK        Correct/Total  
  ---------------  ----------  ---------------
  12               0.6100      153/250        
  35               0.5500      110/200        
  95               0.3800      76/200         
  ...

Top 5 performing categories:
  cat_id=12: 61.00%
  cat_id=35: 55.00%
  cat_id=40: 52.00%
  cat_id=18: 48.00%
  cat_id=95: 38.00%

Bottom 5 performing categories:
  cat_id=67: 28.00%
  cat_id=82: 25.00%
  cat_id=91: 22.00%
  cat_id=14: 20.00%
  cat_id=99: 18.00%

Visualizations saved to: outputs/cape_eval/visualizations
Metrics saved to: outputs/cape_eval/metrics_val.json
```

### Common Use Cases

#### 1. Evaluate Best Checkpoint

```bash
# Evaluate your best model
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck.pth \
    --split val \
    --num-visualizations 100
```

#### 2. Quick Evaluation (Few Samples)

```bash
# Fast evaluation with minimal samples
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010.pth \
    --split val \
    --num-episodes 10 \
    --num-visualizations 5
```

#### 3. Test Set Evaluation

```bash
# Evaluate on test split
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_best_pck.pth \
    --split test \
    --num-episodes 200 \
    --num-visualizations 100 \
    --draw-skeleton
```

#### 4. Compare Multiple Checkpoints

```bash
# Evaluate epoch 10
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e010.pth \
    --output-dir outputs/eval_e010

# Evaluate epoch 20
python scripts/eval_cape_checkpoint.py \
    --checkpoint outputs/cape_run/checkpoint_e020.pth \
    --output-dir outputs/eval_e020

# Compare metrics_val.json files
```

### Sanity Checks

The script performs several sanity checks:
1. ✅ Verifies model has `forward_inference` method
2. ✅ Checks coordinate shapes match expectations
3. ✅ Validates predictions are within valid range
4. ✅ Warns if predictions are shorter than expected
5. ✅ Logs detailed error messages if extraction fails

### Troubleshooting

#### Issue: PCK is 100% or 0%

**If PCK@100%:**
- Checkpoint might be from old (buggy) training code
- See warning message about old checkpoints
- Solution: Use checkpoint from fixed training code

**If PCK@0%:**
- Model might not have trained properly
- Check training logs for actual validation performance
- Try visualizing to see what model is predicting

#### Issue: No visualizations generated

**Possible causes:**
1. Keypoint extraction failed for all episodes
2. Check console for error messages
3. Try with `--num-episodes 20` for more samples

**Solution:**
```bash
# Run with verbose output
python scripts/eval_cape_checkpoint.py \
    --checkpoint <path> \
    --num-episodes 20 \
    --num-visualizations 10 \
    2>&1 | tee eval_debug.log
```

#### Issue: Images look wrong

**Check:**
1. Are support and query images from the same category?
2. Are keypoints in the correct pixel coordinates?
3. Are skeleton edges appropriate for the category?

**Debug:**
- Check `query_metadata` in printed output
- Verify bbox dimensions match image size
- Check if visibility masks are correct

###Dependencies

Required packages (already in `requirements_cape.txt`):
- `torch` - PyTorch
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `opencv-python` (cv2) - Image processing
- `pillow` - Image loading
- `tqdm` - Progress bars

### Integration with Training

This script uses the **same evaluation logic** as training:
- `engine_cape.py::extract_keypoints_from_sequence`
- `util/eval_utils.py::PCKEvaluator`
- `util/eval_utils.py::compute_pck_bbox`

This ensures consistency between training validation and standalone evaluation.

### Performance

**Timing:**
- ~3-4 seconds per episode on CPU
- ~1-2 seconds per episode on GPU
- ~2-3 seconds per episode on MPS (Apple Silicon)

**For 100 episodes:**
- CPU: ~5-6 minutes
- GPU: ~2-3 minutes
- MPS: ~3-4 minutes

### Related Files

- `../visualize_cape_predictions.py` - Original visualization script (test set focus)
- `../engine_cape.py` - Training evaluation logic
- `../util/eval_utils.py` - PCK computation utilities
- `../tests/test_tokenizer_fix_simple.py` - Verify model has tokenizer

### Version History

- **v1.0** (2025-11-25): Initial release
  - Full evaluation pipeline
  - Side-by-side visualizations
  - JSON metrics export
  - Handles old/new checkpoints
  - Automatic device detection

---

## Future Scripts

Planned additions to this folder:
- `compare_checkpoints.py` - Compare multiple checkpoints side-by-side
- `analyze_errors.py` - Analyze failure cases (low PCK examples)
- `export_for_demo.py` - Export model for demo/deployment

---

Last updated: 2025-11-25

