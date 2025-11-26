# Single Image Training Mode

This guide explains how to train the model on a single image from a specific category. This is useful for:
- **Extreme overfitting tests**: Verify the model can memorize a single image
- **Debugging**: Isolate issues to a single data point
- **Quick sanity checks**: Test training pipeline without waiting for full dataset

## 🚀 Quick Start

### Train on First Image from Category 40

```bash
python -m models.train_cape_episodic \
    --debug_single_image 40 \
    --dataset_root . \
    --category_split_file category_splits.json \
    --output_dir output/single_image_test \
    --epochs 50 \
    --batch_size 1 \
    --num_queries_per_episode 1 \
    --episodes_per_epoch 20
```

### Train on Specific Image Index

If you know the specific image index within a category:

```bash
python -m models.train_cape_episodic \
    --debug_single_image 40 \
    --debug_single_image_index 5 \
    --dataset_root . \
    --category_split_file category_splits.json \
    --output_dir output/single_image_test \
    --epochs 50
```

## 📋 Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--debug_single_image` | Category ID to use | `40` (zebra) |
| `--debug_single_image_index` | Image index within category (optional) | `0` (first image) |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size (recommend 1 for single image) | `1` |
| `--num_queries_per_episode` | Queries per episode (recommend 1) | `1` |

## 🔍 How It Works

In single-image mode:
1. **Finds the image**: Locates the specified image from the category
2. **Self-supervised**: Uses the same image as both **support** and **query**
3. **Overfitting test**: Model should achieve near-zero loss within ~10 epochs
4. **Same-image validation**: Validation also uses the **same image** (not unseen categories)

### Example Episode Structure

```
Training Episode:
  Support Image: Image #123 from category 40 (zebra)
  Query Image:   Image #123 from category 40 (zebra)  [SAME IMAGE]
  
Validation Episode:
  Support Image: Image #123 from category 40 (zebra)  [SAME IMAGE]
  Query Image:   Image #123 from category 40 (zebra)  [SAME IMAGE]
  
  The model learns to predict keypoints on the query image
  using the support pose graph from the same image.
  Validation tests if the model can perfectly memorize this single image.
```

### Important Notes

- **Training**: Same image used as both support and query
- **Validation**: Same image used as both support and query (for overfitting verification)
- **Purpose**: Verify the model can learn and memorize a single image perfectly
- **Expected PCK**: Should reach 100% PCK on the same image after overfitting

## ✅ Expected Results

### Training Loss Progression

| Epoch | Expected Loss | Validation PCK | Notes |
|-------|---------------|----------------|-------|
| 1-5   | 10-50        | 0-20%         | Initial learning |
| 5-10  | 1-10         | 20-70%        | Rapid improvement |
| 10-20 | < 1.0        | 70-95%        | Near-perfect fit |
| 20+   | < 0.1        | 95-100%       | Overfitting complete |

### Success Criteria

✅ **Model can learn**: Loss drops to < 0.1 within 20 epochs  
✅ **Perfect memorization**: Validation PCK reaches ~100% (same image)  
✅ **No crashes**: Training completes without errors  
✅ **Memory stable**: No OOM errors  

❌ **If loss stays high**: There's likely a bug in the model/data pipeline  
❌ **If validation PCK stays low**: Model isn't learning (check architecture/optimization)

## 🎯 Use Cases

### 1. Debug Model Architecture

```bash
# Test if model can learn at all
python -m models.train_cape_episodic \
    --debug_single_image 40 \
    --epochs 20 \
    --output_dir output/debug_model
```

**Expected**: Loss → 0  
**If not**: Check model architecture, loss function, data loading

### 2. Test Data Pipeline

```bash
# Verify data loading works correctly
python -m models.train_cape_episodic \
    --debug_single_image 40 \
    --epochs 5 \
    --print_freq 1
```

**Check logs for**:
- Image loading success
- Keypoint extraction
- Sequence tokenization
- No shape mismatches

### 3. Quick Hyperparameter Test

```bash
# Test different learning rates quickly
for lr in 1e-4 5e-4 1e-3; do
    python -m models.train_cape_episodic \
        --debug_single_image 40 \
        --lr $lr \
        --epochs 20 \
        --output_dir output/lr_test_${lr}
done
```

## 🔧 Troubleshooting

### "No images found for category X"

**Problem**: Category doesn't exist or has no images

**Solution**:
1. Check category exists: `grep "category_id" annotations/mp100_split1_train.json | grep "X"`
2. Try a different category (common ones: 40=zebra, 1=person, 5=dog)

### "Image index out of range"

**Problem**: Requested image index doesn't exist in category

**Solution**:
- Don't specify `--debug_single_image_index` (uses first image)
- Or check how many images category has first

### Loss Doesn't Decrease

**Possible causes**:
1. **Learning rate too low**: Try `--lr 5e-4`
2. **Gradient clipping too aggressive**: Try `--clip_max_norm 1.0`
3. **Model bug**: Check model architecture
4. **Data bug**: Verify keypoints are loaded correctly

### Out of Memory

**Solution**:
```bash
# Reduce batch size and disable AMP
python -m models.train_cape_episodic \
    --debug_single_image 40 \
    --batch_size 1 \
    --no-use_amp
```

## 📊 Example Output

```
================================================================================
⚠️  DEBUG SINGLE IMAGE MODE ENABLED
================================================================================
Training on SINGLE IMAGE from category: 40
Selected image index: 1234 (image 1 of 15 in category)
Episodes per epoch: 20
Expected: Training loss → 0 within ~10 epochs
Purpose: Extreme overfitting test on single image
================================================================================

⚠️  SINGLE IMAGE VALIDATION MODE
================================================================================
Validation will use the SAME image as training:
  - Image index: 1234
  - Category ID: 40
  - Same image used as both support and query (self-supervised)
  - Purpose: Verify model can perfectly memorize single image
================================================================================

Epoch: [1]
  loss: 25.4321  loss_ce: 12.3456  loss_coords: 13.0865
  Val PCK@0.2: 15.23% (same image, autoregressive)
  ...
  
Epoch: [10]
  loss: 0.5432  loss_ce: 0.2345  loss_coords: 0.3087
  Val PCK@0.2: 78.45% (same image, autoregressive)
  ...
  
Epoch: [20]
  loss: 0.0123  loss_ce: 0.0045  loss_coords: 0.0078
  Val PCK@0.2: 98.76% (same image, autoregressive)
  ✓ Overfitting complete! Model successfully memorized the single image.
```

## 🔗 Related Features

- **Single Category Mode**: `--debug_overfit_category` (trains on all images from one category)
- **Full Training**: Remove debug flags for normal training

## 📝 Notes

- **Self-supervised**: Same image used as support and query
- **Fast**: Training completes in minutes (not hours)
- **Memory efficient**: Single image uses minimal GPU memory
- **Reproducible**: Same image every time (no randomness in image selection)

---

**Last Updated**: 2025-01-XX

