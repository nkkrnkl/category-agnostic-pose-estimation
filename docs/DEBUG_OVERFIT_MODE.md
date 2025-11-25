# Debug Overfit Mode - Single Category Training

## Overview

The debug overfit mode allows you to train on a **single category** with a **small number of episodes** to quickly verify that:
1. The model can learn (training loss decreases)
2. The data pipeline is working correctly
3. No bugs prevent optimization

This is a critical debugging tool recommended by the PhD mentor for validating the training setup.

---

## When to Use This Mode

✅ **Use debug overfit mode when:**
- Starting work on the project for the first time
- After making changes to model architecture
- After modifying the data pipeline or loss functions
- Debugging why full training isn't working
- Verifying that gradients are flowing correctly

❌ **Don't use it for:**
- Actual training or evaluation
- Hyperparameter tuning
- Measuring generalization performance

---

## How to Enable

### Basic Usage

```bash
python train_cape_episodic.py \
  --debug_overfit_category 40 \
  --debug_overfit_episodes 10 \
  --epochs 50 \
  --batch_size 2 \
  --output_dir outputs/debug_overfit
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--debug_overfit_category` | Single category ID to train on (e.g., 40 for zebra) | None (disabled) |
| `--debug_overfit_episodes` | Episodes per epoch in overfit mode | 10 |

### Finding Category IDs

To see available category IDs:

```bash
# Look at your category_splits.json
cat category_splits.json

# Or inspect annotations
python -c "
import json
with open('annotations/mp100_split1_train.json') as f:
    data = json.load(f)
    cats = {c['id']: c['name'] for c in data['categories']}
    print('Available categories:')
    for cid, name in sorted(cats.items())[:20]:
        print(f'  {cid}: {name}')
"
```

**Common Categories:**
- 1: person
- 16: bird
- 17: cat
- 18: dog
- 40: zebra
- 42: sheep
- 51: giraffe

---

## Expected Behavior

### What Should Happen (Model is Working)

```
Epoch [0] Train Loss: 45.2
Epoch [1] Train Loss: 32.1
Epoch [2] Train Loss: 18.5
Epoch [5] Train Loss: 5.2
Epoch [10] Train Loss: 1.3
Epoch [20] Train Loss: 0.15  ← Near-zero!
Epoch [30] Train Loss: 0.05
```

**✅ Success Indicators:**
- Training loss decreases consistently
- Reaches < 1.0 within 20 epochs
- Reaches < 0.1 within 50 epochs
- Validation PCK increases (if validating on same category)

### What Might Go Wrong (Debugging Needed)

```
Epoch [0] Train Loss: 45.2
Epoch [1] Train Loss: 44.9
Epoch [2] Train Loss: 44.7
Epoch [10] Train Loss: 44.1  ← Not learning!
```

**❌ Failure Indicators:**
- Loss stays high (> 30 after 10 epochs)
- Loss doesn't decrease monotonically
- Loss is NaN or inf
- Gradients are zero or very small

**Common Causes:**
- Learning rate too low (try 5e-4 instead of 1e-4)
- Data pipeline issue (check keypoints are loaded correctly)
- Model architecture bug (check forward pass)
- Loss masking too aggressive (check visibility masks)

---

## Example Debugging Session

### Step 1: Verify Data Loading

```bash
# Check that the category has enough images
python -c "
from datasets.mp100_cape import build_mp100_cape
import argparse

args = argparse.Namespace(
    dataset_root='.',
    mp100_split=1,
    semantic_classes=89,
    image_norm=False,
    vocab_size=2000,
    seq_len=200
)

dataset = build_mp100_cape('train', args)

# Count images per category
from collections import Counter
category_counts = Counter()
for idx in range(len(dataset.ids)):
    ann_ids = dataset.coco.getAnnIds(imgIds=dataset.ids[idx])
    anns = dataset.coco.loadAnns(ann_ids)
    for ann in anns:
        if 'category_id' in ann:
            category_counts[ann['category_id']] += 1

# Show top categories
print('Categories with most images:')
for cat_id, count in category_counts.most_common(10):
    print(f'  Category {cat_id}: {count} images')
"
```

### Step 2: Run Overfit Test

```bash
python train_cape_episodic.py \
  --debug_overfit_category 40 \
  --debug_overfit_episodes 10 \
  --epochs 50 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --lr 1e-4 \
  --output_dir outputs/debug_overfit_cat40
```

### Step 3: Monitor Training

Watch for:
- Loss decreasing each epoch
- By epoch 20: Loss < 1.0
- By epoch 50: Loss < 0.1

### Step 4: Visualize Results

```bash
python visualize_results_simple.py \
  --mode gt \
  --dataset_root . \
  --num_samples 5 \
  --output_dir visualizations/overfit_cat40
```

---

## Implementation Details

### How It Works Internally

When `--debug_overfit_category` is set:

1. **Creates temporary category split:**
   ```json
   {
     "train": [40],  // Only your chosen category
     "val": [],
     "test": []
   }
   ```

2. **Writes to temp file** using Python's `tempfile.mkstemp()`

3. **Passes temp file to episodic sampler** instead of `category_splits.json`

4. **Sampler only sees one category**, so all episodes use that category

5. **Episodes_per_epoch overridden** to `--debug_overfit_episodes` (default: 10)

### Why This Doesn't Require Model Changes

The `EpisodicSampler` already supports arbitrary category lists via the JSON file. By dynamically creating a JSON with just one category, we can constrain training without modifying:
- Model architecture
- Loss functions
- Data pipeline
- Evaluation code

**Only the training script** needs to be aware of this mode.

---

## Tips & Best Practices

### 1. Start with a Common Category

Pick a category with many examples (e.g., `person`, `cat`, `dog`) to ensure enough data for overfitting.

### 2. Use Small Episodes

```bash
--debug_overfit_episodes 10  # Fast iteration
--batch_size 2               # Fits in memory
--accumulation_steps 2       # Effective batch = 4
```

### 3. Increase LR Slightly

For overfitting tests, you can use a higher learning rate:

```bash
--lr 5e-4  # 5× higher for faster overfitting
```

### 4. Disable Early Stopping

```bash
--early_stopping_patience 0  # Run all 50 epochs
```

### 5. Monitor Specific Metrics

Look for:
- Training loss (should → 0)
- Training PCK (should → 100%)
- Validation loss (can increase - we're overfitting!)

---

## Troubleshooting

### "Valid categories: 0"

**Cause:** The chosen category doesn't have enough images in the training split.

**Fix:**
- Check which categories have data: see "Finding Category IDs" above
- Use a different category with more examples
- Ensure you're using the correct `mp100_split` (1-5)

### Loss Stays High

**Cause:** Model cannot overfit - indicates a bug.

**Fix:**
1. Enable debug logging:
   ```bash
   export DEBUG_CAPE=1
   python train_cape_episodic.py --debug_overfit_category 40 ...
   ```

2. Check if loss is computed:
   - Look for `loss_ce` and `loss_coords` in logs
   - Both should be > 0 and decreasing

3. Verify gradients are flowing:
   ```python
   # Add to train_cape_episodic.py temporarily:
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
   ```

### Training is Slow

**Expected:** Even with 10 episodes, each epoch takes ~30 seconds on CPU.

**Speed up:**
- Use GPU if available (automatically detected)
- Reduce `--num_queries_per_episode` to 1
- Reduce image size (not recommended, breaks model)

---

## Example: Complete Overfitting Test

```bash
#!/bin/bash
# run_overfit_test.sh - Quick sanity check that model can learn

echo "🔍 Running overfit test on category 40 (zebra)..."

python train_cape_episodic.py \
  --debug_overfit_category 40 \
  --debug_overfit_episodes 10 \
  --epochs 50 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --lr 5e-4 \
  --early_stopping_patience 0 \
  --output_dir outputs/debug_overfit_cat40 \
  --print_freq 5 \
  2>&1 | tee overfit_test.log

echo ""
echo "✅ Test complete!"
echo "📊 Check overfit_test.log for results"
echo ""
echo "Expected: Loss < 1.0 by epoch 20, Loss < 0.1 by epoch 50"
echo ""
echo "If loss stays high, there's a bug in the model/data pipeline."
```

---

## Cleaning Up After Testing

The temporary category split file is automatically cleaned up by the OS (created with `tempfile.mkstemp`).

To clean up output checkpoints:

```bash
rm -rf outputs/debug_overfit*
```

---

## Next Steps After Successful Overfitting

Once you've verified the model can overfit on one category:

1. ✅ **Confirmed**: Model architecture is correct
2. ✅ **Confirmed**: Data pipeline is working
3. ✅ **Confirmed**: Loss computation is correct
4. ✅ **Confirmed**: Optimizer is updating weights

**Now you can proceed to full training:**

```bash
python train_cape_episodic.py \
  --epochs 300 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --output_dir outputs/full_training
```

**Without** `--debug_overfit_category`, this will train on all 69 categories as normal.

