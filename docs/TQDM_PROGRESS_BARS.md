# ✅ TQDM Progress Bars for Training

Added visual progress bars using `tqdm` to make training progression clearer and easier to monitor.

---

## 🎯 What Was Added

### 1. **Training Progress Bar**

Shows real-time progress during training with key metrics:

```
Epoch 0: 100%|████████| 500/500 [10:23<00:00, 1.25s/it, loss=28.5, loss_ce=0.14, loss_coords=4.8, lr=0.0001]
```

**Displays:**
- Current iteration / total iterations
- Time elapsed and estimated time remaining
- Training speed (iterations per second)
- **Current loss** (total loss)
- **loss_ce** (classification loss)
- **loss_coords** (coordinate regression loss)
- **Learning rate**

### 2. **Validation Progress Bar**

Shows validation progress with losses:

```
Validation: 100%|████████| 100/100 [02:15<00:00, 1.35s/it, loss=24.3, loss_ce=0.12, loss_coords=4.2]
```

**Displays:**
- Validation progress
- Current validation losses
- Processing speed

### 3. **Test (Unseen Categories) Progress Bar**

Shows test progress with real-time PCK metric:

```
Test (Unseen) PCK@0.2: 100%|████████| 50/50 [01:30<00:00, 1.8s/it, PCK=45.23%, correct=1234, visible=2729]
```

**Displays:**
- Test progress on unseen categories
- **Real-time PCK** (Percentage of Correct Keypoints)
- Number of correct predictions
- Total visible keypoints

---

## 📁 Files Modified

### `engine_cape.py`

**Changes:**
1. Added `from tqdm import tqdm` import
2. Wrapped training data loader with tqdm progress bar
3. Added `pbar.set_postfix()` to update metrics in real-time
4. Wrapped validation data loader with tqdm
5. Wrapped test data loader with tqdm
6. Updated all progress bars with relevant metrics

---

## 🎨 Visual Improvements

### Before (Without tqdm)

```
Epoch: [0]  [  0/500]  eta: 1:05:15  lr: 0.000100  loss: 17.9322 ...
Epoch: [0]  [ 10/500]  eta: 0:18:38  lr: 0.000100  loss: 30.3498 ...
Epoch: [0]  [ 20/500]  eta: 0:15:24  lr: 0.000100  loss: 27.6009 ...
```
- Hard to see overall progress
- Cluttered with many metrics
- No visual progress bar

### After (With tqdm)

```
Epoch 0: 100%|███████████████████| 500/500 [10:23<00:00, 1.25s/it]
  loss=28.5, loss_ce=0.14, loss_coords=4.8, lr=0.0001

Validation: 100%|████████████████| 100/100 [02:15<00:00, 1.35s/it]
  loss=24.3, loss_ce=0.12, loss_coords=4.2

Test (Unseen) PCK@0.2: 100%|████| 50/50 [01:30<00:00, 1.8s/it]
  PCK=45.23%, correct=1234, visible=2729
```
- ✅ Clear visual progress bar
- ✅ Easy to see completion percentage
- ✅ Time remaining estimates
- ✅ Key metrics at a glance

---

## 🚀 How to Use

Simply run training as normal:

```bash
python train_cape_episodic.py \
  --dataset_root . \
  --epochs 5 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_queries_per_episode 2 \
  --early_stopping_patience 20 \
  --output_dir ./outputs/cape_run
```

**You'll see:**

1. **Epoch Progress Bar** during training
   - Updates every iteration with current loss
   - Shows estimated time to completion

2. **Validation Progress Bar** after each epoch
   - Shows validation losses
   - Indicates validation speed

3. **Test Progress Bar** (if running test evaluation)
   - Shows real-time PCK metric
   - Tracks number of correct predictions

---

## 📊 What Gets Displayed

### Training Metrics

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `loss` | Total loss | 28.5 |
| `loss_ce` | Classification loss | 0.14 |
| `loss_coords` | Coordinate regression loss | 4.8 |
| `lr` | Current learning rate | 0.0001 |

### Validation Metrics

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `loss` | Total validation loss | 24.3 |
| `loss_ce` | Classification loss | 0.12 |
| `loss_coords` | Coordinate loss | 4.2 |

### Test Metrics (Unseen Categories)

| Metric | Description | Example Value |
|--------|-------------|---------------|
| `PCK` | Percentage of Correct Keypoints | 45.23% |
| `correct` | Number of correct predictions | 1234 |
| `visible` | Total visible keypoints | 2729 |

---

## 🎯 Benefits

1. **Better Visibility**: Clear visual feedback on training progress
2. **Time Estimates**: Know how long until epoch/validation completion
3. **Real-Time Metrics**: See current loss/PCK without waiting for print statements
4. **Professional Look**: Clean, modern progress bars
5. **Less Clutter**: Only shows key metrics in progress bar, full stats still printed at end

---

## 🔧 Technical Details

### Implementation

**Training Loop (`train_one_epoch_episodic`)**:
```python
pbar = tqdm(data_loader, desc=f'Epoch {epoch}', leave=True, ncols=120)

for batch_idx, batch in enumerate(pbar):
    # ... training code ...
    
    # Update progress bar with current metrics
    pbar.set_postfix({
        'loss': f'{loss_value:.4f}',
        'loss_ce': f'{loss_dict_reduced_scaled.get("loss_ce", 0):.4f}',
        'loss_coords': f'{loss_dict_reduced_scaled.get("loss_coords", 0):.4f}',
        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
    })
```

**Validation Loop (`evaluate_cape`)**:
```python
pbar = tqdm(data_loader, desc='Validation', leave=True, ncols=120)

for batch in pbar:
    # ... validation code ...
    
    pbar.set_postfix({
        'loss': f'{val_loss:.4f}',
        'loss_ce': f'{loss_dict_reduced_scaled.get("loss_ce", 0):.4f}',
        'loss_coords': f'{loss_dict_reduced_scaled.get("loss_coords", 0):.4f}'
    })
```

**Test Loop (`evaluate_unseen_categories`)**:
```python
pbar = tqdm(data_loader, desc=f'Test (Unseen) PCK@{pck_threshold}', leave=True, ncols=120)

for batch in pbar:
    # ... test code ...
    
    current_results = pck_evaluator.get_results()
    pbar.set_postfix({
        'PCK': f'{current_results["pck_overall"]:.2%}',
        'correct': current_results['total_correct'],
        'visible': current_results['total_visible']
    })
```

### Parameters

- `desc`: Description shown before the progress bar
- `leave=True`: Keep progress bar on screen after completion
- `ncols=120`: Width of the progress bar (120 characters)
- `set_postfix()`: Update metrics shown at the end of the bar

---

## ✅ Summary

- ✅ **Training** now shows clear progress bar with loss/lr
- ✅ **Validation** shows progress with validation losses
- ✅ **Test** shows progress with real-time PCK metric
- ✅ **All metrics** update in real-time as training progresses
- ✅ **Time estimates** help plan training duration
- ✅ **Original logging** still works (summary printed at end of epoch)

**Training is now much easier to monitor!** 🎉

