# Training Metrics Explanation

## What the Training Output Means

### Epoch Progress Line
```
Epoch: [1]  [   0/2882]  eta: 3:50:41  lr: 0.000100  ...
```
- **[1]**: Current epoch number
- **[0/2882]**: Current batch / total batches (2882 batches = 5765 images / 2 batch_size)
- **eta: 3:50:41**: Estimated time remaining for this epoch
- **lr: 0.000100**: Current learning rate

### Main Loss Components

**Total Loss** = Classification Loss + Coordinate Loss

#### 1. **loss_ce** (Classification Loss)
- Classifies each token in the sequence
- Tokens can be: `<coord>` (keypoint), `<sep>` (separator), `<eos>` (end of sequence)
- **Negative values** are NORMAL - this is log probability from NLL (Negative Log Likelihood) loss
- More negative = better (higher confidence in predictions)
- Values around -100 to -300 are typical

#### 2. **loss_coords** (Coordinate Regression Loss)
- Predicts the actual (x, y) coordinates of keypoints
- L1 loss between predicted and ground truth coordinates
- **Positive values** - lower is better
- Values around 0.2-1.5 are typical

#### 3. **loss_ce_room** (Category Classification Loss)
- Predicts which of the 49 categories the object belongs to
- Currently 0.0 because we're not using CLS tokens

### Auxiliary Losses (loss_ce_0 through loss_ce_4)
These are losses from intermediate decoder layers:
- The model has 6 decoder layers (numbered 0-5)
- Each layer makes predictions
- Training with these intermediate losses helps the model learn better
- **All should decrease as training progresses**

### Unscaled vs Scaled Losses
- **Scaled**: Multiplied by loss coefficients (cls_loss_coef=2, coords_loss_coef=5)
- **Unscaled**: Raw loss values before scaling
- Both are shown for debugging purposes

## What to Look For

### Good Training Signs:
1. **Total loss decreasing** over epochs
2. **Classification loss becoming more negative** (e.g., -100 → -200 → -300)
3. **Coordinate loss decreasing** (e.g., 1.2 → 0.8 → 0.5)
4. **Validation loss tracking training loss** (not diverging)

### Warning Signs:
1. **Loss becoming NaN or Inf** - training unstable
2. **Validation loss much higher than training** - overfitting
3. **Loss not decreasing** - learning rate too low or data issues

## Example Good Progression:

```
Epoch 1:  Train Loss: 66.23  Val Loss: 37.53
Epoch 2:  Train Loss: 45.12  Val Loss: 32.41
Epoch 3:  Train Loss: 35.67  Val Loss: 28.93
...
Epoch 20: Train Loss: 15.23  Val Loss: 18.45
```

The coordinate loss (loss_coords) is particularly important for pose estimation - it should steadily decrease as the model learns to predict keypoint locations more accurately.
