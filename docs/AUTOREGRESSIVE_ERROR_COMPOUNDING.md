# Autoregressive Error Compounding Explained

## What is Autoregressive Generation?

**Autoregressive generation** means the model generates a sequence **one element at a time**, where each new element depends on all the previous elements.

Think of it like writing a sentence:
- You write word 1
- Then you write word 2 (based on word 1)
- Then you write word 3 (based on words 1 and 2)
- And so on...

## Teacher Forcing vs Autoregressive Generation

### Teacher Forcing (Training Mode)

During **training**, the model uses **teacher forcing**:

```
Time Step 1: Model predicts keypoint_1, but sees GT keypoint_1
Time Step 2: Model predicts keypoint_2, but sees GT keypoint_1 AND GT keypoint_2
Time Step 3: Model predicts keypoint_3, but sees GT keypoint_1, GT keypoint_2, AND GT keypoint_3
...
```

**Key point:** The model always sees the **correct** (ground truth) previous keypoints, even if it predicted them wrong.

**Example:**
```
Ground Truth:  [head, neck, shoulder, elbow, wrist]
Model Predicts: [head, neck, shoulder, elbow, wrist]

At step 3, even if model predicted wrong shoulder, it still sees:
- GT head (correct)
- GT neck (correct)  
- GT shoulder (correct) ← Always correct!
```

### Autoregressive Generation (Validation/Inference Mode)

During **validation/inference**, the model uses **autoregressive generation**:

```
Time Step 1: Model predicts keypoint_1 → uses this prediction
Time Step 2: Model predicts keypoint_2 → uses PREDICTED keypoint_1 (not GT!)
Time Step 3: Model predicts keypoint_3 → uses PREDICTED keypoint_1 and keypoint_2
...
```

**Key point:** The model uses its **own predictions** as input for the next step, not ground truth.

**Example:**
```
Ground Truth:  [head, neck, shoulder, elbow, wrist]
Model Predicts: [head, neck, shoulder, elbow, wrist]

At step 3, model uses:
- PREDICTED head (might be wrong!)
- PREDICTED neck (might be wrong!)
- Then predicts shoulder based on these (potentially wrong) inputs
```

## How Errors Compound

### Example: Predicting a Face (5 keypoints)

Let's say we're predicting keypoints for a face in this order:
1. **Left eye** (x=0.3, y=0.4)
2. **Right eye** (x=0.7, y=0.4) - depends on left eye position
3. **Nose** (x=0.5, y=0.6) - depends on both eyes
4. **Left mouth corner** (x=0.4, y=0.8) - depends on nose and eyes
5. **Right mouth corner** (x=0.6, y=0.8) - depends on all previous

### Scenario 1: Perfect Generation (No Errors)

```
Step 1: Predict left_eye = (0.3, 0.4) ✅ CORRECT
Step 2: Predict right_eye = (0.7, 0.4) ✅ CORRECT (used correct left_eye)
Step 3: Predict nose = (0.5, 0.6) ✅ CORRECT (used correct left_eye + right_eye)
Step 4: Predict left_mouth = (0.4, 0.8) ✅ CORRECT (used correct previous keypoints)
Step 5: Predict right_mouth = (0.6, 0.8) ✅ CORRECT (used correct previous keypoints)

Result: All 5 keypoints correct → PCK = 100%
```

### Scenario 2: Error in First Keypoint (Compounds!)

```
Step 1: Predict left_eye = (0.5, 0.4) ❌ WRONG! (should be 0.3, 0.4)
         Error: +0.2 in x-direction

Step 2: Predict right_eye = (0.9, 0.4) ❌ WRONG!
         Why? Model thinks left_eye is at (0.5, 0.4), so it places right_eye
         relative to that. But left_eye is actually at (0.3, 0.4)!
         Error compounds: Now right_eye is also wrong

Step 3: Predict nose = (0.7, 0.6) ❌ WRONG!
         Why? Model uses wrong left_eye and wrong right_eye
         Error compounds further: Nose position is now wrong

Step 4: Predict left_mouth = (0.6, 0.8) ❌ WRONG!
         Why? Model uses wrong left_eye, right_eye, and nose
         Error compounds even more

Step 5: Predict right_mouth = (0.8, 0.8) ❌ WRONG!
         Why? All previous keypoints are wrong, so this is also wrong
         Error has fully compounded

Result: All 5 keypoints wrong → PCK = 0%
```

**Key insight:** One error at step 1 caused **all subsequent keypoints** to be wrong!

### Scenario 3: Error in Middle Keypoint (Partial Compounding)

```
Step 1: Predict left_eye = (0.3, 0.4) ✅ CORRECT
Step 2: Predict right_eye = (0.7, 0.4) ✅ CORRECT
Step 3: Predict nose = (0.6, 0.6) ❌ WRONG! (should be 0.5, 0.6)
         Error: +0.1 in x-direction

Step 4: Predict left_mouth = (0.5, 0.8) ❌ WRONG!
         Why? Model uses wrong nose position
         Error compounds: left_mouth is wrong

Step 5: Predict right_mouth = (0.7, 0.8) ❌ WRONG!
         Why? Model uses wrong nose and left_mouth
         Error compounds further: right_mouth is wrong

Result: First 2 keypoints correct, last 3 wrong → PCK = 40% (2/5)
```

**Key insight:** An error at step 3 caused **all subsequent keypoints** (steps 4-5) to be wrong, but earlier keypoints (steps 1-2) remain correct.

## Why This Matters for Your Model

### In Your Training Logs:

Looking at your results:
- **Best PCK: 55.56% (10/18 keypoints)** at epoch 15
- **Final PCK: 44.44% (8/18 keypoints)** at epoch 50

This suggests:
1. The model correctly predicts the **first ~10 keypoints** most of the time
2. But if there's an error in the first few keypoints, **all subsequent keypoints** become wrong
3. This is why PCK is stuck at ~33-55% instead of approaching 100%

### Why Training Loss is Low but PCK is Low

**Training (Teacher Forcing):**
- Model sees correct GT at each step
- Even if it predicts wrong, it sees correct input next step
- Loss can be low because model learns: "Given correct previous keypoints, predict next one"

**Validation (Autoregressive):**
- Model uses its own (potentially wrong) predictions
- If first keypoint is wrong, all subsequent are wrong
- PCK is low because errors compound

## Visual Example

Imagine predicting a person's pose (head → neck → shoulder → elbow → wrist):

```
Ground Truth Pose:
    head (0.5, 0.1)
      ↓
    neck (0.5, 0.3)
      ↓
  shoulder (0.4, 0.5)
      ↓
    elbow (0.3, 0.7)
      ↓
    wrist (0.2, 0.9)

Autoregressive Prediction (with error in head):
    head (0.6, 0.1) ❌ WRONG (+0.1 in x)
      ↓
    neck (0.6, 0.3) ❌ WRONG (based on wrong head)
      ↓
  shoulder (0.5, 0.5) ❌ WRONG (based on wrong head + neck)
      ↓
    elbow (0.4, 0.7) ❌ WRONG (based on wrong previous)
      ↓
    wrist (0.3, 0.9) ❌ WRONG (based on wrong previous)

Result: All 5 keypoints wrong, even though only head was initially wrong!
```

## Solutions

### 1. Scheduled Sampling
Gradually transition from teacher forcing to autoregressive during training:
- Epoch 1-10: 100% teacher forcing
- Epoch 11-20: 90% teacher forcing, 10% autoregressive
- Epoch 21-30: 50% teacher forcing, 50% autoregressive
- Epoch 31+: 100% autoregressive

This helps the model learn to handle its own (potentially wrong) predictions.

### 2. Better Initial Token
The first keypoint is critical! If it's wrong, everything fails.
- Use support keypoints to initialize the first prediction
- Use image features more effectively for first keypoint
- Add special handling for the first token

### 3. Error Correction Mechanisms
- Add attention to support keypoints (which are always correct)
- Use support pose graph to "correct" predictions
- Add a refinement step that uses support information

## Summary

**Autoregressive error compounding** means:
- If the model makes an error early in the sequence
- That error affects all subsequent predictions
- Because each prediction depends on previous (potentially wrong) predictions

This is why:
- Training loss can be very low (model sees correct inputs)
- But validation PCK can be low (model uses its own wrong predictions)
- And why PCK doesn't approach 100% even on a single image

The solution is to help the model learn to generate correctly even when previous predictions are wrong, or to ensure the first few keypoints are always correct.


