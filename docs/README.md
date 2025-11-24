# Theodoros - CAPE Project with Raster2Seq

This folder contains the essential files from the Raster2Seq framework needed for the Category-Agnostic Pose Estimation (CAPE) project.

## Project Goal
Adapt the Raster2Seq framework to perform category-agnostic pose estimation on the MP-100 dataset, using 2D coordinate sequences as support data instead of annotated support images.

## Directory Structure

```
theodoros/
├── models/                          # Core model architecture
│   ├── __init__.py
│   ├── backbone.py                  # ResNet backbone for feature extraction
│   ├── deformable_transformer.py    # Deformable transformer (original)
│   ├── deformable_transformer_v2.py # Deformable transformer (version 2)
│   ├── position_encoding.py         # Positional encoding utilities
│   ├── losses.py                    # Loss functions (L1, classification, etc.)
│   ├── matcher.py                   # Hungarian matching for training
│   ├── roomformer.py                # Main model (original)
│   ├── roomformer_v2.py             # Main model (version 2) - PRIMARY MODEL
│   └── deformable_points.py         # Deformable attention points
│
├── datasets/                        # Data loading and processing
│   ├── __init__.py
│   ├── poly_data.py                 # Main dataset class (to be adapted for CAPE)
│   ├── discrete_tokenizer.py        # Coordinate tokenization
│   ├── transforms.py                # Image augmentation
│   └── data_utils.py                # Data utilities
│
├── util/                            # Utility functions
│   ├── __init__.py
│   ├── misc.py                      # Miscellaneous utilities
│   ├── poly_ops.py                  # Polygon operations (to be adapted for keypoints)
│   ├── plot_utils.py                # Visualization utilities
│   └── eval_utils.py                # Evaluation metrics
│
├── engine.py                        # Training and evaluation loops
├── main.py                          # Main entry point for training
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Key Components to Adapt for CAPE

### 1. **Model Architecture** (models/roomformer_v2.py)
- **Current**: Predicts polygon sequences (rooms)
- **Adapt to**: Predict keypoint sequences (pose skeleton)
- Keep the anchor-based autoregressive decoder intact
- Modify output heads to predict keypoints instead of room corners

### 2. **Dataset** (datasets/poly_data.py)
- **Current**: Loads floorplan images and room polygons
- **Adapt to**: Load MP-100 images and keypoint annotations
- Modify to handle pose graph as 2D coordinate sequences
- Keep the sequence generation logic

### 3. **Loss Functions** (models/losses.py)
- **Current**: L1 loss for coordinates, classification for room types
- **Adapt to**: L1 loss for keypoints, classification for keypoint types
- May need to add specific losses for pose estimation (e.g., limb length constraints)

### 4. **Evaluation** (util/eval_utils.py)
- **Current**: F1 score, IoU for room detection
- **Adapt to**: PCK (Percentage of Correct Keypoints), mAP for pose estimation

## Workflow for Adaptation

1. **Data Preparation**
   - Convert MP-100 dataset to sequence format
   - Create pose graph as keypoint sequences
   - Modify `datasets/poly_data.py` to load MP-100

2. **Model Modification**
   - Adapt `models/roomformer_v2.py` for keypoint prediction
   - Adjust number of queries, sequence length
   - Modify semantic classes to keypoint types

3. **Training**
   - Use `main.py` with modified arguments
   - Train on MP-100 dataset
   - Fine-tune for specific pose categories

4. **Evaluation**
   - Implement CAPE-specific metrics
   - Compare against baselines (CapeFormer, GraphCape, etc.)

## Important Parameters

From `main.py`, key parameters to adjust:
- `--num_queries`: Max number of keypoint queries (adjust based on skeleton)
- `--semantic_classes`: Number of keypoint types
- `--poly2seq`: Enable sequence-to-sequence mode (KEEP THIS)
- `--seq_len`: Maximum sequence length
- `--num_bins`: Number of bins for coordinate discretization
- `--use_anchor`: Enable learnable anchors (KEEP THIS)

## Next Steps

1. Understand the current Raster2Seq architecture by reading:
   - `models/roomformer_v2.py` - main model
   - `engine.py` - training loop
   - `datasets/poly_data.py` - data loading

2. Design the adaptation:
   - How to represent pose graphs as sequences?
   - What modifications are needed for keypoint prediction?
   - How to handle support data (pose graph)?

3. Implement MP-100 dataset loader

4. Modify model architecture for CAPE

5. Train and evaluate

## References

- Original Raster2Seq paper: For architecture details
- CapeX paper: For CAPE problem formulation
- MP-100 dataset: For data format and evaluation metrics
