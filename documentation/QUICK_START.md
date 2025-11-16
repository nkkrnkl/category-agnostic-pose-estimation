# Quick Start Guide - CAPE with Raster2Seq

## What We Have

You now have a clean, minimal version of the Raster2Seq framework with only the essential files needed for your CAPE project.

**Location**: `/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/theodoros`

## Folder Structure

```
theodoros/
├── models/          # 10 files - Model architecture
├── datasets/        # 5 files  - Data loading
├── util/            # 5 files  - Utilities
├── engine.py        # Training/evaluation loops
├── main.py          # Entry point
└── requirements.txt # Dependencies
```

**Total**: 22 Python files (~392 KB)

## Key Concept

### Raster2Seq (Original)
```
Floorplan Image → Backbone → Transformer → Autoregressive Decoder → Room Polygons
                                             (with anchors)
```

### Your CAPE Adaptation
```
Object Image + Pose Graph → Backbone → Transformer → Autoregressive Decoder → Keypoints
                                                      (with anchors)
```

## The Big Picture

1. **Same Architecture**: Keep the anchor-based autoregressive decoder
2. **Different Data**: Replace room polygons with keypoint sequences
3. **Different Task**: Pose estimation instead of floorplan reconstruction
4. **Key Innovation**: Use 2D coordinate sequences as support (not text like CapeX)

## What Stays the Same

✓ Anchor-based autoregressive decoder
✓ Sequence-to-sequence generation
✓ Deformable transformer
✓ ResNet backbone
✓ Coordinate discretization
✓ L1 loss for coordinates

## What Changes

✗ Input: Object images (not floorplan images)
✗ Output: Keypoint sequences (not room polygons)
✗ Support: Pose graph as 2D coords (new!)
✗ Semantics: Keypoint types (not room types)
✗ Evaluation: PCK, mAP (not Room F1, IoU)
✗ Dataset: MP-100 (not CubiCasa5K/Structured3D)

## First Steps

### 1. Read These Files (in order)
```bash
# Core model - understand how it works
models/roomformer_v2.py

# Training loop - see how training works
engine.py

# Data loader - understand data format
datasets/poly_data.py
```

### 2. Understand the Sequence Format

**Floorplan Example**:
```
Room 1: [x1, y1, x2, y2, x3, y3, <SEP>]
Room 2: [x1, y1, x2, y2, x3, y3, x4, y4, <SEP>]
End:    [<EOS>]
```

**CAPE Example** (what you need):
```
Object: [x1, y1, x2, y2, ..., xN, yN, <EOS>]
        (keypoint 1)  (keypoint 2)  (keypoint N)
```

### 3. Key Questions to Answer

- How is MP-100 dataset formatted?
- How many keypoints per object category?
- How to represent the pose graph?
- What are the keypoint types?
- How to handle support data?

## Main Adaptations Needed

### Priority 1: Data Loading
**File**: `datasets/poly_data.py`

Change from:
- Loading room polygons from COCO format
- Multiple polygons per image

Change to:
- Loading keypoints from MP-100 format
- Single keypoint sequence per object
- Include pose graph as support

### Priority 2: Model Output
**File**: `models/roomformer_v2.py`

Change from:
- Predicting room corners
- Room type classification

Change to:
- Predicting keypoints
- Keypoint type classification

### Priority 3: Evaluation
**File**: `util/eval_utils.py`

Change from:
- Room F1, Corner F1, Angle metrics

Change to:
- PCK (Percentage of Correct Keypoints)
- mAP (mean Average Precision)
- OKS (Object Keypoint Similarity)

## How the Autoregressive Decoder Works

```python
# Simplified pseudocode

# Initialize
anchors = learnable_anchors  # Spatial priors
previous_tokens = [<BOS>]

# Generate sequence
while not done:
    # 1. Attend to previous tokens (masked self-attention)
    context = self_attention(previous_tokens)
    
    # 2. Attend to image features (cross-attention with anchors)
    image_context = deformable_attention(image_features, anchors)
    
    # 3. Predict next token
    next_token = predict(context + image_context)
    
    # 4. Append to sequence
    previous_tokens.append(next_token)
    
    # 5. Check if done
    if next_token == <EOS>:
        done = True
```

This is EXACTLY what you need for keypoint prediction!

## Comparison with CapeX

| Aspect | CapeX | Your Method (Raster2Seq) |
|--------|-------|--------------------------|
| Support Data | Text descriptions | 2D coordinate sequences |
| Example | "left front leg" | [x1, y1, x2, y2, ...] |
| Architecture | Transformer + Graph Decoder | Autoregressive Decoder |
| Advantage | Interpretable | Direct coordinate prediction |

## Key Parameters to Adjust

From `main.py`:

```bash
# Current (Floorplan)
--num_queries 800        # Max corners
--num_polys 20          # Max rooms
--semantic_classes 12   # Room types

# Your CAPE version
--num_queries 50        # Max keypoints (adjust per dataset)
--num_polys 1           # Single object per image
--semantic_classes 17   # Keypoint types (adjust per dataset)
--seq_len 100          # Max sequence length
```

## Common Mistakes to Avoid

1. ❌ Don't change the autoregressive decoder structure
2. ❌ Don't remove the anchor mechanism
3. ❌ Don't change the sequence format drastically
4. ✅ DO keep the coordinate discretization
5. ✅ DO keep the L1 loss for coordinates
6. ✅ DO adapt the data loader carefully

## Resources

- **Raster2Seq Paper**: Read for architecture details
- **CapeX Paper**: Read for CAPE problem formulation
- **MP-100 Paper**: Read for dataset and evaluation
- **Original Repo**: `Raster2Seq_internal-main/` for reference

## Getting Help

If you get stuck:
1. Check the original repo's README
2. Look at training scripts in `tools/` folder
3. Read `REPOSITORY_OVERVIEW.md` in original repo
4. Compare with CapeX implementation

## Timeline Suggestion

- **Week 1-2**: Understand Raster2Seq architecture
- **Week 3-4**: Prepare MP-100 dataset
- **Week 5-6**: Adapt model for CAPE
- **Week 7-8**: Train and debug
- **Week 9-10**: Evaluate and compare baselines
- **Week 11-12**: Write report and prepare presentation

## Success Criteria

Your project is successful if you:
1. ✓ Adapt Raster2Seq for keypoint prediction
2. ✓ Train on MP-100 dataset
3. ✓ Compare against 3+ CAPE baselines
4. ✓ Show quantitative results (PCK, mAP)
5. ✓ Show qualitative results (visualizations)

---

**Ready to Start?**

Begin by reading these 3 files:
1. `models/roomformer_v2.py`
2. `engine.py`
3. `datasets/poly_data.py`

Good luck! 🚀
