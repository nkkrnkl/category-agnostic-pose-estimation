# File Inventory - Theodoros CAPE Project

## Summary
**Total Files**: 24 files (22 Python files + 2 documentation files)
**Total Size**: ~392 KB
**Source**: Extracted from Raster2Seq_internal-main repository

---

## File Categories

### 📋 Documentation (2 files)
- `README.md` - Project overview and adaptation guide
- `FILE_INVENTORY.md` - This file

### 🏗️ Core Training Files (2 files)
- `main.py` (15 KB) - Main entry point, argument parsing, training setup
- `engine.py` (64 KB) - Training loops, evaluation, inference functions

### 🔧 Configuration (1 file)
- `requirements.txt` - Python dependencies

---

## 🧠 Models Directory (10 files)

### Primary Model Files
1. **`roomformer_v2.py`** - **MAIN MODEL** (Version 2, most recent)
   - Complete model architecture
   - Anchor-based autoregressive decoder
   - Output heads for coordinates, classification, semantics

2. **`roomformer.py`** - Original model (for reference)

### Transformer Components
3. **`deformable_transformer_v2.py`** - Deformable transformer architecture (v2)
4. **`deformable_transformer.py`** - Original deformable transformer
5. **`deformable_points.py`** - Deformable attention point sampling

### Supporting Components
6. **`backbone.py`** - ResNet backbone for feature extraction
7. **`position_encoding.py`** - Positional encoding for transformer
8. **`losses.py`** - Loss functions (L1, classification, rasterization)
9. **`matcher.py`** - Hungarian matching algorithm for training
10. **`__init__.py`** - Module initialization

---

## 📊 Datasets Directory (5 files)

1. **`poly_data.py`** - **PRIMARY DATASET CLASS**
   - Loads and processes polygon data
   - Converts to sequence format
   - **NEEDS ADAPTATION** for MP-100 keypoints

2. **`discrete_tokenizer.py`** - Coordinate discretization
   - Converts continuous coordinates to discrete tokens
   - Learnable codebook

3. **`transforms.py`** - Image augmentation and preprocessing

4. **`data_utils.py`** - Data loading utilities

5. **`__init__.py`** - Module initialization

---

## 🛠️ Util Directory (5 files)

1. **`poly_ops.py`** - Polygon operations
   - **NEEDS ADAPTATION** for keypoint operations
   - Contains geometric utilities

2. **`misc.py`** - Miscellaneous utilities
   - General helper functions
   - Tensor operations

3. **`plot_utils.py`** - Visualization utilities
   - Plotting functions for results

4. **`eval_utils.py`** - Evaluation metrics
   - **NEEDS ADAPTATION** for CAPE metrics (PCK, mAP)
   - Currently has floorplan metrics

5. **`__init__.py`** - Module initialization

---

## Key Files to Understand First

### Priority 1 - Core Architecture
1. **`models/roomformer_v2.py`** - Start here to understand the model
2. **`engine.py`** - Understand training/evaluation flow
3. **`datasets/poly_data.py`** - Understand data format

### Priority 2 - Training Setup
4. **`main.py`** - Entry point and configuration
5. **`models/losses.py`** - Loss functions
6. **`models/deformable_transformer_v2.py`** - Transformer details

### Priority 3 - Utilities
7. **`util/poly_ops.py`** - Operations to adapt
8. **`util/eval_utils.py`** - Metrics to adapt
9. **`datasets/discrete_tokenizer.py`** - Tokenization logic

---

## Files That Need Adaptation for CAPE

### Critical Adaptations
1. ✅ **`datasets/poly_data.py`**
   - Change from floorplan polygons → keypoint sequences
   - Load MP-100 dataset format
   - Handle pose graph as support data

2. ✅ **`models/roomformer_v2.py`**
   - Adjust output heads for keypoints
   - Modify semantic classes for keypoint types
   - Keep autoregressive decoder structure

3. ✅ **`util/eval_utils.py`**
   - Replace floorplan metrics with CAPE metrics
   - Implement PCK (Percentage of Correct Keypoints)
   - Implement mAP for pose estimation

### Minor Adaptations
4. ⚠️ **`util/poly_ops.py`**
   - Rename/adapt polygon operations to keypoint operations
   - Keep coordinate manipulation logic

5. ⚠️ **`models/losses.py`**
   - May need pose-specific losses (e.g., limb constraints)
   - Keep L1 and classification losses

### Keep As-Is
- ✓ `models/backbone.py` - No changes needed
- ✓ `models/deformable_transformer_v2.py` - No changes needed
- ✓ `models/position_encoding.py` - No changes needed
- ✓ `models/matcher.py` - No changes needed
- ✓ `datasets/discrete_tokenizer.py` - No changes needed
- ✓ `datasets/transforms.py` - Minor changes for data augmentation
- ✓ `engine.py` - Minor changes for evaluation metrics
- ✓ `main.py` - Minor changes for arguments

---

## Architecture Overview

```
Input Image (Object)
        ↓
    Backbone (ResNet)
        ↓
    Image Features
        ↓
    Deformable Transformer Encoder
        ↓
    Multi-scale Features
        ↓
    ┌─────────────────────────────────────┐
    │  Autoregressive Decoder             │
    │  ├─ Learnable Anchors               │
    │  ├─ Masked Self-Attention           │
    │  ├─ Cross-Attention to Image        │
    │  └─ Deformable Attention            │
    └─────────────────────────────────────┘
        ↓
    Output Heads
    ├─ Coordinate Head → (x, y) keypoints
    ├─ Token Type Head → <CORNER>, <SEP>, <EOS>
    └─ Semantic Head → Keypoint type
        ↓
    Keypoint Sequence
```

---

## Comparison: Floorplan vs CAPE

| Aspect | Raster2Seq (Floorplan) | CAPE Adaptation |
|--------|------------------------|-----------------|
| Input | Floorplan image | Object image |
| Output | Room polygon sequences | Keypoint sequences |
| Support Data | N/A | Pose graph (2D coords) |
| Semantic Classes | Room types (bedroom, kitchen) | Keypoint types (head, leg) |
| Evaluation | Room F1, Corner F1, IoU | PCK, mAP, OKS |
| Sequence Format | [x1,y1,x2,y2,<SEP>,...] | [x1,y1,x2,y2,<SEP>,...] |
| Anchor Points | Room corner candidates | Keypoint candidates |

---

## Next Steps

1. **Phase 1: Understanding** (Week 1)
   - Read the 3 priority files
   - Understand sequence generation
   - Study anchor mechanism

2. **Phase 2: Data Preparation** (Week 2)
   - Download MP-100 dataset
   - Convert to sequence format
   - Create data loader

3. **Phase 3: Model Adaptation** (Week 3-4)
   - Modify `poly_data.py` for MP-100
   - Adapt `roomformer_v2.py` for keypoints
   - Update evaluation metrics

4. **Phase 4: Training** (Week 5-6)
   - Train on MP-100
   - Debug and iterate
   - Hyperparameter tuning

5. **Phase 5: Evaluation** (Week 7-8)
   - Compare against baselines
   - Generate qualitative results
   - Write report

---

## Missing Components (Need to Add)

1. **MP-100 Dataset Loader**
   - Create new file: `datasets/mp100_data.py`

2. **CAPE Evaluation Metrics**
   - Add to `util/eval_utils.py`: PCK, mAP, OKS

3. **Pose Graph Support Module**
   - Create new file: `datasets/pose_graph.py`
   - Handle pose graph as sequence

4. **CAPE-specific Config**
   - Create new file: `configs/cape_config.py`
   - Store MP-100 specific parameters

---

## Dependencies (from requirements.txt)

The copied files depend on these libraries. You'll need to install them:
- PyTorch
- torchvision
- numpy
- opencv-python
- scipy
- matplotlib
- shapely
- pycocotools
- And others specified in requirements.txt

**Note**: The original repo also requires compiling CUDA ops for deformable attention. This may need special setup.

---

**Created**: November 15, 2024
**Purpose**: Essential files for CAPE adaptation from Raster2Seq framework
