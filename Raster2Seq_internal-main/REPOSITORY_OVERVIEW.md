# Raster2Seq Repository Overview

## 🎯 Project Purpose

**Raster2Seq** is a deep learning framework that converts rasterized floorplan images into structured vector-graphics representations (polygon sequences). It reformulates the Raster2Vector conversion problem as a sequence-to-sequence task, where each room is represented as a labeled polygon sequence.

### Key Innovation
- **Autoregressive decoder** that predicts polygon corners sequentially
- **Learnable anchors** that guide attention to informative image regions
- **Flexible output format** that handles complex floorplans with varying numbers of rooms and corners

---

## 📁 Repository Structure

### Core Components

#### 1. **Models** (`models/`)
- **`roomformer.py`** / **`roomformer_v2.py`**: Main model architectures
  - Based on Deformable DETR architecture
  - Encoder-decoder transformer with learnable anchors
  - Supports both non-semantic and semantic-rich floorplans
- **`deformable_transformer.py`**: Transformer backbone with deformable attention
- **`backbone.py`**: CNN backbone (ResNet) for feature extraction
- **`matcher.py`**: Hungarian matching for training
- **`losses.py`**: Loss functions (L1, rasterization, classification)

#### 2. **Data Processing** (`datasets/`)
- **`poly_data.py`**: Main dataset class (`MultiPoly`)
  - Converts COCO-format annotations to model inputs
  - Supports polygon-to-sequence conversion (`poly2seq` mode)
  - Handles discrete tokenization of coordinates
- **`discrete_tokenizer.py`**: Tokenizes continuous coordinates into discrete bins
- **`transforms.py`**: Image augmentation and preprocessing

#### 3. **Training & Evaluation** (`engine.py`)
- **`train_one_epoch()`**: Training loop with loss computation
- **`evaluate()`**: Evaluation on validation/test sets
- **`generate()`** / **`generate_v2()`**: Inference functions
- Includes evaluation metrics (F1, IoU, etc.)

#### 4. **Data Preprocessing** (`data_preprocess/`)
Scripts to convert raw datasets into COCO format:
- **`cubicasa5k/`**: CubiCasa5K dataset preprocessing
- **`raster2graph/`**: Raster2Graph dataset preprocessing
- **`stru3d/`**: Structured3D dataset preprocessing
- **`rplan/`**, **`scenecad/`**, **`waffle/`**: Additional dataset support

#### 5. **Differentiable Rasterization** (`diff_ras/`)
- CUDA kernels for differentiable polygon rasterization
- Used for rasterization loss during training

#### 6. **Evaluation Scripts**
- **`s3d_floorplan_eval/`**: Structured3D evaluation metrics
- **`rplan_eval/`**: RPlan evaluation
- **`scenecad_eval/`**: SceneCAD evaluation
- **`clipseg_eval/`**: CLIPSeg evaluation

#### 7. **Visualization** (`html_generator/`, `gt_html_generator/`)
- HTML visualization generators for predictions and ground truth
- Plotting utilities in `util/plot_utils.py`

#### 8. **Training Scripts** (`tools/`)
Shell scripts for training and evaluation:
- **Pretraining**: `pretrain_*.sh` (structure only, no semantics)
- **Finetuning**: `finetune_*.sh` (with semantic room labels)
- **Evaluation**: `eval_*.sh`
- **Prediction**: `predict_*.sh`

---

## 🔄 Workflow

### 1. **Data Preparation**
```bash
# Example for CubiCasa5K
bash data_preprocess/cubicasa5k/run.sh
```
Converts raw floorplan images and annotations into COCO format:
- RGB images (input)
- Polygon coordinates for each room (output)
- Optional: Semantic room labels (bedroom, kitchen, etc.)

### 2. **Training (Two-Stage)**

#### Stage 1: Pretraining
```bash
bash tools/pretrain_s3d_rgb.sh
```
- Learns to predict polygon structure only
- No semantic room classification
- Focuses on geometric accuracy

#### Stage 2: Finetuning
```bash
bash tools/finetune_s3d_rgb.sh
```
- Adds semantic room classification
- Refines predictions with room type labels

### 3. **Inference**
```bash
python predict.py --checkpoint <path> --input <image>
```
Generates polygon sequences from floorplan images.

### 4. **Evaluation**
```bash
bash tools/eval_s3d_rgb_finetune.sh
```
Computes metrics (F1 score, IoU, etc.) on test sets.

---

## 🏗️ Architecture Details

### Model Architecture

1. **Backbone** (ResNet50)
   - Extracts multi-scale features from input image
   - Outputs feature maps at different resolutions

2. **Encoder** (Deformable Transformer)
   - Processes image features with deformable attention
   - Multi-scale feature fusion

3. **Decoder** (Autoregressive)
   - **Query-based**: Uses learnable queries (anchors) for each potential polygon corner
   - **Autoregressive**: Predicts next corner given previous corners
   - **Attention**: Cross-attention to image features, self-attention to previous corners

4. **Output Heads**
   - **Classification**: Whether a corner is valid (foreground/background)
   - **Coordinates**: (x, y) position of the corner (normalized [0, 1])
   - **Room Class** (finetuning): Semantic label (bedroom, kitchen, etc.)

### Key Features

- **Learnable Anchors**: Spatial coordinates that guide attention
- **Polygon Refinement**: Iterative refinement of polygon predictions
- **Masked Attention**: Optional masking to prevent cross-polygon attention
- **Rasterization Loss**: Differentiable loss on rasterized polygon masks

---

## 📊 Supported Datasets

1. **Structured3D** (S3D)
   - 3,500+ floorplans
   - 16 room types + doors + windows
   - Semantic-rich annotations

2. **CubiCasa5K**
   - 5,000 floorplans
   - Diverse room structures
   - Real-world complexity

3. **Raster2Graph**
   - Large-scale dataset (~300K images)
   - Japanese floorplans
   - Complex geometric variations

4. **Additional**: RPlan, SceneCAD, WAFFLE

---

## 🔧 Key Configuration Parameters

From `main.py`:

- **`--num_queries`**: Max number of corner queries (e.g., 800)
- **`--num_polys`**: Max number of polygons/rooms (e.g., 20)
- **`--semantic_classes`**: Number of room classes (-1 for non-semantic)
- **`--poly2seq`**: Enable sequence-to-sequence mode
- **`--seq_len`**: Maximum sequence length (for poly2seq)
- **`--num_bins`**: Number of bins for coordinate discretization
- **`--use_anchor`**: Enable learnable anchor mechanism
- **`--with_poly_refine`**: Iterative polygon refinement

---

## 📝 Data Format

### Input
- **Images**: RGB floorplan images (typically 512×512 or 256×256)
- **Format**: COCO-style JSON annotations

### Output
- **Polygons**: List of closed polygons (room boundaries)
- **Coordinates**: Normalized [0, 1] relative to image size
- **Labels**: Optional semantic room types

### Sequence Format (poly2seq mode)
- Coordinates are discretized into bins
- Special tokens: `<SEP>` (between polygons), `<EOS>` (end), `<CLS>` (room class)
- Example: `[x1, y1, x2, y2, ..., <SEP>, x1, y1, ...]`

---

## 🎓 Key Concepts

1. **Polygon Sequence**: Rooms represented as sequences of (x, y) coordinates
2. **Autoregressive Generation**: Predicts corners one at a time, conditioned on previous corners
3. **Learnable Anchors**: Spatial priors that help the model focus on relevant image regions
4. **Deformable Attention**: Efficient attention mechanism that samples sparse locations
5. **Differentiable Rasterization**: Allows gradient flow through polygon-to-mask conversion

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   cd models/ops && sh make.sh
   cd ../../diff_ras && python setup.py build develop
   ```

2. **Prepare data** (see `data_preprocess/README.md`)

3. **Train**:
   ```bash
   bash tools/pretrain_s3d_rgb.sh
   bash tools/finetune_s3d_rgb.sh
   ```

4. **Evaluate**:
   ```bash
   bash tools/eval_s3d_rgb_finetune.sh
   ```

5. **Predict**:
   ```bash
   python predict.py --checkpoint <path> --input_dir <images>
   ```

---

## 📚 Related Work

- **RoomFormer**: Base architecture inspiration
- **Deformable DETR**: Transformer architecture
- **PolyFormer**: Seq2seq polygon generation framework
- **HEAT, Raster2Graph, MonteFloor**: Related floorplan reconstruction methods

---

## 🔍 Important Files to Understand

- **`main.py`**: Entry point, argument parsing
- **`models/roomformer.py`**: Main model definition
- **`engine.py`**: Training and evaluation loops
- **`datasets/poly_data.py`**: Data loading and preprocessing
- **`models/deformable_transformer.py`**: Transformer architecture
- **`util/poly_ops.py`**: Polygon manipulation utilities

---

## 💡 Tips for Understanding the Code

1. Start with `main.py` to see the training pipeline
2. Look at `models/roomformer.py` for the forward pass
3. Check `datasets/poly_data.py` to understand data format
4. Review `engine.py` for training/evaluation logic
5. Examine training scripts in `tools/` for typical configurations

