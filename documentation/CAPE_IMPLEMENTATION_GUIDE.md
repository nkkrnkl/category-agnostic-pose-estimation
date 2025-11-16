# CAPE Implementation Guide - Based on Email Clarification

## Email Confirmation Summary

### ✅ Files Verified (All Present)

Based on the email exchange, you need these files - **ALL INCLUDED**:

1. **roomformer.py / roomformer_v2.py** ✓
   - Main model architectures
   - Based on Deformable DETR architecture
   - Encoder-decoder transformer with learnable anchors
   - Supports both non-semantic and semantic-rich floorplans

2. **deformable_transformer.py / deformable_transformer_v2.py** ✓
   - Transformer backbone with deformable attention

3. **backbone.py** ✓
   - CNN backbone (ResNet) for feature extraction

4. **matcher.py** ✓
   - Hungarian matching for training

5. **losses.py** ✓
   - Loss functions (L1, rasterization, classification)

### Process Confirmed (From Email)

```
Step 1: MP100 Image (Rasterized RGB)
         ↓
Step 2: Feature Extractor (Encoder)
         ↓
Step 3: Image Feature Vector
         ↓
Step 4: Autoregressive Decoder (token-by-token)
         ↓ (+ Reference Skeleton Sequence)
Step 5: Vectorized Output (Joint Coordinates)
```

---

## Key Insight from Email Reply

> "Since in CAPE, you also need to provide the model a reference skeleton of another image,
> you should also think of an efficient way to add it as an extra condition to the model,
> apart from the rasterized image and its joint sequence."

### Suggested Approach (From Instructor)

**"One simple way is to present the reference one as another sequence and concatenate it with the joint sequence of the target object in the input image."**

This means:

```python
# Traditional Raster2Seq (Floorplan)
input_sequence = [x1, y1, x2, y2, ..., xN, yN, <EOS>]

# Your CAPE Adaptation
reference_sequence = [x1_ref, y1_ref, x2_ref, y2_ref, ..., xN_ref, yN_ref]  # Pose graph
target_sequence = [x1, y1, x2, y2, ..., xN, yN]  # To predict

input_sequence = [reference_sequence + <SEP> + target_sequence]
```

---

## Implementation Steps (Based on Email Guidance)

### Step 1: Vectorize MP100 Images
**File to modify**: `datasets/poly_data.py`

```python
class MP100Dataset:
    def __getitem__(self, idx):
        # Load query image (RGB rasterized)
        query_image = load_image(idx)

        # Load reference skeleton (pose graph as sequence)
        reference_keypoints = load_reference_skeleton(idx)  # [x1, y1, x2, y2, ...]

        # Load target keypoints (ground truth)
        target_keypoints = load_target_keypoints(idx)  # [x1, y1, x2, y2, ...]

        # Concatenate reference + target sequences
        input_sequence = concatenate_sequences(
            reference_keypoints,
            target_keypoints
        )

        return {
            'image': query_image,
            'input_sequence': input_sequence,
            'target_sequence': target_keypoints
        }
```

### Step 2: Feature Extractor (Already Implemented)
**File**: `models/backbone.py`

✅ No changes needed - ResNet backbone extracts features from RGB images

```python
# In models/roomformer_v2.py
features = self.backbone(image)  # Extract image features
```

### Step 3: Produce Image Feature Vector
**File**: `models/deformable_transformer_v2.py`

✅ No changes needed - Encoder processes features

```python
# In encoder
image_features = self.encoder(features)  # Shape: [batch, L_I, D]
```

### Step 4: Token-by-Token Prediction (Autoregressive)
**File**: `models/roomformer_v2.py`

This is where the **concatenation of reference skeleton** happens:

```python
class RoomFormerV2(nn.Module):
    def forward(self, images, reference_skeleton, target_sequence):
        # Step 1: Extract image features
        image_features = self.backbone(images)
        image_features = self.encoder(image_features)

        # Step 2: Concatenate reference skeleton with target sequence
        # Reference skeleton acts as "support data" or "pose graph"
        input_sequence = torch.cat([
            reference_skeleton,  # [batch, N_ref, 2] - reference keypoints
            self.sep_token,      # Separator token
            target_sequence      # [batch, N_target, 2] - target keypoints
        ], dim=1)

        # Step 3: Tokenize the concatenated sequence
        input_tokens = self.tokenizer(input_sequence)

        # Step 4: Autoregressive decoding
        output_tokens = self.decoder(
            tgt=input_tokens,              # Input sequence tokens
            memory=image_features,         # Image features from encoder
            tgt_mask=causal_mask,          # Causal mask for autoregression
            query_pos=self.anchor_points   # Learnable anchors
        )

        # Step 5: Predict coordinates
        predicted_keypoints = self.coord_head(output_tokens)

        return predicted_keypoints
```

### Step 5: Vectorized Output
**Output format**: Joint coordinates as continuous (x, y) values

```python
output = {
    'pred_coords': predicted_keypoints,  # [batch, N, 2]
    'pred_labels': predicted_labels,     # [batch, N, num_classes]
    'pred_tokens': predicted_tokens      # [batch, N, 3] (corner/sep/eos)
}
```

---

## Critical Implementation Detail: Reference Skeleton Integration

Based on the email, you need to add the **reference skeleton as extra conditioning**. Here are three approaches:

### Approach 1: Sequence Concatenation (Recommended by Instructor)

```python
# Simplest approach - concatenate sequences
input_sequence = [ref_x1, ref_y1, ..., ref_xN, ref_yN, <SEP>,
                  tgt_x1, tgt_y1, ..., tgt_xN, tgt_yN]
```

**Pros**:
- Simple to implement
- Minimal code changes
- Leverages existing sequence processing

**Cons**:
- Longer sequences (2x length)
- May need to increase `--seq_len` parameter

### Approach 2: Cross-Attention to Reference

```python
# Add reference skeleton as separate input to decoder
output = decoder(
    tgt=target_sequence,
    memory=image_features,
    reference=reference_skeleton  # Additional conditioning
)
```

**Pros**:
- More explicit conditioning
- Can attend differently to reference vs. image

**Cons**:
- Requires modifying decoder architecture
- More complex implementation

### Approach 3: Image Concatenation

```python
# Render reference skeleton on image and concatenate channels
image_with_reference = torch.cat([
    query_image,           # [3, H, W]
    reference_skeleton_map # [1, H, W] - rendered skeleton
], dim=0)  # [4, H, W]
```

**Pros**:
- Visual conditioning
- Spatially aligned

**Cons**:
- Requires rendering skeleton on image
- Need to modify backbone input channels

---

## Modified File Checklist

### Files That MUST Be Modified

1. ✏️ **`datasets/poly_data.py`**
   - Load MP100 dataset instead of floorplan datasets
   - Load reference skeleton (pose graph)
   - Concatenate reference + target sequences
   - Convert keypoints to sequence format

2. ✏️ **`models/roomformer_v2.py`**
   - Handle concatenated sequences (reference + target)
   - Adjust sequence length handling
   - Modify forward pass to process reference skeleton

3. ✏️ **`main.py`**
   - Add MP100 dataset arguments
   - Adjust `--seq_len` for concatenated sequences
   - Adjust `--num_queries` for keypoints (not corners)
   - Change `--semantic_classes` to keypoint types

4. ✏️ **`util/eval_utils.py`**
   - Replace floorplan metrics with CAPE metrics
   - Implement PCK (Percentage of Correct Keypoints)
   - Implement mAP for pose estimation

### Files That Can Stay As-Is

✓ `models/backbone.py` - No changes
✓ `models/deformable_transformer_v2.py` - No changes
✓ `models/matcher.py` - No changes (may need minor tweaks)
✓ `models/losses.py` - Minor changes (keep L1 loss)
✓ `datasets/discrete_tokenizer.py` - No changes
✓ `datasets/transforms.py` - Minor changes for augmentation
✓ `engine.py` - Minor changes for evaluation

---

## Example Implementation: Sequence Concatenation

### In `datasets/poly_data.py`

```python
class MP100Dataset(Dataset):
    def __init__(self, root, split='train'):
        self.images = load_mp100_images(root, split)
        self.keypoints = load_mp100_keypoints(root, split)
        self.pose_graphs = load_pose_graphs()  # Reference skeletons

    def __getitem__(self, idx):
        # Load query image
        image = self.images[idx]

        # Load target keypoints
        target_kpts = self.keypoints[idx]  # [N, 2]

        # Get reference skeleton for this category
        category = self.get_category(idx)
        reference_kpts = self.pose_graphs[category]  # [N, 2]

        # Concatenate: [ref_kpts, <SEP>, target_kpts]
        input_sequence = self.concatenate_sequences(
            reference_kpts,
            target_kpts
        )

        return {
            'image': image,
            'sequence': input_sequence,
            'target': target_kpts,
            'reference': reference_kpts
        }

    def concatenate_sequences(self, ref, tgt):
        # Flatten coordinates
        ref_flat = ref.reshape(-1)  # [N*2]
        tgt_flat = tgt.reshape(-1)  # [N*2]

        # Add separator token
        sep_token = torch.tensor([self.SEP_TOKEN])

        # Concatenate
        sequence = torch.cat([ref_flat, sep_token, tgt_flat])

        return sequence
```

### In `models/roomformer_v2.py`

```python
class RoomFormerV2(nn.Module):
    def forward(self, samples, targets):
        # Extract image features
        features = self.backbone(samples['image'])
        memory = self.encoder(features)

        # Get input sequence (already concatenated in dataset)
        input_sequence = samples['sequence']

        # Tokenize
        input_tokens = self.tokenizer(input_sequence)

        # Autoregressive decoding with causal mask
        hs = self.decoder(
            tgt=input_tokens,
            memory=memory,
            query_pos=self.anchor_embeddings
        )

        # Predict coordinates
        outputs = self.coord_head(hs)

        return outputs
```

---

## Key Parameters to Adjust

Based on the concatenation approach:

```bash
# Original Raster2Seq (Floorplan)
--seq_len 800           # Max sequence length
--num_queries 800       # Max corners

# Your CAPE Adaptation
--seq_len 1000          # 2x for concatenation (ref + target)
--num_queries 34        # Number of keypoints (e.g., 17 ref + 17 target)
--semantic_classes 17   # Keypoint types (adjust per dataset)
--num_bins 256          # Coordinate discretization bins
--use_anchor True       # KEEP learnable anchors
```

---

## Validation Checklist

Before training, verify:

- [ ] MP100 dataset loads correctly
- [ ] Reference skeleton (pose graph) is loaded for each category
- [ ] Sequences are concatenated properly: `[ref, <SEP>, target]`
- [ ] Sequence length is sufficient (2x original)
- [ ] Tokenizer handles concatenated sequences
- [ ] Model forward pass processes both reference and target
- [ ] Loss is computed only on target keypoints (not reference)
- [ ] Evaluation metrics are CAPE-specific (PCK, mAP)

---

## Missing vs. Present

### ✅ All Required Files Present

According to the email, you need:
1. ✅ roomformer.py / roomformer_v2.py
2. ✅ deformable_transformer.py / deformable_transformer_v2.py
3. ✅ backbone.py
4. ✅ matcher.py
5. ✅ losses.py

### ✅ Additional Helpful Files Included

6. ✅ position_encoding.py - Needed for transformer
7. ✅ deformable_points.py - Needed for deformable attention
8. ✅ datasets/poly_data.py - Template for MP100 loader
9. ✅ datasets/discrete_tokenizer.py - For coordinate tokenization
10. ✅ engine.py - Training loop
11. ✅ main.py - Entry point

### ❌ Files NOT Needed (Correctly Excluded)

- room_dropout.py - Only for floorplan data augmentation
- All evaluation scripts (s3d_floorplan_eval, etc.)
- All visualization scripts (html_generator, etc.)
- Data preprocessing scripts (data_preprocess/)
- CUDA ops compilation (models/ops/) - May need separately

---

## Summary: Email Confirmation

Based on the email exchange with the instructor:

1. ✅ **Your understanding is correct**
2. ✅ **All required files are included**
3. ✅ **Process is clear**: RGB image → Encoder → Feature vector → Autoregressive decoder → Vectorized output
4. ✅ **Key addition**: Reference skeleton as extra conditioning via sequence concatenation

**Next Step**: Implement the sequence concatenation approach in `datasets/poly_data.py` and test with a small subset of MP100 data.

---

**Created**: November 15, 2024
**Based on**: Email exchange with Raster2Seq instructor (Hao Phung)
