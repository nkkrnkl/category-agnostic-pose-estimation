# CAPE Positional Encoding Design

## Overview

This document explains the positional encoding strategy in our geometry-only CAPE implementation, based on the PhD student's recommendation that "positional encoding for the keypoint sequence" is critical for understanding vertex structure.

## Background: Why Positional Encoding Matters

### The Problem

Transformers are permutation-invariant by default. Without positional encoding:
- Model cannot distinguish [head, neck, shoulder] from [shoulder, head, neck]
- Only spatial position (x, y) available, not sequential position (0th, 1st, 2nd)
- Violates structural understanding needed for pose estimation

### The Solution

We use TWO types of positional encoding for support keypoints:
1. **2D Spatial PE**: Encodes WHERE keypoints are in image space (x, y)
2. **1D Sequence PE**: Encodes WHICH keypoint it is in ordering (index)

## Positional Encoding in CAPE Components

### Image Features (Query Processing)

**Location**: Applied in backbone ([models/backbone.py](../models/backbone.py))

**Type**: 2D sinusoidal positional encoding

**Purpose**: Tell the model where each image patch is in the spatial grid

**Implementation**: `PositionEmbeddingSine` (DETR-style)

### Support Keypoints (NEW Implementation)

**Location**: [models/geometric_support_encoder.py](../models/geometric_support_encoder.py)

**Types**: BOTH 2D spatial AND 1D sequence

**Pipeline**:
```
Coordinates [B, N, 2]
  ↓
1. Coordinate embedding: MLP(coords) → [B, N, D]
2. 2D spatial PE: SinePosEnc2D(coords) → [B, N, D]
3. Combine: coord_emb + spatial_pe
4. 1D sequence PE: SinePosEnc1D(indices) → add to embeddings
5. Optional GCN: graph-aware processing
6. Transformer: self-attention
  ↓
Support features [B, N, D]
```

**Why Both?**:
- Spatial PE: "This keypoint is at (0.3, 0.4) in the image"
- Sequence PE: "This is the 5th keypoint in the skeleton"
- Combined: "The 5th keypoint (e.g., left shoulder) is at (0.3, 0.4)"

### Decoder Output Sequence

**Location**: [models/deformable_transformer_v2.py](../models/deformable_transformer_v2.py)

**Type**: 1D sinusoidal positional encoding

**Purpose**: Tell the model which token position it's generating

**Implementation**: `self.pos_embed` (lines 136-138)

## Comparison with CapeX

| Component | CapeX | Our Implementation |
|-----------|-------|-------------------|
| Support representation | Text embeddings | Coordinate embeddings |
| Support positional encoding | 2D spatial only (for coord refinement) | 2D spatial + 1D sequence |
| Graph encoding | GCN in decoder only | GCN in encoder (optional) |
| Sequence ordering | Implicit in text labels | Explicit via sequence PE |

## Design Rationale

### Why Sinusoidal (not Learned)?

1. **Generalization**: Works for any sequence length (no max_len constraint)
2. **Consistency**: Matches decoder positional encoding
3. **Deterministic**: Aids debugging and reproducibility
4. **Proven**: Standard in Transformer paper

### Why Additive (not Concatenated)?

1. **Standard**: Proven in Transformer literature
2. **Efficient**: No dimension change
3. **Flexible**: Easy to ablate (remove one PE type)

### Why Both Spatial AND Sequence?

- PhD student: "positional encoding for the keypoint sequence"
- CapeX: Uses spatial PE for coordinate refinement
- Our approach: Combine both for maximum information

## Validation

See [tests/test_geometric_support_encoder.py](../tests/test_geometric_support_encoder.py) for comprehensive tests verifying:
- Sequence PE affects output (different orderings → different embeddings)
- Gradients flow correctly
- Batch alignment preserved
- Masked keypoints handled properly

## Future Enhancements

Potential improvements (not in current plan):
1. **Spatial Sorting**: Sort keypoints top-to-bottom for consistency
2. **Relative PE**: Encode relative positions between keypoints
3. **Learned PE**: Experiment with learned vs. sinusoidal
4. **Graph-Aware PE**: PE that incorporates skeleton structure

## References

- PhD student discussion (Nov 2024): "positional encoding for the keypoint sequence"
- CapeX paper: ICLR 2025 submission
- Attention Is All You Need (Vaswani et al., 2017): Original transformer PE
- DETR (Carion et al., 2020): 2D positional encoding for vision

