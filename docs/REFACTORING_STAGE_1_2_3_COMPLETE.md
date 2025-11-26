# CAPE Geometry-Only Refactoring - Stages 1-3 Complete

## Executive Summary

Successfully completed Stages 1-3 of the geometry-only CAPE refactoring plan:

- **Stage 1**: Copied CapeX graph utilities (adjacency matrices, GCN layers, sinusoidal positional encoding)
- **Stage 2**: Implemented `GeometricSupportEncoder` combining coordinate embeddings, positional encoding, and optional GCN
- **Stage 3**: Integrated new encoder into CAPE model with feature flags for easy comparison

**Status**: ✅ All tests passing (47 total tests across 3 test files)

**Next Steps**: 
- Train models with old vs new encoders to compare performance (see `scripts/compare_encoders.sh`)
- If geometric encoder shows improvement, proceed to Stage 4 (decoder GCN)
- If not, tune hyperparameters before proceeding

---

## Stage 1: CapeX Geometry Components (COMPLETE)

### Files Created

1. **`models/graph_utils.py`** - Graph convolution utilities
   - `adj_from_skeleton()`: Builds dual-channel normalized adjacency matrices
   - `GCNLayer`: Graph convolutional layer with dual-channel convolution
   - Source: Copied from CapeX `encoder_decoder.py:507-555`
   - Pure geometry, no text dependencies

2. **`models/positional_encoding.py`** - Positional encoding modules
   - `PositionalEncoding1D`: Standard 1D positional encoding (existing)
   - `SinePositionalEncoding2D`: DETR-style sinusoidal encoding for 2D coordinates
   - Source: Copied from CapeX `positional_encoding.py:97-123`
   - Pure geometry, translation/scale invariant

3. **`tests/test_graph_utils.py`** - Comprehensive unit tests
   - 26 tests covering:
     - Adjacency matrix properties (symmetry, normalization, masking)
     - GCN layer (shape preservation, gradient flow, different configs)
     - Positional encoding (determinism, output shapes)
     - Integration tests (complete pipeline)

### Key Components

#### Adjacency Matrix Construction

```python
adj = adj_from_skeleton(num_pts, skeleton_edges, mask, device)
# Output: [bs, 2, num_pts, num_pts]
#   Channel 0: Self-loops (identity with masked points zeroed)
#   Channel 1: Normalized neighbor adjacency (row-normalized)
```

**Properties**:
- Symmetric undirected graph (before normalization)
- Row-normalized (each row sums to 1)
- Handles masked keypoints correctly
- Works with empty skeletons (isolated nodes)

#### GCN Layer

```python
gcn = GCNLayer(in_features=256, out_features=256, kernel_size=2, batch_first=True)
out = gcn(x, adj)
# x: [bs, num_pts, 256]
# adj: [bs, 2, num_pts, num_pts]
# out: [bs, num_pts, 256]
```

**Features**:
- Dual-channel convolution (self + neighbors)
- Efficient einsum-based implementation
- Supports batch_first and seq_first conventions
- Gradient flow validated

#### Sinusoidal Positional Encoding

```python
pos_enc = SinePositionalEncoding2D(num_feats=128)
pos = pos_enc.forward_coordinates(coords)
# coords: [bs, num_pts, 2] in [0, 1]
# pos: [bs, num_pts, 256]  # 128 * 2
```

**Properties**:
- Deterministic (same coords → same encoding)
- Translation/scale aware (uses 2π scaling)
- Separate encoding for x and y axes
- Concatenates [pos_y, pos_x]

### Test Results

```
tests/test_graph_utils.py::26 tests - ALL PASSED ✓
  - TestAdjFromSkeleton: 6/6 tests passed
  - TestGCNLayer: 6/6 tests passed
  - TestSinePositionalEncoding2D: 5/5 tests passed
  - TestIntegration: 2/2 tests passed
  - Standalone tests: 7/7 tests passed
```

**Validation**:
- Adjacency matrices correct (symmetric, normalized, masked)
- GCN preserves shapes and flows gradients
- Positional encoding deterministic and consistent
- Complete pipeline works end-to-end

---

## Stage 2: Geometric Support Encoder (COMPLETE)

### Files Created

1. **`models/geometric_support_encoder.py`** - Geometry-only support encoder
   - `GeometricSupportEncoder`: Main encoder class
   - Combines: Coordinate MLP + Positional Encoding + Optional GCN + Transformer
   - Pure geometry (no text), CapeX-inspired architecture

2. **`tests/test_geometric_support_encoder.py`** - Comprehensive tests
   - 15 tests covering:
     - Forward pass (with/without GCN)
     - Gradient flow
     - Masked keypoints
     - Variable dimensions
     - Determinism

### Architecture

```
Pipeline:
  Input: support_coords [bs, N, 2] (normalized to [0,1])
         support_mask [bs, N] (True = invalid)
         skeleton_edges (list of edge lists)

  1. Coordinate Embedding:
     coord_emb = MLP(support_coords)
     → [bs, N, 256]
     - 2-layer MLP: Linear(2→256) → ReLU → Linear(256→256)
     - Replaces CapeX's text encoder

  2. Positional Encoding (from CapeX):
     pos_emb = SinePositionalEncoding2D(support_coords)
     → [bs, N, 256]
     - Sinusoidal encoding with scale=2π

  3. Combine:
     embeddings = coord_emb + pos_emb
     → [bs, N, 256]

  4. Optional GCN Pre-Encoding:
     IF use_gcn_preenc:
       adj = adj_from_skeleton(N, skeleton_edges, support_mask, device)
       for gcn_layer in gcn_layers:
         embeddings = gcn_layer(embeddings, adj)
     → [bs, N, 256]

  5. Transformer Self-Attention:
     support_features = TransformerEncoder(embeddings, mask=support_mask)
     → [bs, N, 256]

  Output: support_features [bs, N, 256]
```

### Test Results

```
tests/test_geometric_support_encoder.py::15 tests - ALL PASSED ✓
  - TestGeometricSupportEncoder: 11/11 tests passed
  - Standalone tests: 4/4 tests passed
```

**Key Validations**:
- Forward pass works with/without GCN
- Gradients flow through all components (MLP, GCN, Transformer)
- Handles masked keypoints correctly
- Works with variable keypoint counts (5-50 points tested)
- Deterministic in eval mode
- GCN produces different outputs than no-GCN (proves it's active)

---

## Stage 3: Integration into CAPE Model (COMPLETE)

### Files Modified

1. **`models/cape_model.py`** - Added geometric encoder support
   - Added import: `from .geometric_support_encoder import GeometricSupportEncoder`
   - Modified `__init__`: Added `use_geometric_encoder`, `use_gcn_preenc`, `num_gcn_layers` parameters
   - Conditional encoder selection:
     ```python
     if use_geometric_encoder:
         self.support_encoder = GeometricSupportEncoder(...)
     else:
         self.support_encoder = SupportPoseGraphEncoder(...)  # Old encoder
     ```
   - Updated `build_cape_model()` to pass new parameters

2. **`models/train_cape_episodic.py`** - Added command-line arguments
   - Added documentation block explaining CAPE training strategy
   - Added arguments:
     - `--use_geometric_encoder`: Enable new encoder
     - `--use_gcn_preenc`: Enable GCN pre-encoding
     - `--num_gcn_layers`: Number of GCN layers (default: 2)
   - Updated print statements to show encoder configuration

3. **`datasets/mp100_cape.py`** - Validated coordinate normalization and skeleton indexing
   - Modified `_get_skeleton_for_category()`:
     - Added conversion from 1-indexed to 0-indexed (COCO → PyTorch)
     - Validates skeleton edges are in correct format
   - Coordinates already normalized to [0, 1] (no changes needed)

4. **`datasets/episodic_sampler.py`** - Added batch validation
   - Added assertion in `episodic_collate_fn`:
     ```python
     assert len(support_skeletons_repeated) == batch_size
     ```
   - Validates skeleton edges list length matches batch size

### Files Created

1. **`scripts/compare_encoders.sh`** - Automated comparison script
   - Runs 3 experiments:
     1. Baseline (old encoder)
     2. Geometric encoder (no GCN)
     3. Geometric encoder (with GCN)
   - Each trains for 10 epochs
   - Logs results for comparison

2. **`tests/test_cape_model_integration.py`** - Integration tests
   - 6 tests covering:
     - Support-query batch alignment
     - Skeleton edge propagation
     - Old vs new encoder comparison
     - Coordinate normalization validation
     - Skeleton 0-indexing validation

### Test Results

```
tests/test_cape_model_integration.py::6 tests - ALL PASSED ✓
  - TestCAPEModelIntegration: 5/5 tests passed
  - Standalone test: 1/1 test passed
```

### Integration Points Validated

**1. Coordinate Normalization** ✓
- Dataset normalizes coordinates to [0, 1] in `episodic_sampler.py:214-215`
- Matches CapeX expectations
- Positional encoding works correctly with [0, 1] range

**2. Skeleton Edge Indexing** ✓
- Added conversion from 1-indexed (COCO) to 0-indexed (PyTorch)
- In `mp100_cape.py::_get_skeleton_for_category()`
- Prevents index out of bounds errors in `adj_from_skeleton`

**3. Batch Alignment** ✓
- Support coords, masks, and skeleton edges all have matching batch size
- Assertion added in `episodic_collate_fn` to catch misalignment
- Critical for 1-shot learning (support[i] must match query[i])

**4. Encoder API Compatibility** ✓
- Both old and new encoders have same signature:
  ```python
  forward(support_coords, support_mask, skeleton_edges) -> support_features
  ```
- Output shapes identical: [bs, num_pts, hidden_dim]
- Can switch between encoders with single flag

---

## Usage

### Training with Old Encoder (Baseline)

```bash
python models/train_cape_episodic.py \
    --epochs 10 \
    --batch_size 2 \
    --num_queries_per_episode 2 \
    --dataset_root . \
    --output_dir outputs/baseline
```

### Training with New Geometric Encoder (No GCN)

```bash
python models/train_cape_episodic.py \
    --epochs 10 \
    --batch_size 2 \
    --num_queries_per_episode 2 \
    --dataset_root . \
    --use_geometric_encoder \
    --output_dir outputs/geometric_no_gcn
```

### Training with New Geometric Encoder (With GCN)

```bash
python models/train_cape_episodic.py \
    --epochs 10 \
    --batch_size 2 \
    --num_queries_per_episode 2 \
    --dataset_root . \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --num_gcn_layers 2 \
    --output_dir outputs/geometric_with_gcn
```

### Automated Comparison

```bash
bash scripts/compare_encoders.sh
```

This runs all three experiments and saves results for comparison.

---

## Test Summary

### All Tests Passing ✓

```
Total: 47 tests across 3 test files
- test_graph_utils.py: 26/26 passed
- test_geometric_support_encoder.py: 15/15 passed
- test_cape_model_integration.py: 6/6 passed
```

### Test Coverage

**Unit Tests**:
- Graph utilities (adjacency, GCN, positional encoding)
- Geometric support encoder (forward, gradients, masking)

**Integration Tests**:
- Batch alignment (support-query correspondence)
- Encoder API compatibility
- Data format validation (coordinates, skeleton edges)

**Gradient Tests**:
- All components allow gradient flow
- No NaN/Inf in outputs or gradients
- End-to-end differentiability validated

---

## Implementation Notes

### What Was Copied from CapeX

1. **`adj_from_skeleton` function** (exact copy)
   - Lines 507-521 from CapeX `encoder_decoder.py`
   - No modifications needed
   - Pure geometry, no text dependencies

2. **`GCNLayer` class** (exact copy)
   - Lines 524-555 from CapeX `encoder_decoder.py`
   - No modifications needed
   - Works with arbitrary features + adjacency

3. **`SinePositionalEncoding2D` class** (exact copy)
   - Lines 97-123 from CapeX `positional_encoding.py`
   - No modifications needed
   - DETR-style sinusoidal encoding

### What Was Created New

1. **`GeometricSupportEncoder`** - Original implementation
   - Inspired by CapeX architecture
   - Replaces text encoder with coordinate MLP
   - Combines CapeX components in geometry-only pipeline
   - Fully compatible with existing CAPE model

### What Was Modified

1. **`models/cape_model.py`**:
   - Added conditional encoder selection
   - Maintains backward compatibility (old encoder still works)
   - No breaking changes to API

2. **`models/train_cape_episodic.py`**:
   - Added command-line arguments
   - Added documentation explaining training strategy
   - No changes to training loop logic

3. **`datasets/mp100_cape.py`**:
   - Added skeleton 1-indexed → 0-indexed conversion
   - No changes to coordinate normalization (already correct)

4. **`datasets/episodic_sampler.py`**:
   - Added batch size validation assertion
   - No changes to collation logic

---

## Architecture Comparison

### Old Support Encoder (Baseline)

```
SupportPoseGraphEncoder:
  1. Coordinate Embedding: Linear(2 → 256)
  2. Edge Embedding: Lookup table (connected=1, not=0)
  3. Combine: coord_emb + edge_info
  4. Positional Encoding: Learned 1D positional embeddings
  5. Transformer: Self-attention
```

**Limitations**:
- Binary edge representation (connected/not connected)
- Learned positional embeddings (limited generalization)
- No graph-aware processing

### New Geometric Encoder (CapeX-Inspired)

```
GeometricSupportEncoder:
  1. Coordinate Embedding: MLP(2 → 256 → 256)
  2. Positional Encoding: Sinusoidal (DETR-style, from CapeX)
  3. Combine: coord_emb + pos_emb
  4. Optional GCN: GCN layers with normalized adjacency (from CapeX)
  5. Transformer: Self-attention
```

**Improvements**:
- Richer coordinate representation (2-layer MLP vs 1 linear)
- Better positional encoding (sinusoidal vs learned)
- Proper graph convolution (normalized adjacency vs binary lookup)
- Validated design patterns (CapeX achieves 88.8% PCK)

---

## Validation Checklist

### Data Pipeline ✓

- [x] Coordinates normalized to [0, 1]
- [x] Skeleton edges converted to 0-indexed
- [x] Batch size alignment validated
- [x] Support-query correspondence maintained

### Model Architecture ✓

- [x] Both encoders compatible (same API)
- [x] Feature toggle works (--use_geometric_encoder)
- [x] GCN toggle works (--use_gcn_preenc)
- [x] No breaking changes to existing code

### Testing ✓

- [x] All unit tests pass (26/26 for graph utils)
- [x] All encoder tests pass (15/15)
- [x] All integration tests pass (6/6)
- [x] Gradients flow through all components
- [x] No NaN/Inf in outputs

### Code Quality ✓

- [x] No linter errors
- [x] Comprehensive docstrings
- [x] Type hints where applicable
- [x] Clear comments explaining design decisions

---

## Next Steps (Stage 4 & 5)

### Stage 4: Add Decoder GCN (OPTIONAL - HIGH RISK)

**Decision Point**: Only proceed if encoder GCN shows improvement in Stage 3

**Tasks**:
1. Run comparison experiments: `bash scripts/compare_encoders.sh`
2. Analyze validation PCK:
   - Baseline vs Geometric (no GCN)
   - Geometric (no GCN) vs Geometric (with GCN)
3. If GCN improves PCK by >3%, proceed to decoder modifications
4. If not, tune hyperparameters first

**Risks**:
- Decoder modification is complex (sequence vs keypoint mapping)
- May break existing functionality
- Requires extensive testing

**Mitigation**:
- Implement behind feature flag
- Validate encoder improvements first
- Extensive ablation studies

### Stage 5: Cleanup & Finalization

**Prerequisites**: Stages 3-4 validated via training

**Tasks**:
1. Remove old encoder (if new one performs better)
2. Set geometric encoder as default
3. Update README and documentation
4. Create migration guide

---

## Performance Expectations

### Realistic Targets (Geometry-Only, No Text)

- **Minimum Viable**: >40% validation PCK
  - Proves concept works
  - Worse than CapeX (88.8%) but functional

- **Realistic**: 60-70% validation PCK
  - Competitive performance
  - Graph encoding compensates for lack of text

- **Optimistic**: >70% validation PCK
  - Strong geometry-only performance
  - Sequence generation helps

### Comparison to CapeX

**CapeX (with text)**:
- 88.8% average PCK on MP-100
- Uses CLIP/BERT embeddings for semantic information
- DETR-style parallel prediction

**Our Model (geometry-only)**:
- Expected: 60-70% PCK (realistic)
- Uses only coordinates + skeleton edges
- Autoregressive sequence generation

**Performance Gap Factors**:
- Text provides semantic labels ("left" vs "right", "front" vs "back")
- Model must learn symmetry breaking from geometry + vision only
- Tradeoff: More generalizable (no language bias) but less accurate

---

## Files Created/Modified Summary

### New Files (8)

1. `models/graph_utils.py` - Graph utilities from CapeX
2. `models/positional_encoding.py` - Positional encoding (existing file, added SinePositionalEncoding2D)
3. `models/geometric_support_encoder.py` - New support encoder
4. `tests/test_graph_utils.py` - Graph utilities tests
5. `tests/test_geometric_support_encoder.py` - Encoder tests
6. `tests/test_cape_model_integration.py` - Integration tests
7. `scripts/compare_encoders.sh` - Comparison script
8. `docs/REFACTORING_STAGE_1_2_3_COMPLETE.md` - This document

### Modified Files (4)

1. `models/cape_model.py` - Added geometric encoder support
2. `models/train_cape_episodic.py` - Added command-line arguments
3. `datasets/mp100_cape.py` - Fixed skeleton indexing
4. `datasets/episodic_sampler.py` - Added batch validation

### No Changes (Preserved Stability)

- `datasets/discrete_tokenizer.py` - Tokenization unchanged
- `datasets/data_utils.py` - Data utilities unchanged
- `engine_cape.py` - Training loop unchanged (until Stage 4)
- `models/roomformer_v2.py` - Base model unchanged (until Stage 4)
- All visualization scripts - Unchanged

---

## Conclusion

Stages 1-3 successfully completed. The geometry-only CAPE model now has:

1. **Proven graph encoding components** from CapeX (adjacency matrices, GCN layers)
2. **Better positional encoding** (DETR-style sinusoidal)
3. **Modular architecture** (can toggle encoders, GCN on/off)
4. **Comprehensive test coverage** (47 tests, all passing)
5. **Backward compatibility** (old encoder still works)

**Ready for experimentation**: Use `scripts/compare_encoders.sh` to compare performance.

**Next decision**: Based on training results, proceed to Stage 4 (decoder GCN) or tune and finalize.

---

## Stage 3.5: Add Sequence Positional Encoding (CRITICAL FIX)

### Motivation

After Stage 3 completion, a comprehensive positional encoding audit revealed a critical gap: the NEW `GeometricSupportEncoder` lacks 1D sequence positional encoding, which is essential for the transformer to understand keypoint ordering.

The PhD student's guidance emphasized that "positional encoding for the keypoint sequence" is crucial for understanding vertex structure (e.g., head → neck → shoulders). The OLD `SupportPoseGraphEncoder` had this (line 141 in [models/support_encoder.py](../models/support_encoder.py)), but the NEW `GeometricSupportEncoder` only had 2D spatial positional encoding based on (x,y) coordinates, not sequence position.

### Changes

**File**: [models/geometric_support_encoder.py](../models/geometric_support_encoder.py)

**Additions**:
1. Import `PositionalEncoding1D` from `models.positional_encoding`
2. Initialize `self.sequence_pos_encoding` in `__init__` (after line 86)
3. Apply sequence PE in `forward` after spatial PE (after line 156)
4. Update docstrings to reflect the new 3-component architecture
5. Update `__repr__` to show both spatial and sequence PE

**Result**: Support embeddings now contain THREE types of information:
- **Content**: What are the coordinates? (coord_emb)
- **Spatial**: Where in image space? (spatial_pe)
- **Sequential**: Which keypoint in ordering? (sequence_pe) ← NEW

### Pipeline Before and After

**Before** (Stage 3):
```
1. Coordinate embedding: MLP(coords) → [bs, N, D]
2. 2D spatial PE: SinePosEnc2D(coords) → [bs, N, D]
3. Combine: coord_emb + spatial_pe
4. Optional GCN: GCN(embeddings, adjacency)
5. Transformer: Self-attention
```

**After** (Stage 3.5):
```
1. Coordinate embedding: MLP(coords) → [bs, N, D]
2. 2D spatial PE: SinePosEnc2D(coords) → [bs, N, D]
3. Combine: coord_emb + spatial_pe
4. 1D sequence PE: SinePosEnc1D(indices) → add to embeddings  ← NEW
5. Optional GCN: GCN(embeddings, adjacency)
6. Transformer: Self-attention
```

### Validation

All existing tests pass. New tests added in:
- [tests/test_positional_encoding.py](../tests/test_positional_encoding.py) (NEW file, 10 tests)
- [tests/test_geometric_support_encoder.py](../tests/test_geometric_support_encoder.py) (5 new tests added)

**Key Test**: `test_sequence_pe_affects_output` verifies that different keypoint orderings produce different outputs, confirming sequence PE is working.

**Test Results**:
- 10/10 tests pass in `test_positional_encoding.py`
- 20/20 tests pass in `test_geometric_support_encoder.py` (including 5 new)
- 6/6 integration tests pass in `test_cape_model_integration.py`

### Documentation

See [CAPE_Positional_Encoding_Design.md](./CAPE_Positional_Encoding_Design.md) for full design rationale, including:
- Why transformers need positional encoding for keypoint sequences
- Comparison with CapeX approach
- Design decisions (sinusoidal vs. learned, additive vs. concatenated)
- Validation strategy

### Impact

This fix addresses a fundamental architectural gap:
- **Without sequence PE**: Model only knows WHERE keypoints are (spatial position)
- **With sequence PE**: Model knows BOTH where keypoints are AND which keypoint it is

This aligns with the PhD student's recommendation and combines the best of CapeX (spatial PE for coordinates) with transformer best practices (sequence PE for ordering).

---

## Quick Start

To test the new encoder immediately:

```bash
# Run all tests
python -m pytest tests/test_graph_utils.py tests/test_geometric_support_encoder.py tests/test_cape_model_integration.py -v

# Train for 5 epochs with new encoder
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --epochs 5 \
    --batch_size 2 \
    --dataset_root . \
    --output_dir outputs/test_geometric

# Compare old vs new encoder
bash scripts/compare_encoders.sh
```

Expected: Training should complete without errors. Validation PCK target: >40% (minimum), >60% (realistic).

