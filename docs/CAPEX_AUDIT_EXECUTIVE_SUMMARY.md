# CapeX Audit: Executive Summary

**Date**: November 25, 2025  
**Status**: ✅ COMPLETE  
**Next Action**: Review findings and approve integration plan

---

## TL;DR - Key Findings

### ✅ GOOD NEWS: CapeX's Graph Encoding is Text-Independent!

**The graph encoding components (GCN layers, adjacency matrix construction, positional encoding) are purely geometric and can be directly ported to our geometry-only model.**

### ❌ CHALLENGE: Support Representation is Text-Based

**CapeX uses text embeddings (CLIP/BERT) to represent support keypoints. We must replace this with coordinate-based embeddings.**

### 🎯 SOLUTION: Hybrid Approach

**Adopt CapeX's graph encoding + keep our sequence generation = Best of both worlds**

---

## Critical Discoveries

### Discovery 1: Text is ONLY for Support Encoding

**What I found**:
- CapeX uses CLIP to encode keypoint descriptions ("left eye", "nose", etc.)
- These text embeddings **ARE** the support representation
- **NO coordinate information** is used in the support encoding!
- Coordinates only appear as:
  1. Ground truth (for loss)
  2. Predicted outputs
  3. Positional encodings (generated from proposals)

**Implication**: We only need to replace ONE component (support encoder), not the entire architecture.

**Code location**: `capex-code/models/models/detectors/capex.py:298-337` (`extract_text_features()`)

### Discovery 2: Graph Encoding is Modular

**What I found**:
- Graph encoding happens in decoder FFN layers
- Controlled by ONE config parameter: `graph_decoder='pre'` vs `None`
- Can be added/removed without changing other components
- Uses standard GCN with normalized adjacency matrix

**Implication**: We can add graph encoding to our model incrementally.

**Code location**: `capex-code/models/models/utils/encoder_decoder.py:389-404`

### Discovery 3: Skeleton Format is Simple

**What I found**:
- Skeleton = List of `[src, dst]` edge pairs (0-indexed)
- Converted to adjacency matrix via simple algorithm
- Symmetric (undirected graph)
- Row-normalized (for graph convolution)

**Implication**: Easy to implement, no complex preprocessing needed.

**Code location**: `capex-code/models/models/utils/encoder_decoder.py:507-521` (`adj_from_skeleton()`)

### Discovery 4: Positional Encoding is Standard

**What I found**:
- Uses sinusoidal encoding (same as DETR, Transformer, etc.)
- Applied to BOTH image features AND coordinates
- Formula: `sin(x/10000^(2i/d))` and `cos(x/10000^(2i/d))`

**Implication**: We likely already have this (mmcv), or can trivially port it.

**Code location**: `capex-code/models/models/utils/positional_encoding.py:97-123`

### Discovery 5: Set Prediction vs Sequence Generation

**What I found**:
- CapeX predicts all keypoints in parallel (like DETR object detection)
- No BOS/EOS tokens, no autoregressive generation
- Each decoder layer refines coordinate predictions iteratively

**Implication**: Fundamentally different from our Raster2Seq approach. We should keep our sequence generation but adopt CapeX's graph encoding.

---

## What We Can Port (Line Counts)

| Component | Lines | Complexity | Value |
|-----------|-------|------------|-------|
| `adj_from_skeleton()` | ~15 | Low | High |
| `GCNLayer` | ~35 | Low | High |
| `SinePositionalEncoding.forward_coordinates()` | ~30 | Low | Medium |
| **TOTAL CORE** | **~80 lines** | **Low** | **High** |
| | | | |
| `ProposalGenerator` | ~80 | Medium | Medium (optional) |
| `GraphTransformerDecoderLayer` | ~100 | High | Medium (pattern only) |

**Bottom line**: We need to port ~80 lines of simple, well-contained code to get the core benefits.

---

## What We Must Implement

### 1. Geometric Support Encoder (NEW)

**Purpose**: Replace CapeX's text embeddings with coordinate-based embeddings

**Input**: 
- Support coordinates: `[bs, num_kpts, 2]`
- Skeleton edges: List of `[[src, dst], ...]`
- Visibility mask: `[bs, num_kpts]`

**Output**: 
- Support embeddings: `[bs, num_kpts, 256]`

**Proposed Implementation** (3 variants):

**Variant A: Minimal** (1 day effort)
```python
support_embed = MLP(support_coords)  # [bs, N, 2] → [bs, N, 256]
```

**Variant B: With Positional Encoding** (2 days effort, **RECOMMENDED**)
```python
coord_feat = MLP(support_coords)  # [bs, N, 2] → [bs, N, 256]
pos_feat = SinePositionalEncoding(support_coords)  # [bs, N, 2] → [bs, N, 256]
support_embed = coord_feat + pos_feat
```

**Variant C: With Graph Pre-Encoding** (3-4 days effort)
```python
coord_feat = MLP(support_coords)
pos_feat = SinePositionalEncoding(support_coords)
adj = adj_from_skeleton(skeleton, mask)
graph_feat = GNN(coord_feat, adj)  # [bs, N, 256]
support_embed = coord_feat + pos_feat + graph_feat
```

**Recommendation**: Start with **Variant B**, measure performance, upgrade to **Variant C** if needed.

---

## Integration Roadmap

### Week 1: Foundation

**Goal**: Port CapeX graph utilities and test in isolation

**Tasks**:
1. Create `models/graph_utils.py`
2. Port `adj_from_skeleton()` from CapeX
3. Port `GCNLayer` from CapeX
4. Write unit tests (adjacency symmetry, normalization, GCN shapes)
5. Run tests, verify correctness

**Deliverable**: Tested graph utility functions

**Success criteria**: All tests pass, no errors

### Week 2: Support Encoding

**Goal**: Implement geometric support encoder

**Tasks**:
1. Create `models/support_encoder.py`
2. Implement `GeometricSupportEncoder` (Variant B)
3. Port `SinePositionalEncoding.forward_coordinates()` if needed
4. Test with dummy data (shapes, gradients, no NaNs)
5. Compare to zero/random baselines

**Deliverable**: `GeometricSupportEncoder` class

**Success criteria**: Forward/backward pass works, embeddings have variance

### Week 3: Decoder Integration

**Goal**: Add GCN to our decoder layers

**Tasks**:
1. Modify decoder to accept `skeleton` argument
2. Add `adj_from_skeleton` call in forward pass
3. Add GCN layers as optional FFN replacement
4. Add config flag: `use_graph_decoder: bool`
5. Test forward pass end-to-end

**Deliverable**: Updated decoder with graph support

**Success criteria**: Model trains for 1 epoch without errors

### Week 4: Dataset Integration

**Goal**: Provide skeleton edges in our dataset

**Tasks**:
1. Update `datasets/mp100_cape.py` to include skeleton in `__getitem__`
2. Verify skeleton format matches CapeX (0-indexed, list of lists)
3. Update `build_episodic_dataloader` to pass skeleton through
4. Test episodic sampling with skeleton

**Deliverable**: Dataset provides skeleton

**Success criteria**: Dataloaders yield skeleton in correct format

### Week 5: Training & Validation

**Goal**: Train geometry-only model and measure performance

**Tasks**:
1. Train baseline (no graph) for 20 epochs
2. Train with graph for 20 epochs
3. Compare validation PCK
4. Visualize predictions with skeleton overlay
5. Debug any issues

**Deliverable**: Trained models + performance comparison

**Success criteria**: 
- Graph model > No-graph model (PCK)
- Predictions respect skeleton structure
- PCK > 40% on validation

### Week 6: Optimization & Ablation

**Goal**: Tune and understand components

**Tasks**:
1. Ablate support encoder variants (A, B, C)
2. Ablate GCN layer count (1, 2, 3)
3. Tune learning rate, batch size
4. Test on different category types (animals, furniture, clothing)

**Deliverable**: Ablation study + optimized hyperparameters

**Success criteria**: PCK > 60% on validation

---

## Files to Create

### New Files

1. **`models/graph_utils.py`** (~100 lines)
   - `adj_from_skeleton()` - adjacency matrix construction
   - `GCNLayer` - graph convolutional layer
   - Unit tests

2. **`models/support_encoder.py`** (~150 lines)
   - `GeometricSupportEncoder` - coordinate + positional + graph
   - Helper functions
   - Unit tests

3. **`configs/cape_with_graph.yaml`** (~50 lines)
   - Config for graph-enabled training
   - Hyperparameters based on CapeX

4. **`tests/test_graph_encoding.py`** (~200 lines)
   - Comprehensive tests for graph components

### Modified Files

1. **`models/cape_model.py`**
   - Add `GeometricSupportEncoder` instantiation
   - Modify `forward()` to use geometric support
   - Add GCN to decoder layers

2. **`datasets/mp100_cape.py`**
   - Include `skeleton` in `__getitem__` return dict

3. **`datasets/episodic_sampler.py`**
   - Pass skeleton through in batch collation

4. **`train_cape_episodic.py`**
   - Add `--use_graph_decoder` argument
   - Update training loop to pass skeleton to model

---

## Expected Performance

### Pessimistic (60% probability)

- **Validation PCK**: 50-60%
- **Why**: Geometry alone lacks semantic disambiguation
- **Mitigation**: Increase training data, better support encoding

### Realistic (30% probability)

- **Validation PCK**: 65-75%
- **Why**: Graph structure provides strong prior, positional encoding helps
- **Mitigation**: Ablation studies, hyperparameter tuning

### Optimistic (10% probability)

- **Validation PCK**: 75-85%
- **Why**: Our sequence generation + CapeX graph = powerful combination
- **Bonus**: Might exceed text-based CapeX on some categories!

**Baseline**: CapeX with text achieves **88.81% average PCK** on MP-100.

**Our target**: **60-75% PCK** would be a strong result for geometry-only.

---

## Risks & Mitigations

### Risk 1: Geometry Insufficient for Symmetry Breaking

**Problem**: Can't distinguish "left eye" from "right eye" without text

**Probability**: High (60%)

**Impact**: Model predicts reasonable shapes but swaps left/right

**Mitigation**:
- Use spatial ordering (left < right in x-coordinate)
- Graph topology (left-side vs right-side subgraphs)
- Augmentation: flip horizontally and swap labels

**Fallback**: Accept reduced performance on symmetric categories

### Risk 2: Integration Bugs

**Problem**: Shape mismatches, gradient flow issues, convention differences

**Probability**: Medium (40%)

**Impact**: Delays, debugging time

**Mitigation**:
- Extensive unit tests before integration
- Incremental changes (test after each modification)
- Print tensor shapes at every step

**Fallback**: Simplify integration (fewer CapeX components)

### Risk 3: Graph Doesn't Help

**Problem**: GCN layers don't improve performance (or hurt it)

**Probability**: Low (20%)

**Impact**: Wasted effort on graph encoding

**Mitigation**:
- Ablate early (Week 5)
- Compare graph vs no-graph quantitatively
- Check if adjacency matrix is correct

**Fallback**: Disable graph encoding, focus on better support encoder

---

## Resources Needed

### Computational

- **GPU**: Required (MPS on M4, or CUDA)
- **Memory**: ~8GB VRAM for batch_size=2 (same as current)
- **Training time**: ~2-3 hours per 20 epochs (on M4)

### Data

- ✅ **MP-100 dataset**: Already have
- ✅ **Skeleton annotations**: Already in annotation files
- ✅ **Category splits**: Already derived

### Code Dependencies

- ✅ **PyTorch**: Already installed
- ✅ **mmcv**: Already installed (for positional encoding, might reuse)
- ❌ **CLIP**: NOT needed (we're removing text!)
- ❌ **transformers (HF)**: NOT needed

---

## Questions for User

### Q1: Architecture Choice

**Question**: Should we keep our autoregressive sequence generation, or switch to CapeX's set prediction?

**My recommendation**: **Keep sequence generation**, add CapeX's graph encoding.

**Why**: 
- Sequence generation captures finer spatial detail (bilinear tokenization)
- Less disruption to existing codebase
- Graph encoding is orthogonal (can be added to either approach)

**Your input**: ?

### Q2: Integration Depth

**Question**: How much of CapeX should we integrate?

**Options**:
- **Minimal**: Just GCN layers (~80 lines) - 1 week
- **Moderate**: GCN + geometric support encoder (~200 lines) - 3 weeks ← **RECOMMENDED**
- **Maximal**: Full CapeX decoder (~500 lines) - 6 weeks

**Your input**: ?

### Q3: Performance Target

**Question**: What's the minimum acceptable PCK on validation?

**Context**: CapeX with text gets ~89%, geometry-only will be lower.

**Options**:
- **Conservative**: 40-50% PCK (proof of concept)
- **Realistic**: 60-70% PCK (competitive) ← **RECOMMENDED**
- **Ambitious**: 75-85% PCK (near text-based performance)

**Your input**: ?

### Q4: Timeline

**Question**: How quickly do you need this?

**Options**:
- **Quick prototype**: 1 week (minimal integration, might be buggy)
- **Solid implementation**: 3-4 weeks (moderate integration, tested) ← **RECOMMENDED**
- **Fully optimized**: 6-8 weeks (maximal integration, ablations, tuning)

**Your input**: ?

---

## Recommended Immediate Next Steps

### Step 1: Review Audit Documents (15 minutes)

**Read**:
1. `CAPEX_GRAPH_ENCODING_AUDIT.md` - Full detailed analysis (15 sections)
2. `CAPEX_CODE_SNIPPETS.md` - Exact code to port with examples
3. `CAPEX_VS_OUR_APPROACH.md` - Side-by-side comparison + migration plan
4. This document - Executive summary

**Decide**:
- Which integration level (minimal/moderate/maximal)?
- Keep sequence generation or switch to set prediction?
- Performance target?

### Step 2: Port Core Graph Utilities (2-3 hours)

**Create** `models/graph_utils.py`:
```python
# Copy from CAPEX_CODE_SNIPPETS.md:
# 1. adj_from_skeleton()
# 2. GCNLayer

# Add unit tests
def test_adjacency_symmetry(): ...
def test_adjacency_normalization(): ...
def test_gcn_shape_preservation(): ...
```

**Verify**:
```bash
python -m pytest models/graph_utils.py -v
```

### Step 3: Implement Minimal Geometric Support Encoder (3-4 hours)

**Create** `models/support_encoder.py`:
```python
class GeometricSupportEncoder(nn.Module):
    """Variant B: Coordinate MLP + Positional Encoding"""
    
    def __init__(self, hidden_dim=256):
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_encoding = SinePositionalEncoding(
            num_feats=128, normalize=True
        )
    
    def forward(self, coords, mask):
        coord_feat = self.coord_mlp(coords)
        pos_feat = self.pos_encoding.forward_coordinates(coords)
        return coord_feat + pos_feat
```

**Test**:
```python
encoder = GeometricSupportEncoder()
coords = torch.rand(2, 17, 2)
mask = torch.ones(2, 17)
output = encoder(coords, mask)
assert output.shape == (2, 17, 256)
```

### Step 4: Sanity Test Integration (1 hour)

**Modify** `models/cape_model.py` (minimal):
```python
# In __init__:
from models.support_encoder import GeometricSupportEncoder
self.support_encoder = GeometricSupportEncoder(hidden_dim=256)

# In forward:
support_embed = self.support_encoder(support_coords, support_mask)
# Pass support_embed to decoder (instead of None or zeros)
```

**Test**:
```bash
python train_cape_episodic.py --epochs 1 --batch_size 1
# Should run without errors (even if performance is bad)
```

### Step 5: Add GCN to Decoder (4-6 hours)

**Modify decoder layers** to optionally use GCN:
```python
# In each decoder layer, after cross-attention:
if self.use_graph:
    adj = adj_from_skeleton(num_kpts, skeleton, mask, device)
    tgt = self.gcn_layer(tgt, adj)  # Graph-conditioned refinement
```

**Test**:
```bash
# Without graph
python train_cape_episodic.py --use_graph false --epochs 5

# With graph
python train_cape_episodic.py --use_graph true --epochs 5

# Compare validation PCK (should be higher with graph)
```

### Step 6: Full Training Run (24 hours compute)

**Train for 50 epochs**:
```bash
python train_cape_episodic.py \
  --use_graph true \
  --epochs 50 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --early_stopping_patience 10 \
  --output_dir ./outputs/cape_geometric_graph
```

**Monitor**:
- Training loss (should decrease)
- Validation PCK (should increase)
- Predictions (visualize every 5 epochs)

**Success**: PCK > 40% after 50 epochs

---

## Key Takeaways

### 1. CapeX's Graph Encoding is Geometry-Only ✅

**The good news**: Adjacency matrix construction, GCN layers, and positional encoding have **ZERO text dependencies**.

**What this means**: We can port ~80 lines of code and get graph-aware pose estimation.

### 2. Text is ONLY for Initial Support Encoding ❌

**The challenge**: CapeX uses CLIP/BERT embeddings as support representation.

**What this means**: We must design a replacement using coordinates + graph structure.

### 3. Graph is a Modular Enhancement 🔧

**The opportunity**: Graph encoding is added via FFN replacement, not architectural overhaul.

**What this means**: We can add it incrementally, ablate its contribution, and keep or remove it based on results.

### 4. Set Prediction ≠ Sequence Generation ⚠️

**The difference**: CapeX predicts all keypoints in parallel; we generate token sequences.

**What this means**: We keep our approach, adopt CapeX's graph component (they're compatible!).

### 5. Realistic Performance Expectation 📊

**CapeX with text**: 88.81% PCK  
**Ours without text**: 60-75% PCK (estimated)

**Gap**: ~15-25% due to missing semantic information

**Acceptable?**: Depends on project goals (if the goal is "geometry-only", this is expected).

---

## Go / No-Go Decision

### ✅ GO if:
- You're willing to accept ~15-25% PCK degradation vs text-based CapeX
- You want graph-aware pose estimation without text dependencies
- You have 3-4 weeks for moderate integration
- The goal is to prove geometry-only CAPE is viable

### ⚠️ RECONSIDER if:
- You need performance matching text-based methods (might require text)
- Timeline is < 1 week (not enough time for proper integration)
- Geometry-only is not a hard constraint (could use text)

### ❌ NO-GO if:
- Support coordinates are unreliable/noisy (text more robust)
- Categories have no consistent skeleton structure (graph won't help)
- Computational budget is very limited (GCN adds parameters)

---

## Effort vs Impact Analysis

```
High Impact ↑
           │
           │  🎯 GCN Layers
           │  (Medium Effort, High Impact)
           │
           │     🎯 Geometric Support Encoder
           │     (Medium Effort, Critical)
           │
           │  📍 Positional Encoding
           │  (Low Effort, Medium Impact)
           │
           │                    ⚠️ Full CapeX Decoder
           │                    (High Effort, High Impact)
           │
           │     ⚠️ Proposal Generator
           │     (Medium Effort, Medium Impact)
           │
Low Impact ↓─────────────────────────────────────→ High Effort
           Low                                High
```

**Sweet spot**: GCN layers + Geometric support encoder (moderate effort, high impact)

---

## Confidence Levels

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Graph encoding is text-independent | **99%** | Direct code inspection |
| Text is only for support | **95%** | Code + README confirms |
| GCN improves performance | **85%** | CapeX has `graph_decoder` config, implies ablation was done |
| Our hybrid approach will work | **75%** | Architecture compatible, but untested |
| 60-75% PCK achievable | **60%** | Educated guess based on CapeX's 89% |
| Integration feasible in 3-4 weeks | **80%** | Reasonable estimate given complexity |

---

## Final Recommendation

### 🎯 Adopt CapeX's Graph Encoding with Moderate Integration

**Rationale**:
1. ✅ Graph components are geometry-only (no text dependency)
2. ✅ Small code footprint (~200-300 lines total)
3. ✅ Modular design (can add incrementally)
4. ✅ High expected value (graph should help ~5-15%)
5. ✅ Low risk (well-tested in CapeX, proven to work)

**Plan**:
1. Port `adj_from_skeleton()` and `GCNLayer` (Week 1)
2. Implement `GeometricSupportEncoder` with coords + positional (Week 2)
3. Add GCN to our decoder FFN (Week 3)
4. Integrate with dataset (Week 4)
5. Train and validate (Week 5)
6. Ablate and optimize (Week 6)

**Timeline**: 3-4 weeks

**Expected outcome**: Geometry-only CAPE model with 60-75% PCK on MP-100 validation.

---

## Appendix: Quick Reference

### Files to Read (Prioritized)

1. **MUST READ**:
   - `CAPEX_GRAPH_ENCODING_AUDIT.md` - Complete analysis
   - `CAPEX_CODE_SNIPPETS.md` - Code to port

2. **SHOULD READ**:
   - `CAPEX_VS_OUR_APPROACH.md` - Comparison + migration plan
   - This file - Executive summary

3. **OPTIONAL**:
   - `capex-code/models/models/utils/encoder_decoder.py:507-556` - Original source
   - `capex-code/README.md` - CapeX overview

### Code Snippets to Port

**Priority 1** (copy-paste ready):
- `adj_from_skeleton()` - See `CAPEX_CODE_SNIPPETS.md` Section 1
- `GCNLayer` - See `CAPEX_CODE_SNIPPETS.md` Section 2

**Priority 2** (needs adaptation):
- `SinePositionalEncoding.forward_coordinates()` - Section 3
- `GeometricSupportEncoder` - Section 6 (we designed this)

**Priority 3** (optional):
- `ProposalGenerator` - Section 5 (complex, might skip)

### Commands to Run

**After integration**:
```bash
# 1. Test graph utilities
python -m pytest models/graph_utils.py -v

# 2. Test support encoder
python -m pytest models/support_encoder.py -v

# 3. Train baseline (no graph)
python train_cape_episodic.py --use_graph false --epochs 20 \
  --output_dir ./outputs/baseline_no_graph

# 4. Train with graph
python train_cape_episodic.py --use_graph true --epochs 20 \
  --output_dir ./outputs/with_graph

# 5. Compare results
python compare_runs.py ./outputs/baseline_no_graph ./outputs/with_graph
```

---

**🚀 Ready to proceed when you are! Let me know which integration level you'd like to pursue, and I'll start implementing.**

