# Executive Summary: Support Image & Skeleton Audit

**Date:** 2025-11-26  
**Status:** ✅ Analysis Complete (No Code Changes)

---

## 🔍 Key Findings

### 1. Support Images: **LOADED BUT NOT USED** ❌

| Pipeline Stage | Status |
|----------------|--------|
| Load from disk | ✅ Yes (`datasets/mp100_cape.py:444`) |
| Include in episode | ✅ Yes (`datasets/episodic_sampler.py:345`) |
| Batch into tensor | ✅ Yes (`datasets/episodic_sampler.py:512`) |
| Move to GPU | ✅ Yes (`models/engine_cape.py:116`) |
| **Pass to model** | ❌ **NO** |
| **Encode with backbone** | ❌ **NO** |

**Bottom Line:**
- Support images are loaded, batched, and moved to GPU
- But **never passed** to `model.forward()` or encoded
- Wasted I/O, memory, and GPU bandwidth
- **Architectural intent:** Coordinate-only support encoding (no visual features)

**Why This Happens:**
- `CAPEModel.forward()` signature (line 119): **no `support_images` parameter**
- Both support encoders (`GeometricSupportEncoder`, `SupportPoseGraphEncoder`) **only accept coordinates**
- No CNN/ViT backbone for support images in the architecture

---

### 2. Skeleton Edges: **PROPERLY USED WITH GRACEFUL FALLBACK** ✅

| Pipeline Stage | Status |
|----------------|--------|
| Load from COCO metadata | ✅ Yes (`datasets/mp100_cape.py:511`) |
| Handle missing skeleton | ✅ Yes (returns `[]`) |
| Include in episode | ✅ Yes (`datasets/episodic_sampler.py:348`) |
| Batch and align | ✅ Yes (`datasets/episodic_sampler.py:515`) |
| Pass to model | ✅ Yes (`models/cape_model.py:211`) |
| **Encode in GCN** | ⚠️ **Conditional** (if `--use_gcn_preenc`) |
| **Fallback if missing** | ✅ **Coordinate-only mode** |

**Bottom Line:**
- Skeleton edges are correctly integrated end-to-end
- Missing skeletons handled gracefully (no crashes)
- Both encoders have fallback: coordinate embeddings + transformer attention

**How Fallback Works:**

**GeometricSupportEncoder:**
```python
if self.use_gcn_preenc and skeleton_edges:
    adj = adj_from_skeleton(...)  # Zero adj if edges empty
    embeddings = gcn_layers(embeddings, adj)  # No-op if adj is zero
# Transformer always applied (unrestricted self-attention)
```

**SupportPoseGraphEncoder:**
```python
if skeleton_edges and len(skeleton_edges) > 0:
    embeddings = coord_emb + edge_emb  # Use skeleton
else:
    embeddings = coord_emb  # Coordinate-only fallback
# Transformer always applied
```

---

## 🚨 Red Flags & Recommendations

### 🔴 RED FLAG #1: Support Images Loaded But Not Used

**Issue:**
- Support images: loaded → batched → GPU → **discarded**
- Wastes I/O, memory, GPU bandwidth
- Design inconsistency (why load if not used?)

**Recommendation:**
- **Option A:** Stop loading support images (add `load_support_images=False` flag)
- **Option B:** Use support images (requires architectural changes: support image backbone + cross-attention)
- **Option C:** Document this is intentional (coordinate-only CAPE design)

---

### 🟡 YELLOW FLAG #2: No Skeleton Coverage Visibility

**Issue:**
- Code doesn't log which categories have/lack skeletons
- No statistics on skeleton coverage (% of categories with skeletons)
- Silent behavior when skeleton is missing

**Recommendation:**
```python
# Add at dataset initialization
skeleton_coverage = sum(1 for cat in categories if len(get_skeleton(cat)) > 0)
print(f"Skeleton coverage: {skeleton_coverage}/{len(categories)} ({100*skeleton_coverage/len(categories):.1f}%)")
```

---

### 🟡 YELLOW FLAG #3: No Stratified Evaluation

**Issue:**
- Cannot verify if model performs equally on skeleton vs. no-skeleton categories
- Mixing skeleton/no-skeleton may create bimodal support distributions
- Potential hidden performance bias

**Recommendation:**
```python
# In evaluation, stratify PCK by skeleton presence
pck_with_skeleton = compute_pck(categories_with_skeleton)
pck_without_skeleton = compute_pck(categories_without_skeleton)
print(f"PCK (with skeleton): {pck_with_skeleton:.2%}")
print(f"PCK (without skeleton): {pck_without_skeleton:.2%}")
```

---

## ✅ What's Working Well

### 1. Graceful Skeleton Fallback
- Both encoders handle missing skeletons without errors
- Coordinate-only mode is a reasonable fallback
- Transformer provides global context regardless of skeleton

### 2. Clean Pipeline Integration
- Skeleton edges correctly batched and aligned with support/query
- 1-shot episodic structure preserved (support[i] ↔ query[i])
- No mixing of skeletons across categories

### 3. Flexible Architecture
- `--use_gcn_preenc` flag makes skeleton usage explicit
- Can train with/without skeleton-aware encoding
- Transformer always provides unrestricted self-attention

---

## 📊 Architecture Summary

### Current Support Encoding Strategy: **Coordinate-Only**

```
Support Image (512×512 RGB)
    ↓
    ❌ NOT USED (loaded but discarded)

Support Coords [(x, y), ...]
    ↓
    ✅ Coordinate MLP
    ↓
    ✅ 2D Spatial PE (sine/cosine on x, y)
    ↓
    ✅ 1D Sequence PE (keypoint ordering)
    ↓
    ⚠️  Optional GCN (if skeleton present & --use_gcn_preenc)
    ↓
    ✅ Transformer Self-Attention (unrestricted)
    ↓
Support Features [embeddings]
```

**Information Sources:**
- ✅ Keypoint coordinates (x, y)
- ✅ Spatial position (2D positional encoding)
- ✅ Keypoint ordering (1D positional encoding)
- ⚠️  Graph structure (skeleton edges, conditional)
- ❌ Visual appearance (support image pixels)

---

## 🎯 Quick Action Items

### Immediate (Low Effort, High Value)

1. **Add skeleton coverage logging:**
   ```python
   # In build_mp100_cape() or dataset __init__
   print(f"Skeleton coverage: {num_with_skeleton}/{total_categories}")
   ```

2. **Document support image design:**
   ```python
   # In episodic_collate_fn, add comment:
   # NOTE: support_images are loaded for potential future use (visualization, debugging)
   # but are NOT currently used for support encoding. The model uses coordinate-only
   # support representation (coords + skeleton edges + positional encodings).
   ```

3. **Add optional flag to skip support image loading:**
   ```python
   parser.add_argument('--load_support_images', action='store_true',
                       help='Load support images (for visualization, not used in training)')
   ```

### Future (Requires More Effort)

4. **Add stratified evaluation:**
   - Track skeleton presence per category
   - Compute PCK separately for skeleton/no-skeleton categories
   - Log to tensorboard or wandb

5. **Consider support image encoding:**
   - Add support image backbone (CNN or ViT)
   - Add cross-attention between support visual features and query
   - Requires architectural changes to `CAPEModel`

---

## 📝 Code Locations Reference

### Support Image Loading
- `datasets/mp100_cape.py:444` - Image to tensor
- `datasets/episodic_sampler.py:345` - Include in episode
- `datasets/episodic_sampler.py:512` - Batch into collated dict
- `models/engine_cape.py:116` - Load from batch (but not used)

### Skeleton Edge Usage
- `datasets/mp100_cape.py:494-517` - `_get_skeleton_for_category()`
- `datasets/episodic_sampler.py:235-242` - Validate skeleton indices
- `datasets/episodic_sampler.py:490-498` - Batch and repeat skeletons
- `models/cape_model.py:211` - Pass to support encoder
- `models/geometric_support_encoder.py:182-190` - GCN with adjacency
- `models/support_encoder.py:124-138` - Edge embedding fallback
- `models/graph_utils.py:15-75` - `adj_from_skeleton()` (handles empty)

---

## 🧪 Validation Checklist

To verify the audit findings empirically (optional):

- [ ] Print `batch['support_images'].shape` in training loop → verify loaded
- [ ] Add breakpoint in `CAPEModel.forward()` → verify `support_images` not in signature
- [ ] Add logging in `_get_skeleton_for_category()` → count empty skeletons
- [ ] Add logging in `adj_from_skeleton()` → count zero adjacency matrices
- [ ] Run training with `--use_gcn_preenc` and without → verify difference in support features
- [ ] Stratify validation PCK by skeleton presence → check for performance bias

---

## ✅ Conclusion

**Support Images:**
- **Status:** Loaded but **NOT USED** (intentional coordinate-only design)
- **Action:** Document intent or stop loading
- **Priority:** Low (no bug, just design clarification)

**Skeleton Edges:**
- **Status:** Properly integrated with **graceful fallback**
- **Action:** Add visibility (logging, stratified metrics)
- **Priority:** Medium (works fine, but lacks transparency)

**Overall:**
- ✅ No critical bugs or crashes
- ✅ Architecture is logically consistent
- ⚠️  Improve visibility and documentation
- 💡 Consider using support images if needed for better performance

