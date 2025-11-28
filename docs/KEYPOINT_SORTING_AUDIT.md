# Keypoint Sorting Audit Report

## Executive Summary

This audit evaluates claims about keypoint sorting in the CAPE training pipeline and assesses the risks of implementing geometric sorting.

**Date:** November 28, 2024  
**Auditor:** Code Audit System  
**Scope:** `datasets/mp100_cape.py`, `datasets/data_utils.py`, `models/graph_utils.py`

---

## 1. Verdict: Does the Code Currently Sort Keypoints?

### ❌ FALSE - No Keypoint Sorting is Implemented

**Evidence from `datasets/mp100_cape.py`:**

```python
# Line 100 - This is the ONLY sorted() call in the file:
self.ids = list(sorted(self.coco.imgs.keys()))  # Sorts IMAGE IDs, not keypoints
```

Searched for: `sort`, `reorder`, `lexsort`, `argsort` in `mp100_cape.py`
- **Result:** Only 1 match, which sorts image IDs for deterministic iteration
- **Keypoints are kept in their original COCO annotation order**

**Evidence from `datasets/data_utils.py`:**

```python
# Line 27-57 - A sort_polygons() function EXISTS:
def sort_polygons(polygons, tolerance=20, reverse=False):
    # Sorts polygons from top-left to bottom-right
    ...
    return sorted_polygons, sorted_indices
```

**Is it used?** Searched entire codebase for imports/calls:
- `util/plot_utils.py`: Has a different function `sort_polygons_by_matching()` for visualization
- **NOT imported in `mp100_cape.py`, `episodic_sampler.py`, or any training code**

### Conclusion: Keypoints remain in their original semantic order as defined in COCO annotations.

---

## 2. Risk Assessment: Would Simple Sorting Corrupt Graph Data?

### ⚠️ CRITICAL: YES - Simple Sorting Would BREAK the Skeleton Graph

**The Problem: Index-Semantic Binding**

Skeleton edges are defined as index pairs referring to semantic keypoints:

```python
# From MP-100 COCO annotations:
{
    "keypoints": ["nose", "left_eye", "right_eye", "left_ear", ...],
    "skeleton": [[0, 1], [0, 2], [1, 3], ...]  # nose→left_eye, nose→right_eye, ...
}
```

The skeleton `[[0, 1], ...]` means:
- Index 0 = "nose"
- Index 1 = "left_eye"
- Edge [0,1] = "nose connects to left_eye"

**What Happens if We Sort Keypoints?**

**Before sorting:**
```
Keypoints:  [(100, 50),  (80, 40),   (120, 40)]
             idx 0       idx 1       idx 2
             "nose"      "left_eye"  "right_eye"

Skeleton:   [[0, 1], [0, 2]]
            → nose↔left_eye, nose↔right_eye  ✓ CORRECT
```

**After geometric sorting (by y, then x):**
```
Keypoints:  [(80, 40),   (120, 40),  (100, 50)]
             idx 0       idx 1       idx 2
             WAS left_eye WAS right_eye WAS nose

Skeleton:   [[0, 1], [0, 2]]  ← NOT UPDATED!
            → left_eye↔right_eye, left_eye↔nose  ✗ WRONG!
```

**Evidence from `models/graph_utils.py`:**

```python
# Line 51-63: adj_from_skeleton() uses indices directly
edges = torch.tensor(skeleton[b], device=device)
adj = torch.zeros(num_pts, num_pts, device=device)
if len(valid_edges) > 0:
    adj[valid_edges[:, 0], valid_edges[:, 1]] = 1  # Direct index access!
```

If coordinates are reordered but `skeleton` is not, the adjacency matrix will encode **wrong connections**.

---

## 3. Critical Analysis: Is Geometric Sorting Beneficial?

### The Claimed Benefit

> "Implementing geometric sorting would help the Transformer learn a 'canonical' geometric order, improving generalization."

### Evaluation: Mixed Verdict

**Arguments FOR Sorting:**
1. Provides consistent ordering across categories
2. Autoregressive generation follows a predictable spatial pattern
3. Reduces sequence complexity - model predicts "next spatially"

**Arguments AGAINST Sorting (Stronger):**

1. **Destroys Semantic Structure:**
   - Skeleton edges encode anatomical/structural relationships
   - "Shoulder connects to elbow" is category-invariant semantic knowledge
   - Sorting by (x,y) removes this inductive bias

2. **Category-Agnostic ≠ Structure-Agnostic:**
   - The model should learn that "connected keypoints" matter
   - GCN encoder specifically exploits this structure
   - Sorting would make the GCN meaningless (wrong adjacencies)

3. **Inconsistent Orderings Across Poses:**
   - Same category, different poses → different sort orders
   - "Head" is top-left when standing, not when lying down
   - Model would see same keypoint at different sequence positions

4. **PhD Student Recommendation:**
   - The PhD student suggested sorting for the *floorplan* task
   - Floorplans don't have semantic skeletons
   - Pose estimation explicitly uses skeleton structure

### Conclusion: Sorting is **harmful** for CAPE because we use skeleton-based GCN encoding.

---

## 4. Implementation Strategy (If Sorting is Required)

### The Safe Approach: Sort WITH Index Mapping

If geometric sorting is absolutely required (e.g., ablation study), the skeleton edges MUST be remapped:

**Pseudocode:**

```python
def sort_keypoints_with_skeleton_remap(keypoints, skeleton, visibility=None):
    """
    Sort keypoints geometrically while preserving skeleton edge semantics.
    
    Args:
        keypoints: np.array of shape (N, 2) with (x, y) coordinates
        skeleton: list of [src_idx, dst_idx] edge pairs
        visibility: optional (N,) array of visibility flags
    
    Returns:
        sorted_keypoints: (N, 2) array sorted by (y, x)
        remapped_skeleton: list of edges with updated indices
        sorted_visibility: (N,) array if visibility provided
        sort_indices: original indices after sorting
    """
    N = len(keypoints)
    
    # Step 1: Get sort order (top-to-bottom, left-to-right)
    # lexsort sorts by last key first, so (x, y) means primary=y, secondary=x
    sort_indices = np.lexsort((keypoints[:, 0], keypoints[:, 1]))
    
    # Step 2: Create OLD→NEW index mapping
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sort_indices)}
    
    # Step 3: Remap skeleton edges
    remapped_skeleton = []
    for src, dst in skeleton:
        if src < N and dst < N:  # Validate indices
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            remapped_skeleton.append([new_src, new_dst])
    
    # Step 4: Apply sort to keypoints
    sorted_keypoints = keypoints[sort_indices]
    
    # Step 5: Sort visibility if provided
    sorted_visibility = visibility[sort_indices] if visibility is not None else None
    
    return sorted_keypoints, remapped_skeleton, sorted_visibility, sort_indices


# Usage in mp100_cape.py __getitem__:
keypoints = np.array(record['keypoints'])
skeleton = record['skeleton']
visibility = np.array(record['visibility'])

sorted_kpts, remapped_skel, sorted_vis, _ = sort_keypoints_with_skeleton_remap(
    keypoints, skeleton, visibility
)

record['keypoints'] = sorted_kpts.tolist()
record['skeleton'] = remapped_skel
record['visibility'] = sorted_vis.tolist()
```

**Critical Considerations:**
1. Must sort BOTH support and query keypoints identically
2. Sorting should happen AFTER bbox cropping but BEFORE tokenization
3. The remapped skeleton must be passed to `adj_from_skeleton()`
4. Consistency: Same category should yield same relative ordering

---

## 5. Recommendation

### Do NOT Implement Geometric Sorting

**Rationale:**

1. The GCN encoder (`GeometricSupportEncoder`) relies on correct skeleton→adjacency mapping
2. Skeleton edges encode semantic relationships that are category-invariant
3. The autoregressive decoder already learns from positional encoding + teacher forcing
4. Geometric sorting provides minimal benefit but risks significant harm

**Instead, Focus On:**

1. Verify skeleton edges are correctly loaded (done ✓)
2. Ensure consistent keypoint ordering from COCO annotations (already true ✓)
3. Use the GCN to leverage structural information (implemented ✓)

**If Ablation is Required:**

Compare performance WITH and WITHOUT GCN pre-encoding (`--use_gcn_preenc`).
This tests the value of skeleton structure without breaking index semantics.

---

## 6. Summary Table

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Keypoints are not sorted | ✅ TRUE | No sorting logic in `mp100_cape.py` |
| `sort_polygons` exists but unused | ✅ TRUE | Exists in `data_utils.py`, not imported |
| Simple sorting breaks skeleton | ✅ TRUE | `adj_from_skeleton()` uses raw indices |
| Sorting helps Transformer | ❌ FALSE | Destroys GCN adjacency semantics |
| Safe sorting requires remapping | ✅ TRUE | See implementation strategy above |

---

*End of Audit Report*

