# Support Image & Edge Graph Usage Audit

**Date:** 2025-11-26  
**Scope:** Deep code-level analysis of support image and skeleton edge usage in CAPE training  
**Conclusion:** Support images are loaded but **NOT encoded or used**. Skeleton edges are properly integrated with graceful fallback for missing graphs.

---

## 1. Support Image Usage During Training

### 1.1 Where Support Images Are Loaded

**File:** `datasets/mp100_cape.py`
- **Function:** `MP100CAPE.__getitem__()` (line 201)
- **Lines 437-444:** Converts PIL image to tensor and normalizes:
  ```python
  img_tensor = torch.as_tensor(self._expand_image_dims(img))
  if self.image_normalize:
      img_tensor = self.image_normalize(img_tensor.float() / 255.0)
  else:
      img_tensor = img_tensor.float() / 255.0
  
  record["image"] = img_tensor
  ```

**File:** `datasets/episodic_sampler.py`
- **Function:** `EpisodicDataset.__getitem__()` (line 183)
- **Line 210:** Loads support data from base dataset:
  ```python
  support_data = self.base_dataset[episode['support_idx']]
  ```
- **Line 345:** Includes support image in episode dict:
  ```python
  return {
      'support_image': support_data['image'],  # <-- LOADED HERE
      'support_coords': support_coords,
      'support_mask': support_mask,
      'support_skeleton': support_skeleton,
      ...
  }
  ```

**File:** `datasets/episodic_sampler.py`
- **Function:** `episodic_collate_fn()` (line 376)
- **Line 389:** Collects support images from episodes:
  ```python
  support_images = []
  ```
- **Line 391-392:** Appends to list:
  ```python
  for episode in batch:
      support_images.append(episode['support_image'])
  ```
- **Line 512:** Batches support images and includes in collated batch:
  ```python
  return {
      'support_images': support_images,  # (B*K, C, H, W) - batched tensor
      ...
  }
  ```

**Summary:** Support images are:
- ✅ Loaded from disk via `MP100CAPE.__getitem__()`
- ✅ Included in episode dicts from `EpisodicDataset`
- ✅ Batched into `(B*K, C, H, W)` tensors by `episodic_collate_fn`
- ✅ Passed to training loop in `batch['support_images']`

---

### 1.2 How (or If) Support Images Are Encoded

#### Where Support Images Are Retrieved

**File:** `models/engine_cape.py`
- **Function:** `train_one_epoch_episodic()` (line 46)
- **Line 116:** Support images loaded from batch:
  ```python
  support_images = batch['support_images'].to(device)  # (B*K, C, H, W)
  ```

- **Function:** `evaluate_cape()` (line 389)
- **Line 452:** Support images loaded from batch:
  ```python
  support_images = batch['support_images'].to(device)  # (B*K, C, H, W)
  ```

- **Function:** `evaluate_unseen_categories()` (line 868)
- **Line 931:** Support images loaded from batch:
  ```python
  support_images = batch['support_images'].to(device)  # (B*K, C, H, W)
  ```

#### Critical Finding: Support Images Are NOT Passed to Model

**File:** `models/engine_cape.py`
- **Lines 159-165 (training, with AMP):**
  ```python
  outputs = model(
      samples=query_images,       # ✅ Query images passed
      support_coords=support_coords,  # ✅ Support coords passed
      support_mask=support_masks,     # ✅ Support mask passed
      targets=query_targets,
      skeleton_edges=support_skeletons
      # ❌ support_images NOT passed
  )
  ```

- **Lines 175-181 (training, FP32):**
  ```python
  outputs = model(
      samples=query_images,
      support_coords=support_coords,
      support_mask=support_masks,
      targets=query_targets,
      skeleton_edges=support_skeletons
      # ❌ support_images NOT passed
  )
  ```

**Same pattern in evaluation functions:** `evaluate_cape()` and `evaluate_unseen_categories()` both load `support_images` from the batch but **never pass them to the model**.

#### Model Forward Signature

**File:** `models/cape_model.py`
- **Function:** `CAPEModel.forward()` (line 119)
- **Signature:**
  ```python
  def forward(self, samples, support_coords, support_mask, targets=None, skeleton_edges=None):
  ```
  
  **Parameters:**
  - `samples`: Query images ✅
  - `support_coords`: Support keypoint coordinates ✅
  - `support_mask`: Support keypoint visibility mask ✅
  - `targets`: Query targets (for training) ✅
  - `skeleton_edges`: Skeleton edge lists ✅
  - **`support_images`: NOT PRESENT ❌**

#### What Happens to Support Information

**File:** `models/cape_model.py`
- **Line 211:** Only support **coordinates** are encoded:
  ```python
  support_features = self.support_encoder(support_coords, support_mask, skeleton_edges)
  #                                       ^^^^^^^^^^^^^^  <-- Only coords, not images!
  ```

#### Support Encoder Implementations

**GeometricSupportEncoder** (`models/geometric_support_encoder.py`):
- **Line 134:** Forward signature:
  ```python
  def forward(self, support_coords, support_mask, skeleton_edges):
  #                 ^^^^^^^^^^^^^^  <-- Only coords, no image parameter
  ```
- **Lines 166-173:** Encodes coordinates via MLP and positional encoding:
  ```python
  coord_emb = self.coord_mlp(support_coords)  # MLP on (x, y) only
  pos_emb = self.pos_encoding.forward_coordinates(support_coords)
  embeddings = coord_emb + pos_emb
  ```
- **No backbone processing of support images**

**SupportPoseGraphEncoder** (`models/support_encoder.py`):
- **Line 96:** Forward signature:
  ```python
  def forward(self, support_coords, support_mask=None, skeleton_edges=None):
  #                 ^^^^^^^^^^^^^^  <-- Only coords, no image parameter
  ```
- **Line 121:** Embeds coordinates only:
  ```python
  coord_emb = self.coord_embedding(support_coords)  # MLP on (x, y) only
  ```
- **No backbone processing of support images**

#### Does Support Image Go Through a Backbone?

**Answer: NO**

- ❌ Support images are **never passed** to any encoder or backbone
- ❌ No CNN or vision transformer processes support images
- ❌ No visual features are extracted from support images
- ✅ Only support **coordinates** (x, y positions) are encoded
- ✅ Only support **skeleton edges** (graph structure) are used

---

### 1.3 Do We *Need* to Encode the Support Image?

#### Current Architectural Intent (Based on Code)

The architecture is **explicitly designed for coordinate-only support encoding**:

1. **Support encoders accept only coordinates:**
   - Both `GeometricSupportEncoder` and `SupportPoseGraphEncoder` have no image parameters
   - Only coordinate MLPs and graph encoders are present

2. **No infrastructure for image encoding:**
   - No CNN/ViT backbone for support images
   - No cross-attention between support visual features and query visual features
   - Support features are derived purely from (x, y) coordinates and skeleton structure

3. **Deliberate design choice:**
   - Support images are loaded (for potential future use or debugging/visualization)
   - But intentionally not used in the forward pass
   - Model learns from support **pose structure** (coordinates + skeleton), not appearance

#### Potential Benefits of Current Approach (Coordinate-Only)

✅ **Simplicity:**
- Fewer parameters (no support image encoder)
- Faster training (no support image backbone forward pass)

✅ **Category-agnostic design:**
- Forces model to learn pose structure, not appearance
- Reduces risk of overfitting to support appearance

✅ **Few-shot generalization:**
- Support provides structural prior (keypoint relationships)
- Query image provides appearance information
- Clean separation of concerns

#### Potential Risks of Current Approach

⚠️ **Missing appearance cues:**
- Support appearance could help disambiguate pose (e.g., left vs right limb)
- Texture/scale information from support is lost
- Model cannot learn appearance-conditioned pose patterns

⚠️ **Unused data:**
- Support images are loaded but never used
- Wasted I/O and memory (could skip loading support images entirely)

⚠️ **Design inconsistency:**
- If support images aren't needed, why load them?
- If they are needed, why not use them?

#### Recommendation

**Current behavior is intentional but should be documented:**

1. **If coordinate-only is desired:**
   - Add comment in `episodic_collate_fn` explaining why `support_images` are loaded but not used
   - Consider adding a `load_support_images=False` flag to skip I/O if not needed
   - Document this design choice in model README

2. **If support image encoding is desired:**
   - Add a support image backbone (CNN or ViT)
   - Add cross-attention between support visual features and query features
   - Modify `CAPEModel.forward()` to accept `support_images` parameter
   - This would require significant architectural changes

**Current implementation is valid for coordinate-only CAPE, but the inconsistency (loading but not using) should be clarified.**

---

## 2. Edge Graph (Skeleton) Usage During Training

### 2.1 Dataset-Level Definition

#### Where Skeletons Are Defined

**File:** `datasets/mp100_cape.py`
- **Function:** `_get_skeleton_for_category()` (line 494)
- **Lines 505-517:**
  ```python
  def _get_skeleton_for_category(self, category_id):
      """
      Get skeleton edges for a given category from COCO annotations.
      
      Returns:
          List of [src, dst] edge pairs defining the skeleton structure.
          Returns empty list if no skeleton defined for this category.
      """
      try:
          # Get category info from COCO
          cat_info = self.coco.loadCats(category_id)[0]
          
          # MP-100 stores skeleton in 'skeleton' field
          skeleton = cat_info.get('skeleton', [])
          
          return skeleton if skeleton else []
      
      except Exception as e:
          # If category not found or error, return empty skeleton
          return []
  ```

**Key Behaviors:**
- Line 511: Retrieves skeleton from COCO category metadata via `cat_info.get('skeleton', [])`
- Line 513: Returns empty list `[]` if skeleton not present
- Line 517: Returns empty list `[]` on any exception (category not found, etc.)

#### Skeleton Representation Format

**Format:** List of edge pairs `[[src, dst], ...]`
- Each edge is `[src, dst]` where `src` and `dst` are keypoint indices
- **Indexing:** MP-100 uses **1-indexed** keypoints (1, 2, 3, ..., N)
- **Graph type:** Undirected (edges are bidirectional)
- **Example:**
  ```python
  skeleton = [[1, 2], [2, 3], [3, 4]]  # Linear chain: 1-2-3-4
  ```

#### Categories With/Without Skeletons

**Discovery from code analysis:**

The `_get_skeleton_for_category()` method retrieves skeletons from the COCO annotation file's category metadata. The MP-100 dataset includes skeleton definitions in the `'skeleton'` field of each category.

**From dataset documentation (`docs/MP100_CATEGORY_ANALYSIS.md`):**
- Total categories: 99 (excluding category 80 "hand")
- Train categories (split 1): 69
- Val categories (split 1): 10
- Test categories (split 1): 20

**Skeleton availability is category-specific:**
- Some categories have full skeleton definitions (e.g., body poses)
- Some categories may have empty skeletons (e.g., face landmarks with no defined connectivity)
- The code does **not** log or report which categories lack skeletons
- No statistics are available in the codebase about skeleton coverage

**Finding:** There is **no explicit logging or analysis** in the code to identify which categories have skeletons vs. which don't. The system silently returns `[]` for categories without skeletons.

---

### 2.2 Pipeline Integration (Episodic Sampler → Collate → Model)

#### How Skeletons Are Passed Through Batches

**Step 1: Dataset Loading**

**File:** `datasets/mp100_cape.py`
- **Line 405:** Skeleton attached to each sample:
  ```python
  record["skeleton"] = self._get_skeleton_for_category(category_ids[0])
  ```

**Step 2: Episode Construction**

**File:** `datasets/episodic_sampler.py`
- **Function:** `EpisodicDataset.__getitem__()` (line 183)
- **Line 235-242:** Support skeleton retrieved from support data:
  ```python
  support_skeleton = support_data.get('skeleton', [])
  
  # Validate skeleton indices (optional)
  if support_skeleton and len(support_skeleton) > 0:
      # Check that skeleton indices don't exceed num keypoints
      max_idx = max([max(edge) for edge in support_skeleton if len(edge) == 2])
      if max_idx > len(support_coords):
          # Issue warning but continue
          support_skeleton = []
  ```

- **Line 348:** Skeleton included in episode dict:
  ```python
  return {
      ...
      'support_skeleton': support_skeleton,  # List of [src, dst] pairs or []
      ...
  }
  ```

**Step 3: Batch Collation**

**File:** `datasets/episodic_sampler.py`
- **Function:** `episodic_collate_fn()` (line 376)
- **Lines 490-498:** Skeletons collected and repeated per query:
  ```python
  support_skeletons_repeated = []
  for episode in batch:
      skeleton = episode.get('support_skeleton', [])
      # Repeat skeleton K times (once per query in episode)
      support_skeletons_repeated.extend([skeleton] * queries_per_episode)
  ```

- **Line 515:** Skeletons included in batch:
  ```python
  return {
      ...
      'support_skeletons': support_skeletons_repeated,  # List of B*K skeleton edge lists
      ...
  }
  ```

**Alignment Guarantee:**
- Each support skeleton is repeated `queries_per_episode` times
- `support_skeletons[i]` corresponds to `support_coords[i]` and `query_images[i]`
- Batch size: `(B*K)` where `B` = episodes, `K` = queries per episode

#### Alignment With Support/Query

**File:** `datasets/episodic_sampler.py`
- **Lines 440-498:** Critical alignment fix implemented
- **Key insight (lines 443-457):**
  ```python
  # ========================================================================
  # CRITICAL FIX: Align support and query dimensions for 1-shot learning
  # ========================================================================
  # Currently we have:
  #   - support_coords:  (B, max_kpts, 2)  where B = number of episodes
  #   - query_images:    (B*K, C, H, W)    where K = queries per episode
  # 
  # Problem: The model cannot tell which support goes with which query
  # because support is per-episode but queries are flattened across episodes.
  #
  # Solution: Repeat each support K times (once per query in that episode)
  # so that support_coords[i] corresponds to query_images[i].
  # ========================================================================
  ```

**Result:**
- ✅ `support_coords[i]`, `support_skeletons[i]`, and `query_images[i]` are aligned
- ✅ All have batch size `(B*K)`
- ✅ 1-shot episodic structure is preserved (each query has exactly one support)

---

### 2.3 Model-Level Usage

#### Where Edges Enter the Encoder

**File:** `models/cape_model.py`
- **Line 211:** Skeleton edges passed to support encoder:
  ```python
  support_features = self.support_encoder(support_coords, support_mask, skeleton_edges)
  #                                                                      ^^^^^^^^^^^^^^^
  ```

#### GeometricSupportEncoder Usage

**File:** `models/geometric_support_encoder.py`

**Input (line 137):**
```python
def forward(self, support_coords, support_mask, skeleton_edges):
    """
    Args:
        skeleton_edges (list): List of edge lists (one per batch element)
                              Each element is list of [i, j] pairs (0-indexed)
    """
```

**Usage (lines 182-190):**
```python
# 5. Optional GCN pre-encoding (from CapeX)
if self.use_gcn_preenc and self.gcn_layers is not None:
    # Build adjacency matrix from skeleton
    adj = adj_from_skeleton(num_pts, skeleton_edges, support_mask, device)
    # adj: [bs, 2, num_pts, num_pts]
    
    # Apply GCN layers sequentially
    for gcn_layer in self.gcn_layers:
        embeddings = gcn_layer(embeddings, adj)  # [bs, num_pts, hidden_dim]
```

**Key points:**
- Skeleton edges are **only used if** `use_gcn_preenc=True` (CLI flag `--use_gcn_preenc`)
- If GCN is disabled, skeleton edges are **ignored** (no error, just not used)
- GCN applies graph convolution using the skeleton as adjacency structure

#### Adjacency Matrix Construction

**File:** `models/graph_utils.py`
- **Function:** `adj_from_skeleton()` (line 15)
- **Lines 50-64:** Handling of edge list:
  ```python
  for b in range(batch_size):
      edges = torch.tensor(skeleton[b], device=device)
      adj = torch.zeros(num_pts, num_pts, device=device)
      if len(edges) > 0:  # ✅ Only add edges if skeleton is non-empty
          # CRITICAL FIX: Filter out edges with indices >= num_pts
          valid_mask = (edges[:, 0] < num_pts) & (edges[:, 1] < num_pts)
          valid_edges = edges[valid_mask]
          
          if len(valid_edges) > 0:
              adj[valid_edges[:, 0], valid_edges[:, 1]] = 1
      adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
  ```

**Behavior:**
- Line 53: If `skeleton[b]` is empty (`len(edges) == 0`), creates a **zero adjacency matrix**
- Lines 55-60: Filters edges with out-of-bounds indices (graceful handling)
- Result: `adj` is always a valid tensor, even if skeleton is empty

**Dual-Channel Adjacency:**
- **Channel 0:** Self-loops (identity matrix)
- **Channel 1:** Normalized neighbor adjacency from skeleton

**Lines 66-75:**
```python
# Make symmetric (undirected graph)
trans_adj_mx = torch.transpose(adj_mx, 1, 2)
...
adj = adj_mx + trans_adj_mx * cond - adj_mx * cond

# Zero out rows/columns for masked keypoints
adj = adj * ~mask[..., None] * ~mask[:, None]

# Row-normalize
adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
```

**Key insight:**
- Empty skeleton → zero adjacency matrix → **GCN becomes a no-op** (no message passing)
- Self-loop channel still exists, so node features are preserved

#### SupportPoseGraphEncoder Usage

**File:** `models/support_encoder.py`

**Input (line 96):**
```python
def forward(self, support_coords, support_mask=None, skeleton_edges=None):
    """
    Args:
        skeleton_edges: Optional list of edge lists, one per batch item
                       If None, no skeletal structure is used
    """
```

**Usage (lines 124-138):**
```python
# 2. Add skeleton edge information (adjacency embeddings)
if skeleton_edges is not None and len(skeleton_edges) > 0:
    # Create adjacency matrix from edges
    adj_matrix = self._build_adjacency_matrix(skeleton_edges, N, device)
    
    # Aggregate edge information for each keypoint
    edge_info = self._aggregate_edge_embeddings(adj_matrix, N, device)
    
    # Combine coordinate and edge embeddings
    combined = torch.cat([coord_emb, edge_info], dim=-1)  # (B, N, 2D)
    embeddings = self.coord_edge_proj(combined)  # (B, N, D)
else:
    # ✅ No skeleton information - use only coordinates
    embeddings = coord_emb
```

**Graceful fallback:**
- Line 124: Checks if `skeleton_edges is not None and len(skeleton_edges) > 0`
- Lines 125-135: If skeleton exists, uses edge embeddings
- **Lines 136-138: If skeleton is missing, uses only coordinate embeddings (graceful degradation)**

#### How Edges Affect Attention/Embeddings

**GeometricSupportEncoder:**
- Edges → Adjacency matrix → GCN layers → Graph-aware embeddings
- GCN performs message passing along skeleton edges
- If edges are missing, GCN receives zero adjacency (no message passing, only self-loops)
- Transformer still applies full self-attention (not restricted by skeleton)

**SupportPoseGraphEncoder:**
- Edges → Adjacency matrix → Edge embeddings → Combined with coordinate embeddings
- Edge embeddings encode neighbor connectivity
- If edges are missing, only coordinate embeddings are used
- Transformer still applies full self-attention

**Key difference:**
- Skeleton edges provide **inductive bias** (graph structure prior)
- But transformer **self-attention is unrestricted** (not edge-masked)
- Edges augment node features, they don't constrain attention

---

## 3. Handling Categories Without Skeletons

### 3.1 Detection / Representation of Missing Skeletons

#### How Missing Skeletons Are Indicated

**File:** `datasets/mp100_cape.py`
- **Line 511:** Missing skeleton represented as empty list:
  ```python
  skeleton = cat_info.get('skeleton', [])  # Returns [] if 'skeleton' key missing
  ```
- **Line 513:** Explicit check:
  ```python
  return skeleton if skeleton else []  # Returns [] if skeleton is falsy
  ```

**Representation:**
- Missing skeleton: `[]` (empty list)
- Present skeleton: `[[1, 2], [2, 3], ...]` (list of edge pairs)

#### Any Warnings/Logs

**Finding:** **NO warnings or logs for missing skeletons**

**Code does NOT:**
- ❌ Log when a category has no skeleton
- ❌ Print statistics about skeleton coverage
- ❌ Warn during training about mixed skeleton/no-skeleton batches
- ❌ Track which categories lack skeletons

**Silent behavior:**
- `_get_skeleton_for_category()` silently returns `[]`
- `adj_from_skeleton()` silently creates zero adjacency for `[]`
- `SupportPoseGraphEncoder.forward()` silently uses coordinate-only mode for `[]`
- `GeometricSupportEncoder.forward()` silently skips GCN if adjacency is zero

**Recommendation:** Add optional logging:
```python
if len(skeleton) == 0:
    logger.debug(f"Category {category_id} has no skeleton definition (using coordinate-only encoding)")
```

---

### 3.2 Fallback Behavior in Encoders

#### GeometricSupportEncoder

**When skeleton is empty (`skeleton_edges[i] == []`):**

1. **Lines 182-183:** GCN check:
   ```python
   if self.use_gcn_preenc and self.gcn_layers is not None:
   ```
   - If `use_gcn_preenc=False`, skeleton is **never used** (skipped entirely)

2. **Line 185:** Adjacency matrix construction:
   ```python
   adj = adj_from_skeleton(num_pts, skeleton_edges, support_mask, device)
   ```
   - `adj_from_skeleton()` returns dual-channel adjacency:
     - Channel 0: Identity (self-loops)
     - Channel 1: **All zeros** (no edges)

3. **Lines 188-190:** GCN layers:
   ```python
   for gcn_layer in self.gcn_layers:
       embeddings = gcn_layer(embeddings, adj)
   ```
   - GCN receives zero adjacency for channel 1
   - **Message passing becomes self-loop only** (no neighbor aggregation)
   - Equivalent to MLP applied to each node independently

4. **Line 195-198:** Transformer encoder:
   ```python
   support_features = self.transformer_encoder(
       embeddings,
       src_key_padding_mask=support_mask
   )
   ```
   - **Transformer attention is UNRESTRICTED** (not edge-masked)
   - Full self-attention across all keypoints
   - Provides global context even without skeleton edges

**Net effect:**
- Skeleton edges provide graph structure bias via GCN
- If edges missing: GCN is no-op, but transformer still provides global context
- **Graceful degradation:** coordinate embeddings + positional encoding + transformer attention

---

#### SupportPoseGraphEncoder

**When skeleton is empty (`skeleton_edges is None or skeleton_edges == []`):**

1. **Line 124:** Explicit check:
   ```python
   if skeleton_edges is not None and len(skeleton_edges) > 0:
   ```

2. **Lines 136-138:** Fallback path:
   ```python
   else:
       # No skeleton information - use only coordinates
       embeddings = coord_emb
   ```
   - Skips edge embedding entirely
   - Uses **only coordinate embeddings** from MLP

3. **Line 141:** Positional encoding:
   ```python
   embeddings = self.pos_embedding(embeddings)
   ```
   - 1D sequence positional encoding (keypoint ordering)
   - Applied regardless of skeleton presence

4. **Lines 155-158:** Transformer encoder:
   ```python
   support_features = self.transformer_encoder(
       embeddings,
       src_key_padding_mask=attn_mask
   )
   ```
   - **Full self-attention** (unrestricted)
   - Provides global context even without skeleton

**Net effect:**
- With skeleton: `coordinate_emb + edge_emb + pos_enc + transformer`
- Without skeleton: `coordinate_emb + pos_enc + transformer`
- **Graceful degradation:** loses edge information but retains coordinates and global context

---

### 3.3 Potential Impact on Training Stability

#### Does Mixing Skeleton/No-Skeleton Categories Destabilize Training?

**Analysis:**

**1. Distribution Shift in Support Representations**

**With skeleton:**
- Support embeddings: `f(coords, edges, attention)`
- Richer inductive bias (graph structure prior)
- May lead to more structured embeddings

**Without skeleton:**
- Support embeddings: `f(coords, attention)`
- Less inductive bias (only coordinates and global attention)
- May lead to less structured embeddings

**Potential issue:**
- If 50% of categories have skeletons and 50% don't, the model sees two different "types" of support representations
- Could create a **bimodal distribution** in the support embedding space
- Model may struggle to learn a unified strategy for using support

**Mitigation:**
- Both encoders fall back to **coordinate + transformer** (common base)
- Skeleton edges are **additive** (not required)
- Transformer self-attention provides global context in both cases

**Verdict:** ⚠️ **Minor risk, but mitigated by fallback behavior**

---

**2. Inconsistent Inductive Bias**

**With skeleton:**
- GCN message passing enforces local connectivity
- Model learns to use edge relationships for pose structure

**Without skeleton:**
- No local connectivity bias
- Model relies on transformer's learned attention patterns

**Potential issue:**
- Model may learn different strategies for skeleton vs. no-skeleton categories
- Could lead to **inconsistent generalization** across category types

**Mitigation:**
- Transformer is the final encoder (same for both)
- Transformer can learn to attend to relevant keypoints regardless of skeleton
- Skeleton edges are a **soft bias**, not a hard constraint

**Verdict:** ⚠️ **Minor risk, but model has capacity to adapt**

---

**3. Empirical Evidence (from code)**

**No explicit handling or special cases:**
- Code does **not** separate skeleton/no-skeleton categories into different batches
- Code does **not** use different loss weights for skeleton/no-skeleton samples
- Code does **not** log or monitor performance stratified by skeleton presence

**Implication:**
- The design **assumes mixing is acceptable**
- If instability occurred, it would manifest as:
  - Higher loss variance
  - Poor performance on no-skeleton categories
  - Difficulty learning a unified support encoder

**Current code does not track these metrics, so we cannot verify stability empirically from code alone.**

---

**4. Recommended Improvements**

To better understand and mitigate potential issues:

**A. Add skeleton presence logging:**
```python
# In episodic_sampler.py or mp100_cape.py
def analyze_skeleton_coverage(dataset):
    skeleton_present = 0
    skeleton_missing = 0
    for cat_id in dataset.categories:
        skeleton = dataset._get_skeleton_for_category(cat_id)
        if len(skeleton) > 0:
            skeleton_present += 1
        else:
            skeleton_missing += 1
    print(f"Skeleton coverage: {skeleton_present}/{skeleton_present + skeleton_missing} "
          f"({100 * skeleton_present / (skeleton_present + skeleton_missing):.1f}%)")
```

**B. Add optional skeleton-aware metrics:**
```python
# In evaluation, stratify PCK by skeleton presence
pck_with_skeleton = ...
pck_without_skeleton = ...
print(f"PCK (with skeleton): {pck_with_skeleton:.2%}")
print(f"PCK (without skeleton): {pck_without_skeleton:.2%}")
```

**C. Consider category-conditional encoding:**
```python
# If skeleton presence correlates with category type (e.g., body vs. face),
# add a learnable category-type embedding to support features
```

**D. Add option to filter categories by skeleton presence:**
```python
# For ablation studies
parser.add_argument('--require_skeleton', action='store_true',
                    help='Only use categories with skeleton definitions')
```

---

#### Red Flags or Recommendations

**🔴 RED FLAG #1: No visibility into skeleton coverage**
- **Issue:** Code does not log or track which categories have skeletons
- **Risk:** Silent performance degradation on no-skeleton categories
- **Fix:** Add skeleton coverage analysis at dataset initialization

**🟡 YELLOW FLAG #2: Support images loaded but not used**
- **Issue:** `support_images` loaded from disk, batched, moved to GPU, but never used
- **Risk:** Wasted I/O, memory, and GPU bandwidth
- **Fix:** Either use support images or stop loading them (add `load_support_images=False` flag)

**🟡 YELLOW FLAG #3: No stratified evaluation**
- **Issue:** Cannot verify if model performs equally well on skeleton vs. no-skeleton categories
- **Risk:** Hidden performance bias
- **Fix:** Add skeleton-stratified metrics in evaluation

**✅ GREEN FLAG #1: Graceful fallback behavior**
- Both encoders handle missing skeletons without crashing
- Coordinate-only mode is a reasonable fallback
- No errors or exceptions

**✅ GREEN FLAG #2: Skeleton edges are optional by design**
- `--use_gcn_preenc` flag makes GCN usage explicit
- Transformer provides global context regardless of skeleton
- Flexible architecture

---

## Summary Table

| Component | Loaded? | Passed to Model? | Encoded? | Used in Training? | Fallback Behavior |
|-----------|---------|------------------|----------|-------------------|-------------------|
| **Support Images** | ✅ Yes | ❌ No | ❌ No | ❌ **NOT USED** | N/A (never used) |
| **Support Coords** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A (always present) |
| **Support Mask** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A (always present) |
| **Skeleton Edges** | ✅ Yes | ✅ Yes | ⚠️ Conditional | ⚠️ Conditional | ✅ Coordinate-only mode |
| **Query Images** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A (always present) |

---

## Key Findings

### Support Images
1. **Loaded but not used:** Support images are loaded from disk, batched, and moved to GPU, but **never passed to the model or encoded**.
2. **Wasted resources:** This wastes I/O bandwidth, CPU/GPU memory, and data loading time.
3. **Architectural intent:** The design is **explicitly coordinate-only** for support encoding (no image backbone).
4. **Recommendation:** Either start using support images (requires architectural changes) or stop loading them (add flag to skip).

### Skeleton Edges
1. **Properly integrated:** Skeleton edges are correctly loaded, batched, aligned, and passed through the pipeline.
2. **Graceful fallback:** Both encoders handle missing skeletons without errors (coordinate-only mode).
3. **Conditional usage:** Skeleton edges are only used if `--use_gcn_preenc` is enabled (GeometricSupportEncoder) or by default (SupportPoseGraphEncoder).
4. **Silent behavior:** No logging about which categories have/lack skeletons.
5. **Potential stability concern:** Mixing skeleton/no-skeleton categories may create bimodal support distributions, but graceful fallback mitigates this.
6. **Recommendation:** Add skeleton coverage logging and stratified evaluation metrics.

---

## Final Verdict

**Support Images:**
- ❌ **NOT USED** in current implementation
- ⚠️ Inconsistency between loading and usage
- ✅ Intentional coordinate-only design (valid choice)
- 💡 Clarify intent: document why loaded but not used, or remove loading

**Skeleton Edges:**
- ✅ **PROPERLY INTEGRATED** with graceful fallback
- ✅ Handles missing skeletons without errors
- ⚠️ Lacks visibility into skeleton coverage
- 💡 Add logging and stratified metrics for better understanding

**Overall:**
- Code is **logically consistent** (no bugs or crashes)
- Architecture is **intentionally coordinate-only for support**
- Main improvement: **document design choices** and **add visibility into skeleton usage**

