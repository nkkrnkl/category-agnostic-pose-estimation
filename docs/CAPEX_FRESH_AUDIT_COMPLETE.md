# CapeX Graph Encoding Analysis (Fresh Audit)

**Date**: November 25, 2025  
**Auditor**: AI Assistant  
**Scope**: Complete re-audit of CapeX codebase for geometry-only CAPE adaptation

---

## 1. Overview of Graph and Keypoint Encoding Pipeline

### End-to-End Data Flow

**Raw Annotations → Preprocessed Data → Tokens → Transformer Inputs**

1. **Raw Annotations** (COCO format JSON):
   - Keypoint coordinates: `[x, y, visibility]` for each keypoint
   - Skeleton edges: List of `[idx1, idx2]` pairs (0-indexed in code)
   - Category metadata: `keypoints` (names list), `skeleton` (edges)
   
2. **Dataset Loading** (`transformer_dataset.py:__getitem__`):
   - Support images: Keypoints + skeleton loaded from annotations
   - Query images: Same structure
   - **Text descriptions**: Generated via `rename_points_descriptions()` based on category name/supercategory
   
3. **Text Feature Extraction** (`capex.py:extract_text_features`, lines 298-337):
   - Text descriptions → CLIP/BERT/GTE tokenizer → Text encoder
   - **Output**: `[bs, max_points=100, text_dim]` (e.g., 512 for CLIP)
   - **Critical**: Text embeddings are THE primary support representation
   
4. **Coordinate Processing**:
   - Coordinates normalized to image size (via affine transforms in dataset pipeline)
   - **NO quantization** - kept as continuous `float` values
   - Used for:
     - Initial proposal generation (cross-attention)
     - Positional encoding (sinusoidal)
     - Regression targets

5. **Transformer Forward**:
   - Image features: `[bs, C, H, W]` from backbone
   - Text embeddings: `[bs, num_pts, embed_dim]` projected to model dimension
   - Skeleton: List of edge lists `[[[i,j], ...], ...]` per batch

### Key Architecture Pattern

**CapeX is NOT a sequence model** - it's a **DETR-style set prediction** model:
- Predicts ALL keypoints in parallel
- No BOS/EOS tokens
- No autoregressive decoding
- No discrete tokenization

---

## 2. Keypoint Encoding: Detailed Breakdown

### 2.1 Raw Keypoints Origin

**File**: `transformer_base_dataset.py:_load_coco_keypoint_annotation_kernel`  
**Lines**: 218-224

```python
joints_3d = np.zeros((kpt_num, 3), dtype=np.float32)
joints_3d_visible = np.zeros((kpt_num, 3), dtype=np.float32)

keypoints = np.array(obj['keypoints']).reshape(-1, 3)  # COCO format: [x, y, v]
joints_3d[:cat_kpt_num, :2] = keypoints[:, :2]  # Only (x,y)
joints_3d_visible[:cat_kpt_num, :2] = np.minimum(1, keypoints[:, 2:3])  # visibility
```

### 2.2 Normalization

**Where**: Dataset preprocessing pipeline (affine transforms)  
**How**: 
- Bounding box center + scale computed: `_xywh2cs()` (lines 250-278)
- Coordinates transformed to cropped/resized image space (256x256 default)
- **Final**: Normalized to `[0, 1]` for positional encoding

**Critical Code** (`head.py:forward`, lines 168-169):
```python
point_descriptions = point_descriptions * mask_s  # Zero out invisible points
point_descriptions = self.text_proj(point_descriptions)  # [bs, num_pts, embed_dim]
```

### 2.3 Quantization/Tokenization

**NONE**. CapeX uses continuous coordinate regression, NOT discrete tokens.

**Evidence**:
- `head.py:forward` (lines 186-191): Outputs are continuous via `.sigmoid()`
- No vocabulary, no binning, no discrete token IDs
- Coordinates are regressed directly via MLP (`TokenDecodeMLP`)

### 2.4 Coordinate Embeddings

**CapeX does NOT embed coordinates directly**. Instead:

1. **Text Embeddings** serve as keypoint representations (`capex.py:extract_text_features`):
   ```python
   # CLIP example (lines 312-314)
   tokens = self.tokenizer(all_points).to(device=self.text_backbone_device)
   all_descriptions = self.text_backbone.encode_text(tokens)  # [num_pts, 512]
   all_descriptions = all_descriptions / all_descriptions.norm(dim=1, keepdim=True)
   ```

2. **Coordinates used for positional encoding ONLY** (`positional_encoding.py:forward_coordinates`, lines 97-123):
   ```python
   def forward_coordinates(self, coord):
       # coord: [bs, kpt, 2] in [0, 1]
       x_embed, y_embed = coord[:, :, 0], coord[:, :, 1]
       x_embed = x_embed * self.scale  # scale = 2*pi by default
       y_embed = y_embed * self.scale
       
       # Sinusoidal encoding
       dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=coord.device)
       dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
       
       pos_x = x_embed[:, :, None] / dim_t  # [bs, kpt, num_feats]
       pos_y = y_embed[:, :, None] / dim_t
       
       # Interleave sin/cos
       pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).view(bs, kpt, -1)
       pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).view(bs, kpt, -1)
       
       pos = torch.cat((pos_y, pos_x), dim=2)  # [bs, kpt, num_feats * 2]
       return pos
   ```

### 2.5 Keypoint Ordering

**Semantic ordering via text**, NOT spatial.

**Evidence** (`utils.py:rename_points_descriptions`):
- Hardcoded text descriptions per category (e.g., "front_left_leg", "nose tip", "top left side of backrest")
- Order is defined by semantic meaning, not (x,y) position
- Example (animal_face):
  ```python
  updated_point_names = ['top left side of the left eye', 
                         'bottom right side of the left eye',
                         'bottom left side of the right eye', 
                         'top right side of the right eye',
                         'nose tip', 'left side of the lip', ...]
  ```

**Dataset** (`transformer_dataset.py:__getitem__`, lines 187, 207):
```python
Xs['img_metas'].data['point_descriptions'] = self.cats_points_descriptions[Xs['img_metas'].data['category_id']]
Xq['img_metas'].data['point_descriptions'] = self.cats_points_descriptions[Xq['img_metas'].data['category_id']]
```

### 2.6 Positional Encodings Applied

**Two types**:

1. **Image Feature Positional Encoding** (`head.py:forward`, lines 165-166):
   ```python
   masks = x.new_zeros((x.shape[0], x.shape[2], x.shape[3])).to(torch.bool)
   pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, H, W]
   ```
   - Applied to 2D feature map from backbone
   - DETR-style sinusoidal encoding

2. **Keypoint Coordinate Positional Encoding** (`encoder_decoder.py:forward`, line 181):
   ```python
   initial_position_embedding = position_embedding.forward_coordinates(initial_proposals)
   ```
   - Applied to predicted keypoint coordinates
   - Recalculated dynamically at each decoder layer (line 253):
   ```python
   query_pos_embed = position_embedding.forward_coordinates(bi)  # bi = refined proposals
   ```

### 2.7 Summary: Keypoint Representation

| Component | Implementation | Text-Dependent? |
|-----------|---------------|-----------------|
| Keypoint content | Text embeddings (CLIP/BERT) | ✅ YES |
| Keypoint position | Sinusoidal pos encoding | ❌ NO |
| Keypoint ordering | Semantic (text-based) | ✅ YES |
| Coordinate format | Continuous [0,1] | ❌ NO |
| Tokenization | NONE (set prediction) | N/A |

---

## 3. Edge / Skeleton Encoding: Detailed Breakdown

### 3.1 Skeleton Representation

**Format**: List of edge lists  
**Example**: `skeleton = [[[0,1], [1,2], [2,3]], ...]` (batch of skeletons)

**Source** (`transformer_dataset.py:__getitem__`, lines 210-211):
```python
Xall['skeleton'] = self.db[query_id]['skeleton']  # From COCO 'skeleton' field
```

### 3.2 Adjacency Matrix Construction

**File**: `encoder_decoder.py:adj_from_skeleton`  
**Lines**: 507-521

```python
def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    
    # Build adjacency for each batch element
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b])
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1  # Mark edges
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    
    # Make symmetric (undirected graph)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    
    # Zero out masked keypoints
    adj = adj * ~mask[..., None] * ~mask[:, None]
    
    # Row-normalize
    adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    
    # Dual-channel: [self-loops, adjacency]
    adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)  # [bs, 2, num_pts, num_pts]
    
    return adj
```

**Shape**: `[bs, 2, num_pts, num_pts]`
- Channel 0: Self-loops (identity matrix with masked points zeroed)
- Channel 1: Normalized adjacency matrix

### 3.3 GCN Layer Architecture

**File**: `encoder_decoder.py:GCNLayer`  
**Lines**: 524-555

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=2, use_bias=True, 
                 activation=nn.ReLU(inplace=True), batch_first=True):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features * kernel_size, kernel_size=1,
                              padding=0, stride=1, dilation=1, bias=use_bias)
        self.kernel_size = kernel_size  # Typically 2 (self + neighbors)
        self.activation = activation
        self.batch_first = batch_first

    def forward(self, x, adj):
        assert adj.size(1) == self.kernel_size
        
        # Reshape for conv: [bs, in_feat, num_pts]
        if not self.batch_first:
            x = x.permute(1, 2, 0)  # [num_pts, bs, C] -> [bs, C, num_pts]
        else:
            x = x.transpose(1, 2)   # [bs, num_pts, C] -> [bs, C, num_pts]
        
        x = self.conv(x)  # [bs, out_feat * kernel_size, num_pts]
        
        # Split into kernel_size channels
        b, kc, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)  # [bs, K, out_feat, num_pts]
        
        # Graph convolution via einsum
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))  # [bs, out_feat, num_pts]
        
        if self.activation is not None:
            x = self.activation(x)
        
        # Reshape back
        if not self.batch_first:
            x = x.permute(2, 0, 1)  # [bs, C, num_pts] -> [num_pts, bs, C]
        else:
            x = x.transpose(1, 2)   # [bs, C, num_pts] -> [bs, num_pts, C]
        
        return x
```

**Key Operation**: `torch.einsum('bkcv,bkvw->bcw', (x, adj))`
- Aggregates features from neighbors weighted by adjacency
- `bkcv`: batch, kernel_channel, features, vertices
- `bkvw`: batch, kernel_channel, vertices (source), vertices (target)

### 3.4 Where Graph is Injected

**ONLY in Decoder FFN** (`encoder_decoder.py:GraphTransformerDecoderLayer.forward`, lines 389-401):

```python
if self.graph_decoder is not None:
    num_pts, b, c = refined_support_feat.shape
    adj = adj_from_skeleton(num_pts=num_pts,
                            skeleton=skeleton,
                            mask=tgt_key_padding_mask,
                            device=refined_support_feat.device)
    
    if self.graph_decoder == 'pre':
        # GCN before FFN2
        tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat, adj))))
    elif self.graph_decoder == 'post':
        # GCN after FFN1
        tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat))), adj)
    else:  # 'both'
        # GCN in both FFN1 and FFN2
        tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat, adj))), adj)
else:
    # Standard FFN (no graph)
    tgt2 = self.ffn2(self.dropout(self.activation(self.ffn1(refined_support_feat))))
```

**NOT used in**:
- Encoder (no graph processing)
- Self-attention (standard transformer attention)
- Cross-attention (standard)

### 3.5 Graph vs. Attention

**Graph information is SEPARATE from attention**:
- Attention masks: Only mask padding/invalid keypoints
- Graph structure: Encoded via GCN in FFN, operates on content features
- No "graph-aware attention masks" (no edge-specific masking)

### 3.6 Summary: Edge/Skeleton Encoding

| Component | Implementation | Text-Dependent? |
|-----------|---------------|-----------------|
| Edge representation | List of [i,j] pairs | ❌ NO |
| Adjacency matrix | Dual-channel normalized | ❌ NO |
| GCN architecture | Conv1d + einsum | ❌ NO |
| Integration point | Decoder FFN only | ❌ NO |
| Attention masking | Independent of graph | N/A |

---

## 4. Unified Transformer Sequence Construction

### 4.1 NO Sequence - This is Set Prediction

**Critical Finding**: CapeX does NOT build a unified sequence like autoregressive models.

**Evidence**:
- No BOS/EOS tokens
- No sequential token generation
- All keypoints predicted in parallel

### 4.2 Transformer Input Structure

**Encoder** (`encoder_decoder.py:EncoderDecoder.forward`, lines 164-176):

```python
# Image features: [H*W, bs, C]
src = src.flatten(2).permute(2, 0, 1)

# Image positional encoding: [H*W, bs, C]
pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

# Support text embeddings: [num_pts, bs, C]
query_embed = support_embed.transpose(0, 1)

# Concatenate for joint self-attention
pos_embed = torch.cat((pos_embed, support_order_embed))  # support_order_embed is zeros

# Encoder processes both
query_embed, refined_support_embed = self.encoder(
    src,  # Image features
    query_embed,  # Text embeddings
    src_key_padding_mask=mask,
    query_key_padding_mask=query_padding_mask,
    pos=pos_embed)
```

**Encoder behavior** (`encoder_decoder.py:TransformerEncoder.forward`, lines 433-436):
```python
# Concatenate image and support features
src_cat = torch.cat((src, query), dim=0)  # [H*W + num_pts, bs, C]
mask_cat = torch.cat((src_key_padding_mask, query_key_padding_mask), dim=1)

# Joint self-attention
output = src_cat
for layer in self.layers:
    output = layer(output, ...)
```

**Decoder** (`encoder_decoder.py:GraphTransformerDecoder.forward`, lines 238-269):

```python
# Decoder processes:
# - support_feat: [num_pts, bs, C] (text embeddings refined by encoder)
# - query_feat: [H*W, bs, C] (image features refined by encoder)

for layer_idx, layer in enumerate(self.layers):
    # Recalculate positional encoding from predicted coordinates
    query_pos_embed = position_embedding.forward_coordinates(bi)  # bi = current proposals
    
    refined_support_feat, attn_map = layer(
        refined_support_feat,  # "queries" (one per keypoint to predict)
        refined_query_feat,     # "keys/values" (image features)
        pos=pos,
        query_pos=query_pos_embed,
        skeleton=skeleton)
    
    # Predict coordinate offsets
    delta_bi = kpt_branch[layer_idx](refined_support_feat.transpose(0, 1))
    bi = update(bi, delta_bi)  # Iterative refinement
```

### 4.3 What Gets Predicted

**Per decoder layer**: Coordinate offsets for ALL keypoints simultaneously

**File**: `head.py:forward`, lines 186-191
```python
output_kpts = []
for idx in range(outs_dec.shape[0]):  # For each decoder layer
    layer_delta_unsig = self.kpt_branch[idx](outs_dec[idx])  # [bs, num_pts, 2]
    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(out_points[idx])
    output_kpts.append(layer_outputs_unsig.sigmoid())  # Normalize to [0,1]
```

### 4.4 Summary: NO Sequence Model

| Aspect | CapeX | Autoregressive (Raster2Seq) |
|--------|-------|----------------------------|
| Prediction paradigm | Set (parallel) | Sequence (token-by-token) |
| BOS/EOS tokens | ❌ NO | ✅ YES |
| Causal masking | ❌ NO | ✅ YES |
| Token generation | All at once | One at a time |
| Coordinate format | Continuous | Discrete (quantized) |

---

## 5. What Parts Rely on Text, and Why

### 5.1 Text-Dependent Components

#### 5.1.1 Support Keypoint Representation (CRITICAL)

**Where**: `capex.py:extract_text_features` (lines 298-337)

**Why**: Text embeddings ARE the support representation
- Support images are NOT used (set to 0): `Xs['img'] = 0` (`transformer_base_dataset.py:189`)
- Only text descriptions of keypoint names/locations are encoded
- These text embeddings serve as "queries" in the decoder

**Impact**: This is the CORE semantic information source

#### 5.1.2 Keypoint Ordering

**Where**: `utils.py:rename_points_descriptions`

**Why**: Text descriptions define which keypoint is which
- "front_left_leg" vs "front_right_leg" distinguished by text
- "nose tip" vs "left eye" distinguished by text
- No geometric ordering (spatial position not used for ordering)

**Impact**: Model learns which semantic keypoint index corresponds to which coordinate

#### 5.1.3 Symmetry Breaking

**Where**: Implicit in text descriptions

**Example** (vehicle):
```python
updated_point_names = ['front and right wheel', 'front and left wheel', 
                       'rear and right wheel', 'rear and left wheel', ...]
```

**Why**: Left vs Right is semantic, not purely geometric
- Two wheels at similar y-coordinate distinguished by text "left"/"right"
- Without text, model would need to infer from image context alone

### 5.2 Text-Independent Components

| Component | File | Lines | Pure Geometry? |
|-----------|------|-------|----------------|
| Adjacency matrix | encoder_decoder.py | 507-521 | ✅ YES |
| GCN layers | encoder_decoder.py | 524-555 | ✅ YES |
| Positional encoding | positional_encoding.py | 97-123 | ✅ YES |
| Coordinate regression | head.py | 186-191 | ✅ YES |
| Proposal generation | encoder_decoder.py | 36-110 | ✅ YES (uses cross-attn with image) |

### 5.3 Dependencies Summary

**Text is used for**:
1. **Primary support representation** (replaces support image)
2. **Keypoint identity/semantics** (which keypoint is which)
3. **Ordering** (index 0 = "nose", index 1 = "left_eye", etc.)

**Geometry is used for**:
1. **Positional information** (where keypoints are located)
2. **Graph structure** (which keypoints connect to which)
3. **Prediction targets** (what coordinates to regress to)

---

## 6. Mapping CapeX Logic to Our Geometry-Only Setting

### 6.1 Component-by-Component Analysis

#### 6.1.1 Adjacency Matrix Construction

**Status**: ✅ **Reusable as-is**

**Rationale**: Pure geometry - only needs edge list and mask

**Code to copy**: `encoder_decoder.py:adj_from_skeleton` (lines 507-521)

#### 6.1.2 GCN Layer

**Status**: ✅ **Reusable as-is**

**Rationale**: Operates on arbitrary features + adjacency, no text dependency

**Code to copy**: `encoder_decoder.py:GCNLayer` (lines 524-555)

#### 6.1.3 Sinusoidal Positional Encoding for Coordinates

**Status**: ✅ **Reusable as-is**

**Rationale**: Pure coordinate encoding, no text

**Code to copy**: `positional_encoding.py:SinePositionalEncoding.forward_coordinates` (lines 97-123)

#### 6.1.4 Support Keypoint Representation

**Status**: ❌ **NOT reusable** - must replace entirely

**Current**: Text embeddings from CLIP/BERT
**Replacement**: 
- **Option A**: Coordinate MLP embedding
  ```python
  coord_emb = nn.Sequential(
      nn.Linear(2, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim)
  )(support_coords)  # [bs, num_pts, hidden_dim]
  
  pos_emb = SinePositionalEncoding2D().forward_coordinates(support_coords)
  support_embed = coord_emb + pos_emb
  ```

- **Option B**: Add GCN pre-encoding
  ```python
  adj = adj_from_skeleton(num_pts, skeleton, mask, device)
  support_embed = GCNLayer(hidden_dim, hidden_dim)(support_embed, adj)
  ```

**Critical**: This replaces the TEXT encoder, NOT adds to it

#### 6.1.5 Keypoint Ordering

**Status**: ✏️ **Reusable with modification**

**Current**: Semantic ordering via text
**Options for geometry-only**:

1. **Use annotation order** (RECOMMENDED)
   - Keep order as-is from COCO JSON
   - Assume consistent indexing within category
   - Transformer self-attention is permutation-equivariant
   
2. **Spatial ordering**
   ```python
   # Sort by y-coordinate (top to bottom)
   sorted_indices = torch.argsort(support_coords[:, :, 1], dim=1)
   support_coords = torch.gather(support_coords, 1, sorted_indices)
   ```
   - Risk: Inconsistent across images if keypoints move
   
3. **Graph traversal** (BFS from root)
   - Requires canonical root selection
   - Complex to implement reliably

**Recommendation**: Option 1 (annotation order) + rely on transformer's permutation equivariance

#### 6.1.6 Proposal Generator

**Status**: ✅ **Reusable as-is**

**Rationale**: Uses cross-attention between image and support features
- Doesn't care if support features are text or coordinate embeddings
- Pure geometry: finds where in image each keypoint appears

**Code**: `encoder_decoder.py:ProposalGenerator` (lines 36-110)

#### 6.1.7 Decoder with GCN

**Status**: ✅ **Reusable as-is** (structure) | ✏️ **Needs input adaptation**

**What to keep**:
- Decoder layer structure
- GCN in FFN layers
- Iterative refinement of proposals

**What to change**:
- Input to decoder is NOT text embeddings
- Input is coordinate + positional embeddings

**Code**: `encoder_decoder.py:GraphTransformerDecoder` (lines 199-306)

### 6.2 Challenges Without Text

#### 6.2.1 Symmetry Breaking

**Problem**: Left vs Right keypoints

**CapeX solution**: Text distinguishes "left_wheel" from "right_wheel"

**Geometry-only solutions**:
1. **Spatial context**: Left wheels have smaller x-coordinate
2. **Graph context**: Left wheel connects to left body parts (via skeleton)
3. **Visual features**: Model learns from image which side is which

**Likely outcome**: Model CAN learn this, but may be less reliable than text

#### 6.2.2 Semantic Grouping

**Problem**: Identifying "all leg keypoints" vs "all face keypoints"

**CapeX solution**: Text groups "left_front_leg", "left_back_leg", etc.

**Geometry-only solution**: 
- Graph connectivity (legs connect to hips, not head)
- Spatial proximity
- Visual similarity in image features

#### 6.2.3 Keypoint Identity

**Problem**: Distinguishing "nose" from "tail tip" when both are endpoints

**CapeX solution**: Text labels explicitly

**Geometry-only solution**:
- Relative position in image
- Connection pattern in graph (nose connects to eyes, not body)
- Visual appearance (nose looks different from tail)

### 6.3 Recommended Architecture

**Geometry-Only Support Encoder**:
```
Input: support_coords [bs, N, 2] + skeleton_edges (list) + support_mask [bs, N]

1. Coordinate Embedding:
   coord_emb = MLP(support_coords)  # [bs, N, hidden_dim]

2. Positional Encoding:
   pos_emb = SinePositionalEncoding2D(support_coords)  # [bs, N, hidden_dim]

3. Combine:
   embeddings = coord_emb + pos_emb

4. OPTIONAL: GCN Pre-Encoding
   adj = adj_from_skeleton(N, skeleton_edges, support_mask, device)
   embeddings = GCNLayer(embeddings, adj)

5. Transformer Self-Attention (as in CapeX encoder):
   support_features = TransformerEncoder(embeddings, mask=support_mask)

Output: support_features [bs, N, hidden_dim]
```

**Decoder** (SAME as CapeX):
- Use CapeX's `GraphTransformerDecoder` unchanged
- Feed `support_features` (geometry-based) instead of text embeddings
- Graph is injected via GCN in FFN layers (same as CapeX)

---

## 7. Suggested Implementation Blueprint

### 7.1 Files to Create in Our Repo

**File 1**: `models/graph_utils.py`
```python
# Copy EXACT implementations:
def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    # Lines 507-521 from capex-code/models/models/utils/encoder_decoder.py
    ...

class GCNLayer(nn.Module):
    # Lines 524-555 from capex-code/models/models/utils/encoder_decoder.py
    ...
```

**File 2**: `models/positional_encoding.py` (MODIFY existing)
```python
class SinePositionalEncoding2D(nn.Module):
    def forward_coordinates(self, coord):
        # Lines 97-123 from capex-code/models/models/utils/positional_encoding.py
        ...
```

**File 3**: `models/geometric_support_encoder.py` (NEW)
```python
class GeometricSupportEncoder(nn.Module):
    """
    Geometry-only support encoder inspired by CapeX.
    Replaces text embeddings with coordinate + graph embeddings.
    """
    def __init__(self, hidden_dim=256, num_encoder_layers=3, 
                 use_gcn_preenc=False, nhead=8):
        super().__init__()
        
        # Coordinate embedding (replaces text encoder)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional encoding (from CapeX)
        self.pos_encoding = SinePositionalEncoding2D(
            num_feats=hidden_dim // 2,
            normalize=True,
            scale=2 * math.pi
        )
        
        # Optional GCN pre-encoding
        self.use_gcn_preenc = use_gcn_preenc
        if use_gcn_preenc:
            self.gcn_layers = nn.ModuleList([
                GCNLayer(hidden_dim, hidden_dim, kernel_size=2, batch_first=True)
                for _ in range(2)
            ])
        
        # Transformer encoder (similar to CapeX)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
    
    def forward(self, support_coords, support_mask, skeleton_edges):
        """
        Args:
            support_coords: [bs, N, 2] normalized to [0,1]
            support_mask: [bs, N] (True = invalid)
            skeleton_edges: List of edge lists [[[i,j], ...], ...]
        Returns:
            support_features: [bs, N, hidden_dim]
        """
        # 1. Embed coordinates
        coord_emb = self.coord_mlp(support_coords)  # [bs, N, hidden_dim]
        
        # 2. Add positional encoding
        pos_emb = self.pos_encoding.forward_coordinates(support_coords)
        embeddings = coord_emb + pos_emb
        
        # 3. Optional GCN pre-encoding
        if self.use_gcn_preenc:
            adj = adj_from_skeleton(
                num_pts=support_coords.shape[1],
                skeleton=skeleton_edges,
                mask=support_mask,
                device=support_coords.device
            )
            for gcn in self.gcn_layers:
                embeddings = gcn(embeddings, adj)
        
        # 4. Transformer self-attention
        # [bs, N, C] -> [N, bs, C] (transformer expects seq_len first)
        embeddings = embeddings.transpose(0, 1)
        support_features = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=support_mask
        )
        # [N, bs, C] -> [bs, N, C]
        support_features = support_features.transpose(0, 1)
        
        return support_features
```

### 7.2 Integration with Existing CAPE Model

**Modify**: `models/cape_model.py`

**Changes**:
1. Replace `SupportPoseGraphEncoder` with `GeometricSupportEncoder`
2. Keep existing sequence generation (Raster2Seq) - this is DIFFERENT from CapeX
3. Add GCN to decoder (if using RoomFormer/Deformable Transformer base)

**Critical**: Our model is HYBRID:
- **Support encoding**: CapeX-inspired (parallel + GCN)
- **Query decoding**: Keep Raster2Seq (autoregressive)

This is VALID - we're adapting CapeX's graph encoding, not its entire architecture.

### 7.3 Testing Strategy

**Test 1**: Adjacency Matrix Correctness
```python
def test_adj_from_skeleton():
    skeleton = [[[0, 1], [1, 2], [2, 3]]]
    mask = torch.zeros(1, 4).bool()
    adj = adj_from_skeleton(4, skeleton, mask, 'cpu')
    
    # Check shape
    assert adj.shape == (1, 2, 4, 4)
    
    # Check symmetry
    assert torch.allclose(adj[0, 1], adj[0, 1].T)
    
    # Check normalization (each row sums to 1)
    assert torch.allclose(adj[0, 1].sum(dim=-1), torch.ones(4))
```

**Test 2**: GCN Shape Preservation
```python
def test_gcn_layer():
    gcn = GCNLayer(256, 256, kernel_size=2, batch_first=True)
    x = torch.rand(2, 10, 256)  # [bs, num_pts, feat_dim]
    adj = torch.rand(2, 2, 10, 10)  # [bs, kernel_size, num_pts, num_pts]
    
    out = gcn(x, adj)
    assert out.shape == (2, 10, 256)
```

**Test 3**: Geometric Support Encoder Forward Pass
```python
def test_geometric_support_encoder():
    encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
    
    coords = torch.rand(2, 10, 2)  # [bs, num_pts, 2]
    mask = torch.zeros(2, 10).bool()
    skeleton = [[[0,1], [1,2], [2,3]] for _ in range(2)]
    
    out = encoder(coords, mask, skeleton)
    assert out.shape == (2, 10, 256)
    
    # Check gradients flow
    loss = out.sum()
    loss.backward()
    assert encoder.coord_mlp[0].weight.grad is not None
```

---

## 8. Recommended Tests and Diagnostics

### 8.1 Data Validation Tests

**Test**: Skeleton Edge Indices
```python
def test_skeleton_indices_valid(dataset):
    """Ensure all skeleton edges reference valid keypoint indices."""
    for idx in range(len(dataset)):
        sample = dataset[idx]
        skeleton = sample['skeleton']
        num_keypoints = len(sample['keypoints'])
        
        for edge in skeleton:
            assert edge[0] < num_keypoints, f"Edge {edge} invalid for {num_keypoints} keypoints"
            assert edge[1] < num_keypoints
            assert edge[0] >= 0 and edge[1] >= 0
```

**Test**: Skeleton Connectivity
```python
def test_skeleton_forms_connected_graph(dataset):
    """Check if skeleton edges form a connected graph (optional)."""
    import networkx as nx
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        skeleton = sample['skeleton']
        num_keypoints = len(sample['keypoints'])
        
        G = nx.Graph()
        G.add_nodes_from(range(num_keypoints))
        G.add_edges_from(skeleton)
        
        # Allow disconnected if some keypoints are invisible
        num_components = nx.number_connected_components(G)
        assert num_components >= 1
```

### 8.2 Model Tests

**Test**: Adjacency Matrix Properties
```python
def test_adjacency_symmetry():
    """Adjacency should be symmetric (undirected graph)."""
    skeleton = [[[0,1], [1,2], [0,2]]]
    mask = torch.zeros(1, 3).bool()
    adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
    
    assert torch.allclose(adj[0, 1], adj[0, 1].T, atol=1e-6)

def test_adjacency_normalization():
    """Each row should sum to 1 (or 0 for masked nodes)."""
    skeleton = [[[0,1], [1,2]]]
    mask = torch.tensor([[False, False, False]])
    adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
    
    row_sums = adj[0, 1].sum(dim=-1)
    expected = torch.ones(3)
    assert torch.allclose(row_sums, expected, atol=1e-6)
```

**Test**: GCN Gradient Flow
```python
def test_gcn_gradient_flow():
    """Ensure gradients flow through GCN layers."""
    gcn = GCNLayer(128, 128, kernel_size=2, batch_first=True)
    x = torch.rand(2, 10, 128, requires_grad=True)
    adj = torch.rand(2, 2, 10, 10)
    
    out = gcn(x, adj)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert torch.any(x.grad != 0)
```

### 8.3 Visualization Diagnostics

**Script**: `scripts/visualize_skeleton_graph.py`
```python
def visualize_skeleton_on_image(image, keypoints, skeleton, output_path):
    """
    Overlay skeleton graph on image.
    Args:
        image: numpy array [H, W, 3]
        keypoints: numpy array [N, 2] (x, y coordinates)
        skeleton: list of [i, j] edges
        output_path: where to save
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    # Draw keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=100, zorder=3)
    
    # Draw edges
    lines = []
    for edge in skeleton:
        i, j = edge
        lines.append([keypoints[i], keypoints[j]])
    
    lc = LineCollection(lines, colors='blue', linewidths=2, zorder=2)
    ax.add_collection(lc)
    
    # Add labels
    for idx, (x, y) in enumerate(keypoints):
        ax.text(x, y, str(idx), color='white', fontsize=8, 
                bbox=dict(facecolor='black', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**Script**: `scripts/compare_adjacency_matrices.py`
```python
def visualize_adjacency_matrix(adj, title, output_path):
    """
    Visualize adjacency matrix as heatmap.
    Args:
        adj: [num_pts, num_pts] adjacency matrix
        title: plot title
        output_path: where to save
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(adj, annot=True, fmt='.2f', cmap='Blues', 
                square=True, cbar_kws={'label': 'Weight'})
    ax.set_title(title)
    ax.set_xlabel('Target Keypoint')
    ax.set_ylabel('Source Keypoint')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### 8.4 Consistency Tests

**Test**: Round-Trip Coordinate Encoding
```python
def test_coordinate_positional_encoding_deterministic():
    """Same coordinates should give same encoding."""
    pos_enc = SinePositionalEncoding2D(num_feats=128, normalize=True)
    
    coords = torch.rand(2, 10, 2)
    enc1 = pos_enc.forward_coordinates(coords)
    enc2 = pos_enc.forward_coordinates(coords)
    
    assert torch.allclose(enc1, enc2, atol=1e-6)

def test_different_coordinates_different_encoding():
    """Different coordinates should give different encodings."""
    pos_enc = SinePositionalEncoding2D(num_feats=128, normalize=True)
    
    coords1 = torch.tensor([[[0.1, 0.2]]])
    coords2 = torch.tensor([[[0.5, 0.7]]])
    
    enc1 = pos_enc.forward_coordinates(coords1)
    enc2 = pos_enc.forward_coordinates(coords2)
    
    assert not torch.allclose(enc1, enc2, atol=1e-3)
```

---

## 9. Ambiguities / Open Questions

### 9.1 Support Order Embedding (Disabled in CapeX)

**Code** (`head.py:forward`, line 162):
```python
support_order_embedding = x.new_zeros((bs, self.embed_dims, 1, target_s[0].shape[1])).to(torch.bool)
```

**Question**: Why is this zeros/disabled?
**Impact**: Positional information for support keypoints comes ONLY from coordinate-based positional encoding, NOT from ordering

**Recommendation**: Keep disabled (as CapeX does)

### 9.2 Proposal Generator vs Direct Coordinate Embedding

**CapeX**: Generates initial proposals via cross-attention (image × support features)

**Alternative**: Could we initialize from support coordinates directly?

**Trade-off**:
- CapeX: Proposals are image-aware (good for handling pose variation)
- Direct: Simpler but assumes similar pose in support/query

**Recommendation**: Use CapeX's proposal generator (cross-attention) - it's geometry-agnostic

### 9.3 Graph Decoder Mode

**Options** (`encoder_decoder.py:GraphTransformerDecoderLayer.__init__`, line 321):
- `graph_decoder=None`: No GCN (standard FFN)
- `graph_decoder='pre'`: GCN in FFN1
- `graph_decoder='post'`: GCN in FFN2
- `graph_decoder='both'`: GCN in both FFN1 and FFN2

**CapeX default**: 'pre'

**Question**: Which is best for geometry-only?

**Recommendation**: Start with 'pre' (CapeX default), ablate later

### 9.4 Number of Decoder Layers

**CapeX**: 3 decoder layers (default)

**Question**: Does geometry-only need more/fewer?

**Hypothesis**: 
- More layers might help without text (more refinement needed)
- Fewer might work if graph encoding is strong

**Recommendation**: Start with 3, experiment with 4-6

### 9.5 Coordinate Normalization Scale

**CapeX** (`positional_encoding.py:forward_coordinates`, line 104):
```python
x_embed = x_embed * self.scale  # scale = 2*pi by default
```

**Question**: Is 2π optimal for geometry-only?

**Alternatives**: π, 4π, learnable scale

**Recommendation**: Start with 2π (CapeX default), consider learnable if performance poor

### 9.6 Skeleton Edge Indexing (0 vs 1)

**CapeX code**: Uses 0-indexed edges internally
**MP-100 COCO**: Uses 1-indexed edges in JSON

**Critical**: Must convert 1-indexed → 0-indexed during data loading

**Current code** (`datasets/mp100_cape.py`):
```python
# Check if skeleton uses 1-indexing
if skeleton and min(min(edge) for edge in skeleton) == 1:
    skeleton = [[i-1, j-1] for i, j in skeleton]
```

**Recommendation**: Validate this conversion is applied consistently

---

## 10. Key Takeaways for Geometry-Only CAPE

### 10.1 What We Can Directly Reuse

1. ✅ **Adjacency matrix construction** (`adj_from_skeleton`)
2. ✅ **GCN layers** (dual-channel graph convolution)
3. ✅ **Sinusoidal positional encoding** for coordinates
4. ✅ **Proposal generator** (cross-attention mechanism)
5. ✅ **Decoder architecture** (iterative refinement + GCN)

### 10.2 What We Must Replace

1. ❌ **Text encoder** (CLIP/BERT/GTE) → Coordinate MLP
2. ❌ **Text-based ordering** → Annotation order or spatial
3. ❌ **Semantic keypoint identity** → Learn from geometry + graph

### 10.3 Architecture Differences

| Aspect | CapeX | Our Geometry-Only CAPE |
|--------|-------|------------------------|
| Support representation | Text embeddings | Coord MLP + Pos Enc + GCN |
| Keypoint ordering | Semantic (text) | Annotation order |
| Prediction paradigm | Set (parallel) | Sequence (autoregressive) |
| Coordinate format | Continuous | Discrete (Raster2Seq) |
| Graph usage | Decoder FFN only | Encoder + Decoder |

### 10.4 Expected Performance Impact

**Pessimistic**: 50-60% PCK (text provides significant information)
**Realistic**: 60-70% PCK (graph + geometry helps a lot)
**Optimistic**: 70-80% PCK (our sequence generation might compensate for lack of text)

**Key**: Validation PCK > 60% on unseen categories would validate the geometry-only approach

---

## 11. References

**CapeX Paper**: `capex-code/ICLR-2025-capex-category-agnostic-pose-estimation-from-textual-point-explanation-Paper-Conference.pdf`

**Key Files Audited**:
1. `capex-code/models/models/detectors/capex.py` - Main model, text extraction
2. `capex-code/models/models/keypoint_heads/head.py` - Pose head, text projection
3. `capex-code/models/models/utils/encoder_decoder.py` - Transformer, GCN, adjacency
4. `capex-code/models/models/utils/positional_encoding.py` - Sinusoidal encoding
5. `capex-code/models/datasets/datasets/mp100/transformer_dataset.py` - Data loading
6. `capex-code/models/datasets/datasets/mp100/utils.py` - Text descriptions
7. `capex-code/models/datasets/datasets/mp100/transformer_base_dataset.py` - Base dataset

**Total Lines Reviewed**: ~2000+ lines of core architecture code

---

**END OF AUDIT**

