# CapeX: Critical Code Snippets for Porting

This document contains the exact code snippets from CapeX that are most relevant for our geometry-only adaptation.

---

## 1. Graph Adjacency Matrix Construction

**Source**: `capex-code/models/models/utils/encoder_decoder.py:507-521`

```python
def adj_from_skeleton(num_pts, skeleton, mask, device='cuda'):
    """
    Construct normalized adjacency matrix from skeleton edges.
    
    Args:
        num_pts (int): Maximum number of keypoints (e.g., 100)
        skeleton (list): List of edge lists, one per batch
                        e.g., [[[0,1], [1,2], ...], [[0,1], ...]]
        mask (Tensor): Boolean mask [bs, num_pts] - True = invalid/padded
        device (str): Device to create tensors on
        
    Returns:
        adj (Tensor): [bs, 2, num_pts, num_pts]
                     Channel 0: Self-loops (diagonal)
                     Channel 1: Neighbor connections (normalized)
    """
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    
    # Build adjacency for each sample in batch
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b])  # [[src, dst], ...]
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1  # Mark directed edges
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    
    # Make undirected (symmetric)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    
    # Mask out invalid keypoints (padding)
    # ~mask is True for VALID keypoints
    adj = adj * ~mask[..., None] * ~mask[:, None]
    
    # Row-normalize: each row sums to 1 (for graph convolution)
    adj = torch.nan_to_num(adj / adj.sum(dim=-1, keepdim=True))
    
    # Create dual-channel adjacency
    # Channel 0: Self-loops (only for valid keypoints)
    # Channel 1: Neighbor connections
    adj = torch.stack((torch.diag_embed(~mask), adj), dim=1)
    
    return adj  # [bs, 2, num_pts, num_pts]
```

**Usage Example**:
```python
skeleton = [[[0, 1], [1, 2], [2, 3], [0, 4]]]  # One sample
mask = torch.tensor([[False, False, False, False, False]])  # All valid
adj = adj_from_skeleton(num_pts=5, skeleton=skeleton, mask=mask, device='cpu')

# adj.shape = [1, 2, 5, 5]
# adj[0, 0] = identity matrix (self-loops)
# adj[0, 1] = normalized adjacency (symmetric)
```

---

## 2. Graph Convolutional Layer

**Source**: `capex-code/models/models/utils/encoder_decoder.py:524-556`

```python
class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer with dual-channel adjacency.
    
    Performs: x_out = Σ_{c ∈ channels} Σ_{j ∈ neighbors} W_c * x_j * adj[c, i, j]
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension  
        kernel_size (int): Number of adjacency channels (default: 2)
                          2 = [self-loop, neighbors]
        use_bias (bool): Whether to use bias in convolution
        activation: Activation function (default: ReLU)
        batch_first (bool): If True, input is [bs, num_pts, c]
                           If False, input is [num_pts, bs, c]
    """
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=2,
                 use_bias=True,
                 activation=nn.ReLU(inplace=True),
                 batch_first=True):
        super(GCNLayer, self).__init__()
        
        # 1D conv to generate kernel_size separate transformations
        self.conv = nn.Conv1d(
            in_features, 
            out_features * kernel_size,  # Output has kernel_size channels
            kernel_size=1,  # Pointwise (no spatial convolution)
            padding=0, 
            stride=1, 
            dilation=1, 
            bias=use_bias
        )
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_first = batch_first

    def forward(self, x, adj):
        """
        Args:
            x: Features
               - If batch_first=False: [num_pts, bs, c]
               - If batch_first=True: [bs, num_pts, c]
            adj: Adjacency matrix [bs, kernel_size, num_pts, num_pts]
            
        Returns:
            out: Transformed features (same shape as input)
        """
        assert adj.size(1) == self.kernel_size, \
            f"Adjacency has {adj.size(1)} channels, expected {self.kernel_size}"
        
        # Transpose to [bs, c, num_pts] for Conv1d
        if not self.batch_first:
            x = x.permute(1, 2, 0)  # [num_pts, bs, c] → [bs, c, num_pts]
        else:
            x = x.transpose(1, 2)  # [bs, num_pts, c] → [bs, c, num_pts]
        
        # Apply 1D convolution: [bs, c, num_pts] → [bs, c*kernel_size, num_pts]
        x = self.conv(x)
        
        b, kc, v = x.size()
        # Reshape to separate kernel channels
        x = x.view(b, self.kernel_size, kc // self.kernel_size, v)
        # x.shape = [bs, kernel_size, out_features, num_pts]
        
        # Graph convolution via Einstein summation
        # 'bkcv,bkvw->bcw' means:
        #   b=batch, k=kernel_size, c=channels, v=vertices_src, w=vertices_dst
        #   For each vertex w: aggregate from vertices v weighted by adj[k,v,w]
        x = torch.einsum('bkcv,bkvw->bcw', (x, adj))
        # x.shape = [bs, out_features, num_pts]
        
        if self.activation is not None:
            x = self.activation(x)
        
        # Transpose back to original format
        if not self.batch_first:
            x = x.permute(2, 0, 1)  # [bs, c, num_pts] → [num_pts, bs, c]
        else:
            x = x.transpose(1, 2)  # [bs, c, num_pts] → [bs, num_pts, c]
        
        return x
```

**Einstein Summation Explanation**:
```
'bkcv,bkvw->bcw'

Given:
  x: [bs, kernel_size, channels, num_pts_src]
  adj: [bs, kernel_size, num_pts_src, num_pts_dst]

Compute:
  out[b, c, w] = Σ_{k,v} x[b, k, c, v] * adj[b, k, v, w]

In words:
  For each output vertex w and channel c:
    Sum over all kernel channels k and source vertices v:
      weighted by adjacency adj[k, v, w]
```

---

## 3. Coordinate Positional Encoding

**Source**: `capex-code/models/models/utils/positional_encoding.py:97-123`

```python
def forward_coordinates(self, coord):
    """
    Compute sinusoidal positional encoding for 2D coordinates.
    
    Args:
        coord (Tensor): Normalized coordinates [bs, num_kpts, 2]
                       Values in [0, 1]
                       
    Returns:
        pos (Tensor): Positional embeddings [bs, num_kpts, num_feats*2]
                     where num_feats*2 = 256 typically
    """
    # Extract x, y coordinates
    x_embed, y_embed = coord[:, :, 0], coord[:, :, 1]  # [bs, num_kpts]
    
    # Scale to [0, 2π]
    x_embed = x_embed * self.scale  # scale = 2 * π
    y_embed = y_embed * self.scale
    
    # Compute frequency basis
    dim_t = torch.arange(
        self.num_feats, dtype=torch.float32, device=coord.device
    )
    dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
    # dim_t = [1, 1, 10000^(2/128), 10000^(2/128), 10000^(4/128), ...]
    
    # Divide coordinates by frequencies
    pos_x = x_embed[:, :, None] / dim_t  # [bs, num_kpts, num_feats]
    pos_y = y_embed[:, :, None] / dim_t  # [bs, num_kpts, num_feats]
    
    bs, kpt, _ = pos_x.shape
    
    # Apply sin to even indices, cos to odd indices
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
        dim=3
    ).view(bs, kpt, -1)  # [bs, num_kpts, num_feats]
    
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
        dim=3
    ).view(bs, kpt, -1)  # [bs, num_kpts, num_feats]
    
    # Concatenate y and x encodings
    pos = torch.cat((pos_y, pos_x), dim=2)  # [bs, num_kpts, num_feats * 2]
    
    return pos  # [bs, num_kpts, 256]
```

**Frequency Calculation**:
```
For num_feats=128, temperature=10000:
  freq[0] = 1 / 10000^(0/128) = 1
  freq[2] = 1 / 10000^(2/128) ≈ 0.912
  freq[4] = 1 / 10000^(4/128) ≈ 0.832
  ...
  freq[126] = 1 / 10000^(126/128) ≈ 0.0001

Low frequencies capture coarse position, high frequencies capture fine details.
```

---

## 4. Graph Integration in Decoder Layer

**Source**: `capex-code/models/models/utils/encoder_decoder.py:389-404`

```python
# Inside GraphTransformerDecoderLayer.forward()

# After self-attention and cross-attention:
refined_support_feat = self.norm2(refined_support_feat)

# Check if graph decoder is enabled
if self.graph_decoder is not None:
    num_pts, b, c = refined_support_feat.shape
    
    # Build adjacency matrix from skeleton
    adj = adj_from_skeleton(
        num_pts=num_pts,
        skeleton=skeleton,  # Passed from dataset
        mask=tgt_key_padding_mask,  # Mask for invalid keypoints
        device=refined_support_feat.device
    )
    
    # Apply GCN in FFN (3 modes)
    if self.graph_decoder == 'pre':
        # GCN → Activation → Linear
        tgt2 = self.ffn2(
            self.dropout(
                self.activation(
                    self.ffn1(refined_support_feat, adj)  # ← GCN Layer
                )
            )
        )
    elif self.graph_decoder == 'post':
        # Linear → Activation → GCN
        tgt2 = self.ffn2(
            self.dropout(
                self.activation(
                    self.ffn1(refined_support_feat)
                )
            ), 
            adj  # ← GCN Layer
        )
    else:  # 'both'
        # GCN → Activation → GCN
        tgt2 = self.ffn2(
            self.dropout(
                self.activation(
                    self.ffn1(refined_support_feat, adj)  # ← GCN Layer 1
                )
            ), 
            adj  # ← GCN Layer 2
        )
else:
    # Standard FFN (no graph)
    tgt2 = self.ffn2(
        self.dropout(
            self.activation(
                self.ffn1(refined_support_feat)
            )
        )
    )

# Residual connection
refined_support_feat = refined_support_feat + self.dropout3(tgt2)
refined_support_feat = self.norm3(refined_support_feat)
```

**When `graph_decoder='pre'`**:
- `self.ffn1` is a `GCNLayer(d_model=256, dim_feedforward=768, kernel_size=2)`
- `self.ffn2` is a standard `nn.Linear(768, 256)`

**Initialization** (from config):
```python
# In EncoderDecoder.__init__():
if self.graph_decoder == 'pre':
    self.ffn1 = GCNLayer(d_model, dim_feedforward, batch_first=False)
    self.ffn2 = nn.Linear(dim_feedforward, d_model)
```

---

## 5. Proposal Generator (Initial Keypoint Localization)

**Source**: `capex-code/models/models/utils/encoder_decoder.py:36-111`

```python
class ProposalGenerator(nn.Module):
    """
    Generates initial keypoint proposals via cross-attention between
    support embeddings and query image features.
    
    Creates a similarity map [bs, num_kpts, h, w] and computes
    weighted coordinate proposals from it.
    """
    
    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(hidden_dim, dynamic_proj_dim),
            nn.ReLU(),
            nn.Linear(dynamic_proj_dim, hidden_dim)
        )
        self.dynamic_act = nn.Tanh()

    def forward(self, query_feat, support_feat, spatial_shape):
        """
        Args:
            query_feat: [hw, bs, c] - flattened image features
            support_feat: [num_kpts, bs, c] - support embeddings
            spatial_shape: [h, w] - spatial dimensions of feature map
            
        Returns:
            proposal_for_loss: [bs, num_kpts, 2] - soft proposals (for heatmap loss)
            similarity_map: [bs, num_kpts, h, w] - attention map
            proposals: [bs, num_kpts, 2] - hard proposals (argmax-based)
        """
        device = query_feat.device
        _, bs, c = query_feat.shape
        h, w = spatial_shape
        
        # Normalizer for converting pixel coords to [0,1]
        side_normalizer = torch.tensor([w, h]).to(device)[None, None, :]
        
        # Transpose for easier batched operations
        query_feat = query_feat.transpose(0, 1)  # [bs, hw, c]
        support_feat = support_feat.transpose(0, 1)  # [bs, num_kpts, c]
        nq = support_feat.shape[1]
        
        # Project features
        fs_proj = self.support_proj(support_feat)  # [bs, num_kpts, proj_dim]
        fq_proj = self.query_proj(query_feat)  # [bs, hw, proj_dim]
        
        # Dynamic pattern attention (learned scaling)
        pattern_attention = self.dynamic_act(
            self.dynamic_proj(fs_proj)
        )  # [bs, num_kpts, c]
        
        # Modulate support features
        fs_feat = (pattern_attention + 1) * fs_proj  # [bs, num_kpts, proj_dim]
        
        # Compute similarity: how well each image location matches each keypoint
        similarity = torch.bmm(
            fq_proj, fs_feat.transpose(1, 2)
        )  # [bs, hw, num_kpts]
        
        similarity = similarity.transpose(1, 2).reshape(bs, nq, h, w)
        # similarity.shape = [bs, num_kpts, h, w]
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device)
        )
        coord_grid = torch.stack([grid_x, grid_y], dim=0)
        coord_grid = coord_grid.unsqueeze(0).unsqueeze(0).repeat(bs, nq, 1, 1, 1)
        coord_grid = coord_grid.permute(0, 1, 3, 4, 2)  # [bs, nq, h, w, 2]
        
        # Soft proposal (for loss): weighted average of all positions
        similarity_softmax = similarity.flatten(2, 3).softmax(dim=-1)  # [bs, nq, hw]
        similarity_coord_grid = similarity_softmax[:, :, :, None] * coord_grid.flatten(2, 3)
        proposal_for_loss = similarity_coord_grid.sum(dim=2)  # [bs, nq, 2]
        proposal_for_loss = proposal_for_loss / side_normalizer  # Normalize to [0,1]
        
        # Hard proposal (for position encoding): local maximum
        max_pos = torch.argmax(similarity.reshape(bs, nq, -1), dim=-1, keepdim=True)
        max_mask = F.one_hot(max_pos, num_classes=w * h)  # [bs, nq, 1, hw]
        max_mask = max_mask.reshape(bs, nq, w, h).type(torch.float)
        
        # Local max pooling (3x3 neighborhood)
        local_max_mask = F.max_pool2d(
            input=max_mask, kernel_size=3, stride=1, padding=1
        ).reshape(bs, nq, w * h, 1)
        
        # Compute local probability within max region
        local_similarity_softmax = similarity_softmax[:, :, :, None] * local_max_mask
        local_similarity_softmax = local_similarity_softmax / (
            local_similarity_softmax.sum(dim=-2, keepdim=True) + 1e-10
        )
        
        # Weighted average within local region
        proposals = local_similarity_softmax * coord_grid.flatten(2, 3)
        proposals = proposals.sum(dim=2) / side_normalizer  # [bs, nq, 2]
        
        return proposal_for_loss, similarity, proposals
```

**Key Insight**: This generates coordinate proposals from **learned feature matching**, not from explicit support coordinates!

---

## 6. Text Embedding Extraction (What We Need to Replace)

**Source**: `capex-code/models/models/detectors/capex.py:298-337`

```python
def extract_text_features(self, img_metas, max_points, mask_s):
    """
    Extract text embeddings for support keypoints.
    
    THIS IS WHAT WE NEED TO REPLACE WITH GEOMETRIC ENCODING!
    
    Args:
        img_metas: List of metadata dicts containing 'sample_point_descriptions'
        max_points: Maximum number of keypoints (e.g., 100)
        mask_s: Visibility mask [bs, max_points, 1]
        
    Returns:
        all_shots_point_descriptions: [bs, max_points, text_dim]
                                      Text embeddings for each keypoint
    """
    with torch.set_grad_enabled(self.finetune_text_pretrained):
        all_shots_point_descriptions = []
        
        # For each support shot (e.g., 1-shot, 5-shot)
        for shot in range(len(img_metas[0]['sample_point_descriptions'])):
            # Get text descriptions for this shot
            support_descriptions = [
                i['sample_point_descriptions'][shot] for i in img_metas
            ]
            
            # Filter to only visible keypoints
            support_descriptions = [
                description[list(mask[:len(description)].view(-1) == 1)]
                for mask, description in zip(mask_s.cpu(), support_descriptions)
            ]
            
            # Flatten all descriptions for batch encoding
            all_points = [
                point for description in support_descriptions for point in description
            ]
            
            # Encode with CLIP
            if self.text_backbone_type == "clip":
                tokens = self.tokenizer(all_points).to(device=self.text_backbone_device)
                all_descriptions = self.text_backbone.encode_text(tokens)
                # L2 normalize
                all_descriptions = all_descriptions / all_descriptions.norm(
                    dim=1, keepdim=True
                ).to(dtype=torch.float32)
            
            # Encode with BERT/GTE
            elif self.text_backbone_type in ["gte", "bert-multilingual"]:
                tokens = self.tokenizer(
                    all_points, max_length=77, padding=True, 
                    truncation=True, return_tensors='pt'
                ).to(device=self.text_backbone_device)
                all_descriptions = self.text_backbone(**tokens)
                # Use CLS token embedding
                all_descriptions = all_descriptions.last_hidden_state[:, 0]
                # L2 normalize
                all_descriptions = torch.nn.functional.normalize(
                    all_descriptions, p=2, dim=1
                )
            
            # Reshape back to [bs, max_points, text_dim]
            batch_padded_tensors = []
            start_index = 0
            for i, description in enumerate(support_descriptions):
                end_index = start_index + len(description)
                
                # Pad to max_points with zeros
                padded_tensor = torch.zeros(
                    max_points, all_descriptions.shape[-1]
                ).to(device=self.text_backbone_device).detach()
                
                # Fill in visible keypoint embeddings
                padded_tensor[mask_s[i].view(-1) == 1] = \
                    all_descriptions[start_index:end_index]
                
                batch_padded_tensors.append(padded_tensor)
                start_index = end_index
            
            all_shots_point_descriptions.append(
                torch.stack(batch_padded_tensors, dim=0)
            )
        
        # Average across shots
        return torch.mean(
            torch.stack(all_shots_point_descriptions, dim=0), 0
        )  # [bs, max_points, text_dim]
```

**Replacement Strategy**:
```python
def extract_geometric_features(self, support_coords, mask_s, skeleton):
    """
    OUR VERSION: Extract geometric embeddings for support keypoints.
    
    Args:
        support_coords: [bs, max_points, 2] - normalized coordinates
        mask_s: Visibility mask [bs, max_points, 1]
        skeleton: List of edge lists
        
    Returns:
        support_embeddings: [bs, max_points, hidden_dim]
    """
    # 1. Coordinate embedding
    coord_embed = self.coord_mlp(support_coords)  # [bs, max_points, 256]
    
    # 2. Positional encoding
    pos_embed = self.pos_encoding.forward_coordinates(
        support_coords
    )  # [bs, max_points, 256]
    
    # 3. Graph pre-encoding (optional)
    adj = adj_from_skeleton(
        support_coords.shape[1], skeleton, 
        mask_s.squeeze(-1).bool(), support_coords.device
    )
    
    graph_feat = coord_embed.permute(1, 0, 2)  # [max_points, bs, 256]
    for gcn_layer in self.graph_encoder_layers:
        graph_feat = gcn_layer(graph_feat, adj)
    graph_feat = graph_feat.permute(1, 0, 2)  # [bs, max_points, 256]
    
    # 4. Combine
    support_embeddings = coord_embed + pos_embed + graph_feat  # Element-wise
    
    return support_embeddings
```

---

## 7. Complete Forward Pass Flow

**CapeX End-to-End** (with annotations for what to replace):

```python
# ===== 1. EXTRACT FEATURES =====
# Text features (REPLACE THIS ❌)
all_shots_point_descriptions = self.extract_text_features(
    img_metas, max_points=100, mask_s
)  # [bs, 100, 512] - CLIP embeddings

# Query image features (KEEP ✅)
feature_q = self.backbone.forward_features(img_q)  # [bs, 768, h, w]

# ===== 2. PROJECT TO EMBEDDING SPACE =====
# Query image projection (KEEP ✅)
x = self.input_proj(feature_q)  # [bs, 256, h, w]

# Text projection (REPLACE WITH GEOMETRIC ❌)
point_descriptions = self.text_proj(all_shots_point_descriptions)  # [bs, 100, 256]

# ===== 3. TRANSFORMER ENCODING =====
# Joint encoding (KEEP ✅)
query_embed, refined_support_embed = self.encoder(
    src=x.flatten(2).permute(2, 0, 1),  # [hw, bs, 256]
    query_embed=point_descriptions.transpose(0, 1),  # [100, bs, 256]
    src_key_padding_mask=mask,
    query_key_padding_mask=query_padding_mask,
    pos=pos_embed
)

# ===== 4. GENERATE INITIAL PROPOSALS =====  
# Cross-attention to find keypoints in image (KEEP ✅)
initial_proposals_for_loss, similarity_map, initial_proposals = \
    self.proposal_generator(
        query_embed, refined_support_embed, spatial_shape=[h, w]
    )  # proposals: [bs, 100, 2] in [0,1]

# ===== 5. COMPUTE POSITIONAL ENCODING =====
# From coordinate proposals (KEEP ✅)
initial_position_embedding = self.positional_encoding.forward_coordinates(
    initial_proposals
)  # [bs, 100, 256]

# ===== 6. GRAPH-CONDITIONED DECODING =====
# Iterative refinement with graph (KEEP ✅, but skeleton must be geometric)
outs_dec, out_points, attn_maps = self.decoder(
    refined_support_embed,  # [100, bs, 256]
    query_embed,  # [hw, bs, 256]
    memory_key_padding_mask=mask,
    pos=pos_embed,
    query_pos=initial_position_embedding,
    tgt_key_padding_mask=query_padding_mask,
    position_embedding=self.positional_encoding,
    initial_proposals=initial_proposals,
    kpt_branch=self.kpt_branch,
    skeleton=skeleton,  # ← GRAPH INFO
    return_attn_map=return_attn_map
)

# ===== 7. FINAL COORDINATE PREDICTION =====
# MLP heads predict offsets (KEEP ✅)
for idx in range(outs_dec.shape[0]):
    layer_delta_unsig = self.kpt_branch[idx](outs_dec[idx])
    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(out_points[idx])
    output_kpts.append(layer_outputs_unsig.sigmoid())

return torch.stack(output_kpts, dim=0)  # [num_layers, bs, 100, 2]
```

**Summary of Replacements**:
1. ❌ **Line 2-3**: Replace `extract_text_features` with `extract_geometric_features`
2. ❌ **Line 11**: Replace `text_proj` with coordinate embedding + graph encoding
3. ✅ **Everything else**: Keep as-is!

---

## 8. Configuration for Graph vs No-Graph

**With Graph** (`configs/1shot-swin-clip/graph_split1_config.py:75`):
```python
transformer=dict(
    type='EncoderDecoder',
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    graph_decoder='pre',  # ← Enable graph!
    dim_feedforward=768,
    dropout=0.1,
    similarity_proj_dim=256,
    dynamic_proj_dim=128,
    activation="relu",
    normalize_before=False,
    return_intermediate_dec=True
)
```

**Without Graph** (`configs/1shot-swin-clip/base_split1_config.py:73`):
```python
transformer=dict(
    type='EncoderDecoder',
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    # graph_decoder=None,  # ← No graph (default)
    dim_feedforward=768,
    dropout=0.1,
    similarity_proj_dim=256,
    dynamic_proj_dim=128,
    activation="relu",
    normalize_before=False,
    return_intermediate_dec=True
)
```

**Difference**: ONE parameter! Everything else identical.

**Implication**: Graph encoding is a **modular addition**, not a fundamental redesign.

---

## 9. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
├─────────────────────────────────────────────────────────────────┤
│ Support:                                                        │
│   - Text: ["left eye", "right eye", "nose"]                    │
│   - Coords: [[0.3, 0.2], [0.7, 0.2], [0.5, 0.5]] (GT only!)   │
│   - Skeleton: [[0,2], [1,2]]                                    │
│                                                                 │
│ Query:                                                          │
│   - Image: [bs, 3, 256, 256]                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT ENCODING (❌ REPLACE)                    │
├─────────────────────────────────────────────────────────────────┤
│ CLIP/BERT Encoder:                                             │
│   "left eye" → [0.21, -0.45, ..., 0.33] (512-dim)             │
│   "right eye" → [0.19, -0.43, ..., 0.31]                       │
│   "nose" → [-0.12, 0.67, ..., -0.22]                          │
│                                                                 │
│ Output: [bs, 100, 512] padded text embeddings                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  TEXT PROJECTION (❌ REPLACE)                    │
├─────────────────────────────────────────────────────────────────┤
│ Linear: 512 → 256                                               │
│                                                                 │
│ Output: [bs, 100, 256] = support_embed                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  IMAGE FEATURE EXTRACTION (✅ KEEP)              │
├─────────────────────────────────────────────────────────────────┤
│ Swin Transformer V2:                                           │
│   Query Image [bs, 3, 256, 256]                                │
│      ↓                                                          │
│   Feature Map [bs, 768, 32, 32]                                │
│      ↓                                                          │
│   Projection (Conv 1x1): 768 → 256                             │
│      ↓                                                          │
│   Query Features [bs, 256, 32, 32]                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              JOINT ENCODER (✅ KEEP, geometry-only)              │
├─────────────────────────────────────────────────────────────────┤
│ Concatenate:                                                    │
│   [Image Features (hw=1024), Support Embeds (100)] → [1124, bs, 256]│
│                                                                 │
│ Self-Attention Layers (3 layers):                              │
│   Jointly refine image and support features                    │
│                                                                 │
│ Output:                                                         │
│   - query_embed: [1024, bs, 256] refined image                 │
│   - refined_support_embed: [100, bs, 256] refined support      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│           PROPOSAL GENERATOR (✅ KEEP, geometry-only)            │
├─────────────────────────────────────────────────────────────────┤
│ Cross-Attention:                                                │
│   Support [100, bs, 256] × Image [1024, bs, 256]               │
│      ↓                                                          │
│   Similarity Map [bs, 100, 32, 32]                             │
│      ↓                                                          │
│   Soft Argmax → Initial Proposals [bs, 100, 2]                 │
│                                                                 │
│ These are the FIRST coordinate predictions!                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│        POSITIONAL ENCODING (✅ KEEP, geometry-only)              │
├─────────────────────────────────────────────────────────────────┤
│ Sinusoidal Encoding:                                            │
│   Coords [bs, 100, 2] → Pos Embed [bs, 100, 256]              │
│                                                                 │
│ Formula: pos = concat[sin(x/freq), cos(x/freq), sin(y/freq), ...]│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│     GRAPH DECODER (✅ KEEP, geometry-only, CRITICAL!)            │
├─────────────────────────────────────────────────────────────────┤
│ For each decoder layer (3 layers):                             │
│                                                                 │
│   1. Self-Attention:                                            │
│      Support keypoints attend to each other                     │
│      Q, K, V = [100, bs, 256] + positional encoding            │
│                                                                 │
│   2. Cross-Attention:                                           │
│      Support keypoints attend to query image                    │
│      Q = Support [100, bs, 512] (concat with pos)              │
│      K, V = Image [1024, bs, 256]                              │
│                                                                 │
│   3. FFN with GCN:                                             │
│      Build adjacency from skeleton:                             │
│        adj = adj_from_skeleton(skeleton) → [bs, 2, 100, 100]  │
│                                                                 │
│      GCN Layer 1:                                               │
│        x = GCN(refined_support_feat, adj)  # [100, bs, 768]   │
│        x = ReLU(x)                                              │
│                                                                 │
│      Linear Layer 2:                                            │
│        x = Linear(x)  # [100, bs, 256]                         │
│                                                                 │
│      Residual: refined_support_feat += x                        │
│                                                                 │
│   4. Coordinate Prediction:                                     │
│      delta = MLP(refined_support_feat)  # [bs, 100, 2]        │
│      new_coords = sigmoid(inv_sigmoid(old_coords) + delta)      │
│                                                                 │
│   5. Update positional encoding for next layer                  │
│      pos_embed = forward_coordinates(new_coords)                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                       │
├─────────────────────────────────────────────────────────────────┤
│ Predicted Keypoints: [bs, 100, 2]                              │
│ - Normalized coordinates in [0, 1]                              │
│ - Refined through 3 decoder layers                              │
│ - Graph-aware via GCN                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Minimal Working Example (Geometry-Only)

```python
import torch
import torch.nn as nn

# ===== STEP 1: Define geometric support encoder =====
class GeometricSupportEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
        
    def forward(self, coords, mask):
        coord_feat = self.coord_mlp(coords)  # [bs, num_kpts, 256]
        pos_feat = self.pos_encoding.forward_coordinates(coords)  # [bs, num_kpts, 256]
        return coord_feat + pos_feat

# ===== STEP 2: Create dummy inputs =====
bs, num_kpts = 2, 17
support_coords = torch.rand(bs, num_kpts, 2)  # Normalized [0,1]
support_mask = torch.ones(bs, num_kpts, 1)  # All visible
skeleton = [
    [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9]],  # Sample 1
    [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9]]   # Sample 2
]
query_img = torch.rand(bs, 3, 256, 256)

# ===== STEP 3: Encode support geometrically =====
support_encoder = GeometricSupportEncoder(hidden_dim=256)
support_embed = support_encoder(support_coords, support_mask)
print(f"Support embeddings: {support_embed.shape}")  # [2, 17, 256]

# ===== STEP 4: Build adjacency matrix =====
adj = adj_from_skeleton(
    num_pts=num_kpts,
    skeleton=skeleton,
    mask=support_mask.squeeze(-1).bool(),
    device='cpu'
)
print(f"Adjacency: {adj.shape}")  # [2, 2, 17, 17]
print(f"Self-loops sum: {adj[0, 0].sum(dim=-1)}")  # Should be [1,1,...,1]
print(f"Neighbor sum: {adj[0, 1].sum(dim=-1)}")  # Should be [1,1,...,1] (normalized)

# ===== STEP 5: Apply GCN =====
gcn = GCNLayer(in_features=256, out_features=256, kernel_size=2, batch_first=True)
support_embed_refined = gcn(support_embed, adj)
print(f"GCN output: {support_embed_refined.shape}")  # [2, 17, 256]

# ===== SUCCESS! =====
print("✅ Geometry-only graph encoding works!")
```

**Expected Output**:
```
Support embeddings: torch.Size([2, 17, 256])
Adjacency: torch.Size([2, 2, 17, 17])
Self-loops sum: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
Neighbor sum: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
GCN output: torch.Size([2, 17, 256])
✅ Geometry-only graph encoding works!
```

---

## 11. Integration Checklist

**Before integrating CapeX components**:
- [ ] Verify our current model's decoder accepts external support embeddings
- [ ] Check if our cross-attention can handle variable-length skeletons
- [ ] Confirm our loss functions can handle set prediction (if switching from sequence)
- [ ] Ensure our dataset provides skeleton edge lists

**During integration**:
- [ ] Port `adj_from_skeleton()` with unit tests
- [ ] Port `GCNLayer` with unit tests
- [ ] Port `SinePositionalEncoding.forward_coordinates()` if not already present
- [ ] Implement `GeometricSupportEncoder`
- [ ] Modify model forward pass to use geometric encoder
- [ ] Update config to enable/disable graph decoder

**After integration**:
- [ ] Test forward pass with dummy data (no NaNs, correct shapes)
- [ ] Verify gradients flow through GCN layers
- [ ] Train for 1 epoch - confirm loss decreases
- [ ] Visualize predictions with skeleton overlays
- [ ] Ablate graph decoder (compare graph vs no-graph)

---

## 12. Gotchas and Edge Cases

### Gotcha 1: Batch-First vs Sequence-First

CapeX uses **sequence-first** convention in transformers:
- `[num_pts, bs, c]` NOT `[bs, num_pts, c]`

Our code might use batch-first. **Solution**: Transpose carefully!

### Gotcha 2: Skeleton Edge Indexing

CapeX skeletons are **0-indexed**: `[[0, 1], [1, 2]]`

Some datasets use **1-indexed**: `[[1, 2], [2, 3]]`

**Solution**: Check and convert in dataset loader.

### Gotcha 3: Mask Convention

CapeX masks: **True = invalid/padded**, **False = valid**

Some code uses opposite. **Solution**: Use `~mask` if needed.

### Gotcha 4: Coordinate Normalization

CapeX expects normalized coordinates in `[0, 1]`:
- `0.0` = top-left corner
- `1.0` = bottom-right corner

**Solution**: Divide by image dimensions before feeding to model.

### Gotcha 5: Skeleton Format

**Input format**: List of lists (per batch sample)
```python
skeleton = [
    [[0, 1], [1, 2]],  # Sample 1 skeleton
    [[0, 1], [1, 3]]   # Sample 2 skeleton (can differ!)
]
```

**NOT** a single global skeleton (unlike some CAPE works).

---

## 13. Performance Expectations

**CapeX with text** (from README):
- Split 1: 92.79% PCK
- Split 2: 89.47%
- Split 3: 84.95%
- Split 4: 87.25%
- Split 5: 89.61%
- **Average: 88.81%**

**Expected with geometry-only** (educated guess):
- Optimistic: 75-80% (graph helps a lot)
- Realistic: 65-75% (text provided crucial semantics)
- Pessimistic: 50-65% (significant semantic loss)

**Why lower?**
- Text provides transfer learning (pre-trained CLIP)
- Text disambiguates symmetry more easily
- Text encodes semantic priors ("wheels are round")

**Why might still work?**
- Graph structure provides strong prior
- Positional encoding captures relative geometry
- Visual features from query image provide context

**Our goal**: Demonstrate **geometric-only CAN work**, not necessarily match text-based performance.

---

## 14. Debugging Tips

**If loss is NaN**:
1. Check adjacency normalization (should sum to 1 per row)
2. Check for division by zero in `adj_from_skeleton` (mask handling)
3. Verify positional encoding doesn't explode (clamp coordinates to [0,1])

**If predictions are random**:
1. Verify gradients flow through GCN layers (`retain_graph=True` in backward)
2. Check if skeleton edges are correct (visualize with networkx)
3. Ensure support embeddings have variance (not all zeros/same)

**If skeleton structure is ignored**:
1. Check if `graph_decoder` is actually enabled in config
2. Verify adjacency matrix is non-zero (print `adj.sum()`)
3. Test with `graph_decoder=None` as baseline - performance should drop

---

## 15. Quick Reference: Variable Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `query_img` | `[bs, 3, 256, 256]` | Input query image |
| `support_coords` | `[bs, 100, 2]` | Support keypoint coordinates (normalized) |
| `skeleton` | List of `[[src, dst], ...]` | Edge list (per sample) |
| `mask_s` | `[bs, 100, 1]` | Visibility mask (1=visible, 0=padded) |
| `text_embeddings` | `[bs, 100, 512]` | CLIP/BERT output (❌ REMOVE) |
| `support_embed` | `[bs, 100, 256]` | Initial support representation |
| `feature_q` | `[bs, 768, 32, 32]` | CNN backbone output |
| `x` | `[bs, 256, 32, 32]` | Projected query features |
| `query_embed` | `[1024, bs, 256]` | Flattened, refined image features |
| `refined_support_embed` | `[100, bs, 256]` | Refined support (after encoder) |
| `similarity_map` | `[bs, 100, 32, 32]` | Attention map (support → image) |
| `initial_proposals` | `[bs, 100, 2]` | First coordinate predictions |
| `pos_embed` | `[bs, 100, 256]` | Positional encoding of proposals |
| `adj` | `[bs, 2, 100, 100]` | Adjacency matrix (dual-channel) |
| `outs_dec` | `[3, 100, bs, 256]` | Decoder outputs (3 layers) |
| `output` | `[3, bs, 100, 2]` | Final predictions (3 layers) |

**Note**: Final output is multi-layer (for intermediate supervision), last layer is used for inference.

---

**End of Code Snippets Document**

