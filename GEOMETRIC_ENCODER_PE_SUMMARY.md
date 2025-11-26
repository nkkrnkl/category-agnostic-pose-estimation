# Geometric Support Encoder: Positional Encoding Implementation

## Summary

**Good news:** The `GeometricSupportEncoder` **already implements exactly what you requested!** 

Both spatial (2D coordinate-based) and sequence (1D index-based) positional encodings are properly combined with the learned coordinate embeddings.

---

## Implementation Details

### File: `models/geometric_support_encoder.py`

The encoder combines **three complementary sources of information**:

### 1. **Coordinate MLP Embedding** (Lines 79-84)

```python
self.coord_mlp = nn.Sequential(
    nn.Linear(2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)
```

**Purpose:** Learn a transformation of raw (x, y) coordinates.

---

### 2. **2D Spatial Positional Encoding** (Lines 87-92)

```python
self.pos_encoding = SinePositionalEncoding2D(
    num_feats=hidden_dim // 2,  # Output: hidden_dim (num_feats * 2)
    temperature=10000,
    normalize=True,
    scale=2 * math.pi
)
```

**Purpose:** Encode WHERE the keypoint is in image space (x, y position).

**Implementation:** CapeX-style sinusoidal encoding:
- Input: `[bs, num_pts, 2]` coordinates in [0, 1]
- Output: `[bs, num_pts, hidden_dim]` spatial embeddings
- Uses sine/cosine at different frequencies for x and y

---

### 3. **1D Sequence Positional Encoding** (Lines 98-102)

```python
self.sequence_pos_encoding = PositionalEncoding1D(
    d_model=hidden_dim,
    max_len=100,  # Max keypoints
    dropout=0.0   # Deterministic
)
```

**Purpose:** Encode WHICH keypoint it is in the ordering (0th, 1st, 2nd, ...).

**Implementation:** Standard transformer sinusoidal PE:
- Input: `[bs, num_pts, hidden_dim]` embeddings
- Output: `[bs, num_pts, hidden_dim]` with sequence PE added
- Encodes index position (0, 1, 2, ..., N-1)

---

## Forward Pass Logic (Lines 166-179)

The three components are combined in the forward method:

```python
# 1. Coordinate embedding
coord_emb = self.coord_mlp(support_coords)  # [bs, num_pts, hidden_dim]

# 2. Spatial positional encoding (WHERE in image)
pos_emb = self.pos_encoding.forward_coordinates(support_coords)  # [bs, num_pts, hidden_dim]

# 3. Combine coordinate and spatial positional information
embeddings = coord_emb + pos_emb  # [bs, num_pts, hidden_dim]

# 4. Add sequence positional encoding (WHICH keypoint)
# This tells the transformer which position in the sequence (0th, 1st, 2nd, ...)
embeddings = self.sequence_pos_encoding(embeddings)  # [bs, num_pts, hidden_dim]
```

**Order of operations:**
1. MLP(coords) → learned embedding
2. SinePE2D(coords) → spatial encoding
3. Add them: `coord_emb + spatial_pe`
4. Add sequence PE: encodes ordering

**Result:** Each keypoint embedding contains information about:
- **What** the coordinates are (learned via MLP)
- **Where** in image space (x, y via spatial PE)
- **Which** position in sequence (0, 1, 2, ... via sequence PE)

---

## Verification Test Results

```
✓ Encoder created successfully
  GeometricSupportEncoder(hidden_dim=256, spatial_pe=SinePE2D, sequence_pe=SinePE1D)

✓ All positional encoding components present:
  - coord_mlp: Linear(2 → 256)
  - pos_encoding (spatial): SinePositionalEncoding2D(num_feats=128)
  - sequence_pos_encoding: PositionalEncoding1D(d_model=256, max_len=100)

✓ Forward pass successful:
  Output shape: torch.Size([2, 17, 256])

✓ Sequence PE verification:
  Mean difference after shuffling: 0.302307
  ✅ PASS: Shuffling changes output (sequence PE is working)
```

**Key test:** Shuffling keypoint order produces different embeddings, confirming sequence PE is active.

---

## Integration

### Model Usage (`models/cape_model.py`)

The geometric encoder is used when the flag `--use_geometric_encoder` is set:

```python
if use_geometric_encoder:
    self.support_encoder = GeometricSupportEncoder(
        hidden_dim=hidden_dim,
        num_encoder_layers=support_encoder_layers,
        use_gcn_preenc=use_gcn_preenc,
        num_gcn_layers=num_gcn_layers
    )
```

### Training Script (`models/train_cape_episodic.py`)

Enable via CLI:
```bash
python models/train_cape_episodic.py \
    --use_geometric_encoder \
    --use_gcn_preenc \
    --num_gcn_layers 2
```

**Your current training uses this!** Looking at your terminal output:
```
Support encoder: Geometric (CapeX-inspired)
  - GCN pre-encoding: Enabled
  - GCN layers: 2
```

This confirms you're already using the geometric encoder with both PE types.

---

## Alignment with PhD Recommendations

The implementation **perfectly matches** the PhD guidance:

> **PhD Recommendation:**
> "Support keypoints should combine both:
> 1. Coordinate-based spatial information, and
> 2. Sequence index–based positional information"

**Our Implementation:**
- ✅ **Coordinate-based spatial:** `SinePositionalEncoding2D` on (x, y) coords
- ✅ **Sequence index–based:** `PositionalEncoding1D` for ordering
- ✅ **Combination:** Both are added to learned coordinate embeddings

**Additional components beyond minimum requirement:**
- ✅ Learned coordinate MLP (provides flexibility)
- ✅ Optional GCN pre-encoding (uses skeleton structure)
- ✅ Transformer self-attention (contextual understanding)

---

## Shape Flow Diagram

```
Input: support_coords [bs, num_pts, 2]  (normalized to [0, 1])
   ↓
   ├─→ coord_mlp ────────→ coord_emb [bs, num_pts, hidden_dim]
   │                              ↓
   └─→ SinePE2D ─────────→ spatial_pe [bs, num_pts, hidden_dim]
                                  ↓
                            embeddings = coord_emb + spatial_pe
                                  ↓
                            PositionalEncoding1D (sequence PE)
                                  ↓
                            embeddings [bs, num_pts, hidden_dim]
                                  ↓
                            (Optional GCN layers)
                                  ↓
                            Transformer Encoder
                                  ↓
Output: support_features [bs, num_pts, hidden_dim]
```

---

## Code Reference

### `__init__` Method (Lines 65-132)

```python
def __init__(self,
             hidden_dim: int = 256,
             num_encoder_layers: int = 3,
             nhead: int = 8,
             dim_feedforward: int = 1024,
             dropout: float = 0.1,
             use_gcn_preenc: bool = False,
             num_gcn_layers: int = 2,
             activation: str = 'relu'):
    super().__init__()
    
    self.hidden_dim = hidden_dim
    self.use_gcn_preenc = use_gcn_preenc
    
    # 1. Coordinate embedding MLP
    self.coord_mlp = nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.ReLU() if activation == 'relu' else nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
    
    # 2. Sinusoidal positional encoding (spatial, from CapeX)
    self.pos_encoding = SinePositionalEncoding2D(
        num_feats=hidden_dim // 2,
        temperature=10000,
        normalize=True,
        scale=2 * math.pi
    )
    
    # 3. 1D sequence positional encoding (ordering)
    self.sequence_pos_encoding = PositionalEncoding1D(
        d_model=hidden_dim,
        max_len=100,
        dropout=0.0
    )
    
    # 4. Optional GCN pre-encoding
    if use_gcn_preenc:
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, ...)
            for _ in range(num_gcn_layers)
        ])
    
    # 5. Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True
    )
    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_encoder_layers
    )
```

### `forward` Method (Lines 134-200)

```python
def forward(self,
            support_coords: torch.Tensor,
            support_mask: torch.Tensor,
            skeleton_edges: List[List[List[int]]]) -> torch.Tensor:
    """
    Args:
        support_coords: [bs, num_pts, 2] in [0, 1]
        support_mask: [bs, num_pts] (True = invalid)
        skeleton_edges: List of edge lists
    
    Returns:
        support_features: [bs, num_pts, hidden_dim]
    """
    # 1. Coordinate embedding
    coord_emb = self.coord_mlp(support_coords)
    
    # 2. Spatial PE
    pos_emb = self.pos_encoding.forward_coordinates(support_coords)
    
    # 3. Combine
    embeddings = coord_emb + pos_emb
    
    # 4. Add sequence PE
    embeddings = self.sequence_pos_encoding(embeddings)
    
    # 5. Optional GCN
    if self.use_gcn_preenc and self.gcn_layers is not None:
        adj = adj_from_skeleton(num_pts, skeleton_edges, support_mask, device)
        for gcn_layer in self.gcn_layers:
            embeddings = gcn_layer(embeddings, adj)
    
    # 6. Transformer
    support_features = self.transformer_encoder(
        embeddings,
        src_key_padding_mask=support_mask
    )
    
    return support_features
```

---

## Conclusion

**No changes needed!** The `GeometricSupportEncoder` already implements:

✅ **Coordinate MLP embedding** - Learned transformation of (x, y)  
✅ **2D Spatial PE (CapeX-style)** - WHERE in image space  
✅ **1D Sequence PE** - WHICH keypoint in ordering  
✅ **Proper combination** - All three sources of information  
✅ **PhD-compliant** - Matches recommendations exactly  

Your current training (with `--use_geometric_encoder --use_gcn_preenc`) is already using this implementation.

The encoder gives the model maximum signal from support keypoints by combining:
- Learned coordinate features
- Explicit spatial positional encoding
- Explicit sequence positional encoding
- Optional graph structure (GCN)
- Contextual relationships (Transformer)

