# CAPE Training vs. Inference Input/Output Specification

**Status**: ✅ VERIFIED CORRECT (Comprehensive Audit - Nov 25, 2025)  
**Auditor**: System Audit & Verification  
**Purpose**: Definitive reference for CAPE training/testing input structure

---

## 🎯 Executive Summary

**VERDICT: Implementation is CORRECT** ✅

After comprehensive code audit of all data pipelines, model forwards, and evaluation paths:
- ✅ Training uses query GT keypoints (not support) for teacher forcing
- ✅ Support keypoints are conditioning-only (via cross-attention)
- ✅ Causal masking prevents future token leakage during training
- ✅ Inference is truly autoregressive (BOS → token-by-token generation)
- ✅ Query GT in inference is only used for metrics, never as model input

---

## 📋 Table of Contents

1. [Training Pipeline](#training-pipeline)
2. [Inference Pipeline](#inference-pipeline)
3. [Code Path Verification](#code-path-verification)
4. [Critical Design Principles](#critical-design-principles)
5. [Common Pitfalls (What NOT to Do)](#common-pitfalls)

---

## 1. Training Pipeline

### 1.1 Conceptual Flow

```
Episode (Category c ∈ C_train):
  ├── Support Image I_s (same category as query)
  │   ├── Keypoints V_s = [v₁ˢ, v₂ˢ, ..., vₙˢ]
  │   └── Skeleton G_c = [[edge₁], [edge₂], ...]
  │          ↓
  │    Support Encoder (E_s) with graph structure
  │          ↓
  │    Cross-Attention Context: support_features (B, N, D)
  │
  └── Query Image I_q (same category, different instance)
      ├── Image → ResNet → F_q (image features)
      │                      ↓
      └── Keypoints V_q  Transformer Decoder
            ↓                ├─ Input: V_q (teacher forcing)
       Tokenize              ├─ Image Context: F_q  
            ↓                ├─ Support Context: E_s (cross-attn)
       seq_data              └─ Causal Mask: Future blocked
            ↓                    ↓
     targets (GT)          Predictions V̂_q
            ↓                    ↓
  Decoder Input Sequence   Loss(V̂_q, V_q)
```

### 1.2 Input Structure

**For each training batch:**

| Component | Source | Shape | Description |
|-----------|--------|-------|-------------|
| `query_images` | Query instances | `(B*K, 3, H, W)` | RGB images to predict keypoints on |
| `query_targets` | **Query GT keypoints** | `(B*K, seq_len, 2)` | Ground truth sequence **from query**, NOT support |
| `support_coords` | Support instance | `(B*K, N, 2)` | Support keypoints for conditioning |
| `support_mask` | Support validity | `(B*K, N)` | Mask for valid support keypoints |
| `skeleton_edges` | Category structure | List of `B*K` edge lists | Graph connectivity |

**Key Point**: `query_targets` contains the tokenized keypoint sequence **of the query image itself**, not from any other image.

### 1.3 Code Verification

**Episode Construction** (`datasets/episodic_sampler.py:261-264`):
```python
for query_idx in episode['query_indices']:
    query_data = self.base_dataset[query_idx]  # Load QUERY image
    query_images.append(query_data['image'])
    query_targets.append(query_data['seq_data'])  # ← QUERY keypoints!
```
✅ **Verified**: Targets come from query images, not support.

**Tokenization** (`datasets/mp100_cape.py:450-454`):
```python
record["seq_data"] = self._tokenize_keypoints(
    keypoints=record["keypoints"],  # ← Keypoints from THIS image
    height=record["height"],
    width=record["width"],
    visibility=record.get("visibility")
)
```
✅ **Verified**: Each image's `seq_data` is tokenized from **its own** keypoints.

**Forward Pass** (`engine_cape.py:95-101`):
```python
outputs = model(
    samples=query_images,          # Query images
    support_coords=support_coords,  # Support for conditioning
    support_mask=support_masks,
    targets=query_targets,          # ← QUERY GT for teacher forcing!
    skeleton_edges=support_skeletons
)
```
✅ **Verified**: `targets` argument is query GT, not support.

**Model Processing** (`models/cape_model.py:195, 222`):
```python
# Step 1: Encode support (separate from decoder input)
support_features = self.support_encoder(support_coords, support_mask, skeleton_edges)

# Step 2: Forward with QUERY targets
outputs = self.base_model(samples, seq_kwargs=targets)  # targets = query GT!
```
✅ **Verified**: Support encoded separately; query GT used as `seq_kwargs`.

**Decoder Embedding** (`models/deformable_transformer_v2.py:1020-1023`):
```python
output = self._seq_embed(
    seq11=seq_kwargs['seq11'],      # From query GT!
    seq12=seq_kwargs['seq12'],
    seq21=seq_kwargs['seq21'],
    seq22=seq_kwargs['seq22'],
    ...
)
```
✅ **Verified**: Decoder embeds **query GT sequences**.

### 1.4 Causal Masking

**Mask Creation** (`models/deformable_transformer_v2.py:166-174`):
```python
def _create_causal_attention_mask(self, seq_len):
    """Upper triangular mask: future positions = -inf"""
    mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
    )
    causal_mask = mask.masked_fill(mask == 1, float('-inf'))
    return causal_mask
```

**Mask Structure** (example for seq_len=5):
```
[[  0, -inf, -inf, -inf, -inf],   # Token 0 sees only itself
 [  0,   0, -inf, -inf, -inf],   # Token 1 sees 0,1
 [  0,   0,   0, -inf, -inf],   # Token 2 sees 0,1,2
 [  0,   0,   0,   0, -inf],   # Token 3 sees 0,1,2,3
 [  0,   0,   0,   0,   0]]    # Token 4 sees all previous
```

**Applied During Forward** (`models/deformable_transformer_v2.py:236-241`):
```python
if tgt_masks is None:
    tgt_masks = self._create_causal_attention_mask(
        seq_kwargs['seq11'].shape[1]
    ).to(memory.device)
```
✅ **Verified**: Causal mask automatically applied, preventing future token leakage.

---

## 2. Inference Pipeline

### 2.1 Conceptual Flow

```
Episode (Category c ∈ C_test, UNSEEN):
  ├── Support Image I_s
  │   ├── Keypoints V_s = [v₁ˢ, v₂ˢ, ..., vₙˢ]
  │   └── Skeleton G_c_unseen
  │          ↓
  │    Support Encoder (E_s)
  │          ↓
  │    Cross-Attention Context
  │
  └── Query Image I_q_unseen
      ├── Image → ResNet → F_q
      │                     ↓
      └── GT V_q (stored) Autoregressive Decoder
            │                ├─ Start: BOS
            │                ├─ Image: F_q
            │                ├─ Support: E_s
            │                └─ Loop: v̂ᵢ → v̂ᵢ₊₁
            │                    ↓
            │              Predictions V̂_q
            │                    ↓
            └──────────────> PCK(V̂_q, V_q)
                             (Metric computation only)
```

### 2.2 Input Structure

**For each test batch:**

| Component | Source | Shape | Description |
|-----------|--------|-------|-------------|
| `query_images` | Unseen category | `(B*K, 3, H, W)` | RGB images from unseen category |
| `support_coords` | Different image, same unseen cat | `(B*K, N, 2)` | Support keypoints for conditioning |
| `support_mask` | Support validity | `(B*K, N)` | Mask for valid support keypoints |
| `skeleton_edges` | Unseen category structure | List of `B*K` edge lists | Graph connectivity |
| **NO** `query_targets` in forward pass | - | - | **Targets NOT passed to model** |

**Key Point**: Query GT keypoints (V_q) are **only loaded for metric computation**, never passed to the model's forward method.

### 2.3 Code Verification

**No Query GT Passed** (`engine_cape.py:532-544`):
```python
predictions = model.forward_inference(
    samples=query_images,          # Query image
    support_coords=support_coords,  # Support conditioning
    support_mask=support_masks,
    skeleton_edges=support_skeletons
    # ← NO `targets` argument!
)
```
✅ **Verified**: Query GT **NOT** passed to `forward_inference`.

**Autoregressive Loop** (`models/roomformer_v2.py:436-546`):
```python
# Initialize with BOS
(prev_output_token_11, prev_output_token_12, 
 prev_output_token_21, prev_output_token_22, ...) = self._prepare_sequences(bs)

i = 0
while i < max_len and unfinish_flag.any():
    # Current tokens only (position i)
    seq_kwargs = {
        'seq11': prev_output_tokens_11[:, i:i+1],
        'seq12': prev_output_tokens_12[:, i:i+1],
        ...
    }
    
    # Forward pass
    hs, _, reg_output, cls_output = self.transformer(...)
    
    # Sample next token
    cls_j = torch.argmax(cls_output, 2)[j, 0].item()
    
    if cls_j == TokenType.coord.value:
        # Decode coordinates and feed back
        output_j_x, output_j_y = reg_output[j, 0].detach().cpu().numpy()
        gen_out[j].append([output_j_x, output_j_y])
        
        # Tokenize for next step
        prev_output_token_11[j].append(...)  # Feed prediction back
        ...
    elif cls_j == TokenType.eos.value:
        unfinish_flag[j] = 0  # Stop generation
    
    i += 1
```
✅ **Verified**: True autoregressive: BOS → token₁ → token₂ → ... → EOS

**GT Used for Metrics Only** (`engine_cape.py:560-575`):
```python
pred_coords = predictions.get('coordinates')  # Model output
gt_coords = query_targets.get('target_seq')  # ← Loaded separately!

# Compute PCK
pck_evaluator.add_batch(
    pred_keypoints=pred_kpts,  # From model
    gt_keypoints=gt_kpts,      # From stored GT (not used in forward!)
    ...
)
```
✅ **Verified**: GT loaded separately, used **only** for PCK metric.

---

## 3. Code Path Verification

### 3.1 Support is Conditioning Only

**Proof 1: Support Never in Decoder Input**

Search all model forward calls for `seq_kwargs`:
- `seq_kwargs` always comes from `targets` (query GT)
- Never from `support_coords`

**File**: `models/cape_model.py:222`
```python
outputs = self.base_model(samples, seq_kwargs=targets)  # targets ≠ support!
```

**Proof 2: Support Encoded Separately**

**File**: `models/cape_model.py:195-212`
```python
# Support goes through separate encoder
support_features = self.support_encoder(support_coords, support_mask, skeleton_edges)

# Injected into decoder for cross-attention (not as input sequence)
self.base_model.transformer.decoder.support_features = support_features
self.base_model.transformer.decoder.support_mask = support_mask
```

**Proof 3: Decoder Uses Support via Cross-Attention**

**File**: `models/deformable_transformer_v2.py:289-293`
```python
# In TransformerDecoderLayer:
self.support_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
self.dropout_support = nn.Dropout(dropout)
self.norm_support = nn.LayerNorm(d_model)
```

The decoder has dedicated cross-attention modules for support, separate from the self-attention on the token sequence.

✅ **Verified**: Support used for **cross-attention conditioning**, NOT as decoder input sequence.

### 3.2 Training/Inference Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Query GT provided?** | ✅ Yes (as `targets`) | ❌ No |
| **Decoder input** | Full GT sequence V_q | Auto-generated: BOS → v̂₁ → v̂₂ → ... |
| **Attention mask** | Causal (future blocked) | Causal (future blocked) |
| **Support usage** | Cross-attention conditioning | Cross-attention conditioning |
| **Generation mode** | Teacher forcing | Autoregressive |
| **GT usage** | Training target (loss) | Metric computation only |

---

## 4. Critical Design Principles

### 4.1 Teacher Forcing During Training

**What it means:**
- Model receives the **entire ground truth sequence** as input
- For each position t, model predicts token t
- Causal mask ensures position t only sees tokens < t
- Model learns: `p(v_t | v_{<t}, I_q, G_c, V_s)`

**Why it's correct:**
- Standard practice for autoregressive models
- Faster training (parallel processing across sequence)
- Stable gradients

**Code location:**
- `engine_cape.py:95-101` - Full `query_targets` passed to model
- `models/deformable_transformer_v2.py:236-241` - Causal mask applied

### 4.2 Autoregressive Generation During Inference

**What it means:**
- Model starts with BOS token
- Generates one token at a time
- Each prediction fed back as input for next step
- Stops at EOS or max length

**Why it's correct:**
- Matches real-world usage (no GT available)
- Tests true generalization capability
- Same distribution as deployment

**Code location:**
- `models/roomformer_v2.py:436-546` - Autoregressive loop
- `models/cape_model.py:230-371` - `forward_inference` wrapper

### 4.3 Support as Conditioning

**What it means:**
- Support keypoints V_s encoded into embeddings
- Decoder cross-attends to support embeddings
- Support provides **structural context**, not the target

**Why it's correct:**
- Enables 1-shot learning from single example
- Support acts as "template" or "prior"
- Query prediction conditioned on support structure

**Code location:**
- `models/support_encoder.py` - Support graph encoding
- `models/cape_model.py:195` - Support encoding
- `models/deformable_transformer_v2.py:289-293` - Support cross-attention

---

## 5. Common Pitfalls (What NOT to Do)

### ❌ FORBIDDEN: Support as Decoder Target

**WRONG**:
```python
outputs = model(
    samples=query_images,
    targets=support_data['seq_data']  # ❌ WRONG! Should be query GT!
)
```

**CORRECT**:
```python
outputs = model(
    samples=query_images,
    targets=query_targets  # ✅ From query images!
)
```

### ❌ FORBIDDEN: Query GT in Inference Forward

**WRONG**:
```python
predictions = model.forward_inference(
    samples=query_images,
    targets=query_targets  # ❌ WRONG! No targets in inference!
)
```

**CORRECT**:
```python
predictions = model.forward_inference(
    samples=query_images,
    support_coords=support_coords  # ✅ Only support + image!
)
# Query GT loaded separately for metrics
```

### ❌ FORBIDDEN: No Causal Mask

**WRONG**:
```python
tgt_masks = None  # ❌ Future tokens visible!
```

**CORRECT**:
```python
tgt_masks = self._create_causal_attention_mask(seq_len)  # ✅ Causal!
```

---

## 6. Audit Trail

### Files Examined

**Data Pipeline:**
- ✅ `datasets/episodic_sampler.py` - Episode construction
- ✅ `datasets/mp100_cape.py` - Dataset loading and tokenization
- ✅ `datasets/token_types.py` - Token type definitions

**Model Architecture:**
- ✅ `models/cape_model.py` - CAPE wrapper with support conditioning
- ✅ `models/roomformer_v2.py` - Base Raster2Seq model
- ✅ `models/deformable_transformer_v2.py` - Transformer implementation
- ✅ `models/support_encoder.py` - Support graph encoder

**Training/Evaluation:**
- ✅ `engine_cape.py` - Training and evaluation loops
- ✅ `train_cape_episodic.py` - Training script
- ✅ `util/eval_utils.py` - PCK metric computation

### Verification Methods

1. **Code Reading**: Line-by-line examination of all critical paths
2. **Data Flow Tracing**: Followed tensors from dataset → model → loss
3. **Shape Verification**: Checked tensor dimensions at each stage
4. **Logic Verification**: Confirmed causal masking, autoregressive generation

### Conclusion

✅ **ALL CHECKS PASSED**

The CAPE implementation correctly follows the specified training/testing paradigm:
- Training uses query GT with teacher forcing and causal masking
- Inference is truly autoregressive without query GT in forward pass
- Support is used only for conditioning via cross-attention
- No architectural violations detected

---

## 7. Quick Reference

### Training Forward Call
```python
model(
    samples=query_images,       # (B*K, 3, H, W)
    support_coords=support,     # (B*K, N, 2) - conditioning
    support_mask=mask,          # (B*K, N)
    targets=query_targets,      # (B*K, seq_len, ...) - from QUERY
    skeleton_edges=skeletons    # List[B*K]
)
```

### Inference Forward Call
```python
model.forward_inference(
    samples=query_images,       # (B*K, 3, H, W)
    support_coords=support,     # (B*K, N, 2) - conditioning
    support_mask=mask,          # (B*K, N)
    skeleton_edges=skeletons    # List[B*K]
    # NO targets!
)
```

### Metric Computation
```python
# GT loaded separately (not passed to model)
gt_coords = query_targets.get('target_seq')
pred_coords = predictions.get('coordinates')

pck = compute_pck(pred_coords, gt_coords)
```

---

**Document Status**: ✅ COMPREHENSIVE AUDIT COMPLETE

All claims verified with actual code. Implementation is correct.

