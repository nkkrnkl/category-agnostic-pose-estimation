# Optional Improvements (Not Yet Implemented)

This document lists potential improvements identified during the codebase audit that are **NOT critical** for correctness but could enhance performance, robustness, or maintainability.

These are listed for future consideration and prioritization.

---

## 1. Semantic Keypoint Ordering

**Name**: Canonical keypoint ordering per category

**Files**: `datasets/mp100_cape.py`, possibly config files

**Problem**: Currently, keypoints follow the COCO annotation order, which may vary across images. For the same category, keypoint #0 might be "nose" in one image but "left_eye" in another (depending on annotation order).

**Proposed Solution**:
```python
# Define canonical ordering for each category
CANONICAL_KEYPOINT_ORDER = {
    'person': [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # COCO person order
    'cat': [0, 1, 2, 3, 4, 5, ...],
    # ... for each category
}

# In __getitem__, reorder keypoints and visibility
category_name = self.coco.loadCats(category_id)[0]['name']
if category_name in CANONICAL_KEYPOINT_ORDER:
    order = CANONICAL_KEYPOINT_ORDER[category_name]
    kpts_array = kpts_array[order]
    visibility = visibility[order]
    # Also need to reorder skeleton edges!
```

**Benefits**:
- More consistent semantic meaning across episodes
- Could improve cross-category generalization
- Easier to interpret model predictions

**Effort**: Medium (requires defining canonical orders for all 100 categories, testing)

**Impact**: Medium (could improve accuracy by 1-2% PCK, not verified)

---

## 2. Aspect-Ratio-Preserving Resize

**Name**: Resize to 512×512 while preserving aspect ratio

**Files**: `datasets/mp100_cape.py` (transform pipeline)

**Problem**: Current implementation resizes cropped bboxes to exactly 512×512, potentially distorting non-square objects.

**Proposed Solution**:
```python
# Instead of Resize((512, 512)):
# 1. Resize shorter side to 512
# 2. Pad to 512×512 with zeros
# 3. Update keypoint coordinates accordingly

from albumentations import LongestMaxSize, PadIfNeeded

transforms = A.Compose([
    LongestMaxSize(max_size=512),  # Resize longest edge to 512
    PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0),
    # ... other transforms
], keypoint_params=...)
```

**Benefits**:
- No aspect ratio distortion
- More natural object appearance

**Drawbacks**:
- Adds padding (could confuse model)
- Need to track padding masks
- Slightly more complex coordinate handling

**Effort**: Medium

**Impact**: Low-Medium (may improve PCK by 0.5-1%, especially for elongated objects)

---

## 3. Full Multi-Instance Support

**Name**: Handle all instances in multi-instance images

**Files**: `datasets/mp100_cape.py`, `datasets/episodic_sampler.py`

**Problem**: Currently only uses the first instance per image (see `MULTI_INSTANCE_LIMITATION.md`). ~10% of images have 2+ instances.

**Proposed Solution**:
```python
# In MP100CAPE:
# 1. Change __len__() to count total instances, not images
# 2. Map index → (image_id, instance_idx)
# 3. Store instance-based index list

def __len__(self):
    return len(self.instance_ids)  # List of (img_id, inst_idx) tuples

def __getitem__(self, idx):
    img_id, inst_idx = self.instance_ids[idx]
    # Load image and extract instance inst_idx
    # ...
```

**Benefits**:
- Use 100% of annotated data (currently ~90%)
- More diverse training examples

**Drawbacks**:
- Slightly more complex indexing
- Need to handle instance selection carefully in episodic sampling

**Effort**: Medium

**Impact**: Low-Medium (~10% more data could improve PCK by 0.5-1%)

---

## 4. Skeleton Edge Attention Bias

**Name**: Use skeleton edges as attention bias in decoder

**Files**: `models/deformable_transformer_v2.py`, `models/cape_model.py`

**Problem**: Currently, skeleton edges are used only in the support encoder. The decoder doesn't explicitly know which keypoints are connected.

**Proposed Solution**:
```python
# In decoder cross-attention:
# 1. Build adjacency matrix from skeleton edges
# 2. Use as attention bias (add to attention scores)
# 3. Encourage model to attend to connected keypoints

# In decoder layer:
attn_scores = Q @ K.T  # (B, seq_len, seq_len)
if skeleton_bias is not None:
    # skeleton_bias: (B, N_kpts, N_kpts) → expand to seq_len
    attn_scores = attn_scores + skeleton_bias
attn_weights = softmax(attn_scores)
```

**Benefits**:
- Model explicitly aware of skeleton structure during decoding
- Could improve keypoint relationship modeling
- May reduce erratic predictions (e.g., arm keypoint far from shoulder)

**Effort**: High (requires modifying decoder, careful integration with existing attention)

**Impact**: Medium (could improve structural consistency, ~1-2% PCK)

---

## 5. Uncertainty-Aware Predictions

**Name**: Predict uncertainty/confidence per keypoint

**Files**: `models/roomformer_v2.py`, `models/cape_losses.py`, `util/eval_utils.py`

**Problem**: Model doesn't quantify prediction uncertainty. Some keypoints may be more confident than others (e.g., visible vs occluded).

**Proposed Solution**:
```python
# Add uncertainty head to model
self.uncertainty_head = nn.Linear(hidden_dim, 1)

# In forward:
pred_uncertainty = self.uncertainty_head(hs)  # (B, seq_len, 1)

# In loss:
# Use uncertainty to weight loss (higher uncertainty → lower weight)
loss_coords = torch.exp(-pred_uncertainty) * l1_loss + pred_uncertainty
# (Encourages model to predict high uncertainty for hard keypoints)

# In evaluation:
# Use uncertainty for confidence thresholding or weighted PCK
```

**Benefits**:
- More calibrated predictions
- Can filter low-confidence predictions
- Better understanding of model behavior

**Effort**: High (requires loss redesign, careful tuning)

**Impact**: Medium (may improve PCK by providing better confidence estimates)

---

## 6. Category Embeddings

**Name**: Add category embeddings to support/query encoding

**Files**: `models/support_encoder.py`, `models/cape_model.py`

**Problem**: Model doesn't explicitly know which category it's working with. This could help with category-specific patterns.

**Proposed Solution**:
```python
# Add category embedding layer
self.category_embed = nn.Embedding(num_categories, hidden_dim)

# In forward:
category_emb = self.category_embed(category_id)  # (B, hidden_dim)
# Add to support embeddings or query embeddings
support_features = support_features + category_emb.unsqueeze(1)
```

**Benefits**:
- Model can learn category-specific priors
- May improve within-category predictions

**Drawbacks**:
- Could hurt generalization to unseen categories
- Need to handle unseen category IDs (use a default embedding?)

**Effort**: Low-Medium

**Impact**: Low (may help seen categories but hurt unseen, net unclear)

---

## 7. Keypoint-Level Attention Visualization

**Name**: Visualize which support keypoints the model attends to when predicting each query keypoint

**Files**: New file `util/visualization.py`, `models/cape_model.py`

**Problem**: Hard to debug/understand what the model is learning without attention visualization.

**Proposed Solution**:
```python
# In CAPEModel.forward, save attention weights
self.last_support_attention = support_cross_attn_weights

# In visualization:
def visualize_support_attention(query_img, support_img, query_kpt_idx, attn_weights):
    # Show which support keypoints are attended to for query keypoint query_kpt_idx
    # Overlay attention heatmap on support image
    plt.imshow(support_img)
    for i, attn in enumerate(attn_weights[query_kpt_idx]):
        plt.scatter(support_kpts[i, 0], support_kpts[i, 1], 
                    s=attn*1000, alpha=0.7, c='red')
    plt.show()
```

**Benefits**:
- Better model interpretability
- Easier debugging
- Can identify failure modes

**Effort**: Medium

**Impact**: Low (doesn't improve accuracy, but helps debugging/analysis)

---

## 8. Temperature Scaling for Calibration

**Name**: Apply temperature scaling to coordinate predictions

**Files**: `models/roomformer_v2.py`, post-processing

**Problem**: Coordinate predictions may be overconfident or underconfident (not well-calibrated).

**Proposed Solution**:
```python
# After training, fit a temperature parameter T on validation set
# to calibrate predictions

# In inference:
pred_logits_calibrated = pred_logits / temperature
```

**Benefits**:
- Better calibrated predictions
- May improve downstream tasks that use prediction confidence

**Effort**: Low (standard calibration technique)

**Impact**: Low (doesn't directly improve PCK, but useful for confidence)

---

## 9. Mixed-Resolution Training

**Name**: Train with images at multiple resolutions (256, 512, 768)

**Files**: `datasets/mp100_cape.py`, `train_cape_episodic.py`

**Problem**: Model only sees 512×512 images, may not generalize to other scales.

**Proposed Solution**:
```python
# In data augmentation:
random_size = random.choice([256, 384, 512, 640])
transforms = A.Compose([
    A.Resize(random_size, random_size),
    # ... other transforms
])
```

**Benefits**:
- Better scale robustness
- May improve generalization

**Drawbacks**:
- Need to handle variable-size inputs (batch padding)
- Slower training (variable sizes harder to optimize)

**Effort**: Medium

**Impact**: Low-Medium (may improve robustness, unclear PCK impact)

---

## 10. Curriculum Learning

**Name**: Start training on easy categories, gradually add harder ones

**Files**: `train_cape_episodic.py`, `datasets/episodic_sampler.py`

**Problem**: Training on all categories from the start may be too hard (some categories have many keypoints, complex skeletons).

**Proposed Solution**:
```python
# Define curriculum: easy → hard categories
curriculum = [
    {'epochs': 0-50, 'categories': easy_categories},
    {'epochs': 50-100, 'categories': easy_categories + medium_categories},
    {'epochs': 100+, 'categories': all_categories},
]

# In training loop:
current_categories = get_curriculum_categories(epoch)
sampler.set_category_subset(current_categories)
```

**Benefits**:
- Easier optimization (start with simpler tasks)
- May improve final accuracy

**Drawbacks**:
- Requires defining category difficulty (heuristic)
- More complex training schedule

**Effort**: Medium

**Impact**: Low-Medium (may improve convergence, unclear final PCK impact)

---

## 11. Self-Supervised Pre-Training

**Name**: Pre-train support encoder on self-supervised task (e.g., contrastive learning on pose graphs)

**Files**: New `pretrain_support_encoder.py`, `models/support_encoder.py`

**Problem**: Support encoder initialized randomly, could benefit from pre-training.

**Proposed Solution**:
```python
# Contrastive learning on pose graphs:
# - Positive pairs: augmented versions of same pose graph
# - Negative pairs: different pose graphs
# - Learn embeddings where similar poses are close

# After pre-training, load weights into CAPE model
support_encoder.load_state_dict(pretrained_weights)
```

**Benefits**:
- Better initialization for support encoder
- May improve few-shot generalization

**Effort**: High (requires designing pre-training task, training pipeline)

**Impact**: Medium (could improve accuracy, especially on unseen categories)

---

## 12. Ensemble of Multiple Support Examples (K-Shot Extension)

**Name**: Use multiple support examples instead of just one

**Files**: `datasets/episodic_sampler.py`, `models/cape_model.py`

**Problem**: 1-shot is hardest setting. Using K>1 support examples could improve accuracy.

**Proposed Solution**:
```python
# In episodic sampler:
# Sample K support examples instead of 1

# In model:
# Aggregate K support embeddings (e.g., mean, max, attention pooling)
support_features_agg = support_features.mean(dim=1)  # (B, K, N, D) → (B, N, D)
```

**Benefits**:
- Better support representation (more information)
- Higher accuracy

**Drawbacks**:
- More compute (K times more support encoding)
- Easier task (not true 1-shot anymore)

**Effort**: Medium

**Impact**: High (K-shot typically improves PCK by 5-10% per additional shot)

---

## 13. Adaptive Sequence Length

**Name**: Use variable sequence length based on number of keypoints

**Files**: `datasets/tokenizer.py`, `models/roomformer_v2.py`

**Problem**: Fixed sequence length (e.g., 256) wastes computation for categories with few keypoints (e.g., 5 keypoints → 10 tokens + padding → 246 wasted positions).

**Proposed Solution**:
```python
# Dynamic sequence length based on category
max_seq_len = 2 * num_keypoints + 10  # 2 tokens per kpt + BOS/SEP/EOS + buffer

# In model:
# Adjust attention mask to actual sequence length
# Pad only to batch max, not global max
```

**Benefits**:
- Faster training/inference (less padding)
- More memory efficient

**Drawbacks**:
- Variable-length batching more complex
- Need to handle different sequence lengths in batch

**Effort**: High (requires significant refactoring)

**Impact**: Medium (speed improvement, unclear accuracy impact)

---

## Summary Table

| # | Improvement | Files | Effort | Impact | Priority |
|---|-------------|-------|--------|--------|----------|
| 1 | Semantic keypoint ordering | `mp100_cape.py` | Medium | Medium | Medium |
| 2 | Aspect-ratio-preserving resize | `mp100_cape.py` | Medium | Low-Med | Low |
| 3 | Full multi-instance support | `mp100_cape.py`, `episodic_sampler.py` | Medium | Low-Med | Low |
| 4 | Skeleton edge attention bias | `deformable_transformer_v2.py` | High | Medium | Medium |
| 5 | Uncertainty-aware predictions | `roomformer_v2.py`, losses | High | Medium | Low |
| 6 | Category embeddings | `support_encoder.py` | Low-Med | Low | Low |
| 7 | Attention visualization | New `visualization.py` | Medium | Low (debug) | Medium |
| 8 | Temperature scaling | Post-processing | Low | Low | Low |
| 9 | Mixed-resolution training | `mp100_cape.py` | Medium | Low-Med | Low |
| 10 | Curriculum learning | `train_cape_episodic.py` | Medium | Low-Med | Low |
| 11 | Self-supervised pre-training | New `pretrain_*.py` | High | Medium | Medium |
| 12 | K-shot extension (K>1) | `episodic_sampler.py`, `cape_model.py` | Medium | High | High |
| 13 | Adaptive sequence length | `tokenizer.py`, `roomformer_v2.py` | High | Medium | Low |

**Recommendation**: Consider implementing in order of **Impact/Effort ratio**:
1. **K-shot extension** (High impact, Medium effort) - if moving beyond 1-shot is acceptable
2. **Semantic keypoint ordering** (Medium impact, Medium effort) - could improve consistency
3. **Skeleton edge attention bias** (Medium impact, High effort) - good research direction
4. **Attention visualization** (Debug value, Medium effort) - useful for understanding model
5. Others as needed based on specific requirements

---

**Note**: These improvements are **optional** and **not required for correctness**. The current implementation (after CRITICAL FIX #1 and #2) is correct and ready for training.

