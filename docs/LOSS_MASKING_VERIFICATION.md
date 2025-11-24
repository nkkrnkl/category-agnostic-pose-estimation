# Loss Masking Verification: Visibility-Based Filtering

## ✅ **YES, Loss is Still Computed ONLY on Visible Keypoints**

After fixing the visibility array length mismatch bug, the loss masking logic remains intact and correct. Here's the complete flow:

---

## 📊 **Complete Data Flow: Visibility → Loss Masking**

### **Step 1: Extract Visibility from JSON Annotations**

**File**: `datasets/mp100_cape.py` (lines 291)

```python
# Extract visibility from COCO annotations
# Visibility values: 0 = unlabeled, 1 = occluded, 2 = visible
visibility = kpts[:, 2]  # Shape: (N,) where N = total keypoints
```

**Example**:
```python
visibility = [0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2]
#              ↑                       ↑                       ↑
#         unlabeled              unlabeled               visible
```

---

### **Step 2: Store TOTAL Keypoints + Visibility (Bug Fix)**

**File**: `datasets/mp100_cape.py` (lines 386-401)

```python
# CRITICAL FIX: Store TOTAL keypoints (not just visible count)
record["keypoints"] = kpts_array.tolist()  # All 17 keypoints
record["visibility"] = visibility.tolist()  # All 17 visibility flags
record["num_keypoints"] = len(kpts_array)  # 17 (TOTAL, not visible count)
record["num_visible_keypoints"] = int(np.sum(visibility > 0))  # 9 (for stats)
```

**Before (Buggy)**:
```python
record["num_keypoints"] = num_visible  # ❌ Stored visible count (9)
```

**After (Fixed)**:
```python
record["num_keypoints"] = len(kpts_array)  # ✅ Stores total count (17)
```

---

### **Step 3: Create Visibility Mask for Loss Computation**

**File**: `datasets/mp100_cape.py` (lines 666-694)

```python
# Create visibility mask for loss computation
visibility_mask = torch.zeros(self.tokenizer.seq_len, dtype=torch.bool)

# Mark coordinate tokens based on ACTUAL visibility
token_idx = 0
for kpt_idx, kpt in enumerate(keypoints):
    if token_labels[token_idx] == TokenType.coord.value:
        # Only mark as True if visibility > 0
        if visibility[kpt_idx] > 0:
            visibility_mask[token_idx] = True  # ✅ Will contribute to loss
        else:
            visibility_mask[token_idx] = False  # ❌ Will NOT contribute to loss
        token_idx += 1
```

**Example**:
```python
# Keypoints:         [kpt0,  kpt1,  kpt2,  kpt3,  kpt4, ...]
# Visibility:        [0,     2,     2,     2,     0,    ...]
# visibility_mask:   [False, True,  True,  True,  False, ...]
```

---

### **Step 4: Include Visibility Mask in Sequence Data**

**File**: `datasets/mp100_cape.py` (line 735)

```python
return {
    'target_seq': target_seq,           # Ground truth coordinates
    'token_labels': token_labels,       # Token types (coord/sep/eos)
    'mask': mask,                       # Valid tokens (not padding)
    'visibility_mask': visibility_mask, # ✅ VISIBLE keypoints only
    ...
}
```

---

### **Step 5: Pass Through Episodic Sampler**

**File**: `datasets/episodic_sampler.py` (lines 422-431)

```python
return {
    ...
    'query_targets': batched_seq_data,  # Includes 'visibility_mask'
    ...
}
```

The `visibility_mask` is included in `batched_seq_data` which is passed to the training loop.

---

### **Step 6: Training Loop Passes to Loss Function**

**File**: `engine_cape.py` (lines 89-104)

```python
# Query targets (includes visibility_mask)
query_targets = {}
for key, value in batch['query_targets'].items():
    query_targets[key] = value.to(device)

# Forward pass
outputs = model(samples=query_images, ...)

# Compute loss (visibility_mask is in query_targets)
loss_dict = criterion(outputs, query_targets)
```

---

### **Step 7: CAPE Loss Functions Apply Visibility Masking**

**File**: `models/cape_losses.py` (lines 108-136, 192-216)

#### **7a. Token Classification Loss (`loss_labels`)**

```python
def loss_labels(self, outputs, targets, indices):
    target_classes = targets['token_labels'].to(device)
    
    # Create mask: valid tokens (not padding)
    valid_mask = (target_classes != -1).bool()
    
    # Apply visibility mask (CAPE-specific)
    if 'visibility_mask' in targets:
        visibility_mask = targets['visibility_mask'].to(device)
        # Combine: must be both valid AND visible
        mask = valid_mask & visibility_mask  # ✅ Only visible keypoints
    else:
        mask = valid_mask
    
    # Compute loss ONLY on masked tokens
    loss_ce = label_smoothed_nll_loss(
        src_logits[mask],      # ✅ Only predictions for visible keypoints
        target_classes[mask],  # ✅ Only labels for visible keypoints
        epsilon=self.label_smoothing
    )
    return {'loss_ce': loss_ce}
```

**Example**:
```python
# Token labels:      [coord, coord, coord, coord, coord, sep, eos, pad, pad, ...]
# Visibility mask:   [False, True,  True,  True,  False, False, False, False, False, ...]
# Valid mask:        [True,  True,  True,  True,  True,  True,  True,  False, False, ...]
# Final mask:        [False, True,  True,  True,  False, False, False, False, False, ...]
#                     ↑      ↑      ↑      ↑      ↑
#                   skip   use    use    use    skip
```

#### **7b. Coordinate Regression Loss (`loss_polys`)**

```python
def loss_polys(self, outputs, targets, indices):
    # Create mask: coordinate tokens only
    coord_mask = (token_labels == 0).bool()
    
    # Apply visibility mask (CAPE-specific)
    if 'visibility_mask' in targets:
        visibility_mask = targets['visibility_mask'].to(device)
        # Combine: must be both coordinate token AND visible
        mask = coord_mask & visibility_mask  # ✅ Only visible coordinates
    else:
        mask = coord_mask
    
    # Compute L1 loss ONLY on visible coordinate tokens
    loss_coords = F.l1_loss(
        src_poly[mask],      # ✅ Only predictions for visible keypoints
        target_polys[mask]   # ✅ Only ground truth for visible keypoints
    )
    return {'loss_coords': loss_coords}
```

---

## 📋 **Summary: Loss Masking is Working Correctly**

| Component | What It Does | Status |
|-----------|--------------|--------|
| **JSON Annotations** | Store visibility per keypoint (0/1/2) | ✅ |
| **Dataset Loading** | Extract all keypoints + visibility | ✅ |
| **Visibility Mask Creation** | Mark visible keypoints as `True` | ✅ |
| **Sequence Data** | Include `visibility_mask` in return dict | ✅ |
| **Episodic Sampler** | Pass `visibility_mask` through to batches | ✅ |
| **Training Loop** | Send `visibility_mask` to loss function | ✅ |
| **CAPE Loss (labels)** | Apply mask: `valid_mask & visibility_mask` | ✅ |
| **CAPE Loss (coords)** | Apply mask: `coord_mask & visibility_mask` | ✅ |

---

## 🎯 **Key Takeaways**

### **What Changed in the Bug Fix**:
1. **Fixed `num_keypoints`** meaning: Now stores TOTAL keypoints (17), not visible count (9)
2. **Fixed fallback visibility**: Now creates correct-length array (17 instead of 9)

### **What DIDN'T Change**:
1. **Visibility mask creation**: Still marks only visible keypoints as `True`
2. **Loss masking logic**: Still applies `visibility_mask` to filter loss computation
3. **Gradient computation**: Still only trains on visible keypoints

### **Result**:
- ✅ Loss is computed **ONLY** on visible keypoints (visibility > 0)
- ✅ Invisible/occluded keypoints (visibility == 0) are **excluded** from loss
- ✅ Model does **NOT** get penalized for predictions on unlabeled keypoints
- ✅ Training quality is **preserved** and **correct**

---

## 🧪 **How to Verify**

Add a debug print in `models/cape_losses.py`:

```python
def loss_polys(self, outputs, targets, indices):
    coord_mask = (token_labels == 0).bool()
    if 'visibility_mask' in targets:
        visibility_mask = targets['visibility_mask'].to(device)
        mask = coord_mask & visibility_mask
        
        # Debug: Print masking statistics
        print(f"Coord tokens: {coord_mask.sum().item()}")
        print(f"Visible tokens: {visibility_mask.sum().item()}")
        print(f"Masked tokens: {mask.sum().item()}")
        # Expected: masked < coord (some are filtered out)
    else:
        mask = coord_mask
    
    loss_coords = F.l1_loss(src_poly[mask], target_polys[mask])
    return {'loss_coords': loss_coords}
```

**Expected output**:
```
Coord tokens: 17
Visible tokens: 10
Masked tokens: 10  ← Loss only computed on 10 visible keypoints!
```

---

## ✅ **Conclusion**

**YES**, even after fixing the visibility array length mismatch bug, **loss is still computed ONLY on visible keypoints**. The masking logic is intact and working correctly. The bug fix only corrected the LENGTH of the visibility array, not its CONTENT or APPLICATION in loss computation.

