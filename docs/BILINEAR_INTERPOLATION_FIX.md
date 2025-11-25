# Bilinear Interpolation Fix - Issue #18 Clarification

## 🔍 Summary

**Issue #18** was initially misdiagnosed. The "duplicate sequences" were **NOT** duplicates - they are **required** for bilinear interpolation in the Raster2Seq model.

---

## ❌ What Was Wrong

Early in the audit, we incorrectly identified `seq21`, `seq22`, `delta_x2`, `delta_y2` as "duplicates" and removed them. This was **WRONG**.

### Erroneous Fix (Initially Applied)
```python
# WRONG! Missing seq21, seq22, delta_x2, delta_y2
seq_dict = {
    'seq11': quantized[:, 0],  # x coordinates
    'seq12': quantized[:, 1],  # y coordinates
    'delta_x1': np.ones_like(quantized[:, 0]) * 0.5,
    'delta_y1': np.ones_like(quantized[:, 1]) * 0.5,
}
```

---

## ✅ What Is Correct

Raster2Seq uses **bilinear interpolation** to embed continuous coordinates into a discrete grid. For each coordinate `(x, y)`, we need **4 grid points** representing the corners of the grid cell:

1. **`index11`**: `(floor(x), floor(y))` - bottom-left corner
2. **`index21`**: `(ceil(x), floor(y))` - bottom-right corner
3. **`index12`**: `(floor(x), ceil(y))` - top-left corner
4. **`index22`**: `(ceil(x), ceil(y))` - top-right corner

The model then uses `delta_x` and `delta_y` to interpolate between these 4 points:

- **`delta_x1 = x - floor(x)`**: Distance from left edge
- **`delta_x2 = 1 - delta_x1 = ceil(x) - x`**: Distance from right edge
- **`delta_y1 = y - floor(y)`**: Distance from bottom edge
- **`delta_y2 = 1 - delta_y1 = ceil(y) - y`**: Distance from top edge

### Correct Implementation (in `_tokenize_keypoints`)

```python
# 4 indices for bilinear interpolation (floor/ceil combinations)
index11 = [[math.floor(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
index21 = [[math.ceil(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
index12 = [[math.floor(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]
index22 = [[math.ceil(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]

# Tokenize all 4 sequences
seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
seq22 = self.tokenizer(index22, add_bos=True, add_eos=False, dtype=torch.long)

# Compute deltas
delta_x1 = [p[0] - math.floor(p[0]) for p in polygon]  # Distance from floor
delta_y1 = [p[1] - math.floor(p[1]) for p in polygon]
delta_x2 = 1 - delta_x1  # Distance from ceil
delta_y2 = 1 - delta_y1

return {
    'seq11': seq11,
    'seq21': seq21,  # NOT a duplicate!
    'seq12': seq12,
    'seq22': seq22,  # NOT a duplicate!
    'delta_x1': delta_x1,
    'delta_x2': delta_x2,  # NOT a duplicate!
    'delta_y1': delta_y1,
    'delta_y2': delta_y2,  # NOT a duplicate!
    ...
}
```

---

## 📚 Why This Matters

Bilinear interpolation is a standard technique for continuous-to-discrete coordinate quantization:

1. **Smoother embeddings**: Instead of hard quantization (rounding to nearest bin), bilinear interpolation creates a weighted combination of the 4 surrounding grid points.
2. **Better gradients**: During backpropagation, gradients flow to all 4 corners, not just one.
3. **Required by Raster2Seq**: This is how the original paper implements coordinate embedding.

---

## 🛠️ Current Status

- ✅ **`_tokenize_keypoints()` (lines 636-814)**: Correctly implements bilinear interpolation with all 4 sequences and deltas.
- ✅ **`convert_to_sequence()` (lines 56-68)**: Marked as deprecated to avoid confusion.
- ✅ **Model (`roomformer_v2.py`)**: Uses all 4 sequences correctly during training and inference.

---

## 📖 References

- **Raster2Seq Paper**: Section on coordinate quantization
- **`datasets/mp100_cape.py`**: Lines 686-814 for the correct implementation
- **`models/roomformer_v2.py`**: Lines 338-359, 423-460 for how sequences are used

---

## ⚠️ Lesson Learned

Always verify the **mathematical/algorithmic reason** for data structures before labeling them as "duplicates". In this case:
- The 4 sequences represent the 4 corners of a grid cell.
- The deltas represent interpolation weights.
- Removing them would break the bilinear interpolation logic.

**Rule**: If you see 4 similar-looking arrays in a coordinate quantization context, they're probably for bilinear interpolation, not duplicates!

