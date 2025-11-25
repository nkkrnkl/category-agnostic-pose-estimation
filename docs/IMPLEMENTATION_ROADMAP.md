# CapeX Integration: Concrete Implementation Roadmap

**Last Updated**: November 25, 2025  
**Status**: Ready to implement  
**Recommended Timeline**: 3-4 weeks for moderate integration

---

## Day-by-Day Plan (Moderate Integration)

### Week 1: Foundation & Graph Utilities

#### Day 1 (Monday) - 4 hours
**Goal**: Create graph utility module

**Tasks**:
1. Create `models/graph_utils.py`
2. Port `adj_from_skeleton()` from CapeX
3. Write comprehensive docstring
4. Add type hints

**Code to write**:
```python
# models/graph_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def adj_from_skeleton(
    num_pts: int,
    skeleton: List[List[List[int]]],
    mask: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    [Full docstring from CAPEX_CODE_SNIPPETS.md]
    """
    # [Copy implementation from audit document]
```

**Test**:
```bash
python -c "from models.graph_utils import adj_from_skeleton; print('✅ Import works')"
```

**Success criteria**: ✅ File created, function imports successfully

---

#### Day 2 (Tuesday) - 4 hours
**Goal**: Port GCNLayer and write tests

**Tasks**:
1. Add `GCNLayer` class to `models/graph_utils.py`
2. Create `tests/test_graph_utils.py`
3. Write unit tests for:
   - Adjacency symmetry
   - Adjacency normalization
   - GCN shape preservation
   - GCN gradient flow

**Code to write**:
```python
# In models/graph_utils.py

class GCNLayer(nn.Module):
    """[Full docstring from CAPEX_CODE_SNIPPETS.md]"""
    # [Copy implementation]

# tests/test_graph_utils.py

import torch
import pytest
from models.graph_utils import adj_from_skeleton, GCNLayer

def test_adj_from_skeleton_triangle():
    """Test adjacency construction for simple triangle graph."""
    # [Implementation from audit document]

def test_gcn_shape_preservation():
    """Test GCN maintains correct tensor shapes."""
    # [Implementation from audit document]

# ... more tests ...
```

**Test**:
```bash
python -m pytest tests/test_graph_utils.py -v
```

**Success criteria**: ✅ All tests pass

---

#### Day 3 (Wednesday) - 4 hours
**Goal**: Port positional encoding

**Tasks**:
1. Check if `SinePositionalEncoding.forward_coordinates()` exists in our codebase
2. If not, port from CapeX
3. Add to `models/position_encoding.py` (or create file)
4. Write tests

**Code to write**:
```python
# models/position_encoding.py (if creating new)

import torch
import torch.nn as nn

class SinePositionalEncoding(nn.Module):
    """[Copy from CapeX with full docstring]"""
    
    def forward_coordinates(self, coord):
        """[Copy from CAPEX_CODE_SNIPPETS.md Section 3]"""
        # [Full implementation]

# tests/test_position_encoding.py

def test_positional_encoding_deterministic():
    """Test that same coords give same encoding."""
    # [Implementation]

def test_positional_encoding_normalized():
    """Test that encoding works with [0,1] coords."""
    # [Implementation]
```

**Test**:
```bash
python -m pytest tests/test_position_encoding.py -v
```

**Success criteria**: ✅ Positional encoding works, tests pass

---

#### Day 4 (Thursday) - 4 hours
**Goal**: Design geometric support encoder

**Tasks**:
1. Create `models/support_encoder.py`
2. Implement `GeometricSupportEncoder` (Variant B: Coords + Positional)
3. Write tests
4. Test with dummy data

**Code to write**:
```python
# models/support_encoder.py

import torch
import torch.nn as nn
from .position_encoding import SinePositionalEncoding

class GeometricSupportEncoder(nn.Module):
    """
    Encodes support keypoints using ONLY geometric information.
    
    Replaces CapeX's text-based support encoding with:
      - Coordinate MLP (learns feature representation from (x,y))
      - Positional encoding (sine/cosine functions for translation invariance)
      - (Optional) Graph pre-encoding (GNN for structural context)
    
    Args:
        hidden_dim (int): Output embedding dimension (default: 256)
        use_graph_preencoding (bool): Whether to apply GNN before outputting
        num_graph_layers (int): Number of GCN layers for pre-encoding
        
    Input:
        coords: [bs, num_kpts, 2] - Normalized support coordinates in [0,1]
        mask: [bs, num_kpts] - Visibility mask (True=visible, False=padded)
        skeleton: List of edge lists (optional, for graph pre-encoding)
        
    Output:
        support_embed: [bs, num_kpts, hidden_dim] - Geometric embeddings
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        use_graph_preencoding: bool = False,
        num_graph_layers: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_graph_preencoding = use_graph_preencoding
        
        # 1. Coordinate embedding MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Positional encoding (from CapeX)
        self.pos_encoding = SinePositionalEncoding(
            num_feats=hidden_dim // 2,
            normalize=True,
            scale=2 * 3.14159265359  # 2π
        )
        
        # 3. Optional: Graph pre-encoding
        if use_graph_preencoding:
            from .graph_utils import GCNLayer
            self.graph_layers = nn.ModuleList([
                GCNLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    kernel_size=2,
                    batch_first=True
                )
                for _ in range(num_graph_layers)
            ])
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        skeleton: List[List[List[int]]] = None
    ) -> torch.Tensor:
        """
        Forward pass: coords → geometric embeddings
        
        Args:
            coords: [bs, num_kpts, 2] in [0, 1]
            mask: [bs, num_kpts] - True=visible
            skeleton: List of edge lists (needed if use_graph_preencoding=True)
            
        Returns:
            support_embed: [bs, num_kpts, hidden_dim]
        """
        # 1. Embed coordinates
        coord_feat = self.coord_mlp(coords)  # [bs, num_kpts, hidden_dim]
        
        # 2. Add positional encoding
        pos_feat = self.pos_encoding.forward_coordinates(coords)  # [bs, num_kpts, hidden_dim]
        
        # 3. Combine
        support_embed = coord_feat + pos_feat
        
        # 4. Optional: Graph pre-encoding
        if self.use_graph_preencoding:
            assert skeleton is not None, "Skeleton required for graph pre-encoding"
            
            from .graph_utils import adj_from_skeleton
            
            # Build adjacency
            adj = adj_from_skeleton(
                num_pts=coords.shape[1],
                skeleton=skeleton,
                mask=~mask,  # CapeX convention: True=invalid
                device=coords.device
            )
            
            # Apply GCN layers
            for gcn_layer in self.graph_layers:
                support_embed = gcn_layer(support_embed, adj)
        
        return support_embed

# tests/test_support_encoder.py

def test_geometric_support_encoder_basic():
    """Test basic forward pass without graph pre-encoding."""
    encoder = GeometricSupportEncoder(hidden_dim=256, use_graph_preencoding=False)
    
    coords = torch.rand(2, 17, 2)  # [bs=2, num_kpts=17, 2]
    mask = torch.ones(2, 17).bool()  # All visible
    
    output = encoder(coords, mask)
    
    assert output.shape == (2, 17, 256), f"Wrong shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output has NaNs"
    assert output.std() > 0.01, "Output has no variance"

def test_geometric_support_encoder_with_graph():
    """Test forward pass WITH graph pre-encoding."""
    encoder = GeometricSupportEncoder(
        hidden_dim=256, 
        use_graph_preencoding=True,
        num_graph_layers=2
    )
    
    coords = torch.rand(2, 17, 2)
    mask = torch.ones(2, 17).bool()
    skeleton = [
        [[0, 1], [1, 2], [2, 3]],
        [[0, 1], [1, 2], [2, 3]]
    ]
    
    output = encoder(coords, mask, skeleton)
    
    assert output.shape == (2, 17, 256)
    assert not torch.isnan(output).any()

def test_gradient_flow():
    """Test gradients flow through encoder."""
    encoder = GeometricSupportEncoder(hidden_dim=256)
    
    coords = torch.rand(2, 17, 2, requires_grad=True)
    mask = torch.ones(2, 17).bool()
    
    output = encoder(coords, mask)
    loss = output.sum()
    loss.backward()
    
    assert coords.grad is not None, "No gradient for coords"
    assert not torch.isnan(coords.grad).any(), "Gradient has NaNs"
```

**Test**:
```bash
python -m pytest tests/test_support_encoder.py -v
```

**Success criteria**: ✅ GeometricSupportEncoder works, all tests pass

---

#### Day 5 (Friday) - 4 hours
**Goal**: Integration planning and code review

**Tasks**:
1. Review our current `models/cape_model.py` architecture
2. Identify where to inject support encoder
3. Plan decoder modifications for GCN
4. Create integration checklist

**Code to review**:
```bash
# Read our current model
cat models/cape_model.py

# Check decoder structure
grep -n "class.*Decoder\|def forward" models/cape_model.py
```

**Document**:
Create `INTEGRATION_CHECKLIST.md` with specific line numbers and modification points.

**Success criteria**: ✅ Clear integration plan documented

---

### Week 2: Model Integration

#### Day 6 (Monday) - 6 hours
**Goal**: Add geometric support encoder to main model

**Tasks**:
1. Modify `models/cape_model.py` to import `GeometricSupportEncoder`
2. Add to `__init__` method
3. Modify `forward()` to use it
4. Add config parameters

**Code changes**:
```python
# models/cape_model.py

from .support_encoder import GeometricSupportEncoder

class CAPEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ... existing code ...
        
        # NEW: Geometric support encoder
        self.support_encoder = GeometricSupportEncoder(
            hidden_dim=config.get('hidden_dim', 256),
            use_graph_preencoding=config.get('use_graph_preencoding', False),
            num_graph_layers=config.get('num_graph_layers', 2)
        )
        
        # ... rest of init ...
    
    def forward(self, samples, support_coords, support_mask, skeleton, targets):
        """
        Args:
            samples: Query images [bs, 3, H, W]
            support_coords: [bs, num_kpts, 2] - NEW ARGUMENT
            support_mask: [bs, num_kpts] - NEW ARGUMENT
            skeleton: List of edge lists - NEW ARGUMENT
            targets: Ground truth for training
        """
        # NEW: Encode support geometrically
        support_embed = self.support_encoder(
            support_coords, support_mask, skeleton
        )  # [bs, num_kpts, 256]
        
        # ... rest of forward pass ...
        # (Pass support_embed to decoder/transformer)
```

**Test**:
```python
# Quick sanity test
model = CAPEModel(config)
coords = torch.rand(2, 17, 2)
mask = torch.ones(2, 17).bool()
skeleton = [[[0,1], [1,2]], [[0,1], [1,2]]]
img = torch.rand(2, 3, 256, 256)

output = model(img, coords, mask, skeleton, targets=None)
print(f"✅ Forward pass works! Output shape: {output.shape}")
```

**Success criteria**: ✅ Model instantiates, forward pass runs

---

#### Day 7 (Tuesday) - 6 hours
**Goal**: Add GCN to decoder layers

**Tasks**:
1. Identify our decoder layer class
2. Add `graph_decoder` mode parameter
3. Modify FFN to conditionally use GCN
4. Test with/without graph

**Code changes**:
```python
# In decoder layer class:

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, graph_decoder=None):
        super().__init__()
        
        # ... existing attention layers ...
        
        # Modify FFN based on graph_decoder mode
        self.graph_decoder = graph_decoder
        
        if graph_decoder == 'pre':
            from .graph_utils import GCNLayer
            self.ffn1 = GCNLayer(
                d_model, dim_feedforward, kernel_size=2, batch_first=False
            )
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
        else:
            # Standard FFN (no graph)
            self.ffn1 = nn.Linear(d_model, dim_feedforward)
            self.ffn2 = nn.Linear(dim_feedforward, d_model)
    
    def forward(self, tgt, memory, ..., skeleton=None, mask=None):
        # ... self-attention and cross-attention ...
        
        # FFN with optional GCN
        if self.graph_decoder == 'pre' and skeleton is not None:
            from .graph_utils import adj_from_skeleton
            
            num_pts, bs, c = tgt.shape
            adj = adj_from_skeleton(num_pts, skeleton, mask, tgt.device)
            
            # GCN → ReLU → Linear
            tgt2 = self.ffn1(tgt, adj)  # GCN layer
            tgt2 = F.relu(tgt2)
            tgt2 = self.ffn2(tgt2)  # Linear layer
        else:
            # Standard FFN
            tgt2 = self.ffn1(tgt)
            tgt2 = F.relu(tgt2)
            tgt2 = self.ffn2(tgt2)
        
        # Residual
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        
        return tgt
```

**Test**:
```python
# Test both modes
layer_no_graph = DecoderLayer(256, 8, 768, graph_decoder=None)
layer_with_graph = DecoderLayer(256, 8, 768, graph_decoder='pre')

# Forward pass
tgt = torch.rand(17, 2, 256)
memory = torch.rand(1024, 2, 256)
mask = torch.zeros(2, 17).bool()
skeleton = [[[0,1], [1,2]], [[0,1], [1,2]]]

out_no_graph = layer_no_graph(tgt, memory, skeleton=None)
out_with_graph = layer_with_graph(tgt, memory, skeleton=skeleton, mask=mask)

print(f"✅ Both modes work!")
print(f"   No graph shape: {out_no_graph.shape}")
print(f"   With graph shape: {out_with_graph.shape}")
```

**Success criteria**: ✅ Both modes work, shapes correct

---

#### Day 8 (Wednesday) - 4 hours
**Goal**: Update dataset to provide skeleton

**Tasks**:
1. Modify `datasets/mp100_cape.py` to include skeleton in return dict
2. Ensure skeleton format matches CapeX (0-indexed, list per sample)
3. Test episodic sampler with skeleton

**Code changes**:
```python
# datasets/mp100_cape.py

class MP100CAPEDataset(Dataset):
    def __getitem__(self, idx):
        # ... existing code ...
        
        # Add skeleton to return dict
        result = {
            'img': img_cropped,
            'keypoints': keypoints_normalized,
            'visibility': visibility_mask,
            'skeleton': self.get_skeleton(category_id),  # NEW
            # ... other fields ...
        }
        
        return result
    
    def get_skeleton(self, category_id):
        """
        Get skeleton edges for a category.
        
        Returns:
            List of [src, dst] edge pairs (0-indexed)
        """
        # Get from annotation
        skeleton = self.coco.cats[category_id].get('skeleton', [])
        
        # Ensure 0-indexed (some datasets use 1-indexed)
        skeleton = [[max(0, src-1), max(0, dst-1)] for src, dst in skeleton]
        
        return skeleton
```

**Test**:
```python
dataset = MP100CAPEDataset(...)
sample = dataset[0]

print(f"Skeleton: {sample['skeleton']}")
# Should see: [[0, 1], [1, 2], ...]

assert isinstance(sample['skeleton'], list)
assert all(isinstance(edge, list) and len(edge) == 2 for edge in sample['skeleton'])
```

**Success criteria**: ✅ Dataset provides skeleton in correct format

---

#### Day 9 (Thursday) - 4 hours
**Goal**: Update episodic sampler to batch skeletons

**Tasks**:
1. Modify `datasets/episodic_sampler.py` collate function
2. Handle variable skeleton sizes across categories
3. Test batching

**Code changes**:
```python
# datasets/episodic_sampler.py

def collate_fn(batch):
    """Collate episodes into batches."""
    # ... existing code for images, keypoints, etc. ...
    
    # NEW: Collect skeletons
    skeletons = [sample['skeleton'] for sample in batch]
    # Keep as list (each category can have different skeleton)
    
    return {
        'images': torch.stack(images),
        'keypoints': torch.stack(keypoints),
        'visibility': torch.stack(visibility),
        'skeleton': skeletons,  # List of edge lists
        # ... other fields ...
    }
```

**Test**:
```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))

print(f"Batch skeleton: {batch['skeleton']}")
# Should see list of 4 edge lists

assert len(batch['skeleton']) == 4
assert isinstance(batch['skeleton'][0], list)
```

**Success criteria**: ✅ Batching works, skeletons preserved

---

#### Day 10 (Friday) - 4 hours
**Goal**: End-to-end integration test

**Tasks**:
1. Wire support encoder + GCN + dataset together
2. Run forward pass with real data
3. Debug shape mismatches
4. Verify gradients flow

**Test script**:
```python
# test_integration.py

import torch
from models.cape_model import CAPEModel
from datasets.episodic_sampler import build_episodic_dataloader

# 1. Build model
config = {
    'hidden_dim': 256,
    'use_graph_preencoding': False,
    'graph_decoder': 'pre'
}
model = CAPEModel(config)

# 2. Build dataloader
loader = build_episodic_dataloader(
    base_dataset=...,
    split='train',
    batch_size=2,
    num_queries_per_episode=2
)

# 3. Get one batch
batch = next(iter(loader))

# 4. Forward pass
output = model(
    samples=batch['query_images'],
    support_coords=batch['support_coords'],
    support_mask=batch['support_mask'],
    skeleton=batch['skeleton'],
    targets=batch['targets']
)

print(f"✅ End-to-end works!")
print(f"   Output shape: {output.shape}")

# 5. Backward pass
loss = output.sum()
loss.backward()

print(f"✅ Gradients flow!")

# 6. Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")
```

**Success criteria**: ✅ Forward and backward pass work without errors

---

### Week 3: Training & Debugging

#### Day 11-12 (Mon-Tue) - 8 hours
**Goal**: First training run (baseline - no graph)

**Tasks**:
1. Create config: `configs/cape_baseline_no_graph.yaml`
2. Train for 10 epochs
3. Monitor loss, PCK
4. Debug any issues

**Config**:
```yaml
# configs/cape_baseline_no_graph.yaml

model:
  hidden_dim: 256
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_heads: 8
  graph_decoder: null  # No graph
  use_graph_preencoding: false

training:
  epochs: 20
  batch_size: 2
  accumulation_steps: 4
  learning_rate: 1e-4
  early_stopping_patience: 10

data:
  num_queries_per_episode: 2
  train_episodes_per_epoch: 100
  val_episodes_per_epoch: 50
```

**Command**:
```bash
python train_cape_episodic.py \
  --config configs/cape_baseline_no_graph.yaml \
  --output_dir ./outputs/baseline_no_graph \
  --epochs 10
```

**Monitor**:
```bash
# Watch training
tail -f outputs/baseline_no_graph/train.log

# Check tensorboard (if available)
tensorboard --logdir outputs/baseline_no_graph
```

**Success criteria**: 
- ✅ Training completes without errors
- ✅ Loss decreases
- ✅ PCK > 20% after 10 epochs (very low bar)

---

#### Day 13-14 (Wed-Thu) - 8 hours
**Goal**: Training run with graph encoding

**Tasks**:
1. Create config: `configs/cape_with_graph.yaml`
2. Train for 10 epochs
3. Compare to baseline
4. Visualize predictions

**Config**:
```yaml
# configs/cape_with_graph.yaml

model:
  hidden_dim: 256
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_heads: 8
  graph_decoder: 'pre'  # ← ENABLE GRAPH
  use_graph_preencoding: false  # Start simple

# ... rest same as baseline ...
```

**Command**:
```bash
python train_cape_episodic.py \
  --config configs/cape_with_graph.yaml \
  --output_dir ./outputs/with_graph \
  --epochs 10
```

**Compare**:
```bash
# Compare final validation PCK
grep "Val PCK" outputs/baseline_no_graph/train.log
grep "Val PCK" outputs/with_graph/train.log

# Expected: with_graph > baseline (hopefully!)
```

**Success criteria**: 
- ✅ Training completes
- ✅ PCK with graph ≥ PCK without graph (ideally +3-5%)

---

#### Day 15 (Friday) - 4 hours
**Goal**: Visualization and debugging

**Tasks**:
1. Run `visualize_cape_predictions.py` on both checkpoints
2. Compare predicted skeletons
3. Debug any weird predictions
4. Create comparison images

**Commands**:
```bash
# Visualize baseline
python visualize_cape_predictions.py \
  --checkpoint outputs/baseline_no_graph/best_model.pth \
  --output_dir visualizations/baseline

# Visualize with graph
python visualize_cape_predictions.py \
  --checkpoint outputs/with_graph/best_model.pth \
  --output_dir visualizations/with_graph

# Compare side-by-side
python compare_visualizations.py \
  visualizations/baseline \
  visualizations/with_graph
```

**Look for**:
- Does graph version respect skeleton structure better?
- Are connected keypoints more coherent?
- Any failure modes?

**Success criteria**: ✅ Qualitative improvement visible with graph

---

### Week 3: Full Training & Optimization

#### Day 16-17 (Mon-Tue) - 8 hours
**Goal**: Extended training (50 epochs)

**Tasks**:
1. Train baseline for 50 epochs
2. Train with graph for 50 epochs
3. Monitor convergence
4. Save best checkpoints

**Commands**:
```bash
# Baseline
python train_cape_episodic.py \
  --config configs/cape_baseline_no_graph.yaml \
  --epochs 50 \
  --early_stopping_patience 15 \
  --output_dir ./outputs/baseline_50ep

# With graph
python train_cape_episodic.py \
  --config configs/cape_with_graph.yaml \
  --epochs 50 \
  --early_stopping_patience 15 \
  --output_dir ./outputs/with_graph_50ep
```

**Expected runtime**: ~6-8 hours per run on M4

**Success criteria**: 
- ✅ Training converges (loss plateaus)
- ✅ PCK > 40% on validation

---

#### Day 18 (Wednesday) - 4 hours
**Goal**: Ablation - Graph pre-encoding

**Tasks**:
1. Create config with `use_graph_preencoding: true`
2. Train for 20 epochs
3. Compare to previous runs

**Config**:
```yaml
# configs/cape_with_graph_preenc.yaml

model:
  graph_decoder: 'pre'
  use_graph_preencoding: true  # ← ENABLE PRE-ENCODING
  num_graph_layers: 2
```

**Compare**:
| Config | Graph Decoder | Graph Pre-Enc | Val PCK |
|--------|---------------|---------------|---------|
| Baseline | None | False | ??? |
| Graph | 'pre' | False | ??? |
| Graph+Pre | 'pre' | True | ??? |

**Success criteria**: ✅ Understand contribution of each component

---

#### Day 19 (Thursday) - 4 hours
**Goal**: Hyperparameter tuning

**Tasks**:
1. Try different learning rates (1e-5, 5e-5, 1e-4, 5e-4)
2. Try different GCN layer counts (1, 2, 3)
3. Try different hidden dims (128, 256, 512)

**Grid search** (simplified):
```bash
for lr in 1e-5 5e-5 1e-4; do
  for gcn_layers in 1 2 3; do
    python train_cape_episodic.py \
      --lr $lr \
      --num_graph_layers $gcn_layers \
      --epochs 20 \
      --output_dir ./outputs/sweep_lr${lr}_gcn${gcn_layers}
  done
done
```

**Success criteria**: ✅ Find best hyperparameters

---

#### Day 20 (Friday) - 4 hours
**Goal**: Analysis and reporting

**Tasks**:
1. Collect all results
2. Create performance table
3. Analyze failure cases
4. Write summary report

**Results table**:
```markdown
| Configuration | Val PCK | Train Time | Notes |
|---------------|---------|------------|-------|
| Baseline (no graph) | 45.2% | 6h | Simple, fast |
| Graph (pre) | 52.8% | 7h | +7.6% improvement! |
| Graph+Pre-enc | 54.1% | 8h | +1.3% additional |
| Best tuned | 58.3% | 8h | lr=5e-5, gcn_layers=2 |
```

**Visualizations**:
- Training curves (loss over epochs)
- PCK by category (which categories benefit from graph?)
- Qualitative examples (good predictions, failure cases)

**Success criteria**: ✅ Clear understanding of what works

---

### Week 4: Polish & Documentation

#### Day 21-22 (Mon-Tue) - 8 hours
**Goal**: Code cleanup and optimization

**Tasks**:
1. Refactor code (remove debug prints, add comments)
2. Add comprehensive docstrings
3. Optimize inference speed (if needed)
4. Run linter, fix warnings

**Checklist**:
- [ ] All functions have docstrings
- [ ] Type hints added
- [ ] No unused imports
- [ ] No hardcoded paths
- [ ] Config parameters documented
- [ ] Code follows PEP 8

**Success criteria**: ✅ Clean, professional code

---

#### Day 23 (Wednesday) - 4 hours
**Goal**: Documentation

**Tasks**:
1. Update `README.md` with CapeX integration notes
2. Document new command-line arguments
3. Create example configs
4. Write user guide

**Documents to create/update**:
- `README.md`: Add "CapeX Graph Encoding" section
- `docs/GRAPH_ENCODING.md`: Technical details
- `docs/USAGE.md`: How to enable/disable graph
- `configs/examples/`: Example configs for different scenarios

**Success criteria**: ✅ User can understand and use the new features

---

#### Day 24 (Thursday) - 4 hours
**Goal**: Final validation

**Tasks**:
1. Train best model for 100 epochs (overnight)
2. Evaluate on test set (unseen categories)
3. Compare to any baselines
4. Create final report

**Final evaluation**:
```bash
# Train best configuration
python train_cape_episodic.py \
  --config configs/cape_best.yaml \
  --epochs 100 \
  --output_dir ./outputs/final_model

# Evaluate on test set
python evaluate_cape.py \
  --checkpoint outputs/final_model/best_model.pth \
  --split test \
  --output_file results/test_set_performance.json

# Generate report
python generate_report.py results/test_set_performance.json
```

**Success criteria**: ✅ Test set PCK documented, ready for publication/use

---

## Quick Reference: Files to Create/Modify

### New Files to Create

1. ✅ `models/graph_utils.py` (~120 lines)
   - `adj_from_skeleton()`
   - `GCNLayer`

2. ✅ `models/support_encoder.py` (~100 lines)
   - `GeometricSupportEncoder`

3. ✅ `models/position_encoding.py` (~80 lines) - if not exists
   - `SinePositionalEncoding.forward_coordinates()`

4. ✅ `tests/test_graph_utils.py` (~150 lines)
   - Unit tests for graph functions

5. ✅ `tests/test_support_encoder.py` (~120 lines)
   - Unit tests for support encoder

6. ✅ `configs/cape_baseline_no_graph.yaml` (~50 lines)
7. ✅ `configs/cape_with_graph.yaml` (~50 lines)
8. ✅ `configs/cape_best.yaml` (~50 lines)

9. ✅ `INTEGRATION_CHECKLIST.md` (documentation)
10. ✅ `docs/GRAPH_ENCODING.md` (documentation)

### Existing Files to Modify

1. ⚠️ `models/cape_model.py`
   - Add `GeometricSupportEncoder` to `__init__`
   - Modify `forward()` to use geometric support
   - Add GCN to decoder layers
   - **Lines changed**: ~50-100 (depending on architecture)

2. ⚠️ `datasets/mp100_cape.py`
   - Add skeleton to `__getitem__` return
   - Implement `get_skeleton()` method
   - **Lines changed**: ~20-30

3. ⚠️ `datasets/episodic_sampler.py`
   - Update `collate_fn` to handle skeletons
   - **Lines changed**: ~10-15

4. ⚠️ `train_cape_episodic.py`
   - Add `--use_graph_decoder` argument
   - Pass skeleton to model in training loop
   - **Lines changed**: ~20-30

5. ⚠️ `README.md`
   - Add CapeX integration section
   - Update usage instructions
   - **Lines changed**: ~30-50

**Total new code**: ~700 lines  
**Total modified code**: ~130-225 lines  
**Total effort**: ~3-4 weeks

---

## Minimal Viable Product (MVP) Checklist

**Goal**: Get SOMETHING working quickly (1 week)

- [ ] Day 1: Port `adj_from_skeleton()` and `GCNLayer`
- [ ] Day 2: Port `SinePositionalEncoding.forward_coordinates()`
- [ ] Day 3: Implement minimal `GeometricSupportEncoder` (coords + pos only)
- [ ] Day 4: Integrate into model (skip GCN in decoder for now)
- [ ] Day 5: Train for 5 epochs, verify it works
- [ ] Day 6-7: Add GCN to decoder, re-train

**Expected outcome**: Working geometry-only model, PCK ~40-50%

**Decision point**: If this works, continue to full integration. If not, debug before proceeding.

---

## Tools & Scripts to Create

### 1. Comparison Script

```python
# scripts/compare_runs.py

import json
import matplotlib.pyplot as plt

def compare_runs(baseline_dir, experiment_dir):
    """Compare two training runs and generate report."""
    
    # Load metrics
    baseline_metrics = json.load(open(f"{baseline_dir}/metrics.json"))
    experiment_metrics = json.load(open(f"{experiment_dir}/metrics.json"))
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    axes[0].plot(baseline_metrics['train_loss'], label='Baseline')
    axes[0].plot(experiment_metrics['train_loss'], label='With Graph')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    # PCK curves
    axes[1].plot(baseline_metrics['val_pck'], label='Baseline')
    axes[1].plot(experiment_metrics['val_pck'], label='With Graph')
    axes[1].set_title('Validation PCK')
    axes[1].legend()
    
    plt.savefig('comparison.png')
    
    # Print summary
    print(f"Baseline final PCK: {baseline_metrics['val_pck'][-1]:.2f}%")
    print(f"Graph final PCK: {experiment_metrics['val_pck'][-1]:.2f}%")
    print(f"Improvement: {experiment_metrics['val_pck'][-1] - baseline_metrics['val_pck'][-1]:.2f}%")
```

### 2. Skeleton Visualization

```python
# scripts/visualize_skeleton.py

import networkx as nx
import matplotlib.pyplot as plt

def visualize_skeleton(coords, skeleton, save_path):
    """Visualize pose skeleton on image."""
    
    # Create graph
    G = nx.Graph()
    G.add_edges_from(skeleton)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw edges
    for src, dst in skeleton:
        ax.plot(
            [coords[src, 0], coords[dst, 0]],
            [coords[src, 1], coords[dst, 1]],
            'b-', linewidth=2
        )
    
    # Draw keypoints
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    
    # Labels
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=12, ha='center', va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.savefig(save_path)
```

### 3. Debugging Helper

```python
# scripts/debug_graph_encoding.py

def debug_graph_encoding(model, batch):
    """Debug helper to print shapes and values at each step."""
    
    print("=" * 80)
    print("DEBUGGING GRAPH ENCODING")
    print("=" * 80)
    
    # Input shapes
    print(f"\n[INPUT]")
    print(f"  Query images: {batch['query_images'].shape}")
    print(f"  Support coords: {batch['support_coords'].shape}")
    print(f"  Support mask: {batch['support_mask'].shape}")
    print(f"  Skeleton: {len(batch['skeleton'])} samples")
    print(f"    Sample 0: {len(batch['skeleton'][0])} edges")
    
    # Forward with hooks
    activations = {}
    
    def hook(name):
        def fn(module, input, output):
            activations[name] = output
        return fn
    
    # Register hooks
    model.support_encoder.register_forward_hook(hook('support_encoder'))
    if hasattr(model, 'graph_layers'):
        for i, layer in enumerate(model.graph_layers):
            layer.register_forward_hook(hook(f'graph_layer_{i}'))
    
    # Forward
    output = model(
        batch['query_images'],
        batch['support_coords'],
        batch['support_mask'],
        batch['skeleton'],
        targets=None
    )
    
    # Print activations
    print(f"\n[ACTIVATIONS]")
    for name, act in activations.items():
        if isinstance(act, torch.Tensor):
            print(f"  {name}: {act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")
    
    print(f"\n[OUTPUT]")
    print(f"  Predictions: {output.shape}")
    print(f"  Min: {output.min():.4f}, Max: {output.max():.4f}")
    
    # Check for issues
    if torch.isnan(output).any():
        print(f"  ❌ WARNING: NaNs detected in output!")
    if output.std() < 0.01:
        print(f"  ⚠️ WARNING: Very low variance in predictions")
    
    print("=" * 80)
```

---

## Commit Strategy

### Week 1 Commits

```bash
# Day 1
git add models/graph_utils.py
git commit -m "feat: add adj_from_skeleton() from CapeX"

# Day 2
git add models/graph_utils.py tests/test_graph_utils.py
git commit -m "feat: add GCNLayer and unit tests"

# Day 3
git add models/position_encoding.py tests/test_position_encoding.py
git commit -m "feat: add SinePositionalEncoding.forward_coordinates()"

# Day 4
git add models/support_encoder.py tests/test_support_encoder.py
git commit -m "feat: implement GeometricSupportEncoder"

# Day 5
git add INTEGRATION_CHECKLIST.md
git commit -m "docs: create integration checklist"
```

### Week 2 Commits

```bash
# Day 6
git add models/cape_model.py
git commit -m "feat: integrate GeometricSupportEncoder into CAPEModel"

# Day 7
git add models/cape_model.py
git commit -m "feat: add GCN to decoder layers with graph_decoder config"

# Day 8
git add datasets/mp100_cape.py
git commit -m "feat: add skeleton to dataset __getitem__"

# Day 9
git add datasets/episodic_sampler.py
git commit -m "feat: handle skeleton in episodic sampler collation"

# Day 10
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test"
```

### Week 3 Commits

```bash
# Day 11
git add configs/cape_baseline_no_graph.yaml outputs/baseline_no_graph/
git commit -m "exp: baseline training without graph (10 epochs)"

# Day 13
git add configs/cape_with_graph.yaml outputs/with_graph/
git commit -m "exp: training with graph encoding (10 epochs)"

# Day 16
git add outputs/baseline_50ep/ outputs/with_graph_50ep/
git commit -m "exp: extended training (50 epochs), graph shows +7% improvement"
```

**Note**: Use conventional commit format (`feat:`, `fix:`, `docs:`, `test:`, `exp:`)

---

## Environment Setup Checklist

**Before starting**:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed:
  ```bash
  pip install torch torchvision torchaudio
  pip install opencv-python albumentations
  pip install pycocotools matplotlib networkx
  pip install pytest
  ```
- [ ] Dataset downloaded and annotations cleaned
- [ ] GPU available (MPS on M4 or CUDA)
  ```bash
  python -c "import torch; print(torch.backends.mps.is_available())"
  ```

**New dependencies** (if any):
- [ ] NetworkX (for skeleton visualization): `pip install networkx`
- [ ] TensorBoard (for training monitoring): `pip install tensorboard`

---

## Common Issues & Solutions

### Issue 1: Import Error for CapeX Modules

**Error**: `ImportError: cannot import name 'adj_from_skeleton'`

**Cause**: File not created or wrong path

**Solution**:
```bash
# Check file exists
ls -l models/graph_utils.py

# Check function is defined
grep "def adj_from_skeleton" models/graph_utils.py

# Try import in Python
python -c "from models.graph_utils import adj_from_skeleton"
```

### Issue 2: Shape Mismatch in GCN

**Error**: `RuntimeError: size mismatch, m1: [100 x 256], m2: [512 x 1536]`

**Cause**: Adjacency matrix shape doesn't match feature dimensions

**Debug**:
```python
print(f"tgt.shape: {tgt.shape}")  # Should be [num_pts, bs, c]
print(f"adj.shape: {adj.shape}")  # Should be [bs, 2, num_pts, num_pts]
print(f"Expected: adj[bs=?, 2, {tgt.shape[0]}, {tgt.shape[0]}]")
```

**Solution**: Check `num_pts` parameter in `adj_from_skeleton` call.

### Issue 3: NaN in Loss

**Error**: `Loss is NaN at step 10`

**Cause**: Division by zero in adjacency normalization, or gradient explosion

**Debug**:
```python
# In adj_from_skeleton(), add:
print(f"Adjacency sum before norm: {adj.sum(dim=-1)}")
print(f"Any zeros? {(adj.sum(dim=-1) == 0).any()}")

# If zeros exist, check:
print(f"Mask: {mask}")  # Are all keypoints masked?
print(f"Skeleton: {skeleton}")  # Any edges?
```

**Solution**: Add epsilon in normalization:
```python
adj = torch.nan_to_num(adj / (adj.sum(dim=-1, keepdim=True) + 1e-10))
```

### Issue 4: Graph Doesn't Improve Performance

**Symptom**: With graph PCK = Without graph PCK

**Possible causes**:
1. Skeleton edges are wrong (check dataset)
2. GCN layers not actually being used (check `graph_decoder` config)
3. Adjacency is all zeros (print `adj.sum()`)
4. Learning rate too high (GCN needs smaller LR)

**Debug**:
```python
# In forward pass, add:
if skeleton is not None:
    print(f"Graph decoder mode: {self.graph_decoder}")
    print(f"Skeleton edges: {skeleton[0][:5]}...")  # First 5 edges
    print(f"Adjacency sum: {adj.sum()}")  # Should be > 0
    print(f"Adjacency max: {adj.max()}")  # Should be > 0
```

**Solution**: Verify config, check data, try different `graph_decoder` modes.

---

## Performance Benchmarks

### Expected Training Speed (M4 MacBook Pro)

| Configuration | Batch Size | Time per Epoch | GPU Memory |
|---------------|------------|----------------|------------|
| Baseline (no graph) | 2 | ~8 min | ~4 GB |
| With graph (pre) | 2 | ~10 min | ~5 GB |
| With graph (both) | 2 | ~12 min | ~6 GB |
| Graph + Pre-enc | 2 | ~14 min | ~7 GB |

**If memory is limited**:
- Reduce batch_size to 1
- Increase accumulation_steps to 8
- Disable graph pre-encoding (use only decoder GCN)

### Expected Inference Speed

| Configuration | FPS (batch=1) | FPS (batch=4) |
|---------------|---------------|---------------|
| Baseline | ~15 FPS | ~50 FPS |
| With graph | ~12 FPS | ~40 FPS |

**Graph adds ~20% overhead** (acceptable for +5-10% accuracy)

---

## Rollback Plan

**If integration fails or performance is worse**:

### Step 1: Identify Issue
```bash
# Compare logs
diff outputs/baseline/train.log outputs/with_graph/train.log

# Check for errors
grep "ERROR\|NaN\|failed" outputs/with_graph/train.log
```

### Step 2: Disable Components
```yaml
# Try minimal config
model:
  graph_decoder: null  # Disable GCN
  use_graph_preencoding: false  # Disable pre-encoding

# Or
model:
  graph_decoder: 'pre'  # Keep GCN
  use_graph_preencoding: false  # Disable pre-encoding only
```

### Step 3: Revert Code
```bash
# Revert to before integration
git log --oneline  # Find commit before integration
git checkout <commit_hash>

# Or create feature branch
git checkout -b capex-integration
# ... make changes ...
# If it doesn't work:
git checkout main  # Go back to working version
```

### Step 4: Incremental Debugging

**Test in isolation**:
1. Test `adj_from_skeleton()` alone (unit test)
2. Test `GCNLayer` alone (unit test)
3. Test `GeometricSupportEncoder` alone (integration test)
4. Test full model without training (forward pass only)
5. Test training for 1 iteration (check gradients)

**Identify bottleneck, fix, retry.**

---

## Success Indicators

### After Week 1
- ✅ 3 utility functions ported and tested
- ✅ Unit tests pass
- ✅ No import errors

### After Week 2
- ✅ Model accepts geometric support
- ✅ GCN integrated in decoder
- ✅ Dataset provides skeleton
- ✅ Forward pass works end-to-end

### After Week 3
- ✅ Model trains for 50 epochs
- ✅ Loss decreases (not stuck)
- ✅ PCK > 40% on validation
- ✅ Graph shows measurable benefit (+3-10%)

### After Week 4
- ✅ Code is clean and documented
- ✅ Best hyperparameters found
- ✅ Test set evaluation complete
- ✅ Ready for use/publication

---

## FAQs

**Q1: Do I need to understand graph theory to implement this?**

A: No! The graph operations are simple:
- Adjacency matrix = "which keypoints are connected"
- GCN = "aggregate features from neighbors"
- You can treat it as a black box and just follow the implementation

**Q2: Can I skip the graph pre-encoding and just use GCN in decoder?**

A: Yes! Start with:
- `use_graph_preencoding: false`
- `graph_decoder: 'pre'`

This is simpler and likely gives 80% of the benefit.

**Q3: What if I don't have skeleton annotations?**

A: Two options:
1. Use fully-connected graph (all keypoints connect to all)
2. Use distance-based edges (connect nearby keypoints)

But skeleton is better if available!

**Q4: Can I use this with other datasets (not MP-100)?**

A: Yes! As long as your dataset provides:
- Keypoint coordinates
- Skeleton edges (or you can construct them)

The code is dataset-agnostic.

**Q5: Will this work for single-category pose estimation (not category-agnostic)?**

A: Yes! The graph encoding helps for any pose estimation task.
- Category-agnostic: Train on cats, test on dogs
- Single-category: Train and test on humans only

Both benefit from graph structure.

---

## Contact & Support

**If you encounter issues**:

1. Check `CAPEX_CODE_SNIPPETS.md` for exact code to port
2. Check `CAPEX_ARCHITECTURE_DIAGRAM.md` for visual reference
3. Check `CAPEX_VS_OUR_APPROACH.md` for comparison
4. Read CapeX paper (in `capex-code/` folder)
5. Consult CapeX GitHub: https://github.com/MR-hyj/CapeX

**Debugging checklist**:
- [ ] Print all tensor shapes in forward pass
- [ ] Check for NaNs at each step
- [ ] Verify skeleton format (0-indexed, list of lists)
- [ ] Test with `batch_size=1` first
- [ ] Disable graph to isolate issue

---

## Appendix: Configuration Template

```yaml
# configs/cape_geometric_graph.yaml

# Experiment name
name: "cape_geometric_with_graph"
description: "CapeX graph encoding with geometric support (no text)"

# Model architecture
model:
  name: "CAPEModel"
  
  # Encoder
  image_backbone: "resnet50"  # or "swin_v2_tiny"
  backbone_pretrained: true
  
  # Support encoding (NEW)
  support_encoder:
    type: "geometric"  # Not text!
    hidden_dim: 256
    use_graph_preencoding: false  # Start simple
    num_graph_layers: 2
  
  # Transformer
  transformer:
    num_encoder_layers: 3
    num_decoder_layers: 3
    num_heads: 8
    hidden_dim: 256
    feedforward_dim: 768
    dropout: 0.1
  
  # Graph encoding (CapeX contribution)
  graph_decoder: 'pre'  # Options: null, 'pre', 'post', 'both'
  
  # Prediction
  prediction_type: "sequence"  # or "direct" for CapeX-style
  
# Training
training:
  epochs: 50
  batch_size: 2
  accumulation_steps: 4
  learning_rate: 1e-4
  weight_decay: 1e-4
  
  optimizer: "AdamW"
  scheduler: "cosine"
  warmup_epochs: 5
  
  early_stopping:
    patience: 15
    metric: "val_pck"
    mode: "max"

# Data
data:
  dataset: "MP-100"
  dataset_root: "."
  annotation_dir: "data/annotations"
  
  # Episodic sampling
  num_queries_per_episode: 2
  train_episodes_per_epoch: 100
  val_episodes_per_epoch: 50
  
  # Augmentation
  train_augmentation: true
  img_size: 256

# Logging
logging:
  log_interval: 10  # Log every 10 iterations
  save_interval: 5  # Save checkpoint every 5 epochs
  visualize_interval: 5  # Generate visualizations every 5 epochs
  
# Hardware
device: "mps"  # or "cuda" or "cpu"
num_workers: 4
pin_memory: true
```

**Usage**:
```bash
python train_cape_episodic.py --config configs/cape_geometric_graph.yaml
```

---

## Validation Checklist (Before Declaring Success)

**Quantitative**:
- [ ] Validation PCK > 50% (minimum viable)
- [ ] Validation PCK > 60% (good performance)
- [ ] Validation PCK > 70% (excellent for geometry-only)
- [ ] Graph encoding adds +5% PCK vs baseline
- [ ] Training loss converges (plateaus after 30-40 epochs)
- [ ] No overfitting (val PCK tracks train PCK)

**Qualitative**:
- [ ] Predictions look reasonable (not random)
- [ ] Skeleton structure is respected (connected keypoints are coherent)
- [ ] Symmetry is handled (left/right not swapped often)
- [ ] Works on different categories (animals, furniture, clothing)
- [ ] Generalizes to unseen categories (test set)

**Technical**:
- [ ] No NaNs in training
- [ ] Gradients flow through all modules
- [ ] Adjacency matrix is correct (symmetric, normalized)
- [ ] GCN layers actually activate (check with hooks)
- [ ] Support embeddings have variance (not collapsed)

**Code Quality**:
- [ ] All functions documented
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] No linter errors
- [ ] Code is readable

**Reproducibility**:
- [ ] Results consistent across runs (with same seed)
- [ ] Config files saved with checkpoints
- [ ] Randomness controlled (seed set)

---

## Final Checklist: Ready for Production

- [ ] Code reviewed and cleaned
- [ ] Documentation complete
- [ ] Tests passing (unit + integration)
- [ ] Model trained and validated
- [ ] Performance meets expectations
- [ ] Checkpoints saved
- [ ] Visualizations generated
- [ ] Comparison to baselines done
- [ ] README updated
- [ ] Config examples provided

**When all checked**: ✅ **READY FOR USE!**

---

**End of Implementation Roadmap**

