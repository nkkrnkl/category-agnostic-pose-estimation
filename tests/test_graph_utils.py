"""
Unit tests for graph utilities (adjacency matrix and GCN layers).

Tests components copied from CapeX for geometry-only CAPE.
"""

import math
import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.graph_utils import adj_from_skeleton, GCNLayer
from models.positional_encoding import SinePositionalEncoding2D


class TestAdjFromSkeleton:
    """Test adjacency matrix construction from skeleton edges."""
    
    def test_basic_triangle(self):
        """Test adjacency for a simple triangle graph."""
        skeleton = [[[0, 1], [1, 2], [2, 0]]]
        mask = torch.zeros(1, 3).bool()
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        # Check shape
        assert adj.shape == (1, 2, 3, 3), f"Expected (1, 2, 3, 3), got {adj.shape}"
        
        # Channel 0: self-loops (all 1s for unmasked)
        expected_self = torch.eye(3)
        assert torch.allclose(adj[0, 0], expected_self, atol=1e-6)
    
    def test_symmetry(self):
        """Adjacency matrix should be symmetric (undirected graph)."""
        skeleton = [[[0, 1], [1, 2], [0, 2]]]
        mask = torch.zeros(1, 3).bool()
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        # Channel 1 should be symmetric
        assert torch.allclose(adj[0, 1], adj[0, 1].T, atol=1e-6)
    
    def test_normalization(self):
        """Each row should sum to 1 (for unmasked keypoints)."""
        skeleton = [[[0, 1], [1, 2]]]
        mask = torch.tensor([[False, False, False]])
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        # Channel 1: row sums should be 1
        row_sums = adj[0, 1].sum(dim=-1)
        expected = torch.ones(3)
        assert torch.allclose(row_sums, expected, atol=1e-6), \
            f"Row sums: {row_sums}, expected: {expected}"
    
    def test_masked_keypoints(self):
        """Masked keypoints should have zero adjacency."""
        skeleton = [[[0, 1], [1, 2]]]
        mask = torch.tensor([[False, True, False]])  # Mask middle keypoint
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        # Channel 0: masked keypoint should have 0 on diagonal
        assert adj[0, 0, 1, 1] == 0.0, "Masked keypoint should have 0 self-loop"
        
        # Channel 1: row 1 should be all zeros (masked)
        assert torch.allclose(adj[0, 1, 1, :], torch.zeros(3), atol=1e-6)
        
        # Channel 1: column 1 should be all zeros (masked)
        assert torch.allclose(adj[0, 1, :, 1], torch.zeros(3), atol=1e-6)
    
    def test_batch(self):
        """Test with batch size > 1."""
        skeleton = [
            [[0, 1], [1, 2]],
            [[0, 1], [1, 2], [2, 0]]
        ]
        mask = torch.zeros(2, 3).bool()
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        assert adj.shape == (2, 2, 3, 3)
        
        # Channel 0: self-loops should be identity
        assert torch.allclose(adj[0, 0], torch.eye(3), atol=1e-6)
        assert torch.allclose(adj[1, 0], torch.eye(3), atol=1e-6)
        
        # Channel 1: rows should sum to 1
        assert torch.allclose(adj[0, 1].sum(dim=-1), torch.ones(3), atol=1e-6)
        assert torch.allclose(adj[1, 1].sum(dim=-1), torch.ones(3), atol=1e-6)
    
    def test_empty_skeleton(self):
        """Test with no edges (isolated nodes)."""
        skeleton = [[]]  # No edges
        mask = torch.zeros(1, 3).bool()
        adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
        
        # Channel 0: only self-loops
        assert torch.allclose(adj[0, 0], torch.eye(3), atol=1e-6)
        
        # Channel 1: all zeros (no neighbors)
        # Note: rows sum to 0 after normalization, becomes NaN, then 0 via nan_to_num
        assert torch.allclose(adj[0, 1], torch.zeros(3, 3), atol=1e-6)


class TestGCNLayer:
    """Test Graph Convolutional Layer."""
    
    def test_shape_preservation_batch_first(self):
        """GCN should preserve tensor shapes (batch_first=True)."""
        gcn = GCNLayer(128, 128, kernel_size=2, batch_first=True)
        x = torch.rand(2, 10, 128)  # [bs, num_pts, features]
        adj = torch.rand(2, 2, 10, 10)  # [bs, kernel_size, num_pts, num_pts]
        
        out = gcn(x, adj)
        assert out.shape == (2, 10, 128), f"Expected (2, 10, 128), got {out.shape}"
    
    def test_shape_preservation_seq_first(self):
        """GCN should preserve tensor shapes (batch_first=False)."""
        gcn = GCNLayer(128, 128, kernel_size=2, batch_first=False)
        x = torch.rand(10, 2, 128)  # [num_pts, bs, features]
        adj = torch.rand(2, 2, 10, 10)  # [bs, kernel_size, num_pts, num_pts]
        
        out = gcn(x, adj)
        assert out.shape == (10, 2, 128), f"Expected (10, 2, 128), got {out.shape}"
    
    def test_gradient_flow(self):
        """Gradients should flow through GCN layers."""
        gcn = GCNLayer(128, 128, kernel_size=2, batch_first=True)
        x = torch.rand(2, 10, 128, requires_grad=True)
        adj = torch.rand(2, 2, 10, 10)
        
        out = gcn(x, adj)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "Gradients should flow to input"
        assert torch.any(x.grad != 0), "Gradients should be non-zero"
    
    def test_different_in_out_dims(self):
        """GCN should handle different input/output dimensions."""
        gcn = GCNLayer(128, 256, kernel_size=2, batch_first=True)
        x = torch.rand(2, 10, 128)
        adj = torch.rand(2, 2, 10, 10)
        
        out = gcn(x, adj)
        assert out.shape == (2, 10, 256)
    
    def test_kernel_size_3(self):
        """GCN should work with kernel_size=3."""
        gcn = GCNLayer(128, 128, kernel_size=3, batch_first=True)
        x = torch.rand(2, 10, 128)
        adj = torch.rand(2, 3, 10, 10)  # 3 channels
        
        out = gcn(x, adj)
        assert out.shape == (2, 10, 128)
    
    def test_no_activation(self):
        """GCN should work without activation."""
        gcn = GCNLayer(128, 128, kernel_size=2, activation=None, batch_first=True)
        x = torch.rand(2, 10, 128)
        adj = torch.rand(2, 2, 10, 10)
        
        out = gcn(x, adj)
        assert out.shape == (2, 10, 128)


class TestSinePositionalEncoding2D:
    """Test sinusoidal positional encoding for 2D coordinates."""
    
    def test_deterministic(self):
        """Same coordinates should give same encoding."""
        pos_enc = SinePositionalEncoding2D(num_feats=128)
        coords = torch.rand(2, 10, 2)
        
        enc1 = pos_enc.forward_coordinates(coords)
        enc2 = pos_enc.forward_coordinates(coords)
        
        assert torch.allclose(enc1, enc2, atol=1e-6)
    
    def test_different_coords_different_encoding(self):
        """Different coordinates should give different encodings."""
        pos_enc = SinePositionalEncoding2D(num_feats=128)
        coords1 = torch.tensor([[[0.1, 0.2]]])
        coords2 = torch.tensor([[[0.5, 0.7]]])
        
        enc1 = pos_enc.forward_coordinates(coords1)
        enc2 = pos_enc.forward_coordinates(coords2)
        
        assert not torch.allclose(enc1, enc2, atol=1e-3)
    
    def test_output_shape(self):
        """Output shape should be [bs, num_pts, num_feats*2]."""
        pos_enc = SinePositionalEncoding2D(num_feats=128)
        coords = torch.rand(2, 10, 2)
        
        pos = pos_enc.forward_coordinates(coords)
        assert pos.shape == (2, 10, 256), f"Expected (2, 10, 256), got {pos.shape}"
    
    def test_batch_independence(self):
        """Each batch element should be encoded independently."""
        pos_enc = SinePositionalEncoding2D(num_feats=128)
        
        # Create batch where first two elements are identical
        coords = torch.rand(3, 10, 2)
        coords[1] = coords[0]
        
        pos = pos_enc.forward_coordinates(coords)
        
        # First two should be identical
        assert torch.allclose(pos[0], pos[1], atol=1e-6)
        
        # First and third should be different
        assert not torch.allclose(pos[0], pos[2], atol=1e-3)
    
    def test_scale_parameter(self):
        """Different scales should produce different encodings."""
        pos_enc_2pi = SinePositionalEncoding2D(num_feats=128, scale=2*math.pi)
        pos_enc_pi = SinePositionalEncoding2D(num_feats=128, scale=math.pi)
        
        coords = torch.rand(2, 10, 2)
        
        enc_2pi = pos_enc_2pi.forward_coordinates(coords)
        enc_pi = pos_enc_pi.forward_coordinates(coords)
        
        assert not torch.allclose(enc_2pi, enc_pi, atol=1e-3)


class TestIntegration:
    """Integration tests for graph utils + positional encoding."""
    
    def test_gcn_with_real_adjacency(self):
        """Test GCN with adjacency from skeleton."""
        # Create simple skeleton
        skeleton = [[[0, 1], [1, 2], [2, 3]]]
        mask = torch.zeros(1, 4).bool()
        
        # Build adjacency
        adj = adj_from_skeleton(4, skeleton, mask, 'cpu')
        
        # Apply GCN
        gcn = GCNLayer(256, 256, kernel_size=2, batch_first=True)
        x = torch.rand(1, 4, 256)
        out = gcn(x, adj)
        
        assert out.shape == (1, 4, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_complete_pipeline(self):
        """Test complete encoding pipeline: coords → pos enc → GCN."""
        # Setup
        coords = torch.rand(2, 10, 2)  # [bs, num_pts, 2]
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(2)]
        mask = torch.zeros(2, 10).bool()
        
        # 1. Coordinate MLP (simulated)
        coord_mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        coord_emb = coord_mlp(coords)  # [2, 10, 256]
        
        # 2. Positional encoding
        pos_enc = SinePositionalEncoding2D(num_feats=128)
        pos_emb = pos_enc.forward_coordinates(coords)  # [2, 10, 256]
        
        # 3. Combine
        embeddings = coord_emb + pos_emb
        
        # 4. Build adjacency
        adj = adj_from_skeleton(10, skeleton, mask, 'cpu')
        
        # 5. Apply GCN
        gcn = GCNLayer(256, 256, kernel_size=2, batch_first=True)
        out = gcn(embeddings, adj)
        
        # Validate
        assert out.shape == (2, 10, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Test gradients flow end-to-end
        loss = out.sum()
        loss.backward()
        assert coords.grad is None  # coords not requires_grad
        assert coord_mlp[0].weight.grad is not None


def test_adj_from_skeleton_symmetry():
    """Adjacency matrix should be symmetric (undirected graph)."""
    skeleton = [[[0, 1], [1, 2], [0, 2]]]
    mask = torch.zeros(1, 3).bool()
    adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
    assert torch.allclose(adj[0, 1], adj[0, 1].T, atol=1e-6)


def test_adj_from_skeleton_normalization():
    """Each row should sum to 1 (or 0 for masked keypoints)."""
    skeleton = [[[0, 1], [1, 2]]]
    mask = torch.tensor([[False, False, False]])
    adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
    row_sums = adj[0, 1].sum(dim=-1)
    expected = torch.ones(3)
    assert torch.allclose(row_sums, expected, atol=1e-6)


def test_gcn_shape_preservation():
    """GCN should maintain tensor shapes."""
    gcn = GCNLayer(128, 128, kernel_size=2, batch_first=True)
    x = torch.rand(2, 10, 128)
    adj = torch.rand(2, 2, 10, 10)
    out = gcn(x, adj)
    assert out.shape == (2, 10, 128)


def test_gcn_gradient_flow():
    """Gradients should flow through GCN layers."""
    gcn = GCNLayer(128, 128, kernel_size=2, batch_first=True)
    x = torch.rand(2, 10, 128, requires_grad=True)
    adj = torch.rand(2, 2, 10, 10)
    out = gcn(x, adj)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_adj_handles_masked_keypoints():
    """Masked keypoints should have zero adjacency."""
    skeleton = [[[0, 1], [1, 2]]]
    mask = torch.tensor([[False, True, False]])  # Mask middle keypoint
    adj = adj_from_skeleton(3, skeleton, mask, 'cpu')
    # Row 1 should be all zeros (masked)
    assert torch.allclose(adj[0, 1, 1, :], torch.zeros(3))


def test_sine_pos_enc_deterministic():
    """Same coordinates should give same encoding."""
    pos_enc = SinePositionalEncoding2D(num_feats=128)
    coords = torch.rand(2, 10, 2)
    enc1 = pos_enc.forward_coordinates(coords)
    enc2 = pos_enc.forward_coordinates(coords)
    assert torch.allclose(enc1, enc2, atol=1e-6)


def test_sine_pos_enc_different():
    """Different coordinates should give different encodings."""
    pos_enc = SinePositionalEncoding2D(num_feats=128)
    coords1 = torch.tensor([[[0.1, 0.2]]])
    coords2 = torch.tensor([[[0.5, 0.7]]])
    enc1 = pos_enc.forward_coordinates(coords1)
    enc2 = pos_enc.forward_coordinates(coords2)
    assert not torch.allclose(enc1, enc2, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

