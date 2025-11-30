"""
Unit tests for GeometricSupportEncoder.

Tests the geometry-only support encoder that combines coordinate embeddings,
positional encoding, optional GCN, and transformer self-attention.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.geometric_support_encoder import GeometricSupportEncoder


class TestGeometricSupportEncoder:
    """Test geometry-only support encoder."""
    
    def test_basic_forward(self):
        """Test forward pass with basic inputs (no GCN)."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=False,
            num_encoder_layers=2
        )
        
        coords = torch.rand(2, 10, 2)  # [bs, num_pts, 2] in [0, 1]
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        
        assert out.shape == (2, 10, 256), f"Expected (2, 10, 256), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
    
    def test_forward_with_gcn(self):
        """Test forward pass with GCN pre-encoding enabled."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True,
            num_gcn_layers=2
        )
        
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        
        assert out.shape == (2, 10, 256), f"Expected (2, 10, 256), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
    
    def test_gradient_flow_no_gcn(self):
        """Test gradients flow through encoder (no GCN)."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=False
        )
        
        coords = torch.rand(2, 10, 2, requires_grad=True)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        loss = out.sum()
        loss.backward()
        
        assert coords.grad is not None, "Gradients should flow to input"
        assert torch.any(coords.grad != 0), "Gradients should be non-zero"
        assert encoder.coord_mlp[0].weight.grad is not None, "Gradients should reach MLP"
    
    def test_gradient_flow_with_gcn(self):
        """Test gradients flow through encoder with GCN."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True,
            num_gcn_layers=2
        )
        
        coords = torch.rand(2, 10, 2, requires_grad=True)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        loss = out.sum()
        loss.backward()
        
        assert coords.grad is not None, "Gradients should flow to input"
        assert torch.any(coords.grad != 0), "Gradients should be non-zero"
        assert encoder.coord_mlp[0].weight.grad is not None, "Gradients should reach MLP"
        assert encoder.gcn_layers[0].conv.weight.grad is not None, "Gradients should reach GCN"
    
    def test_with_masked_keypoints(self):
        """Test encoder handles masked keypoints correctly."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True
        )
        
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        mask[0, 5] = True  # Mask one keypoint in first batch
        mask[1, 3] = True  # Mask different keypoint in second batch
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        
        assert out.shape == (2, 10, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Masked keypoint outputs should still exist (they're just masked in attention)
        # They won't be all zeros because coordinate embedding still processes them
        assert out[0, 5].abs().sum() > 0, "Masked keypoint should still have embeddings"
        assert out[1, 3].abs().sum() > 0, "Masked keypoint should still have embeddings"
    
    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [128, 256, 512]:
            encoder = GeometricSupportEncoder(hidden_dim=hidden_dim)
            coords = torch.rand(2, 10, 2)
            mask = torch.zeros(2, 10).bool()
            skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
            
            out = encoder(coords, mask, skeleton)
            assert out.shape == (2, 10, hidden_dim)
    
    def test_variable_num_keypoints(self):
        """Test with different numbers of keypoints."""
        encoder = GeometricSupportEncoder(hidden_dim=256)
        
        for num_pts in [5, 10, 20, 50]:
            coords = torch.rand(2, num_pts, 2)
            mask = torch.zeros(2, num_pts).bool()
            # Create simple chain skeleton
            skeleton = [[[i, i+1] for i in range(num_pts-1)] for _ in range(2)]
            
            out = encoder(coords, mask, skeleton)
            assert out.shape == (2, num_pts, 256)
    
    def test_empty_skeleton(self):
        """Test with no skeleton edges (isolated keypoints)."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True
        )
        
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[] for _ in range(2)]  # No edges
        
        out = encoder(coords, mask, skeleton)
        
        assert out.shape == (2, 10, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_single_batch_element(self):
        """Test with batch size = 1."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
        
        coords = torch.rand(1, 10, 2)
        mask = torch.zeros(1, 10).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]]]
        
        out = encoder(coords, mask, skeleton)
        assert out.shape == (1, 10, 256)
    
    def test_output_different_with_without_gcn(self):
        """Outputs should differ with and without GCN."""
        encoder_no_gcn = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=False,
            num_encoder_layers=2
        )
        encoder_with_gcn = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True,
            num_gcn_layers=2,
            num_encoder_layers=2
        )
        
        # Set same weights for coord_mlp (to isolate GCN effect)
        encoder_with_gcn.coord_mlp.load_state_dict(encoder_no_gcn.coord_mlp.state_dict())
        
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(2)]
        
        out_no_gcn = encoder_no_gcn(coords, mask, skeleton)
        out_with_gcn = encoder_with_gcn(coords, mask, skeleton)
        
        # Outputs should be different (GCN adds graph processing)
        assert not torch.allclose(out_no_gcn, out_with_gcn, atol=1e-3), \
            "GCN should produce different outputs"
    
    def test_deterministic(self):
        """Same input should produce same output (deterministic)."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
        encoder.eval()  # Disable dropout
        
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out1 = encoder(coords, mask, skeleton)
        out2 = encoder(coords, mask, skeleton)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Should be deterministic in eval mode"
    
    # ========================================================================
    # NEW TESTS FOR SEQUENCE POSITIONAL ENCODING
    # ========================================================================
    
    def test_geometric_encoder_with_sequence_pe(self):
        """Test encoder has sequence_pos_encoding module."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=False)
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        assert out.shape == (2, 10, 256)
        
        # Verify sequence_pos_encoding module exists
        assert hasattr(encoder, 'sequence_pos_encoding'), \
            "Encoder should have sequence_pos_encoding attribute"
    
    def test_sequence_pe_affects_output(self):
        """Test that sequence PE makes ordering matter."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=False)
        encoder.eval()  # Disable dropout for deterministic output
        
        # Same coordinates, different order
        coords_ordered = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        ])
        coords_shuffled = torch.tensor([
            [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
        ])
        
        mask = torch.zeros(1, 3).bool()
        skeleton = [[[0, 1], [1, 2]]]
        
        out_ordered = encoder(coords_ordered, mask, skeleton)
        out_shuffled = encoder(coords_shuffled, mask, skeleton)
        
        # Outputs should be DIFFERENT due to sequence PE
        # (spatial PE would be same, but sequence PE differs)
        assert not torch.allclose(out_ordered, out_shuffled, atol=1e-3), \
            "Different orderings should produce different outputs (sequence PE working)"
    
    def test_geometric_encoder_sequence_pe_with_gcn(self):
        """Test sequence PE works with GCN pre-encoding."""
        encoder = GeometricSupportEncoder(
            hidden_dim=256,
            use_gcn_preenc=True,
            num_gcn_layers=2
        )
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        assert out.shape == (2, 10, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_gradient_flow_with_sequence_pe(self):
        """Test gradients flow through sequence PE."""
        encoder = GeometricSupportEncoder(hidden_dim=256)
        coords = torch.rand(2, 10, 2, requires_grad=True)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        loss = out.sum()
        loss.backward()
        
        assert coords.grad is not None, "Gradients should flow to coords"
        assert encoder.coord_mlp[0].weight.grad is not None, "Gradients should reach MLP"
        # Sequence PE has no learnable parameters (sinusoidal), so no grad check needed
    
    def test_sequence_pe_with_masked_keypoints(self):
        """Test sequence PE works correctly with masked keypoints."""
        encoder = GeometricSupportEncoder(hidden_dim=256)
        coords = torch.rand(1, 10, 2)
        mask = torch.zeros(1, 10).bool()
        mask[0, 5] = True  # Mask one keypoint
        skeleton = [[[0, 1], [1, 2]]]
        
        out = encoder(coords, mask, skeleton)
        assert out.shape == (1, 10, 256)
        assert not torch.isnan(out).any()
        # Masked position should have output (sequence PE still applied, but masked in attention)


def test_geometric_encoder_forward():
    """Test forward pass with various inputs."""
    encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=False)
    coords = torch.rand(2, 10, 2)
    mask = torch.zeros(2, 10).bool()
    skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
    
    out = encoder(coords, mask, skeleton)
    assert out.shape == (2, 10, 256)


def test_geometric_encoder_with_gcn():
    """Test encoder with GCN pre-encoding."""
    encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
    coords = torch.rand(2, 10, 2)
    mask = torch.zeros(2, 10).bool()
    skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
    
    out = encoder(coords, mask, skeleton)
    assert out.shape == (2, 10, 256)


def test_geometric_encoder_gradient_flow():
    """Test that gradients flow through all components."""
    encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
    coords = torch.rand(2, 10, 2, requires_grad=True)
    mask = torch.zeros(2, 10).bool()
    skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
    
    out = encoder(coords, mask, skeleton)
    loss = out.sum()
    loss.backward()
    
    assert coords.grad is not None
    assert encoder.coord_mlp[0].weight.grad is not None


def test_geometric_encoder_with_masked_keypoints():
    """Test encoder handles masked keypoints correctly."""
    encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
    coords = torch.rand(2, 10, 2)
    mask = torch.zeros(2, 10).bool()
    mask[0, 5] = True  # Mask one keypoint
    skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
    
    out = encoder(coords, mask, skeleton)
    assert out.shape == (2, 10, 256)
    # Masked keypoint output should still exist (masked in attention)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

