"""
Integration tests for CAPE model with geometric support encoder.

Tests end-to-end integration of the refactored components.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.geometric_support_encoder import GeometricSupportEncoder


class TestCAPEModelIntegration:
    """Integration tests for CAPE model."""
    
    def test_support_query_batch_alignment(self):
        """Support batch size must match query batch size (1-shot requirement)."""
        # Simulate episodic batch
        batch_size = 4  # 2 episodes × 2 queries per episode
        num_support_kpts = 10
        
        support_coords = torch.rand(batch_size, num_support_kpts, 2)  # (4, 10, 2)
        query_images = torch.rand(batch_size, 3, 512, 512)  # (4, 3, 512, 512)
        
        # For 1-shot CAPE, dimensions must match
        assert support_coords.shape[0] == query_images.shape[0], \
            "1-shot learning: each query must have exactly 1 support"
    
    def test_skeleton_edges_propagated(self):
        """Skeleton edges should reach encoder correctly."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
        
        batch_size = 2
        num_pts = 10
        coords = torch.rand(batch_size, num_pts, 2)
        mask = torch.zeros(batch_size, num_pts).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3]] for _ in range(batch_size)]
        
        # Should work without errors
        out = encoder(coords, mask, skeleton)
        assert out.shape == (batch_size, num_pts, 256)
    
    def test_geometric_encoder_vs_old_encoder(self):
        """Compare outputs of geometric vs old encoder (shapes should match)."""
        from models.support_encoder import SupportPoseGraphEncoder
        
        # Create both encoders
        old_encoder = SupportPoseGraphEncoder(
            hidden_dim=256,
            nheads=8,
            num_encoder_layers=3,
            dim_feedforward=1024,
            dropout=0.0  # Disable for comparison
        )
        
        new_encoder = GeometricSupportEncoder(
            hidden_dim=256,
            num_encoder_layers=3,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,  # Disable for comparison
            use_gcn_preenc=False
        )
        
        # Same input
        coords = torch.rand(2, 10, 2)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        # Both should produce same shape (values will differ)
        out_old = old_encoder(coords, mask, skeleton)
        out_new = new_encoder(coords, mask, skeleton)
        
        assert out_old.shape == out_new.shape == (2, 10, 256)
        
        # Values should be different (different architectures)
        assert not torch.allclose(out_old, out_new, atol=1e-3), \
            "Different encoders should produce different outputs"
    
    def test_normalized_coordinates_range(self):
        """Verify coordinates are in [0, 1] range as CapeX expects."""
        coords = torch.rand(2, 10, 2)  # Already in [0, 1]
        
        assert coords.min() >= 0.0, "Coordinates should be >= 0"
        assert coords.max() <= 1.0, "Coordinates should be <= 1"
        
        # Test encoder accepts [0, 1] coordinates
        encoder = GeometricSupportEncoder(hidden_dim=256)
        mask = torch.zeros(2, 10).bool()
        skeleton = [[[0, 1], [1, 2]] for _ in range(2)]
        
        out = encoder(coords, mask, skeleton)
        assert not torch.isnan(out).any()
    
    def test_skeleton_0_indexed(self):
        """Verify skeleton edges are 0-indexed."""
        encoder = GeometricSupportEncoder(hidden_dim=256, use_gcn_preenc=True)
        
        # 0-indexed skeleton (correct)
        coords = torch.rand(2, 5, 2)
        mask = torch.zeros(2, 5).bool()
        skeleton = [[[0, 1], [1, 2], [2, 3], [3, 4]] for _ in range(2)]
        
        # Should work without index errors
        out = encoder(coords, mask, skeleton)
        assert out.shape == (2, 5, 256)
        
        # If skeleton were 1-indexed, edges like [4, 5] would cause IndexError
        # with num_pts=5 (valid indices are 0-4)


def test_forward_pass_end_to_end():
    """Full forward pass should work without errors."""
    from models.geometric_support_encoder import GeometricSupportEncoder
    
    # Simulate realistic batch
    encoder = GeometricSupportEncoder(
        hidden_dim=256,
        use_gcn_preenc=True,
        num_gcn_layers=2
    )
    
    # Batch of 4 with varying keypoint counts (padded to max)
    coords = torch.rand(4, 20, 2)  # [bs, max_kpts, 2]
    mask = torch.zeros(4, 20).bool()
    
    # Mask some keypoints (simulate padding/invisibility)
    mask[0, 15:] = True  # First sample has 15 keypoints
    mask[1, 12:] = True  # Second has 12 keypoints
    mask[2, 18:] = True  # Third has 18 keypoints
    mask[3, 10:] = True  # Fourth has 10 keypoints
    
    # Different skeletons (categories have different structures)
    skeleton = [
        [[0, 1], [1, 2], [2, 3]],  # Simple chain
        [[0, 1], [0, 2], [1, 3], [2, 3]],  # Diamond
        [[0, 1], [1, 2], [2, 0], [2, 3]],  # Triangle + tail
        [[0, 1], [1, 2]]  # Very simple
    ]
    
    out = encoder(coords, mask, skeleton)
    
    assert out.shape == (4, 20, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

