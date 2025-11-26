"""
Unit tests for positional encoding modules.

Tests both 1D sequence positional encoding and 2D spatial positional encoding.
"""

import torch
import pytest
from models.positional_encoding import PositionalEncoding1D, SinePositionalEncoding2D


class TestPositionalEncoding1D:
    """Tests for 1D sequence positional encoding."""
    
    def test_positional_encoding_1d_basic(self):
        """Test basic forward pass with standard dimensions."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        x = torch.rand(2, 10, 256)  # [batch, seq_len, d_model]
        out = pe(x)
        assert out.shape == (2, 10, 256), f"Expected shape (2, 10, 256), got {out.shape}"
    
    def test_positional_encoding_1d_deterministic(self):
        """Test that output is deterministic (dropout=0)."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        x = torch.rand(2, 10, 256)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Output should be deterministic with dropout=0"
    
    def test_positional_encoding_1d_variable_length(self):
        """Test with different sequence lengths."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        x1 = torch.rand(2, 5, 256)
        x2 = torch.rand(2, 20, 256)
        out1 = pe(x1)
        out2 = pe(x2)
        assert out1.shape == (2, 5, 256), f"Expected shape (2, 5, 256), got {out1.shape}"
        assert out2.shape == (2, 20, 256), f"Expected shape (2, 20, 256), got {out2.shape}"
    
    def test_positional_encoding_1d_unique_positions(self):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        # Check that position 0 != position 1
        pe_buffer = pe.pe  # [1, max_len, d_model]
        assert not torch.allclose(pe_buffer[0, 0], pe_buffer[0, 1], atol=1e-3), \
            "Different positions should have different encodings"
    
    def test_positional_encoding_1d_adds_to_input(self):
        """Test that positional encoding is added to input."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        x = torch.zeros(1, 5, 256)  # All zeros
        out = pe(x)
        # Output should not be all zeros (PE was added)
        assert not torch.allclose(out, x, atol=1e-6), "PE should be added to input"
    
    def test_positional_encoding_1d_gradient_flow(self):
        """Test that gradients flow through the module."""
        pe = PositionalEncoding1D(d_model=256, max_len=100, dropout=0.0)
        x = torch.rand(2, 10, 256, requires_grad=True)
        out = pe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow through PE"
        assert x.grad.shape == x.shape, "Gradient shape should match input shape"


class TestSinePositionalEncoding2D:
    """Tests for 2D spatial positional encoding."""
    
    def test_forward_coordinates_basic(self):
        """Test basic forward pass with coordinates."""
        pe = SinePositionalEncoding2D(num_feats=128)  # Will output 256 dims
        coords = torch.rand(2, 10, 2)  # [batch, num_pts, 2] in [0, 1]
        out = pe.forward_coordinates(coords)
        assert out.shape == (2, 10, 256), f"Expected shape (2, 10, 256), got {out.shape}"
    
    def test_forward_coordinates_normalized_input(self):
        """Test that input coordinates are expected to be in [0, 1]."""
        pe = SinePositionalEncoding2D(num_feats=128, normalize=True, scale=2*3.14159)
        coords = torch.rand(2, 10, 2)  # Already in [0, 1]
        # Should work without error
        out = pe.forward_coordinates(coords)
        assert out.shape == (2, 10, 256)
    
    def test_forward_coordinates_deterministic(self):
        """Test that same coordinates give same encoding."""
        pe = SinePositionalEncoding2D(num_feats=128)
        coords = torch.rand(2, 10, 2)
        out1 = pe.forward_coordinates(coords)
        out2 = pe.forward_coordinates(coords)
        assert torch.allclose(out1, out2, atol=1e-6), "Same coords should give same encoding"
    
    def test_forward_coordinates_different_positions(self):
        """Test that different coordinates give different encodings."""
        pe = SinePositionalEncoding2D(num_feats=128)
        coords1 = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        coords2 = torch.tensor([[[0.5, 0.6], [0.7, 0.8]]])
        out1 = pe.forward_coordinates(coords1)
        out2 = pe.forward_coordinates(coords2)
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "Different coordinates should give different encodings"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

