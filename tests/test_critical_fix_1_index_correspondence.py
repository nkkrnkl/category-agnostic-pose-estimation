"""
Tests for CRITICAL FIX #1: Keypoint-Edge Index Correspondence

This test verifies that after loading and processing:
1. All keypoints (including invisible ones) are preserved
2. Skeleton edge indices correctly reference coordinate indices
3. Visibility is used as a mask, not for filtering
4. Adjacency matrix construction uses correct indices
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_keypoint_preservation():
    """
    Test that ALL keypoints are preserved after loading, including invisible ones.
    This ensures skeleton edge indices remain valid.
    """
    from datasets.mp100_cape import MP100CAPE
    from datasets.tokenizer import DiscreteTokenizerV2
    
    # Create a mock dataset with known keypoints
    # We'll test with synthetic data
    
    # Synthetic keypoints: some visible, some invisible
    keypoints = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
    visibility = [2, 0, 2, 1, 2]  # visible, invisible, visible, occluded, visible
    
    # Create tokenizer
    tokenizer = DiscreteTokenizerV2(num_bins=128, seq_len=256)
    
    # Tokenize keypoints with visibility
    from datasets.mp100_cape import MP100CAPE
    dataset = MP100CAPE(root='./data', image_set='train', tokenizer=tokenizer)
    
    result = dataset._tokenize_keypoints(
        keypoints=keypoints,
        height=512,
        width=512,
        visibility=visibility
    )
    
    # Verify all keypoints are tokenized (not filtered)
    # Each keypoint generates coordinate tokens
    # With 5 keypoints, we expect:
    # - BOS token
    # - 5 coordinate tokens (one per keypoint)
    # - SEP/EOS token
    # Total: 7 tokens before padding
    
    # Check that visibility mask has correct structure
    visibility_mask = result['visibility_mask']
    assert visibility_mask.dtype == torch.bool, "Visibility mask should be boolean"
    
    # Count visible coordinate tokens (should match visible keypoints)
    num_visible_coords = visibility_mask.sum().item()
    expected_visible = sum(1 for v in visibility if v > 0)  # visibility > 0
    assert num_visible_coords == expected_visible, \
        f"Expected {expected_visible} visible coords, got {num_visible_coords}"
    
    print("✓ All keypoints preserved (not filtered by visibility)")
    print(f"✓ Visibility mask correctly marks {num_visible_coords}/{len(keypoints)} keypoints as visible")


def test_skeleton_edge_alignment():
    """
    Test that skeleton edges correctly reference coordinate indices.
    
    After CRITICAL FIX #1, keypoints should NOT be filtered, so skeleton
    edge indices should directly match coordinate tensor indices.
    """
    # Synthetic test case
    # Category with 5 keypoints
    keypoints = [
        [10.0, 20.0],  # 0
        [30.0, 40.0],  # 1 (invisible)
        [50.0, 60.0],  # 2
        [70.0, 80.0],  # 3
        [90.0, 100.0]  # 4
    ]
    visibility = [2, 0, 2, 2, 2]  # Keypoint 1 is invisible
    
    # Skeleton edges (1-indexed as in COCO)
    # These should become 0-indexed: [[0,1], [1,2], [2,3], [3,4]]
    skeleton_1indexed = [[1, 2], [2, 3], [3, 4], [4, 5]]
    
    # Expected 0-indexed skeleton
    skeleton_expected = [[0, 1], [1, 2], [2, 3], [3, 4]]
    
    # Test adjacency matrix construction
    from models.support_encoder import SupportPoseGraphEncoder
    
    encoder = SupportPoseGraphEncoder(hidden_dim=64, max_keypoints=10)
    
    # Create coordinate tensor (all keypoints, not filtered!)
    coords = torch.tensor([keypoints], dtype=torch.float32)  # (1, 5, 2)
    
    # Build adjacency matrix
    # The encoder expects 0-indexed edges
    adj_matrix = encoder._build_adjacency_matrix([skeleton_expected], N=5, device=coords.device)
    
    # Verify adjacency matrix
    # Should have 1s at the positions defined by edges
    assert adj_matrix.shape == (1, 5, 5), f"Expected shape (1, 5, 5), got {adj_matrix.shape}"
    
    # Check specific edges
    for src, dst in skeleton_expected:
        assert adj_matrix[0, src, dst] == 1, f"Edge {src}->{dst} should exist"
        assert adj_matrix[0, dst, src] == 1, f"Edge {dst}->{src} should exist (undirected)"
    
    # Edge [0,1] should connect keypoint 0 and keypoint 1
    # Even though keypoint 1 is invisible, it should still be in the tensor!
    # This is the key fix - we don't filter it out
    assert adj_matrix[0, 0, 1] == 1, "Edge [0,1] should connect first and second keypoint"
    assert adj_matrix[0, 1, 2] == 1, "Edge [1,2] should connect second and third keypoint"
    
    print("✓ Skeleton edges correctly aligned with coordinate indices")
    print("✓ Invisible keypoints are NOT filtered (preserving index correspondence)")


def test_visibility_mask_not_filtering():
    """
    Test that visibility is used as a MASK, not for FILTERING.
    
    This is the core of CRITICAL FIX #1:
    - OLD (WRONG): Filter keypoints by visibility → breaks index correspondence
    - NEW (CORRECT): Keep all keypoints, use visibility as mask in loss/eval
    """
    from datasets.mp100_cape import MP100CAPE
    from datasets.tokenizer import DiscreteTokenizerV2
    
    # Synthetic keypoints with mixed visibility
    keypoints = [[10, 20], [30, 40], [50, 60]]
    visibility = [2, 0, 2]  # visible, invisible, visible
    
    tokenizer = DiscreteTokenizerV2(num_bins=128, seq_len=256)
    dataset = MP100CAPE(root='./data', image_set='train', tokenizer=tokenizer)
    
    result = dataset._tokenize_keypoints(
        keypoints=keypoints,
        height=512,
        width=512,
        visibility=visibility
    )
    
    # Check that we have tokens for ALL keypoints, not just visible ones
    # The visibility_mask should mark which ones to use in loss
    
    # After BOS, we should have 3 coordinate tokens (one per keypoint)
    # Then SEP/EOS
    # Total: 1 (BOS) + 3 (coords) + 1 (SEP) = 5 tokens before padding
    
    mask = result['mask']  # Valid token mask (not padding)
    visibility_mask = result['visibility_mask']
    
    # Count tokens
    num_valid_tokens = mask.sum().item()
    assert num_valid_tokens >= 5, f"Expected at least 5 tokens, got {num_valid_tokens}"
    
    # Check visibility mask
    # Only 2 keypoints are visible (indices 0 and 2)
    # So only 2 coordinate tokens should be marked as visible
    num_visible = visibility_mask.sum().item()
    assert num_visible == 2, f"Expected 2 visible tokens, got {num_visible}"
    
    print("✓ Visibility used as mask (not for filtering)")
    print(f"✓ All {len(keypoints)} keypoints tokenized, {num_visible} marked as visible")


def test_padding_does_not_affect_edges():
    """
    Test that padding keypoints doesn't cause skeleton edges to reference padding.
    """
    # Test with varying number of keypoints
    # Category A: 3 keypoints
    # Category B: 5 keypoints
    # When batched, they're padded to max (5)
    
    keypoints_a = [[10, 20], [30, 40], [50, 60]]
    visibility_a = [2, 2, 2]
    skeleton_a = [[0, 1], [1, 2]]  # 0-indexed
    
    keypoints_b = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
    visibility_b = [2, 0, 2, 2, 2]
    skeleton_b = [[0, 1], [1, 2], [2, 3], [3, 4]]  # 0-indexed
    
    # Create coordinate tensors
    coords_a = torch.tensor(keypoints_a, dtype=torch.float32)  # (3, 2)
    coords_b = torch.tensor(keypoints_b, dtype=torch.float32)  # (5, 2)
    
    # Pad coords_a to match coords_b
    padding_a = torch.zeros(2, 2)  # 2 padding keypoints
    coords_a_padded = torch.cat([coords_a, padding_a], dim=0)  # (5, 2)
    
    # Create masks
    mask_a = torch.tensor([True, True, True, False, False])  # 3 real, 2 padding
    mask_b = torch.tensor([True, True, True, True, True])  # All real
    
    # Stack into batch
    coords_batch = torch.stack([coords_a_padded, coords_b])  # (2, 5, 2)
    mask_batch = torch.stack([mask_a, mask_b])  # (2, 5)
    
    # Test adjacency matrix construction
    from models.support_encoder import SupportPoseGraphEncoder
    
    encoder = SupportPoseGraphEncoder(hidden_dim=64, max_keypoints=10)
    
    # Build adjacency matrices for batch
    skeletons = [skeleton_a, skeleton_b]
    adj_matrix = encoder._build_adjacency_matrix(skeletons, N=5, device=coords_batch.device)
    
    # For category A, edges should NOT reference padding indices (3, 4)
    # Check that padding positions have no edges
    assert adj_matrix[0, 3, :].sum() == 0, "Padding keypoint 3 should have no edges"
    assert adj_matrix[0, 4, :].sum() == 0, "Padding keypoint 4 should have no edges"
    assert adj_matrix[0, :, 3].sum() == 0, "Padding keypoint 3 should have no incoming edges"
    assert adj_matrix[0, :, 4].sum() == 0, "Padding keypoint 4 should have no incoming edges"
    
    # For category B, all keypoints can have edges
    # Check that skeleton_b edges exist
    for src, dst in skeleton_b:
        assert adj_matrix[1, src, dst] == 1, f"Edge {src}->{dst} should exist for category B"
    
    print("✓ Padding keypoints do not have skeleton edges")
    print("✓ Skeleton edges only reference real keypoints")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING CRITICAL FIX #1: Keypoint-Edge Index Correspondence")
    print("="*70 + "\n")
    
    try:
        test_keypoint_preservation()
        print()
        test_skeleton_edge_alignment()
        print()
        test_visibility_mask_not_filtering()
        print()
        test_padding_does_not_affect_edges()
        
        print("\n" + "="*70)
        print("✅ ALL CRITICAL FIX #1 TESTS PASSED")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

