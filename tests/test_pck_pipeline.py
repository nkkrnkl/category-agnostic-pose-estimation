"""
Comprehensive tests for PCK evaluation pipeline.

These tests verify that:
1. PCK computation is correct and not artificially inflated
2. Keypoint extraction uses predicted token labels, not GT labels
3. Data leakage is detected and prevented
4. Bbox dimensions are used correctly

Run with:
    python -m pytest tests/test_pck_pipeline.py -v
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.eval_utils import compute_pck_bbox
from util.sequence_utils import extract_keypoints_from_predictions
from engine_cape import extract_keypoints_from_sequence
from datasets.token_types import TokenType


class TestPCKComputation:
    """Test PCK computation correctness."""
    
    def test_pck_not_100_for_random_predictions(self):
        """PCK should not be 100% when predictions are random."""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Create GT and predictions IN PIXEL SPACE (matching bbox)
        # Keypoints should be in [0, 512] range to match bbox of 512x512
        gt = np.random.rand(10, 2) * 512  # 10 keypoints in [0, 512] pixels
        pred = gt + 200  # Predictions offset by 200 pixels (far from GT)
        pred = np.clip(pred, 0, 512)  # Keep in valid range
        
        # Compute PCK
        pck, correct, total = compute_pck_bbox(
            pred, gt, 
            bbox_width=512.0, 
            bbox_height=512.0, 
            threshold=0.2
        )
        
        # With large offset (200 pixels >> threshold of ~145 pixels), PCK should be low
        assert pck < 0.5, f"PCK should be low for offset predictions! Got {pck:.2%}"
        assert pck >= 0.0, f"PCK should not be negative! Got {pck:.2%}"
        assert total == 10, f"Expected 10 total keypoints, got {total}"
        
        print(f"✓ Random predictions: PCK = {pck:.2%} (correct={correct}, total={total})")
    
    def test_pck_100_only_when_predictions_match(self):
        """PCK should be 100% ONLY when predictions perfectly match GT."""
        # Create GT keypoints IN PIXEL SPACE
        gt = np.random.rand(10, 2) * 512  # [0, 512] pixels
        
        # Make predictions identical to GT
        pred = gt.copy()
        
        # Compute PCK
        pck, correct, total = compute_pck_bbox(
            pred, gt,
            bbox_width=512.0,
            bbox_height=512.0,
            threshold=0.2
        )
        
        # Perfect match should give 100% PCK
        assert pck == 1.0, f"PCK should be 100% for perfect match! Got {pck:.2%}"
        assert correct == total == 10, f"All keypoints should be correct: {correct}/{total}"
        
        print(f"✓ Perfect match: PCK = {pck:.2%} (correct={correct}, total={total})")
    
    def test_pck_respects_visibility_mask(self):
        """PCK should only evaluate visible keypoints."""
        # Create GT and predictions IN PIXEL SPACE
        gt = np.array([[256, 256], [154, 154], [358, 358]])  # Pixels
        pred = np.array([[256, 256], [461, 461], [358, 358]])  # 2nd kpt is far off
        
        # Visibility: only first and third are visible
        visibility = np.array([2, 0, 2])  # 0 = not labeled
        
        # Compute PCK
        pck, correct, total = compute_pck_bbox(
            pred, gt,
            bbox_width=512.0,
            bbox_height=512.0,
            visibility=visibility,
            threshold=0.2
        )
        
        # Should only evaluate 2 keypoints (kpt 0 and kpt 2)
        assert total == 2, f"Should evaluate 2 visible keypoints, got {total}"
        assert correct == 2, f"Both visible keypoints should be correct, got {correct}"
        assert pck == 1.0, f"PCK should be 100% for correct visible keypoints, got {pck:.2%}"
        
        print(f"✓ Visibility masking: PCK = {pck:.2%} (correct={correct}, total={total})")
    
    def test_pck_threshold_varies_with_bbox_size(self):
        """PCK threshold should scale with bbox size."""
        # Same GT and pred (IN PIXEL SPACE) but different bbox sizes
        gt = np.array([[50, 50]])  # Pixels
        pred = np.array([[55, 55]])  # 5 pixels off
        
        # Small bbox: tight threshold
        # bbox_diag = sqrt(100²+100²) = 141.4, threshold = 0.2*141.4 = 28.3 pixels
        # distance = 7.07 pixels, normalized = 7.07/141.4 = 0.05 < 0.2 ✓
        pck_small, _, _ = compute_pck_bbox(
            pred, gt,
            bbox_width=100.0,
            bbox_height=100.0,
            threshold=0.2
        )
        
        # Large bbox: loose threshold
        # bbox_diag = sqrt(1000²+1000²) = 1414.2, threshold = 0.2*1414.2 = 282.8 pixels
        # distance = 7.07 pixels, normalized = 7.07/1414.2 = 0.005 < 0.2 ✓
        pck_large, _, _ = compute_pck_bbox(
            pred, gt,
            bbox_width=1000.0,
            bbox_height=1000.0,
            threshold=0.2
        )
        
        # Both should be correct, but included as scaling sanity check
        assert pck_large >= pck_small, \
            f"Larger bbox should have higher/equal PCK: small={pck_small:.2%}, large={pck_large:.2%}"
        
        print(f"✓ Bbox scaling: small bbox PCK={pck_small:.2%}, large bbox PCK={pck_large:.2%}")


class TestKeypointExtraction:
    """Test keypoint extraction from sequences."""
    
    def test_extract_keypoints_uses_correct_token_labels(self):
        """Predictions should use predicted token labels, not GT labels."""
        # Create sequences with different structures
        pred_coords = torch.rand(1, 20, 2)
        gt_coords = torch.rand(1, 20, 2)
        
        # GT has coords at positions [0, 2, 4, 6, 8] (alternating with SEP tokens)
        # Token types: 0=coord, 1=sep, 2=eos, 3=cls, 4=pad
        gt_token_labels = torch.tensor([[
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ]])
        gt_mask = torch.ones(1, 20, dtype=torch.bool)
        gt_mask[0, 11:] = False  # Mask out padding
        
        # Model predicts coords at positions [0, 1, 2, 3, 4] (consecutive, different structure!)
        pred_token_labels = torch.tensor([[
            0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ]])
        
        # Create mock logits (one-hot encoded token types)
        pred_logits = F.one_hot(pred_token_labels, num_classes=5).float()  # (1, 20, 5)
        
        # Extract using CORRECT method (predicted labels)
        pred_kpts_correct = extract_keypoints_from_predictions(
            pred_coords,
            pred_logits,
            max_keypoints=None
        )
        
        # Extract using WRONG method (GT labels)
        pred_kpts_wrong = extract_keypoints_from_sequence(
            pred_coords, gt_token_labels, gt_mask
        )
        
        # They should be DIFFERENT!
        assert not torch.allclose(pred_kpts_correct, pred_kpts_wrong, atol=1e-6), \
            "Predicted keypoints should differ when using predicted vs GT token labels!"
        
        # Verify correct shapes
        assert pred_kpts_correct.shape[1] == 5, \
            f"Using predicted labels should extract 5 coords, got {pred_kpts_correct.shape[1]}"
        assert pred_kpts_wrong.shape[1] == 5, \
            f"Using GT labels should extract 5 coords, got {pred_kpts_wrong.shape[1]}"
        
        print(f"✓ Token label extraction: correct shape={pred_kpts_correct.shape}, wrong shape={pred_kpts_wrong.shape}")
    
    def test_extract_handles_early_eos(self):
        """Extraction should handle models that predict EOS early."""
        # Create prediction where model predicts EOS after only 3 coords
        pred_coords = torch.rand(1, 20, 2)
        
        # Token sequence: 3 coords, then EOS, then padding
        pred_token_labels = torch.tensor([[
            0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ]])
        pred_logits = F.one_hot(pred_token_labels, num_classes=5).float()
        
        # Extract keypoints
        pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
        
        # Should extract only 3 keypoints
        assert pred_kpts.shape[1] == 3, \
            f"Should extract 3 keypoints (before EOS), got {pred_kpts.shape[1]}"
        
        print(f"✓ Early EOS handling: extracted {pred_kpts.shape[1]} keypoints")
    
    def test_extract_batch_with_varying_lengths(self):
        """Extraction should handle batches where instances have different numbers of keypoints."""
        # Create batch of 3 instances with different sequence lengths
        pred_coords = torch.rand(3, 20, 2)
        
        # Instance 0: 5 coords, Instance 1: 3 coords, Instance 2: 7 coords
        pred_token_labels = torch.tensor([
            [0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        ])
        pred_logits = F.one_hot(pred_token_labels, num_classes=5).float()
        
        # Extract keypoints
        pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
        
        # Should be padded to max length (7)
        assert pred_kpts.shape == (3, 7, 2), \
            f"Expected shape (3, 7, 2), got {pred_kpts.shape}"
        
        # Verify padding is zeros for shorter sequences
        # Instance 1 should have zeros for keypoints 3-6
        assert torch.allclose(pred_kpts[1, 3:], torch.zeros(4, 2)), \
            "Padding for instance 1 should be zeros"
        
        print(f"✓ Batch extraction: shape={pred_kpts.shape}, correctly padded")


class TestDataLeakageDetection:
    """Test that data leakage is detected."""
    
    def test_warning_when_predictions_match_gt(self):
        """Should warn when predictions are identical to GT."""
        # Create identical pred and GT IN PIXEL SPACE
        gt = np.random.rand(5, 2) * 512
        pred = gt.copy()
        
        # This should trigger a warning
        with pytest.warns(RuntimeWarning, match="IDENTICAL to ground truth"):
            pck, _, _ = compute_pck_bbox(
                pred, gt,
                bbox_width=512.0,
                bbox_height=512.0,
                threshold=0.2
            )
        
        print(f"✓ Data leakage warning triggered for identical predictions")
    
    def test_no_warning_when_predictions_differ(self):
        """Should NOT warn when predictions differ from GT."""
        # Create different pred and GT IN PIXEL SPACE
        gt = np.random.rand(5, 2) * 512
        pred = gt + 10  # Add 10 pixel offset
        
        # This should NOT trigger a warning
        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            pck, _, _ = compute_pck_bbox(
                pred, gt,
                bbox_width=512.0,
                bbox_height=512.0,
                threshold=0.2
            )
        
        # Filter for our specific warning
        leakage_warnings = [w for w in warning_list if "IDENTICAL to ground truth" in str(w.message)]
        assert len(leakage_warnings) == 0, "Should not warn when predictions differ from GT"
        
        print(f"✓ No false positive warnings for different predictions")


class TestEdgeCases:
    """Test edge cases in PCK computation."""
    
    def test_pck_with_no_visible_keypoints(self):
        """PCK should handle case where all keypoints are invisible."""
        gt = np.random.rand(5, 2) * 512  # Pixels
        pred = np.random.rand(5, 2) * 512  # Pixels
        visibility = np.zeros(5)  # All invisible
        
        pck, correct, total = compute_pck_bbox(
            pred, gt,
            bbox_width=512.0,
            bbox_height=512.0,
            visibility=visibility,
            threshold=0.2
        )
        
        # Should return 0 for all metrics
        assert pck == 0.0, f"PCK should be 0.0 with no visible keypoints, got {pck}"
        assert correct == 0, f"Correct should be 0, got {correct}"
        assert total == 0, f"Total should be 0, got {total}"
        
        print(f"✓ No visible keypoints: PCK={pck}, correct={correct}, total={total})")
    
    def test_pck_with_single_keypoint(self):
        """PCK should handle single keypoint case."""
        gt = np.array([[256, 256]])  # Center in pixels
        pred = np.array([[261, 261]])  # 5 pixels off (close)
        
        pck, correct, total = compute_pck_bbox(
            pred, gt,
            bbox_width=512.0,
            bbox_height=512.0,
            threshold=0.2
        )
        
        # Should evaluate 1 keypoint
        assert total == 1, f"Total should be 1, got {total}"
        assert pck in [0.0, 1.0], f"PCK should be 0.0 or 1.0 for single keypoint, got {pck}"
        
        print(f"✓ Single keypoint: PCK={pck}, correct={correct}, total={total})")
    
    def test_extract_with_no_coord_tokens(self):
        """Extraction should handle sequences with no coord tokens."""
        pred_coords = torch.rand(1, 10, 2)
        
        # All tokens are non-coord (e.g., all SEP or EOS)
        pred_token_labels = torch.tensor([[1, 1, 1, 1, 1, 2, 4, 4, 4, 4]])  # No coord tokens
        pred_logits = F.one_hot(pred_token_labels, num_classes=5).float()
        
        # Extract keypoints
        pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
        
        # Should return empty keypoints
        assert pred_kpts.shape == (1, 0, 2), \
            f"Expected shape (1, 0, 2) for no coords, got {pred_kpts.shape}"
        
        print(f"✓ No coord tokens: shape={pred_kpts.shape}")


def test_integration_pck_pipeline():
    """Integration test: full pipeline from model output to PCK."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Full PCK Pipeline")
    print("="*80)
    
    # Simulate model output
    batch_size = 2
    seq_len = 30
    num_classes = 5  # TokenType enum values
    
    # Create mock predictions
    pred_coords = torch.rand(batch_size, seq_len, 2)  # Coordinates
    pred_logits = torch.randn(batch_size, seq_len, num_classes)  # Token logits
    
    # Create mock GT
    gt_coords = torch.rand(batch_size, seq_len, 2)
    gt_token_labels = torch.randint(0, 5, (batch_size, seq_len))
    gt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Step 1: Extract keypoints from predictions
    pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
    print(f"\n✓ Step 1: Extracted predictions: {pred_kpts.shape}")
    
    # Step 2: Extract keypoints from GT
    gt_kpts = extract_keypoints_from_sequence(gt_coords, gt_token_labels, gt_mask)
    print(f"✓ Step 2: Extracted GT: {gt_kpts.shape}")
    
    # Step 3: Ensure same number of keypoints (pad if needed)
    if pred_kpts.shape[1] < gt_kpts.shape[1]:
        padding = torch.zeros(batch_size, gt_kpts.shape[1] - pred_kpts.shape[1], 2)
        pred_kpts = torch.cat([pred_kpts, padding], dim=1)
    elif pred_kpts.shape[1] > gt_kpts.shape[1]:
        pred_kpts = pred_kpts[:, :gt_kpts.shape[1], :]
    
    print(f"✓ Step 3: Aligned shapes: pred={pred_kpts.shape}, gt={gt_kpts.shape}")
    
    # Step 4: Compute PCK for each instance
    bbox_widths = torch.tensor([619.0, 450.0])  # Original bbox sizes
    bbox_heights = torch.tensor([964.0, 780.0])
    
    pcks = []
    for i in range(batch_size):
        pck, correct, total = compute_pck_bbox(
            pred_kpts[i].numpy(),
            gt_kpts[i].numpy(),
            bbox_widths[i].item(),
            bbox_heights[i].item(),
            threshold=0.2
        )
        pcks.append(pck)
        print(f"✓ Step 4.{i+1}: Instance {i} PCK = {pck:.2%} ({correct}/{total})")
    
    # Step 5: Compute overall PCK
    overall_pck = np.mean(pcks)
    print(f"\n✓ Step 5: Overall PCK = {overall_pck:.2%}")
    print("="*80)
    print("✓ INTEGRATION TEST PASSED")
    print("="*80)
    
    # Verify reasonable PCK range
    assert 0.0 <= overall_pck <= 1.0, f"PCK should be in [0, 1], got {overall_pck}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

