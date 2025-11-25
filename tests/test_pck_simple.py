#!/usr/bin/env python3
"""
Simple test of PCK@bbox logic without dependencies.
Tests the mathematical formulas and edge cases.
"""

import math


def compute_pck_simple(pred_kpts, gt_kpts, bbox_w, bbox_h, threshold=0.2):
    """
    Simple PCK computation for testing (no numpy/torch).
    
    Args:
        pred_kpts: List of [x, y] predictions
        gt_kpts: List of [x, y] ground truth
        bbox_w: Bbox width
        bbox_h: Bbox height
        threshold: Distance threshold
    
    Returns:
        pck: PCK score
        correct: Number of correct keypoints
        total: Total keypoints
    """
    assert len(pred_kpts) == len(gt_kpts), "Mismatched keypoint counts"
    
    # Compute bbox diagonal
    bbox_diag = math.sqrt(bbox_w ** 2 + bbox_h ** 2)
    
    correct = 0
    total = len(pred_kpts)
    
    for pred, gt in zip(pred_kpts, gt_kpts):
        # Compute Euclidean distance
        dist = math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        
        # Normalize by bbox diagonal
        norm_dist = dist / bbox_diag
        
        # Check if within threshold
        if norm_dist < threshold:
            correct += 1
    
    pck = correct / total if total > 0 else 0.0
    return pck, correct, total


def test_perfect_predictions():
    """Test with perfect predictions."""
    print("=" * 70)
    print("TEST 1: Perfect Predictions")
    print("=" * 70)
    
    pred = [[0.5, 0.3], [0.2, 0.8], [0.7, 0.6]]
    gt = [[0.5, 0.3], [0.2, 0.8], [0.7, 0.6]]  # Same as pred
    
    pck, correct, total = compute_pck_simple(pred, gt, 512, 512, 0.2)
    
    print(f"Predictions: {pred}")
    print(f"Ground truth: {gt}")
    print(f"Result: PCK@0.2 = {pck:.2%} ({correct}/{total})")
    
    assert pck == 1.0, f"Expected 100%, got {pck:.2%}"
    print("✓ PASSED\n")


def test_bbox_normalization():
    """Test bbox diagonal normalization."""
    print("=" * 70)
    print("TEST 2: Bbox Diagonal Normalization")
    print("=" * 70)
    
    # Non-square bbox
    bbox_w, bbox_h = 1000, 500
    bbox_diag = math.sqrt(bbox_w**2 + bbox_h**2)
    
    print(f"Bbox: {bbox_w} × {bbox_h}")
    print(f"Diagonal: {bbox_diag:.2f}")
    
    # Keypoint with 0.1 offset in normalized coords
    pred = [[0.6, 0.6]]
    gt = [[0.5, 0.5]]
    
    # In pixel coords: [100, 50] offset
    pixel_dist = math.sqrt((0.1 * bbox_w)**2 + (0.1 * bbox_h)**2)
    norm_dist = pixel_dist / bbox_diag
    
    print(f"\nPrediction: {pred[0]}")
    print(f"Ground truth: {gt[0]}")
    print(f"Pixel distance: {pixel_dist:.2f}")
    print(f"Normalized distance: {norm_dist:.4f}")
    print(f"Threshold: 0.2")
    print(f"Within threshold? {norm_dist < 0.2}")
    
    pck, correct, total = compute_pck_simple(pred, gt, bbox_w, bbox_h, 0.2)
    print(f"\nResult: PCK@0.2 = {pck:.2%} ({correct}/{total})")
    
    expected_pck = 1.0 if norm_dist < 0.2 else 0.0
    assert pck == expected_pck
    print("✓ PASSED\n")


def test_threshold_boundary():
    """Test behavior at threshold boundary."""
    print("=" * 70)
    print("TEST 3: Threshold Boundary Cases")
    print("=" * 70)
    
    bbox_w, bbox_h = 512, 512
    bbox_diag = math.sqrt(bbox_w**2 + bbox_h**2)
    threshold = 0.2
    
    # Max allowed distance
    max_allowed_dist = threshold * bbox_diag
    
    print(f"Bbox: {bbox_w} × {bbox_h}")
    print(f"Diagonal: {bbox_diag:.2f}")
    print(f"Threshold: {threshold}")
    print(f"Max allowed pixel distance: {max_allowed_dist:.2f}\n")
    
    # Just within threshold (in normalized coordinates)
    # We want normalized distance = 0.19
    # So pixel distance should be 0.19 * bbox_diag
    # In normalized coords: offset_norm = pixel_offset / bbox_w (for x) or / bbox_h (for y)
    # For simplicity, create offset along diagonal
    offset_norm = 0.19  # Normalized distance we want
    offset_x_pixels = offset_norm * bbox_diag / math.sqrt(2)  # Split equally x and y
    offset_y_pixels = offset_norm * bbox_diag / math.sqrt(2)
    
    pred = [[0.5 + offset_x_pixels/bbox_w, 0.5 + offset_y_pixels/bbox_h]]
    gt = [[0.5, 0.5]]
    pck1, c1, t1 = compute_pck_simple(pred, gt, bbox_w, bbox_h, threshold)
    
    # Verify actual distance
    actual_dist_pixels = math.sqrt(offset_x_pixels**2 + offset_y_pixels**2)
    norm_dist1 = actual_dist_pixels / bbox_diag
    
    print(f"Case 1: Just within threshold")
    print(f"  Distance: {actual_dist_pixels:.2f} pixels")
    print(f"  Normalized: {norm_dist1:.4f}")
    print(f"  PCK: {pck1:.2%}")
    assert pck1 == 1.0, f"Should be within threshold (norm_dist={norm_dist1:.4f})"
    
    # Just outside threshold (use larger margin to avoid floating point issues)
    offset_norm = 0.25  # Clearly above threshold
    offset_x_pixels = offset_norm * bbox_diag / math.sqrt(2)
    offset_y_pixels = offset_norm * bbox_diag / math.sqrt(2)
    
    pred = [[0.5 + offset_x_pixels/bbox_w, 0.5 + offset_y_pixels/bbox_h]]
    gt = [[0.5, 0.5]]
    pck2, c2, t2 = compute_pck_simple(pred, gt, bbox_w, bbox_h, threshold)
    
    actual_dist_pixels = math.sqrt(offset_x_pixels**2 + offset_y_pixels**2)
    norm_dist2 = actual_dist_pixels / bbox_diag
    
    print(f"\nCase 2: Outside threshold")
    print(f"  Distance: {actual_dist_pixels:.2f} pixels")
    print(f"  Normalized: {norm_dist2:.4f}")
    print(f"  PCK: {pck2:.2%}")
    assert pck2 == 0.0, f"Should be outside threshold (norm_dist={norm_dist2:.4f}, threshold={threshold})"
    
    print("\n✓ PASSED\n")


def test_mixed_accuracy():
    """Test with some correct and some incorrect keypoints."""
    print("=" * 70)
    print("TEST 4: Mixed Accuracy")
    print("=" * 70)
    
    # 5 keypoints: 3 correct, 2 incorrect
    pred = [
        [0.50, 0.50],  # Perfect
        [0.30, 0.30],  # Perfect
        [0.70, 0.70],  # Perfect
        [0.20, 0.90],  # Error in y
        [0.90, 0.10]   # Diagonal error
    ]
    
    gt = [
        [0.50, 0.50],
        [0.30, 0.30],
        [0.70, 0.70],
        [0.20, 0.20],  # 0.7 offset
        [0.10, 0.90]   # Large diagonal offset
    ]
    
    pck, correct, total = compute_pck_simple(pred, gt, 512, 512, 0.2)
    
    print(f"Total keypoints: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"PCK@0.2: {pck:.2%}")
    
    # Verify manual calculation
    # First 3 should be correct (perfect match)
    # Last 2 should be incorrect (large errors)
    expected_correct = 3
    expected_pck = 3.0 / 5.0
    
    assert correct == expected_correct, f"Expected {expected_correct} correct, got {correct}"
    assert abs(pck - expected_pck) < 0.01, f"Expected {expected_pck:.2%}, got {pck:.2%}"
    
    print("✓ PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PCK@bbox LOGIC VERIFICATION")
    print("=" * 70 + "\n")
    
    try:
        test_perfect_predictions()
        test_bbox_normalization()
        test_threshold_boundary()
        test_mixed_accuracy()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nVerified:")
        print("  ✓ Euclidean distance computation")
        print("  ✓ Bbox diagonal normalization")
        print("  ✓ Threshold checking")
        print("  ✓ Mixed accuracy scenarios")
        print("  ✓ Boundary cases")
        print("\nImplementation ready for use in CAPE evaluation!")
        print()
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

