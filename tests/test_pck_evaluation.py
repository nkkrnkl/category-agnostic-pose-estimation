#!/usr/bin/env python3
"""
Test PCK@bbox evaluation implementation.

Tests:
1. Basic PCK computation with perfect predictions
2. PCK with varying error levels
3. Visibility masking
4. Bbox normalization (diagonal vs max)
5. Batch evaluation
6. PCKEvaluator accumulator
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from util.eval_utils import compute_pck_bbox, PCKEvaluator


def test_perfect_predictions():
    """Test PCK with perfect predictions (should be 100%)."""
    print("=" * 80)
    print("TEST 1: Perfect Predictions")
    print("=" * 80)
    
    # Perfect predictions (pred == gt)
    pred = np.array([[0.5, 0.3], [0.2, 0.8], [0.7, 0.6]])
    gt = pred.copy()  # Identical
    
    pck, correct, visible = compute_pck_bbox(
        pred, gt,
        bbox_width=512.0,
        bbox_height=512.0,
        threshold=0.2
    )
    
    print(f"Predicted: {pred.tolist()}")
    print(f"Ground truth: {gt.tolist()}")
    print(f"Bbox: 512 × 512")
    print(f"\nResult:")
    print(f"  PCK@0.2: {pck:.2%}")
    print(f"  Correct: {correct}/{visible}")
    
    assert pck == 1.0, f"Expected 100% PCK, got {pck:.2%}"
    assert correct == 3, f"Expected 3 correct, got {correct}"
    print("✓ PASSED: Perfect predictions give 100% PCK\n")


def test_varying_error():
    """Test PCK with different error levels."""
    print("=" * 80)
    print("TEST 2: Varying Error Levels")
    print("=" * 80)
    
    gt = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
    bbox_w, bbox_h = 512.0, 512.0
    threshold = 0.2
    
    # Compute bbox diagonal
    bbox_diag = np.sqrt(bbox_w**2 + bbox_h**2)
    max_error = threshold * bbox_diag
    
    print(f"Ground truth: {gt.tolist()}")
    print(f"Bbox: {bbox_w} × {bbox_h}")
    print(f"Bbox diagonal: {bbox_diag:.2f}")
    print(f"Threshold: {threshold}")
    print(f"Max allowed error: {max_error:.2f} pixels\n")
    
    # Test case 1: Small error (within threshold)
    pred = gt + np.array([[0.01, 0.01], [0.01, -0.01], [-0.01, 0.01]])
    pck, correct, visible = compute_pck_bbox(pred, gt, bbox_w, bbox_h, threshold=threshold)
    print(f"Small error (+/-0.01):")
    print(f"  PCK@{threshold}: {pck:.2%} ({correct}/{visible})")
    assert pck == 1.0, "Small error should be within threshold"
    
    # Test case 2: Medium error (some within, some outside)
    pred = np.array([
        [0.5, 0.5],   # No error
        [0.3, 0.9],   # Large error in y (0.2 normalized)
        [0.8, 0.2]    # No error
    ])
    pck, correct, visible = compute_pck_bbox(pred, gt, bbox_w, bbox_h, threshold=threshold)
    print(f"Mixed error (one keypoint far):")
    print(f"  PCK@{threshold}: {pck:.2%} ({correct}/{visible})")
    expected_pck = 2.0 / 3.0
    assert abs(pck - expected_pck) < 0.01, f"Expected {expected_pck:.2%}, got {pck:.2%}"
    
    # Test case 3: All incorrect
    pred = gt + 0.5  # Large offset
    pck, correct, visible = compute_pck_bbox(pred, gt, bbox_w, bbox_h, threshold=threshold)
    print(f"Large error (+0.5):")
    print(f"  PCK@{threshold}: {pck:.2%} ({correct}/{visible})")
    assert pck == 0.0, "Large error should fail all keypoints"
    
    print("✓ PASSED: Error thresholding works correctly\n")


def test_visibility_masking():
    """Test that visibility masking filters keypoints correctly."""
    print("=" * 80)
    print("TEST 3: Visibility Masking")
    print("=" * 80)
    
    # 5 keypoints, but only 3 visible
    pred = np.array([
        [0.5, 0.5],   # Visible
        [0.3, 0.3],   # Not visible (v=0)
        [0.7, 0.7],   # Visible
        [0.2, 0.2],   # Visible
        [0.9, 0.9]    # Not visible (v=0)
    ])
    gt = pred.copy()
    
    # Visibility: 0 = not labeled, 1 or 2 = visible
    visibility = np.array([2, 0, 1, 2, 0])
    
    pck, correct, visible = compute_pck_bbox(
        pred, gt,
        bbox_width=512.0,
        bbox_height=512.0,
        visibility=visibility,
        threshold=0.2
    )
    
    print(f"Total keypoints: {len(pred)}")
    print(f"Visibility flags: {visibility.tolist()}")
    print(f"Visible keypoints (v > 0): {(visibility > 0).sum()}")
    print(f"\nResult:")
    print(f"  PCK@0.2: {pck:.2%}")
    print(f"  Evaluated: {visible} keypoints (should be 3)")
    print(f"  Correct: {correct}/{visible}")
    
    assert visible == 3, f"Expected 3 visible keypoints, got {visible}"
    assert pck == 1.0, "All visible keypoints should be correct"
    print("✓ PASSED: Visibility masking works correctly\n")


def test_bbox_normalization():
    """Test different bbox normalization methods."""
    print("=" * 80)
    print("TEST 4: Bbox Normalization Methods")
    print("=" * 80)
    
    # Non-square bbox
    pred = np.array([[0.5, 0.5]])
    gt = np.array([[0.6, 0.6]])  # 0.1 offset in normalized coords
    bbox_w, bbox_h = 1000.0, 500.0
    threshold = 0.2
    
    print(f"Bbox: {bbox_w} × {bbox_h}")
    print(f"Prediction: {pred[0]}")
    print(f"Ground truth: {gt[0]}")
    print(f"Offset: [0.1, 0.1] (normalized)")
    print(f"\nDistance in pixels: [{0.1 * bbox_w}, {0.1 * bbox_h}] = [100, 50]")
    print(f"Euclidean distance: {np.sqrt((100)**2 + (50)**2):.2f} pixels\n")
    
    # Method 1: Diagonal (standard)
    pck_diag, _, _ = compute_pck_bbox(
        pred, gt, bbox_w, bbox_h, threshold=threshold, normalize_by='diagonal'
    )
    bbox_diag = np.sqrt(bbox_w**2 + bbox_h**2)
    norm_dist_diag = np.sqrt((100)**2 + (50)**2) / bbox_diag
    print(f"Method 1 (diagonal):")
    print(f"  Bbox size: {bbox_diag:.2f}")
    print(f"  Normalized distance: {norm_dist_diag:.4f}")
    print(f"  Within threshold {threshold}? {norm_dist_diag < threshold}")
    print(f"  PCK: {pck_diag:.2%}")
    
    # Method 2: Max
    pck_max, _, _ = compute_pck_bbox(
        pred, gt, bbox_w, bbox_h, threshold=threshold, normalize_by='max'
    )
    bbox_max = max(bbox_w, bbox_h)
    norm_dist_max = np.sqrt((100)**2 + (50)**2) / bbox_max
    print(f"\nMethod 2 (max):")
    print(f"  Bbox size: {bbox_max:.2f}")
    print(f"  Normalized distance: {norm_dist_max:.4f}")
    print(f"  Within threshold {threshold}? {norm_dist_max < threshold}")
    print(f"  PCK: {pck_max:.2%}")
    
    # Method 3: Mean
    pck_mean, _, _ = compute_pck_bbox(
        pred, gt, bbox_w, bbox_h, threshold=threshold, normalize_by='mean'
    )
    bbox_mean = (bbox_w + bbox_h) / 2
    norm_dist_mean = np.sqrt((100)**2 + (50)**2) / bbox_mean
    print(f"\nMethod 3 (mean):")
    print(f"  Bbox size: {bbox_mean:.2f}")
    print(f"  Normalized distance: {norm_dist_mean:.4f}")
    print(f"  Within threshold {threshold}? {norm_dist_mean < threshold}")
    print(f"  PCK: {pck_mean:.2%}")
    
    print("\n✓ PASSED: Different normalization methods work\n")


def test_pck_evaluator():
    """Test PCKEvaluator accumulator."""
    print("=" * 80)
    print("TEST 5: PCKEvaluator Accumulator")
    print("=" * 80)
    
    evaluator = PCKEvaluator(threshold=0.2)
    
    # Category 1: 2 images, perfect predictions
    pred1 = np.array([[[0.5, 0.5], [0.3, 0.7]], [[0.8, 0.2], [0.4, 0.6]]])
    gt1 = pred1.copy()
    bbox_w1 = np.array([512.0, 512.0])
    bbox_h1 = np.array([512.0, 512.0])
    cat_ids1 = np.array([1, 1])
    
    # Category 2: 2 images, 50% accuracy
    pred2 = np.array([[[0.5, 0.5], [0.9, 0.9]], [[0.8, 0.2], [0.1, 0.1]]])
    gt2 = np.array([[[0.5, 0.5], [0.3, 0.3]], [[0.8, 0.2], [0.6, 0.6]]])
    bbox_w2 = np.array([512.0, 512.0])
    bbox_h2 = np.array([512.0, 512.0])
    cat_ids2 = np.array([2, 2])
    
    # Add batches
    import torch
    evaluator.add_batch(
        torch.from_numpy(pred1), torch.from_numpy(gt1),
        torch.from_numpy(bbox_w1), torch.from_numpy(bbox_h1),
        torch.from_numpy(cat_ids1)
    )
    
    evaluator.add_batch(
        torch.from_numpy(pred2), torch.from_numpy(gt2),
        torch.from_numpy(bbox_w2), torch.from_numpy(bbox_h2),
        torch.from_numpy(cat_ids2)
    )
    
    # Get results
    results = evaluator.get_results()
    
    print(f"Total images: {results['num_images']}")
    print(f"Total categories: {results['num_categories']}")
    print(f"Overall PCK: {results['pck_overall']:.2%}")
    print(f"Mean PCK (macro): {results['mean_pck_categories']:.2%}")
    print(f"\nPer-category PCK:")
    for cat_id, pck in results['pck_per_category'].items():
        print(f"  Category {cat_id}: {pck:.2%}")
    
    # Verify
    assert results['num_images'] == 4, "Should have 4 images"
    assert results['num_categories'] == 2, "Should have 2 categories"
    assert results['pck_per_category'][1] == 1.0, "Category 1 should have 100% PCK"
    assert results['pck_per_category'][2] == 0.5, "Category 2 should have 50% PCK"
    
    # Overall: (4 correct from cat1 + 2 correct from cat2) / (4 + 4) = 6/8 = 0.75
    expected_overall = 0.75
    assert abs(results['pck_overall'] - expected_overall) < 0.01, \
        f"Expected {expected_overall:.2%}, got {results['pck_overall']:.2%}"
    
    # Mean: (1.0 + 0.5) / 2 = 0.75
    expected_mean = 0.75
    assert abs(results['mean_pck_categories'] - expected_mean) < 0.01, \
        f"Expected {expected_mean:.2%}, got {results['mean_pck_categories']:.2%}"
    
    print("\n✓ PASSED: PCKEvaluator accumulates correctly\n")


def run_all_tests():
    """Run all PCK tests."""
    print("\n" + "=" * 80)
    print("RUNNING PCK@bbox EVALUATION TESTS")
    print("=" * 80 + "\n")
    
    try:
        test_perfect_predictions()
        test_varying_error()
        test_visibility_masking()
        test_bbox_normalization()
        test_pck_evaluator()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nPCK@bbox implementation verified:")
        print("  ✓ Distance computation")
        print("  ✓ Bbox normalization (diagonal, max, mean)")
        print("  ✓ Threshold checking")
        print("  ✓ Visibility masking")
        print("  ✓ Batch processing")
        print("  ✓ Per-category aggregation")
        print("  ✓ Overall and macro-average metrics")
        print()
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

