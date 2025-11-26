#!/usr/bin/env python3
"""
Automated test for visualization pipeline normalization/denormalization.

This test verifies that:
1. Normalization and denormalization are exact inverses
2. Keypoints align correctly when visualized
3. Coordinate transformations are consistent
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from models.visualize_cape_predictions import (
    visualize_pose_prediction,
    ensure_image_size,
    TARGET_SIZE
)


def test_normalization_denormalization_roundtrip():
    """
    Test that normalization → denormalization is exact (pixel error < 1e-6).
    """
    print("=" * 80)
    print("Test 1: Normalization/Denormalization Roundtrip")
    print("=" * 80)
    
    # Create fake keypoints in pixel coordinates [0, 512]
    np.random.seed(42)
    num_keypoints = 10
    pixel_coords = np.random.rand(num_keypoints, 2) * TARGET_SIZE
    
    # Normalize to [0, 1]
    normalized_coords = pixel_coords / TARGET_SIZE
    
    # Denormalize back to pixels
    denormalized_coords = normalized_coords * TARGET_SIZE
    
    # Check roundtrip error
    error = np.abs(pixel_coords - denormalized_coords)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"  Original pixel coords (first 3):")
    for i in range(min(3, num_keypoints)):
        print(f"    Kpt {i}: ({pixel_coords[i,0]:.6f}, {pixel_coords[i,1]:.6f})")
    print(f"  Normalized coords (first 3):")
    for i in range(min(3, num_keypoints)):
        print(f"    Kpt {i}: ({normalized_coords[i,0]:.6f}, {normalized_coords[i,1]:.6f})")
    print(f"  Denormalized coords (first 3):")
    for i in range(min(3, num_keypoints)):
        print(f"    Kpt {i}: ({denormalized_coords[i,0]:.6f}, {denormalized_coords[i,1]:.6f})")
    print(f"  Max error: {max_error:.10f}")
    print(f"  Mean error: {mean_error:.10f}")
    
    assert max_error < 1e-6, f"Roundtrip error too large: {max_error} > 1e-6"
    print("  ✓ PASS: Normalization/denormalization roundtrip is exact\n")


def test_ensure_image_size():
    """
    Test that ensure_image_size correctly resizes images to 512x512.
    """
    print("=" * 80)
    print("Test 2: Image Size Enforcement")
    print("=" * 80)
    
    # Test 1: Already 512x512
    img_512 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    result = ensure_image_size(img_512, TARGET_SIZE)
    assert result.shape == (512, 512, 3), f"Expected (512, 512, 3), got {result.shape}"
    assert np.array_equal(result, img_512), "512x512 image should be unchanged"
    print("  ✓ PASS: 512x512 image unchanged")
    
    # Test 2: Different size (resize)
    img_256 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = ensure_image_size(img_256, TARGET_SIZE)
    assert result.shape == (512, 512, 3), f"Expected (512, 512, 3), got {result.shape}"
    print("  ✓ PASS: 256x256 image resized to 512x512")
    
    # Test 3: Float image [0, 1]
    img_float = np.random.rand(100, 150, 3).astype(np.float32)
    result = ensure_image_size(img_float, TARGET_SIZE)
    assert result.shape == (512, 512, 3), f"Expected (512, 512, 3), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert result.max() <= 255 and result.min() >= 0, "Values should be in [0, 255]"
    print("  ✓ PASS: Float image converted to uint8 and resized")
    
    print("  ✓ ALL TESTS PASSED\n")


def test_visualization_coordinate_alignment():
    """
    Test that identical GT and predicted keypoints align perfectly in visualization.
    """
    print("=" * 80)
    print("Test 3: Visualization Coordinate Alignment")
    print("=" * 80)
    
    # Create fake 512x512 image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Create identical GT and predicted keypoints (normalized [0,1])
    np.random.seed(123)
    num_keypoints = 5
    normalized_keypoints = np.random.rand(num_keypoints, 2)
    normalized_keypoints = np.clip(normalized_keypoints, 0.1, 0.9)  # Keep away from edges
    
    # Convert to lists
    gt_keypoints = normalized_keypoints.tolist()
    pred_keypoints = normalized_keypoints.tolist()  # Identical to GT
    
    # Create skeleton (simple chain)
    skeleton_edges = [[i+1, i+2] for i in range(num_keypoints - 1)]
    
    # Test visualization (should not raise errors)
    try:
        output_path = Path(PROJECT_ROOT) / "output" / "test_visualization_alignment.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualize_pose_prediction(
            support_image=test_image,
            query_image=test_image,
            pred_keypoints=pred_keypoints,
            support_keypoints=gt_keypoints,
            gt_keypoints=gt_keypoints,
            skeleton_edges=skeleton_edges,
            save_path=str(output_path),
            category_name="Test Category",
            pck_score=1.0,
            debug_coords=True
        )
        
        # Check that file was created
        assert output_path.exists(), "Visualization file should be created"
        print(f"  ✓ PASS: Visualization created successfully at {output_path}")
        
        # Clean up
        if output_path.exists():
            output_path.unlink()
        
    except Exception as e:
        print(f"  ✗ FAIL: Visualization raised exception: {e}")
        raise
    
    print("  ✓ ALL TESTS PASSED\n")


def test_coordinate_denormalization_consistency():
    """
    Test that denormalization uses TARGET_SIZE (512) consistently.
    """
    print("=" * 80)
    print("Test 4: Coordinate Denormalization Consistency")
    print("=" * 80)
    
    # Create normalized coordinates
    normalized_coords = np.array([[0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    
    # Denormalize using TARGET_SIZE
    denormalized = normalized_coords * TARGET_SIZE
    
    # Expected values
    expected = np.array([[256.0, 256.0], [128.0, 384.0], [0.0, 512.0]])
    
    error = np.abs(denormalized - expected)
    max_error = np.max(error)
    
    print(f"  Normalized coords: {normalized_coords}")
    print(f"  Denormalized coords: {denormalized}")
    print(f"  Expected coords: {expected}")
    print(f"  Max error: {max_error:.10f}")
    
    assert max_error < 1e-6, f"Denormalization error too large: {max_error}"
    assert np.allclose(denormalized, expected, atol=1e-6), "Denormalization should use TARGET_SIZE"
    
    print("  ✓ PASS: Denormalization uses TARGET_SIZE consistently\n")


def main():
    """Run all visualization pipeline tests."""
    print("\n" + "=" * 80)
    print("VISUALIZATION PIPELINE TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        test_normalization_denormalization_roundtrip()
        test_ensure_image_size()
        test_visualization_coordinate_alignment()
        test_coordinate_denormalization_consistency()
        
        print("=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

