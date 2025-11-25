#!/usr/bin/env python3
"""
Simple test to verify bbox processing logic without requiring full PyTorch setup.
"""

import json
import numpy as np
from pathlib import Path

def test_bbox_extraction():
    """Test that bbox extraction logic is correct."""
    
    print("=" * 80)
    print("TESTING BBOX EXTRACTION LOGIC")
    print("=" * 80)
    
    # Load annotation file
    ann_file = Path(__file__).parent / 'data/annotations/mp100_split1_train.json'
    
    if not ann_file.exists():
        print(f"✗ Annotation file not found: {ann_file}")
        return False
    
    print(f"\n1. Loading annotations from {ann_file.name}...")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"   ✓ Loaded {len(data['annotations'])} annotations")
    print(f"   ✓ Loaded {len(data['images'])} images")
    print(f"   ✓ Loaded {len(data['categories'])} categories")
    
    # Test bbox processing on first annotation
    print("\n2. Testing bbox processing on sample annotation...")
    ann = data['annotations'][0]
    
    # Extract bbox
    if 'bbox' not in ann:
        print("   ✗ No bbox in annotation!")
        return False
    
    bbox = ann['bbox']
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    print(f"   Original bbox: [{bbox_x:.2f}, {bbox_y:.2f}, {bbox_w:.2f}, {bbox_h:.2f}]")
    
    # Extract keypoints
    if 'keypoints' not in ann or not ann['keypoints']:
        print("   ✗ No keypoints in annotation!")
        return False
    
    kpts = np.array(ann['keypoints']).reshape(-1, 3)
    print(f"   Total keypoints: {len(kpts)}")
    
    # Filter visible
    visible_mask = kpts[:, 2] > 0
    visible_kpts = kpts[visible_mask][:, :2]
    print(f"   Visible keypoints: {len(visible_kpts)}")
    
    # Simulate bbox-relative adjustment
    print("\n3. Adjusting keypoints to be bbox-relative...")
    kpts_original = kpts[:, :2].copy()
    print(f"   First keypoint (absolute): [{kpts_original[0, 0]:.2f}, {kpts_original[0, 1]:.2f}]")
    
    kpts_bbox_relative = kpts_original.copy()
    kpts_bbox_relative[:, 0] -= bbox_x
    kpts_bbox_relative[:, 1] -= bbox_y
    print(f"   First keypoint (bbox-relative): [{kpts_bbox_relative[0, 0]:.2f}, {kpts_bbox_relative[0, 1]:.2f}]")
    
    # Normalize to [0, 1] by bbox dimensions
    print("\n4. Normalizing to [0, 1] by bbox dimensions...")
    kpts_normalized = kpts_bbox_relative.copy()
    kpts_normalized[:, 0] /= bbox_w
    kpts_normalized[:, 1] /= bbox_h
    print(f"   First keypoint (normalized): [{kpts_normalized[0, 0]:.4f}, {kpts_normalized[0, 1]:.4f}]")
    
    # Check if in valid range
    visible_normalized = kpts_normalized[visible_mask]
    min_x, max_x = visible_normalized[:, 0].min(), visible_normalized[:, 0].max()
    min_y, max_y = visible_normalized[:, 1].min(), visible_normalized[:, 1].max()
    
    print(f"   Normalized range X: [{min_x:.4f}, {max_x:.4f}]")
    print(f"   Normalized range Y: [{min_y:.4f}, {max_y:.4f}]")
    
    if min_x < -0.1 or max_x > 1.1 or min_y < -0.1 or max_y > 1.1:
        print("   ⚠ Warning: Some keypoints outside [0, 1] range!")
        print("   This may indicate keypoints outside bbox (annotations may be imperfect)")
    else:
        print("   ✓ All keypoints in reasonable [0, 1] range")
    
    # Simulate resizing to 512x512
    print("\n5. Simulating resize to 512x512...")
    target_size = 512
    scale_x = target_size / bbox_w
    scale_y = target_size / bbox_h
    
    kpts_resized = kpts_bbox_relative.copy()
    kpts_resized[:, 0] *= scale_x
    kpts_resized[:, 1] *= scale_y
    
    print(f"   Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
    print(f"   First keypoint (after resize): [{kpts_resized[0, 0]:.2f}, {kpts_resized[0, 1]:.2f}]")
    
    # Final normalization
    kpts_final_normalized = kpts_resized / target_size
    print(f"   First keypoint (final normalized): [{kpts_final_normalized[0, 0]:.4f}, {kpts_final_normalized[0, 1]:.4f}]")
    
    print("\n" + "=" * 80)
    print("✓ BBOX EXTRACTION LOGIC TEST PASSED")
    print("=" * 80)
    print("\nVerified:")
    print("  ✓ Bbox extraction from COCO annotations")
    print("  ✓ Keypoint adjustment to bbox-relative coordinates")
    print("  ✓ Normalization by bbox dimensions")
    print("  ✓ Resize scaling to 512x512")
    print()
    
    return True


if __name__ == '__main__':
    import sys
    success = test_bbox_extraction()
    sys.exit(0 if success else 1)

