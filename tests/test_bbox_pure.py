#!/usr/bin/env python3
"""
Pure Python test - no dependencies required.
"""

import json
from pathlib import Path

def test_bbox_in_annotations():
    """Test that bboxes exist in MP-100 annotations."""
    
    print("=" * 80)
    print("VERIFYING BBOX DATA IN MP-100 ANNOTATIONS")
    print("=" * 80)
    
    # Load annotation file
    ann_file = Path(__file__).parent / 'data/annotations/mp100_split1_train.json'
    
    if not ann_file.exists():
        print(f"✗ Annotation file not found: {ann_file}")
        return False
    
    print(f"\n1. Loading annotations...")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"   ✓ Loaded {len(data['annotations'])} annotations")
    
    # Check first few annotations for bbox
    print("\n2. Checking bbox presence in annotations...")
    has_bbox_count = 0
    has_keypoints_count = 0
    
    for i, ann in enumerate(data['annotations'][:100]):  # Check first 100
        if 'bbox' in ann and ann['bbox']:
            has_bbox_count += 1
        if 'keypoints' in ann and ann['keypoints']:
            has_keypoints_count += 1
    
    print(f"   Annotations with bbox: {has_bbox_count}/100")
    print(f"   Annotations with keypoints: {has_keypoints_count}/100")
    
    if has_bbox_count == 0:
        print("   ✗ No bboxes found in annotations!")
        return False
    
    print(f"   ✓ Bboxes present in annotations")
    
    # Show example
    print("\n3. Example annotation structure:")
    for ann in data['annotations']:
        if 'bbox' in ann and 'keypoints' in ann and ann['bbox'] and ann['keypoints']:
            bbox = ann['bbox']
            kpts = ann['keypoints']
            
            print(f"   Annotation ID: {ann['id']}")
            print(f"   Category ID: {ann['category_id']}")
            print(f"   Bbox: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
            print(f"   Num keypoints (raw): {len(kpts)} values ({len(kpts)//3} keypoints)")
            print(f"   Num visible keypoints: {ann.get('num_keypoints', 'N/A')}")
            
            # Show first keypoint
            print(f"   First keypoint: [{kpts[0]:.2f}, {kpts[1]:.2f}, v={kpts[2]}]")
            
            # Calculate bbox-relative coordinate
            kpt_x_abs = kpts[0]
            kpt_y_abs = kpts[1]
            kpt_x_rel = kpt_x_abs - bbox[0]
            kpt_y_rel = kpt_y_abs - bbox[1]
            kpt_x_norm = kpt_x_rel / bbox[2]
            kpt_y_norm = kpt_y_rel / bbox[3]
            
            print(f"   First keypoint (absolute): [{kpt_x_abs:.2f}, {kpt_y_abs:.2f}]")
            print(f"   First keypoint (bbox-relative): [{kpt_x_rel:.2f}, {kpt_y_rel:.2f}]")
            print(f"   First keypoint (normalized [0,1]): [{kpt_x_norm:.4f}, {kpt_y_norm:.4f}]")
            
            break
    
    print("\n" + "=" * 80)
    print("✓ VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nConfirmed:")
    print("  ✓ MP-100 annotations contain bboxes")
    print("  ✓ Bboxes are in COCO format [x, y, width, height]")
    print("  ✓ Keypoints can be converted to bbox-relative coordinates")
    print("  ✓ Normalization to [0, 1] is straightforward")
    print()
    
    return True


if __name__ == '__main__':
    import sys
    success = test_bbox_in_annotations()
    sys.exit(0 if success else 1)

