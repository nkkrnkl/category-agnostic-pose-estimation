#!/usr/bin/env python3
"""
Quick test script to verify MP-100 dataset loading works correctly
"""

import sys
from pathlib import Path
import torch

# Add theodoros to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.mp100_cape import build_mp100_cape
import argparse


def test_dataset_loading():
    print("=" * 80)
    print("Testing MP-100 CAPE Dataset Loading")
    print("=" * 80)

    # Create minimal args
    args = argparse.Namespace(
        dataset_root='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/theodoros',
        mp100_split=1,
        semantic_classes=49,
        image_norm=True,
        vocab_size=2000,
        seq_len=200
    )

    print(f"\nDataset root: {args.dataset_root}")
    print(f"Split: {args.mp100_split}")

    # Try to load train dataset
    try:
        print("\n" + "-" * 80)
        print("Loading TRAIN dataset...")
        print("-" * 80)
        dataset_train = build_mp100_cape('train', args)
        print(f"✓ Train dataset loaded: {len(dataset_train)} samples")

        # Test loading a single sample
        print("\nTesting single sample loading...")
        sample = dataset_train[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Num keypoints: {sample.get('num_keypoints', 'N/A')}")
        print(f"  - Category ID: {sample.get('category_id', 'N/A')}")
        print(f"  - Image ID: {sample.get('image_id', 'N/A')}")
        if 'keypoints' in sample:
            print(f"  - First 3 keypoints: {sample['keypoints'][:3]}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease check that annotation files exist at:")
        print(f"  {args.dataset_root}/annotations/mp100_split1_train.json")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to load val dataset
    try:
        print("\n" + "-" * 80)
        print("Loading VAL dataset...")
        print("-" * 80)
        dataset_val = build_mp100_cape('val', args)
        print(f"✓ Val dataset loaded: {len(dataset_val)} samples")

        # Test loading a single sample
        print("\nTesting single sample loading...")
        sample = dataset_val[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Num keypoints: {sample.get('num_keypoints', 'N/A')}")

    except Exception as e:
        print(f"✗ Error loading val dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
