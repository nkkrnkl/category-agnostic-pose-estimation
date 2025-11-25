#!/usr/bin/env python3
"""
Tests for appearance-only augmentation pipeline.

Verifies that:
1. Augmentations do NOT modify keypoint annotations (bitwise identical)
2. Augmentations DO modify image appearance (visual changes)
3. Training uses augmentation, validation/test do not
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.mp100_cape import MP100CAPE


class MockArgs:
    """Mock args for dataset construction"""
    def __init__(self, dataset_root='.'):
        self.dataset_root = dataset_root
        self.mp100_split = 1
        self.num_bins = 256
        self.image_normalize = True


class TestAppearanceAugmentation:
    """Test suite for appearance-only augmentation"""
    
    @pytest.fixture
    def train_dataset(self):
        """Create training dataset with augmentation"""
        args = MockArgs()
        try:
            dataset = MP100CAPE(
                img_folder=Path(args.dataset_root) / "data",
                ann_file=Path(args.dataset_root) / "data" / "annotations" / "mp100_split1_train.json",
                transforms=None,  # Will be built internally
                image_set='train'
            )
            return dataset
        except Exception as e:
            pytest.skip(f"Cannot load dataset: {e}")
    
    @pytest.fixture
    def val_dataset(self):
        """Create validation dataset without augmentation"""
        args = MockArgs()
        try:
            dataset = MP100CAPE(
                img_folder=Path(args.dataset_root) / "data",
                ann_file=Path(args.dataset_root) / "data" / "annotations" / "mp100_split1_val.json",
                transforms=None,
                image_set='val'
            )
            return dataset
        except Exception as e:
            pytest.skip(f"Cannot load dataset: {e}")
    
    def test_keypoints_unchanged_by_augmentation(self, train_dataset):
        """
        CRITICAL TEST: Verify that augmentation does NOT modify keypoint annotations.
        
        Keypoints must be BITWISE IDENTICAL before and after transforms.
        """
        # Get a sample from training set (which has augmentation)
        idx = 0
        
        # Load sample multiple times
        sample1 = train_dataset[idx]
        sample2 = train_dataset[idx]
        sample3 = train_dataset[idx]
        
        # Extract keypoint data
        kpts1 = np.array(sample1['keypoints'])
        kpts2 = np.array(sample2['keypoints'])
        kpts3 = np.array(sample3['keypoints'])
        
        # Keypoints must be EXACTLY identical (bitwise)
        # Even though images will differ due to augmentation,
        # keypoints should not change at all
        np.testing.assert_array_equal(
            kpts1, kpts2,
            err_msg="Keypoints changed between augmented samples! "
                    "Augmentation must NOT modify keypoint annotations."
        )
        np.testing.assert_array_equal(
            kpts1, kpts3,
            err_msg="Keypoints changed between augmented samples! "
                    "Augmentation must NOT modify keypoint annotations."
        )
        
        # Also check visibility (must be identical)
        vis1 = np.array(sample1.get('visibility', []))
        vis2 = np.array(sample2.get('visibility', []))
        vis3 = np.array(sample3.get('visibility', []))
        
        if len(vis1) > 0:
            np.testing.assert_array_equal(vis1, vis2)
            np.testing.assert_array_equal(vis1, vis3)
    
    def test_images_changed_by_augmentation(self, train_dataset):
        """
        Verify that augmentation DOES modify image appearance.
        
        When we load the same sample multiple times, images should differ
        due to random augmentations.
        """
        idx = 0
        
        # Load same sample 5 times
        samples = [train_dataset[idx] for _ in range(5)]
        
        # Extract image tensors
        images = [s['image'] for s in samples]
        
        # Count how many pairs differ
        num_different = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img_i = images[i]
                img_j = images[j]
                
                # Check if images differ (at least one pixel different)
                if not torch.allclose(img_i, img_j):
                    num_different += 1
        
        # With random augmentation (p > 0), we expect most pairs to differ
        # Out of 10 pairs (5 choose 2), at least 5 should differ
        assert num_different >= 5, (
            f"Only {num_different}/10 image pairs differ. "
            f"Augmentation may not be active or probability is too low."
        )
    
    def test_validation_deterministic(self, val_dataset):
        """
        Verify that validation dataset is DETERMINISTIC (no augmentation).
        
        Loading the same sample multiple times should produce identical images.
        """
        idx = 0
        
        # Load same sample 3 times
        sample1 = val_dataset[idx]
        sample2 = val_dataset[idx]
        sample3 = val_dataset[idx]
        
        # Images should be EXACTLY identical (no augmentation)
        img1 = sample1['image']
        img2 = sample2['image']
        img3 = sample3['image']
        
        torch.testing.assert_close(
            img1, img2,
            msg="Validation images differ! Validation should be deterministic (no augmentation)."
        )
        torch.testing.assert_close(
            img1, img3,
            msg="Validation images differ! Validation should be deterministic (no augmentation)."
        )
    
    def test_bbox_unchanged_by_augmentation(self, train_dataset):
        """
        Verify that bounding box coordinates remain unchanged.
        
        Since we don't use geometric augmentation, bbox should be identical.
        """
        idx = 0
        
        sample1 = train_dataset[idx]
        sample2 = train_dataset[idx]
        
        bbox1 = sample1.get('bbox', [])
        bbox2 = sample2.get('bbox', [])
        
        if len(bbox1) > 0:
            np.testing.assert_array_equal(
                bbox1, bbox2,
                err_msg="Bounding box changed! Geometric augmentation must be disabled."
            )
    
    def test_image_shape_preserved(self, train_dataset):
        """
        Verify that image shape is preserved (512x512).
        
        Appearance-only augmentation should not change image dimensions.
        """
        idx = 0
        
        sample = train_dataset[idx]
        img = sample['image']
        
        # Should be [C, H, W] = [3, 512, 512]
        assert img.shape == (3, 512, 512), (
            f"Image shape is {img.shape}, expected (3, 512, 512). "
            f"Augmentation must preserve image dimensions."
        )


if __name__ == '__main__':
    print("Run tests with: pytest tests/test_appearance_augmentation.py -v")

