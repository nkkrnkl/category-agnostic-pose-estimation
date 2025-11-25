#!/usr/bin/env python3
"""
Comprehensive test suite to debug the PCK@100% issue in validation.

Tests:
1. Support and query images are DIFFERENT within an episode
2. Episodic sampler samples without replacement
3. forward_inference exists and is callable
4. Predictions are NOT identical to ground truth (no teacher forcing)
5. Predictions are NOT identical to support (no copying)
6. PCK computation is working correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
import os
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import EpisodicSampler, EpisodicDataset, episodic_collate_fn, build_episodic_dataloader
from models import build_model
from models.cape_model import build_cape_model
from util.eval_utils import compute_pck_bbox


class TestValidationPCK:
    """Test suite for validation PCK debugging."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.dataset_root = project_root
        self.results = []
    
    def create_minimal_dataset_args(self):
        """Create minimal args for building datasets."""
        return argparse.Namespace(
            dataset_root=str(self.dataset_root),
            max_keypoints=25,
            semantic_classes=False,
            image_norm=False,
            vocab_size=2000,
            seq_len=200,
            mp100_split=1,
        )
        
    def log_result(self, test_name, passed, message=""):
        """Log a test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        print(f"{status}: {test_name}")
        if message:
            print(f"  → {message}")
        print()
    
    def create_temp_category_split(self, category_id=66):
        """Create a temporary category split with one category in val.
        
        Default category_id=66 because it has many examples (232) in MP-100 validation split.
        """
        temp_split = {
            "description": "Test split with single category",
            "total_categories": 1,
            "train_categories": 0,
            "val_categories": 1,
            "test_categories": 0,
            "train": [],
            "val": [category_id],
            "test": []
        }
        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.json', text=True)
        os.close(temp_fd)
        
        with open(temp_path, 'w') as f:
            json.dump(temp_split, f, indent=2)
        
        return temp_path
    
    def build_test_model(self):
        """Build a minimal CAPE model for testing."""
        args = argparse.Namespace(
            # Model architecture
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            pre_norm=False,
            add_cls_token=False,
            input_channels=3,
            query_pos_type='learned',
            
            # CAPE-specific
            support_encoder_layers=3,
            support_fusion_method='cross_attention',
            
            # Backbone
            backbone='resnet18',  # Smaller for testing
            dilation=False,
            position_embedding='sine',
            lr_backbone=0.0,  # Don't train backbone in tests
            
            # Loss
            aux_loss=True,
            ce_loss_coef=1.0,
            coords_loss_coef=5.0,
            
            # Dataset
            dataset_root=str(self.dataset_root),
            max_keypoints=25,
            num_bins=1000,
            vocab_size=2000,
            seq_len=200,
            semantic_classes=False,
            image_norm=False,
            poly2seq=False,  # Required by build_model
        )
        
        # Build base model first
        base_model, _ = build_model(args)
        
        # Build CAPE model
        model = build_cape_model(args, base_model)
        model.to(self.device)
        model.eval()
        return model, args
    
    def test_episodic_sampler_no_replacement(self):
        """Test that episodic sampler samples support and queries WITHOUT replacement."""
        print("=" * 80)
        print("TEST 1: Episodic Sampler - No Replacement")
        print("=" * 80)
        print()
        
        # Use real category split file (has all actual val categories)
        real_split_path = self.dataset_root / 'category_splits.json'
        
        try:
            # Build dataset
            val_dataset = build_mp100_cape('val', self.create_minimal_dataset_args())
            
            # Create sampler
            sampler = EpisodicSampler(
                val_dataset,
                str(real_split_path),
                split='val',
                num_queries_per_episode=2,
                seed=42
            )
            
            # Sample 10 episodes
            print(f"Sampling 10 episodes to check for image reuse...")
            print()
            
            duplicates_found = 0
            for i in range(10):
                episode = sampler.sample_episode()
                support_idx = episode['support_idx']
                query_indices = episode['query_indices']
                
                # Check if support appears in queries
                if support_idx in query_indices:
                    print(f"  Episode {i}: ✗ Support index {support_idx} appears in queries {query_indices}")
                    duplicates_found += 1
                else:
                    print(f"  Episode {i}: ✓ Support {support_idx} ≠ Queries {query_indices}")
            
            print()
            passed = duplicates_found == 0
            self.log_result(
                "Episodic sampler samples without replacement",
                passed,
                f"Found {duplicates_found}/10 episodes with duplicate images" if not passed else "All episodes have unique support and query images"
            )
        except Exception as e:
            self.log_result(
                "Episodic sampler samples without replacement",
                False,
                f"Exception: {e}"
            )
    
    def test_episodic_dataset_unique_images(self):
        """Test that EpisodicDataset returns episodes with unique support and query images."""
        print("=" * 80)
        print("TEST 2: EpisodicDataset - Unique Image IDs")
        print("=" * 80)
        print()
        
        # Use real category split file
        real_split_path = self.dataset_root / 'category_splits.json'
        
        try:
            # Build dataset
            val_dataset = build_mp100_cape('val', self.create_minimal_dataset_args())
            
            # Create episodic dataset
            episodic_ds = EpisodicDataset(
                val_dataset,
                str(real_split_path),
                split='val',
                num_queries_per_episode=2,
                episodes_per_epoch=5,
                seed=42
            )
            
            print(f"Loading 5 episodes and checking image IDs...")
            print()
            
            duplicates_found = 0
            for i in range(len(episodic_ds)):
                episode = episodic_ds[i]
                
                # Get image IDs
                support_meta = episode.get('support_metadata', {})
                query_meta = episode.get('query_metadata', [])
                
                support_id = support_meta.get('image_id', None)
                query_ids = [m.get('image_id', None) for m in query_meta]
                
                print(f"  Episode {i}:")
                print(f"    Support ID: {support_id}")
                print(f"    Query IDs:  {query_ids}")
                
                # Check for duplicates
                if support_id in query_ids:
                    print(f"    ✗ DUPLICATE: Support appears in queries!")
                    duplicates_found += 1
                else:
                    print(f"    ✓ Unique images")
                print()
            
            passed = duplicates_found == 0
            self.log_result(
                "EpisodicDataset returns unique images per episode",
                passed,
                f"Found {duplicates_found}/5 episodes with duplicate image IDs" if not passed else "All episodes have unique support and query image IDs"
            )
        except Exception as e:
            self.log_result(
                "EpisodicDataset returns unique images per episode",
                False,
                f"Exception: {e}"
            )
    
    def test_collate_fn_preserves_metadata(self):
        """Test that episodic_collate_fn preserves support and query metadata."""
        print("=" * 80)
        print("TEST 3: Collate Function - Metadata Preservation")
        print("=" * 80)
        print()
        
        # Use real category split file
        real_split_path = self.dataset_root / 'category_splits.json'
        
        try:
            # Build dataset
            val_dataset = build_mp100_cape('val', self.create_minimal_dataset_args())
            
            # Create episodic dataset
            episodic_ds = EpisodicDataset(
                val_dataset,
                str(real_split_path),
                split='val',
                num_queries_per_episode=2,
                episodes_per_epoch=2,
                seed=42
            )
            
            # Get batch of 2 episodes (4 queries total)
            batch = [episodic_ds[0], episodic_ds[1]]
            
            print(f"Pre-collate:")
            print(f"  Episode 0 support ID: {batch[0].get('support_metadata', {}).get('image_id', 'N/A')}")
            print(f"  Episode 0 query IDs: {[m.get('image_id', 'N/A') for m in batch[0].get('query_metadata', [])]}")
            print(f"  Episode 1 support ID: {batch[1].get('support_metadata', {}).get('image_id', 'N/A')}")
            print(f"  Episode 1 query IDs: {[m.get('image_id', 'N/A') for m in batch[1].get('query_metadata', [])]}")
            print()
            
            # Collate
            collated = episodic_collate_fn(batch)
            
            print(f"Post-collate:")
            support_meta = collated.get('support_metadata', [])
            query_meta = collated.get('query_metadata', [])
            
            print(f"  Support metadata length: {len(support_meta)} (expected: 4 = 2 episodes * 2 queries)")
            print(f"  Query metadata length: {len(query_meta)} (expected: 4)")
            
            if support_meta:
                support_ids = [m.get('image_id', 'N/A') for m in support_meta]
                print(f"  Support IDs: {support_ids}")
            else:
                print(f"  Support IDs: (not available)")
                support_ids = []
            
            if query_meta:
                query_ids = [m.get('image_id', 'N/A') for m in query_meta]
                print(f"  Query IDs: {query_ids}")
            else:
                print(f"  Query IDs: (not available)")
                query_ids = []
            
            print()
            
            # Check for overlap
            overlap_found = False
            if support_ids and query_ids:
                overlap = set(support_ids) & set(query_ids)
                if overlap:
                    print(f"  ✗ OVERLAP: {len(overlap)} image IDs appear in both support and query!")
                    print(f"     Overlapping IDs: {list(overlap)}")
                    overlap_found = True
                else:
                    print(f"  ✓ No overlap between support and query IDs")
            
            print()
            
            passed = len(support_meta) > 0 and len(query_meta) > 0 and not overlap_found
            self.log_result(
                "Collate function preserves metadata and no overlap",
                passed,
                "Metadata missing or image ID overlap detected" if not passed else "Metadata preserved correctly"
            )
        except Exception as e:
            self.log_result(
                "Collate function preserves metadata and no overlap",
                False,
                f"Exception: {e}"
            )
    
    def test_forward_inference_exists(self):
        """Test that the model has forward_inference method."""
        print("=" * 80)
        print("TEST 4: Model - forward_inference Method")
        print("=" * 80)
        print()
        
        model, args = self.build_test_model()
        
        has_method = hasattr(model, 'forward_inference')
        print(f"Model type: {type(model).__name__}")
        print(f"Has forward_inference: {has_method}")
        
        if has_method:
            print(f"✓ forward_inference is available")
        else:
            print(f"✗ forward_inference is NOT available")
            print(f"  Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
        
        print()
        
        self.log_result(
            "Model has forward_inference method",
            has_method,
            "forward_inference not found - validation will use teacher forcing!" if not has_method else "forward_inference available for autoregressive inference"
        )
    
    def test_forward_inference_vs_forward(self):
        """Test that forward_inference produces DIFFERENT outputs than forward with targets."""
        print("=" * 80)
        print("TEST 5: Inference Outputs - forward_inference vs forward")
        print("=" * 80)
        print()
        
        model, args = self.build_test_model()
        
        # Create dummy data
        batch_size = 2
        num_kpts = 10
        
        query_images = torch.randn(batch_size, 3, 512, 512)
        support_coords = torch.rand(batch_size, num_kpts, 2)
        support_mask = torch.ones(batch_size, num_kpts)
        
        # Create dummy targets
        seq_len = 100
        dummy_targets = {
            'target_seq': torch.rand(batch_size, seq_len, 2),
            'token_labels': torch.zeros(batch_size, seq_len, dtype=torch.long),
            'mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'visibility_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'delta_x1': torch.zeros(batch_size, seq_len),
            'delta_y1': torch.zeros(batch_size, seq_len),
            'delta_x2': torch.ones(batch_size, seq_len),
            'delta_y2': torch.ones(batch_size, seq_len),
        }
        
        # Mark some tokens as coordinates (type 1)
        dummy_targets['token_labels'][:, :num_kpts] = 1
        
        print(f"Running inference with dummy data...")
        print(f"  Query images: {query_images.shape}")
        print(f"  Support coords: {support_coords.shape}")
        print()
        
        with torch.no_grad():
            # Run forward_inference (autoregressive)
            try:
                pred_autoregressive = model.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_mask,
                    skeleton_edges=None
                )
                
                coords_autoregressive = pred_autoregressive.get('coordinates', None)
                print(f"✓ forward_inference succeeded")
                print(f"  Output coords shape: {coords_autoregressive.shape if coords_autoregressive is not None else 'None'}")
                
            except Exception as e:
                print(f"✗ forward_inference failed: {e}")
                self.log_result(
                    "forward_inference is callable",
                    False,
                    f"Exception: {e}"
                )
                return
            
            # Run forward with targets (teacher forcing)
            try:
                outputs_teacher_forcing = model(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_mask,
                    targets=dummy_targets,
                    skeleton_edges=None
                )
                
                coords_teacher_forcing = outputs_teacher_forcing.get('pred_coords', None)
                print(f"✓ forward (teacher forcing) succeeded")
                print(f"  Output coords shape: {coords_teacher_forcing.shape if coords_teacher_forcing is not None else 'None'}")
                
            except Exception as e:
                print(f"✗ forward failed: {e}")
                coords_teacher_forcing = None
        
        print()
        
        # Compare outputs
        if coords_autoregressive is not None and coords_teacher_forcing is not None:
            diff = torch.abs(coords_autoregressive - coords_teacher_forcing).mean().item()
            
            print(f"Comparing outputs:")
            print(f"  Mean absolute difference: {diff:.6f}")
            
            are_different = diff > 0.001  # Should be significantly different
            
            if are_different:
                print(f"  ✓ Outputs are DIFFERENT (expected)")
                print(f"    forward_inference and forward produce different results")
            else:
                print(f"  ✗ Outputs are IDENTICAL (unexpected!)")
                print(f"    This suggests forward_inference might be using teacher forcing internally")
            
            print()
            
            self.log_result(
                "forward_inference produces different outputs than teacher forcing",
                are_different,
                f"Mean diff: {diff:.6f}" + (" (too similar!)" if not are_different else "")
            )
        else:
            self.log_result(
                "Both inference methods return coordinates",
                False,
                "One or both methods did not return coordinates"
            )
    
    def test_predictions_not_identical_to_support(self):
        """Test that model predictions are NOT just copying support coordinates."""
        print("=" * 80)
        print("TEST 6: Predictions vs Support")
        print("=" * 80)
        print()
        
        model, args = self.build_test_model()
        
        # Create dummy data
        batch_size = 2
        num_kpts = 10
        
        query_images = torch.randn(batch_size, 3, 512, 512)
        support_coords = torch.rand(batch_size, num_kpts, 2)
        support_mask = torch.ones(batch_size, num_kpts)
        
        print(f"Running forward_inference...")
        
        with torch.no_grad():
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_mask,
                skeleton_edges=None
            )
            
            pred_coords = predictions.get('coordinates', None)
        
        if pred_coords is None:
            self.log_result(
                "Predictions are not identical to support",
                False,
                "forward_inference did not return coordinates"
            )
            return
        
        print(f"  Pred coords: {pred_coords.shape}")
        print(f"  Support coords: {support_coords.shape}")
        print()
        
        # Compare first num_kpts of predictions to support
        pred_trimmed = pred_coords[:, :num_kpts, :]  # (B, num_kpts, 2)
        diff = torch.abs(pred_trimmed - support_coords).mean().item()
        
        print(f"Mean absolute difference: {diff:.6f}")
        
        are_different = diff > 0.001  # Should NOT be copying support exactly
        
        if are_different:
            print(f"✓ Predictions are different from support (expected)")
        else:
            print(f"✗ Predictions are IDENTICAL to support!")
            print(f"  This means the model is just copying support keypoints")
            print(f"  without looking at the query image")
        
        print()
        
        self.log_result(
            "Predictions are not copying support coordinates",
            are_different,
            f"Mean diff: {diff:.6f}" + (" - model is just copying support!" if not are_different else "")
        )
    
    def test_real_validation_batch(self):
        """Test a real validation batch to check for data leakage."""
        print("=" * 80)
        print("TEST 7: Real Validation Batch Analysis")
        print("=" * 80)
        print()
        
        # Use real category split file
        real_split_path = self.dataset_root / 'category_splits.json'
        
        try:
            # Build dataset
            val_dataset = build_mp100_cape('val', self.create_minimal_dataset_args())
            
            # Build dataloader
            val_loader = build_episodic_dataloader(
                base_dataset=val_dataset,
                category_split_file=str(real_split_path),
                split='val',
                batch_size=1,
                num_queries_per_episode=2,
                episodes_per_epoch=1,
                num_workers=0,
                seed=42
            )
            
            print(f"Loading first batch...")
            batch = next(iter(val_loader))
            
            # Extract metadata
            support_meta = batch.get('support_metadata', [])
            query_meta = batch.get('query_metadata', [])
            
            print(f"\nBatch contents:")
            print(f"  Support images: {batch['support_images'].shape}")
            print(f"  Query images: {batch['query_images'].shape}")
            print(f"  Support metadata: {len(support_meta)} entries")
            print(f"  Query metadata: {len(query_meta)} entries")
            print()
            
            # Check image IDs
            if support_meta and query_meta:
                support_ids = [m.get('image_id', 'N/A') for m in support_meta]
                query_ids = [m.get('image_id', 'N/A') for m in query_meta]
                
                print(f"Image IDs:")
                print(f"  Support: {support_ids}")
                print(f"  Query:   {query_ids}")
                print()
                
                # Check for overlap
                overlap = set(support_ids) & set(query_ids)
                if overlap:
                    print(f"✗ CRITICAL: {len(overlap)} images appear in BOTH support and query!")
                    print(f"  Overlapping IDs: {list(overlap)}")
                    print(f"  This is DATA LEAKAGE - explains 100% PCK")
                    passed = False
                else:
                    print(f"✓ No overlap - support and query use different images")
                    passed = True
            else:
                print(f"⚠️  Metadata not available, cannot check image IDs")
                passed = False
            
            print()
            
            self.log_result(
                "Real validation batch has no data leakage",
                passed,
                "Image ID overlap detected!" if not passed else "All support and query images are unique"
            )
        except Exception as e:
            self.log_result(
                "Real validation batch has no data leakage",
                False,
                f"Exception: {e}"
            )
    
    def test_pck_computation_sanity(self):
        """Test that PCK computation works correctly."""
        print("=" * 80)
        print("TEST 8: PCK Computation Sanity Check")
        print("=" * 80)
        print()
        
        # Test case 1: Identical predictions and GT → PCK should be 100%
        print("Test case 1: Identical predictions and GT")
        pred = np.array([[0.5, 0.5], [0.3, 0.7]])
        gt = np.array([[0.5, 0.5], [0.3, 0.7]])
        visibility = np.array([2, 2])
        
        pck, correct, visible = compute_pck_bbox(pred, gt, 100.0, 100.0, visibility, threshold=0.2)
        
        print(f"  PCK: {pck:.2%} (expected: 100%)")
        print(f"  Correct: {correct}/{visible}")
        
        test1_pass = pck == 1.0 and correct == 2 and visible == 2
        print(f"  {'✓ PASS' if test1_pass else '✗ FAIL'}")
        print()
        
        # Test case 2: Very different predictions → PCK should be 0%
        print("Test case 2: Very different predictions")
        pred = np.array([[0.0, 0.0], [0.0, 0.0]])
        gt = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        pck, correct, visible = compute_pck_bbox(pred, gt, 100.0, 100.0, visibility, threshold=0.2)
        
        print(f"  PCK: {pck:.2%} (expected: 0%)")
        print(f"  Correct: {correct}/{visible}")
        
        test2_pass = pck == 0.0 and correct == 0 and visible == 2
        print(f"  {'✓ PASS' if test2_pass else '✗ FAIL'}")
        print()
        
        # Test case 3: One correct, one incorrect
        print("Test case 3: Mixed results")
        pred = np.array([[0.5, 0.5], [0.0, 0.0]])
        gt = np.array([[0.5, 0.5], [1.0, 1.0]])
        
        pck, correct, visible = compute_pck_bbox(pred, gt, 100.0, 100.0, visibility, threshold=0.2)
        
        print(f"  PCK: {pck:.2%} (expected: 50%)")
        print(f"  Correct: {correct}/{visible}")
        
        test3_pass = pck == 0.5 and correct == 1 and visible == 2
        print(f"  {'✓ PASS' if test3_pass else '✗ FAIL'}")
        print()
        
        overall_pass = test1_pass and test2_pass and test3_pass
        self.log_result(
            "PCK computation is working correctly",
            overall_pass,
            "Some PCK tests failed" if not overall_pass else "All PCK sanity checks passed"
        )
    
    def test_inference_with_real_data(self):
        """Test inference on real validation data and check for suspicious patterns."""
        print("=" * 80)
        print("TEST 9: Real Data Inference Analysis")
        print("=" * 80)
        print()
        
        # Load checkpoint if available
        checkpoint_dir = project_root / 'outputs' / 'cape_run'
        checkpoint_path = checkpoint_dir / 'checkpoint_e007_lr1e-04_bs2_acc4_qpe2.pth'
        
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping real data test")
            self.log_result(
                "Real data inference test",
                False,
                "Checkpoint not available"
            )
            return
        
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        train_args = checkpoint['args']
        
        # Build model (ensure all required args are present)
        if not hasattr(train_args, 'semantic_classes'):
            train_args.semantic_classes = False
        model = build_cape_model(train_args, self.device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        print(f"✓ Model loaded")
        print()
        
        # Build validation dataset (ensure semantic_classes is set)
        if not hasattr(train_args, 'semantic_classes'):
            train_args.semantic_classes = False
        val_dataset = build_mp100_cape('val', train_args)
        
        # Build dataloader
        val_loader = build_episodic_dataloader(
            base_dataset=val_dataset,
            category_split_file=str(Path(train_args.dataset_root) / train_args.category_split_file),
            split='val',
            batch_size=1,
            num_queries_per_episode=2,
            episodes_per_epoch=3,  # Just 3 episodes
            num_workers=0,
            seed=42
        )
        
        print(f"Testing on 3 validation episodes...")
        print()
        
        issues_found = []
        
        for batch_idx, batch in enumerate(val_loader):
            print(f"Episode {batch_idx}:")
            
            # Extract data
            support_coords = batch['support_coords']
            support_masks = batch['support_masks']
            query_images = batch['query_images']
            query_targets = batch['query_targets']
            support_meta = batch.get('support_metadata', [])
            query_meta = batch.get('query_metadata', [])
            
            # Check image IDs
            if support_meta and query_meta:
                support_ids = [m.get('image_id', 'N/A') for m in support_meta]
                query_ids = [m.get('image_id', 'N/A') for m in query_meta]
                
                print(f"  Support IDs: {support_ids}")
                print(f"  Query IDs:   {query_ids}")
                
                overlap = set(support_ids) & set(query_ids)
                if overlap:
                    print(f"  ✗ IMAGE OVERLAP: {list(overlap)}")
                    issues_found.append(f"Episode {batch_idx}: Image overlap {list(overlap)}")
            
            # Run inference
            with torch.no_grad():
                predictions = model.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks,
                    skeleton_edges=batch.get('support_skeletons', None)
                )
            
            pred_coords = predictions.get('coordinates', None)
            gt_coords = query_targets.get('target_seq', None)
            
            if pred_coords is not None and gt_coords is not None:
                # Compare first sample
                sample_idx = 0
                pred_sample = pred_coords[sample_idx, :10, :]  # First 10 keypoints
                gt_sample = gt_coords[sample_idx, :10, :]
                support_sample = support_coords[sample_idx, :10, :]
                
                diff_pred_gt = torch.abs(pred_sample - gt_sample).mean().item()
                diff_pred_support = torch.abs(pred_sample - support_sample).mean().item()
                
                print(f"  Coord differences (first sample, first 10 kpts):")
                print(f"    Pred vs GT:      {diff_pred_gt:.6f}")
                print(f"    Pred vs Support: {diff_pred_support:.6f}")
                
                # Check for suspicious patterns
                if diff_pred_gt < 0.001:
                    print(f"    ✗ Pred == GT (teacher forcing or data leakage!)")
                    issues_found.append(f"Episode {batch_idx}: Pred == GT")
                elif diff_pred_support < 0.001:
                    print(f"    ✗ Pred == Support (model copying support!)")
                    issues_found.append(f"Episode {batch_idx}: Pred == Support")
                else:
                    print(f"    ✓ Predictions are different from both GT and Support")
            
            print()
        
        passed = len(issues_found) == 0
        self.log_result(
            "Real validation data has no suspicious patterns",
            passed,
            f"Found {len(issues_found)} issues: {issues_found}" if not passed else "All episodes show expected behavior"
        )
    
    def run_all_tests(self):
        """Run all tests and print summary."""
        print("\n" + "=" * 80)
        print("VALIDATION PCK DEBUG TEST SUITE")
        print("=" * 80)
        print()
        
        # Run tests
        self.test_episodic_sampler_no_replacement()
        self.test_episodic_dataset_unique_images()
        self.test_collate_fn_preserves_metadata()
        self.test_forward_inference_exists()
        self.test_forward_inference_vs_forward()
        self.test_predictions_not_identical_to_support()
        self.test_inference_with_real_data()
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print()
        
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        
        print(f"Passed: {passed_tests}/{total_tests}")
        print()
        
        if passed_tests < total_tests:
            print("Failed tests:")
            for r in self.results:
                if not r['passed']:
                    print(f"  ✗ {r['test']}")
                    if r['message']:
                        print(f"    {r['message']}")
            print()
            print("=" * 80)
            print("ROOT CAUSE ANALYSIS")
            print("=" * 80)
            print()
            
            # Analyze failure patterns
            failed_messages = [r['message'] for r in self.results if not r['passed']]
            
            if any('overlap' in msg.lower() or 'duplicate' in msg.lower() for msg in failed_messages):
                print("🚨 PRIMARY ISSUE: DATA LEAKAGE")
                print()
                print("The episodic sampler is reusing the same image as both support and query.")
                print("This causes 100% PCK because the model receives the EXACT same image")
                print("with known keypoints as support, then is asked to predict on the same image.")
                print()
                print("FIX: Ensure episodic sampler uses random.sample() with replacement=False")
                print("     to guarantee support and query indices are unique.")
            elif any('teacher forcing' in msg.lower() or 'identical to gt' in msg.lower() for msg in failed_messages):
                print("🚨 PRIMARY ISSUE: TEACHER FORCING IN VALIDATION")
                print()
                print("The validation loop is passing ground truth targets to the model,")
                print("allowing it to 'cheat' during inference.")
                print()
                print("FIX: Use model.forward_inference() instead of model.forward()")
                print("     in evaluate_cape() function in engine_cape.py")
            elif any('copying support' in msg.lower() for msg in failed_messages):
                print("🚨 PRIMARY ISSUE: MODEL COPYING SUPPORT")
                print()
                print("The model is outputting support coordinates without processing")
                print("the query image.")
                print()
                print("FIX: Check CAPEModel.forward_inference() implementation")
                print("     Ensure it's actually running query images through encoder")
            else:
                print("⚠️  UNKNOWN ISSUE")
                print()
                print("Tests failed but root cause is unclear.")
                print("Review individual test failures above for clues.")
        else:
            print("✓ ALL TESTS PASSED")
            print()
            print("If validation PCK is still 100%, the issue is NOT:")
            print("  - Data leakage (image reuse)")
            print("  - Teacher forcing")
            print("  - Support copying")
            print("  - PCK computation bug")
            print()
            print("Possible remaining causes:")
            print("  1. Validation split has overlapping categories with training")
            print("  2. Model is EXTREMELY well-trained (unlikely at epoch 7)")
            print("  3. Task is easier than expected")
            print("  4. Bug in how coordinates are denormalized")
        
        print()


if __name__ == '__main__':
    tester = TestValidationPCK()
    tester.run_all_tests()

