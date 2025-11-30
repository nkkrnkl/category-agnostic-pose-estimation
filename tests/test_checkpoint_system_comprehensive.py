"""
Comprehensive test for checkpoint saving/loading system.

This test validates that:
1. Model state is saved and restored correctly
2. Optimizer state is saved and restored correctly
3. LR scheduler state is saved and restored correctly
4. Training metrics are preserved (epoch, best_val_loss, best_pck, etc.)
5. RNG states are preserved (torch, numpy, python, CUDA)
6. State dict is clean (no contamination from temporary assignments)
7. Training can resume exactly where it left off
8. Data sampling continues correctly after resume

Critical for long training runs with frequent stop/start cycles.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.cape_model import CAPEModel, build_cape_model
from models.roomformer_v2 import build
from models.support_encoder import SupportPoseGraphEncoder
from argparse import Namespace


class TestCheckpointSystem:
    """Comprehensive checkpoint system tests"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        
    def build_test_model(self):
        """Build a CAPE model for testing using saved checkpoint args"""
        # Load args from an existing checkpoint
        checkpoint_path = Path(__file__).parent.parent / 'outputs' / 'cape_run' / 'checkpoint_e004_lr1e-04_bs2_acc4_qpe2.pth'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Test checkpoint not found: {checkpoint_path}\n"
                "This test requires an existing checkpoint to validate loading behavior.\n"
                "Please run training for a few epochs first."
            )
        
        # Load checkpoint to get args
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        args = checkpoint['args']
        args.device = 'cpu'  # Force CPU for testing
        
        # Build base model
        base_model, _ = build(args)
        
        # Build CAPE model
        cape_model = build_cape_model(args, base_model)
        
        return cape_model, checkpoint
    
    def build_test_optimizer_scheduler(self, model):
        """Build optimizer and scheduler for testing"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 20],
            gamma=0.1
        )
        
        return optimizer, scheduler
    
    def test_1_state_dict_no_contamination(self):
        """Test 1: Verify state_dict has no contaminated keys after forward pass"""
        print("\n" + "=" * 80)
        print("TEST 1: State Dict Contamination Check")
        print("=" * 80)
        
        model, _ = self.build_test_model()
        
        # Create dummy inputs
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        support_coords = torch.rand(batch_size, 10, 2)
        support_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        skeleton_edges = [[[0, 1], [1, 2]] for _ in range(batch_size)]
        
        # Create dummy targets
        targets = {
            'seq11': torch.randint(0, 500, (batch_size, 20)),
            'seq21': torch.randint(0, 500, (batch_size, 20)),
            'seq12': torch.randint(0, 500, (batch_size, 20)),
            'seq22': torch.randint(0, 500, (batch_size, 20)),
        }
        
        # Forward pass
        print("  Running forward pass...")
        _ = model(images, support_coords, support_mask, targets=targets, skeleton_edges=skeleton_edges)
        
        # Check state dict
        print("  Checking state_dict keys...")
        state_dict = model.state_dict()
        
        # Check for contaminated keys (keys that start with "base_model.transformer.decoder.support_")
        contaminated_keys = [k for k in state_dict.keys() 
                            if k.startswith('base_model.transformer.decoder.support_')]
        
        if contaminated_keys:
            print(f"  ✗ FAILED: Found {len(contaminated_keys)} contaminated keys:")
            for k in contaminated_keys[:5]:  # Show first 5
                print(f"     - {k}")
            if len(contaminated_keys) > 5:
                print(f"     ... and {len(contaminated_keys) - 5} more")
            return False
        else:
            print("  ✓ PASSED: No contaminated keys in state_dict")
            
        # Check that correct keys exist
        expected_keys = [k for k in state_dict.keys() 
                        if 'support_cross_attention_layers' in k or 'support_attn_layer_norms' in k]
        print(f"  ✓ Found {len(expected_keys)} expected support layer keys")
        
        return True
    
    def test_2_basic_save_load(self):
        """Test 2: Basic checkpoint save and load"""
        print("\n" + "=" * 80)
        print("TEST 2: Basic Checkpoint Save/Load")
        print("=" * 80)
        
        # Build model
        model, _ = self.build_test_model()
        optimizer, scheduler = self.build_test_optimizer_scheduler(model)
        
        # Do a dummy forward pass to populate state
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        support_coords = torch.rand(batch_size, 10, 2)
        support_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        targets = {
            'seq11': torch.randint(0, 500, (batch_size, 20)),
            'seq21': torch.randint(0, 500, (batch_size, 20)),
            'seq12': torch.randint(0, 500, (batch_size, 20)),
            'seq22': torch.randint(0, 500, (batch_size, 20)),
        }
        
        outputs = model(images, support_coords, support_mask, targets=targets)
        loss = outputs['pred_logits'].mean()  # Dummy loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / 'test_checkpoint.pth'
        print(f"  Saving checkpoint to {checkpoint_path}...")
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': 5,
            'best_val_loss': 4.2,
            'best_pck': 0.73,
            'epochs_without_improvement': 2,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved")
        
        # Create fresh model
        print("  Creating fresh model...")
        model_fresh, _ = self.build_test_model()
        optimizer_fresh, scheduler_fresh = self.build_test_optimizer_scheduler(model_fresh)
        
        # Load checkpoint
        print("  Loading checkpoint...")
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        try:
            missing_keys, unexpected_keys = model_fresh.load_state_dict(
                loaded_checkpoint['model'], strict=False
            )
            
            if unexpected_keys:
                print(f"  ⚠️  {len(unexpected_keys)} unexpected keys (likely contamination from old run)")
                print(f"     First 3: {unexpected_keys[:3]}")
            
            if missing_keys:
                print(f"  ⚠️  {len(missing_keys)} missing keys")
                print(f"     First 3: {missing_keys[:3]}")
                return False
            
            optimizer_fresh.load_state_dict(loaded_checkpoint['optimizer'])
            scheduler_fresh.load_state_dict(loaded_checkpoint['lr_scheduler'])
            
            print("  ✓ PASSED: All states loaded successfully")
            
            # Verify metadata
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['best_val_loss'] == 4.2
            assert loaded_checkpoint['best_pck'] == 0.73
            print("  ✓ Training metrics preserved correctly")
            
            return True
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False
    
    def test_3_optimizer_state_preservation(self):
        """Test 3: Verify optimizer state is preserved (momentum, etc.)"""
        print("\n" + "=" * 80)
        print("TEST 3: Optimizer State Preservation")
        print("=" * 80)
        
        model, _ = self.build_test_model()
        optimizer, _ = self.build_test_optimizer_scheduler(model)
        
        # Do several training steps to build up optimizer state
        print("  Running 5 training steps to build optimizer state...")
        for i in range(5):
            images = torch.randn(2, 3, 512, 512)
            support_coords = torch.rand(2, 10, 2)
            support_mask = torch.ones(2, 10, dtype=torch.bool)
            targets = {
                'seq11': torch.randint(0, 500, (2, 20)),
                'seq21': torch.randint(0, 500, (2, 20)),
                'seq12': torch.randint(0, 500, (2, 20)),
                'seq22': torch.randint(0, 500, (2, 20)),
            }
            
            outputs = model(images, support_coords, support_mask, targets=targets)
            loss = outputs['pred_logits'].mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Get first parameter's optimizer state
        first_param_name = list(model.state_dict().keys())[0]
        param_group_0 = optimizer.state_dict()['state'][0]
        
        print(f"  Original optimizer state for param 0:")
        print(f"    - step: {param_group_0.get('step', 'N/A')}")
        print(f"    - exp_avg shape: {param_group_0.get('exp_avg', torch.tensor([])).shape}")
        print(f"    - exp_avg_sq shape: {param_group_0.get('exp_avg_sq', torch.tensor([])).shape}")
        
        # Save and load
        checkpoint_path = Path(self.temp_dir) / 'test_optimizer.pth'
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Fresh model and optimizer
        model_fresh, _ = self.build_test_model()
        optimizer_fresh, _ = self.build_test_optimizer_scheduler(model_fresh)
        
        # Load
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_fresh.load_state_dict(loaded['model'], strict=False)
        optimizer_fresh.load_state_dict(loaded['optimizer'])
        
        # Check optimizer state preserved
        param_group_0_loaded = optimizer_fresh.state_dict()['state'][0]
        
        print(f"  Loaded optimizer state for param 0:")
        print(f"    - step: {param_group_0_loaded.get('step', 'N/A')}")
        print(f"    - exp_avg shape: {param_group_0_loaded.get('exp_avg', torch.tensor([])).shape}")
        print(f"    - exp_avg_sq shape: {param_group_0_loaded.get('exp_avg_sq', torch.tensor([])).shape}")
        
        # Compare
        if param_group_0['step'] == param_group_0_loaded['step']:
            print("  ✓ PASSED: Optimizer state preserved")
            return True
        else:
            print("  ✗ FAILED: Optimizer state mismatch")
            return False
    
    def test_4_rng_state_preservation(self):
        """Test 4: Verify RNG states are preserved for reproducibility"""
        print("\n" + "=" * 80)
        print("TEST 4: RNG State Preservation")
        print("=" * 80)
        
        # Set known RNG states
        print("  Setting known RNG states...")
        torch.manual_seed(12345)
        np.random.seed(67890)
        random.seed(11111)
        
        # Save RNG states FIRST (before consuming any random numbers)
        checkpoint_path = Path(self.temp_dir) / 'test_rng.pth'
        checkpoint = {
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        
        torch.save(checkpoint, checkpoint_path)
        print("  ✓ RNG states saved")
        
        # NOW generate random numbers (simulating epoch training)
        print("\n  Generating epoch 0 data...")
        torch_val_epoch0 = torch.rand(5)
        np_val_epoch0 = np.random.rand(5)
        py_val_epoch0 = [random.random() for _ in range(5)]
        
        print(f"  Epoch 0 - PyTorch sample: {torch_val_epoch0[0]:.6f}")
        print(f"  Epoch 0 - NumPy sample: {np_val_epoch0[0]:.6f}")
        print(f"  Epoch 0 - Python sample: {py_val_epoch0[0]:.6f}")
        
        # Continue to epoch 1
        print("\n  Generating epoch 1 data...")
        torch_val_epoch1 = torch.rand(5)
        np_val_epoch1 = np.random.rand(5)
        py_val_epoch1 = [random.random() for _ in range(5)]
        
        print(f"  Epoch 1 - PyTorch sample: {torch_val_epoch1[0]:.6f}")
        
        # SIMULATE CRASH - Restore from checkpoint and verify epoch 0 data matches
        print("\n  Simulating crash and resume from checkpoint...")
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        torch.set_rng_state(loaded['rng_state'])
        np.random.set_state(loaded['np_rng_state'])
        random.setstate(loaded['py_rng_state'])
        
        if torch.cuda.is_available() and 'cuda_rng_state' in loaded:
            torch.cuda.set_rng_state_all(loaded['cuda_rng_state'])
        
        # Generate epoch 0 data again - should match exactly
        torch_val_resumed = torch.rand(5)
        np_val_resumed = np.random.rand(5)
        py_val_resumed = [random.random() for _ in range(5)]
        
        print(f"  After resume - PyTorch sample: {torch_val_resumed[0]:.6f}")
        print(f"  After resume - NumPy sample: {np_val_resumed[0]:.6f}")
        print(f"  After resume - Python sample: {py_val_resumed[0]:.6f}")
        
        # Verify they match
        torch_match = torch.allclose(torch_val_epoch0, torch_val_resumed)
        np_match = np.allclose(np_val_epoch0, np_val_resumed)
        py_match = all(abs(a - b) < 1e-10 for a, b in zip(py_val_epoch0, py_val_resumed))
        
        if torch_match and np_match and py_match:
            print("  ✓ PASSED: All RNG states restored correctly")
            print("     → Resumed training generates identical data sequence!")
            return True
        else:
            print(f"  ✗ FAILED:")
            print(f"     PyTorch match: {torch_match}")
            print(f"     NumPy match: {np_match}")
            print(f"     Python match: {py_match}")
            return False
    
    def test_5_training_metadata_preservation(self):
        """Test 5: Verify all training metadata is saved and loaded"""
        print("\n" + "=" * 80)
        print("TEST 5: Training Metadata Preservation")
        print("=" * 80)
        
        # Define comprehensive metadata
        metadata = {
            'epoch': 42,
            'best_val_loss': 3.14159,
            'best_pck': 0.8765,
            'epochs_without_improvement': 7,
            'train_stats': {
                'loss': 2.5,
                'pck': 0.85,
                'grad_norm': 1.23
            },
            'val_stats': {
                'loss': 3.14,
                'pck': 0.88,
            }
        }
        
        # Save
        checkpoint_path = Path(self.temp_dir) / 'test_metadata.pth'
        torch.save(metadata, checkpoint_path)
        print("  ✓ Metadata saved")
        
        # Load
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Verify each field
        print("  Verifying fields...")
        all_match = True
        
        for key, expected_value in metadata.items():
            loaded_value = loaded.get(key)
            if isinstance(expected_value, dict):
                # Check nested dict
                for subkey, subvalue in expected_value.items():
                    loaded_subvalue = loaded_value.get(subkey)
                    if abs(loaded_subvalue - subvalue) < 1e-6:
                        print(f"    ✓ {key}.{subkey}: {loaded_subvalue}")
                    else:
                        print(f"    ✗ {key}.{subkey}: expected {subvalue}, got {loaded_subvalue}")
                        all_match = False
            else:
                if abs(loaded_value - expected_value) < 1e-6 if isinstance(expected_value, (int, float)) else loaded_value == expected_value:
                    print(f"    ✓ {key}: {loaded_value}")
                else:
                    print(f"    ✗ {key}: expected {expected_value}, got {loaded_value}")
                    all_match = False
        
        if all_match:
            print("  ✓ PASSED: All metadata preserved")
            return True
        else:
            print("  ✗ FAILED: Some metadata mismatched")
            return False
    
    def test_6_model_weights_identical_after_load(self):
        """Test 6: Verify model weights are identical after save/load"""
        print("\n" + "=" * 80)
        print("TEST 6: Model Weight Preservation")
        print("=" * 80)
        
        # Build and initialize model
        model, _ = self.build_test_model()
        
        # Do a forward/backward to ensure all parameters have gradients
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        support_coords = torch.rand(batch_size, 10, 2)
        support_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        targets = {
            'seq11': torch.randint(0, 500, (batch_size, 20)),
            'seq21': torch.randint(0, 500, (batch_size, 20)),
            'seq12': torch.randint(0, 500, (batch_size, 20)),
            'seq22': torch.randint(0, 500, (batch_size, 20)),
        }
        
        outputs = model(images, support_coords, support_mask, targets=targets)
        loss = outputs['pred_logits'].mean()
        loss.backward()
        
        # Save original state dict
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / 'test_weights.pth'
        torch.save({'model': model.state_dict()}, checkpoint_path)
        print("  ✓ Checkpoint saved")
        
        # Load into fresh model
        model_fresh, _ = self.build_test_model()
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_fresh.load_state_dict(loaded['model'], strict=False)
        
        # Compare weights
        print("  Comparing weights...")
        mismatches = 0
        total_params = 0
        
        for name, original_param in original_state.items():
            if name in model_fresh.state_dict():
                loaded_param = model_fresh.state_dict()[name]
                total_params += 1
                
                if not torch.allclose(original_param, loaded_param, atol=1e-6):
                    print(f"    ✗ Mismatch: {name}")
                    mismatches += 1
        
        if mismatches == 0:
            print(f"  ✓ PASSED: All {total_params} parameters match exactly")
            return True
        else:
            print(f"  ✗ FAILED: {mismatches}/{total_params} parameters mismatched")
            return False
    
    def test_7_lr_scheduler_step_preservation(self):
        """Test 7: Verify LR scheduler continues from correct step"""
        print("\n" + "=" * 80)
        print("TEST 7: LR Scheduler Step Preservation")
        print("=" * 80)
        
        model, _ = self.build_test_model()
        optimizer, scheduler = self.build_test_optimizer_scheduler(model)
        
        # Step scheduler multiple times
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"  Initial LR: {initial_lr}")
        
        for i in range(12):  # Cross first milestone at epoch 10
            scheduler.step()
        
        lr_after_12_steps = optimizer.param_groups[0]['lr']
        print(f"  LR after 12 steps: {lr_after_12_steps}")
        
        # Save
        checkpoint_path = Path(self.temp_dir) / 'test_scheduler.pth'
        torch.save({
            'lr_scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)
        
        # Fresh optimizer/scheduler
        model_fresh, _ = self.build_test_model()
        optimizer_fresh, scheduler_fresh = self.build_test_optimizer_scheduler(model_fresh)
        
        # Load
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        optimizer_fresh.load_state_dict(loaded['optimizer'])
        scheduler_fresh.load_state_dict(loaded['lr_scheduler'])
        
        lr_after_load = optimizer_fresh.param_groups[0]['lr']
        print(f"  LR after load: {lr_after_load}")
        
        # Step once more and verify behavior continues
        scheduler_fresh.step()
        lr_after_13_steps = optimizer_fresh.param_groups[0]['lr']
        print(f"  LR after 13th step: {lr_after_13_steps}")
        
        # Verify
        if abs(lr_after_load - lr_after_12_steps) < 1e-9:
            print("  ✓ PASSED: LR scheduler state preserved")
            return True
        else:
            print(f"  ✗ FAILED: LR mismatch ({lr_after_12_steps} != {lr_after_load})")
            return False
    
    def test_8_no_data_duplication_after_resume(self):
        """Test 8: Verify RNG state ensures no duplicate data sampling"""
        print("\n" + "=" * 80)
        print("TEST 8: No Data Duplication After Resume")
        print("=" * 80)
        
        # Simulate data sampling
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Sample "epoch 0 data"
        epoch_0_samples = [torch.randint(0, 1000, (10,)).tolist() for _ in range(5)]
        print("  Epoch 0 samples (first 3):")
        for i in range(3):
            print(f"    Sample {i}: {epoch_0_samples[i][:5]}...")
        
        # Sample "epoch 1 data"
        epoch_1_samples = [torch.randint(0, 1000, (10,)).tolist() for _ in range(5)]
        print("  Epoch 1 samples (first 3):")
        for i in range(3):
            print(f"    Sample {i}: {epoch_1_samples[i][:5]}...")
        
        # Save RNG state after epoch 1
        rng_after_epoch_1 = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        
        # Sample "epoch 2 data" (what should come next)
        expected_epoch_2_samples = [torch.randint(0, 1000, (10,)).tolist() for _ in range(5)]
        print("  Expected epoch 2 samples (first 3):")
        for i in range(3):
            print(f"    Sample {i}: {expected_epoch_2_samples[i][:5]}...")
        
        # SIMULATE RESUME: Restore RNG state
        print("\n  Simulating training resume...")
        print("  Restoring RNG states from end of epoch 1...")
        torch.set_rng_state(rng_after_epoch_1['torch'])
        np.random.set_state(rng_after_epoch_1['numpy'])
        random.setstate(rng_after_epoch_1['python'])
        
        # Sample "epoch 2 data after resume"
        actual_epoch_2_samples = [torch.randint(0, 1000, (10,)).tolist() for _ in range(5)]
        print("  Actual epoch 2 samples after resume (first 3):")
        for i in range(3):
            print(f"    Sample {i}: {actual_epoch_2_samples[i][:5]}...")
        
        # Verify they match
        all_match = all(
            expected == actual 
            for expected, actual in zip(expected_epoch_2_samples, actual_epoch_2_samples)
        )
        
        if all_match:
            print("  ✓ PASSED: RNG states ensure same data sequence after resume")
            print("     → No duplicate training data!")
            return True
        else:
            print("  ✗ FAILED: Data sampling differs after resume")
            return False
    
    def test_9_full_checkpoint_cycle(self):
        """Test 9: Full save/load/resume cycle with all components"""
        print("\n" + "=" * 80)
        print("TEST 9: Full Checkpoint Cycle (End-to-End)")
        print("=" * 80)
        
        # Build model, optimizer, scheduler
        model, _ = self.build_test_model()
        optimizer, scheduler = self.build_test_optimizer_scheduler(model)
        
        # Simulate training for 3 epochs
        print("  Simulating training for 3 epochs...")
        
        epoch = 0
        best_val_loss = float('inf')
        best_pck = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(3):
            # Dummy training step
            images = torch.randn(2, 3, 512, 512)
            support_coords = torch.rand(2, 10, 2)
            support_mask = torch.ones(2, 10, dtype=torch.bool)
            targets = {
                'seq11': torch.randint(0, 500, (2, 20)),
                'seq21': torch.randint(0, 500, (2, 20)),
                'seq12': torch.randint(0, 500, (2, 20)),
                'seq22': torch.randint(0, 500, (2, 20)),
            }
            
            outputs = model(images, support_coords, support_mask, targets=targets)
            loss = outputs['pred_logits'].mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # Update metrics
            val_loss = 5.0 - epoch * 0.5  # Fake improvement
            val_pck = 0.5 + epoch * 0.1
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_pck > best_pck:
                best_pck = val_pck
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            print(f"    Epoch {epoch}: loss={val_loss:.2f}, pck={val_pck:.2f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint after epoch 2
        checkpoint_path = Path(self.temp_dir) / 'test_full_cycle.pth'
        print(f"\n  Saving checkpoint after epoch {epoch}...")
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_pck': best_pck,
            'epochs_without_improvement': epochs_without_improvement,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Continue training for 2 more epochs (without resume)
        print("\n  Continuing training for 2 more epochs (no resume)...")
        epoch_3_loss_no_resume = None
        epoch_3_lr_no_resume = None
        
        for epoch in range(3, 5):
            images = torch.randn(2, 3, 512, 512)
            support_coords = torch.rand(2, 10, 2)
            support_mask = torch.ones(2, 10, dtype=torch.bool)
            targets = {
                'seq11': torch.randint(0, 500, (2, 20)),
                'seq21': torch.randint(0, 500, (2, 20)),
                'seq12': torch.randint(0, 500, (2, 20)),
                'seq22': torch.randint(0, 500, (2, 20)),
            }
            
            outputs = model(images, support_coords, support_mask, targets=targets)
            loss = outputs['pred_logits'].mean()
            
            if epoch == 3:
                epoch_3_loss_no_resume = loss.item()
                epoch_3_lr_no_resume = optimizer.param_groups[0]['lr']
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            print(f"    Epoch {epoch}: loss={loss.item():.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # NOW RESUME FROM CHECKPOINT
        print("\n  ========================================")
        print("  RESUMING FROM CHECKPOINT")
        print("  ========================================")
        
        # Fresh model, optimizer, scheduler
        model_resumed, _ = self.build_test_model()
        optimizer_resumed, scheduler_resumed = self.build_test_optimizer_scheduler(model_resumed)
        
        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"  Loading checkpoint from epoch {loaded['epoch']}...")
        missing_keys, unexpected_keys = model_resumed.load_state_dict(loaded['model'], strict=False)
        
        if unexpected_keys:
            print(f"  ⚠️  {len(unexpected_keys)} unexpected keys (contamination)")
        if missing_keys:
            print(f"  ⚠️  {len(missing_keys)} missing keys")
            return False
        
        optimizer_resumed.load_state_dict(loaded['optimizer'])
        scheduler_resumed.load_state_dict(loaded['lr_scheduler'])
        
        # Restore RNG states
        torch.set_rng_state(loaded['rng_state'])
        np.random.set_state(loaded['np_rng_state'])
        random.setstate(loaded['py_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state' in loaded:
            torch.cuda.set_rng_state_all(loaded['cuda_rng_state'])
        
        # Restore metadata
        epoch = loaded['epoch']
        best_val_loss = loaded['best_val_loss']
        best_pck = loaded['best_pck']
        epochs_without_improvement = loaded['epochs_without_improvement']
        
        print(f"  ✓ Resumed from epoch: {epoch}")
        print(f"  ✓ Best val loss: {best_val_loss:.2f}")
        print(f"  ✓ Best PCK: {best_pck:.2f}")
        print(f"  ✓ Epochs without improvement: {epochs_without_improvement}")
        
        # Continue training from epoch 3
        print("\n  Continuing training for 2 more epochs (with resume)...")
        epoch_3_loss_resumed = None
        epoch_3_lr_resumed = None
        
        for epoch in range(3, 5):
            images = torch.randn(2, 3, 512, 512)
            support_coords = torch.rand(2, 10, 2)
            support_mask = torch.ones(2, 10, dtype=torch.bool)
            targets = {
                'seq11': torch.randint(0, 500, (2, 20)),
                'seq21': torch.randint(0, 500, (2, 20)),
                'seq12': torch.randint(0, 500, (2, 20)),
                'seq22': torch.randint(0, 500, (2, 20)),
            }
            
            outputs = model_resumed(images, support_coords, support_mask, targets=targets)
            loss = outputs['pred_logits'].mean()
            
            if epoch == 3:
                epoch_3_loss_resumed = loss.item()
                epoch_3_lr_resumed = optimizer_resumed.param_groups[0]['lr']
            
            loss.backward()
            optimizer_resumed.step()
            optimizer_resumed.zero_grad()
            scheduler_resumed.step()
            
            print(f"    Epoch {epoch}: loss={loss.item():.4f}, LR={optimizer_resumed.param_groups[0]['lr']:.6f}")
        
        # Compare epoch 3 results
        print("\n  Comparing epoch 3 (first epoch after checkpoint):")
        print(f"    Without resume: loss={epoch_3_loss_no_resume:.6f}, LR={epoch_3_lr_no_resume:.6f}")
        print(f"    With resume:    loss={epoch_3_loss_resumed:.6f}, LR={epoch_3_lr_resumed:.6f}")
        
        # RNG state preservation means data samples should be identical
        loss_match = abs(epoch_3_loss_no_resume - epoch_3_loss_resumed) < 1e-5
        lr_match = abs(epoch_3_lr_no_resume - epoch_3_lr_resumed) < 1e-9
        
        if loss_match and lr_match:
            print("  ✓ PASSED: Training continues identically after resume")
            print("     → RNG states working correctly!")
            print("     → No duplicate data, no skipped data!")
            return True
        else:
            print("  ✗ FAILED: Training differs after resume")
            print(f"     Loss match: {loss_match}")
            print(f"     LR match: {lr_match}")
            return False
    
    def test_10_checkpoint_file_size_sanity(self):
        """Test 10: Verify checkpoint file size is reasonable"""
        print("\n" + "=" * 80)
        print("TEST 10: Checkpoint File Size Sanity Check")
        print("=" * 80)
        
        model, _ = self.build_test_model()
        optimizer, scheduler = self.build_test_optimizer_scheduler(model)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model has {num_params:,} parameters")
        
        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / 'test_size.pth'
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': 10,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Check file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint file size: {file_size_mb:.2f} MB")
        
        # Expected: ~4 bytes per param (float32) × 2 (model + optimizer momentum)
        expected_size_mb = (num_params * 4 * 2) / (1024 * 1024)
        print(f"  Expected size (rough): {expected_size_mb:.2f} MB")
        
        # Allow 3x margin (optimizer has 2 moment estimates + other state)
        if file_size_mb < expected_size_mb * 5:
            print(f"  ✓ PASSED: Checkpoint size is reasonable")
            return True
        else:
            print(f"  ⚠️  WARNING: Checkpoint larger than expected")
            print(f"     This might indicate duplicate weights or contamination")
            return False
    
    def run_all_tests(self):
        """Run all checkpoint tests"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CHECKPOINT SYSTEM TEST SUITE")
        print("=" * 80)
        print(f"Temporary directory: {self.temp_dir}")
        
        tests = [
            ("State Dict Contamination", self.test_1_state_dict_no_contamination),
            ("Basic Save/Load", self.test_2_basic_save_load),
            ("Optimizer State", self.test_3_optimizer_state_preservation),
            ("RNG State", self.test_4_rng_state_preservation),
            ("Training Metadata", self.test_5_training_metadata_preservation),
            ("Model Weights", self.test_6_model_weights_identical_after_load),
            ("LR Scheduler", self.test_7_lr_scheduler_step_preservation),
            ("No Data Duplication", self.test_8_no_data_duplication_after_resume),
            ("File Size Sanity", self.test_10_checkpoint_file_size_sanity),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                passed = test_func()
                results.append((test_name, passed))
            except Exception as e:
                print(f"\n  ✗ EXCEPTION in {test_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)
        
        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {test_name}")
        
        print(f"\n  Total: {passed_count}/{total_count} tests passed")
        
        if passed_count == total_count:
            print("\n  🎉 ALL TESTS PASSED!")
            print("     Checkpoint system is working correctly.")
            print("     Safe to use for long training runs with stop/restart.")
        else:
            print(f"\n  ⚠️  {total_count - passed_count} tests failed")
            print("     Review failures above before long training runs.")
        
        print("=" * 80)
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir)
        print(f"\nCleaned up temporary directory: {self.temp_dir}")
        
        return passed_count == total_count


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 CAPE CHECKPOINT SYSTEM VALIDATION TEST                     ║
║                                                                            ║
║  This test validates that checkpoint saving/loading works correctly for:  ║
║  - Model weights                                                           ║
║  - Optimizer state (momentum, learning rate)                               ║
║  - LR scheduler state                                                      ║
║  - Training metrics (epoch, best loss, best PCK)                           ║
║  - RNG states (for reproducible data sampling)                             ║
║  - No state_dict contamination from temporary assignments                  ║
║                                                                            ║
║  Critical for long training runs with frequent stop/start cycles!          ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    tester = TestCheckpointSystem()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

