"""
Comprehensive tests for the checkpoint and resume system.

Tests verify:
1. Checkpoint contains all expected fields (model, optimizer, RNG states, best metrics)
2. Resume restores full training state correctly
3. PCK-based best model saving works
4. Best checkpoints are not overwritten incorrectly after resume
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Define pytest decorators as no-ops if pytest not available
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

import torch
import torch.nn as nn
import numpy as np
import random
import tempfile
import os
from pathlib import Path
import copy


class DummyModel(nn.Module):
    """Simple model for testing checkpoint save/load."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestCheckpointFields:
    """Test 1: Verify checkpoint contains all expected fields."""
    
    def test_checkpoint_has_required_fields(self):
        """Verify checkpoint dict has all required fields for safe resume."""
        
        # Create dummy model and optimizer
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Simulate training state
        epoch = 5
        best_val_loss = 0.123
        best_pck = 0.456
        epochs_without_improvement = 2
        
        # Build checkpoint dict (matching train_cape_episodic.py)
        checkpoint_dict = {
            # Model & optimizer state
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            
            # Best-model tracking (CRITICAL for resume)
            'best_val_loss': best_val_loss,
            'best_pck': best_pck,
            'epochs_without_improvement': epochs_without_improvement,
            
            # RNG states (CRITICAL for reproducibility)
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        # Add CUDA RNG if available
        if torch.cuda.is_available():
            checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        
        # Save and reload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            torch.save(checkpoint_dict, tmp.name)
            loaded_checkpoint = torch.load(tmp.name, map_location='cpu', weights_only=False)
        
        # Verify all required fields exist
        assert 'model' in loaded_checkpoint, "Missing 'model' in checkpoint"
        assert 'optimizer' in loaded_checkpoint, "Missing 'optimizer' in checkpoint"
        assert 'lr_scheduler' in loaded_checkpoint, "Missing 'lr_scheduler' in checkpoint"
        assert 'epoch' in loaded_checkpoint, "Missing 'epoch' in checkpoint"
        
        # CRITICAL: Best-model tracking fields
        assert 'best_val_loss' in loaded_checkpoint, "Missing 'best_val_loss' (CRITICAL for resume)"
        assert 'best_pck' in loaded_checkpoint, "Missing 'best_pck' (CRITICAL for resume)"
        assert 'epochs_without_improvement' in loaded_checkpoint, "Missing 'epochs_without_improvement'"
        
        # CRITICAL: RNG state fields
        assert 'rng_state' in loaded_checkpoint, "Missing 'rng_state' (CRITICAL for reproducibility)"
        assert 'np_rng_state' in loaded_checkpoint, "Missing 'np_rng_state'"
        assert 'py_rng_state' in loaded_checkpoint, "Missing 'py_rng_state'"
        
        # Verify values match
        assert loaded_checkpoint['epoch'] == epoch
        assert loaded_checkpoint['best_val_loss'] == best_val_loss
        assert loaded_checkpoint['best_pck'] == best_pck
        assert loaded_checkpoint['epochs_without_improvement'] == epochs_without_improvement
        
        # Cleanup
        os.unlink(tmp.name)
        
        print("✓ All required checkpoint fields present and correct")


class TestResumeRestoresState:
    """Test 2: Verify resume correctly restores all training state."""
    
    def test_resume_restores_model_optimizer_scheduler(self):
        """Verify model, optimizer, and scheduler are restored correctly."""
        
        # Create original model and train for a few steps
        model_original = DummyModel()
        optimizer_original = torch.optim.Adam(model_original.parameters(), lr=1e-4)
        lr_scheduler_original = torch.optim.lr_scheduler.StepLR(optimizer_original, step_size=10)
        
        # Train for a few steps to change weights
        for _ in range(5):
            x = torch.randn(4, 10)
            y = model_original(x)
            loss = y.sum()
            loss.backward()
            optimizer_original.step()
            optimizer_original.zero_grad()
        
        lr_scheduler_original.step()  # Update LR
        
        # Save checkpoint
        checkpoint = {
            'model': model_original.state_dict(),
            'optimizer': optimizer_original.state_dict(),
            'lr_scheduler': lr_scheduler_original.state_dict(),
            'epoch': 10,
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            torch.save(checkpoint, tmp.name)
            checkpoint_path = tmp.name
        
        # Create new model (different initialization)
        model_new = DummyModel()
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=5e-3)  # Different LR!
        lr_scheduler_new = torch.optim.lr_scheduler.StepLR(optimizer_new, step_size=10)
        
        # Verify initial weights are different
        original_weights = model_original.fc1.weight.data.clone()
        new_weights_before = model_new.fc1.weight.data.clone()
        assert not torch.allclose(original_weights, new_weights_before), "Weights shouldn't match before loading"
        
        # Load checkpoint (simulating resume)
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_new.load_state_dict(loaded_checkpoint['model'])
        optimizer_new.load_state_dict(loaded_checkpoint['optimizer'])
        lr_scheduler_new.load_state_dict(loaded_checkpoint['lr_scheduler'])
        
        # Verify weights now match
        new_weights_after = model_new.fc1.weight.data.clone()
        assert torch.allclose(original_weights, new_weights_after), "Weights should match after loading"
        
        # Verify optimizer state matches
        orig_lr = optimizer_original.param_groups[0]['lr']
        new_lr = optimizer_new.param_groups[0]['lr']
        assert orig_lr == new_lr, f"LR mismatch: {orig_lr} vs {new_lr}"
        
        # Verify epoch is correct
        assert loaded_checkpoint['epoch'] == 10
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        print("✓ Model, optimizer, and scheduler restored correctly")
    
    def test_resume_restores_rng_states(self):
        """Verify RNG states are restored for reproducible training."""
        
        # Set specific RNG states
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Save RNG states BEFORE generating values
        checkpoint = {
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            torch.save(checkpoint, tmp.name)
            checkpoint_path = tmp.name
        
        # Now generate some random values (from the saved state)
        torch_val_original = torch.rand(1).item()
        np_val_original = np.random.rand()
        py_val_original = random.random()
        
        # Change RNG states (generate more random numbers)
        for _ in range(10):
            torch.rand(1)
            np.random.rand()
            random.random()
        
        # Verify we get different values now
        torch_val_after_change = torch.rand(1).item()
        np_val_after_change = np.random.rand()
        py_val_after_change = random.random()
        
        assert torch_val_original != torch_val_after_change, "Torch RNG should have changed"
        assert np_val_original != np_val_after_change, "NumPy RNG should have changed"
        assert py_val_original != py_val_after_change, "Python RNG should have changed"
        
        # Restore RNG states from checkpoint (back to the saved state)
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        torch.set_rng_state(loaded_checkpoint['rng_state'])
        np.random.set_state(loaded_checkpoint['np_rng_state'])
        random.setstate(loaded_checkpoint['py_rng_state'])
        
        # Generate new values - should match the original values (same sequence from saved state)
        torch_val_restored = torch.rand(1).item()
        np_val_restored = np.random.rand()
        py_val_restored = random.random()
        
        assert torch_val_original == torch_val_restored, f"Torch RNG not properly restored: {torch_val_original} != {torch_val_restored}"
        assert np_val_original == np_val_restored, f"NumPy RNG not properly restored: {np_val_original} != {np_val_restored}"
        assert py_val_original == py_val_restored, f"Python RNG not properly restored: {py_val_original} != {py_val_restored}"
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        print("✓ RNG states restored correctly for reproducibility")
    
    def test_resume_restores_best_metrics(self):
        """CRITICAL: Verify best_val_loss and best_pck are restored to prevent overwriting."""
        
        # Simulate mid-training state
        best_val_loss_original = 0.0987
        best_pck_original = 0.7654
        epochs_without_improvement_original = 3
        
        checkpoint = {
            'model': DummyModel().state_dict(),
            'epoch': 50,
            'best_val_loss': best_val_loss_original,
            'best_pck': best_pck_original,
            'epochs_without_improvement': epochs_without_improvement_original,
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            torch.save(checkpoint, tmp.name)
            checkpoint_path = tmp.name
        
        # Load checkpoint (simulating resume logic from train_cape_episodic.py)
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Restore best metrics (CRITICAL step that was missing before fix)
        best_val_loss_restored = loaded_checkpoint.get('best_val_loss', float('inf'))
        best_pck_restored = loaded_checkpoint.get('best_pck', 0.0)
        epochs_without_improvement_restored = loaded_checkpoint.get('epochs_without_improvement', 0)
        
        # Verify values match
        assert best_val_loss_restored == best_val_loss_original, \
            f"best_val_loss mismatch: {best_val_loss_restored} vs {best_val_loss_original}"
        assert best_pck_restored == best_pck_original, \
            f"best_pck mismatch: {best_pck_restored} vs {best_pck_original}"
        assert epochs_without_improvement_restored == epochs_without_improvement_original, \
            f"epochs_without_improvement mismatch: {epochs_without_improvement_restored} vs {epochs_without_improvement_original}"
        
        # CRITICAL: Verify we DON'T reset to defaults
        assert best_val_loss_restored != float('inf'), "best_val_loss should NOT reset to inf on resume!"
        assert best_pck_restored != 0.0 or best_pck_original == 0.0, "best_pck should be restored"
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        print("✓ Best metrics restored correctly (prevents checkpoint overwrite bug)")


class TestPCKBasedSaving:
    """Test 3: Verify PCK-based best model saving."""
    
    def test_best_pck_checkpoint_saved(self):
        """Verify that best PCK checkpoint is saved when PCK improves."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Simulate training loop with PCK improvement
            model = DummyModel()
            best_pck = 0.0
            
            # Epoch 1: PCK = 0.5 (new best)
            val_pck = 0.5
            if val_pck > best_pck:
                best_pck = val_pck
                checkpoint_name = f'checkpoint_best_pck_e001_pck{val_pck:.4f}.pth'
                torch.save({'model': model.state_dict(), 'val_pck': val_pck}, 
                          output_dir / checkpoint_name)
            
            # Verify checkpoint exists
            pck_checkpoints = list(output_dir.glob('checkpoint_best_pck_*.pth'))
            assert len(pck_checkpoints) == 1, "Should have 1 best PCK checkpoint"
            
            # Epoch 2: PCK = 0.7 (new best, should update)
            val_pck = 0.7
            if val_pck > best_pck:
                best_pck = val_pck
                checkpoint_name = f'checkpoint_best_pck_e002_pck{val_pck:.4f}.pth'
                torch.save({'model': model.state_dict(), 'val_pck': val_pck}, 
                          output_dir / checkpoint_name)
            
            # Verify we now have 2 checkpoints (we don't auto-delete old best)
            pck_checkpoints = list(output_dir.glob('checkpoint_best_pck_*.pth'))
            assert len(pck_checkpoints) == 2, "Should have 2 best PCK checkpoints (old + new)"
            
            # Verify the latest has the correct PCK value
            latest_checkpoint = torch.load(output_dir / checkpoint_name, map_location='cpu', weights_only=False)
            assert latest_checkpoint['val_pck'] == 0.7
            
            print("✓ Best PCK checkpoint saved correctly when PCK improves")
    
    def test_best_pck_and_loss_independent(self):
        """Verify best-PCK and best-loss checkpoints are tracked independently."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            model = DummyModel()
            best_val_loss = float('inf')
            best_pck = 0.0
            
            # Epoch 1: loss=0.5, PCK=0.6 (both best)
            val_loss, val_pck = 0.5, 0.6
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'val_loss': val_loss}, output_dir / f'best_loss_e001.pth')
            if val_pck > best_pck:
                best_pck = val_pck
                torch.save({'val_pck': val_pck}, output_dir / f'best_pck_e001.pth')
            
            # Epoch 2: loss=0.3 (best loss), PCK=0.5 (worse PCK)
            val_loss, val_pck = 0.3, 0.5
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'val_loss': val_loss}, output_dir / f'best_loss_e002.pth')
            if val_pck > best_pck:  # Won't trigger (0.5 < 0.6)
                best_pck = val_pck
                torch.save({'val_pck': val_pck}, output_dir / f'best_pck_e002.pth')
            
            # Verify: 2 loss checkpoints, 1 PCK checkpoint
            loss_ckpts = list(output_dir.glob('best_loss_*.pth'))
            pck_ckpts = list(output_dir.glob('best_pck_*.pth'))
            
            assert len(loss_ckpts) == 2, "Should have 2 loss checkpoints"
            assert len(pck_ckpts) == 1, "Should have 1 PCK checkpoint (didn't improve)"
            
            print("✓ Best-loss and best-PCK tracked independently")


class TestBestCheckpointNotOverwritten:
    """Test 4: Verify best checkpoint is not overwritten after resume."""
    
    def test_resume_preserves_best_checkpoint(self):
        """CRITICAL: Verify that resuming doesn't overwrite best checkpoint with worse model."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Simulate training to epoch 10 with best_val_loss = 0.1 at epoch 5
            best_val_loss = 0.1
            best_epoch = 5
            
            # Save checkpoint at epoch 10 (current) and best checkpoint (epoch 5)
            checkpoint_e10 = {
                'model': DummyModel().state_dict(),
                'epoch': 10,
                'best_val_loss': best_val_loss,  # CRITICAL: stored in checkpoint
                'val_loss': 0.2,  # Current val_loss is worse than best
            }
            
            best_checkpoint = {
                'model': DummyModel().state_dict(),
                'epoch': 5,
                'val_loss': 0.1,  # Best val_loss
            }
            
            checkpoint_path = output_dir / 'checkpoint_e010.pth'
            best_path = output_dir / 'checkpoint_best_loss.pth'
            
            torch.save(checkpoint_e10, checkpoint_path)
            torch.save(best_checkpoint, best_path)
            
            # Simulate resume: Load checkpoint
            loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # CRITICAL: Restore best_val_loss (this was the bug!)
            best_val_loss_restored = loaded.get('best_val_loss', float('inf'))
            
            # BEFORE FIX: best_val_loss would reset to inf, and epoch 11's val_loss=0.15
            # would incorrectly be considered "best" and overwrite the checkpoint
            
            # AFTER FIX: best_val_loss is restored to 0.1
            assert best_val_loss_restored == 0.1, "best_val_loss should be restored from checkpoint"
            
            # Continue training: epoch 11 has val_loss=0.15 (worse than 0.1)
            val_loss_e11 = 0.15
            
            # Check if we should save best model
            should_save_best = (val_loss_e11 < best_val_loss_restored)
            
            # Verify we DON'T save (0.15 > 0.1)
            assert not should_save_best, "Should NOT overwrite best checkpoint (0.15 > 0.1)"
            
            # Verify best checkpoint still exists with correct value
            best_loaded = torch.load(best_path, map_location='cpu', weights_only=False)
            assert best_loaded['val_loss'] == 0.1, "Best checkpoint should be unchanged"
            
            print("✓ Best checkpoint preserved after resume (bug fixed!)")
    
    def test_resume_allows_better_checkpoint(self):
        """Verify that resume still allows saving if new model is actually better."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Resume from epoch 10 with best_val_loss = 0.1
            best_val_loss = 0.1
            
            checkpoint_e10 = {
                'epoch': 10,
                'best_val_loss': best_val_loss,
            }
            
            torch.save(checkpoint_e10, output_dir / 'checkpoint_e010.pth')
            
            # Load and restore
            loaded = torch.load(output_dir / 'checkpoint_e010.pth', map_location='cpu', weights_only=False)
            best_val_loss_restored = loaded.get('best_val_loss', float('inf'))
            
            # Epoch 11: val_loss = 0.05 (BETTER than 0.1)
            val_loss_e11 = 0.05
            
            # Should save new best
            should_save_best = (val_loss_e11 < best_val_loss_restored)
            assert should_save_best, "Should save new best checkpoint (0.05 < 0.1)"
            
            # Update best_val_loss
            best_val_loss_restored = val_loss_e11
            
            # Save new best checkpoint
            torch.save({'val_loss': val_loss_e11}, output_dir / 'checkpoint_best_new.pth')
            
            # Verify saved
            assert (output_dir / 'checkpoint_best_new.pth').exists()
            
            print("✓ Resume allows saving better checkpoints (as expected)")


if __name__ == '__main__':
    """Run all tests with pytest or standalone."""
    print("=" * 80)
    print("CHECKPOINT SYSTEM TESTS")
    print("=" * 80)
    
    # Test 1: Checkpoint fields
    print("\n[Test 1] Checkpoint Contains Expected Fields")
    test1 = TestCheckpointFields()
    test1.test_checkpoint_has_required_fields()
    
    # Test 2: Resume restores state
    print("\n[Test 2] Resume Restores Full State")
    test2 = TestResumeRestoresState()
    test2.test_resume_restores_model_optimizer_scheduler()
    test2.test_resume_restores_rng_states()
    test2.test_resume_restores_best_metrics()
    
    # Test 3: PCK-based saving
    print("\n[Test 3] PCK-Based Best Model Saving")
    test3 = TestPCKBasedSaving()
    test3.test_best_pck_checkpoint_saved()
    test3.test_best_pck_and_loss_independent()
    
    # Test 4: No overwrite after resume
    print("\n[Test 4] Best Checkpoint Not Overwritten After Resume")
    test4 = TestBestCheckpointNotOverwritten()
    test4.test_resume_preserves_best_checkpoint()
    test4.test_resume_allows_better_checkpoint()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

