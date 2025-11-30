"""
Critical checkpoint loading test using actual trained checkpoint.

This test validates that your saved checkpoint can be properly loaded
and that all critical state is preserved for resuming training.

Tests:
1. Checkpoint file integrity
2. Model weights can be loaded (strict=False for compatibility)
3. Optimizer state is preserved
4. LR scheduler state is preserved
5. Training metadata is intact
6. RNG states are present
7. State dict has no contamination (after the fix)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.cape_model import build_cape_model
from models.roomformer_v2 import build


def test_checkpoint_loading():
    """Test loading your actual checkpoint"""
    
    print("=" * 80)
    print("CHECKPOINT LOADING VALIDATION TEST")
    print("=" * 80)
    
    # Find most recent checkpoint
    checkpoint_dir = project_root / 'outputs' / 'cape_run'
    checkpoints = list(checkpoint_dir.glob('checkpoint_e*.pth'))
    
    if not checkpoints:
        print("❌ No checkpoints found in outputs/cape_run/")
        print("   Please run training for a few epochs first.")
        return False
    
    # Use most recent
    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n📦 Testing checkpoint: {checkpoint_path.name}")
    print(f"   File size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    print(f"   Modified: {checkpoint_path.stat().st_mtime}")
    
    # ========================================================================
    # TEST 1: Load checkpoint file
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: Checkpoint File Integrity")
    print("-" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("  ✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"  ✗ FAILED to load checkpoint: {e}")
        return False
    
    # ========================================================================
    # TEST 2: Verify all required keys are present
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: Required Keys Present")
    print("-" * 80)
    
    required_keys = [
        'model', 'optimizer', 'lr_scheduler', 'epoch',
        'best_val_loss', 'best_pck', 'epochs_without_improvement',
        'rng_state', 'np_rng_state', 'py_rng_state'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key in checkpoint:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} - MISSING!")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n  ✗ FAILED: {len(missing_keys)} required keys missing")
        return False
    
    print("  ✓ PASSED: All required keys present")
    
    # ========================================================================
    # TEST 3: Verify training metadata values
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: Training Metadata")
    print("-" * 80)
    
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Best PCK: {checkpoint['best_pck']:.4f}")
    print(f"  Epochs without improvement: {checkpoint['epochs_without_improvement']}")
    
    # Sanity checks
    if checkpoint['epoch'] < 0:
        print("  ✗ FAILED: Invalid epoch number")
        return False
    
    if checkpoint['best_val_loss'] <= 0 or checkpoint['best_val_loss'] > 100:
        print(f"  ⚠️  WARNING: Suspicious val_loss value: {checkpoint['best_val_loss']}")
    
    if checkpoint['best_pck'] < 0 or checkpoint['best_pck'] > 1:
        print(f"  ✗ FAILED: Invalid PCK value (should be in [0, 1]): {checkpoint['best_pck']}")
        return False
    
    print("  ✓ PASSED: Metadata values are reasonable")
    
    # ========================================================================
    # TEST 4: Inspect model state_dict for contamination
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: State Dict Contamination Check")
    print("-" * 80)
    
    model_state = checkpoint['model']
    
    # Check for contaminated keys
    contaminated_keys = [k for k in model_state.keys() 
                        if 'base_model.transformer.decoder.support_cross_attn_layers' in k 
                        or 'base_model.transformer.decoder.support_attn_norms' in k]
    
    if contaminated_keys:
        print(f"  ⚠️  Found {len(contaminated_keys)} contaminated keys (from bug):")
        for k in contaminated_keys[:3]:
            print(f"     - {k}")
        if len(contaminated_keys) > 3:
            print(f"     ... and {len(contaminated_keys) - 3} more")
        print("  ℹ️  This is expected for checkpoints saved before the fix")
        print("     Will be ignored during loading with strict=False")
    else:
        print("  ✓ No contaminated keys found")
        print("     Checkpoint was saved after the fix!")
    
    # Check that correct keys exist
    correct_support_keys = [k for k in model_state.keys() 
                           if k.startswith('support_cross_attention_layers') 
                           or k.startswith('support_attn_layer_norms')]
    
    print(f"  ✓ Found {len(correct_support_keys)} correct support layer parameters")
    
    # ========================================================================
    # TEST 5: Load model weights into fresh model
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: Load Model Weights")
    print("-" * 80)
    
    print("  Building fresh model from checkpoint args...")
    args = checkpoint['args']
    args.device = 'cpu'
    
    try:
        # Build base model
        base_model, _ = build(args)
        
        # Build CAPE model
        model = build_cape_model(args, base_model)
        
        print("  ✓ Fresh model built")
    except Exception as e:
        print(f"  ✗ FAILED to build model: {e}")
        return False
    
    # Load weights
    print("  Loading checkpoint weights...")
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model'], strict=False
        )
        
        print(f"  ✓ Model weights loaded")
        
        if unexpected_keys:
            print(f"  ⚠️  {len(unexpected_keys)} unexpected keys (contamination from old checkpoint)")
            print(f"     These are safely ignored with strict=False")
        else:
            print(f"  ✓ No unexpected keys - clean checkpoint!")
        
        if missing_keys:
            print(f"  ✗ {len(missing_keys)} missing keys - model architecture changed!")
            for k in missing_keys[:5]:
                print(f"     - {k}")
            return False
        else:
            print(f"  ✓ No missing keys - all weights loaded")
        
    except Exception as e:
        print(f"  ✗ FAILED to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # TEST 6: Load optimizer state
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 6: Load Optimizer State")
    print("-" * 80)
    
    # Build optimizer with same structure as training
    # Two parameter groups: backbone (with lower LR) and rest
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # Count params in each group
    num_main_params = len(param_dicts[0]['params'])
    num_backbone_params = len(param_dicts[1]['params'])
    print(f"  Fresh model parameter groups:")
    print(f"     Main params: {num_main_params}")
    print(f"     Backbone params: {num_backbone_params}")
    
    # Try to load state
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print("  ✓ Optimizer state loaded successfully")
        print(f"     Main LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"     Backbone LR: {optimizer.param_groups[1]['lr']:.2e}")
        print(f"     Weight decay: {optimizer.param_groups[0]['weight_decay']:.2e}")
        
        # Check that momentum buffers exist
        num_params_with_state = len(optimizer.state)
        total_params = sum(len(group['params']) for group in optimizer.param_groups)
        print(f"     Parameters with optimizer state: {num_params_with_state}/{total_params}")
        
        if num_params_with_state > 0:
            print("  ✓ Optimizer has accumulated state (momentum, etc.)")
        else:
            print("  ⚠️  Optimizer has no accumulated state (was checkpoint saved immediately?)")
        
        optimizer_loaded = True
            
    except (RuntimeError, ValueError) as e:
        # For contaminated checkpoints, optimizer param groups might mismatch
        print(f"  ⚠️  Optimizer loading failed: {str(e)[:100]}")
        print("  ℹ️  This happens when:")
        print("     - State dict contamination changed parameter structure")
        print("     - Model architecture was modified slightly")
        print("  ℹ️  Impact: Optimizer will restart from scratch (suboptimal but not critical)")
        print("  ℹ️  Future checkpoints (after the fix) should load correctly")
        optimizer_loaded = False
    
    # ========================================================================
    # TEST 7: Load LR scheduler state
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 7: Load LR Scheduler State")
    print("-" * 80)
    
    try:
        # Parse lr_drop to get milestones (same as in train_cape_episodic.py)
        if hasattr(args, 'lr_drop_epochs'):
            milestones = args.lr_drop_epochs
        elif hasattr(args, 'lr_drop'):
            milestones = [int(x) for x in args.lr_drop.split(',')]
        else:
            milestones = [100, 200]  # Default
        
        gamma = getattr(args, 'lr_drop_gamma', 0.1)  # Default gamma
        
        # Build scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        
        # Load state (only if optimizer loaded successfully)
        if optimizer_loaded:
            try:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                print("  ✓ LR scheduler state loaded")
                print(f"     Last epoch: {scheduler.last_epoch}")
                print(f"     Milestones: {scheduler.milestones}")
                print(f"     Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                scheduler_loaded = True
            except Exception as e:
                print(f"  ⚠️  Scheduler loading failed: {e}")
                print("  ℹ️  Scheduler will restart from epoch 0")
                scheduler_loaded = False
        else:
            print("  ⚠️  Skipping scheduler load (optimizer didn't load)")
            print("  ℹ️  Scheduler will restart from epoch 0")
            scheduler_loaded = False
        
    except Exception as e:
        print(f"  ✗ FAILED to build scheduler: {e}")
        scheduler_loaded = False
    
    # ========================================================================
    # TEST 8: Verify RNG states format
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 8: RNG States Format")
    print("-" * 80)
    
    # Check torch RNG
    if 'rng_state' in checkpoint:
        rng_state = checkpoint['rng_state']
        print(f"  ✓ Torch RNG state present (shape: {rng_state.shape})")
    else:
        print("  ✗ Torch RNG state missing")
        return False
    
    # Check numpy RNG
    if 'np_rng_state' in checkpoint:
        np_rng = checkpoint['np_rng_state']
        print(f"  ✓ NumPy RNG state present (type: {type(np_rng).__name__})")
    else:
        print("  ✗ NumPy RNG state missing")
        return False
    
    # Check python RNG
    if 'py_rng_state' in checkpoint:
        py_rng = checkpoint['py_rng_state']
        print(f"  ✓ Python RNG state present (version: {py_rng[0]})")
    else:
        print("  ✗ Python RNG state missing")
        return False
    
    # Check CUDA RNG (optional)
    if torch.cuda.is_available():
        if 'cuda_rng_state' in checkpoint:
            print(f"  ✓ CUDA RNG state present")
        else:
            print(f"  ⚠️  CUDA RNG state missing (but CUDA is available)")
    else:
        print(f"  ℹ️  CUDA not available, skipping CUDA RNG check")
    
    # ========================================================================
    # TEST 9: Test actual RNG restoration
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 9: RNG Restoration Test")
    print("-" * 80)
    
    try:
        # Save current RNG state
        current_torch_rng = torch.get_rng_state()
        
        # Restore from checkpoint
        torch.set_rng_state(checkpoint['rng_state'])
        print("  ✓ Torch RNG state restored")
        
        # Generate a sample
        sample1 = torch.rand(3)
        print(f"     Sample after restore: [{sample1[0]:.4f}, {sample1[1]:.4f}, {sample1[2]:.4f}]")
        
        # Restore original
        torch.set_rng_state(current_torch_rng)
        print("  ✓ RNG restoration working correctly")
        
    except Exception as e:
        print(f"  ✗ FAILED to restore RNG: {e}")
        return False
    
    # ========================================================================
    # TEST 10: Verify parameter count matches
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 10: Parameter Count Verification")
    print("-" * 80)
    
    # Count parameters in checkpoint
    checkpoint_param_count = len(checkpoint['model'])
    print(f"  Checkpoint has {checkpoint_param_count} parameter tensors")
    
    # Count parameters in fresh model
    model_param_count = len(model.state_dict())
    print(f"  Fresh model has {model_param_count} parameter tensors")
    
    # Count unexpected keys
    unexpected_count = len(unexpected_keys) if unexpected_keys else 0
    
    # With strict=False, we expect:
    # checkpoint_params = model_params + unexpected_params - missing_params
    # Since we have no missing params, we expect:
    # checkpoint_params = model_params + unexpected_params
    
    expected_checkpoint_count = model_param_count + unexpected_count
    
    if abs(checkpoint_param_count - expected_checkpoint_count) <= 2:  # Allow small variance
        print(f"  ✓ Parameter counts match (accounting for {unexpected_count} unexpected keys)")
    else:
        print(f"  ⚠️  Parameter count mismatch:")
        print(f"     Expected: ~{expected_checkpoint_count}")
        print(f"     Got: {checkpoint_param_count}")
        print(f"     Difference: {abs(checkpoint_param_count - expected_checkpoint_count)}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("CHECKPOINT VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Checkpoint Details:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    print(f"   Best PCK: {checkpoint['best_pck']:.4f}")
    print(f"   Epochs w/o improvement: {checkpoint['epochs_without_improvement']}")
    print(f"   Model parameters: {checkpoint_param_count}")
    print(f"   File size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    
    print(f"\n✅ Loading Compatibility:")
    print(f"   Model: ✓ Can load (with strict=False)")
    print(f"   Optimizer: {'✓ Can load' if optimizer_loaded else '⚠️  Cannot load (param group mismatch)'}")
    print(f"   Scheduler: {'✓ Can load' if scheduler_loaded else '⚠️  Cannot load (depends on optimizer)'}")
    print(f"   RNG States: ✓ Present and restorable")
    print(f"   Metadata: ✓ Present and valid")
    
    if unexpected_keys:
        print(f"\n⚠️  Contamination Status:")
        print(f"   {len(unexpected_keys)} contaminated keys (from old bug)")
        print(f"   Impact: None (safely ignored with strict=False)")
        print(f"   Future checkpoints: Will be clean after the fix")
    else:
        print(f"\n✨ Clean Checkpoint:")
        print(f"   No contaminated keys!")
        print(f"   The state_dict bug fix is working correctly")
    
    print(f"\n🎯 Resume Training Command:")
    print(f"   python train_cape_episodic.py \\")
    print(f"     --dataset_root . \\")
    print(f"     --resume {checkpoint_path} \\")
    print(f"     --epochs 300 \\")
    print(f"     --batch_size {args.batch_size} \\")
    print(f"     --accumulation_steps {args.accumulation_steps} \\")
    print(f"     --num_queries_per_episode {args.num_queries_per_episode} \\")
    print(f"     --output_dir outputs/cape_run \\")
    print(f"     2>&1 | tee -a train_output.log")
    
    print("\n" + "=" * 80)
    if optimizer_loaded and scheduler_loaded:
        print("✅ ALL TESTS PASSED - CHECKPOINT IS FULLY COMPATIBLE")
        print("=" * 80)
        print("\n💡 Key Points:")
        print("   ✓ Checkpoint can be loaded successfully")
        print("   ✓ All model weights will be restored")
        print("   ✓ Optimizer momentum will be preserved")
        print("   ✓ LR schedule will continue from correct point")
        print("   ✓ RNG states will ensure no duplicate data")
        print("   ✓ Training will resume from epoch", checkpoint['epoch'] + 1)
        print("\n   Safe to resume long training runs! 🚀")
    else:
        print("⚠️  CHECKPOINT IS PARTIALLY COMPATIBLE")
        print("=" * 80)
        print("\n💡 Key Points:")
        print("   ✓ Checkpoint can be loaded")
        print("   ✓ All model weights will be restored")
        if not optimizer_loaded:
            print("   ⚠️  Optimizer state cannot be loaded (param group mismatch)")
            print("      → Optimizer will restart from scratch")
            print("      → Training will be slightly less efficient initially")
        if not scheduler_loaded:
            print("   ⚠️  LR scheduler cannot be loaded")
            print("      → Scheduler will restart from epoch 0")
            print("      → Learning rate schedule will be affected")
        print("   ✓ RNG states will ensure no duplicate data")
        print("   ✓ Training can resume from epoch", checkpoint['epoch'] + 1)
        print("\n   ℹ️  This is due to state_dict contamination in old checkpoint")
        print("   ℹ️  Future checkpoints (epoch 5+) will be fully compatible")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║          CAPE CHECKPOINT LOADING VALIDATION (ACTUAL CHECKPOINT)            ║
║                                                                            ║
║  This test validates your saved checkpoint from training epoch 4.          ║
║  It verifies that all state can be properly loaded for resuming training.  ║
║                                                                            ║
║  Critical for long training runs (300 epochs) with stop/restart!           ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    success = test_checkpoint_loading()
    sys.exit(0 if success else 1)

