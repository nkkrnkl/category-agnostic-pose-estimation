#!/usr/bin/env python3
"""
Test the evaluate_cape function specifically to understand the PCK@100% issue.

This test:
1. Loads a checkpoint
2. Runs evaluate_cape with detailed logging
3. Intercepts the model call to see which method is actually invoked
4. Checks if predictions match GT (indicating teacher forcing)
"""

import sys
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch
import os

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import build_episodic_dataloader
from models import build_model
from models.cape_model import build_cape_model
from models.cape_losses import build_cape_criterion
from engine_cape import evaluate_cape


def test_evaluate_cape_inference_method():
    """
    Test which inference method evaluate_cape actually uses.
    
    This is THE key test to understand the PCK@100% issue.
    """
    print("=" * 80)
    print("TEST: evaluate_cape() Inference Method")
    print("=" * 80)
    print()
    
    # Load checkpoint
    checkpoint_path = project_root / 'outputs' / 'cape_run' / 'checkpoint_e007_lr1e-04_bs2_acc4_qpe2.pth'
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  Cannot run test without a checkpoint")
        return
    
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    train_args = checkpoint['args']
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best PCK: {checkpoint.get('best_pck', 'N/A')}")
    print()
    
    # Build model
    device = torch.device('cpu')
    if not hasattr(train_args, 'poly2seq'):
        train_args.poly2seq = False
    base_model, _ = build_model(train_args)
    model = build_cape_model(train_args, base_model)
    model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"✓ Model loaded")
    
    # Check for forward_inference
    has_forward_inference = hasattr(model, 'forward_inference')
    print(f"✓ Model has forward_inference: {has_forward_inference}")
    print()
    
    # Build validation dataset (ensure args are complete)
    if not hasattr(train_args, 'semantic_classes'):
        train_args.semantic_classes = False
    val_dataset = build_mp100_cape('val', train_args)
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
    print(f"✓ Validation dataloader created: {len(val_loader)} episodes")
    print()
    
    # Build criterion
    criterion = build_cape_criterion(train_args)
    
    # Track which methods are called
    inference_calls = []
    forward_calls = []
    
    original_forward_inference = model.forward_inference if has_forward_inference else None
    original_forward = model.forward
    
    def tracked_forward_inference(*args, **kwargs):
        """Wrapper to track forward_inference calls."""
        inference_calls.append({
            'method': 'forward_inference',
            'has_targets': 'targets' in kwargs
        })
        print(f"  🔍 forward_inference called (targets in kwargs: {'targets' in kwargs})")
        return original_forward_inference(*args, **kwargs)
    
    def tracked_forward(*args, **kwargs):
        """Wrapper to track forward calls."""
        forward_calls.append({
            'method': 'forward',
            'has_targets': 'targets' in kwargs
        })
        print(f"  🔍 forward called (targets in kwargs: {'targets' in kwargs})")
        return original_forward(*args, **kwargs)
    
    # Patch model methods
    if has_forward_inference:
        model.forward_inference = tracked_forward_inference
    model.forward = tracked_forward
    
    print("=" * 80)
    print("RUNNING evaluate_cape() WITH METHOD TRACKING")
    print("=" * 80)
    print()
    
    # Run evaluate_cape
    stats = evaluate_cape(
        model=model,
        criterion=criterion,
        data_loader=val_loader,
        device=device,
        compute_pck=True,
        pck_threshold=0.2
    )
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print(f"Method calls:")
    print(f"  forward_inference: {len(inference_calls)} calls")
    print(f"  forward (teacher forcing): {len(forward_calls)} calls")
    print()
    
    print(f"Evaluation stats:")
    print(f"  PCK: {stats.get('pck', 'N/A'):.2%}")
    print(f"  Correct: {stats.get('pck_num_correct', 'N/A')}")
    print(f"  Visible: {stats.get('pck_num_visible', 'N/A')}")
    print()
    
    # Analyze results
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    pck = stats.get('pck', 0.0)
    
    if len(forward_calls) > 0 and len(inference_calls) == 0:
        print("🚨 ISSUE FOUND: Using forward (teacher forcing) instead of forward_inference")
        print()
        print("evaluate_cape is calling model.forward() with targets,")
        print("which allows the model to see ground truth during validation.")
        print()
        print("This explains the 100% PCK.")
        print()
        print("FIX: Modify evaluate_cape() in engine_cape.py to use")
        print("     model.forward_inference() without targets")
        
        return False
        
    elif len(inference_calls) > 0:
        print(f"✓ Using forward_inference ({len(inference_calls)} calls)")
        
        # Check if targets were passed (they shouldn't be)
        targets_passed = any(call['has_targets'] for call in inference_calls)
        if targets_passed:
            print("  ✗ WARNING: Targets were passed to forward_inference!")
            print("    This might cause teacher forcing behavior")
            return False
        else:
            print("  ✓ No targets passed to forward_inference")
        
        print()
        
        if pck >= 0.99:
            print("⚠️  PCK is still 100% despite using forward_inference")
            print()
            print("This rules out teacher forcing as the cause.")
            print()
            print("Remaining possibilities:")
            print("  1. Data leakage (support image == query image)")
            print("  2. Support coords == Query GT coords (annotation issue)")
            print("  3. Model has memorized the validation set")
            print("  4. Validation categories overlap with training")
            print()
            print("Run: python tests/test_validation_pck_debug.py")
            print("     for detailed data leakage analysis")
            
            return False
        else:
            print(f"✓ PCK is reasonable ({pck:.2%})")
            print("  forward_inference is working correctly")
            return True
    else:
        print("⚠️  No model methods were called?")
        print("  This is unexpected - check the test setup")
        return False


def test_evaluate_cape_with_random_model():
    """
    Test evaluate_cape with a RANDOM (untrained) model.
    
    A random model should get ~0-10% PCK on unseen categories.
    If it gets 100%, there's definitely a bug.
    """
    print("\n" + "=" * 80)
    print("TEST: evaluate_cape() with Random Model")
    print("=" * 80)
    print()
    
    print("Building random (untrained) model...")
    
    # Create minimal args
    args = argparse.Namespace(
        # Model architecture
        hidden_dim=128,  # Smaller for faster testing
        dropout=0.1,
        nheads=4,
        num_queries=50,
        enc_layers=2,
        dec_layers=2,
        dim_feedforward=512,
        pre_norm=False,
        add_cls_token=False,
        input_channels=3,
        query_pos_type='learned',
        
        # CAPE-specific
        support_encoder_layers=2,
        support_fusion_method='cross_attention',
        
        # Backbone
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        
        # Loss
        aux_loss=True,
        ce_loss_coef=1.0,
        coords_loss_coef=5.0,
        
        # Dataset
        dataset_root=str(project_root),
        category_split_file='category_splits.json',
        max_keypoints=25,
        num_bins=1000,
        vocab_size=2000,
        seq_len=200,
        image_norm=False,
        semantic_classes=False,
        poly2seq=False,
    )
    
    device = torch.device('cpu')
    base_model, _ = build_model(args)
    model = build_cape_model(args, base_model)
    model.to(device)
    model.eval()
    print(f"✓ Random model created")
    print()
    
    # Build validation dataset
    val_dataset = build_mp100_cape('val', args)
    val_loader = build_episodic_dataloader(
        base_dataset=val_dataset,
        category_split_file=str(Path(args.dataset_root) / args.category_split_file),
        split='val',
        batch_size=1,
        num_queries_per_episode=2,
        episodes_per_epoch=10,  # Small number for fast testing
        num_workers=0,
        seed=42
    )
    print(f"✓ Validation dataloader created: {len(val_loader)} episodes")
    print()
    
    # Run evaluation
    print("Running evaluate_cape with random model...")
    print("Expected PCK: 0-20% (random predictions on unseen categories)")
    print()
    
    criterion = build_cape_criterion(args)
    
    stats = evaluate_cape(
        model=model,
        criterion=criterion,
        data_loader=val_loader,
        device=device,
        compute_pck=True,
        pck_threshold=0.2
    )
    
    print()
    print(f"Results:")
    print(f"  PCK: {stats.get('pck', 'N/A'):.2%}")
    print(f"  Correct: {stats.get('pck_num_correct', 'N/A')}/{stats.get('pck_num_visible', 'N/A')}")
    print()
    
    pck = stats.get('pck', 0.0)
    
    if pck >= 0.5:
        print("🚨 CRITICAL BUG: Random model achieves ≥50% PCK!")
        print()
        print("This is statistically impossible for a random model on unseen categories.")
        print("There is definitely a bug in the validation pipeline.")
        print()
        if pck >= 0.99:
            print("PCK ≥99% indicates:")
            print("  - Teacher forcing (model sees GT during inference)")
            print("  - OR data leakage (support image == query image)")
        else:
            print("PCK is high but not perfect, indicates:")
            print("  - Partial data leakage")
            print("  - OR validation categories overlap with training")
        
        return False
    else:
        print(f"✓ Random model PCK is reasonable ({pck:.2%})")
        print("  Validation pipeline appears to be working correctly")
        return True


if __name__ == '__main__':
    import argparse
    
    print("=" * 80)
    print("VALIDATION PCK@100% DEBUG TEST SUITE")
    print("=" * 80)
    print()
    print("This test suite will identify the root cause of PCK@100% in validation.")
    print()
    
    # Test 1: Check which inference method is used
    result1 = test_evaluate_cape_inference_method()
    
    # Test 2: Check with random model
    result2 = test_evaluate_cape_with_random_model()
    
    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)
    print()
    
    if not result1:
        print("The issue is in evaluate_cape() - it's not using forward_inference correctly")
    elif not result2:
        print("The issue is systematic - even random models get high PCK")
        print("This indicates data leakage or teacher forcing")
    else:
        print("Tests passed but PCK is still 100% in training?")
        print("Check category split configuration")

