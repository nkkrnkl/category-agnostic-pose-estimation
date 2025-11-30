#!/usr/bin/env python3
"""
FOCUSED DIAGNOSTIC: Why is validation PCK stuck at 100%?

This script runs the ACTUAL validation code with debug instrumentation to identify:
1. Are we using forward_inference or forward (teacher forcing)?
2. Are predictions identical to GT?
3. Are predictions identical to support?
4. Is there image ID overlap between support and query?

Run this to get a definitive answer.
"""

import sys
import torch
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set
from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import build_episodic_dataloader
from models import build_model
from models.cape_model import build_cape_model
from models.cape_losses import build_cape_criterion
import argparse


def diagnose_pck_100():
    """Run diagnostic on the PCK@100% issue."""
    
    print("=" * 80)
    print("PCK@100% DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Load checkpoint
    checkpoint_path = project_root / 'outputs' / 'cape_run' / 'checkpoint_e007_lr1e-04_bs2_acc4_qpe2.pth'
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Cannot diagnose without a checkpoint")
        return
    
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    
    # Ensure args are complete
    if not hasattr(args, 'semantic_classes'):
        args.semantic_classes = False
    if not hasattr(args, 'poly2seq'):
        args.poly2seq = False
    
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best PCK: {checkpoint.get('best_pck', 'N/A')}")
    print()
    
    # Build model WITH tokenizer (critical fix)
    print("Building model...")
    device = torch.device('cpu')  # Use CPU for easier debugging
    
    # Build dataset to get tokenizer
    temp_dataset = build_mp100_cape('train', args)
    tokenizer = temp_dataset.get_tokenizer()
    print(f"  Tokenizer: {tokenizer}")
    print(f"  Vocab size: {len(tokenizer) if tokenizer else 'N/A'}")
    
    # Build base model WITH tokenizer
    base_model, _ = build_model(args, tokenizer=tokenizer)
    model = build_cape_model(args, base_model)
    model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"✓ Model loaded with tokenizer")
    print()
    
    # Check for forward_inference
    has_forward_inference = hasattr(model, 'forward_inference')
    print(f"Model has forward_inference: {has_forward_inference}")
    if not has_forward_inference:
        print("  🚨 ISSUE FOUND: forward_inference doesn't exist!")
        print("     evaluate_cape will use teacher forcing")
        print("     This explains PCK@100%")
        return
    print()
    
    # Build validation dataloader
    print("Building validation dataloader...")
    val_dataset = build_mp100_cape('val', args)
    val_loader = build_episodic_dataloader(
        base_dataset=val_dataset,
        category_split_file=str(Path(args.dataset_root) / args.category_split_file),
        split='val',
        batch_size=1,
        num_queries_per_episode=2,
        episodes_per_epoch=5,  # Just 5 episodes for diagnosis
        num_workers=0,
        seed=42
    )
    print(f"✓ Dataloader created: {len(val_loader)} episodes")
    print()
    
    print("=" * 80)
    print("ANALYZING VALIDATION EPISODES")
    print("=" * 80)
    print()
    
    issues_found = []
    
    for batch_idx, batch in enumerate(val_loader):
        print(f"\nEpisode {batch_idx}:")
        print("-" * 40)
        
        # Extract batch data
        support_coords = batch['support_coords']
        support_masks = batch['support_masks']
        query_images = batch['query_images']
        query_targets = batch['query_targets']
        support_meta = batch.get('support_metadata', [])
        query_meta = batch.get('query_metadata', [])
        
        # Check 1: Image ID overlap
        if support_meta and query_meta:
            support_ids = [m.get('image_id', 'N/A') for m in support_meta]
            query_ids = [m.get('image_id', 'N/A') for m in query_meta]
            
            print(f"Support IDs: {support_ids}")
            print(f"Query IDs:   {query_ids}")
            
            overlap = set(support_ids) & set(query_ids)
            if overlap:
                print(f"  🚨 DATA LEAKAGE: {len(overlap)} overlapping image IDs!")
                issues_found.append(f"Episode {batch_idx}: Data leakage")
            else:
                print(f"  ✓ No image ID overlap")
        
        # Check 2: Run inference and compare coordinates
        with torch.no_grad():
            # Use forward_inference (autoregressive)
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=batch.get('support_skeletons', None)
            )
        
        pred_coords = predictions.get('coordinates', None)
        gt_coords = query_targets.get('target_seq', None)
        
        if pred_coords is not None and gt_coords is not None:
            # Analyze first sample
            sample_idx = 0
            pred_sample = pred_coords[sample_idx, :10, :]  # First 10 coords
            gt_sample = gt_coords[sample_idx, :10, :]
            support_sample = support_coords[sample_idx, :10, :]
            
            diff_pred_gt = torch.abs(pred_sample - gt_sample).mean().item()
            diff_pred_support = torch.abs(pred_sample - support_sample).mean().item()
            
            print(f"Coordinate differences (sample 0, first 10 coords):")
            print(f"  Pred vs GT:      {diff_pred_gt:.6f}")
            print(f"  Pred vs Support: {diff_pred_support:.6f}")
            
            # Check for issues
            if diff_pred_gt < 0.001:
                print(f"  🚨 ISSUE: Pred == GT (teacher forcing!)")
                issues_found.append(f"Episode {batch_idx}: Pred == GT")
            elif diff_pred_support < 0.001:
                print(f"  🚨 ISSUE: Pred == Support (copying support!)")
                issues_found.append(f"Episode {batch_idx}: Pred == Support")
            elif diff_pred_gt < 0.05:
                print(f"  ⚠️  WARNING: Pred very close to GT (suspicious)")
            else:
                print(f"  ✓ Predictions are different from GT and Support")
    
    print()
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print()
    
    if not issues_found:
        print("✓ NO ISSUES FOUND in validation pipeline!")
        print()
        print("The validation code appears to be working correctly:")
        print("  - Using forward_inference (not teacher forcing)")
        print("  - No image ID overlap")
        print("  - Predictions differ from GT and support")
        print()
        print("If PCK is still 100%, possible explanations:")
        print("  1. Model is genuinely VERY well trained")
        print("  2. Validation categories are too easy")
        print("  3. There's a subtle bug in coordinate denormalization")
        print("  4. The PCK threshold is too lenient")
        print()
        print("Recommendations:")
        print("  - Try with a RANDOM (untrained) model → should get ~0-20% PCK")
        print("  - Check if validation categories overlap with training")
        print("  - Visualize some predictions to see if they're actually correct")
    else:
        print(f"🚨 FOUND {len(issues_found)} ISSUES:")
        for issue in issues_found:
            print(f"  - {issue}")
        print()
        
        if any('Data leakage' in issue for issue in issues_found):
            print("ROOT CAUSE: Data Leakage")
            print("  The same image appears as both support and query")
            print("  FIX: Check episodic_sampler.py - ensure random.sample without replacement")
        elif any('Pred == GT' in issue for issue in issues_found):
            print("ROOT CAUSE: Teacher Forcing")
            print("  Model is receiving ground truth during inference")
            print("  FIX: Ensure evaluate_cape uses forward_inference, not forward")
        elif any('Pred == Support' in issue for issue in issues_found):
            print("ROOT CAUSE: Copying Support")
            print("  Model is copying support keypoints without processing query image")
            print("  FIX: Check CAPEModel.forward_inference implementation")
    
    print()


if __name__ == '__main__':
    diagnose_pck_100()

