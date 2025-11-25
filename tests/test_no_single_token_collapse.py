#!/usr/bin/env python3
"""
Regression Test: Ensure no single-token collapse during inference.

Tests that the model generates sequences of reasonable length,
not collapsing to a single token due to output bugs.
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets.episodic_sampler import build_episodic_dataloader
from datasets.mp100_cape import build_mp100_cape


def test_inference_with_real_data():
    """Test inference on real validation data."""
    
    print("=" * 80)
    print("TEST: No Single-Token Collapse")
    print("=" * 80)
    print()
    
    checkpoint_path = Path('outputs/cape_run/checkpoint_e010_lr1e-04_bs2_acc4_qpe2.pth')
    
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found, skipping")
        return True
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    
    from models import build_model
    from models.cape_model import build_cape_model
    
    # Build dataset and model
    dataset = build_mp100_cape('val', args)
    tokenizer = dataset.get_tokenizer()
    
    build_result = build_model(args, train=False, tokenizer=tokenizer)
    if isinstance(build_result, tuple):
        base_model = build_result[0]
    else:
        base_model = build_result
    model = build_cape_model(args, base_model)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    # Build dataloader
    category_split_file = Path(args.dataset_root) / args.category_split_file
    dataloader = build_episodic_dataloader(
        base_dataset=dataset,
        category_split_file=str(category_split_file),
        split='val',
        batch_size=1,
        num_queries_per_episode=2,
        episodes_per_epoch=3,
        num_workers=0,
        seed=42
    )
    
    print("Running inference on 3 episodes...")
    print()
    
    all_passed = True
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            support_coords = batch['support_coords']
            support_masks = batch['support_masks']
            query_images = batch['query_images']
            
            # Run inference
            try:
                outputs = model.forward_inference(
                    samples=query_images,
                    support_coords=support_coords,
                    support_mask=support_masks,
                    skeleton_edges=None
                )
            except Exception as e:
                print(f"Episode {batch_idx}: ❌ FAIL - Exception: {e}")
                all_passed = False
                continue
            
            # Check output
            pred_coords = outputs.get('coordinates')
            if pred_coords is None:
                print(f"Episode {batch_idx}: ❌ FAIL - No coordinates returned")
                all_passed = False
                continue
            
            seq_len = pred_coords.shape[1]
            
            # CRITICAL TEST: Sequence length must be > 1
            if seq_len <= 1:
                print(f"Episode {batch_idx}: ❌ FAIL - Sequence collapsed to {seq_len} token(s)")
                print(f"  pred_coords shape: {pred_coords.shape}")
                all_passed = False
            else:
                print(f"Episode {batch_idx}: ✅ PASS - Generated {seq_len} tokens")
    
    print()
    print("=" * 80)
    
    if all_passed:
        print("✅ TEST PASSED: No single-token collapse detected")
        print()
        print("All episodes generated sequences with length > 1")
        print("The forward_inference output bug is FIXED!")
        return True
    else:
        print("❌ TEST FAILED: Single-token collapse detected!")
        print()
        print("The bug has REGRESSED or was not fully fixed.")
        print("Check models/roomformer_v2.py::forward_inference()")
        return False


if __name__ == '__main__':
    success = test_inference_with_real_data()
    exit(0 if success else 1)

