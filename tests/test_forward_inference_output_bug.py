#!/usr/bin/env python3
"""
Test to diagnose the forward_inference output bug.

HYPOTHESIS:
The forward_inference loop generates the full sequence in gen_out,
but only returns the LAST cls_output and reg_output, leading to
shape (B, 1, 2) instead of (B, seq_len, 2).
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import argparse

from datasets.mp100_cape import build_mp100_cape
from models import build_model
from models.cape_model import build_cape_model


def test_forward_inference_outputs():
    """Test that forward_inference returns full sequence, not just last token."""
    
    print("=" * 80)
    print("TEST: Forward Inference Output Shapes")
    print("=" * 80)
    print()
    
    # Build minimal args
    args = argparse.Namespace(
        dataset_root='.',
        annotation_file='annotations/mp100_split1_train.json',
        category_split_file='category_splits.json',
        image_root='data/MP-100',
        image_size=512,
        image_norm='imagenet',
        poly2seq=True,
        vocab_size=1936,
        seq_len=200,
        num_bins=44,
        add_cls_token=False,
        semantic_classes=-1,
        lr_backbone=1e-5,
        num_feature_levels=4,
        device='cpu',
        cape_mode=True,
        support_input_dim=2,
        support_hidden_dim=256,
        support_num_layers=3,
        support_dropout=0.1,
        cross_attn_heads=8,
        num_decoder_layers=6,
        hidden_dim=256,
        mask_stride=16,
        dilation=False,
        lr=1e-4,
        lr_drop=40,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        backbone='resnet50',
        position_embedding='sine',
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        nheads=8,
        num_queries=200,
        num_polys=20,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type='none',
        aux_loss=True,
        with_poly_refine=False,
        masked_attn=False,
        use_anchor=False,
        add_input_embed=False,
        input_channels=3,
        pre_decoder_pos_embed=False,
        learnable_dec_pe=False,
        dec_attn_concat_src=False,
        dec_qkv_proj=True,
        dec_layer_type='v1',
        pad_idx=None,
        inject_cls_embed=False,
    )
    
    # Build dataset to get tokenizer
    print("Building dataset...")
    dataset = build_mp100_cape('train', args)
    tokenizer = dataset.get_tokenizer()
    print(f"✓ Tokenizer: {tokenizer}")
    print()
    
    # Build model
    print("Building model...")
    base_model = build_model(args, train=False, tokenizer=tokenizer)
    if isinstance(base_model, tuple):
        base_model = base_model[0]
    model = build_cape_model(args, base_model)
    model.eval()
    print(f"✓ Model built")
    print()
    
    # Create dummy input
    print("Creating dummy input...")
    B = 2
    C, H, W = 3, 512, 512
    N = 17  # Number of support keypoints
    
    query_images = torch.randn(B, C, H, W)
    support_coords = torch.rand(B, N, 2)  # Normalized [0,1]
    support_mask = torch.ones(B, N).bool()
    print(f"  query_images: {query_images.shape}")
    print(f"  support_coords: {support_coords.shape}")
    print(f"  support_mask: {support_mask.shape}")
    print()
    
    # Run forward_inference
    print("Running forward_inference...")
    with torch.no_grad():
        try:
            outputs = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_mask,
                skeleton_edges=None
            )
        except Exception as e:
            print(f"❌ forward_inference failed: {type(e).__name__}: {e}")
            return False
    
    print("✓ forward_inference completed")
    print()
    
    # Check outputs
    print("Checking outputs...")
    print(f"  Output keys: {outputs.keys()}")
    print()
    
    pred_logits = outputs.get('pred_logits')
    pred_coords = outputs.get('coordinates')
    gen_out = outputs.get('gen_out')
    
    print("Shape Analysis:")
    print("  pred_logits:", pred_logits.shape if pred_logits is not None else "None")
    print("  pred_coords:", pred_coords.shape if pred_coords is not None else "None")
    print("  gen_out (list):", f"{len(gen_out)} batches" if isinstance(gen_out, list) else type(gen_out))
    if isinstance(gen_out, list) and len(gen_out) > 0:
        print(f"    Batch 0 length: {len(gen_out[0])} coordinates")
        print(f"    Batch 1 length: {len(gen_out[1])} coordinates")
    print()
    
    # Diagnosis
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()
    
    if pred_coords is not None:
        expected_seq_len = len(gen_out[0]) if isinstance(gen_out, list) and len(gen_out) > 0 else 0
        actual_seq_len = pred_coords.shape[1] if pred_coords.ndim >= 2 else 1
        
        print(f"Expected sequence length (from gen_out): {expected_seq_len}")
        print(f"Actual pred_coords sequence length: {actual_seq_len}")
        print()
        
        if actual_seq_len == 1 and expected_seq_len > 1:
            print("❌ BUG CONFIRMED!")
            print()
            print("The loop generates a full sequence but only returns the LAST token!")
            print()
            print("This is because:")
            print("  1. The while loop iterates multiple times")
            print("  2. Each iteration stores coordinates in gen_out[j].append(...)")
            print("  3. But cls_output and reg_output are OVERWRITTEN each iteration")
            print("  4. The final return statement uses the LAST cls_output/reg_output")
            print()
            print("FIX NEEDED:")
            print("  Accumulate cls_output and reg_output lists across iterations,")
            print("  similar to how output_hs_list is accumulated.")
            print()
            return False
        elif actual_seq_len == expected_seq_len:
            print("✅ Outputs match expected sequence length")
            return True
        else:
            print(f"⚠️  Unexpected: actual={actual_seq_len}, expected={expected_seq_len}")
            return False
    else:
        print("❌ pred_coords is None!")
        return False


if __name__ == '__main__':
    success = test_forward_inference_outputs()
    if success:
        print()
        print("✅ TEST PASSED")
    else:
        print()
        print("❌ TEST FAILED - Bug exists in forward_inference output!")

