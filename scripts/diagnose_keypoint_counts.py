"""
Diagnostic script to investigate keypoint count mismatches between predictions and ground truth.

This script performs a comprehensive analysis to determine if the model generates more
keypoints than expected, which would cause PCK computation errors and shape mismatches.

Usage:
    # Single sample diagnostic
    DEBUG_KEYPOINT_COUNT=1 python scripts/diagnose_keypoint_counts.py \
        --checkpoint outputs/cape_run/checkpoint_e024_lr1e-04_bs2_acc4_qpe2.pth \
        --num-samples 1
    
    # Per-category analysis
    python scripts/diagnose_keypoint_counts.py \
        --checkpoint outputs/cape_run/checkpoint_e024_lr1e-04_bs2_acc4_qpe2.pth \
        --per-category \
        --num-samples 10
    
    # Token sequence inspection
    python scripts/diagnose_keypoint_counts.py \
        --checkpoint outputs/cape_run/checkpoint_e024_lr1e-04_bs2_acc4_qpe2.pth \
        --dump-tokens \
        --num-samples 3
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.mp100_cape import MP100CAPE
from datasets.episodic_sampler import build_episodic_dataloader
from datasets.token_types import TokenType
from models import build_model
from models.cape_model import build_cape_model
from util.sequence_utils import extract_keypoints_from_predictions
from models.engine_cape import extract_keypoints_from_sequence


def load_checkpoint_and_model(checkpoint_path, device):
    """Load checkpoint and build model."""
    print(f"\n{'='*80}")
    print(f"LOADING CHECKPOINT")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_args = checkpoint['args']
    
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"Checkpoint: {checkpoint_name}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best PCK: {checkpoint.get('best_pck', 'N/A')}")
    
    # Build dataset to get tokenizer
    print(f"\nBuilding dataset to get tokenizer...")
    import albumentations as A
    transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    train_dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_train.json',
        transforms=transforms,
        split='train',
        vocab_size=2000,
        seq_len=200
    )
    tokenizer = train_dataset.tokenizer
    print(f"  ✓ Tokenizer: vocab_size={tokenizer.vocab_size}, num_bins={tokenizer.num_bins}")
    
    # Build model
    print(f"\nBuilding model...")
    build_result = build_model(checkpoint_args, tokenizer=tokenizer, train=False)
    if isinstance(build_result, tuple):
        base_model, _ = build_result
    else:
        base_model = build_result
    model = build_cape_model(checkpoint_args, base_model)
    
    # Load weights
    print(f"Loading model weights...")
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"  ✓ Model loaded")
    except Exception as e:
        print(f"  ⚠️  Warning during load: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint_args, tokenizer


def analyze_sample(
    batch_idx,
    batch,
    model,
    tokenizer,
    device,
    dump_tokens=False,
    debug=False
):
    """Analyze a single batch to detect keypoint count mismatches."""
    
    # Move batch to device
    support_images = batch['support_images'].to(device)
    support_coords = batch['support_coords'].to(device)
    support_masks = batch['support_masks'].to(device)
    query_images = batch['query_images'].to(device)
    support_skeletons = batch.get('support_skeletons', None)
    query_targets = {k: v.to(device) for k, v in batch['query_targets'].items()}
    query_metadata = batch.get('query_metadata', [])
    
    # Run forward inference
    with torch.no_grad():
        if hasattr(model, 'module'):
            predictions = model.module.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=support_skeletons
            )
        else:
            predictions = model.forward_inference(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                skeleton_edges=support_skeletons
            )
    
    # Extract coordinates and logits
    pred_coords = predictions.get('coordinates', None)
    pred_logits = predictions.get('logits', None)
    gt_coords = query_targets['target_seq']
    token_labels = query_targets['token_labels']
    mask = query_targets['mask']
    
    # Extract keypoints
    if pred_logits is not None:
        pred_kpts = extract_keypoints_from_predictions(pred_coords, pred_logits)
    else:
        pred_kpts = extract_keypoints_from_sequence(pred_coords, token_labels, mask)
    
    gt_kpts = extract_keypoints_from_sequence(gt_coords, token_labels, mask)
    
    # Analyze each sample in batch
    results = []
    batch_size = pred_coords.shape[0]
    
    for idx in range(batch_size):
        meta = query_metadata[idx] if idx < len(query_metadata) else {}
        
        # Get GT information
        category_id = meta.get('category_id', 'N/A')
        num_keypoints = meta.get('num_keypoints', 0)
        visibility = meta.get('visibility', [])
        
        # Get pred information
        pred_kpts_count = pred_kpts[idx].shape[0] if len(pred_kpts[idx].shape) > 1 else 0
        gt_kpts_count = gt_kpts[idx].shape[0] if len(gt_kpts[idx].shape) > 1 else 0
        
        # Token analysis
        gt_token_types = token_labels[idx][mask[idx].bool()]
        gt_token_dist = Counter(gt_token_types.cpu().numpy())
        
        if pred_logits is not None:
            pred_token_types = pred_logits[idx].argmax(-1)
            # Only count tokens before first PAD or up to sequence length
            valid_pred_tokens = pred_token_types[:len(gt_token_types)]
            pred_token_dist = Counter(valid_pred_tokens.cpu().numpy())
        else:
            pred_token_dist = gt_token_dist  # Fallback
        
        # Compute token type names
        token_names = {
            TokenType.coord.value: 'COORD',
            TokenType.sep.value: 'SEP',
            TokenType.eos.value: 'EOS',
            TokenType.cls.value: 'CLS'
        }
        
        result = {
            'batch_idx': batch_idx,
            'sample_idx': idx,
            'category_id': category_id,
            'num_keypoints_expected': num_keypoints,
            'gt_kpts_count': gt_kpts_count,
            'pred_kpts_count': pred_kpts_count,
            'mismatch': pred_kpts_count != gt_kpts_count,
            'extra_predictions': max(0, pred_kpts_count - num_keypoints),
            'missing_predictions': max(0, num_keypoints - pred_kpts_count),
            'gt_token_dist': {token_names.get(k, f'UNK{k}'): v for k, v in gt_token_dist.items()},
            'pred_token_dist': {token_names.get(k, f'UNK{k}'): v for k, v in pred_token_dist.items()},
            'gt_seq_len': gt_coords.shape[1],
            'pred_seq_len': pred_coords.shape[1],
            'visibility_count': len(visibility),
            'pred_kpts': pred_kpts[idx].cpu().numpy() if pred_kpts_count > 0 else None,
            'gt_kpts': gt_kpts[idx].cpu().numpy() if gt_kpts_count > 0 else None,
        }
        
        if dump_tokens:
            result['gt_tokens'] = gt_token_types.cpu().numpy()
            if pred_logits is not None:
                result['pred_tokens'] = valid_pred_tokens.cpu().numpy()
        
        results.append(result)
    
    return results


def print_sample_report(result, dump_tokens=False):
    """Print detailed report for a single sample."""
    print(f"\n{'='*80}")
    print(f"Sample {result['batch_idx']}.{result['sample_idx']}")
    print(f"{'='*80}")
    
    print(f"\n📋 Basic Information:")
    print(f"  Category ID: {result['category_id']}")
    print(f"  Expected num_keypoints (from category): {result['num_keypoints_expected']}")
    
    print(f"\n📊 Ground Truth:")
    print(f"  GT keypoints loaded: {result['gt_kpts_count']}")
    print(f"  GT sequence length: {result['gt_seq_len']}")
    print(f"  GT token types distribution: {result['gt_token_dist']}")
    
    print(f"\n🔮 Predictions:")
    print(f"  Predicted sequence length: {result['pred_seq_len']}")
    print(f"  Predicted token types: {result['pred_token_dist']}")
    print(f"  Predicted keypoints extracted: {result['pred_kpts_count']}")
    
    print(f"\n⚖️  Comparison:")
    if result['mismatch']:
        print(f"  ❌ MISMATCH DETECTED!")
        if result['extra_predictions'] > 0:
            print(f"     {result['extra_predictions']} EXTRA predictions (pred > expected)")
        if result['missing_predictions'] > 0:
            print(f"     {result['missing_predictions']} MISSING predictions (pred < expected)")
    else:
        print(f"  ✅ Counts match (pred == gt == expected)")
    
    print(f"\n🔧 After Trimming:")
    print(f"  After trimming to num_kpts_for_category: {result['num_keypoints_expected']}")
    print(f"  Visibility mask length: {result['visibility_count']}")
    print(f"  PCK will compare: {min(result['pred_kpts_count'], result['num_keypoints_expected'])} vs {min(result['gt_kpts_count'], result['num_keypoints_expected'])}")
    
    if dump_tokens and 'gt_tokens' in result:
        print(f"\n🔍 Token Sequences (first 50 tokens):")
        print(f"  GT tokens:   {result['gt_tokens'][:50]}")
        if 'pred_tokens' in result:
            print(f"  Pred tokens: {result['pred_tokens'][:50]}")
    
    # Show coordinate samples
    if result['pred_kpts'] is not None and result['gt_kpts'] is not None:
        print(f"\n📍 Keypoint Coordinates (first 3):")
        pred_sample = result['pred_kpts'][:3]
        gt_sample = result['gt_kpts'][:3]
        for i, (pred, gt) in enumerate(zip(pred_sample, gt_sample)):
            print(f"  Keypoint {i}: pred={pred}, gt={gt}")


def analyze_per_category(results):
    """Aggregate results per category."""
    print(f"\n{'='*80}")
    print(f"PER-CATEGORY ANALYSIS")
    print(f"{'='*80}\n")
    
    category_stats = defaultdict(lambda: {
        'samples': 0,
        'mismatches': 0,
        'total_extra': 0,
        'total_missing': 0,
        'pred_counts': [],
        'gt_counts': [],
        'expected_counts': []
    })
    
    for r in results:
        cat_id = r['category_id']
        stats = category_stats[cat_id]
        stats['samples'] += 1
        if r['mismatch']:
            stats['mismatches'] += 1
        stats['total_extra'] += r['extra_predictions']
        stats['total_missing'] += r['missing_predictions']
        stats['pred_counts'].append(r['pred_kpts_count'])
        stats['gt_counts'].append(r['gt_kpts_count'])
        stats['expected_counts'].append(r['num_keypoints_expected'])
    
    for cat_id, stats in sorted(category_stats.items()):
        print(f"\nCategory {cat_id}:")
        print(f"  Samples: {stats['samples']}")
        print(f"  Mismatches: {stats['mismatches']} ({100*stats['mismatches']/stats['samples']:.1f}%)")
        print(f"  Expected keypoints: {stats['expected_counts'][0]} (constant={len(set(stats['expected_counts'])) == 1})")
        print(f"  GT keypoints: min={min(stats['gt_counts'])}, max={max(stats['gt_counts'])}, avg={np.mean(stats['gt_counts']):.1f}")
        print(f"  Pred keypoints: min={min(stats['pred_counts'])}, max={max(stats['pred_counts'])}, avg={np.mean(stats['pred_counts']):.1f}")
        if stats['total_extra'] > 0:
            print(f"  ⚠️  Total extra predictions: {stats['total_extra']}")
        if stats['total_missing'] > 0:
            print(f"  ⚠️  Total missing predictions: {stats['total_missing']}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose keypoint count mismatches')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to analyze')
    parser.add_argument('--per-category', action='store_true',
                        help='Show per-category statistics')
    parser.add_argument('--dump-tokens', action='store_true',
                        help='Dump token sequences for inspection')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda/mps), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"KEYPOINT COUNT DIAGNOSTIC")
    print(f"{'='*80}\n")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Per-category: {args.per_category}")
    print(f"Dump tokens: {args.dump_tokens}")
    
    # Load model
    model, checkpoint_args, tokenizer = load_checkpoint_and_model(args.checkpoint, device)
    
    # Build validation dataloader
    print(f"\n{'='*80}")
    print(f"BUILDING VALIDATION DATALOADER")
    print(f"{'='*80}\n")
    
    # Same transforms for validation (A is already imported above)
    import albumentations as A
    val_transforms = A.Compose([
        A.Resize(height=512, width=512)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    val_dataset = MP100CAPE(
        img_folder='data',
        ann_file='annotations/mp100_split1_val.json',
        transforms=val_transforms,
        split='val',
        vocab_size=2000,
        seq_len=200
    )
    
    val_dataloader = build_episodic_dataloader(
        val_dataset,
        category_split_file='category_splits.json',
        split='val',
        batch_size=2,
        num_queries_per_episode=2,
        episodes_per_epoch=min(args.num_samples, 100),
        num_workers=0
    )
    
    print(f"✓ Validation dataloader built")
    
    # Run diagnostics
    print(f"\n{'='*80}")
    print(f"RUNNING DIAGNOSTICS")
    print(f"{'='*80}")
    
    all_results = []
    debug_mode = os.environ.get('DEBUG_KEYPOINT_COUNT', '0') == '1'
    
    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= args.num_samples:
            break
        
        results = analyze_sample(
            batch_idx,
            batch,
            model,
            tokenizer,
            device,
            dump_tokens=args.dump_tokens,
            debug=debug_mode
        )
        
        all_results.extend(results)
        
        # Print first few samples in detail
        if batch_idx < 3:
            for result in results:
                print_sample_report(result, dump_tokens=args.dump_tokens)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    total_samples = len(all_results)
    total_mismatches = sum(1 for r in all_results if r['mismatch'])
    total_extra = sum(r['extra_predictions'] for r in all_results)
    total_missing = sum(r['missing_predictions'] for r in all_results)
    
    print(f"Total samples analyzed: {total_samples}")
    print(f"Samples with mismatches: {total_mismatches} ({100*total_mismatches/total_samples:.1f}%)")
    print(f"Total extra predictions: {total_extra}")
    print(f"Total missing predictions: {total_missing}")
    
    # Hypothesis test
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS TEST")
    print(f"{'='*80}\n")
    
    if total_mismatches > total_samples * 0.5:
        print(f"✅ HYPOTHESIS CONFIRMED")
        print(f"   The model generates a DIFFERENT number of keypoints than expected")
        print(f"   in {total_mismatches}/{total_samples} samples.")
        if total_extra > total_missing:
            print(f"\n   ⚠️  PRIMARY ISSUE: Model generates TOO MANY keypoints")
            print(f"      - {total_extra} extra predictions across all samples")
            print(f"      - Trimming operation is discarding valid predictions!")
        elif total_missing > total_extra:
            print(f"\n   ⚠️  PRIMARY ISSUE: Model generates TOO FEW keypoints")
            print(f"      - {total_missing} missing predictions across all samples")
            print(f"      - Model stops generation too early!")
        else:
            print(f"\n   ⚠️  MIXED ISSUE: Both over and under-generation")
    elif total_mismatches == 0:
        print(f"❌ HYPOTHESIS REJECTED")
        print(f"   All samples have matching keypoint counts.")
        print(f"   The issue likely lies elsewhere (coordinate accuracy, visibility, etc.)")
    else:
        print(f"⚠️  PARTIAL CONFIRMATION")
        print(f"   Some samples ({total_mismatches}/{total_samples}) have mismatches.")
        print(f"   This may be category-specific behavior.")
    
    # Per-category analysis
    if args.per_category:
        analyze_per_category(all_results)
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
