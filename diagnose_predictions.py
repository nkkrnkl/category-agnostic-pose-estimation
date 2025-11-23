#!/usr/bin/env python3
"""
Diagnose why model predicts 198 keypoints instead of ~13

Run this with: python diagnose_predictions.py

This will show you:
1. Token type predictions (coord vs eos vs sep)
2. Where the model should have stopped
3. Training loss breakdown
"""
import torch
import numpy as np
from pathlib import Path
import argparse

# Import your model and dataset
from models.cape_model import build_cape_model
from models import build_model
from datasets.mp100_cape import build_mp100_cape
from datasets.episodic_sampler import build_episodic_dataloader


def analyze_token_predictions(model, data_loader, device, num_samples=5):
    """
    Analyze what token types the model is predicting
    """
    model.eval()

    print("=" * 80)
    print("TOKEN PREDICTION ANALYSIS")
    print("=" * 80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_samples:
                break

            # Move to device
            support_coords = batch['support_coords'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_images = batch['query_images'].to(device)
            support_skeletons = batch.get('support_skeletons', None)

            query_targets = {}
            for key, value in batch['query_targets'].items():
                query_targets[key] = value.to(device)

            # Forward pass
            outputs = model(
                samples=query_images,
                support_coords=support_coords,
                support_mask=support_masks,
                targets=query_targets,
                skeleton_edges=support_skeletons
            )

            # Get token predictions
            # outputs['pred_logits'] shape: (batch, seq_len, num_token_types)
            pred_logits = outputs['pred_logits'][0]  # First sample
            pred_token_types = pred_logits.argmax(dim=-1)  # (seq_len,)

            # Get ground truth token types
            gt_token_labels = query_targets['token_labels'][0]  # (seq_len,)

            # Token type mapping (from token_types.py)
            # 0: coord, 1: sep, 2: eos, 3: cls
            token_names = ['coord', 'sep', 'eos', 'cls', 'pad']

            print(f"\n{'=' * 80}")
            print(f"Sample {batch_idx + 1}:")
            print(f"{'=' * 80}")

            # Count predicted token types
            pred_coords = (pred_token_types == 0).sum().item()
            pred_sep = (pred_token_types == 1).sum().item()
            pred_eos = (pred_token_types == 2).sum().item()
            pred_cls = (pred_token_types == 3).sum().item()

            # Count ground truth token types
            gt_coords = (gt_token_labels == 0).sum().item()
            gt_sep = (gt_token_labels == 1).sum().item()
            gt_eos = (gt_token_labels == 2).sum().item()

            print("\nPredicted Token Counts:")
            print(f"  coord: {pred_coords:3d}  <-- THIS IS YOUR 198 KEYPOINTS PROBLEM!")
            print(f"  sep:   {pred_sep:3d}")
            print(f"  eos:   {pred_eos:3d}  <-- Should be > 0 to stop sequence")
            print(f"  cls:   {pred_cls:3d}")

            print("\nGround Truth Token Counts:")
            print(f"  coord: {gt_coords:3d}  <-- Should match actual # of keypoints (~13 for bird)")
            print(f"  sep:   {gt_sep:3d}")
            print(f"  eos:   {gt_eos:3d}")

            # Find where sequence should end
            gt_mask = query_targets['mask'][0]
            valid_length = gt_mask.sum().item()

            print(f"\nSequence Analysis:")
            print(f"  Ground truth sequence length: {valid_length} tokens")
            print(f"  Predicted keypoints: {pred_coords}")
            print(f"  Expected keypoints: {gt_coords}")

            # Show first 50 predictions vs ground truth
            print(f"\nFirst 50 Token Predictions vs Ground Truth:")
            print("  Pos | Predicted  | Ground Truth")
            print("  " + "-" * 35)
            for i in range(min(50, len(pred_token_types))):
                pred_name = token_names[pred_token_types[i]] if pred_token_types[i] < 4 else 'unk'
                gt_name = token_names[gt_token_labels[i]] if gt_token_labels[i] >= 0 and gt_token_labels[i] < 4 else 'pad'
                marker = " ✓" if pred_token_types[i] == gt_token_labels[i] else " ✗"
                print(f"  {i:3d} | {pred_name:10s} | {gt_name:10s} {marker}")

            print("\n" + "=" * 80)

            # Check if model ever predicts EOS
            if pred_eos == 0:
                print("⚠️  WARNING: Model NEVER predicts EOS token!")
                print("   This means it will always generate the full sequence (200 tokens)")
                print("   → Token classifier needs more training to learn EOS prediction")

            # Check coordinate prediction accuracy
            coord_mask = gt_token_labels == 0
            if coord_mask.sum() > 0:
                coord_accuracy = ((pred_token_types[coord_mask] == 0).float().mean() * 100).item()
                print(f"\n📊 Coordinate token classification accuracy: {coord_accuracy:.1f}%")

            eos_mask = gt_token_labels == 2
            if eos_mask.sum() > 0:
                eos_accuracy = ((pred_token_types[eos_mask] == 2).float().mean() * 100).item()
                print(f"📊 EOS token classification accuracy: {eos_accuracy:.1f}%")
                if eos_accuracy < 50:
                    print("   ⚠️  EOS accuracy is LOW - this is why you get 198 keypoints!")


def analyze_training_losses(checkpoint_path):
    """
    Load checkpoint and analyze loss trends
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\n" + "=" * 80)
    print("TRAINING LOSS ANALYSIS")
    print("=" * 80)

    train_stats = checkpoint.get('train_stats', {})
    val_stats = checkpoint.get('val_stats', {})
    epoch = checkpoint.get('epoch', 0)

    print(f"\nEpoch {epoch} Statistics:")
    print(f"\n{'Metric':<35s} {'Train':>10s} {'Val':>10s}")
    print("-" * 80)

    for key in sorted(set(train_stats.keys()) | set(val_stats.keys())):
        train_val = train_stats.get(key, 0)
        val_val = val_stats.get(key, 0)
        print(f"{key:<35s} {train_val:10.4f} {val_val:10.4f}")

    # Analyze specific losses
    print("\n" + "=" * 80)
    print("KEY LOSS COMPONENTS:")
    print("=" * 80)

    loss_ce = train_stats.get('loss_ce', 0)
    loss_coords = train_stats.get('loss_coords', 0)

    print(f"\n1. Token Classification Loss (loss_ce): {loss_ce:.4f}")
    if loss_ce > 3.0:
        print("   ⚠️  HIGH - Model hasn't learned token types yet")
        print("   → This is WHY you get 198 keypoints (can't predict EOS)")
        print("   → Train for more epochs (target: < 1.0)")
    elif loss_ce > 1.0:
        print("   ⚡ DECREASING - Model is learning, but needs more epochs")
    else:
        print("   ✓ GOOD - Model should be predicting EOS correctly")

    print(f"\n2. Coordinate Regression Loss (loss_coords): {loss_coords:.4f}")
    if loss_coords > 0.5:
        print("   ⚠️  HIGH - Coordinate predictions are inaccurate")
    else:
        print("   ✓ GOOD - Coordinates are being learned")

    print(f"\n3. Total Loss: {train_stats.get('loss', 0):.4f}")

    return epoch, loss_ce


def main():
    parser = argparse.ArgumentParser('Diagnose CAPE Predictions')
    parser.add_argument('--checkpoint', default='output/cape_episodic_20251123_144641/checkpoint_best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--num_samples', default=5, type=int,
                       help='Number of samples to analyze')
    parser.add_argument('--dataset_root',
                       default='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation',
                       type=str)
    args = parser.parse_args()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}\n")

    # First analyze training losses from checkpoint
    epoch, loss_ce = analyze_training_losses(args.checkpoint)

    print("\n" + "=" * 80)
    print("RECOMMENDATION BASED ON TRAINING LOSSES:")
    print("=" * 80)

    if epoch < 10:
        print(f"\n⚠️  You've only trained for {epoch + 1} epochs")
        print("   → CAPE needs 50-100 epochs to learn properly")
        print("   → Continue training to see improvement in EOS prediction")

    if loss_ce > 2.0:
        print(f"\n⚠️  Token classification loss is still {loss_ce:.4f} (high)")
        print("   → Model hasn't learned when to stop generating keypoints")
        print("   → This is the ROOT CAUSE of your 198 keypoints issue")
        print("   → Keep training until loss_ce < 1.0")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Continue training for more epochs (target: 50-100 epochs)")
    print("2. Monitor loss_ce (token classification loss) - should decrease below 1.0")
    print("3. Check predictions every 10 epochs to see improvement")
    print("4. Once loss_ce < 1.0, you should see ~13 keypoints instead of 198")

    print("\n" + "=" * 80)
    print("\nRun with token analysis: python diagnose_predictions.py --analyze-tokens")
    print("(Note: This requires loading the full model, which takes time)")


if __name__ == '__main__':
    main()
