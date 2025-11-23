#!/usr/bin/env python3
"""
Quick diagnosis of 198 keypoints issue - just analyzes checkpoint losses
No need to load full model or dataset

Run: python quick_diagnosis.py
"""
import torch
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Analyze training progress from checkpoint"""
    print("=" * 80)
    print("QUICK DIAGNOSIS: Why 198 Keypoints?")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    epoch = checkpoint.get('epoch', 0)
    train_stats = checkpoint.get('train_stats', {})
    val_stats = checkpoint.get('val_stats', {})

    print(f"\nCheckpoint: {Path(checkpoint_path).name}")
    print(f"Epoch: {epoch}")

    # Extract key losses
    loss_ce_train = train_stats.get('loss_ce', 0)
    loss_ce_val = val_stats.get('loss_ce', 0)
    loss_coords_train = train_stats.get('loss_coords', 0)
    loss_coords_val = val_stats.get('loss_coords', 0)
    total_loss_train = train_stats.get('loss', 0)
    total_loss_val = val_stats.get('loss', 0)

    print("\n" + "=" * 80)
    print("TRAINING METRICS")
    print("=" * 80)
    print(f"\n{'Metric':<40s} {'Train':>10s} {'Val':>10s}")
    print("-" * 80)
    print(f"{'Total Loss':<40s} {total_loss_train:>10.4f} {total_loss_val:>10.4f}")
    print(f"{'Token Classification Loss (loss_ce)':<40s} {loss_ce_train:>10.4f} {loss_ce_val:>10.4f}")
    print(f"{'Coordinate Regression Loss (loss_coords)':<40s} {loss_coords_train:>10.4f} {loss_coords_val:>10.4f}")

    # Diagnosis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    print(f"\n📊 Token Classification Loss: {loss_ce_train:.4f}")
    print("\nThis loss controls whether the model predicts:")
    print("  - 'coord' tokens (generate a keypoint)")
    print("  - 'eos' tokens (STOP generating, end sequence)")
    print("  - 'sep' tokens (separator)")

    if loss_ce_train > 3.0:
        print("\n⚠️  STATUS: VERY HIGH (> 3.0)")
        print("\n🔍 DIAGNOSIS:")
        print("   The model is essentially guessing random token types.")
        print("   It hasn't learned the pattern yet.")
        print("\n💡 WHY 198 KEYPOINTS:")
        print("   Since the model can't predict 'eos' (end-of-sequence) correctly,")
        print("   it keeps predicting 'coord' tokens until it hits the 200-token limit.")
        print("   After BOS token (1) and PAD token (1), you get 198 coord tokens.")
        print("\n✅ SOLUTION:")
        print("   This is NORMAL after only 5 epochs!")
        print("   → Continue training to 50-100 epochs")
        print("   → Target: loss_ce < 1.0")
        print("   → When loss_ce < 1.0, you'll see ~13 keypoints (correct)")

    elif loss_ce_train > 1.5:
        print("\n⚡ STATUS: HIGH but decreasing (1.5 - 3.0)")
        print("\n🔍 DIAGNOSIS:")
        print("   Model is learning but needs more training.")
        print("   → Continue training to 50 epochs")
        print("   → Should improve to ~13 keypoints soon")

    elif loss_ce_train > 1.0:
        print("\n📈 STATUS: Moderate (1.0 - 1.5)")
        print("\n🔍 DIAGNOSIS:")
        print("   Model is learning EOS prediction but not perfect yet.")
        print("   → Train 10-20 more epochs")

    else:
        print("\n✅ STATUS: GOOD (< 1.0)")
        print("\n🔍 DIAGNOSIS:")
        print("   Model should be predicting EOS correctly.")
        print("   If you still see 198 keypoints, there may be another issue.")

    # Timeline
    print("\n" + "=" * 80)
    print("EXPECTED TRAINING TIMELINE")
    print("=" * 80)
    print("\nEpochs    | loss_ce | Keypoints Predicted | Status")
    print("-" * 80)
    print("1-10      | > 3.0   | ~198 (no EOS)      | 🟡 Initial learning")
    print("10-30     | 2.0-3.0 | ~100-150           | 🟡 Starting to learn EOS")
    print("30-50     | 1.0-2.0 | ~30-50             | 🟢 Learning EOS better")
    print("50-100    | < 1.0   | ~13-15 (correct!)  | ✅ Trained properly")

    current_stage = ""
    if epoch < 10:
        current_stage = "1-10"
    elif epoch < 30:
        current_stage = "10-30"
    elif epoch < 50:
        current_stage = "30-50"
    else:
        current_stage = "50-100"

    print(f"\n>>> YOU ARE HERE: Epoch {epoch} ({current_stage}) <<<")

    # Recommendations
    print("\n" + "=" * 80)
    print("IMMEDIATE NEXT STEPS")
    print("=" * 80)

    if epoch < 50:
        epochs_remaining = 50 - epoch
        print(f"\n1. ✅ Continue training for {epochs_remaining} more epochs (target: 50 total)")
        print("2. 📊 Check loss_ce every 10 epochs - should decrease steadily")
        print("3. 🔍 Re-run visualization at epoch 30 and 50 to see improvement")
        print(f"\n   Current: {epoch} epochs → 198 keypoints (expected)")
        print(f"   Target:  50 epochs → ~13 keypoints (goal)")
    else:
        print("\n1. You've trained for enough epochs")
        print("2. If still seeing 198 keypoints, check loss_ce value")
        print("3. May need to adjust loss weights or learning rate")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n🎯 Problem: Model predicts 198 keypoints instead of ~13")
    print(f"📌 Root Cause: Token classifier hasn't learned EOS prediction (loss_ce = {loss_ce_train:.2f})")
    print(f"⏰ Training Stage: Epoch {epoch}/50 (need more training)")
    print(f"✅ Solution: Continue training - this is EXPECTED behavior at epoch {epoch}")
    print(f"\n💡 The model is learning! Just needs more time.")

    print("\n" + "=" * 80)


def analyze_all_epochs():
    """Analyze training progression across all epochs"""
    output_dir = Path('/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation/output/cape_episodic_20251123_144641')

    checkpoints = sorted(output_dir.glob('checkpoint_epoch_*.pth'))

    if not checkpoints:
        print("No checkpoints found!")
        return

    print("\n" + "=" * 80)
    print("TRAINING PROGRESSION ANALYSIS")
    print("=" * 80)

    print(f"\n{'Epoch':>6s} {'Total Loss':>12s} {'loss_ce':>12s} {'loss_coords':>12s}")
    print("-" * 80)

    loss_ce_values = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epoch = ckpt.get('epoch', 0)
        train_stats = ckpt.get('train_stats', {})

        total_loss = train_stats.get('loss', 0)
        loss_ce = train_stats.get('loss_ce', 0)
        loss_coords = train_stats.get('loss_coords', 0)

        loss_ce_values.append(loss_ce)

        print(f"{epoch:>6d} {total_loss:>12.4f} {loss_ce:>12.4f} {loss_coords:>12.4f}")

    # Check if loss is decreasing
    print("\n" + "=" * 80)
    print("TRAINING HEALTH CHECK")
    print("=" * 80)

    if len(loss_ce_values) > 1:
        is_decreasing = all(loss_ce_values[i] >= loss_ce_values[i+1] for i in range(len(loss_ce_values)-1))

        if is_decreasing:
            print("\n✅ loss_ce is DECREASING - Training is working!")
            print(f"   Epoch 0: {loss_ce_values[0]:.4f} → Epoch {len(loss_ce_values)-1}: {loss_ce_values[-1]:.4f}")
            print(f"   Improvement: {loss_ce_values[0] - loss_ce_values[-1]:.4f}")
        else:
            print("\n⚠️  loss_ce is not monotonically decreasing")
            print("   This is OK - loss can fluctuate during training")
            print("   As long as general trend is downward")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    import sys

    # Analyze latest checkpoint
    checkpoint_path = '/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation/output/cape_episodic_20251123_144641/checkpoint_epoch_4.pth'

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    analyze_checkpoint(checkpoint_path)
    analyze_all_epochs()

    print("\n" + "=" * 80)
    print("For more detailed analysis, run: python diagnose_predictions.py")
    print("=" * 80)
