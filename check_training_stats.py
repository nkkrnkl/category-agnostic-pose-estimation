#!/usr/bin/env python3
"""
Check training statistics from checkpoint to diagnose 198 keypoints issue
"""
import torch

# Load checkpoint
checkpoint = torch.load('output/cape_episodic_20251123_144641/checkpoint_epoch_4.pth',
                       map_location='cpu')

print('=' * 80)
print('TRAINING DIAGNOSTICS - Why 198 Keypoints?')
print('=' * 80)

print('\n=== Epoch 4 Training Stats ===')
train_stats = checkpoint.get('train_stats', {})
for key, value in sorted(train_stats.items()):
    print(f'{key:35s}: {value:.6f}')

print('\n=== Epoch 4 Validation Stats ===')
val_stats = checkpoint.get('val_stats', {})
for key, value in sorted(val_stats.items()):
    print(f'{key:35s}: {value:.6f}')

print('\n=== Key Metrics to Watch ===')
print(f"Total Loss:              {train_stats.get('loss', 0):.4f}")
print(f"Token Classification Loss: {train_stats.get('loss_ce', 0):.4f}  <- Should decrease for EOS learning")
print(f"Coordinate Loss:         {train_stats.get('loss_coords', 0):.4f}")

if train_stats.get('loss_ce', 0) > 3.0:
    print("\n⚠️  WARNING: Token classification loss is still high (> 3.0)")
    print("   This means the model hasn't learned to predict EOS tokens yet.")
    print("   Expected behavior after only 5 epochs.")
    print("   → Train for more epochs (50-100) to see improvement.")

print('\n=== All Checkpoint Keys ===')
for key in checkpoint.keys():
    if key not in ['model', 'optimizer', 'lr_scheduler']:
        print(f"  - {key}: {checkpoint[key]}")
