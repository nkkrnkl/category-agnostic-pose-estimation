"""Quick test of training loop."""
import torch
from dataset import MP100Dataset
from episodic_sampler import create_episodic_dataloader
from model.model import Raster2SeqCAPE
from utils.masking import create_keypoint_mask

print("Loading dataset...")
train_dataset = MP100Dataset(
    annotation_file='data/cleaned_annotations/mp100_split1_train.json',
    image_root='data',
    image_size=512,
    augment=True
)

print("Creating dataloader...")
train_loader = create_episodic_dataloader(
    dataset=train_dataset,
    batch_size=2,  # Small batch
    num_episodes=10,  # Just 10 episodes for test
    shuffle=True,
    num_workers=0
)

print("Creating model...")
model = Raster2SeqCAPE(
    hidden_dim=256,
    num_encoder_layers=3,
    num_decoder_layers=6,
    num_heads=8,
    dropout=0.1,
    max_keypoints=100,
    pretrained_resnet=True
)

print("Testing one batch...")
for batch_idx, batch in enumerate(train_loader):
    print(f"\nBatch {batch_idx + 1}")
    print(f"  Support images: {batch['support_images'].shape}")
    print(f"  Support coords: {batch['support_coords'].shape}")
    print(f"  Query images: {batch['query_images'].shape}")
    print(f"  Query coords: {batch['query_coords'].shape}")
    print(f"  Num keypoints: {batch['num_keypoints']}")

    # Create keypoint mask
    B, T_max = batch['support_coords'].shape[:2]
    keypoint_mask = create_keypoint_mask(batch['num_keypoints'], T_max)

    # Forward pass
    print("  Running forward pass...")
    outputs = model(
        query_images=batch['query_images'],
        support_coords=batch['support_coords'],
        target_coords=batch['query_coords'],
        num_keypoints=batch['num_keypoints'],
        keypoint_mask=keypoint_mask,
        teacher_forcing=True
    )

    print(f"  Output coords shape: {outputs['coords'].shape}")
    print(f"  Output token_logits shape: {outputs['token_logits'].shape}")

    # Compute loss
    print("  Computing loss...")
    loss_dict = model.compute_loss(
        outputs=outputs,
        target_coords=batch['query_coords'],
        visibility=batch['query_visibility'],
        num_keypoints=batch['num_keypoints'],
        coord_loss_weight=5.0
    )

    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Coord loss: {loss_dict['coord_loss'].item():.4f}")

    # Test backward pass
    print("  Running backward pass...")
    loss_dict['total_loss'].backward()
    print("  ✓ Backward pass successful!")

    break  # Only test one batch

print("\n✓ Training loop test passed!")
