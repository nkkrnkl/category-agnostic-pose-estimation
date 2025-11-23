"""Test MPS training with small batch."""
import torch
from dataset import MP100Dataset
from episodic_sampler import create_episodic_dataloader
from model.model import Raster2SeqCAPE
from utils.masking import create_keypoint_mask

# Get device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device

device = get_device()

print("Loading dataset...")
train_dataset = MP100Dataset(
    annotation_file='annotations/mp100_split1_train.json',
    image_root='data',
    image_size=512,
    augment=True
)

print("Creating dataloader...")
train_loader = create_episodic_dataloader(
    dataset=train_dataset,
    batch_size=2,
    num_episodes=10,
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
).to(device)

print("\nTesting one batch on MPS...")
for batch_idx, batch in enumerate(train_loader):
    # Move to device
    support_coords = batch['support_coords'].to(device)
    query_images = batch['query_images'].to(device)
    query_coords = batch['query_coords'].to(device)
    query_visibility = batch['query_visibility'].to(device)
    num_keypoints = batch['num_keypoints'].to(device)

    B, T_max = support_coords.shape[:2]
    keypoint_mask = create_keypoint_mask(num_keypoints, T_max).to(device)

    # Forward pass
    print("Running forward pass on MPS...")
    outputs = model(
        query_images=query_images,
        support_coords=support_coords,
        target_coords=query_coords,
        num_keypoints=num_keypoints,
        keypoint_mask=keypoint_mask,
        teacher_forcing=True
    )

    # Compute loss
    print("Computing loss...")
    loss_dict = model.compute_loss(
        outputs=outputs,
        target_coords=query_coords,
        visibility=query_visibility,
        num_keypoints=num_keypoints,
        coord_loss_weight=5.0
    )

    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")

    # Backward pass
    print("Running backward pass...")
    loss_dict['total_loss'].backward()

    print("✓ MPS training test passed!")
    break
