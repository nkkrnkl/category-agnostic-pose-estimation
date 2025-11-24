"""
Training script for Raster2Seq CAPE model.
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml

from dataset import MP100Dataset
from episodic_sampler import create_episodic_dataloader
from model.model import Raster2SeqCAPE
from utils.logging import MetricLogger, print_model_summary
from utils.masking import create_keypoint_mask, create_visibility_mask


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """Create Raster2Seq model."""
    model_config = config['model']

    model = Raster2SeqCAPE(
        hidden_dim=model_config['hidden_dim'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        max_keypoints=model_config['max_keypoints'],
        pretrained_resnet=model_config['pretrained_resnet']
    )

    model = model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile for faster training...")
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled successfully")

    return model


def create_optimizer(model, config):
    """Create optimizer and learning rate scheduler."""
    training_config = config['training']

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Cosine learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config['num_epochs']
    )

    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, device, config, logger, epoch, scaler=None):
    """Train for one epoch."""
    model.train()

    training_config = config['training']
    coord_loss_weight = training_config['coord_loss_weight']
    
    # Use mixed precision if CUDA is available and scaler is provided
    use_amp = (device.type == 'cuda' and scaler is not None)

    total_loss = 0
    total_coord_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device (use non_blocking for faster transfer when using CUDA)
        non_blocking = (device.type == 'cuda')
        support_images = batch['support_images'].to(device, non_blocking=non_blocking)
        support_coords = batch['support_coords'].to(device, non_blocking=non_blocking)
        support_visibility = batch['support_visibility'].to(device, non_blocking=non_blocking)
        query_images = batch['query_images'].to(device, non_blocking=non_blocking)
        query_coords = batch['query_coords'].to(device, non_blocking=non_blocking)
        query_visibility = batch['query_visibility'].to(device, non_blocking=non_blocking)
        num_keypoints = batch['num_keypoints'].to(device, non_blocking=non_blocking)

        B, T_max = support_coords.shape[:2]

        # Create keypoint mask
        keypoint_mask = create_keypoint_mask(num_keypoints, T_max).to(device, non_blocking=non_blocking)

        # Forward pass with teacher forcing (using mixed precision if enabled)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(
                    query_images=query_images,
                    support_coords=support_coords,
                    target_coords=query_coords,
                    num_keypoints=num_keypoints,
                    keypoint_mask=keypoint_mask,
                    teacher_forcing=True
                )
                
                # Compute loss
                loss_dict = model.compute_loss(
                    outputs=outputs,
                    target_coords=query_coords,
                    visibility=query_visibility,
                    num_keypoints=num_keypoints,
                    coord_loss_weight=coord_loss_weight
                )
                
                loss = loss_dict['total_loss']
                coord_loss = loss_dict['coord_loss']
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(
                query_images=query_images,
                support_coords=support_coords,
                target_coords=query_coords,
                num_keypoints=num_keypoints,
                keypoint_mask=keypoint_mask,
                teacher_forcing=True
            )
            
            # Compute loss
            loss_dict = model.compute_loss(
                outputs=outputs,
                target_coords=query_coords,
                visibility=query_visibility,
                num_keypoints=num_keypoints,
                coord_loss_weight=coord_loss_weight
            )
            
            loss = loss_dict['total_loss']
            coord_loss = loss_dict['coord_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Logging
        total_loss += loss.item()
        total_coord_loss += coord_loss.item()
        num_batches += 1

        if (batch_idx + 1) % training_config['log_interval'] == 0:
            avg_loss = total_loss / num_batches
            avg_coord_loss = total_coord_loss / num_batches

            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Coord Loss: {avg_coord_loss:.4f}")

            logger.update_epoch({
                'loss': loss.item(),
                'coord_loss': coord_loss.item()
            }, prefix='train/')

    # Return average metrics
    return {
        'loss': total_loss / num_batches,
        'coord_loss': total_coord_loss / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Raster2Seq CAPE model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--preliminary', action='store_true',
                        help='Run preliminary training (5 epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config['experiment']['seed'])

    # Device
    device = get_device()

    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)

    # Create dataset
    print("\nLoading training dataset...")
    train_dataset = MP100Dataset(
        annotation_file=config['data']['annotation_train'],
        image_root=config['data']['image_root'],
        image_size=config['data']['image_size'],
        augment=config['training']['augment']
    )

    # Create episodic dataloader
    print("\nCreating episodic dataloader...")
    # Use num_workers from config if available, otherwise use 0 for CPU or 4+ for CUDA
    num_workers = config.get('num_workers', 0)
    if device.type == 'cuda' and num_workers == 0:
        # Default to 4 workers for CUDA to enable parallel data loading
        num_workers = 4
        print(f"Using {num_workers} workers for data loading (CUDA detected)")
    
    train_loader = create_episodic_dataloader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        num_episodes=config['training']['num_episodes_per_epoch'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')  # Pin memory for faster GPU transfer
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    print_model_summary(model)

    # Create optimizer
    optimizer, scheduler = create_optimizer(model, config)
    
    # Create GradScaler for mixed precision training (CUDA only)
    scaler = None
    if device.type == 'cuda':
        scaler = GradScaler()
        print("✓ Mixed precision training enabled (FP16)")

    # Logger
    logger = MetricLogger(
        log_dir=config['paths']['log_dir'],
        experiment_name=config['experiment']['name']
    )

    # Training loop
    num_epochs = (config['training']['num_epochs_preliminary']
                  if args.preliminary
                  else config['training']['num_epochs'])

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print('=' * 80)

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            logger=logger,
            epoch=epoch,
            scaler=scaler
        )

        # Log epoch metrics
        logger.log(train_metrics, step=epoch, prefix='train/')
        logger.print_epoch(epoch, train_metrics)

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                f"{config['experiment']['name']}_epoch{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'train_metrics': train_metrics
            }, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")

        # Save logs
        logger.save()

    print("\nTraining complete!")

    # Save final model
    final_checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        f"{config['experiment']['name']}_final.pth"
    )
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_checkpoint_path)
    print(f"Saved final model: {final_checkpoint_path}")


if __name__ == '__main__':
    main()
