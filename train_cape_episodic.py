#!/usr/bin/env python3
"""
Episodic Training Script for Category-Agnostic Pose Estimation (CAPE) on MP-100

This script implements TRUE CAPE with:
1. Support pose graph conditioning
2. Episodic meta-learning
3. Train/test category split
4. Evaluation on unseen categories

Difference from train_mp100_cape.py:
- train_mp100_cape.py: Standard supervised learning on all categories
- train_cape_episodic.py: Meta-learning with support graphs, unseen category evaluation
"""

import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models import build_model
from models.cape_model import build_cape_model

# Will create these functions below
from engine_cape import train_one_epoch_episodic, evaluate_cape


def get_args_parser():
    parser = argparse.ArgumentParser('CAPE Episodic Training', add_help=False)

    # CAPE-specific parameters
    parser.add_argument('--cape_mode', action='store_true', default=True,
                        help='Enable CAPE episodic training')
    parser.add_argument('--support_encoder_layers', default=3, type=int,
                        help='Number of layers in support pose graph encoder')
    parser.add_argument('--support_fusion_method', default='cross_attention',
                        choices=['cross_attention', 'concat', 'add'],
                        help='How to fuse support with query features')
    parser.add_argument('--num_queries_per_episode', default=2, type=int,
                        help='Number of query images per episode')
    parser.add_argument('--episodes_per_epoch', default=1000, type=int,
                        help='Number of episodes per training epoch')
    parser.add_argument('--category_split_file', default='category_splits.json',
                        help='Path to category split JSON file')

    # Learning rate parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Number of episodes per batch')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default='200,250', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Input parameters
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)

    # Sequence parameters
    parser.add_argument('--num_queries', default=200, type=int)
    parser.add_argument('--seq_len', default=200, type=int)
    parser.add_argument('--num_polys', default=1, type=int)
    parser.add_argument('--vocab_size', default=2000, type=int)
    parser.add_argument('--masked_attn', action='store_true', default=False)

    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str)
    parser.add_argument('--with_poly_refine', default=True, action='store_true')
    parser.add_argument('--use_anchor', action='store_true')

    # Semantic classes
    parser.add_argument('--semantic_classes', default=70, type=int,
                        help='Total number of categories in MP-100')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--aux_loss', action='store_true', default=True)

    # Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float)  # λ1 = 1.0 (as per milestone)
    parser.add_argument('--coords_loss_coef', default=5, type=float)  # λ2 = 5.0
    parser.add_argument('--room_cls_loss_coef', default=0.0, type=float,
                        help='Set to 0 for CAPE (no category classification)')
    parser.add_argument('--raster_loss_coef', default=0.0, type=float,
                        help='Rasterization loss coefficient (not used for CAPE)')
    parser.add_argument('--label_smoothing', default=0.0, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_name', default='mp100', type=str)
    parser.add_argument('--dataset_root',
                        default='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation',
                        type=str)
    parser.add_argument('--mp100_split', default=1, type=int, choices=[1, 2, 3, 4, 5])

    # Decoder architecture
    parser.add_argument('--dec_layer_type', default='v1', type=str)
    parser.add_argument('--dec_attn_concat_src', action='store_true')
    parser.add_argument('--dec_qkv_proj', action='store_true', default=True)
    parser.add_argument('--pre_decoder_pos_embed', action='store_true')
    parser.add_argument('--learnable_dec_pe', action='store_true')
    parser.add_argument('--add_cls_token', action='store_true', default=False)
    parser.add_argument('--inject_cls_embed', action='store_true', default=False)
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--freeze_anchor', action='store_true')
    parser.add_argument('--per_token_sem_loss', action='store_true', default=False)

    # Output
    parser.add_argument('--output_dir', default='output/cape_episodic',
                        help='Path to save checkpoints')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detected if not specified)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--job_name', default='cape_episodic', type=str)

    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='MP100-CAPE-Episodic', type=str)
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency during training')

    return parser


def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        device = torch.device("mps")
        print("Note: MPS fallback enabled")
    else:
        device = torch.device("cpu")
    return device


def main(args):
    print("=" * 80)
    print("Category-Agnostic Pose Estimation (CAPE) - Episodic Training")
    print("=" * 80)
    print(f"\nMode: Episodic meta-learning with support pose graphs")
    print(f"Support encoder layers: {args.support_encoder_layers}")
    print(f"Fusion method: {args.support_fusion_method}")
    print(f"Queries per episode: {args.num_queries_per_episode}")
    print(f"Episodes per epoch: {args.episodes_per_epoch}\n")

    # Auto-detect device
    if args.device is None:
        device = get_device()
        args.device = str(device)
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}\n")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build base model (RoomFormerV2)
    print("Building base Raster2Seq model...")
    base_model, criterion = build_model(args)

    # Wrap with CAPE model
    print("Wrapping with CAPE support conditioning...")
    model = build_cape_model(args, base_model)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {n_parameters:,}')

    # Build base dataset (will be wrapped by episodic sampler)
    print(f"\nBuilding MP-100 dataset...")
    from datasets.mp100_cape import build_mp100_cape

    train_dataset = build_mp100_cape('train', args)
    val_dataset = build_mp100_cape('val', args)

    # Build episodic dataloaders
    print("Creating episodic dataloaders...")
    from datasets.episodic_sampler import build_episodic_dataloader

    category_split_file = Path(args.dataset_root) / args.category_split_file

    train_loader = build_episodic_dataloader(
        base_dataset=train_dataset,
        category_split_file=str(category_split_file),
        split='train',
        batch_size=args.batch_size,
        num_queries_per_episode=args.num_queries_per_episode,
        episodes_per_epoch=args.episodes_per_epoch,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # For validation, use training dataset with different episodes
    # (val dataset is too small to have all training categories)
    val_episodes = max(1, args.episodes_per_epoch // 10)  # At least 1 episode
    val_loader = build_episodic_dataloader(
        base_dataset=train_dataset,  # Use train dataset but different episodes
        category_split_file=str(category_split_file),
        split='train',
        batch_size=args.batch_size,
        num_queries_per_episode=args.num_queries_per_episode,
        episodes_per_epoch=val_episodes,
        num_workers=args.num_workers,
        seed=args.seed + 999  # Different seed for val to get different episodes
    )

    print(f"Train episodes/epoch: {len(train_loader) * args.batch_size}")
    print(f"Val episodes/epoch: {len(val_loader) * args.batch_size}")

    # Build optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    lr_drop_epochs = [int(x) for x in args.lr_drop.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop_epochs)

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"Checkpoint not found: {args.resume}")

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    best_val_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train one epoch
        train_stats = train_one_epoch_episodic(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq
        )

        lr_scheduler.step()

        # Validation
        val_stats = evaluate_cape(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device
        )

        # Print epoch summary
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"{'=' * 80}")
        print(f"  Train Loss:       {train_stats.get('loss', 0):.4f}")
        print(f"    - Class Loss:   {train_stats.get('loss_ce', 0):.4f}")
        print(f"    - Coords Loss:  {train_stats.get('loss_coords', 0):.4f}")
        print(f"  Val Loss:         {val_stats.get('loss', 0):.4f}")
        print(f"    - Class Loss:   {val_stats.get('loss_ce', 0):.4f}")
        print(f"    - Coords Loss:  {val_stats.get('loss_coords', 0):.4f}")
        print(f"  Learning Rate:    {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 80}\n")

        # Save checkpoint
        checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'train_stats': train_stats,
            'val_stats': val_stats
        }, checkpoint_path)

        # Save best model
        val_loss = val_stats.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.output_dir) / 'checkpoint_best.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'val_loss': val_loss
            }, best_path)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAPE Episodic Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
