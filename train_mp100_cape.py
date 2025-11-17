#!/usr/bin/env python3
"""
Training script for Category-Agnostic Pose Estimation (CAPE) on MP-100
Adapts Raster2Seq framework for pose estimation task
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
from engine import evaluate, train_one_epoch
from models import build_model


def trivial_batch_collator(batch):
    """
    Split batch records into batched_inputs and batched_extras
    For MP-100 CAPE, we need to separate image/instances from sequence data
    Filters out None values (missing files that were skipped at runtime)
    """
    # Filter out None values (missing files skipped at runtime)
    batch = [record for record in batch if record is not None]
    
    if not batch:
        return [], {}

    batched_inputs = []
    batched_extras_keys = None
    batched_extras = {}

    for record in batch:
        # Create input dict with image and instances (if applicable)
        input_dict = {
            'image': record['image'],
            'image_id': record['image_id'],
            'height': record['height'],
            'width': record['width'],
        }

        # For MP-100 CAPE: add instances if needed (currently we don't use detectron2 Instances)
        # Just add a placeholder for compatibility
        input_dict['instances'] = None

        batched_inputs.append(input_dict)

        # Collect sequence data as batched_extras
        if 'seq_data' in record:
            if batched_extras_keys is None:
                batched_extras_keys = list(record['seq_data'].keys())
                for key in batched_extras_keys:
                    batched_extras[key] = []

            for key in batched_extras_keys:
                batched_extras[key].append(record['seq_data'][key])

    # Stack batched_extras tensors
    for key in batched_extras:
        batched_extras[key] = torch.stack(batched_extras[key])

    return batched_inputs, batched_extras


def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        # Enable MPS fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        device = torch.device("mps")
        print("Note: MPS fallback enabled for unsupported operations (e.g., grid_sampler)")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def get_args_parser():
    parser = argparse.ArgumentParser('MP-100 CAPE Training', add_help=False)

    # Learning rate parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int, help='Reduced for pose estimation')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default='200,250', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')

    # Input parameters
    parser.add_argument('--input_channels', default=3, type=int, help='RGB images')
    parser.add_argument('--image_size', default=256, type=int, help='Input image size (256 or 512)')
    parser.add_argument('--image_norm', action='store_true', help='Normalize images')
    parser.add_argument('--debug', action='store_true')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Sequence parameters for keypoints
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Max number of keypoints (e.g., 100 keypoints * 2 coords)")
    parser.add_argument('--seq_len', default=200, type=int,
                        help="Maximum sequence length for keypoints")
    parser.add_argument('--num_polys', default=1, type=int,
                        help="Number of polygons/sequences (for CAPE, always 1)")
    parser.add_argument('--vocab_size', default=2000, type=int,
                        help="Vocabulary size for discrete tokenizer")
    parser.add_argument('--masked_attn', action='store_true', default=False,
                        help="Use masked attention in decoder")

    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="Iteratively refine reference points (keypoint coordinates)")
    parser.add_argument('--use_anchor', action='store_true',
                        help="Use learnable anchors for keypoint prediction")

    # Semantic classes - for CAPE, use category count
    parser.add_argument('--semantic_classes', default=49, type=int,
                        help="Number of object categories in MP-100 (49 available)")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--aux_loss', action='store_true', default=True)

    # Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float, help='Keypoint classification loss')
    parser.add_argument('--coords_loss_coef', default=5, type=float, help='Coordinate regression loss')
    parser.add_argument('--room_cls_loss_coef', default=0.5, type=float, help='Category classification loss')
    parser.add_argument('--raster_loss_coef', default=0.0, type=float, help='Rasterization loss coefficient (not used for CAPE)')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing factor')
    parser.add_argument('--per_token_sem_loss', action='store_true', default=False,
                        help='Compute semantic loss per token instead of per sequence')

    # Dataset parameters
    parser.add_argument('--dataset_name', default='mp100', type=str)
    parser.add_argument('--dataset_root', default='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/theodoros',
                        type=str, help='Path to dataset root (contains data/ and annotations/)')
    parser.add_argument('--mp100_split', default=1, type=int, choices=[1, 2, 3, 4, 5],
                        help='Which MP-100 split to use (1-5)')
    parser.add_argument('--skip_missing_files', action='store_true',
                        help='Pre-filter dataset to skip missing image files during initialization (slower startup but prevents runtime errors)')
    parser.add_argument('--skip_missing_at_runtime', action='store_true',
                        help='Skip missing files at runtime (faster than pre-filtering, only checks files as they are accessed)')

    # Decoder architecture
    parser.add_argument('--dec_layer_type', default='v1', type=str,
                        choices=['v1', 'v2', 'v3', 'v4', 'v41', 'v5', 'v6'],
                        help='Decoder layer type')
    parser.add_argument('--dec_attn_concat_src', action='store_true',
                        help='Concatenate source features in decoder self-attention')
    parser.add_argument('--dec_qkv_proj', action='store_true', default=True,
                        help='Use QKV projection in decoder')
    parser.add_argument('--pre_decoder_pos_embed', action='store_true',
                        help='Add positional embedding before decoder')
    parser.add_argument('--learnable_dec_pe', action='store_true',
                        help='Use learnable decoder positional embeddings')
    parser.add_argument('--add_cls_token', action='store_true', default=False,
                        help='Add CLS token to sequences')

    # Output
    parser.add_argument('--output_dir', default='output/mp100_cape',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default=None,
                        help='device to use for training / testing (auto-detected if not specified)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--job_name', default='mp100_cape_split1', type=str)

    # WandB logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', default='MP100-CAPE', type=str)

    return parser


def main(args):
    print("=" * 80)
    print("Training Category-Agnostic Pose Estimation on MP-100")
    print("=" * 80)
    print(f"\nGit SHA: {utils.get_sha()}\n")
    print(args)

    # Setup wandb if requested
    if args.use_wandb:
        import wandb
        utils.setup_wandb()
        wandb.init(project=args.wandb_project)
        wandb.run.name = args.run_name

    # Auto-detect device if not specified
    if args.device is None:
        device = get_device()
        args.device = str(device)  # Set args.device for build_model
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    print("\nBuilding model...")
    model, criterion = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters:,}')

    # Build datasets
    print(f"\nBuilding datasets for split {args.mp100_split}...")
    try:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
        print(f"Train dataset: {len(dataset_train)} samples")
        print(f"Val dataset: {len(dataset_val)} samples")
    except Exception as e:
        print(f"Error building dataset: {e}")
        raise

    # Debug mode - overfit single sample
    if args.debug:
        print("\nDEBUG MODE: Overfitting on single sample")
        dataset_val = torch.utils.data.Subset(copy.deepcopy(dataset_val), [0])
        dataset_train = copy.deepcopy(dataset_val)
        args.batch_size = 1

    # Build dataloaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # Disable pin_memory for MPS devices (not supported)
    use_pin_memory = device.type == 'cuda'

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=trivial_batch_collator, num_workers=args.num_workers,
                                   pin_memory=use_pin_memory)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=trivial_batch_collator,
                                 num_workers=args.num_workers,
                                 pin_memory=use_pin_memory)

    # Build optimizer
    def match_name_keywords(n, name_keywords):
        return any(keyword in n for keyword in name_keywords)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not match_name_keywords(n, args.lr_backbone_names) and
                      not match_name_keywords(n, args.lr_linear_proj_names) and
                      p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(x) for x in args.lr_drop.split(',')])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        # Train (with poly2seq=True for autoregressive sequence generation)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
            poly2seq=True)
        lr_scheduler.step()

        # Evaluate (with poly2seq=True for autoregressive sequence generation)
        test_stats = evaluate(model, criterion, args.dataset_name, data_loader_val, device, poly2seq=True)

        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"{'='*80}")
        print(f"  Learning Rate:     {train_stats.get('lr', args.lr):.6f}")
        print(f"  Train Loss:        {train_stats.get('loss', 0):.4f}")
        print(f"    - Class Loss:    {train_stats.get('loss_ce', 0):.4f}")
        print(f"    - Coords Loss:   {train_stats.get('loss_coords', 0):.4f}")
        print(f"  Val Loss:          {test_stats.get('loss', 0):.4f}")
        print(f"    - Class Loss:    {test_stats.get('loss_ce', 0):.4f}")
        print(f"    - Coords Loss:   {test_stats.get('loss_coords', 0):.4f}")
        print(f"{'='*80}\n")

        # Save checkpoint
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 20 == 0 or (epoch + 1) in [int(x) for x in args.lr_drop.split(',')]:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "lr": train_stats.get('lr', args.lr)})
            wandb.log({"train/loss": train_stats.get('loss', 0)})
            wandb.log({"val/loss": test_stats.get('loss', 0)})

        # Save log
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n{"=" * 80}')
    print(f'Training completed in {total_time_str}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MP-100 CAPE training script', parents=[get_args_parser()])
    args = parser.parse_args()

    now = datetime.datetime.now()
    args.run_name = args.job_name
    args.output_dir = os.path.join(args.output_dir, args.run_name)

    if args.debug:
        args.batch_size = 1
        args.num_workers = 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
