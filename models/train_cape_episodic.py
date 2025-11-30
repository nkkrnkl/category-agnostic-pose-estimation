#!/usr/bin/env python3
"""
Episodic Training Script for Category-Agnostic Pose Estimation (CAPE) on MP-100.
"""
import argparse
import datetime
import json
import random
import os
import sys
import time
from pathlib import Path
import copy
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models import build_model
from models.cape_model import build_cape_model
from models.cape_losses import build_cape_criterion
from models.engine_cape import train_one_epoch_episodic, evaluate_cape
def get_args_parser():
    parser = argparse.ArgumentParser('CAPE Episodic Training', add_help=False)
    parser.add_argument('--cape_mode', action='store_true', default=True,
                        help='Enable CAPE episodic training')
    parser.add_argument('--support_encoder_layers', default=3, type=int,
                        help='Number of layers in support pose graph encoder')
    parser.add_argument('--support_fusion_method', default='cross_attention',
                        choices=['cross_attention', 'concat', 'add'],
                        help='How to fuse support with query features')
    parser.add_argument('--num_queries_per_episode', default=2, type=int,
                        help='Number of query images per episode')
    parser.add_argument('--num_support_per_episode', default=1, type=int,
                        help='Number of support images per episode (1-shot, 5-shot, etc.)')
    parser.add_argument('--episodes_per_epoch', default=1000, type=int,
                        help='Number of episodes per training epoch')
    parser.add_argument('--category_split_file', default='category_splits.json',
                        help='Path to category split JSON file')
    parser.add_argument('--use_geometric_encoder', action='store_true', default=False,
                        help='Use GeometricSupportEncoder (CapeX-inspired) instead of old SupportPoseGraphEncoder')
    parser.add_argument('--use_gcn_preenc', action='store_true', default=False,
                        help='Use GCN pre-encoding in geometric support encoder (requires --use_geometric_encoder)')
    parser.add_argument('--num_gcn_layers', default=2, type=int,
                        help='Number of GCN layers if use_gcn_preenc=True')
    parser.add_argument('--debug_overfit_category', default=None, type=int,
                        help='DEBUG: Train on single category ID for overfitting test (ignores category_split_file)')
    parser.add_argument('--debug_overfit_episodes', default=10, type=int,
                        help='DEBUG: Number of episodes per epoch when using --debug_overfit_category')
    parser.add_argument('--debug_single_image', default=None, type=int,
                        help='DEBUG: Train on single image from specified category ID. Uses first available image from that category.')
    parser.add_argument('--debug_single_image_index', default=None, type=int,
                        help='DEBUG: Specific image index to use (within the category). If not specified, uses first image.')
    parser.add_argument('--debug_single_image_path', default=None, type=str,
                        help='DEBUG: Exact image file path to use (e.g., "bison_body/000000001120.jpg"). Takes precedence over --debug_single_image.')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Number of episodes per batch (optimized for A100: 64)')
    parser.add_argument('--accumulation_steps', default=4, type=int,
                        help='Number of mini-batches to accumulate gradients over (effective_batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default='200,250', type=str)
    parser.add_argument('--early_stopping_patience', default=20, type=int,
                        help='Stop training if PCK does not improve for N epochs (0 to disable)')
    parser.add_argument('--stop_when_loss_zero', action='store_true',
                        help='For single-image mode: Stop training when all losses reach near-zero (perfect overfitting). '
                             'Overrides early stopping. Requires --debug_single_image or --debug_single_image_path.')
    parser.add_argument('--loss_zero_threshold', default=1e-5, type=float,
                        help='Threshold for considering loss as "zero" when using --stop_when_loss_zero (default: 1e-5)')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--image_norm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--poly2seq', action='store_true', default=True,
                        help='Enable poly2seq mode (sequence-to-sequence for keypoints)')
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
    parser.add_argument('--semantic_classes', default=70, type=int,
                        help='Total number of categories in MP-100')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--aux_loss', action='store_true', default=True)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--coords_loss_coef', default=5, type=float)
    parser.add_argument('--room_cls_loss_coef', default=0.0, type=float,
                        help='Set to 0 for CAPE (no category classification)')
    parser.add_argument('--raster_loss_coef', default=0.0, type=float,
                        help='Rasterization loss coefficient (not used for CAPE)')
    parser.add_argument('--eos_weight', default=20.0, type=float,
                        help='Class weight for EOS token to combat class imbalance (default: 20.0)')
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--dataset_name', default='mp100', type=str)
    parser.add_argument('--dataset_root',
                        default='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation',
                        type=str)
    parser.add_argument('--mp100_split', default=1, type=int, choices=[1, 2, 3, 4, 5])
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
    parser.add_argument('--output_dir', default='output/cape_episodic',
                        help='Path to save checkpoints')
    parser.add_argument('--device', default=None,
                        help='Device (auto-detected if not specified)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of DataLoader worker processes (optimized for A100: 16)')
    parser.add_argument('--job_name', default='cape_episodic', type=str)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='MP100-CAPE-Episodic', type=str)
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency during training')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) for faster training (default: True)')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=True,
                        help='Enable cuDNN benchmark mode for faster convolutions (default: True)')
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='Use deterministic cuDNN algorithms (slower but reproducible)')
    return parser
def get_device():
    """
    Auto-detect best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Note: Using MPS (Apple Silicon GPU)")
        print("      MPS fallback enabled for unsupported ops (e.g., grid_sampler)")
    else:
        device = torch.device("cpu")
    return device
def setup_cuda_optimizations(args, device):
    """
    Setup CUDA optimizations for faster training.
    
    Args:
        args: Training arguments
        device: PyTorch device
    
    Returns:
        scaler: GradScaler for AMP or None
        device: PyTorch device
    """
    if device.type != 'cuda':
        return None, device
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark mode enabled (faster convolutions)")
    if args.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ cuDNN deterministic mode enabled (reproducible but slower)")
    scaler = None
    if args.use_amp and device.type == 'cuda':
        try:
            if hasattr(torch.amp, 'GradScaler'):
                scaler = torch.amp.GradScaler('cuda')
            else:
                scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision training (AMP) enabled")
            print("  → Expected speedup: ~2x on modern GPUs")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create AMP scaler: {e}")
            print("   Continuing without AMP...")
    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"✓ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    return scaler, device
def main(args):
    """
    Main training function for CAPE.
    
    Args:
        args: Training arguments
    """
    print("=" * 80)
    print("Category-Agnostic Pose Estimation (CAPE) - Episodic Training")
    print("=" * 80)
    print(f"\nMode: Episodic meta-learning with support pose graphs")
    print(f"Support encoder: {'Geometric (CapeX-inspired)' if args.use_geometric_encoder else 'Original'}")
    if args.use_geometric_encoder:
        print(f"  - GCN pre-encoding: {'Enabled' if args.use_gcn_preenc else 'Disabled'}")
        if args.use_gcn_preenc:
            print(f"  - GCN layers: {args.num_gcn_layers}")
    print(f"Support encoder layers: {args.support_encoder_layers}")
    print(f"Fusion method: {args.support_fusion_method}")
    print(f"Queries per episode: {args.num_queries_per_episode}")
    print(f"Episodes per epoch: {args.episodes_per_epoch}\n")
    if args.device is None:
        device = get_device()
        args.device = str(device)
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    scaler, device = setup_cuda_optimizations(args, device)
    if scaler is None and device.type == 'cuda':
        print("⚠️  Warning: AMP scaler not created (AMP disabled or not on CUDA)")
    print()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    from datasets.mp100_cape import build_mp100_cape
    train_dataset = build_mp100_cape('train', args)
    val_dataset = build_mp100_cape('val', args)
    tokenizer = train_dataset.get_tokenizer()
    print(f"Tokenizer: {tokenizer}")
    print(f"  vocab_size: {len(tokenizer) if tokenizer else 'N/A'}")
    print(f"  num_bins: {tokenizer.num_bins if tokenizer else 'N/A'}")
    print()
    print("Building base Raster2Seq model...")
    base_model, _ = build_model(args, tokenizer=tokenizer)
    print("Building CAPE-specific loss criterion...")
    num_classes = 3 if not args.add_cls_token else 4
    criterion = build_cape_criterion(args, num_classes=num_classes)
    criterion.to(device)
    print(f"✓ CAPE criterion created with visibility masking support")
    print("Wrapping with CAPE support conditioning...")
    model = build_cape_model(args, base_model)
    model.to(device)
    model_device = next(model.parameters()).device
    device_type_matches = (
        str(model_device).split(':')[0] == str(device).split(':')[0]
    )
    if not device_type_matches:
        print(f"⚠️  Warning: Model device ({model_device}) doesn't match expected device ({device})")
    else:
        print(f"✓ Model moved to device: {model_device}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {n_parameters:,}')
    print("Creating episodic dataloaders...")
    from datasets.episodic_sampler import build_episodic_dataloader
    category_split_file = Path(args.dataset_root) / args.category_split_file
    if args.debug_overfit_category is not None:
        print("\n" + "=" * 80)
        print("⚠️  DEBUG OVERFIT MODE ENABLED")
        print("=" * 80)
        print(f"Training on SINGLE category: {args.debug_overfit_category}")
        print(f"Episodes per epoch: {args.debug_overfit_episodes}")
        print(f"Expected: Training loss → 0 within ~20 epochs")
        print(f"Purpose: Verify model can learn (debugging tool)")
        print("=" * 80 + "\n")
        args.episodes_per_epoch = args.debug_overfit_episodes
        import tempfile
        temp_split = {
            "description": f"DEBUG: Single-category overfitting mode (category {args.debug_overfit_category})",
            "total_categories": 1,
            "train_categories": 1,
            "test_categories": 0,
            "val_categories": 0,
            "train": [args.debug_overfit_category],
            "val": [],
            "test": []
        }
        temp_split_fd, temp_split_path = tempfile.mkstemp(suffix='.json', text=True)
        os.close(temp_split_fd)
        with open(temp_split_path, 'w') as f:
            json.dump(temp_split, f, indent=2)
        category_split_file = Path(temp_split_path)
        print(f"Using temporary category split: {category_split_file}\n")
    single_image_mode = False
    single_image_category = None
    single_image_idx = None
    if args.debug_single_image_path is not None:
        print("\n" + "=" * 80)
        print("⚠️  DEBUG SINGLE IMAGE MODE ENABLED (by file path)")
        print("=" * 80)
        print(f"Training on SINGLE IMAGE with path: {args.debug_single_image_path}")
        from datasets.mp100_cape import build_mp100_cape
        temp_dataset = build_mp100_cape('train', args)
        image_path = args.debug_single_image_path.strip()
        dataset_root_normalized = os.path.normpath(args.dataset_root)
        data_folder = os.path.normpath(os.path.join(dataset_root_normalized, 'data'))
        image_path_normalized = os.path.normpath(image_path)
        if image_path_normalized.startswith(data_folder + os.sep) or image_path_normalized == data_folder:
            image_path = os.path.relpath(image_path_normalized, data_folder)
        elif image_path_normalized.startswith(dataset_root_normalized + os.sep):
            relative_to_root = os.path.relpath(image_path_normalized, dataset_root_normalized)
            if relative_to_root.startswith('data' + os.sep):
                image_path = relative_to_root[len('data' + os.sep):]
            elif relative_to_root == 'data':
                raise ValueError(f"Path points to data folder, not an image: {args.debug_single_image_path}")
            else:
                image_path = relative_to_root
        else:
            image_path = image_path_normalized
        if image_path.startswith('dl-category-agnostic-pose-mp100-data/'):
            image_path = image_path.replace('dl-category-agnostic-pose-mp100-data/', '')
        image_path = image_path.replace(os.sep, '/').lstrip('/')
        def normalize_path_for_comparison(path):
            """
            Normalize path for comparison.
            
            Args:
                path: Path string
            
            Returns:
                Normalized path
            """
            return path.replace('\\', '/').strip('/')
        image_path_normalized = normalize_path_for_comparison(image_path)
        filename = os.path.basename(image_path)
        print(f"🔍 Searching for image:")
        print(f"   Normalized path: {image_path_normalized}")
        print(f"   Filename: {filename}")
        print(f"   Dataset root: {temp_dataset.root}")
        full_path_check = os.path.join(temp_dataset.root, image_path)
        file_exists = os.path.exists(full_path_check)
        print(f"   File exists at {full_path_check}: {file_exists}")
        found_idx = None
        sample_file_names = []
        same_dir_files = []
        image_dir = os.path.dirname(image_path_normalized)
        for idx in range(len(temp_dataset)):
            try:
                img_id = temp_dataset.ids[idx]
                img_info = temp_dataset.coco.loadImgs(img_id)[0]
                coco_file_name = img_info['file_name']
                if len(sample_file_names) < 5:
                    sample_file_names.append(coco_file_name)
                coco_dir = os.path.dirname(normalize_path_for_comparison(coco_file_name))
                if coco_dir == image_dir and len(same_dir_files) < 10:
                    same_dir_files.append(coco_file_name)
                coco_file_name_normalized = normalize_path_for_comparison(coco_file_name)
                if coco_file_name_normalized == image_path_normalized:
                    full_path = os.path.join(temp_dataset.root, coco_file_name)
                    if os.path.exists(full_path):
                        found_idx = idx
                        ann_ids = temp_dataset.coco.getAnnIds(imgIds=img_id)
                        anns = temp_dataset.coco.loadAnns(ann_ids)
                        if len(anns) > 0:
                            single_image_category = anns[0].get('category_id', 0)
                        print(f"✅ Found exact match at index {idx}: {coco_file_name}")
                        break
                elif coco_file_name_normalized.endswith(image_path_normalized) or image_path_normalized.endswith(coco_file_name_normalized):
                    full_path = os.path.join(temp_dataset.root, coco_file_name)
                    if os.path.exists(full_path):
                        found_idx = idx
                        ann_ids = temp_dataset.coco.getAnnIds(imgIds=img_id)
                        anns = temp_dataset.coco.loadAnns(ann_ids)
                        if len(anns) > 0:
                            single_image_category = anns[0].get('category_id', 0)
                        print(f"✅ Found endswith match at index {idx}: {coco_file_name}")
                        break
                elif os.path.basename(coco_file_name) == filename:
                    full_path = os.path.join(temp_dataset.root, coco_file_name)
                    if os.path.exists(full_path):
                        found_idx = idx
                        ann_ids = temp_dataset.coco.getAnnIds(imgIds=img_id)
                        anns = temp_dataset.coco.loadAnns(ann_ids)
                        if len(anns) > 0:
                            single_image_category = anns[0].get('category_id', 0)
                        print(f"✅ Found filename match at index {idx}: {coco_file_name}")
                        break
            except Exception as e:
                continue
        if found_idx is None and file_exists:
            print(f"⚠️  File exists but not matched in train annotations. Trying filename-only search...")
            for idx in range(len(temp_dataset)):
                try:
                    img_id = temp_dataset.ids[idx]
                    img_info = temp_dataset.coco.loadImgs(img_id)[0]
                    coco_file_name = img_info['file_name']
                    if os.path.basename(coco_file_name) == filename:
                        full_path = os.path.join(temp_dataset.root, coco_file_name)
                        if os.path.exists(full_path):
                            try:
                                same_file = os.path.samefile(full_path, full_path_check)
                            except (OSError, ValueError):
                                same_file = os.path.normpath(full_path) == os.path.normpath(full_path_check)
                            if same_file:
                                found_idx = idx
                                ann_ids = temp_dataset.coco.getAnnIds(imgIds=img_id)
                                anns = temp_dataset.coco.loadAnns(ann_ids)
                                if len(anns) > 0:
                                    single_image_category = anns[0].get('category_id', 0)
                                print(f"✅ Found by filename and path verification at index {idx}: {coco_file_name}")
                                break
                except Exception as e:
                    continue
        if found_idx is None and file_exists:
            print(f"⚠️  File not found in train split. Checking val and test splits...")
            for split in ['val', 'test']:
                try:
                    from datasets.mp100_cape import build_mp100_cape
                    split_dataset = build_mp100_cape(split, args)
                    for idx in range(len(split_dataset)):
                        try:
                            img_id = split_dataset.ids[idx]
                            img_info = split_dataset.coco.loadImgs(img_id)[0]
                            coco_file_name = img_info['file_name']
                            coco_file_name_normalized = normalize_path_for_comparison(coco_file_name)
                            if (coco_file_name_normalized == image_path_normalized or 
                                os.path.basename(coco_file_name) == filename):
                                full_path = os.path.join(split_dataset.root, coco_file_name)
                                if os.path.exists(full_path):
                                    try:
                                        same_file = os.path.samefile(full_path, full_path_check)
                                    except (OSError, ValueError):
                                        same_file = os.path.normpath(full_path) == os.path.normpath(full_path_check)
                                    if same_file:
                                        print(f"⚠️  Found image in {split} split, not train split!")
                                        print(f"   This image is in the {split} annotations, not train.")
                                        print(f"   Consider using a different image from the train split.")
                                        print(f"   Or modify the code to allow training on {split} images.")
                                        raise ValueError(
                                            f"Image found in {split} split, not train split.\n"
                                            f"  File: {args.debug_single_image_path}\n"
                                            f"  Found in: {split} annotations\n"
                                            f"  COCO file_name: {coco_file_name}\n"
                                            f"  Please use an image from the train split for training."
                                        )
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
        if found_idx is None:
            error_msg = (
                f"Image not found in train annotations: {args.debug_single_image_path}\n"
                f"  Searched for (normalized): {image_path_normalized}\n"
                f"  Filename: {filename}\n"
                f"  Directory: {image_dir}\n"
                f"  Full path checked: {full_path_check}\n"
                f"  File exists: {file_exists}\n"
            )
            if same_dir_files:
                error_msg += f"\n  Files from same directory ({image_dir}) in train annotations:\n"
                for sample in same_dir_files[:10]:
                    error_msg += f"    - {sample}\n"
                error_msg += f"  (Found {len(same_dir_files)} files from {image_dir} directory)\n"
            elif sample_file_names:
                error_msg += f"\n  Sample file names in COCO train annotations:\n"
                for sample in sample_file_names[:10]:
                    error_msg += f"    - {sample}\n"
            error_msg += (
                f"\n  Possible issues:\n"
                f"    1. Image is in val/test split, not train split\n"
                f"    2. Image exists on disk but is not annotated\n"
                f"    3. File name format differs in annotations\n"
                f"    4. Image was excluded from train split\n"
                f"\n  Suggestions:\n"
                f"    - Try a different image from the same category\n"
                f"    - Check if the image is in val/test annotations\n"
                f"    - Verify the image has annotations in the train split"
            )
            raise ValueError(error_msg)
        single_image_idx = found_idx
        single_image_mode = True
        print(f"✅ Found image at dataset index: {single_image_idx}")
        if single_image_category is not None:
            print(f"   Category ID: {single_image_category}")
        print(f"   File path: {image_path}")
        print(f"Episodes per epoch: {args.episodes_per_epoch}")
        print(f"Expected: Training loss → 0 within ~10 epochs")
        print(f"Purpose: Extreme overfitting test on single image")
        print("=" * 80 + "\n")
        args.episodes_per_epoch = min(args.episodes_per_epoch, 50)
    elif args.debug_single_image is not None:
        print("\n" + "=" * 80)
        print("⚠️  DEBUG SINGLE IMAGE MODE ENABLED")
        print("=" * 80)
        print(f"Training on SINGLE IMAGE from category: {args.debug_single_image}")
        from datasets.mp100_cape import build_mp100_cape
        temp_dataset = build_mp100_cape('train', args)
        category_images = []
        for idx in range(len(temp_dataset)):
            try:
                img_id = temp_dataset.ids[idx]
                ann_ids = temp_dataset.coco.getAnnIds(imgIds=img_id)
                anns = temp_dataset.coco.loadAnns(ann_ids)
                if len(anns) > 0:
                    cat_id = anns[0].get('category_id', 0)
                    if cat_id == args.debug_single_image:
                        img_info = temp_dataset.coco.loadImgs(img_id)[0]
                        path = img_info['file_name']
                        file_name = os.path.join(temp_dataset.root, path)
                        if os.path.exists(file_name):
                            category_images.append(idx)
                        else:
                            print(f"  ⚠️  Skipping image {idx} (file not found: {file_name})")
            except Exception as e:
                continue
        if len(category_images) == 0:
            raise ValueError(
                f"No images found for category {args.debug_single_image} with existing files. "
                f"Please check that the data directory is properly mounted and contains image files."
            )
        if args.debug_single_image_index is not None:
            if args.debug_single_image_index >= len(category_images):
                raise ValueError(f"Image index {args.debug_single_image_index} out of range. "
                               f"Category {args.debug_single_image} has {len(category_images)} images with existing files.")
            single_image_idx = category_images[args.debug_single_image_index]
        else:
            single_image_idx = category_images[0]
        single_image_category = args.debug_single_image
        single_image_mode = True
        print(f"Selected image index: {single_image_idx} (image {category_images.index(single_image_idx) + 1} of {len(category_images)} in category)")
        print(f"Episodes per epoch: {args.episodes_per_epoch}")
        print(f"Expected: Training loss → 0 within ~10 epochs")
        print(f"Purpose: Extreme overfitting test on single image")
        print("=" * 80 + "\n")
        args.episodes_per_epoch = min(args.episodes_per_epoch, 50)
    train_loader = build_episodic_dataloader(
        base_dataset=train_dataset,
        category_split_file=str(category_split_file),
        split='train',
        batch_size=args.batch_size,
        num_queries_per_episode=args.num_queries_per_episode,
        num_support_per_episode=args.num_support_per_episode,
        episodes_per_epoch=args.episodes_per_epoch,
        num_workers=args.num_workers,
        seed=args.seed,
        debug_single_image=single_image_idx if single_image_mode else None,
        debug_single_image_category=single_image_category if single_image_mode else None
    )
    val_episodes = max(1, args.episodes_per_epoch // 10)
    if single_image_mode:
        print("\n" + "=" * 80)
        print("⚠️  SINGLE IMAGE VALIDATION MODE")
        print("=" * 80)
        print(f"Validation will use the SAME image as training:")
        print(f"  - Image index: {single_image_idx}")
        print(f"  - Category ID: {single_image_category}")
        print(f"  - Same image used as both support and query (self-supervised)")
        print(f"  - Purpose: Verify model can perfectly memorize single image")
        print("=" * 80 + "\n")
        val_loader = build_episodic_dataloader(
            base_dataset=train_dataset,
            category_split_file=str(category_split_file),
            split='train',
            batch_size=1,
            num_queries_per_episode=args.num_queries_per_episode,
            num_support_per_episode=args.num_support_per_episode,
            episodes_per_epoch=val_episodes,
            num_workers=args.num_workers,
            seed=args.seed + 999,
            debug_single_image=single_image_idx,
            debug_single_image_category=single_image_category
        )
    else:
        val_loader = build_episodic_dataloader(
            base_dataset=val_dataset,
            category_split_file=str(category_split_file),
            split='val',
            batch_size=1,
            num_queries_per_episode=args.num_queries_per_episode,
            num_support_per_episode=args.num_support_per_episode,
            episodes_per_epoch=val_episodes,
            num_workers=args.num_workers,
            seed=args.seed + 999
        )
    print(f"Train episodes/epoch: {len(train_loader) * args.batch_size}")
    print(f"Val episodes/epoch: {len(val_loader) * args.batch_size}")
    print(f"\nGradient Accumulation:")
    print(f"  - Physical batch size: {args.batch_size} episodes")
    print(f"  - Accumulation steps: {args.accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.accumulation_steps} episodes")
    print(f"  - Memory usage: Same as {args.batch_size} episodes (no extra memory!)")
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
    lr_drop_epochs = [int(x) for x in args.lr_drop.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop_epochs)
    best_pck = 0.0
    epochs_without_improvement = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n{'=' * 80}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'=' * 80}")
            print(f"Checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            if unexpected_keys:
                contaminated = [k for k in unexpected_keys if 'decoder.support_cross_attn' in k or 'decoder.support_attn_norm' in k]
                if contaminated:
                    print(f"  ⚠️  Checkpoint has {len(contaminated)} contaminated keys (from state_dict bug)")
                    print(f"     These are duplicate support layer weights saved in wrong location")
                    print(f"     Will be safely ignored - correct weights loaded from proper location")
                if len(unexpected_keys) > len(contaminated):
                    other_unexpected = len(unexpected_keys) - len(contaminated)
                    print(f"  ⚠️  Checkpoint has {other_unexpected} other unexpected keys")
                    print(f"     These may indicate architecture changes")
            if missing_keys:
                print(f"  ⚠️  Current model has {len(missing_keys)} new keys not in checkpoint")
                print(f"     These will use freshly initialized weights")
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"  ✓ Model weights restored")
            print(f"  ✓ Optimizer state restored")
            print(f"  ✓ LR scheduler restored")
            print(f"  ✓ Will resume from epoch {args.start_epoch}")
            best_pck = checkpoint.get('best_pck', 0.0)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"  ✓ Best PCK restored: {best_pck:.4f}")
            print(f"  ✓ Epochs without improvement: {epochs_without_improvement}")
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state'].cpu())
                print(f"  ✓ Torch RNG state restored")
            if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
                print(f"  ✓ CUDA RNG state restored")
            if 'np_rng_state' in checkpoint:
                np.random.set_state(checkpoint['np_rng_state'])
                print(f"  ✓ NumPy RNG state restored")
            if 'py_rng_state' in checkpoint:
                random.setstate(checkpoint['py_rng_state'])
                print(f"  ✓ Python RNG state restored")
            print(f"{'=' * 80}\n")
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")
            print(f"   Starting training from scratch...\n")
    if args.stop_when_loss_zero and not single_image_mode:
        raise ValueError(
            "ERROR: --stop_when_loss_zero can only be used with single-image mode.\n"
            "Please use --debug_single_image or --debug_single_image_path to enable single-image mode."
        )
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    if single_image_mode and args.stop_when_loss_zero:
        print(f"\n🎯 LOSS-BASED STOPPING ENABLED (Perfect Overfitting Mode)")
        print(f"   Training will stop when all losses reach ≤ {args.loss_zero_threshold:.2e}")
        print(f"   This will create a model that perfectly memorizes the single image.")
        print(f"   Early stopping (PCK-based) is DISABLED in this mode.")
        print()
    print()
    early_stop_triggered = False
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        train_stats = train_one_epoch_episodic(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            accumulation_steps=args.accumulation_steps,
            scaler=scaler
        )
        lr_scheduler.step()
        val_stats = evaluate_cape(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device
        )
        train_loss_ce_final = train_stats.get('loss_ce', 0.0)
        train_loss_coords_final = train_stats.get('loss_coords', 0.0)
        train_loss_total = train_stats.get('loss', 0.0)
        train_loss_final_layer = train_loss_ce_final + train_loss_coords_final
        val_loss = val_stats.get('loss', 0.0)
        val_loss_ce = val_stats.get('loss_ce', 0.0)
        val_loss_coords = val_stats.get('loss_coords', 0.0)
        val_pck = val_stats.get('pck', 0.0)
        val_pck_mean = val_stats.get('pck_mean_categories', 0.0)
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"{'=' * 80}")
        print(f"  Train Loss (all layers):    {train_loss_total:.4f}")
        print(f"  Train Loss (final layer):   {train_loss_final_layer:.4f}")
        print(f"    - Class Loss:             {train_loss_ce_final:.4f}")
        print(f"    - Coords Loss:            {train_loss_coords_final:.4f}")
        print(f"")
        print(f"  Val Loss (final layer):     {val_loss:.4f}")
        print(f"    - Class Loss:             {val_loss_ce:.4f}")
        print(f"    - Coords Loss:            {val_loss_coords:.4f}")
        print(f"")
        print(f"  Val PCK@0.2:                {val_pck:.2%}")
        print(f"    - Mean PCK (categories):  {val_pck_mean:.2%}")
        print(f"")
        if val_loss > 0 and train_loss_final_layer > 0:
            loss_ratio = val_loss / train_loss_final_layer
            if loss_ratio > 1.5:
                print(f"  🛑 OVERFITTING ALERT:  Val/Train = {loss_ratio:.2f}x (val >> train)")
            elif loss_ratio > 1.2:
                print(f"  ⚠️  Overfitting watch:  Val/Train = {loss_ratio:.2f}x (val > train)")
            elif loss_ratio > 0.8:
                print(f"  ✅ Generalization OK:  Val/Train = {loss_ratio:.2f}x (balanced)")
            else:
                print(f"  ℹ️  Val < Train:        Val/Train = {loss_ratio:.2f}x (early training)")
        print(f"  Learning Rate:              {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 80}\n")
        checkpoint_name = (
            f'checkpoint_e{epoch:03d}_'
            f'lr{args.lr:.0e}_'
            f'bs{args.batch_size}_'
            f'acc{args.accumulation_steps}_'
            f'qpe{args.num_queries_per_episode}.pth'
        )
        checkpoint_path = Path(args.output_dir) / checkpoint_name
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'train_stats': train_stats,
            'val_stats': val_stats,
            'best_pck': best_pck,
            'epochs_without_improvement': epochs_without_improvement,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint_dict, checkpoint_path)
        val_pck = val_stats.get('pck', 0.0)
        val_pck_mean = val_stats.get('pck_mean_categories', 0.0)
        if 'train_pck' in val_stats:
            import warnings
            warnings.warn(
                "WARNING: val_stats contains 'train_pck' - this should not happen! "
                "Checkpoint selection should use VALIDATION PCK only."
            )
        pck_improved = False
        if val_pck > best_pck:
            pck_improved = True
            best_pck = val_pck
            epochs_without_improvement = 0
            best_pck_name = (
                f'checkpoint_best_pck_e{epoch:03d}_'
                f'pck{val_pck:.4f}_'
                f'meanpck{val_pck_mean:.4f}.pth'
            )
            best_pck_path = Path(args.output_dir) / best_pck_name
            best_pck_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'val_pck': val_pck,
                'val_pck_mean': val_pck_mean,
                'best_pck': best_pck,
                'epochs_without_improvement': epochs_without_improvement,
                'rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate(),
            }
            if torch.cuda.is_available():
                best_pck_dict['cuda_rng_state'] = torch.cuda.get_rng_state_all()
            torch.save(best_pck_dict, best_pck_path)
            print(f"  ✓ Saved BEST PCK model (VALIDATION PCK: {val_pck:.4f}, Mean PCK: {val_pck_mean:.4f})")
            print(f"     → This checkpoint is selected based on VALIDATION performance, not training.")
        if not pck_improved:
            epochs_without_improvement += 1
            print(f"  → No improvement in VALIDATION PCK for {epochs_without_improvement} epoch(s)")
            print(f"     Best VALIDATION PCK:    {best_pck:.4f}")
            print(f"     Current VALIDATION PCK: {val_pck:.4f}")
            if single_image_mode:
                print(f"     (on {val_stats.get('pck_num_visible', 0)} visible keypoints - same image validation)")
            else:
                print(f"     (on {val_stats.get('pck_num_visible', 0)} visible keypoints across unseen validation categories)")
            if (args.early_stopping_patience > 0 and 
                epochs_without_improvement >= args.early_stopping_patience and
                not (single_image_mode and args.stop_when_loss_zero)):
                print(f"\n{'!' * 80}")
                print(f"Early stopping triggered!")
                print(f"No improvement in PCK for {args.early_stopping_patience} epochs.")
                print(f"Best PCK: {best_pck:.4f} (epoch {epoch - epochs_without_improvement + 1})")
                print(f"{'!' * 80}\n")
                early_stop_triggered = True
                break
        if single_image_mode and args.stop_when_loss_zero:
            train_loss = train_stats.get('loss', float('inf'))
            loss_ce = train_stats.get('loss_ce', float('inf'))
            loss_coords = train_stats.get('loss_coords', float('inf'))
            threshold = args.loss_zero_threshold
            all_losses_zero = (
                train_loss <= threshold and
                loss_ce <= threshold and
                loss_coords <= threshold
            )
            if all_losses_zero:
                print(f"\n{'🎯' * 40}")
                print(f"PERFECT OVERFITTING ACHIEVED!")
                print(f"{'🎯' * 40}")
                print(f"All training losses have reached near-zero:")
                print(f"  - Total Loss:    {train_loss:.2e} (threshold: {threshold:.2e})")
                print(f"  - Class Loss:    {loss_ce:.2e} (threshold: {threshold:.2e})")
                print(f"  - Coords Loss:   {loss_coords:.2e} (threshold: {threshold:.2e})")
                print(f"\nThe model has perfectly memorized the single training image!")
                print(f"Stopping training at epoch {epoch + 1}.")
                print(f"{'🎯' * 40}\n")
                early_stop_triggered = True
                break
            else:
                if epoch % 5 == 0 or train_loss < 0.01:
                    print(f"  📊 Progress toward zero loss:")
                    print(f"     Total Loss:    {train_loss:.6f} (target: ≤{threshold:.2e})")
                    print(f"     Class Loss:    {loss_ce:.6f} (target: ≤{threshold:.2e})")
                    print(f"     Coords Loss:   {loss_coords:.6f} (target: ≤{threshold:.2e})")
    print("\n" + "=" * 80)
    if early_stop_triggered:
        print("Training Stopped Early!")
        print(f"Stopped at epoch {epoch + 1}/{args.epochs}")
    else:
        print("Training Complete!")
        print(f"Completed all {args.epochs} epochs")
    print("=" * 80)
    if single_image_mode:
        print(f"Best PCK (on same image): {best_pck:.4f}")
    else:
        print(f"Best PCK (on unseen val categories): {best_pck:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"\nLook for:")
    if single_image_mode:
        print(f"  - checkpoint_best_pck_*.pth   (highest PCK on same image)")
    else:
        print(f"  - checkpoint_best_pck_*.pth   (highest PCK on unseen categories)")
    print(f"  - checkpoint_e***.pth         (per-epoch checkpoints)")
    if early_stop_triggered:
        print(f"\nEarly stopping saved {args.epochs - epoch - 1} epochs of compute time!")
    print("=" * 80)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAPE Episodic Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)