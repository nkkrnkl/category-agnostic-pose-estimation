"""
Evaluation script for Raster2Seq CAPE model (1-shot inference only).
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from collections import defaultdict

from dataset import MP100Dataset, collate_fn
from model.model import Raster2SeqCAPE
from utils.geometry import compute_pck
from utils.masking import create_keypoint_mask


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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def infer_oneshot(model, query_image, support_coords, num_keypoints, device):
    """
    1-shot inference: use a real support instance.

    Args:
        model: Raster2SeqCAPE model
        query_image: [1, 3, 512, 512]
        support_coords: [1, T_max, 2]
        num_keypoints: [1]
        device: torch device

    Returns:
        predicted_coords: [N, 2]
    """
    with torch.no_grad():
        # Create keypoint mask
        keypoint_mask = create_keypoint_mask(num_keypoints, support_coords.shape[1]).to(device)

        # Forward pass (autoregressive)
        outputs = model(
            query_images=query_image,
            support_coords=support_coords,
            num_keypoints=num_keypoints,
            keypoint_mask=keypoint_mask,
            teacher_forcing=False
        )

        predicted_coords = outputs['predicted_coords'][0]  # [N, 2]

        # Trim to actual number of keypoints
        N = num_keypoints[0].item()
        predicted_coords = predicted_coords[:N]

        return predicted_coords


def evaluate_oneshot(model, dataset, device, pck_threshold=0.2):
    """
    Evaluate model in 1-shot mode.

    For each query instance:
    1. Sample another instance from the same category as support
    2. Run inference
    3. Compute PCK

    Args:
        model: trained model
        dataset: MP100Dataset
        device: torch device
        pck_threshold: PCK threshold

    Returns:
        dict with evaluation metrics
    """
    model.eval()

    all_pcks = []
    category_pcks = defaultdict(list)

    print("Evaluating 1-shot mode...")

    for idx in tqdm(range(len(dataset))):
        query_item = dataset[idx]

        # Get category and sample a support instance
        cat_id = query_item['category_id']
        category_instances = dataset.get_instances_by_category(cat_id)

        # Filter out the query instance itself
        support_candidates = [i for i in category_instances if i != idx]

        if len(support_candidates) == 0:
            continue

        # Randomly sample a support instance
        support_idx = np.random.choice(support_candidates)
        support_item = dataset[support_idx]

        # Prepare inputs
        query_image = query_item['image'].unsqueeze(0).to(device)
        support_coords = torch.from_numpy(support_item['keypoints']).unsqueeze(0).to(device)
        num_keypoints = torch.tensor([query_item['num_keypoints']]).to(device)

        # Pad support coords
        T_max = model.max_keypoints
        if support_coords.shape[1] < T_max:
            padding = torch.zeros(1, T_max - support_coords.shape[1], 2, device=device)
            support_coords = torch.cat([support_coords, padding], dim=1)

        # Inference
        predicted_coords = infer_oneshot(
            model=model,
            query_image=query_image,
            support_coords=support_coords,
            num_keypoints=num_keypoints,
            device=device
        )

        # Ground truth
        gt_coords = query_item['keypoints']
        visibility = query_item['visibility']
        bbox = query_item['bbox']

        # Compute PCK
        pck, _ = compute_pck(
            pred_coords=predicted_coords.cpu().numpy(),
            gt_coords=gt_coords,
            bbox=bbox,
            visibility=visibility,
            threshold=pck_threshold
        )

        all_pcks.append(pck)
        category_pcks[cat_id].append(pck)

    # Aggregate results
    mean_pck = np.mean(all_pcks)

    # Per-category PCK
    category_mean_pcks = {
        cat_id: np.mean(pcks)
        for cat_id, pcks in category_pcks.items()
    }

    # Mean across categories
    mean_category_pck = np.mean(list(category_mean_pcks.values()))

    results = {
        'mean_pck': mean_pck,
        'mean_category_pck': mean_category_pck,
        'category_pcks': category_mean_pcks,
        'num_samples': len(all_pcks)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Raster2Seq CAPE model (1-shot)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = get_device()

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, model_config = load_model(args.checkpoint, device)

    # Load dataset
    annotation_file = config['data'][f'annotation_{args.split}']
    print(f"\nLoading {args.split} dataset from {annotation_file}...")

    dataset = MP100Dataset(
        annotation_file=annotation_file,
        image_root=config['data']['image_root'],
        image_size=config['data']['image_size'],
        augment=False
    )

    # Evaluate
    pck_threshold = config['evaluation']['pck_threshold']
    results = evaluate_oneshot(model, dataset, device, pck_threshold)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (1-SHOT)")
    print("=" * 80)
    print(f"Mean PCK@{pck_threshold}: {results['mean_pck']:.4f}")
    print(f"Mean Category PCK@{pck_threshold}: {results['mean_category_pck']:.4f}")
    print(f"Number of samples: {results['num_samples']}")
    print("=" * 80)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
