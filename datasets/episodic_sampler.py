"""
Episodic Sampler for Category-Agnostic Pose Estimation (CAPE)

This module implements episodic training where each episode consists of:
1. A category sampled from training categories
2. A support example from that category (provides pose graph)
3. Query examples from the same category (to predict keypoints on)

This trains the model to use support pose graphs to generalize to new categories.
"""

import torch
import torch.utils.data as data
import numpy as np
import random
import json
from pathlib import Path
from collections import defaultdict
from datasets.mp100_cape import ImageNotFoundError


class EpisodicSampler:
    """
    Episodic sampler for CAPE training.

    Each episode:
    1. Sample a category c from training categories
    2. Sample 1 support image (provides pose graph)
    3. Sample K query images (predict keypoints using support graph)
    """

    def __init__(self, dataset, category_split_file, split='train',
                 num_queries_per_episode=2, seed=None):
        """
        Args:
            dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train' or 'test' (determines which categories to use)
            num_queries_per_episode: Number of query images per episode
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.split = split
        self.num_queries = num_queries_per_episode

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load category splits
        with open(category_split_file) as f:
            category_splits = json.load(f)

        if split == 'train':
            self.categories = category_splits['train']
        elif split == 'test':
            self.categories = category_splits['test']
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Episodic sampler for {split} split: {len(self.categories)} categories")

        # Build category -> image indices mapping
        self.category_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            try:
                # Get category from dataset without loading full sample
                img_id = dataset.ids[idx]
                ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
                anns = dataset.coco.loadAnns(ann_ids)

                if len(anns) > 0:
                    cat_id = anns[0].get('category_id', 0)
                    if cat_id in self.categories:
                        self.category_to_indices[cat_id].append(idx)
            except:
                continue

        # Filter out categories with too few examples
        min_examples = num_queries_per_episode + 1  # Need at least support + queries
        self.categories = [
            cat for cat in self.categories
            if len(self.category_to_indices[cat]) >= min_examples
        ]

        print(f"Valid categories (>={min_examples} examples): {len(self.categories)}")
        print(f"Samples per category: min={min([len(self.category_to_indices[c]) for c in self.categories])}, "
              f"max={max([len(self.category_to_indices[c]) for c in self.categories])}")

    def sample_episode(self):
        """
        Sample one episode.

        Returns:
            episode: dict containing:
                - category_id: sampled category
                - support_idx: index of support image in dataset
                - query_indices: list of query image indices
        """
        # Sample a category
        category_id = random.choice(self.categories)

        # Get all image indices for this category
        indices = self.category_to_indices[category_id]

        # Sample support + query indices (without replacement)
        sampled_indices = random.sample(indices, self.num_queries + 1)

        support_idx = sampled_indices[0]
        query_indices = sampled_indices[1:]

        return {
            'category_id': category_id,
            'support_idx': support_idx,
            'query_indices': query_indices
        }

    def __len__(self):
        """
        Return approximate number of possible episodes.
        This is used for determining epoch length.
        """
        # Approximate: total images / queries per episode
        total_images = sum(len(indices) for indices in self.category_to_indices.values())
        return total_images // self.num_queries


class EpisodicDataset(data.Dataset):
    """
    Dataset wrapper that provides episodic sampling.

    Each item returned is an episode containing:
    - Support image + extracted pose graph
    - Multiple query images from the same category
    """

    def __init__(self, base_dataset, category_split_file, split='train',
                 num_queries_per_episode=2, episodes_per_epoch=1000, seed=None):
        """
        Args:
            base_dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train' or 'test'
            num_queries_per_episode: Number of query images per episode
            episodes_per_epoch: Number of episodes per training epoch
            seed: Random seed
        """
        self.base_dataset = base_dataset
        self.episodes_per_epoch = episodes_per_epoch

        # Create episodic sampler
        self.sampler = EpisodicSampler(
            base_dataset,
            category_split_file,
            split=split,
            num_queries_per_episode=num_queries_per_episode,
            seed=seed
        )

        print(f"EpisodicDataset: {episodes_per_epoch} episodes/epoch, "
              f"{num_queries_per_episode} queries/episode")

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        """
        Sample and return one episode.
        Retries with different episodes if images are missing.

        Returns:
            episode_data: dict containing:
                - support_image: Support image tensor
                - support_coords: Support keypoint coordinates (N, 2)
                - support_mask: Mask for valid support keypoints (N,)
                - query_images: List of query image tensors
                - query_targets: List of query keypoint targets (seq_data)
                - category_id: Category ID
        """
        # Keep trying until we find a valid episode (skip missing images)
        max_retries = 100  # High limit to prevent infinite loops, but should rarely be hit
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Sample episode
                episode = self.sampler.sample_episode()

                # Load support image
                support_data = self.base_dataset[episode['support_idx']]

                # Extract support pose graph (normalized coordinates)
                support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)

                # Normalize support coordinates to [0, 1] using bbox dimensions
                # Note: After mp100_cape.py modifications, keypoints are already bbox-relative
                # and height/width are the dimensions AFTER transform (512x512)
                h, w = support_data['height'], support_data['width']
                support_coords[:, 0] /= w  # x normalized by width
                support_coords[:, 1] /= h  # y normalized by height

                # Create support mask (all valid for now)
                support_mask = torch.ones(len(support_coords), dtype=torch.bool)

                # Extract skeleton edges for support pose graph
                support_skeleton = support_data.get('skeleton', [])

                # Load query images
                query_images = []
                query_targets = []
                query_metadata = []

                for query_idx in episode['query_indices']:
                    query_data = self.base_dataset[query_idx]
                    query_images.append(query_data['image'])
                    query_targets.append(query_data['seq_data'])
                    query_metadata.append({
                        'image_id': query_data['image_id'],
                        'height': query_data['height'],
                        'width': query_data['width'],
                        'keypoints': query_data['keypoints'],
                        'num_keypoints': query_data['num_keypoints'],
                        'bbox': query_data.get('bbox', [0, 0, query_data['width'], query_data['height']]),
                        'bbox_width': query_data.get('bbox_width', query_data['width']),
                        'bbox_height': query_data.get('bbox_height', query_data['height']),
                        'visibility': query_data.get('visibility', [1] * query_data['num_keypoints'])
                    })

                # Successfully loaded all images - return the episode
                return {
                    'support_image': support_data['image'],
                    'support_coords': support_coords,
                    'support_mask': support_mask,
                    'support_skeleton': support_skeleton,
                    'support_bbox': support_data.get('bbox', [0, 0, support_data['width'], support_data['height']]),
                    'support_bbox_width': support_data.get('bbox_width', support_data['width']),
                    'support_bbox_height': support_data.get('bbox_height', support_data['height']),
                    'query_images': query_images,
                    'query_targets': query_targets,
                    'query_metadata': query_metadata,
                    'category_id': episode['category_id']
                }
            except ImageNotFoundError as e:
                # If image is missing, skip and try a different episode
                retry_count += 1
                # Only print warning every 10 retries to avoid spam
                if retry_count % 10 == 0:
                    print(f"Warning: Skipping episode due to missing image (attempt {retry_count})...")
                continue  # Skip and try again with a different episode
        
        # If we've exhausted retries (should be very rare), raise an error
        raise RuntimeError(f"Failed to find valid episode after {max_retries} attempts. "
                         f"This may indicate too many missing images in the dataset.")


def episodic_collate_fn(batch):
    """
    Custom collate function for episodic batches.

    Args:
        batch: List of episode dicts from EpisodicDataset

    Returns:
        collated_batch: dict with batched tensors
    """
    # Each batch contains multiple episodes
    # We need to handle variable number of keypoints per category

    support_images = []
    support_coords_list = []
    support_masks = []
    support_skeletons = []
    query_images_list = []
    query_targets_list = []
    category_ids = []

    for episode in batch:
        support_images.append(episode['support_image'])
        support_coords_list.append(episode['support_coords'])
        support_masks.append(episode['support_mask'])
        support_skeletons.append(episode['support_skeleton'])
        query_images_list.extend(episode['query_images'])
        query_targets_list.extend(episode['query_targets'])
        category_ids.append(episode['category_id'])

    # Stack support images
    support_images = torch.stack(support_images)  # (B, C, H, W)

    # Pad support coordinates to max length in batch
    max_support_kpts = max(coords.shape[0] for coords in support_coords_list)
    support_coords_padded = []
    support_masks_padded = []

    for coords, mask in zip(support_coords_list, support_masks):
        num_kpts = coords.shape[0]
        if num_kpts < max_support_kpts:
            # Pad
            padding = max_support_kpts - num_kpts
            coords = torch.cat([coords, torch.zeros(padding, 2)], dim=0)
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)], dim=0)

        support_coords_padded.append(coords)
        support_masks_padded.append(mask)

    support_coords = torch.stack(support_coords_padded)  # (B, max_kpts, 2)
    support_masks = torch.stack(support_masks_padded)  # (B, max_kpts)

    # Stack query images
    query_images = torch.stack(query_images_list)  # (B*Q, C, H, W)

    # Collate query targets (seq_data)
    batched_seq_data = {}
    for key in query_targets_list[0].keys():
        batched_seq_data[key] = torch.stack([qt[key] for qt in query_targets_list])

    return {
        'support_images': support_images,
        'support_coords': support_coords,
        'support_masks': support_masks,
        'support_skeletons': support_skeletons,  # List of skeleton edge lists
        'query_images': query_images,
        'query_targets': batched_seq_data,
        'category_ids': torch.tensor(category_ids, dtype=torch.long)
    }


def build_episodic_dataloader(base_dataset, category_split_file, split='train',
                              batch_size=2, num_queries_per_episode=2,
                              episodes_per_epoch=1000, num_workers=2, seed=None):
    """
    Build episodic dataloader for CAPE training.

    Args:
        base_dataset: MP100CAPE dataset
        category_split_file: Path to category splits JSON
        split: 'train' or 'test'
        batch_size: Number of episodes per batch
        num_queries_per_episode: Number of query images per episode
        episodes_per_epoch: Total episodes per epoch
        num_workers: Number of worker processes
        seed: Random seed

    Returns:
        dataloader: DataLoader with episodic sampling
    """
    # Create episodic dataset
    episodic_dataset = EpisodicDataset(
        base_dataset=base_dataset,
        category_split_file=category_split_file,
        split=split,
        num_queries_per_episode=num_queries_per_episode,
        episodes_per_epoch=episodes_per_epoch,
        seed=seed
    )

    # Create dataloader
    dataloader = data.DataLoader(
        episodic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=episodic_collate_fn,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    # Test episodic sampler
    print("Testing episodic sampler...")
    from mp100_cape import build_mp100_cape
    import argparse

    # Create dummy args
    args = argparse.Namespace(
        dataset_root='/Users/theodorechronopoulos/Desktop/Cornell Courses/Deep Learning/Project/category-agnostic-pose-estimation',
        mp100_split=1,
        semantic_classes=70,
        image_norm=False,
        vocab_size=2000,
        seq_len=200
    )

    # Build base dataset
    base_dataset = build_mp100_cape('train', args)

    # Test episodic sampler
    sampler = EpisodicSampler(
        base_dataset,
        category_split_file='../category_splits.json',
        split='train',
        num_queries_per_episode=2
    )

    print(f"\nSampling test episode...")
    episode = sampler.sample_episode()
    print(f"Category: {episode['category_id']}")
    print(f"Support idx: {episode['support_idx']}")
    print(f"Query indices: {episode['query_indices']}")

    # Test episodic dataset
    print(f"\nTesting episodic dataset...")
    episodic_dataset = EpisodicDataset(
        base_dataset,
        category_split_file='../category_splits.json',
        split='train',
        num_queries_per_episode=2,
        episodes_per_epoch=10
    )

    episode_data = episodic_dataset[0]
    print(f"Support coords shape: {episode_data['support_coords'].shape}")
    print(f"Number of queries: {len(episode_data['query_images'])}")
    print(f"Category ID: {episode_data['category_id']}")

    print("\n✓ Tests passed!")
