"""
Episodic sampler for meta-learning training.
Creates episodes with support and query instances from the same category.
"""
import random
import numpy as np
import torch
from torch.utils.data import Sampler


class EpisodicBatchSampler(Sampler):
    """
    Sampler that creates episodes for meta-learning.

    Each episode contains:
    - 1 support instance
    - 1 query instance
    - Both from the same category (but different instances)
    """

    def __init__(self, dataset, batch_size, num_episodes, shuffle=True):
        """
        Args:
            dataset: MP100Dataset instance
            batch_size: number of episodes per batch
            num_episodes: total number of episodes per epoch
            shuffle: whether to shuffle categories
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.shuffle = shuffle

        self.category_ids = dataset.category_ids
        self.category_to_instances = dataset.category_to_instances

        # Filter out categories with less than 2 instances
        self.valid_categories = [
            cat_id for cat_id in self.category_ids
            if len(self.category_to_instances[cat_id]) >= 2
        ]

        print(f"Valid categories for episodic sampling: {len(self.valid_categories)}/{len(self.category_ids)}")

    def __iter__(self):
        """
        Generate batches of episode indices.

        Returns:
            iterator over batches, where each batch is a list of (support_idx, query_idx) tuples
        """
        episodes = []

        for _ in range(self.num_episodes):
            # Sample a category
            if self.shuffle:
                cat_id = random.choice(self.valid_categories)
            else:
                cat_id = self.valid_categories[_ % len(self.valid_categories)]

            # Get all instances for this category
            category_instances = self.category_to_instances[cat_id]

            # Sample 2 different instances
            if len(category_instances) < 2:
                continue

            sampled = random.sample(category_instances, 2)
            support_ann = sampled[0]
            query_ann = sampled[1]

            # Get indices in dataset
            support_idx = self.dataset.annotations.index(support_ann)
            query_idx = self.dataset.annotations.index(query_ann)

            episodes.append((support_idx, query_idx))

        # Batch episodes
        for i in range(0, len(episodes), self.batch_size):
            batch = episodes[i:i + self.batch_size]
            if len(batch) == self.batch_size:  # Only yield full batches
                yield batch

    def __len__(self):
        return self.num_episodes // self.batch_size


def collate_episodes(batch):
    """
    Custom collate function for episodic batches.

    Args:
        batch: list of (support_instance, query_instance) tuples

    Returns:
        dict with:
            - support_images: [B, 3, H, W]
            - support_coords: [B, T_max, 2]
            - support_visibility: [B, T_max]
            - query_images: [B, 3, H, W]
            - query_coords: [B, T_max, 2]
            - query_visibility: [B, T_max]
            - num_keypoints: [B]
            - category_ids: list of length B
            - support_bboxes: list of length B
            - query_bboxes: list of length B
    """
    B = len(batch)

    # Unpack support and query
    support_items = [item[0] for item in batch]
    query_items = [item[1] for item in batch]

    # Find max keypoints in this batch
    max_kpts = max(
        max(s['num_keypoints'], q['num_keypoints'])
        for s, q in zip(support_items, query_items)
    )

    # Stack images
    support_images = torch.stack([item['image'] for item in support_items])
    query_images = torch.stack([item['image'] for item in query_items])

    # Pad keypoints and visibility
    support_coords_list = []
    support_vis_list = []
    query_coords_list = []
    query_vis_list = []
    num_keypoints_list = []

    for s_item, q_item in zip(support_items, query_items):
        N = s_item['num_keypoints']
        assert N == q_item['num_keypoints'], "Support and query must have same number of keypoints"

        # Pad support
        s_padded_kpts = np.zeros((max_kpts, 2), dtype=np.float32)
        s_padded_vis = np.zeros(max_kpts, dtype=np.float32)
        s_padded_kpts[:N] = s_item['keypoints'][:N]
        s_padded_vis[:N] = s_item['visibility'][:N]

        # Pad query
        q_padded_kpts = np.zeros((max_kpts, 2), dtype=np.float32)
        q_padded_vis = np.zeros(max_kpts, dtype=np.float32)
        q_padded_kpts[:N] = q_item['keypoints'][:N]
        q_padded_vis[:N] = q_item['visibility'][:N]

        support_coords_list.append(s_padded_kpts)
        support_vis_list.append(s_padded_vis)
        query_coords_list.append(q_padded_kpts)
        query_vis_list.append(q_padded_vis)
        num_keypoints_list.append(N)

    support_coords = torch.from_numpy(np.stack(support_coords_list))
    support_visibility = torch.from_numpy(np.stack(support_vis_list))
    query_coords = torch.from_numpy(np.stack(query_coords_list))
    query_visibility = torch.from_numpy(np.stack(query_vis_list))
    num_keypoints = torch.tensor(num_keypoints_list)

    return {
        'support_images': support_images,
        'support_coords': support_coords,
        'support_visibility': support_visibility,
        'query_images': query_images,
        'query_coords': query_coords,
        'query_visibility': query_visibility,
        'num_keypoints': num_keypoints,
        'category_ids': [s['category_id'] for s in support_items],
        'support_bboxes': [s['bbox'] for s in support_items],
        'query_bboxes': [q['bbox'] for q in query_items],
    }


def create_episodic_dataloader(dataset, batch_size, num_episodes, shuffle=True, num_workers=0):
    """
    Create a DataLoader with episodic sampling.

    Args:
        dataset: MP100Dataset instance
        batch_size: number of episodes per batch
        num_episodes: total number of episodes per epoch
        shuffle: whether to shuffle
        num_workers: number of worker processes

    Returns:
        DataLoader that yields episodic batches
    """
    # Custom sampler
    sampler = EpisodicBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_episodes=num_episodes,
        shuffle=shuffle
    )

    # Custom batch loader
    def batch_loader(episode_batch):
        """Load a batch of episodes."""
        support_query_pairs = []
        for support_idx, query_idx in episode_batch:
            support_item = dataset[support_idx]
            query_item = dataset[query_idx]
            support_query_pairs.append((support_item, query_item))
        return collate_episodes(support_query_pairs)

    # Create dataloader
    from torch.utils.data import DataLoader

    class EpisodicDataset:
        """Wrapper dataset that yields episodes."""
        def __init__(self, base_dataset, sampler):
            self.base_dataset = base_dataset
            self.sampler = sampler
            self.episodes = list(sampler)

        def __len__(self):
            return len(self.episodes)

        def __getitem__(self, idx):
            return self.episodes[idx]

    episodic_dataset = EpisodicDataset(dataset, sampler)

    dataloader = DataLoader(
        episodic_dataset,
        batch_size=1,  # Each "item" is already a batch of episodes
        shuffle=False,  # Shuffling is handled by the sampler
        num_workers=num_workers,
        collate_fn=lambda x: batch_loader(x[0])  # x[0] because batch_size=1
    )

    return dataloader
