"""
Episodic Sampler for Category-Agnostic Pose Estimation (CAPE).
"""
import os
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
    """
    def __init__(self, dataset, category_split_file, split='train',
                 num_queries_per_episode=2, num_support_per_episode=1, seed=None):
        """
        Initialize episodic sampler.
        
        Args:
            dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train', 'val', or 'test'
            num_queries_per_episode: Number of query images per episode
            num_support_per_episode: Number of support images per episode
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.split = split
        self.num_queries = num_queries_per_episode
        self.num_support = num_support_per_episode
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        with open(category_split_file) as f:
            category_splits = json.load(f)
        if split == 'train':
            self.categories = category_splits['train']
        elif split == 'val':
            self.categories = category_splits['val']
        elif split == 'test':
            self.categories = category_splits['test']
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
        print(f"Episodic sampler for {split} split: {len(self.categories)} categories")
        original_categories = self.categories.copy()
        self.category_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            try:
                img_id = dataset.ids[idx]
                ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
                anns = dataset.coco.loadAnns(ann_ids)
                if len(anns) > 0:
                    cat_id = anns[0].get('category_id', 0)
                    if cat_id in self.categories:
                        self.category_to_indices[cat_id].append(idx)
            except:
                continue
        min_examples = num_queries_per_episode + num_support_per_episode
        self.categories = [
            cat for cat in self.categories
            if len(self.category_to_indices[cat]) >= min_examples
        ]
        print(f"Valid categories (>={min_examples} examples): {len(self.categories)}")
        if len(self.categories) == 0:
            category_counts = {cat: len(self.category_to_indices[cat]) for cat in original_categories}
            categories_with_samples = {cat: count for cat, count in category_counts.items() if count > 0}
            categories_without_samples = {cat: count for cat, count in category_counts.items() if count == 0}
            error_msg = (
                f"\n❌ No valid categories found for {split} split after filtering.\n"
                f"   Required: at least {min_examples} examples per category "
                f"(num_support={num_support_per_episode} + num_queries={num_queries_per_episode})\n\n"
            )
            if categories_with_samples:
                error_msg += f"   Categories with samples (but insufficient):\n"
                for cat, count in sorted(categories_with_samples.items()):
                    error_msg += f"      Category {cat}: {count} samples (need {min_examples})\n"
                error_msg += "\n"
            if categories_without_samples:
                error_msg += f"   Categories with NO samples in dataset: {len(categories_without_samples)}\n"
                error_msg += f"      {sorted(categories_without_samples.keys())}\n\n"
            error_msg += (
                f"   Please check:\n"
                f"   1. The dataset annotation file matches the split (e.g., mp100_splitX_{split}.json)\n"
                f"   2. The category_split_file contains the correct categories for this split\n"
                f"   3. The dataset_root path is correct\n"
                f"   4. Each category has at least {min_examples} samples in the annotation file"
            )
            raise ValueError(error_msg)
        print(f"Samples per category: min={min([len(self.category_to_indices[c]) for c in self.categories])}, "
              f"max={max([len(self.category_to_indices[c]) for c in self.categories])}")
    def sample_episode(self):
        """
        Sample one episode.
        
        Returns:
            episode: Dict with category_id, support_indices, query_indices
        """
        category_id = random.choice(self.categories)
        indices = self.category_to_indices[category_id]
        sampled_indices = random.sample(indices, self.num_queries + self.num_support)
        support_indices = sampled_indices[:self.num_support]
        query_indices = sampled_indices[self.num_support:]
        return {
            'category_id': category_id,
            'support_indices': support_indices,
            'query_indices': query_indices
        }
    def __len__(self):
        """
        Return approximate number of possible episodes.
        
        Returns:
            Number of episodes
        """
        total_images = sum(len(indices) for indices in self.category_to_indices.values())
        return total_images // self.num_queries
class EpisodicDataset(data.Dataset):
    """
    Dataset wrapper that provides episodic sampling.
    """
    def __init__(self, base_dataset, category_split_file, split='train',
                 num_queries_per_episode=2, num_support_per_episode=1, episodes_per_epoch=1000, seed=None,
                 fixed_episodes=False, load_support_images=True):
        """
        Initialize episodic dataset.
        
        Args:
            base_dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train', 'val', or 'test'
            num_queries_per_episode: Number of query images per episode
            num_support_per_episode: Number of support images per episode
            episodes_per_epoch: Number of episodes per training epoch
            seed: Random seed
            fixed_episodes: If True, pre-generate episodes once and reuse (for stable val)
            load_support_images: If True, load support images (for visualization).
                                If False, skip loading to save I/O, CPU, and memory.
                                The model only uses support_coords, not support images.
        """
        self.base_dataset = base_dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.num_support = num_support_per_episode
        self.fixed_episodes = fixed_episodes
        self.load_support_images = load_support_images
        self._cached_episodes = None  # Will store pre-generated episodes if fixed_episodes=True
        self.debug_single_image = None  # For debug mode (single image overfitting test)
        self.debug_single_image_category = None  # Category ID for debug mode

        # Create episodic sampler
        self.sampler = EpisodicSampler(
            base_dataset,
            category_split_file,
            split=split,
            num_queries_per_episode=num_queries_per_episode,
            num_support_per_episode=num_support_per_episode,
            seed=seed
        )

        # Pre-generate episodes if using fixed mode
        if self.fixed_episodes:
            print(f"EpisodicDataset: Pre-generating {episodes_per_epoch} fixed episodes "
                  f"for {split} split (stable curves)...")
            self._cached_episodes = []
            for _ in range(episodes_per_epoch):
                episode = self.sampler.sample_episode()
                self._cached_episodes.append(episode)
            print(f"✓ Cached {len(self._cached_episodes)} episodes")
        else:
            print(f"EpisodicDataset: {episodes_per_epoch} episodes/epoch, "
                  f"{num_support_per_episode}-shot ({num_support_per_episode} support), "
                  f"{num_queries_per_episode} queries/episode")
    def __len__(self):
        return self.episodes_per_epoch
    def __getitem__(self, idx):
        """
        Sample and return one episode.
        
        Returns:
            episode_data: Dict with support_image, support_coords, support_mask,
                         query_images, query_targets, category_id
        """
        if self.debug_single_image is not None:
            try:
                image_data = self.base_dataset[self.debug_single_image]
            except ImageNotFoundError as e:
                raise ImageNotFoundError(
                    f"Single image mode: Image at index {self.debug_single_image} not found. "
                    f"Original error: {e}. "
                    f"Please check that the image file exists in the data directory."
                )
            support_data = image_data
            query_data = image_data
            support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)
            h, w = support_data['height'], support_data['width']
            support_coords[:, 0] /= w
            support_coords[:, 1] /= h
            support_visibility = support_data['visibility']
            support_mask = torch.tensor(
                [v > 0 for v in support_visibility], 
                dtype=torch.bool
            )
            support_skeleton = support_data.get('skeleton', [])
            query_targets = query_data['seq_data']
            return {
                'support_image': support_data['image'],
                'support_coords': support_coords,
                'support_mask': support_mask,
                'support_skeleton': support_skeleton,
                'support_metadata': {
                    'image_id': support_data.get('image_id'),
                    'category_id': self.debug_single_image_category,
                    'height': h,
                    'width': w,
                    'bbox_width': support_data.get('bbox_width', w),
                    'bbox_height': support_data.get('bbox_height', h),
                    'visibility': support_visibility
                },
                'query_images': [query_data['image']],
                'query_targets': [query_targets],
                'query_metadata': [{
                    'image_id': query_data.get('image_id'),
                    'category_id': self.debug_single_image_category,
                    'height': query_data.get('height', h),
                    'width': query_data.get('width', w),
                    'bbox_width': query_data.get('bbox_width', w),
                    'bbox_height': query_data.get('bbox_height', h),
                    'visibility': query_data['visibility']
                }],
                'category_id': self.debug_single_image_category
            }
        max_retries = 100
        retry_count = 0
        
        # Track if we're using fixed episodes
        use_fixed = self.fixed_episodes and self._cached_episodes is not None
        
        while retry_count < max_retries:
            try:
                # Sample episode
                # CRITICAL FIX: On retry (count > 0), fall back to random sampling
                # because the fixed episode at 'idx' is evidently broken/missing
                if use_fixed and retry_count == 0:
                    # First attempt: use the cached fixed episode
                    episode = self._cached_episodes[idx % len(self._cached_episodes)]
                else:
                    # Retry OR random mode: sample a new random episode
                    if retry_count == 1 and use_fixed:
                        print(f"⚠️  Fixed episode {idx} failed (missing image). "
                              f"Falling back to random sampling.")
                    episode = self.sampler.sample_episode()

                # Load support images (support multiple supports per episode)
                support_data_list = []
                support_coords_list = []
                support_masks_list = []
                support_skeletons_list = []
                for support_idx in episode['support_indices']:
                    support_data = self.base_dataset[support_idx]
                    support_data_list.append(support_data)
                    support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)
                    h, w = support_data['height'], support_data['width']
                    support_coords[:, 0] /= w
                    support_coords[:, 1] /= h
                    # Clamp coordinates to [0, 1] after augmentation
                    support_coords = support_coords.clamp(0.0, 1.0)
                    if 'visibility' not in support_data:
                        raise KeyError(
                            f"Support data for image {support_data.get('image_id', 'unknown')} "
                            f"is missing 'visibility' field."
                        )
                    support_visibility = support_data['visibility']
                    if len(support_visibility) != len(support_coords):
                        raise ValueError(
                            f"Support visibility length ({len(support_visibility)}) doesn't match "
                            f"keypoints length ({len(support_coords)}) for image {support_data.get('image_id', 'unknown')}"
                        )
                    # Mask convention: True=ignore, False=use
                    support_mask = torch.tensor(
                        [v == 0 for v in support_visibility], 
                        dtype=torch.bool
                    )
                    support_skeleton = support_data.get('skeleton', [])
                    support_coords_list.append(support_coords)
                    support_masks_list.append(support_mask)
                    support_skeletons_list.append(support_skeleton)
                first_support = support_data_list[0]

                # Load query images
                query_images = []
                query_targets = []
                query_metadata = []
                for query_idx in episode['query_indices']:
                    query_data = self.base_dataset[query_idx]
                    query_images.append(query_data['image'])
                    query_targets.append(query_data['seq_data'])
                    if 'visibility' not in query_data:
                        raise KeyError(
                            f"Query data for image {query_data.get('image_id', 'unknown')} "
                            f"is missing 'visibility' field. This indicates mp100_cape.py "
                            f"did not set it properly. Available keys: {list(query_data.keys())}"
                        )
                    num_kpts = len(query_data['keypoints'])
                    visibility = query_data['visibility']
                    if len(visibility) != num_kpts:
                        raise ValueError(
                            f"Visibility length ({len(visibility)}) doesn't match "
                            f"keypoints length ({num_kpts}) for image {query_data.get('image_id', 'unknown')}. "
                            f"This indicates a bug in mp100_cape.py coordinate/visibility handling."
                        )
                    query_metadata.append({
                        'image_id': query_data['image_id'],
                        'height': query_data['height'],
                        'width': query_data['width'],
                        'keypoints': query_data['keypoints'],
                        'num_keypoints': query_data['num_keypoints'],
                        'num_visible_keypoints': query_data.get('num_visible_keypoints', query_data['num_keypoints']),
                        'bbox': query_data.get('bbox', [0, 0, query_data['width'], query_data['height']]),
                        'bbox_width': query_data.get('bbox_width', query_data['width']),
                        'bbox_height': query_data.get('bbox_height', query_data['height']),
                        'visibility': visibility
                    })
                    DEBUG_PCK = os.environ.get('DEBUG_PCK', '0') == '1'
                    if DEBUG_PCK and len(query_metadata) == 1:
                        bbox_w = query_metadata[0]['bbox_width']
                        bbox_h = query_metadata[0]['bbox_height']
                        if bbox_w == 512.0 and bbox_h == 512.0:
                            import warnings
                            warnings.warn(
                                f"Bbox dimensions are exactly 512x512 for image {query_data['image_id']}. "
                                "This may indicate preprocessed dims instead of original bbox size.",
                                RuntimeWarning
                            )
                        print(f"\n[DEBUG_PCK] Bbox metadata check:")
                        print(f"  Image ID: {query_data['image_id']}")
                        print(f"  Original bbox_width: {bbox_w}")
                        print(f"  Original bbox_height: {bbox_h}")
                
                # PERFORMANCE OPTIMIZATION: Skip support images if not needed
                # The model only uses support_coords and support_mask (geometry).
                # Support images are only needed for visualization (eval script).
                if self.load_support_images:
                    support_images = [sd['image'] for sd in support_data_list]
                else:
                    support_images = [None] * len(support_data_list)
                
                return {
                    'support_images': support_images,
                    'support_coords': support_coords_list,
                    'support_masks': support_masks_list,
                    'support_skeletons': support_skeletons_list,
                    'support_metadata': {
                        'image_id': first_support.get('image_id'),
                        'category_id': first_support.get('category_id'),
                        'bbox_width': first_support.get('bbox_width', first_support.get('width')),
                        'bbox_height': first_support.get('bbox_height', first_support.get('height')),
                    },
                    'query_images': query_images,
                    'query_targets': query_targets,
                    'query_metadata': query_metadata,
                    'category_id': episode['category_id']
                }
            except ImageNotFoundError as e:
                retry_count += 1
                if retry_count % 10 == 0:
                    print(f"Warning: Skipping episode due to missing image (attempt {retry_count})...")
                continue
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
    support_images_all = []
    support_coords_all = []
    support_masks_all = []
    support_skeletons_all = []
    support_metadata_list = []
    query_images_list = []
    query_targets_list = []
    query_metadata_list = []
    category_ids = []
    num_support_per_episode = len(batch[0]['support_images']) if len(batch) > 0 else 1
    for episode in batch:
        support_images_all.extend(episode['support_images'])
        support_coords_all.extend(episode['support_coords'])
        support_masks_all.extend(episode['support_masks'])
        support_skeletons_all.extend(episode['support_skeletons'])
        for _ in range(num_support_per_episode):
            support_metadata_list.append(episode.get('support_metadata', {}))
        query_images_list.extend(episode['query_images'])
        query_targets_list.extend(episode['query_targets'])
        query_metadata_list.extend(episode['query_metadata'])
        category_ids.append(episode['category_id'])
    
    # Stack support images (if they exist - may be None for training optimization)
    # When load_support_images=False, support_images are None to save I/O.
    if support_images_all and support_images_all[0] is not None:
        support_images = torch.stack(support_images_all)
    else:
        support_images = None
    
    # Pad support coordinates to max length in batch
    max_support_kpts = max(coords.shape[0] for coords in support_coords_all)
    support_coords_padded = []
    support_masks_padded = []
    for coords, mask in zip(support_coords_all, support_masks_all):
        num_kpts = coords.shape[0]
        if num_kpts < max_support_kpts:
            padding = max_support_kpts - num_kpts
            coords = torch.cat([coords, torch.zeros(padding, 2)], dim=0)
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)], dim=0)
        support_coords_padded.append(coords)
        support_masks_padded.append(mask)
    support_coords = torch.stack(support_coords_padded)
    support_masks = torch.stack(support_masks_padded)
    query_images = torch.stack(query_images_list)
    batched_seq_data = {}
    for key in query_targets_list[0].keys():
        batched_seq_data[key] = torch.stack([qt[key] for qt in query_targets_list])
    
    # ========================================================================
    # CRITICAL FIX: Align support and query dimensions for multi-shot learning
    # ========================================================================
    # Aggregate multiple supports per episode (mean for coords, any for masks)
    # Then repeat each aggregated support K times (once per query in that episode)
    # ========================================================================
    num_episodes = len(batch)
    num_total_queries = len(query_images_list)
    queries_per_episode = num_total_queries // num_episodes
    
    # Aggregate multiple supports: reshape to (B, num_support, max_kpts, 2)
    support_coords_reshaped = support_coords.view(num_episodes, num_support_per_episode, max_support_kpts, 2)
    support_coords_aggregated = support_coords_reshaped.mean(dim=1)  # (B, max_kpts, 2)
    support_masks_reshaped = support_masks.view(num_episodes, num_support_per_episode, max_support_kpts)
    support_masks_aggregated = support_masks_reshaped.any(dim=1)  # (B, max_kpts)
    
    # Handle support images (may be None if load_support_images=False)
    if support_images_all and support_images_all[0] is not None:
        support_images = torch.stack(support_images_all)
        support_images_reshaped = support_images.view(num_episodes, num_support_per_episode, *support_images.shape[1:])
        support_images_aggregated = support_images_reshaped[:, 0]  # Use first support image
        support_images = support_images_aggregated.repeat_interleave(queries_per_episode, dim=0)
    else:
        support_images = None
    
    # Repeat aggregated supports K times (once per query)
    support_coords = support_coords_aggregated.repeat_interleave(queries_per_episode, dim=0)
    support_masks = support_masks_aggregated.repeat_interleave(queries_per_episode, dim=0)
    
    # Aggregate skeletons (use first support's skeleton)
    support_skeletons_aggregated = []
    for i in range(num_episodes):
        support_skeletons_aggregated.append(support_skeletons_all[i * num_support_per_episode])
    
    # Repeat support metadata (list) K times
    support_metadata_repeated = []
    for i in range(num_episodes):
        meta = support_metadata_list[i * num_support_per_episode]
        support_metadata_repeated.extend([meta] * queries_per_episode)
    category_ids_tensor = torch.tensor(category_ids, dtype=torch.long)
    category_ids_tensor = category_ids_tensor.repeat_interleave(queries_per_episode)
    support_skeletons_repeated = []
    for skeleton in support_skeletons_aggregated:
        support_skeletons_repeated.extend([skeleton] * queries_per_episode)
    return {
        'support_images': support_images,
        'support_coords': support_coords,
        'support_masks': support_masks,
        'support_skeletons': support_skeletons_repeated,
        'support_metadata': support_metadata_repeated,
        'query_images': query_images,
        'query_targets': batched_seq_data,
        'query_metadata': query_metadata_list,
        'category_ids': category_ids_tensor
    }
def build_episodic_dataloader(base_dataset, category_split_file, split='train',
                              batch_size=2, num_queries_per_episode=2, num_support_per_episode=1,
                              episodes_per_epoch=1000, num_workers=16, seed=None,
                              fixed_episodes=False, load_support_images=True):
    """
    Build episodic dataloader for CAPE training/validation/testing.
    
    Args:
        base_dataset: MP100CAPE dataset
        category_split_file: Path to category splits JSON
        split: 'train', 'val', or 'test'
        batch_size: Number of episodes per batch
        num_queries_per_episode: Number of query images per episode
        num_support_per_episode: Number of support images per episode
        episodes_per_epoch: Total episodes per epoch
        num_workers: Number of worker processes
        seed: Random seed
        fixed_episodes: If True, pre-generate episodes once and reuse (for stable val curves)
        load_support_images: If True, load support images (needed for visualization).
                            If False, skip to save I/O, CPU, and GPU memory.
                            The model only uses support_coords, not images.
    Returns:
        dataloader: DataLoader with episodic sampling
    """
    episodic_dataset = EpisodicDataset(
        base_dataset=base_dataset,
        category_split_file=category_split_file,
        split=split,
        num_queries_per_episode=num_queries_per_episode,
        num_support_per_episode=num_support_per_episode,
        episodes_per_epoch=episodes_per_epoch,
        seed=seed,
        fixed_episodes=fixed_episodes,
        load_support_images=load_support_images
    )
    dataloader = data.DataLoader(
        episodic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=episodic_collate_fn,
        pin_memory=True
    )
    return dataloader