"""
Episodic Sampler for Category-Agnostic Pose Estimation (CAPE)

This module implements episodic training where each episode consists of:
1. A category sampled from training categories
2. A support example from that category (provides pose graph)
3. Query examples from the same category (to predict keypoints on)

This trains the model to use support pose graphs to generalize to new categories.
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

    Each episode:
    1. Sample a category c from training categories
    2. Sample N support images (provides pose graphs) - N-shot learning
    3. Sample K query images (predict keypoints using support graphs)
    """

    def __init__(self, dataset, category_split_file, split='train',
                 num_queries_per_episode=2, num_support_per_episode=1, seed=None):
        """
        Args:
            dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train', 'val', or 'test' (determines which categories to use)
            num_queries_per_episode: Number of query images per episode
            num_support_per_episode: Number of support images per episode (1-shot, 5-shot, etc.)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.split = split
        self.num_queries = num_queries_per_episode
        self.num_support = num_support_per_episode

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load category splits
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
        
        # Store original categories for diagnostics
        original_categories = self.categories.copy()

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
        min_examples = num_queries_per_episode + num_support_per_episode  # Need at least N support + K queries
        self.categories = [
            cat for cat in self.categories
            if len(self.category_to_indices[cat]) >= min_examples
        ]

        print(f"Valid categories (>={min_examples} examples): {len(self.categories)}")
        
        # Check if we have any valid categories
        if len(self.categories) == 0:
            # Collect diagnostic information
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
            episode: dict containing:
                - category_id: sampled category
                - support_indices: list of support image indices (N-shot)
                - query_indices: list of query image indices
        """
        # Sample a category
        category_id = random.choice(self.categories)

        # Get all image indices for this category
        indices = self.category_to_indices[category_id]

        # Sample support + query indices (without replacement)
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
                 num_queries_per_episode=2, num_support_per_episode=1, episodes_per_epoch=1000, seed=None,
                 debug_single_image=None, debug_single_image_category=None):
        """
        Args:
            base_dataset: MP100CAPE dataset instance
            category_split_file: Path to category_splits.json
            split: 'train', 'val', or 'test'
            num_queries_per_episode: Number of query images per episode
            num_support_per_episode: Number of support images per episode (1-shot, 5-shot, etc.)
            episodes_per_epoch: Number of episodes per training epoch
            seed: Random seed
            debug_single_image: Optional image index for single-image debug mode
            debug_single_image_category: Optional category ID for single-image debug mode
        """
        self.base_dataset = base_dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.num_support = num_support_per_episode
        
        # Single-image debug mode
        self.debug_single_image = debug_single_image
        self.debug_single_image_category = debug_single_image_category
        
        if self.debug_single_image is not None:
            print(f"⚠️  SINGLE IMAGE MODE: Using image index {self.debug_single_image} from category {self.debug_single_image_category}")
            print(f"   Same image will be used as both support and query (self-supervised)")
            # Don't create sampler for single-image mode
            self.sampler = None
        else:
            # Create episodic sampler
            self.sampler = EpisodicSampler(
                base_dataset,
                category_split_file,
                split=split,
                num_queries_per_episode=num_queries_per_episode,
                num_support_per_episode=num_support_per_episode,
                seed=seed
            )

        print(f"EpisodicDataset: {episodes_per_epoch} episodes/epoch, "
              f"{num_support_per_episode}-shot ({num_support_per_episode} support), "
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
        # ========================================================================
        # SINGLE IMAGE MODE: Use the same image as both support and query
        # ========================================================================
        if self.debug_single_image is not None:
            # Load the single image (with error handling)
            try:
                image_data = self.base_dataset[self.debug_single_image]
            except ImageNotFoundError as e:
                # If image doesn't exist, raise a clearer error
                raise ImageNotFoundError(
                    f"Single image mode: Image at index {self.debug_single_image} not found. "
                    f"Original error: {e}. "
                    f"Please check that the image file exists in the data directory."
                )
            
            # Use same image for both support and query (self-supervised)
            support_data = image_data
            query_data = image_data
            
            # Extract support pose graph
            support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)
            h, w = support_data['height'], support_data['width']
            support_coords[:, 0] /= w
            support_coords[:, 1] /= h
            
            # Create support mask
            support_visibility = support_data['visibility']
            support_mask = torch.tensor(
                [v > 0 for v in support_visibility], 
                dtype=torch.bool
            )
            support_skeleton = support_data.get('skeleton', [])
            
            # Extract query targets (same as support for self-supervised)
            query_targets = query_data['seq_data']
            
            # Return episode with single query (same image)
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
        
        # ========================================================================
        # NORMAL MODE: Sample episodes from dataset
        # ========================================================================
        # Keep trying until we find a valid episode (skip missing images)
        max_retries = 100  # High limit to prevent infinite loops, but should rarely be hit
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Sample episode
                episode = self.sampler.sample_episode()

                # Load support images (N-shot: multiple support images)
                support_data_list = []
                support_coords_list = []
                support_masks_list = []
                support_skeletons_list = []
                
                for support_idx in episode['support_indices']:
                    support_data = self.base_dataset[support_idx]
                    support_data_list.append(support_data)
                    
                    # Extract support pose graph (normalized coordinates)
                    support_coords = torch.tensor(support_data['keypoints'], dtype=torch.float32)

                    # ================================================================
                    # Normalize support coordinates to [0, 1]
                    # ================================================================
                    # Pipeline of transformations (from mp100_cape.py):
                    #   1. Keypoints made bbox-relative (subtract bbox_x, bbox_y)
                    #   2. Image cropped to bbox and resized to 512x512
                    #   3. Keypoints scaled proportionally: kpt × (512 / bbox_dim)
                    #   4. Here: Normalize by 512 to get [0, 1]
                    #
                    # Mathematical equivalence:
                    #   (kpt × 512/bbox_dim) / 512 = kpt / bbox_dim
                    #
                    # Result: Keypoints are normalized relative to original bbox dimensions,
                    # which provides scale and translation invariance for N-shot learning.
                    # ================================================================
                    h, w = support_data['height'], support_data['width']  # Both are 512 after resize
                    support_coords[:, 0] /= w  # Normalize x to [0, 1]
                    support_coords[:, 1] /= h  # Normalize y to [0, 1]

                    # ================================================================
                    # CRITICAL FIX: Create support mask based on visibility
                    # ================================================================
                    # Previously: support_mask was all True (ignoring visibility)
                    # Now: Use visibility information to mark only visible keypoints as valid
                    #
                    # Visibility values (COCO format):
                    #   0 = not labeled (keypoint outside image or not annotated)
                    #   1 = labeled but not visible (occluded)
                    #   2 = labeled and visible
                    #
                    # For support mask:
                    #   - True (valid) if visibility > 0 (labeled, may be occluded)
                    #   - False (invalid) if visibility == 0 (not labeled)
                    # ================================================================
                    
                    # Ensure visibility is present
                    if 'visibility' not in support_data:
                        raise KeyError(
                            f"Support data for image {support_data.get('image_id', 'unknown')} "
                            f"is missing 'visibility' field."
                        )
                    
                    support_visibility = support_data['visibility']
                    
                    # Verify length matches keypoints
                    if len(support_visibility) != len(support_coords):
                        raise ValueError(
                            f"Support visibility length ({len(support_visibility)}) doesn't match "
                            f"keypoints length ({len(support_coords)}) for image {support_data.get('image_id', 'unknown')}"
                        )
                    
                    support_mask = torch.tensor(
                        [v > 0 for v in support_visibility], 
                        dtype=torch.bool
                    )

                    # Extract skeleton edges for support pose graph
                    support_skeleton = support_data.get('skeleton', [])
                    
                    support_coords_list.append(support_coords)
                    support_masks_list.append(support_mask)
                    support_skeletons_list.append(support_skeleton)
                
                # For N-shot: Use first support for metadata (or aggregate if needed)
                first_support = support_data_list[0]

                # Load query images
                query_images = []
                query_targets = []
                query_metadata = []

                for query_idx in episode['query_indices']:
                    query_data = self.base_dataset[query_idx]
                    query_images.append(query_data['image'])
                    query_targets.append(query_data['seq_data'])
                    
                    # ================================================================
                    # CRITICAL: Ensure visibility is always present
                    # ================================================================
                    # The fallback should NEVER be used if the dataset is correct.
                    # If 'visibility' is missing, it indicates a bug in mp100_cape.py
                    # ================================================================
                    if 'visibility' not in query_data:
                        raise KeyError(
                            f"Query data for image {query_data.get('image_id', 'unknown')} "
                            f"is missing 'visibility' field. This indicates mp100_cape.py "
                            f"did not set it properly. Available keys: {list(query_data.keys())}"
                        )
                    
                    # Verify lengths match
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
                        'num_keypoints': query_data['num_keypoints'],  # Total keypoints (after fix)
                        'num_visible_keypoints': query_data.get('num_visible_keypoints', query_data['num_keypoints']),
                        'bbox': query_data.get('bbox', [0, 0, query_data['width'], query_data['height']]),
                        'bbox_width': query_data.get('bbox_width', query_data['width']),
                        'bbox_height': query_data.get('bbox_height', query_data['height']),
                        'visibility': visibility  # Guaranteed to be correct length
                    })
                    
                    # ================================================================
                    # DEBUG: Verify bbox dimensions are ORIGINAL, not preprocessed
                    # ================================================================
                    # Bbox dimensions should be the ORIGINAL bbox size from COCO annotations,
                    # NOT the preprocessed 512x512 size. These are used for PCK threshold.
                    # ================================================================
                    DEBUG_PCK = os.environ.get('DEBUG_PCK', '0') == '1'
                    if DEBUG_PCK and len(query_metadata) == 1:  # First query only
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
                    # ================================================================

                # Successfully loaded all images - return the episode
                # For N-shot: Return lists of support data
                support_images = [sd['image'] for sd in support_data_list]
                
                return {
                    'support_images': support_images,  # List of N support images
                    'support_coords': support_coords_list,  # List of N support coordinate tensors
                    'support_masks': support_masks_list,  # List of N support masks
                    'support_skeletons': support_skeletons_list,  # List of N skeleton edge lists
                    'support_metadata': {
                        'image_id': first_support['image_id'],
                        'category_id': first_support['category_id'],
                        'bbox_width': first_support.get('bbox_width', first_support['width']),
                        'bbox_height': first_support.get('bbox_height', first_support['height']),
                    },
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

    support_images_all = []  # List of all support images (B*N)
    support_coords_all = []  # List of all support coordinates (B*N)
    support_masks_all = []  # List of all support masks (B*N)
    support_skeletons_all = []  # List of all skeletons (B*N)
    support_metadata_list = []  # NEW: Collect support metadata for debugging
    query_images_list = []
    query_targets_list = []
    query_metadata_list = []  # NEW: Collect query metadata
    category_ids = []

    # Determine number of support images per episode (N)
    num_support_per_episode = len(batch[0]['support_images']) if len(batch) > 0 else 1

    for episode in batch:
        # For N-shot: Each episode has N support images
        support_images_all.extend(episode['support_images'])  # Add N support images
        support_coords_all.extend(episode['support_coords'])  # Add N coordinate tensors
        support_masks_all.extend(episode['support_masks'])  # Add N masks
        support_skeletons_all.extend(episode['support_skeletons'])  # Add N skeletons
        
        # Repeat metadata N times (once per support)
        for _ in range(num_support_per_episode):
            support_metadata_list.append(episode.get('support_metadata', {}))
        
        query_images_list.extend(episode['query_images'])
        query_targets_list.extend(episode['query_targets'])
        query_metadata_list.extend(episode['query_metadata'])
        category_ids.append(episode['category_id'])

    # Stack support images: (B*N, C, H, W) where N = num_support_per_episode
    support_images = torch.stack(support_images_all)  # (B*N, C, H, W)

    # Pad support coordinates to max length in batch
    max_support_kpts = max(coords.shape[0] for coords in support_coords_all)
    support_coords_padded = []
    support_masks_padded = []

    for coords, mask in zip(support_coords_all, support_masks_all):
        num_kpts = coords.shape[0]
        if num_kpts < max_support_kpts:
            # Pad
            padding = max_support_kpts - num_kpts
            coords = torch.cat([coords, torch.zeros(padding, 2)], dim=0)
            mask = torch.cat([mask, torch.zeros(padding, dtype=torch.bool)], dim=0)

        support_coords_padded.append(coords)
        support_masks_padded.append(mask)

    support_coords = torch.stack(support_coords_padded)  # (B*N, max_kpts, 2)
    support_masks = torch.stack(support_masks_padded)  # (B*N, max_kpts)

    # Stack query images
    query_images = torch.stack(query_images_list)  # (B*K, C, H, W)

    # Collate query targets (seq_data)
    batched_seq_data = {}
    for key in query_targets_list[0].keys():
        batched_seq_data[key] = torch.stack([qt[key] for qt in query_targets_list])

    # ========================================================================
    # CRITICAL FIX: Align support and query dimensions for N-shot learning
    # ========================================================================
    # Currently we have:
    #   - support_coords:  (B*N, max_kpts, 2)  where N = num_support_per_episode
    #   - query_images:    (B*K, C, H, W)      where K = queries per episode
    # 
    # For N-shot learning, we need to aggregate N support images per episode.
    # Strategy: Group supports by episode, then repeat aggregated support K times.
    #
    # After aggregation and alignment:
    #   - support_coords:  (B*K, max_kpts, 2)  - aggregated N supports, repeated K times
    #   - query_images:    (B*K, C, H, W)
    # Now index i of support matches index i of query ✓
    # ========================================================================
    
    # Determine how many queries per episode (K)
    num_episodes = len(batch)  # B
    num_total_queries = len(query_images_list)  # B*K
    queries_per_episode = num_total_queries // num_episodes  # K
    
    # For N-shot: Aggregate N support images per episode
    # Strategy: Mean pooling of support coordinates (simple aggregation)
    # Reshape: (B*N, max_kpts, 2) -> (B, N, max_kpts, 2) -> (B, max_kpts, 2) via mean
    support_coords_reshaped = support_coords.view(num_episodes, num_support_per_episode, max_support_kpts, 2)
    support_coords_aggregated = support_coords_reshaped.mean(dim=1)  # (B, max_kpts, 2) - mean over N supports
    
    support_masks_reshaped = support_masks.view(num_episodes, num_support_per_episode, max_support_kpts)
    # For masks: Use OR (any valid keypoint across N supports is valid)
    support_masks_aggregated = support_masks_reshaped.any(dim=1)  # (B, max_kpts) - OR over N supports
    
    # For images: Use first support image (or could average, but mean pooling features is better)
    support_images_reshaped = support_images.view(num_episodes, num_support_per_episode, *support_images.shape[1:])
    support_images_aggregated = support_images_reshaped[:, 0]  # (B, C, H, W) - use first support image
    
    # Repeat aggregated support K times (once per query in that episode)
    support_coords = support_coords_aggregated.repeat_interleave(queries_per_episode, dim=0)  # (B*K, max_kpts, 2)
    support_masks = support_masks_aggregated.repeat_interleave(queries_per_episode, dim=0)  # (B*K, max_kpts)
    support_images = support_images_aggregated.repeat_interleave(queries_per_episode, dim=0)  # (B*K, C, H, W)
    
    # Aggregate skeletons: Use first support's skeleton (or could merge, but first is simpler)
    support_skeletons_aggregated = []
    for i in range(num_episodes):
        # Use first support's skeleton for each episode
        support_skeletons_aggregated.append(support_skeletons_all[i * num_support_per_episode])
    
    # Repeat support metadata (list) K times
    support_metadata_repeated = []
    for i in range(num_episodes):
        # Use first support's metadata
        meta = support_metadata_list[i * num_support_per_episode]
        support_metadata_repeated.extend([meta] * queries_per_episode)  # Repeat K times
    
    # ========================================================================
    # ALREADY FIXED (Issue #11): Category IDs repeated per query
    # ========================================================================
    # Category IDs must match the (B*K) batch dimension, not just (B).
    # Each category ID is repeated K times (once per query in that episode).
    #
    # Example with B=2 episodes, K=3 queries per episode:
    #   Before: category_ids = [cat_A, cat_B]  (length 2)
    #   After:  category_ids = [cat_A, cat_A, cat_A, cat_B, cat_B, cat_B]  (length 6)
    #
    # This ensures category_ids[i] corresponds to query[i] in the batch.
    # ========================================================================
    
    # Repeat category_ids to match query dimension
    category_ids_tensor = torch.tensor(category_ids, dtype=torch.long)  # (B,)
    category_ids_tensor = category_ids_tensor.repeat_interleave(queries_per_episode)  # (B*K,)
    
    # Repeat skeleton edge lists (each aggregated skeleton repeated K times)
    support_skeletons_repeated = []
    for skeleton in support_skeletons_aggregated:
        support_skeletons_repeated.extend([skeleton] * queries_per_episode)
    # Now support_skeletons_repeated has B*K entries

    # ========================================================================
    # CRITICAL FIX: Include query_metadata for evaluation
    # ========================================================================
    # query_metadata contains essential information for PCK evaluation:
    #   - bbox_width, bbox_height: Original bbox dimensions for PCK normalization
    #   - visibility: Keypoint visibility flags for masking in evaluation
    #   - image_id, height, width: Additional metadata for debugging
    #
    # This was previously collected but not passed through, breaking evaluation.
    # Now it's properly included in the batch for use in engine_cape.py
    # ========================================================================

    return {
        'support_images': support_images,  # (B*K, C, H, W)
        'support_coords': support_coords,  # (B*K, max_kpts, 2)
        'support_masks': support_masks,    # (B*K, max_kpts)
        'support_skeletons': support_skeletons_repeated,  # List of B*K skeleton edge lists
        'support_metadata': support_metadata_repeated,  # List of B*K support metadata dicts (NEW for debugging)
        'query_images': query_images,      # (B*K, C, H, W)
        'query_targets': batched_seq_data, # Dict with tensors of shape (B*K, ...)
        'query_metadata': query_metadata_list,  # List of B*K metadata dicts
        'category_ids': category_ids_tensor  # (B*K,)
    }


def build_episodic_dataloader(base_dataset, category_split_file, split='train',
                              batch_size=2, num_queries_per_episode=2, num_support_per_episode=1,
                              episodes_per_epoch=1000, num_workers=16, seed=None,
                              debug_single_image=None, debug_single_image_category=None):
    """
    Build episodic dataloader for CAPE training/validation/testing.

    Args:
        base_dataset: MP100CAPE dataset
        category_split_file: Path to category splits JSON
        split: 'train', 'val', or 'test' (determines which categories to use)
        batch_size: Number of episodes per batch
        num_queries_per_episode: Number of query images per episode
        episodes_per_epoch: Total episodes per epoch
        num_workers: Number of worker processes
        seed: Random seed
        debug_single_image: Optional image index for single-image debug mode
        debug_single_image_category: Optional category ID for single-image debug mode

    Returns:
        dataloader: DataLoader with episodic sampling
    """
    # Create episodic dataset
    episodic_dataset = EpisodicDataset(
        base_dataset=base_dataset,
        category_split_file=category_split_file,
        split=split,
        num_queries_per_episode=num_queries_per_episode,
        num_support_per_episode=num_support_per_episode,
        episodes_per_epoch=episodes_per_epoch,
        seed=seed,
        debug_single_image=debug_single_image,
        debug_single_image_category=debug_single_image_category
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
