"""
Dataset loader for MP-100 in COCO format.
Handles loading, cropping, and normalization of keypoint annotations.
"""
import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.geometry import normalize_keypoints


class MP100Dataset(Dataset):
    """
    MP-100 dataset in COCO format.

    Loads images and keypoint annotations, handles cropping by bounding boxes,
    and provides normalized keypoints.
    """

    def __init__(
        self,
        annotation_file,
        image_root='data',
        image_size=512,
        augment=False
    ):
        """
        Args:
            annotation_file: path to COCO-format JSON annotation file
            image_root: root directory containing images
            image_size: target image size (will resize to image_size x image_size)
            augment: whether to apply data augmentation
        """
        self.image_root = image_root
        self.image_size = image_size
        self.augment = augment

        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data['categories']}

        # Load all annotations - file existence will be checked lazily during training
        self.annotations = coco_data['annotations']
        
        # Track missing files to avoid repeated checks
        self._missing_files = set()

        # Build category-to-instances mapping
        self.category_to_instances = {}
        for ann in self.annotations:
            cat_id = ann['category_id']
            if cat_id not in self.category_to_instances:
                self.category_to_instances[cat_id] = []
            self.category_to_instances[cat_id].append(ann)

        # Get list of all category IDs
        self.category_ids = sorted(list(self.category_to_instances.keys()))

        # Compute max keypoints across all categories
        self.max_keypoints = max(cat['keypoints'].__len__() for cat in self.categories.values())

        print(f"Loaded {len(self.annotations)} annotations")
        print(f"Categories: {len(self.category_ids)}")
        print(f"Max keypoints: {self.max_keypoints}")

        # Image transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Augmentation transforms
        if self.augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        else:
            self.color_jitter = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a single instance. If the file doesn't exist, tries to find a valid alternative.

        Returns:
            dict with keys:
                - image: torch.Tensor [3, H, W] - cropped and resized image
                - keypoints: np.array [N, 2] - normalized keypoints in [0, 1]^2
                - visibility: np.array [N] - visibility flags
                - category_id: int
                - num_keypoints: int - actual number of keypoints (before padding)
                - bbox: [x, y, w, h] - original bounding box
                - image_id: int
                - annotation_id: int
        """
        ann = self.annotations[idx]

        # Load image
        img_info = self.images[ann['image_id']]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        
        # Check if file exists - if not, try to find a valid alternative
        if not os.path.exists(img_path):
            # Mark this file as missing
            self._missing_files.add(ann['image_id'])
            
            # Try to find another valid item from the same category
            cat_id = ann['category_id']
            category_instances = self.category_to_instances.get(cat_id, [])
            
            # Try up to 10 random instances from the same category
            max_retries = min(10, len(category_instances))
            for _ in range(max_retries):
                alt_ann = random.choice(category_instances)
                alt_img_info = self.images[alt_ann['image_id']]
                alt_img_path = os.path.join(self.image_root, alt_img_info['file_name'])
                
                if os.path.exists(alt_img_path):
                    # Use this alternative annotation
                    ann = alt_ann
                    img_info = alt_img_info
                    img_path = alt_img_path
                    break
            else:
                # If no valid alternative found, try any random annotation
                # This is a fallback to avoid raising an error
                max_fallback_retries = 100
                for _ in range(max_fallback_retries):
                    fallback_idx = random.randint(0, len(self.annotations) - 1)
                    fallback_ann = self.annotations[fallback_idx]
                    fallback_img_info = self.images[fallback_ann['image_id']]
                    fallback_img_path = os.path.join(self.image_root, fallback_img_info['file_name'])
                    
                    if os.path.exists(fallback_img_path):
                        ann = fallback_ann
                        img_info = fallback_img_info
                        img_path = fallback_img_path
                        break
                else:
                    # Last resort: raise an error if we can't find any valid file
                    raise FileNotFoundError(
                        f"Could not find valid image file. Tried {max_retries} from category "
                        f"{cat_id} and {max_fallback_retries} random files. "
                        f"Original missing file: {img_path}"
                    )
        
        image = Image.open(img_path).convert('RGB')

        # Get bbox and keypoints
        bbox = ann['bbox']  # [x, y, w, h]
        keypoints_flat = np.array(ann['keypoints']).reshape(-1, 3)  # [N, 3] - (x, y, v)

        # Get category info
        cat_id = ann['category_id']
        category = self.categories[cat_id]
        num_keypoints = len(category['keypoints'])

        # Crop image by bbox
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure bbox is within image bounds
        img_w, img_h = image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Ensure minimum size
        w = max(1, w)
        h = max(1, h)

        # Ensure crop coordinates are valid
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)

        cropped = image.crop((x, y, x2, y2))

        # Resize to target size
        cropped = cropped.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Apply augmentation
        if self.augment:
            if self.color_jitter is not None:
                cropped = self.color_jitter(cropped)

            # Random horizontal flip
            if np.random.rand() > 0.5:
                cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
                # Flip keypoints
                keypoints_flat_copy = keypoints_flat.copy()
                keypoints_flat_copy[:, 0] = bbox[2] - (keypoints_flat[:, 0] - bbox[0])
                keypoints_flat = keypoints_flat_copy

        # Convert to tensor
        image_tensor = transforms.ToTensor()(cropped)
        image_tensor = self.normalize(image_tensor)

        # Normalize keypoints
        normalized_coords, visibility = normalize_keypoints(keypoints_flat, bbox)

        return {
            'image': image_tensor,
            'keypoints': normalized_coords,  # [N, 2]
            'visibility': visibility,  # [N]
            'category_id': cat_id,
            'num_keypoints': num_keypoints,
            'bbox': bbox,
            'image_id': ann['image_id'],
            'annotation_id': ann['id']
        }

    def get_instances_by_category(self, category_id):
        """
        Get all instances for a given category.

        Args:
            category_id: category ID

        Returns:
            list of annotation indices
        """
        instances = self.category_to_instances.get(category_id, [])
        return [self.annotations.index(ann) for ann in instances]

    def get_category_info(self, category_id):
        """
        Get category metadata.

        Args:
            category_id: category ID

        Returns:
            dict with category information
        """
        return self.categories[category_id]

    def get_skeleton(self, category_id):
        """
        Get skeleton (edge list) for a category.

        Args:
            category_id: category ID

        Returns:
            list of [src, dst] edges
        """
        return self.categories[category_id]['skeleton']


def collate_fn(batch):
    """
    Custom collate function for batching variable-length keypoints.

    Args:
        batch: list of dicts from __getitem__

    Returns:
        dict with batched tensors
    """
    # Find max keypoints in this batch
    max_kpts = max(item['num_keypoints'] for item in batch)

    # Stack images
    images = torch.stack([item['image'] for item in batch])

    # Pad keypoints and visibility
    keypoints_list = []
    visibility_list = []
    num_keypoints_list = []

    for item in batch:
        N = item['num_keypoints']
        # Pad to max_kpts
        padded_kpts = np.zeros((max_kpts, 2), dtype=np.float32)
        padded_vis = np.zeros(max_kpts, dtype=np.float32)

        padded_kpts[:N] = item['keypoints'][:N]
        padded_vis[:N] = item['visibility'][:N]

        keypoints_list.append(padded_kpts)
        visibility_list.append(padded_vis)
        num_keypoints_list.append(N)

    keypoints = torch.from_numpy(np.stack(keypoints_list))
    visibility = torch.from_numpy(np.stack(visibility_list))
    num_keypoints = torch.tensor(num_keypoints_list)

    return {
        'images': images,
        'keypoints': keypoints,
        'visibility': visibility,
        'num_keypoints': num_keypoints,
        'category_ids': [item['category_id'] for item in batch],
        'bboxes': [item['bbox'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'annotation_ids': [item['annotation_id'] for item in batch]
    }
