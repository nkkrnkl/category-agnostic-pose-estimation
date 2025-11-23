"""
MP-100 Dataset for Category-Agnostic Pose Estimation (CAPE)
Adapts Raster2Seq framework for pose estimation task
"""

from pathlib import Path
import torch
import torch.utils.data
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
from copy import deepcopy
import torchvision

# Google Cloud Storage support
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available. GCS bucket checks will be skipped.")

# Import transforms (with fallback if needed)
try:
    from datasets.transforms import Resize, ResizeAndPad
except ImportError:
    print("Warning: Using fallback transforms (detectron2 not available)")

    class Resize:
        def __init__(self, size):
            self.size = size if isinstance(size, tuple) else (size, size)

        def __call__(self, image):
            import cv2
            resized = cv2.resize(image, self.size[::-1])  # cv2 uses (W,H)
            return {'image': resized}

    class ResizeAndPad:
        def __init__(self, size):
            self.size = size if isinstance(size, tuple) else (size, size)

        def __call__(self, image):
            import cv2
            resized = cv2.resize(image, self.size[::-1])
            return {'image': resized}

    # Simplified tokenizer
    class DiscreteTokenizerV2:
        def __init__(self, vocab_size=2000, seq_len=200, **kwargs):
            self.vocab_size = vocab_size
            self.seq_len = seq_len

        def __len__(self):
            return self.vocab_size

        def convert_to_sequence(self, coords, add_eos=True, add_sep_between_polygons=False):
            # Simple quantization: scale coords to vocab bins
            bins = int(np.sqrt(self.vocab_size / 2))
            quantized = (coords * bins).astype(int).clip(0, bins-1)

            # Return dict with tokenized data
            seq_dict = {
                'seq11': quantized[:, 0],  # x coordinates
                'seq21': quantized[:, 0],  # duplicate for compatibility
                'seq12': quantized[:, 1],  # y coordinates
                'seq22': quantized[:, 1],  # duplicate for compatibility
                'delta_x1': np.ones_like(quantized[:, 0]) * 0.5,
                'delta_x2': np.ones_like(quantized[:, 0]) * 0.5,
                'delta_y1': np.ones_like(quantized[:, 1]) * 0.5,
                'delta_y2': np.ones_like(quantized[:, 1]) * 0.5,
            }
            return seq_dict

# Always import the real tokenizer (it's always available)
from datasets.discrete_tokenizer import DiscreteTokenizer, DiscreteTokenizerV2


def _check_gcs_object_exists(bucket_name, blob_name):
    """
    Check if an object exists in a GCS bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the object in the bucket (e.g., 'data/wren_body/0001.jpg')
    
    Returns:
        bool: True if object exists, False otherwise
    """
    if not GCS_AVAILABLE:
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        # If GCS check fails (e.g., authentication issues), return False
        # This allows the code to continue with local-only checks
        return False


def _check_image_exists(local_path, gcs_bucket_name=None, gcs_blob_path=None):
    """
    Check if an image exists locally or in GCS bucket.
    
    Args:
        local_path: Local file path
        gcs_bucket_name: Optional GCS bucket name
        gcs_blob_path: Optional GCS blob path (relative to bucket root)
    
    Returns:
        bool: True if image exists locally or in GCS, False otherwise
    """
    # First check local file system
    if os.path.exists(local_path):
        return True
    
    # If not found locally, check GCS if configured
    if gcs_bucket_name and gcs_blob_path:
        return _check_gcs_object_exists(gcs_bucket_name, gcs_blob_path)
    
    return False


class MP100CAPE(torch.utils.data.Dataset):
    """
    MP-100 Dataset for Category-Agnostic Pose Estimation

    Following the project proposal, this dataset:
    1. Takes a query image
    2. Takes a pose graph (keypoint sequence) as support data
    3. Predicts keypoints on the query image

    Key difference from CapeX: Uses 2D coordinate sequences directly
    instead of textual descriptions
    """

    def __init__(self, img_folder, ann_file, transforms, semantic_classes=-1,
                 dataset_name='mp100', image_norm=False, poly2seq=True,
                 converter_version='v3', split='train', gcs_bucket_name=None, **kwargs):
        super(MP100CAPE, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.dataset_name = dataset_name
        self.split = split
        # Set GCS bucket name, but only if not empty string
        gcs_bucket = gcs_bucket_name or kwargs.get('gcs_bucket_name', None)
        self.gcs_bucket_name = gcs_bucket if gcs_bucket else None

        # Load COCO format annotations
        self.coco = COCO(ann_file)
        # Keep all image IDs - filtering will happen during training in __getitem__
        self.ids = list(sorted(self.coco.imgs.keys()))

        # For CAPE, we always use poly2seq (keypoint sequence representation)
        self.poly2seq = poly2seq

        # Image normalization
        if image_norm:
            self.image_normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.image_normalize = None

        # Initialize tokenizer for keypoint coordinates
        if poly2seq:
            # Convert vocab_size to num_bins for DiscreteTokenizer
            # vocab_size = num_bins * num_bins (approx, ignoring special tokens)
            vocab_size = kwargs.get('vocab_size', 2000)
            seq_len = kwargs.get('seq_len', 200)
            num_bins = int(np.sqrt(vocab_size))
            self.tokenizer = DiscreteTokenizerV2(num_bins=num_bins, seq_len=seq_len, add_cls=False)

        print(f"Loaded MP-100 {split} dataset: {len(self.ids)} images (missing images will be skipped during training)")

    def get_vocab_size(self):
        if self.poly2seq:
            return len(self.tokenizer)
        return None

    def get_tokenizer(self):
        if self.poly2seq:
            return self.tokenizer
        return None

    def __len__(self):
        return len(self.ids)

    def _expand_image_dims(self, x):
        """Expand image dimensions to (C, H, W)"""
        if len(x.shape) == 2:
            exp_img = np.expand_dims(x, 0)
        else:
            exp_img = x.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        return exp_img

    def __getitem__(self, index):
        """
        Returns:
            dict with keys:
                - image: query image tensor
                - keypoints: list of keypoint coordinates [[x1,y1], [x2,y2], ...]
                - category_id: object category
                - num_keypoints: number of keypoints
                - image_id: image identifier
                - file_name: image path
                - seq_data: tokenized keypoint sequence
        """
        coco = self.coco
        img_id = self.ids[index]

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        file_name = os.path.join(self.root, path)

        # Construct GCS blob path if bucket is configured
        gcs_blob_path = None
        if self.gcs_bucket_name:
            # Remove 'data/' prefix if present
            clean_path = path
            if clean_path.startswith('data/'):
                clean_path = clean_path[5:]
            gcs_blob_path = clean_path

        # Check if file exists locally or in GCS bucket
        if not _check_image_exists(file_name, self.gcs_bucket_name, gcs_blob_path):
            location_info = f" (local: {file_name}"
            if self.gcs_bucket_name:
                location_info += f", GCS: gs://{self.gcs_bucket_name}/{gcs_blob_path or path}"
            location_info += ")"
            print(f"Warning: Image file not found at index {index}{location_info}")
            # Return None to be filtered out by collator
            return None
        
        try:
            img = np.array(Image.open(file_name).convert('RGB'))
        except (IOError, OSError, FileNotFoundError) as e:
            # If local file fails and GCS is available, try downloading from GCS
            if self.gcs_bucket_name and gcs_blob_path and not os.path.exists(file_name):
                try:
                    if GCS_AVAILABLE:
                        client = storage.Client()
                        bucket = client.bucket(self.gcs_bucket_name)
                        blob = bucket.blob(gcs_blob_path)
                        
                        # Download to local path
                        os.makedirs(os.path.dirname(file_name), exist_ok=True)
                        blob.download_to_filename(file_name)
                        
                        # Now try loading again
                        img = np.array(Image.open(file_name).convert('RGB'))
                    else:
                        raise e
                except Exception as gcs_error:
                    print(f"Warning: Failed to load/download image at index {index}: {file_name}, error: {e}, GCS error: {gcs_error}")
                    return None
            else:
                print(f"Warning: Failed to load image at index {index}: {file_name}, error: {e}")
                # Return None to be filtered out by collator
                return None

        # Handle image dimensions
        if len(img.shape) >= 3:
            if img.shape[-1] > 3:  # drop alpha channel
                img = img[:, :, :3]
            h, w = img.shape[:2]
        else:
            h, w = img.shape

        # Build record
        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record["image_id"] = img_id

        # Process keypoints from annotations
        keypoints_list = []
        category_ids = []
        num_keypoints_list = []

        for ann in target:
            if 'keypoints' in ann and ann['keypoints']:
                # COCO keypoints format: [x1,y1,v1,x2,y2,v2,...]
                # where v is visibility (0=not labeled, 1=labeled but not visible, 2=labeled and visible)
                kpts = np.array(ann['keypoints']).reshape(-1, 3)

                # Filter visible keypoints (v > 0)
                visible_kpts = kpts[kpts[:, 2] > 0][:, :2]  # only take x, y

                if len(visible_kpts) > 0:
                    keypoints_list.append(visible_kpts.tolist())
                    category_ids.append(ann.get('category_id', 0))
                    num_keypoints_list.append(len(visible_kpts))

        # For now, take the first annotation (can be extended for multi-object)
        if len(keypoints_list) > 0:
            record["keypoints"] = keypoints_list[0]
            record["category_id"] = category_ids[0]
            record["num_keypoints"] = num_keypoints_list[0]
        else:
            # Empty annotation - use dummy values
            record["keypoints"] = [[0.0, 0.0]]
            record["category_id"] = 0
            record["num_keypoints"] = 1

        # Apply transforms
        if self._transforms is not None:
            img, record = self._apply_transforms(img, record)

        # Convert image to tensor
        img_tensor = torch.as_tensor(self._expand_image_dims(img))
        if self.image_normalize:
            img_tensor = self.image_normalize(img_tensor.float() / 255.0)
        else:
            img_tensor = img_tensor.float() / 255.0

        record["image"] = img_tensor

        # Tokenize keypoint sequence for autoregressive generation
        if self.poly2seq:
            # Store category_id temporarily for use in _tokenize_keypoints
            self._current_category_id = record["category_id"]
            record["seq_data"] = self._tokenize_keypoints(record["keypoints"],
                                                          record["height"],
                                                          record["width"])
            # Clean up temporary attribute
            delattr(self, '_current_category_id')

        return record

    def _apply_transforms(self, img, record):
        """Apply image transformations and update keypoint coordinates"""
        if self._transforms is None:
            return img, record

        # Get original dimensions
        orig_h, orig_w = record["height"], record["width"]

        # Apply transform
        transformed = self._transforms(image=img)
        img = transformed['image']

        # Update dimensions
        new_h, new_w = img.shape[:2]
        record["height"] = new_h
        record["width"] = new_w

        # Scale keypoints
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        scaled_keypoints = []
        for kpt in record["keypoints"]:
            x, y = kpt
            scaled_keypoints.append([x * scale_x, y * scale_y])

        record["keypoints"] = scaled_keypoints

        return img, record

    def _tokenize_keypoints(self, keypoints, height, width):
        """
        Tokenize keypoint coordinates using bilinear interpolation
        Following poly_data.py pattern for compatibility with Raster2Seq

        Args:
            keypoints: List of [x, y] coordinates
            height, width: Image dimensions for normalization

        Returns:
            dict with tokenized sequence data
        """
        import math
        from enum import Enum

        # Token types: 0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>
        class TokenType(Enum):
            coord = 0
            sep = 1
            eos = 2
            cls = 3

        # Normalize keypoints to [0, 1]
        normalized_kpts = []
        for x, y in keypoints:
            normalized_kpts.append([x / width, y / height])

        # For CAPE, treat all keypoints as one "polygon" (one continuous sequence)
        polygons = [np.array(normalized_kpts)]

        # Quantize coordinates using bilinear interpolation
        num_bins = self.tokenizer.num_bins
        quant_poly = [poly * (num_bins - 1) for poly in polygons]

        # 4 indices for bilinear interpolation (floor/ceil combinations)
        index11 = [[math.floor(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index21 = [[math.ceil(p[0])*num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index12 = [[math.floor(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]
        index22 = [[math.ceil(p[0])*num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]

        # Tokenize each index sequence
        seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
        seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
        seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
        seq22 = self.tokenizer(index22, add_bos=True, add_eos=False, dtype=torch.long)

        # Build target sequence and token labels
        target_seq = []
        token_labels = []  # 0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>
        add_cls_token = False  # For CAPE, we don't use CLS tokens

        for poly in polygons:
            token_labels.extend([TokenType.coord.value] * len(poly))
            target_seq.extend(poly)
            if add_cls_token:
                token_labels.append(TokenType.cls.value)
                target_seq.append([0, 0])
            token_labels.append(TokenType.sep.value)
            target_seq.append([0, 0])  # EOS padding

        # Change last separator to EOS
        if len(token_labels) > 0:
            token_labels[-1] = TokenType.eos.value

        # Create mask
        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        if len(token_labels) < self.tokenizer.seq_len:
            mask[len(token_labels):] = 0

        # Pad sequences
        target_seq = self.tokenizer._padding(target_seq, [0, 0], dtype=torch.float32)
        token_labels = self.tokenizer._padding(token_labels, -1, dtype=torch.long)

        # Compute deltas for bilinear interpolation
        delta_x1 = [0]  # BOS token
        delta_y1 = [0]
        for polygon in quant_poly:
            delta_x = [p[0] - math.floor(p[0]) for p in polygon]
            delta_y = [p[1] - math.floor(p[1]) for p in polygon]
            delta_x1.extend(delta_x)
            delta_y1.extend(delta_y)
            delta_x1.append(0)  # EOS/SEP
            delta_y1.append(0)

        delta_x1 = delta_x1[:-1]  # Remove last EOS
        delta_y1 = delta_y1[:-1]
        delta_x1 = self.tokenizer._padding(delta_x1, 0, dtype=torch.float32)
        delta_y1 = self.tokenizer._padding(delta_y1, 0, dtype=torch.float32)
        delta_x2 = 1 - delta_x1
        delta_y2 = 1 - delta_y1

        # For CAPE, the polygon label is the category_id
        # This will be set in __getitem__
        target_polygon_labels = torch.full((self.tokenizer.seq_len,), -1, dtype=torch.long)
        # Set the category for all coordinate tokens
        category_id = getattr(self, '_current_category_id', 0)
        for i in range(len(keypoints)):
            if i < self.tokenizer.seq_len:
                target_polygon_labels[i] = category_id

        return {
            'seq11': seq11,
            'seq21': seq21,
            'seq12': seq12,
            'seq22': seq22,
            'target_seq': target_seq,
            'token_labels': token_labels,
            'mask': mask,
            'target_polygon_labels': target_polygon_labels,
            'delta_x1': delta_x1,
            'delta_x2': delta_x2,
            'delta_y1': delta_y1,
            'delta_y2': delta_y2,
        }


def build_mp100_cape(image_set, args):
    """
    Build MP-100 CAPE dataset

    Args:
        image_set: 'train', 'val', or 'test'
        args: argparse arguments
    """
    # Determine which split to use (1-5)
    split_num = getattr(args, 'mp100_split', 1)

    # Paths
    # data folder is in dataset_root/data
    root = Path(args.dataset_root) / "data"
    # annotations are in dataset_root/annotations
    # Use resolve() to convert relative paths to absolute first
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"mp100_split{split_num}_{image_set}.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # Build transforms
    if image_set == 'train':
        transforms = Resize((256, 256))  # Simple resize for now
    else:
        transforms = Resize((256, 256))

    # Get GCS bucket name from args if provided
    gcs_bucket_name = getattr(args, 'gcs_bucket_name', None)
    # If empty string is provided, disable GCS checks
    if gcs_bucket_name == '':
        gcs_bucket_name = None
    # Default bucket name if not specified (and not explicitly disabled)
    elif gcs_bucket_name is None:
        gcs_bucket_name = 'dl-category-agnostic-pose-mp100-data'  # Default from sync script
    
    dataset = MP100CAPE(
        img_folder=str(root),
        ann_file=str(ann_file),
        transforms=transforms,
        semantic_classes=args.semantic_classes,
        dataset_name='mp100',
        image_norm=args.image_norm,
        poly2seq=True,
        converter_version='v3',
        split=image_set,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        gcs_bucket_name=gcs_bucket_name
    )

    return dataset
