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
        """Custom transform that resizes (preserving aspect ratio) and then pads to fixed size"""
        def __init__(self, target_size, pad_value=0, interp=None):
            """
            Args:
                target_size: (height, width) tuple
                pad_value: value to use for padding
                interp: interpolation method (unused in fallback)
            """
            self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
            self.pad_value = pad_value

        def __call__(self, image):
            """
            Args:
                image: numpy array (H, W, C)
            Returns:
                dict with 'image' key containing transformed image
            """
            import cv2
            if isinstance(image, Image.Image):
                image = np.array(image)

            h, w = image.shape[:2]
            target_h, target_w = self.target_size

            # Calculate scale preserving aspect ratio
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize preserving aspect ratio
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Create padded image
            padded = np.full((target_h, target_w, image.shape[2]), self.pad_value, dtype=image.dtype)

            # Calculate padding offsets (center the image)
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2

            # Place resized image in center
            padded[top:top + new_h, left:left + new_w] = resized

            return {'image': padded}

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
                 converter_version='v3', split='train', **kwargs):
        super(MP100CAPE, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.dataset_name = dataset_name
        self.split = split

        # Load COCO format annotations
        self.coco = COCO(ann_file)
        all_ids = list(sorted(self.coco.imgs.keys()))

        # Pre-filter: check which image files actually exist
        # NOTE: This can be slow for large datasets, especially with GCS mounts
        # Set skip_missing_files=False to disable and handle missing files at runtime
        self.skip_missing = kwargs.get('skip_missing_files', False)
        root_abs = os.path.abspath(os.path.expanduser(self.root))
        
        # Detect if bucket name is a prefix in the mount structure (quick check)
        # Check if bucket name directory exists in root or parent
        bucket_name = "dl-category-agnostic-pose-mp100-data"
        self.bucket_prefix = None
        if os.path.exists(os.path.join(root_abs, bucket_name)):
            self.bucket_prefix = bucket_name
            print(f"Detected bucket name prefix in mount: {bucket_name}")
        elif os.path.exists(os.path.join(os.path.dirname(root_abs), bucket_name)):
            self.bucket_prefix = bucket_name
            print(f"Detected bucket name prefix in parent directory: {bucket_name}")
        
        if self.skip_missing:
            # WARNING: This can be slow for large datasets (10k+ images)
            # Each os.path.exists() on GCS mount has network latency
            print(f"Checking file existence for {len(all_ids)} images...")
            print("  (This may take a few minutes for large datasets)")
            valid_ids = []
            missing_count = 0
            checked = 0
            
            # Sample a few files first to detect pattern
            sample_size = min(10, len(all_ids))
            sample_ids = all_ids[:sample_size]
            bucket_prefix_detected = False
            
            for img_id in sample_ids:
                img_info = self.coco.loadImgs(img_id)[0]
                path = img_info['file_name']
                file_name = os.path.join(root_abs, path)
                
                if not os.path.exists(file_name) and self.bucket_prefix:
                    file_name = os.path.join(root_abs, self.bucket_prefix, path)
                    if os.path.exists(file_name):
                        bucket_prefix_detected = True
                
                if os.path.exists(file_name):
                    valid_ids.append(img_id)
                else:
                    missing_count += 1
            
            # If bucket prefix works in sample, use it for all
            if bucket_prefix_detected and not self.bucket_prefix:
                self.bucket_prefix = bucket_name
                print(f"  Detected bucket prefix pattern from sample, using for all files")
            
            # Now check remaining files (with progress for large datasets)
            for img_id in all_ids[sample_size:]:
                img_info = self.coco.loadImgs(img_id)[0]
                path = img_info['file_name']
                file_name = os.path.join(root_abs, path)
                
                # Try with bucket prefix if detected
                if not os.path.exists(file_name) and self.bucket_prefix:
                    file_name = os.path.join(root_abs, self.bucket_prefix, path)
                
                if os.path.exists(file_name):
                    valid_ids.append(img_id)
                else:
                    missing_count += 1
                    if missing_count <= 5:  # Print first 5 missing files
                        print(f"  Warning: Missing file: {path}")
                
                checked += 1
                if checked % 1000 == 0:
                    print(f"  Checked {checked}/{len(all_ids) - sample_size} files...")
            
            self.ids = valid_ids
            if missing_count > 0:
                print(f"  Skipped {missing_count} missing files. Using {len(valid_ids)} valid images.")
        else:
            self.ids = all_ids
            print("  (Skipping file existence check - missing files will be handled at runtime)")

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

        print(f"Loaded MP-100 {split} dataset: {len(self.ids)} images")

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
        
        # Make root path absolute to avoid relative path issues
        root_abs = os.path.abspath(os.path.expanduser(self.root))
        file_name = os.path.join(root_abs, path)
        
        # Handle path mismatches: if file doesn't exist at annotated path, try to find it
        # by extracting just the filename and searching in the category directory
        if not os.path.exists(file_name):
            # Extract category and filename
            path_parts = path.split('/')
            if len(path_parts) >= 2:
                category = path_parts[0]  # e.g., 'human_face'
                filename = path_parts[-1]  # e.g., '2436720309_1.jpg'
                
                # Try multiple fallback strategies in order of likelihood
                fallback_paths = []
                
                # Strategy 1: Try with bucket name prefix (ONLY if detected during init)
                # Some mounts might have: dl-category-agnostic-pose-mp100-data/amur_tiger_body/...
                # Only try this if bucket_prefix was actually detected, otherwise skip
                if getattr(self, 'bucket_prefix', None):
                    bucket_prefix_path = os.path.join(root_abs, self.bucket_prefix, path)
                    fallback_paths.append(bucket_prefix_path)
                
                # Strategy 2: Try without intermediate subdirectories (e.g., skip 'flickr/0/')
                # Original: amur_tiger_body/flickr/0/image04686.jpg
                # Try: amur_tiger_body/image04686.jpg
                if len(path_parts) > 2:
                    fallback_paths.append(os.path.join(root_abs, category, filename))
                    # Also try with bucket prefix (only if detected)
                    if getattr(self, 'bucket_prefix', None):
                        fallback_paths.append(os.path.join(root_abs, self.bucket_prefix, category, filename))
                
                # Strategy 3: Direct path in category folder (already tried above if len > 2)
                if len(path_parts) == 2:
                    fallback_paths.append(os.path.join(root_abs, category, filename))
                    # Also try with bucket prefix (only if detected)
                    if getattr(self, 'bucket_prefix', None):
                        fallback_paths.append(os.path.join(root_abs, self.bucket_prefix, category, filename))
                
                # Strategy 4: Try at root level (unlikely but possible)
                fallback_paths.append(os.path.join(root_abs, filename))
                # Also try with bucket prefix (only if detected)
                if getattr(self, 'bucket_prefix', None):
                    fallback_paths.append(os.path.join(root_abs, self.bucket_prefix, filename))
                
                # Strategy 5: Try with bucket prefix and full path structure (only if detected)
                # Check if parent of root_abs might be the actual mount point
                if getattr(self, 'bucket_prefix', None):
                    parent_dir = os.path.dirname(root_abs)
                    if os.path.basename(root_abs) == "data":
                        # If root is "data", check if bucket is mounted at parent level
                        bucket_mount_path = os.path.join(parent_dir, self.bucket_prefix, path)
                        fallback_paths.append(bucket_mount_path)
                        # Also try without the "data" part
                        fallback_paths.append(os.path.join(parent_dir, self.bucket_prefix, category, *path_parts[1:]))
                
                # Try fallback paths (in order of likelihood)
                # NOTE: We try the most likely paths first to minimize os.path.exists() calls
                # which are slow on GCS mounts
                for alt_path in fallback_paths:
                    if alt_path and os.path.exists(alt_path):
                        file_name = alt_path
                        break
                else:
                    # Last resort: limited recursive search (SLOW - avoid if possible)
                    # Only search if we haven't found the file yet
                    # Limit depth to avoid very slow searches on GCS mounts
                    search_dirs = [os.path.join(root_abs, category)]
                    if getattr(self, 'bucket_prefix', None):
                        bucket_dir = os.path.join(root_abs, self.bucket_prefix, category)
                        if os.path.exists(os.path.join(root_abs, self.bucket_prefix)):
                            search_dirs.append(bucket_dir)
                    search_dirs = [d for d in search_dirs if d and os.path.isdir(d)]
                    
                    # Only do recursive search if really necessary (it's very slow on GCS)
                    # Limit to max_depth=2 to avoid deep recursion
                    max_depth = 2
                    for category_dir in search_dirs:
                        try:
                            depth = 0
                            for root_dir, dirs, files in os.walk(category_dir):
                                # Limit search depth
                                rel_depth = root_dir[len(category_dir):].count(os.sep)
                                if rel_depth > max_depth:
                                    dirs[:] = []  # Don't recurse deeper
                                    continue
                                
                                if filename in files:
                                    file_name = os.path.join(root_dir, filename)
                                    break
                            if os.path.exists(file_name):
                                break
                        except (OSError, PermissionError) as e:
                            # If we can't walk the directory (e.g., permission issues with GCS mount)
                            pass
        
        # Final check: if file still doesn't exist, provide detailed debugging info
        if not os.path.exists(file_name):
            # Try to get more info about what exists
            path_parts = path.split('/')
            category = path_parts[0] if len(path_parts) >= 1 else 'unknown'
            category_dir = os.path.join(root_abs, category)
            
            # Verify the image ID actually exists in annotations
            annotation_check = ""
            try:
                if img_id in self.coco.imgs:
                    img_info_check = self.coco.loadImgs(img_id)[0]
                    annotation_check = f"  ✅ Image ID {img_id} exists in annotations\n    Annotation path: {img_info_check.get('file_name')}\n"
                else:
                    annotation_check = f"  ❌ Image ID {img_id} NOT found in annotations!\n    This suggests a data corruption issue.\n"
            except:
                annotation_check = "  ⚠️  Could not verify annotation (error checking)\n"
            
            # Check what's actually in the category directory
            dir_contents = []
            subdir_contents = {}
            try:
                if os.path.isdir(category_dir):
                    dir_contents = os.listdir(category_dir)[:10]  # First 10 items
                    # If 'flickr' is in the directory, check what's inside
                    if 'flickr' in dir_contents:
                        flickr_dir = os.path.join(category_dir, 'flickr')
                        if os.path.isdir(flickr_dir):
                            flickr_subdirs = os.listdir(flickr_dir)[:5]
                            subdir_contents['flickr'] = flickr_subdirs
            except (OSError, PermissionError):
                dir_contents = ["(cannot list directory)"]
            
            # Build detailed error message
            error_msg = (
                f"Image file not found: {file_name}\n"
                f"{annotation_check}"
                f"  Root directory: {root_abs}\n"
                f"  Path from annotation: {path}\n"
                f"  Image ID: {img_id}\n"
                f"  Category directory: {category_dir}\n"
                f"  Category directory exists: {os.path.isdir(category_dir)}\n"
                f"  Category directory contents (first 10): {dir_contents}\n"
            )
            
            if subdir_contents:
                error_msg += f"  Subdirectory contents:\n"
                for subdir, contents in subdir_contents.items():
                    error_msg += f"    {subdir}/: {contents}\n"
            
            error_msg += (
                f"  Please check:\n"
                f"    1. GCS bucket is mounted correctly\n"
                f"    2. Data symlink exists: {os.path.exists(os.path.join(os.path.dirname(root_abs), 'data'))}\n"
                f"    3. File might be missing from the dataset\n"
                f"    4. Try: ls -la {category_dir} to see what files exist\n"
                f"    5. Run: python check_annotation.py {img_id} to verify annotation"
            )
            
            # For now, raise the error (can be changed to skip if needed)
            raise FileNotFoundError(error_msg)

        img = np.array(Image.open(file_name).convert('RGB'))

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

        # Check if transform is ResizeAndPad (preserves aspect ratio with padding)
        is_resize_and_pad = isinstance(self._transforms, ResizeAndPad)
        
        if is_resize_and_pad:
            # ResizeAndPad: calculate scale and padding offsets
            target_h, target_w = self._transforms.target_size
            scale = min(target_h / orig_h, target_w / orig_w)
            new_resized_h, new_resized_w = int(orig_h * scale), int(orig_w * scale)
            
            # Calculate padding offsets (centered)
            pad_top = (target_h - new_resized_h) // 2
            pad_left = (target_w - new_resized_w) // 2
            
            # Scale keypoints and add padding offset
            scaled_keypoints = []
            for kpt in record["keypoints"]:
                x, y = kpt
                # Scale first
                x_scaled = x * scale
                y_scaled = y * scale
                # Add padding offset
                x_final = x_scaled + pad_left
                y_final = y_scaled + pad_top
                scaled_keypoints.append([x_final, y_final])
        else:
            # Simple Resize: just scale (may distort aspect ratio)
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

        # Clamp quantized values to [0, num_bins - 1] to prevent out-of-bounds indices
        # This ensures floor/ceil operations don't exceed valid vocabulary range
        quant_poly = [np.clip(poly, 0, num_bins - 1) for poly in quant_poly]

        # 4 indices for bilinear interpolation (floor/ceil combinations)
        # Clamp indices to valid range [0, num_bins * num_bins - 1]
        def safe_index(x, y, use_ceil_x, use_ceil_y):
            """Compute index with bounds checking"""
            x_idx = math.ceil(x) if use_ceil_x else math.floor(x)
            y_idx = math.ceil(y) if use_ceil_y else math.floor(y)
            # Clamp to valid range
            x_idx = max(0, min(int(x_idx), num_bins - 1))
            y_idx = max(0, min(int(y_idx), num_bins - 1))
            return x_idx * num_bins + y_idx
        
        index11 = [[safe_index(p[0], p[1], False, False) for p in poly] for poly in quant_poly]
        index21 = [[safe_index(p[0], p[1], True, False) for p in poly] for poly in quant_poly]
        index12 = [[safe_index(p[0], p[1], False, True) for p in poly] for poly in quant_poly]
        index22 = [[safe_index(p[0], p[1], True, True) for p in poly] for poly in quant_poly]

        # Validate indices are within vocabulary range before tokenization
        max_valid_idx = num_bins * num_bins - 1
        for idx_list in [index11, index21, index12, index22]:
            for sublist in idx_list:
                for idx in sublist:
                    if idx < 0 or idx > max_valid_idx:
                        raise ValueError(
                            f"Invalid token index {idx} (valid range: 0-{max_valid_idx}). "
                            f"This indicates a bug in index computation."
                        )
        
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
    # Use resolve() to convert relative paths to absolute first
    dataset_root_abs = Path(args.dataset_root).resolve()
    root = dataset_root_abs / "data"
    # annotations are in dataset_root/annotations
    ann_file = dataset_root_abs / "annotations" / f"mp100_split{split_num}_{image_set}.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # Build transforms
    # Use ResizeAndPad to preserve aspect ratio and pad to fixed size
    # This prevents image distortion while ensuring all images are the same size
    image_size = getattr(args, 'image_size', 256)
    if image_set == 'train':
        transforms = ResizeAndPad((image_size, image_size), pad_value=0)
    else:
        transforms = ResizeAndPad((image_size, image_size), pad_value=0)

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
        skip_missing_files=getattr(args, 'skip_missing_files', False)  # False by default (slow for large datasets)
    )

    return dataset
