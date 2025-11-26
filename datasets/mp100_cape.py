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


class ImageNotFoundError(Exception):
    """Raised when an image file is not found in the dataset."""
    pass

from .token_types import TokenType

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
            """
            DEPRECATED/UNUSED: This is a placeholder method from the original codebase.
            
            The actual tokenization for CAPE is done in _tokenize_keypoints() which
            properly implements bilinear interpolation with all 4 sequences (seq11,
            seq21, seq12, seq22) and their corresponding deltas.
            
            This method is kept for backwards compatibility but should NOT be used.
            """
            raise NotImplementedError(
                "convert_to_sequence is deprecated. Use _tokenize_keypoints() instead."
            )

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

        # Analyze multi-instance statistics
        self._analyze_multi_instance_stats()

        print(f"Loaded MP-100 {split} dataset: {len(self.ids)} images")

    def _analyze_multi_instance_stats(self):
        """
        Analyze and report statistics about multi-instance images.
        
        This helps users understand how much data is potentially unused
        due to the single-instance-per-image limitation.
        """
        total_images = len(self.ids)
        multi_instance_images = 0
        total_instances = 0
        max_instances_per_image = 0
        
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Count valid annotations (those with keypoints and bbox)
            valid_anns = 0
            for ann in anns:
                if 'keypoints' in ann and ann['keypoints'] and 'bbox' in ann:
                    kpts = np.array(ann['keypoints']).reshape(-1, 3)
                    if np.any(kpts[:, 2] > 0):  # Has visible keypoints
                        valid_anns += 1
            
            if valid_anns > 0:
                total_instances += valid_anns
                if valid_anns > 1:
                    multi_instance_images += 1
                max_instances_per_image = max(max_instances_per_image, valid_anns)
        
        # Store statistics
        self.multi_instance_stats = {
            'total_images': total_images,
            'total_instances': total_instances,
            'multi_instance_images': multi_instance_images,
            'max_instances_per_image': max_instances_per_image,
            'instances_used': total_images,  # We use 1 per image
            'instances_unused': total_instances - total_images
        }
        
        # Report if significant data is being skipped
        if multi_instance_images > 0:
            pct_multi = (multi_instance_images / total_images) * 100
            pct_unused = (self.multi_instance_stats['instances_unused'] / total_instances) * 100
            print(f"📊 Multi-instance statistics:")
            print(f"   - Images with multiple instances: {multi_instance_images}/{total_images} ({pct_multi:.1f}%)")
            print(f"   - Total instances available: {total_instances}")
            print(f"   - Instances actually used: {total_images} ({100-pct_unused:.1f}%)")
            print(f"   - Instances skipped: {self.multi_instance_stats['instances_unused']} ({pct_unused:.1f}%)")
            print(f"   - Max instances in single image: {max_instances_per_image}")
            print(f"   ⚠️  Note: Currently using only first instance per image")

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
                - image: query image tensor (cropped to bbox, resized to 512x512)
                - keypoints: list of keypoint coordinates [[x1,y1], [x2,y2], ...] (relative to bbox)
                - category_id: object category
                - num_keypoints: number of keypoints
                - image_id: image identifier
                - file_name: image path
                - bbox: bounding box [x, y, w, h]
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

        # Check if image file exists (skip if not in GS bucket)
        # This check happens during training/testing, not at initialization
        if not os.path.exists(file_name):
            raise ImageNotFoundError(f"Image not found: {file_name}")

        img = np.array(Image.open(file_name).convert('RGB'))

        # ========================================================================
        # Validate image dimensions before processing
        # ========================================================================
        # Check if image loaded correctly and has valid dimensions
        if img is None or img.size == 0:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) failed to load or is empty"
            )
        
        if len(img.shape) < 2:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) has invalid shape: {img.shape}"
            )

        # Handle image dimensions
        if len(img.shape) >= 3:
            if img.shape[-1] > 3:  # drop alpha channel
                img = img[:, :, :3]
            orig_h, orig_w = img.shape[:2]
        else:
            orig_h, orig_w = img.shape
        
        # Validate dimensions are positive
        if orig_h <= 0 or orig_w <= 0:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) has invalid dimensions: {orig_w}x{orig_h}"
            )

        # Build record
        record = {}
        record["file_name"] = file_name
        record["image_id"] = img_id

        # Process keypoints and bbox from annotations
        keypoints_list = []
        category_ids = []
        num_keypoints_list = []
        bbox_list = []
        visibility_list = []

        for ann in target:
            if 'keypoints' in ann and ann['keypoints']:
                # COCO keypoints format: [x1,y1,v1,x2,y2,v2,...]
                # where v is visibility (0=not labeled, 1=labeled but not visible, 2=labeled and visible)
                kpts = np.array(ann['keypoints']).reshape(-1, 3)

                # Store visibility flags for all keypoints
                visibility = kpts[:, 2]

                # Filter visible keypoints (v > 0) - but keep track of all keypoint positions
                visible_mask = kpts[:, 2] > 0
                visible_kpts = kpts[visible_mask][:, :2]  # only take x, y

                if len(visible_kpts) > 0 and 'bbox' in ann:
                    keypoints_list.append(kpts[:, :2].tolist())  # Store ALL keypoints (including non-visible)
                    visibility_list.append(visibility.tolist())
                    category_ids.append(ann.get('category_id', 0))
                    num_keypoints_list.append(len(visible_kpts))
                    bbox_list.append(ann['bbox'])

        # ========================================================================
        # LIMITATION: Multi-instance images only use first instance
        # ========================================================================
        # If an image has multiple annotated objects (e.g., 2 people), we only
        # use the first one. This is a simplification for the initial implementation.
        #
        # Future improvement: Treat each instance as a separate datapoint.
        # This would require:
        #   1. Modify __len__() to count total instances, not images
        #   2. Modify __getitem__() to map index → (image_id, instance_idx)
        #   3. Update self.ids to be instance-based rather than image-based
        #
        # Impact: If 10% of images have 2+ instances, we're using ~90% of potential data.
        # For MP-100, most images have single instances, so impact is minimal.
        # ========================================================================
        
        if len(keypoints_list) > 0:
            # Log multi-instance images (for debugging/analysis)
            if len(keypoints_list) > 1:
                # Only log occasionally to avoid spam
                if not hasattr(self, '_multi_instance_count'):
                    self._multi_instance_count = 0
                self._multi_instance_count += 1
                if self._multi_instance_count <= 5:  # Log first 5 occurrences
                    print(f"\nNote: Image {img_id} has {len(keypoints_list)} instances, using first only")
            
            # Extract bbox [x, y, width, height] from FIRST instance
            bbox = bbox_list[0]
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            
            # Ensure bbox is within image bounds
            bbox_x = max(0, int(bbox_x))
            bbox_y = max(0, int(bbox_y))
            bbox_w = min(int(bbox_w), orig_w - bbox_x)
            bbox_h = min(int(bbox_h), orig_h - bbox_y)
            
            # ================================================================
            # Step 1: Crop image to bounding box
            # ================================================================
            img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            
            # Validate cropped image has valid dimensions
            if img_cropped.size == 0 or img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
                raise ImageNotFoundError(
                    f"Image {img_id} produced empty crop with bbox [{bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}]. "
                    f"Original image size: {orig_w}x{orig_h}"
                )
            
            # ================================================================
            # Step 2: Make keypoints relative to cropped bbox
            # ================================================================
            # Subtract bbox offset so keypoints are in bbox coordinate system
            # Before: keypoints in range [bbox_x, bbox_x+bbox_w] × [bbox_y, bbox_y+bbox_h]
            # After:  keypoints in range [0, bbox_w] × [0, bbox_h]
            kpts_array = np.array(keypoints_list[0])
            kpts_array[:, 0] -= bbox_x  # x relative to bbox top-left
            kpts_array[:, 1] -= bbox_y  # y relative to bbox top-left
            
            # ================================================================
            # CRITICAL FIX #1: Keep ALL keypoints to preserve index correspondence
            # ================================================================
            # PROBLEM: Filtering keypoints by visibility breaks skeleton edge alignment
            #   - Skeleton edges reference original keypoint indices (e.g., [0,1,2,...,N])
            #   - If we filter out invisible keypoints, indices get renumbered
            #   - Edge [i,j] no longer connects the correct keypoints!
            #
            # SOLUTION: Keep ALL keypoints including invisible ones
            #   - Use visibility as a MASK in loss computation and evaluation
            #   - Do NOT remove keypoints based on visibility
            #   - Ensures skeleton edges correctly reference coordinate indices
            #
            # Example:
            #   Original: [kpt0, kpt1(invisible), kpt2, kpt3]
            #   Skeleton: [[0,1], [1,2], [2,3]]
            #   After filtering (OLD BAD WAY): [kpt0, kpt2, kpt3] with indices [0,1,2]
            #   → Edge [0,1] connects kpt0→kpt2 (WRONG! Should be kpt0→kpt1)
            #   
            #   After fix (NEW CORRECT WAY): [kpt0, kpt1, kpt2, kpt3] with visibility [2,0,2,2]
            #   → Edge [0,1] connects kpt0→kpt1 (CORRECT!)
            #   → Loss/eval mask out kpt1 using visibility
            # ================================================================
            
            visibility = np.array(visibility_list[0])
            
            # Store ALL keypoints (including invisible ones) to preserve indices
            record["keypoints"] = kpts_array.tolist()  # All keypoints, not filtered!
            record["visibility"] = visibility.tolist()
            record["category_id"] = category_ids[0]
            
            # ================================================================
            # CRITICAL FIX: num_keypoints should mean TOTAL keypoints
            # ================================================================
            # Previously stored only visible count, which broke fallback logic
            # in episodic_sampler when visibility wasn't passed through.
            # Now we store:
            #   - num_keypoints: TOTAL keypoints (visible + invisible)
            #   - num_visible_keypoints: Visible count (for statistics)
            # ================================================================
            record["num_keypoints"] = len(kpts_array)  # Total keypoints
            record["num_visible_keypoints"] = int(np.sum(visibility > 0))  # Visible count
            record["bbox"] = [bbox_x, bbox_y, bbox_w, bbox_h]
            
            # Store bbox dimensions for normalization
            record["bbox_width"] = bbox_w
            record["bbox_height"] = bbox_h
            
            # Use cropped image
            img = img_cropped
            record["height"] = bbox_h
            record["width"] = bbox_w

            # Get skeleton edges for this category
            record["skeleton"] = self._get_skeleton_for_category(category_ids[0])
        else:
            # ========================================================================
            # CRITICAL FIX: Skip empty annotations instead of using dummy values
            # ========================================================================
            # Problem: Dummy values (single keypoint at [0, 0]) corrupt training:
            #   - Model learns invalid pose patterns
            #   - Loss computation is meaningless on fake data
            #   - Evaluation metrics are contaminated
            #
            # Solution: Raise exception for empty annotations
            #   - Episodic sampler's retry logic will skip this image
            #   - Only valid annotations are used for training
            #   - Clean, meaningful training data
            #
            # Note: This may reduce dataset size slightly, but ensures data quality
            # ========================================================================
            raise ImageNotFoundError(
                f"Image {img_id} has no valid annotations (no visible keypoints or missing bbox). "
                f"Skipping this image to avoid training on dummy data."
            )

        # Apply transforms with error handling
        if self._transforms is not None:
            try:
                img, record = self._apply_transforms(img, record)
            except Exception as e:
                # If transforms fail (e.g., resize on corrupted image), skip this sample
                raise ImageNotFoundError(
                    f"Image {img_id} ({record.get('file_name', 'unknown')}) failed during transforms: {str(e)}"
                ) from e

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
            record["seq_data"] = self._tokenize_keypoints(
                keypoints=record["keypoints"],
                height=record["height"],
                width=record["width"],
                visibility=record.get("visibility", None)  # Pass visibility info
            )
            # Clean up temporary attribute
            delattr(self, '_current_category_id')

        # ================================================================
        # CRITICAL VALIDATION: Ensure keypoints and visibility are aligned
        # ================================================================
        # This is our last chance to catch misalignment before data goes to training.
        # We validate against BOTH each other AND the category definition.
        # ================================================================
        if 'visibility' in record and 'keypoints' in record and 'category_id' in record:
            num_kpts = len(record['keypoints'])
            num_vis = len(record['visibility'])
            category_id = record['category_id']
            
            # Get expected number of keypoints for this category
            expected_num_kpts = self._get_num_keypoints_for_category(category_id)
            
            # Check 1: keypoints and visibility must match each other
            if num_kpts != num_vis:
                raise ValueError(
                    f"CRITICAL BUG in mp100_cape.py __getitem__: "
                    f"keypoints length ({num_kpts}) != visibility length ({num_vis}) "
                    f"for image {record.get('image_id', 'unknown')}, category {category_id}. "
                    f"This indicates a bug in transform logic (likely Albumentations dropped keypoints)."
                )
            
            # Check 2: Both must match the category definition
            if expected_num_kpts is not None and num_kpts != expected_num_kpts:
                raise ValueError(
                    f"CRITICAL BUG in mp100_cape.py __getitem__: "
                    f"keypoints length ({num_kpts}) != expected for category {category_id} ({expected_num_kpts}) "
                    f"for image {record.get('image_id', 'unknown')}. "
                    f"This indicates Albumentations dropped keypoints or incorrect data loading. "
                    f"Image will be skipped."
                )

        return record

    def _get_skeleton_for_category(self, category_id):
        """
        Get skeleton edges for a given category from COCO annotations.

        Args:
            category_id: Category ID

        Returns:
            List of [src, dst] edge pairs defining the skeleton structure.
            Returns empty list if no skeleton defined for this category.
        """
        try:
            # Get category info from COCO
            cat_info = self.coco.loadCats(category_id)[0]

            # MP-100 stores skeleton in 'skeleton' field
            # Format: [[src1, dst1], [src2, dst2], ...]
            skeleton = cat_info.get('skeleton', [])

            return skeleton if skeleton else []

        except Exception as e:
            # If category not found or error, return empty skeleton
            return []
    
    def _get_num_keypoints_for_category(self, category_id):
        """
        Get the expected number of keypoints for a category.
        
        Args:
            category_id: Category ID
        
        Returns:
            Number of keypoints for this category, or None if unknown
        """
        try:
            cat_info = self.coco.loadCats(category_id)[0]
            # MP-100 stores keypoint names in 'keypoints' field
            keypoint_names = cat_info.get('keypoints', [])
            return len(keypoint_names) if keypoint_names else None
        except Exception as e:
            return None

    def _apply_transforms(self, img, record):
        """
        Apply image transformations and update keypoint coordinates.
        
        This is Step 3 of the normalization pipeline:
        - Resizes cropped bbox image from (bbox_w × bbox_h) to (512 × 512)
        - Scales keypoints proportionally to maintain relative positions
        - Applies data augmentation (training only)
        
        After this:
        - Image is 512×512
        - Keypoints are in range [0, 512] × [0, 512]
        - record['height'] and record['width'] are both 512
        
        Final normalization (divide by 512) happens in episodic_sampler.py
        """
        if self._transforms is None:
            return img, record

        # Get original dimensions (bbox dimensions after cropping)
        orig_h, orig_w = record["height"], record["width"]

        # ========================================================================
        # Apply transform with keypoint-aware augmentation (Issue #19 fix)
        # ========================================================================
        # Albumentations automatically transforms keypoints along with the image,
        # maintaining their relative positions during geometric augmentations.
        # ========================================================================
        
        # Prepare keypoints in (x, y) format for albumentations
        keypoints_list = record.get('keypoints', [])
        num_keypoints_before = len(keypoints_list)
        
        # Apply transform (resize + augmentation)
        transformed = self._transforms(image=img, keypoints=keypoints_list)
        img = transformed['image']

        # Update keypoints with transformed coordinates
        transformed_keypoints = transformed.get('keypoints', keypoints_list)
        
        # ================================================================
        # CRITICAL CHECK: Albumentations might drop keypoints!
        # ================================================================
        # Even with remove_invisible=False, Albumentations can drop keypoints
        # that fall outside image bounds after transforms. This breaks our
        # index correspondence with skeleton edges and visibility array.
        #
        # When this happens, we MUST skip this image entirely because:
        # 1. We can't update visibility (don't know WHICH keypoints were dropped)
        # 2. Skeleton edges would reference wrong indices
        # 3. Would corrupt training data
        # ================================================================
        num_keypoints_after = len(transformed_keypoints)
        if num_keypoints_after != num_keypoints_before:
            # Albumentations dropped some keypoints - skip this image
            raise ImageNotFoundError(
                f"Albumentations dropped keypoints! Before: {num_keypoints_before}, "
                f"After: {num_keypoints_after}. Skipping image {record.get('image_id', 'unknown')} "
                f"to maintain data integrity (skeleton/visibility alignment)."
            )
        
        record['keypoints'] = list(transformed_keypoints)
        
        # IMPORTANT: record['visibility'] is NOT updated here!
        # It was set in __getitem__ and must remain synchronized with keypoints.
        # The check above ensures they stay in sync.

        # Update dimensions to post-resize values
        new_h, new_w = img.shape[:2]  # Typically (512, 512)
        record["height"] = new_h
        record["width"] = new_w

        # ================================================================
        # NOTE: Keypoints are already scaled by Albumentations!
        # ================================================================
        # Albumentations automatically scales keypoints when it resizes the image.
        # The transformed_keypoints from line 535 are already in the range [0, 512].
        # We do NOT need to manually scale them again - that would double-scale them!
        #
        # Previous buggy code:
        #   scale_x = new_w / orig_w
        #   scaled_keypoints = [kpt × scale for kpt in keypoints]  # WRONG! Already scaled
        #
        # Correct: Just use the keypoints as-is from Albumentations
        # ================================================================

        return img, record

    def _tokenize_keypoints(self, keypoints, height, width, visibility=None):
        """
        Tokenize keypoint coordinates using bilinear interpolation.
        Following poly_data.py pattern for compatibility with Raster2Seq.
        
        CRITICAL: After FIX #1, keypoints includes ALL keypoints (visible and invisible)
        to preserve index correspondence with skeleton edges.

        Args:
            keypoints: List of [x, y] coordinates for ALL keypoints (including invisible!)
            height, width: Image dimensions for normalization
            visibility: List of visibility flags (same length as keypoints)
                - 0 = not labeled (no annotation)
                - 1 = labeled but occluded/not visible  
                - 2 = labeled and visible
                - If None, assumes all keypoints are visible

        Returns:
            dict with tokenized sequence data including visibility_mask
        """
        import math

        # ========================================================================
        # CRITICAL FIX #1: Create visibility mask for loss computation
        # ========================================================================
        # The visibility mask indicates which coordinate tokens should be used
        # for loss computation. Only visible keypoints (visibility > 0) contribute
        # to the loss. Invisible keypoints (visibility == 0) are masked out.
        #
        # IMPORTANT: We still tokenize ALL keypoints (including invisible ones)
        # to preserve index correspondence with skeleton edges. The visibility
        # mask ensures invisible keypoints don't affect the loss.
        # ========================================================================

        # Set default visibility if not provided
        if visibility is None:
            visibility = [2] * len(keypoints)  # Assume all visible

        # Normalize keypoints to [0, 1]
        normalized_kpts = []
        for x, y in keypoints:
            normalized_kpts.append([x / width, y / height])

        # For CAPE, treat all keypoints as one "polygon" (one continuous sequence)
        polygons = [np.array(normalized_kpts)]

        # Quantize coordinates using bilinear interpolation
        num_bins = self.tokenizer.num_bins
        quant_poly = [poly * (num_bins - 1) for poly in polygons]

        # Clamp quantized coordinates to valid range [0, num_bins-1]
        # This prevents out-of-bounds issues when augmentation pushes coords > 1.0
        quant_poly = [np.clip(poly, 0, num_bins - 1) for poly in quant_poly]

        # ========================================================================
        # CRITICAL FIX #2: Bilinear interpolation requires 4 sequences
        # ========================================================================
        # Raster2Seq uses bilinear interpolation to embed coordinates.
        # For each coordinate (x, y), we need 4 grid points:
        #   - (floor_x, floor_y)  -> index11
        #   - (ceil_x, floor_y)   -> index21
        #   - (floor_x, ceil_y)   -> index12
        #   - (ceil_x, ceil_y)    -> index22
        #
        # These are NOT duplicates! They represent the 4 corners of the grid cell.
        # The model uses delta_x and delta_y to interpolate between them.
        #
        # Previous Fix #18 incorrectly removed seq21 and seq22, thinking they were
        # duplicates. They are NOT - they are required for bilinear interpolation!
        # ========================================================================

        # ========================================================================
        # CRITICAL FIX #3: Clamp indices to prevent vocab overflow
        # ========================================================================
        # Problem: Data augmentation can push coordinates slightly > 1.0
        #          Example: coord = 1.001 -> quantized = 1.001 * 43 = 43.043
        #          Then: ceil(43.043) = 44, and 44 * 44 = 1936 (BOS token!)
        #          This causes CUDA index out of bounds errors.
        #
        # Solution: Clamp floor/ceil values to [0, num_bins-1] before computing
        #           the flattened 2D index. This ensures indices stay in [0, 1935].
        # ========================================================================
        
        # 4 indices for bilinear interpolation (floor/ceil combinations with clamping)
        index11 = [[min(num_bins-1, max(0, math.floor(p[0])))*num_bins + min(num_bins-1, max(0, math.floor(p[1]))) for p in poly] for poly in quant_poly]
        index21 = [[min(num_bins-1, max(0, math.ceil(p[0])))*num_bins + min(num_bins-1, max(0, math.floor(p[1]))) for p in poly] for poly in quant_poly]
        index12 = [[min(num_bins-1, max(0, math.floor(p[0])))*num_bins + min(num_bins-1, max(0, math.ceil(p[1]))) for p in poly] for poly in quant_poly]
        index22 = [[min(num_bins-1, max(0, math.ceil(p[0])))*num_bins + min(num_bins-1, max(0, math.ceil(p[1]))) for p in poly] for poly in quant_poly]

        # Tokenize all 4 sequences
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

        # Create mask for valid tokens (not padding)
        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        if len(token_labels) < self.tokenizer.seq_len:
            mask[len(token_labels):] = 0

        # ========================================================================
        # Create visibility mask for loss computation (CRITICAL FIX #1)
        # ========================================================================
        # This mask indicates which coordinate tokens should contribute to loss.
        # After FIX #1, keypoints includes ALL keypoints (visible and invisible).
        # We use the visibility array to determine which tokens to use in loss.
        #
        # Visibility rules:
        #   - Coordinate tokens from VISIBLE keypoints (visibility > 0): True
        #   - Coordinate tokens from INVISIBLE keypoints (visibility == 0): False
        #   - SEP/EOS/padding tokens: False (don't use in loss)
        # ========================================================================
        
        visibility_mask = torch.zeros(self.tokenizer.seq_len, dtype=torch.bool)
        
        # Mark coordinate tokens based on actual visibility
        # Each keypoint generates one coordinate token (after BOS)
        token_idx = 0  # Start after BOS
        for kpt_idx, kpt in enumerate(keypoints):
            if token_idx >= len(token_labels):
                break
            # Skip BOS token
            if token_idx == 0 and token_labels[token_idx] != TokenType.coord.value:
                token_idx += 1
            # Check if this is a coordinate token
            if token_idx < len(token_labels) and token_labels[token_idx] == TokenType.coord.value:
                # Use visibility: only mark as True if visibility > 0
                if visibility[kpt_idx] > 0:
                    visibility_mask[token_idx] = True
                token_idx += 1
        
        # ========================================================================
        # CRITICAL FIX: Include EOS token in visibility mask
        # ========================================================================
        # BUG: Previously EOS was excluded from loss (line 737 comment)
        # IMPACT: Model never learned to predict EOS → always generates 200 tokens
        # FIX: Mark EOS token as True so model receives gradient signal to learn
        #      when to stop generation
        # ========================================================================
        for i, label in enumerate(token_labels):
            if label == TokenType.eos.value:
                visibility_mask[i] = True
                break  # Only mark first EOS

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
            'seq21': seq21,  # CRITICAL FIX #2: Need all 4 sequences for bilinear interpolation
            'seq12': seq12,
            'seq22': seq22,  # CRITICAL FIX #2: Need all 4 sequences for bilinear interpolation
            'target_seq': target_seq,
            'token_labels': token_labels,
            'mask': mask,  # Valid token mask (not padding)
            'visibility_mask': visibility_mask,  # Visibility mask for loss (visible keypoints only)
            'target_polygon_labels': target_polygon_labels,
            'delta_x1': delta_x1,
            'delta_x2': delta_x2,  # CRITICAL FIX #2: Need all 4 deltas for bilinear interpolation
            'delta_y1': delta_y1,
            'delta_y2': delta_y2,  # CRITICAL FIX #2: Need all 4 deltas for bilinear interpolation
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

    # ========================================================================
    # APPEARANCE-ONLY AUGMENTATION FOR TRAINING
    # ========================================================================
    # Apply photometric augmentations that improve robustness WITHOUT changing
    # geometric relationships between pixels and keypoints.
    #
    # ✅ ALLOWED (appearance-only):
    #   - Color jitter (brightness, contrast, saturation, hue)
    #   - Gaussian noise
    #   - Gaussian blur
    #
    # ❌ FORBIDDEN (would change geometry):
    #   - Random crop, flip, rotation, affine, perspective
    #   - Any transform that moves pixels or changes spatial layout
    #
    # CRITICAL GUARANTEES:
    #   1. Keypoint tensors remain BITWISE IDENTICAL (not modified at all)
    #   2. Image geometry preserved (pixel positions unchanged)
    #   3. Only training uses augmentation; val/test are deterministic
    #
    # WHY NO GEOMETRIC AUGMENTATION:
    #   Geometric transforms require updating keypoint coordinates accordingly.
    #   Since we want keypoint annotations to remain untouched, we only use
    #   appearance transforms that don't affect pixel positions.
    # ========================================================================
    
    # Build transforms with augmentation for training
    if image_set == 'train':
        # Training: APPEARANCE-ONLY augmentation
        import albumentations as A
        transforms = A.Compose([
            # Color jitter: Vary brightness, contrast, saturation, hue
            # This helps model generalize across different lighting conditions
            A.ColorJitter(
                brightness=0.2,  # ±20% brightness variation
                contrast=0.2,    # ±20% contrast variation
                saturation=0.2,  # ±20% saturation variation
                hue=0.05,        # ±5% hue shift (small to avoid unrealistic colors)
                p=0.6            # Apply 60% of the time
            ),
            
            # Gaussian noise: Add small random noise to image
            # Helps model be robust to sensor noise and compression artifacts
            A.GaussNoise(
                var_limit=(5.0, 25.0),  # Low variance to avoid corrupting image
                mean=0,                   # Zero-mean noise
                p=0.4                     # Apply 40% of the time
            ),
            
            # Gaussian blur: Slight blur to simulate focus variations
            # Helps with images that may be slightly out of focus
            A.GaussianBlur(
                blur_limit=(3, 5),  # Small kernel size (3x3 or 5x5)
                sigma_limit=0,      # Auto-select sigma based on kernel
                p=0.2               # Apply 20% of the time (less frequent)
            ),
            
            # Final deterministic resize to 512x512
            # This is NOT augmentation - it's required normalization
            A.Resize(height=512, width=512),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        # Validation/Test: ONLY deterministic resize (no augmentation)
        import albumentations as A
        transforms = A.Compose([
            A.Resize(height=512, width=512)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

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
        seq_len=args.seq_len
    )

    return dataset
