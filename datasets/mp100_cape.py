"""
MP-100 Dataset for Category-Agnostic Pose Estimation (CAPE).
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
    """
    Raised when an image file is not found in the dataset.
    """
    pass
from .token_types import TokenType
try:
    from datasets.transforms import Resize, ResizeAndPad
except ImportError:
    print("Warning: Using fallback transforms (detectron2 not available)")
    class Resize:
        def __init__(self, size):
            self.size = size if isinstance(size, tuple) else (size, size)
        def __call__(self, image):
            import cv2
            resized = cv2.resize(image, self.size[::-1])
            return {'image': resized}
    class ResizeAndPad:
        def __init__(self, size):
            self.size = size if isinstance(size, tuple) else (size, size)
        def __call__(self, image):
            import cv2
            resized = cv2.resize(image, self.size[::-1])
            return {'image': resized}
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
from datasets.discrete_tokenizer import DiscreteTokenizer, DiscreteTokenizerV2
class MP100CAPE(torch.utils.data.Dataset):
    """
    MP-100 Dataset for Category-Agnostic Pose Estimation.
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
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.poly2seq = poly2seq
        if image_norm:
            self.image_normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.image_normalize = None
        if poly2seq:
            vocab_size = kwargs.get('vocab_size', 2000)
            seq_len = kwargs.get('seq_len', 200)
            num_bins = int(np.sqrt(vocab_size))
            self.tokenizer = DiscreteTokenizerV2(num_bins=num_bins, seq_len=seq_len, add_cls=False)
        self._analyze_multi_instance_stats()
        print(f"Loaded MP-100 {split} dataset: {len(self.ids)} images")
    def _analyze_multi_instance_stats(self):
        """
        Analyze and report statistics about multi-instance images.
        """
        total_images = len(self.ids)
        multi_instance_images = 0
        total_instances = 0
        max_instances_per_image = 0
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_anns = 0
            for ann in anns:
                if 'keypoints' in ann and ann['keypoints'] and 'bbox' in ann:
                    kpts = np.array(ann['keypoints']).reshape(-1, 3)
                    if np.any(kpts[:, 2] > 0):
                        valid_anns += 1
            if valid_anns > 0:
                total_instances += valid_anns
                if valid_anns > 1:
                    multi_instance_images += 1
                max_instances_per_image = max(max_instances_per_image, valid_anns)
        self.multi_instance_stats = {
            'total_images': total_images,
            'total_instances': total_instances,
            'multi_instance_images': multi_instance_images,
            'max_instances_per_image': max_instances_per_image,
            'instances_used': total_images,
            'instances_unused': total_instances - total_images
        }
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
        """
        Expand image dimensions to (C, H, W).
        
        Args:
            x: Image array
        
        Returns:
            Expanded image array
        """
        if len(x.shape) == 2:
            exp_img = np.expand_dims(x, 0)
        else:
            exp_img = x.transpose((2, 0, 1))
        return exp_img
    def __getitem__(self, index):
        """
        Get dataset item.
        
        Args:
            index: Item index
        
        Returns:
            dict with image, keypoints, category_id, num_keypoints, image_id,
            file_name, bbox, seq_data
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        file_name = os.path.join(self.root, path)
        if not os.path.exists(file_name):
            raise ImageNotFoundError(f"Image not found: {file_name}")
        img = np.array(Image.open(file_name).convert('RGB'))
        if img is None or img.size == 0:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) failed to load or is empty"
            )
        if len(img.shape) < 2:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) has invalid shape: {img.shape}"
            )
        if len(img.shape) >= 3:
            if img.shape[-1] > 3:
                img = img[:, :, :3]
            orig_h, orig_w = img.shape[:2]
        else:
            orig_h, orig_w = img.shape
        if orig_h <= 0 or orig_w <= 0:
            raise ImageNotFoundError(
                f"Image {img_id} ({file_name}) has invalid dimensions: {orig_w}x{orig_h}"
            )
        record = {}
        record["file_name"] = file_name
        record["image_id"] = img_id
        keypoints_list = []
        category_ids = []
        num_keypoints_list = []
        bbox_list = []
        visibility_list = []
        for ann in target:
            if 'keypoints' in ann and ann['keypoints']:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                visibility = kpts[:, 2]
                visible_mask = kpts[:, 2] > 0
                visible_kpts = kpts[visible_mask][:, :2]
                if len(visible_kpts) > 0 and 'bbox' in ann:
                    keypoints_list.append(kpts[:, :2].tolist())
                    visibility_list.append(visibility.tolist())
                    category_ids.append(ann.get('category_id', 0))
                    num_keypoints_list.append(len(visible_kpts))
                    bbox_list.append(ann['bbox'])
        if len(keypoints_list) > 0:
            if len(keypoints_list) > 1:
                if not hasattr(self, '_multi_instance_count'):
                    self._multi_instance_count = 0
                self._multi_instance_count += 1
                if self._multi_instance_count <= 5:
                    print(f"\nNote: Image {img_id} has {len(keypoints_list)} instances, using first only")
            bbox = bbox_list[0]
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            bbox_x = max(0, int(bbox_x))
            bbox_y = max(0, int(bbox_y))
            bbox_w = min(int(bbox_w), orig_w - bbox_x)
            bbox_h = min(int(bbox_h), orig_h - bbox_y)
            img_cropped = img[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
            if img_cropped.size == 0 or img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
                raise ImageNotFoundError(
                    f"Image {img_id} produced empty crop with bbox [{bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}]. "
                    f"Original image size: {orig_w}x{orig_h}"
                )
            kpts_array = np.array(keypoints_list[0])
            kpts_array[:, 0] -= bbox_x
            kpts_array[:, 1] -= bbox_y
            visibility = np.array(visibility_list[0])
            record["keypoints"] = kpts_array.tolist()
            record["visibility"] = visibility.tolist()
            record["category_id"] = category_ids[0]
            record["num_keypoints"] = len(kpts_array)
            record["num_visible_keypoints"] = int(np.sum(visibility > 0))
            record["bbox"] = [bbox_x, bbox_y, bbox_w, bbox_h]
            record["bbox_width"] = bbox_w
            record["bbox_height"] = bbox_h
            img = img_cropped
            record["height"] = bbox_h
            record["width"] = bbox_w
            record["skeleton"] = self._get_skeleton_for_category(category_ids[0])
        else:
            raise ImageNotFoundError(
                f"Image {img_id} has no valid annotations (no visible keypoints or missing bbox). "
                f"Skipping this image to avoid training on dummy data."
            )
        if self._transforms is not None:
            try:
                img, record = self._apply_transforms(img, record)
            except Exception as e:
                raise ImageNotFoundError(
                    f"Image {img_id} ({record.get('file_name', 'unknown')}) failed during transforms: {str(e)}"
                ) from e
        img_tensor = torch.as_tensor(self._expand_image_dims(img))
        if self.image_normalize:
            img_tensor = self.image_normalize(img_tensor.float() / 255.0)
        else:
            img_tensor = img_tensor.float() / 255.0
        record["image"] = img_tensor
        if self.poly2seq:
            self._current_category_id = record["category_id"]
            record["seq_data"] = self._tokenize_keypoints(
                keypoints=record["keypoints"],
                height=record["height"],
                width=record["width"],
                visibility=record.get("visibility", None)
            )
            delattr(self, '_current_category_id')
        if 'visibility' in record and 'keypoints' in record and 'category_id' in record:
            num_kpts = len(record['keypoints'])
            num_vis = len(record['visibility'])
            category_id = record['category_id']
            expected_num_kpts = self._get_num_keypoints_for_category(category_id)
            if num_kpts != num_vis:
                raise ValueError(
                    f"CRITICAL BUG in mp100_cape.py __getitem__: "
                    f"keypoints length ({num_kpts}) != visibility length ({num_vis}) "
                    f"for image {record.get('image_id', 'unknown')}, category {category_id}. "
                    f"This indicates a bug in transform logic (likely Albumentations dropped keypoints)."
                )
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
            cat_info = self.coco.loadCats(category_id)[0]
            skeleton = cat_info.get('skeleton', [])
            return skeleton if skeleton else []
        except Exception as e:
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
        orig_h, orig_w = record["height"], record["width"]
        keypoints_list = record.get('keypoints', [])
        num_keypoints_before = len(keypoints_list)
        transformed = self._transforms(image=img, keypoints=keypoints_list)
        img = transformed['image']
        transformed_keypoints = transformed.get('keypoints', keypoints_list)
        num_keypoints_after = len(transformed_keypoints)
        if num_keypoints_after != num_keypoints_before:
            raise ImageNotFoundError(
                f"Albumentations dropped keypoints! Before: {num_keypoints_before}, "
                f"After: {num_keypoints_after}. Skipping image {record.get('image_id', 'unknown')} "
                f"to maintain data integrity (skeleton/visibility alignment)."
            )
        record['keypoints'] = list(transformed_keypoints)
        new_h, new_w = img.shape[:2]
        record["height"] = new_h
        record["width"] = new_w
        record["bbox_width"] = new_w
        record["bbox_height"] = new_h
        return img, record
    def _tokenize_keypoints(self, keypoints, height, width, visibility=None):
        """
        Tokenize keypoint coordinates using bilinear interpolation.
        Following poly_data.py pattern for compatibility with Raster2Seq.
        CRITICAL: After FIX
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
        if visibility is None:
            visibility = [2] * len(keypoints)
        normalized_kpts = []
        for x, y in keypoints:
            normalized_kpts.append([x / width, y / height])
        polygons = [np.array(normalized_kpts)]
        num_bins = self.tokenizer.num_bins
        quant_poly = [poly * (num_bins - 1) for poly in polygons]
        quant_poly = [np.clip(poly, 0, num_bins - 1) for poly in quant_poly]
        index11 = [[min(num_bins-1, max(0, math.floor(p[0])))*num_bins + min(num_bins-1, max(0, math.floor(p[1]))) for p in poly] for poly in quant_poly]
        index21 = [[min(num_bins-1, max(0, math.ceil(p[0])))*num_bins + min(num_bins-1, max(0, math.floor(p[1]))) for p in poly] for poly in quant_poly]
        index12 = [[min(num_bins-1, max(0, math.floor(p[0])))*num_bins + min(num_bins-1, max(0, math.ceil(p[1]))) for p in poly] for poly in quant_poly]
        index22 = [[min(num_bins-1, max(0, math.ceil(p[0])))*num_bins + min(num_bins-1, max(0, math.ceil(p[1]))) for p in poly] for poly in quant_poly]
        seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
        seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
        seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
        seq22 = self.tokenizer(index22, add_bos=True, add_eos=False, dtype=torch.long)
        target_seq = []
        token_labels = []
        add_cls_token = False
        for poly in polygons:
            token_labels.extend([TokenType.coord.value] * len(poly))
            target_seq.extend(poly)
            if add_cls_token:
                token_labels.append(TokenType.cls.value)
                target_seq.append([0, 0])
            token_labels.append(TokenType.sep.value)
            target_seq.append([0, 0])
        if len(token_labels) > 0:
            token_labels[-1] = TokenType.eos.value
        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        if len(token_labels) < self.tokenizer.seq_len:
            mask[len(token_labels):] = 0
        visibility_mask = torch.zeros(self.tokenizer.seq_len, dtype=torch.bool)
        token_idx = 0
        for kpt_idx, kpt in enumerate(keypoints):
            if token_idx >= len(token_labels):
                break
            if token_idx == 0 and token_labels[token_idx] != TokenType.coord.value:
                token_idx += 1
            if token_idx < len(token_labels) and token_labels[token_idx] == TokenType.coord.value:
                if visibility[kpt_idx] > 0:
                    visibility_mask[token_idx] = True
                token_idx += 1
        for i, label in enumerate(token_labels):
            if label == TokenType.eos.value:
                visibility_mask[i] = True
                break
        target_seq = self.tokenizer._padding(target_seq, [0, 0], dtype=torch.float32)
        token_labels = self.tokenizer._padding(token_labels, -1, dtype=torch.long)
        delta_x1 = [0]
        delta_y1 = [0]
        for polygon in quant_poly:
            delta_x = [p[0] - math.floor(p[0]) for p in polygon]
            delta_y = [p[1] - math.floor(p[1]) for p in polygon]
            delta_x1.extend(delta_x)
            delta_y1.extend(delta_y)
            delta_x1.append(0)
            delta_y1.append(0)
        delta_x1 = delta_x1[:-1]
        delta_y1 = delta_y1[:-1]
        delta_x1 = self.tokenizer._padding(delta_x1, 0, dtype=torch.float32)
        delta_y1 = self.tokenizer._padding(delta_y1, 0, dtype=torch.float32)
        delta_x2 = 1 - delta_x1
        delta_y2 = 1 - delta_y1
        target_polygon_labels = torch.full((self.tokenizer.seq_len,), -1, dtype=torch.long)
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
            'visibility_mask': visibility_mask,
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
    split_num = getattr(args, 'mp100_split', 1)
    root = Path(args.dataset_root) / "data"
    ann_file = Path(args.dataset_root).resolve() / "annotations" / f"mp100_split{split_num}_{image_set}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if image_set == 'train':
        import albumentations as A
        import inspect
        transform_list = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.6
            ),
        ]
        try:
            sig = inspect.signature(A.GaussNoise.__init__)
            if 'var_limit' in sig.parameters:
                transform_list.append(A.GaussNoise(var_limit=15.0, p=0.4))
            else:
                transform_list.append(A.GaussNoise(p=0.4))
        except Exception:
            pass
        transform_list.extend([
            A.GaussianBlur(
                blur_limit=(3, 5),
                sigma_limit=(0.1, 2.0),
                p=0.2
            ),
            A.Resize(height=512, width=512),
        ])
        transforms = A.Compose(transform_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
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