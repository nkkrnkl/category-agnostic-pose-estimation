"""
Simple image transforms without detectron2 dependency
"""
from PIL import Image
import cv2
import numpy as np


class Resize:
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BICUBIC):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL/cv2 interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = tuple(shape)
        self.interp = cv2.INTER_CUBIC  # Use cv2 interpolation

    def __call__(self, image):
        """
        Args:
            image: numpy array (H, W, C)
        Returns:
            dict with 'image' key containing resized image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = self.shape
        resized = cv2.resize(image, (w, h), interpolation=self.interp)

        return {'image': resized}


class ResizeAndPad:
    """Custom transform that resizes (preserving aspect ratio) and then pads to fixed size"""

    def __init__(self, target_size, pad_value=0, interp=Image.BICUBIC):
        """
        Args:
            target_size: (height, width) tuple
            pad_value: value to use for padding
            interp: interpolation method
        """
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.interp = cv2.INTER_CUBIC
        self.pad_value = pad_value

    def __call__(self, image):
        """
        Args:
            image: numpy array (H, W, C)
        Returns:
            dict with 'image' key containing transformed image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # Calculate scale preserving aspect ratio
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize preserving aspect ratio
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interp)

        # Create padded image
        padded = np.full((target_h, target_w, image.shape[2]), self.pad_value, dtype=image.dtype)

        # Calculate padding offsets (center the image)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2

        # Place resized image in center
        padded[top:top + new_h, left:left + new_w] = resized

        return {'image': padded}
