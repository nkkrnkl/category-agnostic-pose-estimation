"""
Simple image transforms without detectron2 dependency.
"""
from PIL import Image
import cv2
import numpy as np
class Resize:
    """
    Resize image to a fixed target size.
    """
    def __init__(self, shape, interp=Image.BICUBIC):
        """
        Initialize resize transform.
        
        Args:
            shape: (h, w) tuple or int
            interp: Interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = tuple(shape)
        self.interp = cv2.INTER_CUBIC
    def __call__(self, image):
        """
        Resize image.
        
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
    """
    Custom transform that resizes and then pads to fixed size.
    """
    def __init__(self, target_size, pad_value=0, interp=Image.BICUBIC):
        """
        Initialize resize and pad transform.
        
        Args:
            target_size: (height, width) tuple
            pad_value: Value to use for padding
            interp: Interpolation method
        """
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.interp = cv2.INTER_CUBIC
        self.pad_value = pad_value
    def __call__(self, image):
        """
        Resize and pad image.
        
        Args:
            image: numpy array (H, W, C)
        
        Returns:
            dict with 'image' key containing transformed image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interp)
        padded = np.full((target_h, target_w, image.shape[2]), self.pad_value, dtype=image.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        padded[top:top + new_h, left:left + new_w] = resized
        return {'image': padded}