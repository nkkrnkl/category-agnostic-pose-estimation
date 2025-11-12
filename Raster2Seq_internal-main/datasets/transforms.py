from PIL import Image
from detectron2.data import transforms as T

class Resize(T.Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BICUBIC):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return T.ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


# Custom transform that resizes and then pads to fixed size
class ResizeAndPad(T.Augmentation):
    def __init__(self, target_size, pad_value=0, interp=Image.BICUBIC):
        super().__init__()
        self.target_size = target_size  # (height, width)
        self.interp = interp
        self.pad_value = pad_value
        
    def get_transform(self, img):
        h, w = img.shape[:2]
        scale = min(self.target_size[0]/h, self.target_size[1]/w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # First resize preserving aspect ratio
        resize_t = T.ResizeTransform(h, w, new_h, new_w, self.interp)
        
        # Then pad to target size
        pad_h, pad_w = self.target_size[0] - new_h, self.target_size[1] - new_w
        top = pad_h // 2
        left = pad_w // 2
        pad_t = T.PadTransform(left, top, pad_w - left, pad_h - top, new_h, new_w, 
                               pad_value=self.pad_value)
        
        return T.TransformList([resize_t, pad_t])