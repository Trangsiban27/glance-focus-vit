"""Shared image processing: geometry → FoV crop → colour constancy → views.

Returns unnormalized RGB in [0,1]. The model handles normalization and creates
its global/local views in the same coordinate system.

Front-end [A] (spec §5): when ``front_end`` is on, the square field-of-view is
cropped out (removing the endoscope's black border) and Shades-of-Gray colour
constancy is applied over the valid pixels, so images from different cameras
share a tone. It is parameter-free, label-free and identical for train and test.
"""
import random
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def fov_valid(image):
    """Boolean field-of-view mask: dark border removed, largest region kept."""
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255
    dark = gray <= .05
    seeds = np.zeros_like(dark)
    seeds[0] = dark[0]
    seeds[-1] = dark[-1]
    seeds[:, 0] = dark[:, 0]
    seeds[:, -1] = dark[:, -1]
    background = ndimage.binary_propagation(seeds, mask=dark)
    valid = ndimage.binary_closing(~background, structure=np.ones((3, 3)), border_value=1)
    labels, count = ndimage.label(valid)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        valid = labels == sizes.argmax()
    if not valid.any():
        valid = np.ones_like(dark, dtype=bool)
    return valid.astype(bool)


def fov_mask(image):
    """FoV mask as a float tensor [1,H,W] (kept for backward compatibility)."""
    return torch.from_numpy(fov_valid(image).astype(np.float32)).unsqueeze(0)


def _square_fov_box(valid):
    """Smallest in-bounds square covering the valid region: (x0,y0,x1,y1)."""
    height, width = valid.shape
    ys, xs = np.where(valid)
    if xs.size == 0:
        return 0, 0, width, height
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(min(max(round(cx - side / 2), 0), width - side))
    y0 = int(min(max(round(cy - side / 2), 0), height - side))
    return x0, y0, x0 + side, y0 + side


def shades_of_gray(image, valid=None, power=6):
    """Shades-of-Gray colour constancy; illuminant estimated over valid pixels."""
    arr = np.asarray(image, dtype=np.float32) / 255
    pixels = arr[valid] if valid is not None and valid.any() else arr.reshape(-1, 3)
    illuminant = np.mean(pixels ** power, axis=0) ** (1 / power)
    illuminant = np.where(illuminant < 1e-6, 1e-6, illuminant)
    arr = np.clip(arr * (illuminant.mean() / illuminant), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


class ImageProcessor:
    def __init__(self, hi_size=448, training=False, use_fov=True, front_end=True):
        if not isinstance(hi_size, int) or hi_size < 1:
            raise ValueError('hi_size must be a positive integer')
        self.hi_size = hi_size
        self.training = training
        self.use_fov = use_fov
        self.front_end = front_end

    def __call__(self, image):
        image = image.convert('RGB')
        if self.training:
            if random.random() < .5:
                image = TF.hflip(image)
            if random.random() < .5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.uniform(-15, 15),
                              interpolation=InterpolationMode.BILINEAR)
        valid = fov_valid(image) if (self.use_fov or self.front_end) else None
        if self.front_end:
            x0, y0, x1, y1 = _square_fov_box(valid)   # [A] FoV crop
            image = image.crop((x0, y0, x1, y1))
            valid = valid[y0:y1, x0:x1]
            image = shades_of_gray(image, valid)      # [A] colour constancy
        if self.use_fov:
            plane = valid if valid is not None else np.ones((image.height, image.width), dtype=bool)
            mask = torch.from_numpy(plane.astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.ones(1, image.height, image.width)
        hi = TF.resize(TF.to_tensor(image), [self.hi_size, self.hi_size], antialias=True)
        mask = F.interpolate(mask[None], size=(self.hi_size, self.hi_size), mode='area')[0]
        return hi, mask
