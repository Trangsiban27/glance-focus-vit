"""Shared image processing; geometry is applied before creating the view and FoV mask.

Returns unnormalized RGB in [0,1]. The model handles normalization and creates
its global/local views in the same coordinate system.
"""
import random
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def fov_mask(image):
    """Remove dark background connected to borders; retain largest valid region."""
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
        valid = np.ones_like(dark)
    return torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)


class ImageProcessor:
    def __init__(self, hi_size=448, training=False, use_fov=True):
        if not isinstance(hi_size, int) or hi_size < 1:
            raise ValueError('hi_size must be a positive integer')
        self.hi_size = hi_size
        self.training = training
        self.use_fov = use_fov

    def __call__(self, image):
        image = image.convert('RGB')
        if self.training:
            if random.random() < .5:
                image = TF.hflip(image)
            if random.random() < .5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.uniform(-15, 15),
                              interpolation=InterpolationMode.BILINEAR)
        mask = fov_mask(image) if self.use_fov else torch.ones(1, image.height, image.width)
        hi = TF.resize(TF.to_tensor(image), [self.hi_size, self.hi_size], antialias=True)
        mask = F.interpolate(mask[None], size=(self.hi_size, self.hi_size), mode='area')[0]
        return hi, mask
