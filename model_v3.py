"""Glance-Focus ViT v3: v2 architecture with v3 training defaults.

The model structure, attention maps, ROI selection and Grad-CAM behavior are
inherited from :mod:`model_v2`. Version 3 identifies checkpoints trained with
the updated optimizer and ROI recipe, and initializes the fusion scale ``s``
to a small positive value.
"""
import math

import torch

from model_v2 import GlanceFocusViTV2, ModelEMA


class GlanceFocusViTV3(GlanceFocusViTV2):
    """V2 model with v3 configuration and a nonzero initial fusion scale."""

    def __init__(self, *args, s_init=0.1, branch_drop=0.1, roi_coverage=0.45,
                 **kwargs):
        if not math.isfinite(s_init):
            raise ValueError('s_init must be finite')
        super().__init__(*args, branch_drop=branch_drop,
                         roi_coverage=roi_coverage, **kwargs)
        self.config.update(roi_version='v3', s_init=float(s_init))
        if hasattr(self, 's'):
            with torch.no_grad():
                self.s.fill_(float(s_init))

    @classmethod
    def from_config(cls, config, pretrained=False):
        config = {key: value for key, value in config.items()
                  if key != 'roi_version'}
        return cls(**config, pretrained=pretrained)


__all__ = ['GlanceFocusViTV3', 'ModelEMA']
