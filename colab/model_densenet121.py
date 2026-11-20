"""Colab-ready DenseNet-121 global image classifier.

Copy this file into a Colab cell or save it as ``model_densenet121.py`` and
import ``DenseNet121Classifier``. It installs ``timm`` if missing and uses the
ImageNet-1K pretrained DenseNet-121 weights by default.

Input: RGB float tensor [B,3,H,W] in [0,1]. The forward result is a dictionary
with ``logits`` and the auxiliary keys expected by this project's predictors.
This is a global-view classifier; it does not select or fuse ROI crops.
"""
import copy
import importlib.util
import math
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

if importlib.util.find_spec('timm') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm'])
import timm


class CosineHead(nn.Module):
    """Length-normalized linear classifier."""
    def __init__(self, in_dim, num_classes, tau=16., learnable_tau=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        else:
            self.log_tau = None
            self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, x):
        tau = self.log_tau.exp() if self.log_tau is not None else self.tau
        return tau * F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))


class ModelEMA:
    """Exponential moving average copy for inference."""
    def __init__(self, model, decay=.999):
        if not 0 < decay < 1:
            raise ValueError('decay must be in (0,1)')
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_value, model_value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value.detach().to(ema_value.dtype),
                                                alpha=1 - self.decay)
            else:
                ema_value.copy_(model_value)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 classifier with ``full``, ``last_stage`` or ``frozen`` tuning."""
    DEFAULT_BACKBONE = 'densenet121.ra_in1k'

    def __init__(self, num_classes, backbone=DEFAULT_BACKBONE, tuning='last_stage',
                 cosine_head=True, tau=16., learnable_tau=False,
                 drop_rate=.1, pretrained=True):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if tuning not in {'full', 'last_stage', 'frozen'}:
            raise ValueError('tuning must be full, last_stage or frozen')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be finite and positive')
        if not 0 <= drop_rate < 1:
            raise ValueError('drop_rate must be in [0,1)')

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.fusion = 'global'
        self.use_fov = False
        self.tuning = tuning
        self.input_size = tuple(self.backbone.pretrained_cfg.get('input_size', (3, 224, 224))[-2:])
        mean = self.backbone.pretrained_cfg.get('mean', (0.485, 0.456, 0.406))
        std = self.backbone.pretrained_cfg.get('std', (0.229, 0.224, 0.225))
        self.register_buffer('pixel_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.config = {
            'model_type': 'densenet121',
            'num_classes': num_classes,
            'backbone': backbone,
            'tuning': tuning,
            'cosine_head': cosine_head,
            'tau': tau,
            'learnable_tau': learnable_tau,
            'drop_rate': drop_rate,
        }

        self._configure_tuning()
        self.dropout = nn.Dropout(drop_rate)
        self.head = (CosineHead(self.backbone.num_features, num_classes, tau, learnable_tau)
                     if cosine_head else nn.Linear(self.backbone.num_features, num_classes))

    @classmethod
    def from_config(cls, config, pretrained=False):
        accepted = {'num_classes', 'backbone', 'tuning', 'cosine_head', 'tau',
                    'learnable_tau', 'drop_rate'}
        return cls(**{key: value for key, value in config.items() if key in accepted},
                   pretrained=pretrained)

    def _configure_tuning(self):
        if self.tuning == 'full':
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.tuning == 'last_stage':
            last_stage = self.backbone.features.denseblock4
            final_norm = self.backbone.features.norm5
            for module in (last_stage, final_norm):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.tuning != 'full':
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    if not any(parameter.requires_grad for parameter in module.parameters()):
                        module.eval()
        return self

    def forward(self, hi, mask=None, selector=None):
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W] in [0,1]')
        x = F.interpolate(hi, size=self.input_size, mode='bicubic',
                           align_corners=False, antialias=True)
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.backbone(x)
        logits = self.head(self.dropout(features))
        return {'logits': logits, 'boxes': None, 'gate': None,
                'aux_g': None, 'aux_l': None, 'attention': None}

    def loss(self, output, target, criterion):
        return criterion(output['logits'], target)

