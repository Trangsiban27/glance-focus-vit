"""Colab-ready Swin-Tiny image classifier.

Copy this whole file into a Colab cell, or save it as ``model_swin_tiny.py``
and import ``SwinTinyClassifier``. It installs ``timm`` if needed. The default
backbone is ImageNet-22K pretrained and fine-tuned on ImageNet-1K.

The forward interface follows this project's evaluators:
    output = model(hi, mask=None, selector=None)
    logits = output['logits']

Input ``hi`` is RGB float tensor [B,3,H,W] in [0,1]. This is a global-view
baseline; it does not implement Glance-Focus ROI selection or fusion.
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


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank update."""
    def __init__(self, base, rank=8, alpha=16.):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRALinear wraps nn.Linear only')
        if rank < 1 or alpha <= 0:
            raise ValueError('rank and alpha must be positive')
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scaling


class AdapterBlock(nn.Module):
    """Frozen Swin block followed by a zero-initialized residual adapter."""
    def __init__(self, block, dim, hidden=64):
        super().__init__()
        for parameter in block.parameters():
            parameter.requires_grad_(False)
        self.block = block
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.block(x)
        return x + self.up(F.gelu(self.down(self.norm(x))))


class CosineHead(nn.Module):
    """Length-normalized classifier: logits = tau * cosine(feature, weight)."""
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
    """Exponential moving average copy for evaluation checkpoints."""
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


class SwinTinyClassifier(nn.Module):
    """Swin-Tiny global classifier with a pipeline-compatible forward output.

    ``tuning`` can be ``lora``, ``adapter``, ``full`` or ``frozen``. LoRA and
    adapter modes freeze the pretrained Swin backbone while training the
    injected small modules and classification head.
    """
    DEFAULT_BACKBONE = 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'

    def __init__(self, num_classes, backbone=DEFAULT_BACKBONE, tuning='lora',
                 lora_rank=8, lora_alpha=16., adapter_hidden=64,
                 cosine_head=True, tau=16., learnable_tau=False,
                 drop_rate=.1, pretrained=True):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if tuning not in {'lora', 'adapter', 'full', 'frozen'}:
            raise ValueError('tuning must be lora, adapter, full or frozen')
        if lora_rank < 1 or adapter_hidden < 1:
            raise ValueError('lora_rank and adapter_hidden must be >= 1')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be finite and positive')
        if not 0 <= drop_rate < 1:
            raise ValueError('drop_rate must be in [0,1)')

        self.swin = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.fusion = 'global'
        self.use_fov = False
        self.tuning = tuning
        self.input_size = tuple(self.swin.pretrained_cfg.get('input_size', (3, 224, 224))[-2:])
        mean = self.swin.pretrained_cfg.get('mean', (0.485, 0.456, 0.406))
        std = self.swin.pretrained_cfg.get('std', (0.229, 0.224, 0.225))
        self.register_buffer('pixel_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.config = {
            'model_type': 'swin_tiny',
            'num_classes': num_classes,
            'backbone': backbone,
            'tuning': tuning,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'adapter_hidden': adapter_hidden,
            'cosine_head': cosine_head,
            'tau': tau,
            'learnable_tau': learnable_tau,
            'drop_rate': drop_rate,
        }

        self._apply_tuning(tuning, lora_rank, lora_alpha, adapter_hidden)
        feature_dim = self.swin.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.head = (CosineHead(feature_dim, num_classes, tau, learnable_tau)
                     if cosine_head else nn.Linear(feature_dim, num_classes))

    @classmethod
    def from_config(cls, config, pretrained=False):
        accepted = {
            'num_classes', 'backbone', 'tuning', 'lora_rank', 'lora_alpha',
            'adapter_hidden', 'cosine_head', 'tau', 'learnable_tau', 'drop_rate',
        }
        return cls(**{key: value for key, value in config.items() if key in accepted},
                   pretrained=pretrained)

    def _apply_tuning(self, tuning, rank, lora_alpha, adapter_hidden):
        if tuning == 'full':
            return
        for parameter in self.swin.parameters():
            parameter.requires_grad_(False)
        if tuning == 'frozen':
            return

        for stage_index, stage in enumerate(self.swin.layers):
            for block_index, block in enumerate(list(stage.blocks)):
                if tuning == 'lora':
                    if not (hasattr(block.attn, 'qkv') and hasattr(block.mlp, 'fc1')
                            and hasattr(block.mlp, 'fc2')):
                        raise TypeError('Unexpected timm Swin block layout for LoRA')
                    block.attn.qkv = LoRALinear(block.attn.qkv, rank, lora_alpha)
                    block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, lora_alpha)
                    block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, lora_alpha)
                else:
                    dim = getattr(block, 'dim', None)
                    if dim is None:
                        base_dim = getattr(self.swin, 'embed_dim', 96)
                        dim = base_dim * (2 ** stage_index)
                    stage.blocks[block_index] = AdapterBlock(block, dim, adapter_hidden)

    def forward(self, hi, mask=None, selector=None):
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W] in [0,1]')
        x = F.interpolate(hi, size=self.input_size, mode='bicubic',
                           align_corners=False, antialias=True)
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.swin(x)
        logits = self.head(self.dropout(features))
        return {
            'logits': logits,
            'boxes': None,
            'gate': None,
            'aux_g': None,
            'aux_l': None,
            'attention': None,
        }

    def loss(self, output, target, criterion):
        return criterion(output['logits'], target)

