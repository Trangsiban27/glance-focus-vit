"""Colab-ready Glance→Focus ViT v3 (V2 architecture and Grad-CAM ROI).

This file is standalone: copy its full contents into a Colab code cell, or save
it as ``model_v3.py`` beside the notebook and import ``GlanceFocusViTV3``.
It installs ``timm`` if that package is missing. The pretrained backbone still
needs internet access the first time it is created.

Same architecture and parameters as GlanceFocusViT (model.py): [A] front-end
(upstream), [B] frozen ViT-B/16 + LoRA, [D] gated fusion + branch dropout + aux
heads, [E] cosine head, EMA weights. The ONLY change is how the local region is
chosen — ROI v2 adds, over plain attention rollout:

  ① Class-discriminative maps instead of class-agnostic rollout
     - attention_method='gradcam'   : Grad-CAM on the last block, predicted class
     - attention_method='gradrollout': attention×gradient rollout (Chefer 2021)
     (plus the original 'rollout' and 'last'). Maps use the model's PREDICTED
     class — never the true label — so region choice stays leakage-safe (spec §11).
  ② Outlier/register-token suppression (Darcet 2024): winsorise the map to kill
     spurious background peaks before picking the box (suppress_outliers).
  ③ Adaptive crop scale (region_method='adaptive'): smallest window covering a
     target fraction of attention mass, so the zoom matches the lesion size,
     instead of a fixed window of side crop_ratio·min(H,W).

ROI selection stays non-parametric and stop-gradient; grad maps run on a
separate frozen selector (EMA or frozen E1), so no gradient leaks into training.
The v3 model keeps the V2 state structure and marks checkpoints with
config['roi_version']='v3'. The gate scale ``s`` starts at ``s_init`` (default .1).
"""
import copy
import importlib.util
import math
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if importlib.util.find_spec('timm') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm'])
import timm

GRAD_METHODS = {'gradcam', 'gradrollout'}
ATTENTION_METHODS = {'rollout', 'last'} | GRAD_METHODS
REGION_METHODS = {'window', 'threshold', 'adaptive'}


class LoRALinear(nn.Module):
    """[B] Frozen Linear with a trainable low-rank update W' = W + (B·A)·scaling."""
    def __init__(self, base, rank=8, alpha=16.):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRALinear wraps nn.Linear only')
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scaling


class Adapter(nn.Module):
    """[B] Frozen transformer block + residual bottleneck adapter."""
    def __init__(self, block, dim, hidden=64):
        super().__init__()
        self.block = block
        for p in self.block.parameters():
            p.requires_grad_(False)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.block(x)
        return x + self.up(F.gelu(self.down(self.ln(x))))


class CosineHead(nn.Module):
    """[E] Length-normalised classifier: logits_c = tau · cos(w_c, z)."""
    def __init__(self, in_dim, num_classes, tau=16., learnable_tau=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        else:
            self.log_tau = None
            self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, z):
        tau = self.log_tau.exp() if self.log_tau is not None else self.tau
        return tau * F.linear(F.normalize(z, dim=-1), F.normalize(self.weight, dim=-1))


class ModelEMA:
    """Exponential moving average of weights; eval + frozen, used as selector."""
    def __init__(self, model, decay=.999):
        if not 0 < decay < 1:
            raise ValueError('decay must be in (0,1)')
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.is_floating_point():
                e.mul_(d).add_(m.detach().to(e.dtype), alpha=1 - d)
            else:
                e.copy_(m)


class GlanceFocusViTV3(nn.Module):
    """Shared ViT-B/16 with improved (class-aware, adaptive) ROI selection.

    forward(hi, mask=None, selector=None) returns the same dict as GlanceFocusViT.
    Grad-based attention methods require a separate frozen ``selector`` (EMA or
    frozen E1); its predicted-class gradients build the map, and the box is
    detached so no gradient reaches training.
    """
    def __init__(self, num_classes, fusion='gate', crop_ratio=.5, use_fov=True,
                 backbone='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                 gate_hidden=256, drop_rate=.1, region_method='adaptive',
                 attention_method='gradcam', alpha=1., pretrained=True,
                 tuning='lora', lora_rank=8, lora_alpha=16., adapter_hidden=64,
                 cosine_head=True, tau=16., learnable_tau=False,
                 branch_drop=.1, aux_weight=.3, view_embed=False,
                 suppress_outliers=True, roi_coverage=.45, legacy=False, s_init=.1):
        super().__init__()
        if not math.isfinite(s_init):
            raise ValueError('s_init must be finite')
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if region_method not in REGION_METHODS:
            raise ValueError(f'region_method must be one of {sorted(REGION_METHODS)}')
        if attention_method not in ATTENTION_METHODS:
            raise ValueError(f'attention_method must be one of {sorted(ATTENTION_METHODS)}')
        if tuning not in {'lora', 'adapter', 'full', 'frozen'}:
            raise ValueError('tuning must be lora, adapter, full or frozen')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be finite and positive')
        if lora_rank < 1 or adapter_hidden < 1:
            raise ValueError('lora_rank and adapter_hidden must be >= 1')
        if gate_hidden < 1 or not 0 <= drop_rate < 1:
            raise ValueError('Require gate_hidden >= 1 and drop_rate in [0,1)')
        if not 0 <= branch_drop < 1:
            raise ValueError('branch_drop must be in [0,1)')
        if aux_weight < 0:
            raise ValueError('aux_weight must be >= 0')
        if not 0 < roi_coverage <= 1:
            raise ValueError('roi_coverage must be in (0,1]')
        if fusion not in {'global', 'local', 'concat', 'gate'}:
            raise ValueError('Unknown fusion mode')
        if not 0 < crop_ratio <= 1:
            raise ValueError('crop_ratio must be in (0,1]')
        self.legacy = bool(legacy)
        self.config = dict(num_classes=num_classes, fusion=fusion, crop_ratio=crop_ratio,
                           use_fov=use_fov, backbone=backbone, gate_hidden=gate_hidden,
                           drop_rate=drop_rate, region_method=region_method,
                           attention_method=attention_method, alpha=alpha, tuning=tuning,
                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                           adapter_hidden=adapter_hidden, cosine_head=cosine_head, tau=tau,
                           learnable_tau=learnable_tau, branch_drop=branch_drop,
                           aux_weight=aux_weight, view_embed=view_embed,
                           suppress_outliers=suppress_outliers, roi_coverage=roi_coverage,
                           s_init=s_init, roi_version='v3')
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.vit.num_features
        self.fusion, self.crop_ratio, self.use_fov = fusion, crop_ratio, use_fov
        self.region_method, self.attention_method, self.alpha = region_method, attention_method, alpha
        self.branch_drop, self.aux_weight, self.view_embed = branch_drop, aux_weight, view_embed
        self.suppress_outliers, self.roi_coverage = suppress_outliers, roi_coverage
        self._mode = None          # drives capture: rollout|last|gradrollout|gradcam
        self._rollout = None
        self._last = None
        self._attn = []            # raw attention tensors (gradrollout)
        self._feat = None          # pre-attention tokens from the final block (gradcam)
        for block in self.vit.blocks:
            block.attn.fused_attn = False
            block.attn.attn_drop.register_forward_pre_hook(self._attn_hook)
        self._apply_tuning(tuning, lora_rank, lora_alpha, adapter_hidden)
        last_block = self.vit.blocks[-1]
        while isinstance(last_block, Adapter):
            last_block = last_block.block
        # Capture tokens before the final self-attention. The classifier reads
        # the CLS token, so patch-token gradients are only available here,
        # before this attention layer mixes patch information into CLS.
        last_block.norm1.register_forward_hook(self._feat_hook)
        if view_embed:
            self.view_g = nn.Parameter(torch.zeros(d))
            self.view_l = nn.Parameter(torch.zeros(d))
        if fusion == 'gate':
            self.ln_g, self.ln_l = nn.LayerNorm(d), nn.LayerNorm(d)
            self.gate = nn.Sequential(nn.Linear(2 * d, gate_hidden), nn.GELU(),
                                      nn.Linear(gate_hidden, d), nn.Sigmoid())
            self.s = nn.Parameter(torch.full((1,), float(s_init)))
        if fusion == 'concat':
            self.ln_cat = nn.LayerNorm(2 * d)
        feat_dim = 2 * d if fusion == 'concat' else d
        self.drop = nn.Dropout(drop_rate)
        self.head = (CosineHead(feat_dim, num_classes, tau, learnable_tau)
                     if cosine_head else nn.Linear(feat_dim, num_classes))
        if fusion in {'concat', 'gate'}:
            self.aux_g = nn.Linear(d, num_classes)
            self.aux_l = nn.Linear(d, num_classes)

    @classmethod
    def from_config(cls, config, pretrained=False):
        config = {k: v for k, v in config.items() if k != 'roi_version'}
        return cls(**config, pretrained=pretrained)

    def _apply_tuning(self, tuning, rank, lora_alpha, hidden):
        if tuning == 'full':
            return
        for p in self.vit.parameters():
            p.requires_grad_(False)
        if tuning == 'frozen':
            return
        if tuning == 'lora':
            for block in self.vit.blocks:
                block.attn.qkv = LoRALinear(block.attn.qkv, rank, lora_alpha)
                block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, lora_alpha)
                block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, lora_alpha)
        else:
            blocks = self.vit.blocks
            for i in range(len(blocks)):
                blocks[i] = Adapter(blocks[i], self.vit.num_features, hidden)

    def _attn_hook(self, module, inputs):
        if self._mode in (None, 'gradcam'):
            return
        a = inputs[0]
        if self._mode == 'gradrollout':
            self._attn.append(a)                 # keep in the graph for gradients
            return
        am = a.detach().float().mean(1)          # rollout / last: stream, detached
        self._last = am[:, 0, self.vit.num_prefix_tokens:]
        if self._mode == 'rollout':
            am = .5 * am + .5 * torch.eye(am.shape[-1], device=am.device)[None]
            am = am / am.sum(-1, keepdim=True)
            self._rollout = am if self._rollout is None else am @ self._rollout

    def _feat_hook(self, module, inputs, output):
        if self._mode == 'gradcam':
            self._feat = output

    def _grid(self, n):
        g = math.isqrt(n)
        if g * g != n:
            raise ValueError('Backbone must have a square patch grid')
        return g

    def encode_map(self, x):
        """Rollout/last map (no gradients). Returns (z, map[B,1,g,g])."""
        self._mode = self.attention_method
        self._rollout = self._last = None
        try:
            z = self.vit(x)
        finally:
            self._mode = None
        m = (self._rollout[:, 0, self.vit.num_prefix_tokens:]
             if self.attention_method == 'rollout' else self._last)
        self._rollout = self._last = None
        g = self._grid(m.shape[-1])
        return z, m.reshape(-1, 1, g, g)

    def _grad_map(self, x):
        """Class-discriminative map (gradcam / gradrollout). Returns map[B,1,g,g]."""
        prefix = self.vit.num_prefix_tokens
        self._mode = self.attention_method
        self._attn = []
        self._feat = None
        x = x.detach().requires_grad_(True)
        try:
            with torch.enable_grad():
                z = self.vit(x)
                logits = self.head(z)
                pred = logits.argmax(1)
                score = logits.gather(1, pred[:, None]).sum()
                if self.attention_method == 'gradcam':
                    feat = self._feat
                    if feat is None:
                        raise RuntimeError('Grad-CAM did not capture the final-block features')
                    grad, = torch.autograd.grad(score, feat)
                    weight = grad[:, prefix:, :].mean(1, keepdim=True)
                    cam = (weight * feat[:, prefix:, :]).sum(-1).clamp(min=0)
                else:
                    rollout = None
                    attn_grads = torch.autograd.grad(score, tuple(self._attn))
                    for a, grad in zip(self._attn, attn_grads):
                        cam_l = (a * grad).clamp(min=0).mean(1)             # [B,N,N]
                        cam_l = cam_l + torch.eye(cam_l.shape[-1], device=cam_l.device)[None]
                        cam_l = cam_l / cam_l.sum(-1, keepdim=True).clamp(min=1e-6)
                        rollout = cam_l if rollout is None else cam_l @ rollout
                    cam = rollout[:, 0, prefix:].clamp(min=0)
        finally:
            self._mode = None
            self._attn = []
            self._feat = None
        g = self._grid(cam.shape[-1])
        return cam.detach().reshape(-1, 1, g, g)

    def region_map(self, x):
        """Raw ROI map for this model's attention_method (used as the selector)."""
        if self.attention_method in GRAD_METHODS:
            return self._grad_map(x)
        with torch.no_grad():
            _, m = self.encode_map(x)
        return m

    @torch.no_grad()
    def select_boxes(self, m, mask, height, width):
        if self.use_fov:
            m = m * F.adaptive_avg_pool2d(mask, m.shape[-2:])
        kernel = m.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None] / 16
        m = F.conv2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), kernel)
        if self.suppress_outliers:                 # ② winsorise spurious peaks
            cap = m.flatten(1).quantile(.99, dim=1).clamp(min=1e-6)[:, None, None, None]
            m = torch.minimum(m, cap)
        if self.use_fov:
            m = m * F.adaptive_avg_pool2d(mask, m.shape[-2:])
        m = m.clamp(min=0)
        boxes = []
        for item in m:
            side = self.crop_ratio * min(height, width)
            dense = F.interpolate(item[None], (height, width), mode='bilinear', align_corners=False)[0, 0]
            mass = float(dense.sum())
            if not math.isfinite(mass) or mass <= 1e-8:
                # A zero Grad-CAM map has no meaningful argmax: integral-window
                # argmax would otherwise tie everywhere and select the top-left.
                cx, cy = width / 2, height / 2
                if self.region_method == 'adaptive':
                    side = max(1., round(.3 * min(height, width)))
            elif self.region_method == 'threshold':
                try:
                    from scipy import ndimage
                except ImportError as exc:
                    raise ImportError(
                        "region_method='threshold' needs scipy. Run "
                        "`%pip install -q scipy` in Colab, then rerun this cell."
                    ) from exc
                arr = item[0].cpu().numpy()
                labels, count = ndimage.label(arr > self.alpha * arr.mean())
                if count:
                    sizes = np.bincount(labels.ravel()); sizes[0] = 0
                    ys, xs = np.where(labels == sizes.argmax())
                    cx = (xs.min()+xs.max()+1)/2 / arr.shape[1]*width
                    cy = (ys.min()+ys.max()+1)/2 / arr.shape[0]*height
                    side = max((xs.max()-xs.min()+1)/arr.shape[1]*width,
                               (ys.max()-ys.min()+1)/arr.shape[0]*height)
                else:
                    cx, cy = width/2, height/2
            elif self.region_method == 'adaptive':   # ③ smallest window covering roi_coverage mass
                cx, cy, side = self._adaptive_window(dense, height, width)
            else:
                k = max(1, round(side))
                cx, cy, side = self._best_window(dense, k)
            if self.training:
                cx += random.uniform(-.1, .1)*side
                cy += random.uniform(-.1, .1)*side
                side *= random.uniform(.9, 1.1)
            side = min(max(side, 1.), height, width)
            x = min(max(cx-side/2, 0.), width-side)
            y = min(max(cy-side/2, 0.), height-side)
            boxes.append([x, y, x+side, y+side])
        return m.new_tensor(boxes), m

    @staticmethod
    def _best_window(dense, k):
        """Max-sum k×k window via integral image. Returns (cx, cy, side)."""
        integral = F.pad(dense.cumsum(0).cumsum(1), (1, 0, 1, 0))
        sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
        index = int(sums.argmax())
        y, x = divmod(index, sums.shape[1])
        return x + k/2, y + k/2, float(k)

    def _adaptive_window(self, dense, height, width):
        """Smallest square whose best window holds >= roi_coverage of the mass."""
        integral = F.pad(dense.cumsum(0).cumsum(1), (1, 0, 1, 0))
        total = float(dense.sum().clamp(min=1e-6))
        smallest = int(round(.3 * min(height, width)))
        largest = int(round(.8 * min(height, width)))
        best = None
        for k in torch.linspace(smallest, largest, 8).round().int().tolist():
            k = max(1, min(k, min(height, width)))
            sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
            peak = float(sums.max())
            index = int(sums.argmax())
            y, x = divmod(index, sums.shape[1])
            if best is None:
                best = (x + k/2, y + k/2, float(k))
            if peak >= self.roi_coverage * total:
                return x + k/2, y + k/2, float(k)
        return best

    def forward(self, hi, mask=None, selector=None):
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W]')
        if self.use_fov and self.fusion != 'global':
            if mask is None or mask.shape != (hi.shape[0], 1, *hi.shape[-2:]):
                raise ValueError('use_fov requires mask [B,1,H,W] matching hi')
        if self.fusion == 'local' and selector is None:
            raise ValueError('E2/local requires a frozen selector (e.g. a frozen E1 or EMA copy)')
        if (self.training and self.attention_method in GRAD_METHODS
                and self.fusion != 'global' and selector is None):
            raise ValueError('training grad ROI needs a separate frozen selector (EMA/E1) to avoid gradient leakage')
        if selector is not None:
            if selector is self:
                raise ValueError('Selector must be a separate model, not self')
            if selector.training:
                raise ValueError('Selector must be in eval mode')
            if any(p.requires_grad for p in selector.parameters()):
                raise ValueError('Selector parameters must be frozen')
        global_view = F.interpolate(hi, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        global_view = (global_view - .5) / .5
        if self.fusion == 'global':
            zg = self.vit(global_view)
            if self.view_embed:
                zg = zg + self.view_g
            logits = self.head(self.drop(zg))
            return {'logits': logits, 'boxes': None, 'gate': None,
                    'aux_g': None, 'aux_l': None, 'attention': None}
        if selector is not None:
            m = selector.region_map(global_view)
            zg = self.vit(global_view) if self.fusion != 'local' else None
        elif self.attention_method in GRAD_METHODS:
            # Eval-time self-selection (training path is blocked by the guard above).
            m = self.region_map(global_view)
            zg = self.vit(global_view) if self.fusion != 'local' else None
        else:
            zg, m = self.encode_map(global_view)
        boxes, m = self.select_boxes(m, mask, *hi.shape[-2:])
        t = (torch.arange(224, device=hi.device, dtype=hi.dtype)+.5)/224
        xs = boxes[:, 0, None] + (boxes[:, 2]-boxes[:, 0])[:, None]*t
        ys = boxes[:, 1, None] + (boxes[:, 3]-boxes[:, 1])[:, None]*t
        gx = (2*xs/hi.shape[-1]-1)[:, None, :].expand(-1, 224, -1)
        gy = (2*ys/hi.shape[-2]-1)[:, :, None].expand(-1, -1, 224)
        local = F.grid_sample(hi, torch.stack([gx, gy], -1), align_corners=False, padding_mode='border')
        zl = self.vit((local-.5)/.5)
        if self.view_embed:
            if zg is not None:
                zg = zg + self.view_g
            zl = zl + self.view_l
        gate = aux_g = aux_l = None
        if self.fusion == 'local':
            z = zl
        else:
            if self.training and self.branch_drop > 0:
                keep = torch.rand(zl.shape[0], 1, device=zl.device, dtype=zl.dtype) >= self.branch_drop
                zl_used = zl * keep
            else:
                zl_used = zl
            if self.fusion == 'concat':
                z = self.ln_cat(torch.cat([zg, zl_used], -1))
            else:
                gate = self.gate(torch.cat([self.ln_g(zg), self.ln_l(zl_used)], -1))
                z = zg + self.s*gate*zl_used
            if self.training:
                aux_g, aux_l = self.aux_g(zg), self.aux_l(zl)
        logits = self.head(self.drop(z))
        return {'logits': logits, 'boxes': boxes, 'gate': gate,
                'aux_g': aux_g, 'aux_l': aux_l, 'attention': m}

    def loss(self, out, target, criterion):
        """[D] CE(logits) + aux_weight·(CE(aux_g) + CE(aux_l))."""
        total = criterion(out['logits'], target)
        if out.get('aux_g') is not None and out.get('aux_l') is not None:
            total = total + self.aux_weight * (criterion(out['aux_g'], target)
                                               + criterion(out['aux_l'], target))
        return total
