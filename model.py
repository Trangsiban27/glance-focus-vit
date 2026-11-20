"""Glance→Focus ViT — integrated spec (spec/Glance_Focus_ViT_model_spec.md, 11/11/2026).

Modules implemented here (spec §4–5):
  [B] ViT-B/16 frozen + parameter-efficient tuning (LoRA or bottleneck adapter).
  [C] Region selector (attention rollout × FoV); accepts an external frozen
      selector so region choice can use smoothed EMA/E1 attention.
  [D] Gated fusion + per-sample local-branch dropout + auxiliary per-branch heads.
  [E] Cosine classification head (length-normalised, domain-robust logits).
  ModelEMA: weight EMA, used as the inference model and as the selector.

Front-end [A] (FoV crop + Shades-of-Gray) is upstream preprocessing and lives
in the data pipeline: this module expects ``hi`` to already be the normalised
high-resolution image I' in [0,1].

Fusion modes: global (E1); local (E2, needs a frozen selector); concat (E3);
gate (E4). Backbone CLS dimension 768; pretrained normalisation mean=std=0.5.
Attention rollout accumulates mean-head attention with residuals in layer order
under no_grad; only the global pass captures attention. Local crops come from
``hi`` via bilinear grid_sample, never from the 224x224 global view.

Parameter-efficient tuning keeps the pretrained backbone frozen (tuning='lora'
or 'adapter', ~1–2M trainable params); tuning='full' fine-tunes everything and
'frozen' trains only the heads/fusion. LoRA B and the adapter up-projection are
zero-initialised, so a freshly built model starts exactly at the frozen
backbone; the gate scale s starts at 0, so in eval E4 reduces to head(z_g).

Example:
    model = GlanceFocusViT(num_classes=2, fusion='gate')
    ema = ModelEMA(model)
    out = model(hi, fov_mask, selector=ema.module)   # train: dict incl aux_*
    prob = out['logits'].softmax(-1)[:, 1]            # P(Abnormal)
"""
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
try:
    import timm
except ModuleNotFoundError as exc:
    if exc.name != 'timm':
        raise
    raise ImportError(
        "GlanceFocusViT needs timm. In Colab run `%pip install -q timm` before importing model.py."
    ) from exc


class LoRALinear(nn.Module):
    """[B] Frozen Linear with a trainable low-rank update W' = W + (B·A)·scaling.

    B is zero-initialised so the module equals the frozen base at init.
    """
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
    """[B] Frozen transformer block + residual bottleneck adapter.

    h ← block(h); h ← h + W_up · GELU(W_down · LN(h)); W_up zero-initialised.
    """
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
    """Exponential moving average of weights (spec §5, "EMA trọng số").

    The EMA module is eval + frozen, so it can be passed directly as the
    ``selector`` for region choice and used as the inference model. Call
    ``update(model)`` after each optimiser step.
    """
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


class GlanceFocusViT(nn.Module):
    """Shared ViT-B/16 with global, local, concatenation or gated fusion.

    forward(hi, mask=None, selector=None):
        hi: float RGB [B,3,H,W] in [0,1], the high-resolution I' before
            pretrained normalisation.
        mask: FoV [B,1,H,W]; required for two-branch modes when use_fov=True.
        selector: a *separate* frozen, eval-mode model whose attention picks the
            region (a frozen E1 for E2/local, or an EMA copy for E4). Optional
            for concat/gate (falls back to this model's own attention).

    Returns a dict: logits [B,C]; boxes [B,4] in hi pixel-edge coords
    (x1,y1,x2,y2); gate [B,D] (gate mode only); aux_g / aux_l [B,C] (two-branch
    modes, training only); attention (filtered 14x14 map). Global mode returns
    boxes=gate=aux_*=attention=None. Region selection is detached,
    class-independent and never uses labels.
    """
    def __init__(self, num_classes, fusion='gate', crop_ratio=.5, use_fov=True,
                 backbone='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                 gate_hidden=256, drop_rate=.1, region_method='window',
                 attention_method='rollout', alpha=1., pretrained=True,
                 tuning='lora', lora_rank=8, lora_alpha=16., adapter_hidden=64,
                 cosine_head=True, tau=16., learnable_tau=False,
                 branch_drop=.2, aux_weight=.3, view_embed=False, legacy=False):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if region_method not in {'window', 'threshold'}:
            raise ValueError('region_method must be window or threshold')
        if attention_method not in {'rollout', 'last'}:
            raise ValueError('attention_method must be rollout or last')
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
        if fusion not in {'global', 'local', 'concat', 'gate'}:
            raise ValueError('Unknown fusion mode')
        if not 0 < crop_ratio <= 1:
            raise ValueError('crop_ratio must be in (0,1]')
        legacy_config = dict(num_classes=num_classes, fusion=fusion, crop_ratio=crop_ratio,
                             use_fov=use_fov, backbone=backbone, gate_hidden=gate_hidden,
                             drop_rate=drop_rate, region_method=region_method,
                             attention_method=attention_method, alpha=alpha)
        self.legacy = bool(legacy)
        self.config = (legacy_config if self.legacy else
                       dict(**legacy_config, tuning=tuning, lora_rank=lora_rank,
                            lora_alpha=lora_alpha, adapter_hidden=adapter_hidden,
                            cosine_head=cosine_head, tau=tau, learnable_tau=learnable_tau,
                            branch_drop=branch_drop, aux_weight=aux_weight,
                            view_embed=view_embed))
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.vit.num_features
        self.fusion, self.crop_ratio, self.use_fov = fusion, crop_ratio, use_fov
        self.region_method, self.attention_method, self.alpha = region_method, attention_method, alpha
        self.branch_drop, self.aux_weight, self.view_embed = branch_drop, aux_weight, view_embed
        self._capture = False
        self._rollout = None
        self._last = None
        # Disable fused attention and hook the softmaxed weights *before* wrapping
        # blocks, so the hooks stay attached to each block's attn_drop submodule.
        for block in self.vit.blocks:
            block.attn.fused_attn = False
            block.attn.attn_drop.register_forward_pre_hook(self._attention_hook)
        if not self.legacy:
            self._apply_tuning(tuning, lora_rank, lora_alpha, adapter_hidden)
        if view_embed:
            self.view_g = nn.Parameter(torch.zeros(d))
            self.view_l = nn.Parameter(torch.zeros(d))
        if fusion == 'gate':
            self.ln_g, self.ln_l = nn.LayerNorm(d), nn.LayerNorm(d)
            self.gate = nn.Sequential(nn.Linear(2 * d, gate_hidden), nn.GELU(),
                                      nn.Linear(gate_hidden, d), nn.Sigmoid())
            self.s = nn.Parameter(torch.zeros(1))
        if fusion == 'concat':
            self.ln_cat = nn.LayerNorm(2 * d)
        feat_dim = 2 * d if fusion == 'concat' else d
        self.drop = nn.Dropout(drop_rate)
        if self.legacy:
            # Match checkpoints made before LoRA, cosine heads and auxiliary
            # heads were added: head.0 is Dropout and head.1 is Linear.
            self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(feat_dim, num_classes))
        else:
            self.head = (CosineHead(feat_dim, num_classes, tau, learnable_tau)
                         if cosine_head else nn.Linear(feat_dim, num_classes))
        if not self.legacy and fusion in {'concat', 'gate'}:
            self.aux_g = nn.Linear(d, num_classes)
            self.aux_l = nn.Linear(d, num_classes)

    @classmethod
    def from_config(cls, config, pretrained=False):
        """Build from a saved config, detecting pre-LoRA legacy checkpoints."""
        config = dict(config)
        legacy = 'tuning' not in config and 'cosine_head' not in config
        return cls(**config, pretrained=pretrained, legacy=legacy)

    def _apply_tuning(self, tuning, rank, lora_alpha, hidden):
        """[B] Freeze the backbone and inject LoRA / adapters (or leave full)."""
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

    @torch.no_grad()
    def _attention_hook(self, module, inputs):
        if not self._capture:
            return
        a = inputs[0].detach().float().mean(1)
        self._last = a[:, 0, self.vit.num_prefix_tokens:]
        if self.attention_method == 'rollout':
            a = .5 * a + .5 * torch.eye(a.shape[-1], device=a.device)[None]
            a = a / a.sum(-1, keepdim=True)
            self._rollout = a if self._rollout is None else a @ self._rollout

    def encode_map(self, x):
        self._rollout = self._last = None
        self._capture = True
        try:
            z = self.vit(x)
        finally:
            self._capture = False
        m = (self._rollout[:, 0, self.vit.num_prefix_tokens:]
             if self.attention_method == 'rollout' else self._last)
        grid = math.isqrt(m.shape[-1])
        if grid * grid != m.shape[-1]:
            raise ValueError('Backbone must have a square patch grid')
        self._rollout = self._last = None
        return z, m.reshape(-1, 1, grid, grid)

    @torch.no_grad()
    def select_boxes(self, m, mask, height, width):
        if self.use_fov:
            m = m * F.adaptive_avg_pool2d(mask, m.shape[-2:])
        kernel = m.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[None, None] / 16
        m = F.conv2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), kernel)
        if self.use_fov:
            m = m * F.adaptive_avg_pool2d(mask, m.shape[-2:])
        boxes = []
        for item in m:
            side = self.crop_ratio * min(height, width)
            if self.region_method == 'threshold':
                try:
                    from scipy import ndimage
                except ImportError as exc:
                    raise ImportError(
                        "region_method='threshold' needs scipy. In Colab run `%pip install -q scipy`."
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
            else:
                # Integral-image sums over all pixel-aligned windows.
                dense = F.interpolate(item[None], (height, width), mode='bilinear', align_corners=False)[0, 0]
                k = max(1, round(side))
                integral = F.pad(dense.cumsum(0).cumsum(1), (1, 0, 1, 0))
                sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
                index = int(sums.argmax())
                y, x = divmod(index, sums.shape[1])
                cx, cy, side = x+k/2, y+k/2, float(k)
            if self.training:
                cx += random.uniform(-.1, .1)*side
                cy += random.uniform(-.1, .1)*side
                side *= random.uniform(.9, 1.1)
            side = min(max(side, 1.), height, width)
            x = min(max(cx-side/2, 0.), width-side)
            y = min(max(cy-side/2, 0.), height-side)
            boxes.append([x, y, x+side, y+side])
        return m.new_tensor(boxes), m

    def forward(self, hi, mask=None, selector=None):
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W]')
        if self.use_fov and self.fusion != 'global':
            if mask is None or mask.shape != (hi.shape[0], 1, *hi.shape[-2:]):
                raise ValueError('use_fov requires mask [B,1,H,W] matching hi')
        if self.fusion == 'local' and selector is None:
            raise ValueError('E2/local requires a frozen selector (e.g. a frozen E1 or EMA copy)')
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
            if not self.legacy and self.view_embed:
                zg = zg + self.view_g
            logits = self.head(zg) if self.legacy else self.head(self.drop(zg))
            return {'logits': logits, 'boxes': None, 'gate': None,
                    'aux_g': None, 'aux_l': None, 'attention': None}
        if selector is not None:
            with torch.no_grad():
                _, m = selector.encode_map(global_view)
            zg = self.vit(global_view) if self.fusion != 'local' else None
        else:
            zg, m = self.encode_map(global_view)
        boxes, m = self.select_boxes(m, mask, *hi.shape[-2:])
        # Pixel-edge coordinates sampled at crop pixel centers, align_corners=False.
        t = (torch.arange(224, device=hi.device, dtype=hi.dtype)+.5)/224
        xs = boxes[:, 0, None] + (boxes[:, 2]-boxes[:, 0])[:, None]*t
        ys = boxes[:, 1, None] + (boxes[:, 3]-boxes[:, 1])[:, None]*t
        gx = (2*xs/hi.shape[-1]-1)[:, None, :].expand(-1, 224, -1)
        gy = (2*ys/hi.shape[-2]-1)[:, :, None].expand(-1, -1, 224)
        local = F.grid_sample(hi, torch.stack([gx, gy], -1), align_corners=False, padding_mode='border')
        zl = self.vit((local-.5)/.5)
        if not self.legacy and self.view_embed:
            if zg is not None:
                zg = zg + self.view_g
            zl = zl + self.view_l
        gate = aux_g = aux_l = None
        if self.fusion == 'local':
            z = zl
        else:
            if not self.legacy and self.training and self.branch_drop > 0:  # [D] per-sample local branch dropout
                keep = torch.rand(zl.shape[0], 1, device=zl.device, dtype=zl.dtype) >= self.branch_drop
                zl_used = zl * keep
            else:
                zl_used = zl
            if self.fusion == 'concat':
                z = self.ln_cat(torch.cat([zg, zl_used], -1))
            else:
                gate = self.gate(torch.cat([self.ln_g(zg), self.ln_l(zl_used)], -1))
                z = zg + self.s*gate*zl_used
            if not self.legacy and self.training:                # [D] auxiliary heads (train only)
                aux_g, aux_l = self.aux_g(zg), self.aux_l(zl)
        logits = self.head(z) if self.legacy else self.head(self.drop(z))
        return {'logits': logits, 'boxes': boxes, 'gate': gate,
                'aux_g': aux_g, 'aux_l': aux_l, 'attention': m}

    def loss(self, out, target, criterion):
        """Training loss: CE(fused logits) + λ·(CE(aux_g) + CE(aux_l)).

        λ is aux_weight (0.3 by default). The auxiliary terms are returned for
        concat/gate during training; they are None at evaluation time.
        """
        total = criterion(out['logits'], target)
        if out.get('aux_g') is not None and out.get('aux_l') is not None:
            total = total + self.aux_weight * (criterion(out['aux_g'], target)
                                               + criterion(out['aux_l'], target))
        return total
