"""E1 baseline — ViT-B/16 + LoRA + cosine head on the GLOBAL view only.

Standalone counterpart of GlanceFocusViT for the E1 vs E4 comparison. It keeps
the exact same "nền chung" as E4 — [A] front-end (upstream, in the data
pipeline), [B] frozen ViT-B/16 + LoRA, [E] cosine head, and the EMA recipe — but
has NO local branch, NO region selector and NO gated fusion. Only the global
224×224 view is encoded, so E1 differs from E4 in exactly one thing: the local
information.

Compatibility: this model's module names, state_dict keys and ``.config`` match
``GlanceFocusViT(fusion='global')`` from model.py, so a checkpoint trained with
train_e1.py loads straight into the shared evaluate.py (``from_config`` rebuilds
an equivalent global model) — the E1 and E4 tables therefore come from the same
evaluator. Front-end [A] lives in processing.py; this module expects ``hi`` to
be the normalised high-resolution image I' in [0,1].
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
try:
    import timm
except ModuleNotFoundError as exc:
    if exc.name != 'timm':
        raise
    raise ImportError(
        "model_e1 needs timm. In Colab run `%pip install -q timm` before importing it."
    ) from exc


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
    """Exponential moving average of weights; eval + frozen, used at inference."""
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


class GlanceFocusE1(nn.Module):
    """ViT-B/16 (frozen) + LoRA + cosine head on the global 224×224 view.

    forward(hi, mask=None, selector=None) -> dict with the same keys as
    GlanceFocusViT (boxes/gate/aux_*/attention are always None), so train_e1,
    train.predict and evaluate.predict all work unchanged. ``selector`` is
    accepted and ignored (E1 has no region branch).
    """
    def __init__(self, num_classes, use_fov=True,
                 backbone='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                 drop_rate=.1, pretrained=True, tuning='lora', lora_rank=8,
                 lora_alpha=16., adapter_hidden=64, cosine_head=True,
                 tau=16., learnable_tau=False):
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
        self.fusion = 'global'            # lets train.predict / evaluate treat it as E1
        self.use_fov = use_fov
        self.attention_method = 'rollout'  # unused; present for evaluate compatibility
        self.legacy = False
        # Config mirrors GlanceFocusViT(fusion='global') so evaluate.from_config
        # rebuilds an architecturally identical model and loads this state_dict.
        self.config = dict(num_classes=num_classes, fusion='global', crop_ratio=.5,
                           use_fov=use_fov, backbone=backbone, gate_hidden=256,
                           drop_rate=drop_rate, region_method='window',
                           attention_method='rollout', alpha=1., tuning=tuning,
                           lora_rank=lora_rank, lora_alpha=lora_alpha,
                           adapter_hidden=adapter_hidden, cosine_head=cosine_head,
                           tau=tau, learnable_tau=learnable_tau, branch_drop=.2,
                           aux_weight=.3, view_embed=False)
        self.vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self._apply_tuning(tuning, lora_rank, lora_alpha, adapter_hidden)
        feat = self.vit.num_features
        self.drop = nn.Dropout(drop_rate)
        self.head = (CosineHead(feat, num_classes, tau, learnable_tau)
                     if cosine_head else nn.Linear(feat, num_classes))

    def _apply_tuning(self, tuning, rank, lora_alpha, hidden):
        """[B] Freeze the backbone and inject LoRA / adapters (identical to E4)."""
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

    def forward(self, hi, mask=None, selector=None):
        if hi.ndim != 4 or hi.shape[1] != 3 or not hi.is_floating_point():
            raise ValueError('hi must be floating point RGB [B,3,H,W]')
        global_view = F.interpolate(hi, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        global_view = (global_view - .5) / .5
        logits = self.head(self.drop(self.vit(global_view)))
        return {'logits': logits, 'boxes': None, 'gate': None,
                'aux_g': None, 'aux_l': None, 'attention': None}

    def loss(self, out, target, criterion):
        """E1 has a single branch, so the loss is plain cross-entropy."""
        return criterion(out['logits'], target)
