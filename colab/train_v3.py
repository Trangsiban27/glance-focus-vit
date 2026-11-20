"""Self-contained Glance→Focus ViT **v3** training script for Colab.

Copy the entire file into one Colab code cell and run it. V3 keeps the V2 model
and Grad-CAM ROI implementation, with the revised optimizer/training settings:
  - ATTENTION_METHOD 'gradcam'/'gradrollout' = class-aware maps (vs plain rollout)
  - SUPPRESS_OUTLIERS = winsorise register-token peaks
  - REGION_METHOD 'adaptive' = window scale follows the attention mass
Checkpoints carry model_config['roi_version']='v3'. This training cell is
self-contained and does not import train.py or model_v3.py.
CSVs must have path,label columns; relative image paths resolve from their folder.
"""
# ---- EDIT THESE SETTINGS IF NEEDED ----
from pathlib import Path
TRAIN_CSV = Path('/content/data/Datos/train.csv')
VAL_CSV = Path('/content/data/Datos/valid.csv')
DATA_ROOT = None
OUTPUT_DIR = Path('/content/runs/glance_focus_gate_v3')
FUSION = 'gate'                     # global=E1, local=E2, concat=E3, gate=E4
TASK = 'binary'                      # Normal vs Abnormal
SELECTOR_CHECKPOINT = None           # E2: frozen v3 E1 best.pt; other modes leave None
CROP_RATIO = 0.5
HI_SIZE = 448
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
WARMUP_EPOCHS = 2
BATCH_SIZE = 2                       # grad ROI uses more memory; lower if OOM
NUM_WORKERS = 0
FUSION_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True
USE_FOV = True
PRETRAINED = True
SEED = 42
# ---- ROI v3 ----
ATTENTION_METHOD = 'gradcam'        # rollout | last | gradcam | gradrollout
REGION_METHOD = 'adaptive'          # window | threshold | adaptive
ROI_COVERAGE = 0.45                   # adaptive: target attention-mass fraction
SUPPRESS_OUTLIERS = True            # ② winsorise register-token peaks
ALPHA = 1.0
# ---- shared recipe ----
TUNING = 'lora'                     # [B] lora | adapter | full | frozen
LORA_RANK = 8
COSINE_HEAD = True                  # [E]
BRANCH_DROP = 0.1                   # [D]
AUX_WEIGHT = 0.3                    # [D]
S_INIT = 0.1
EMA_DECAY = 0.997                   # [C]
USE_EMA = True
FRONT_END = True                    # [A]
# --------------------------------------

import importlib.util
import subprocess
import sys

_missing = [name for name in ('timm', 'scipy', 'sklearn') if importlib.util.find_spec(name) is None]
if _missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'scipy', 'scikit-learn'])

import copy
import csv
import json
import math
import random
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

GRAD_METHODS = {'gradcam', 'gradrollout'}
ATTENTION_METHODS = {'rollout', 'last'} | GRAD_METHODS
REGION_METHODS = {'window', 'threshold', 'adaptive'}


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

def _square_fov_box(valid):
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
    arr = np.asarray(image, dtype=np.float32) / 255
    pixels = arr[valid] if valid is not None and valid.any() else arr.reshape(-1, 3)
    illuminant = np.mean(pixels ** power, axis=0) ** (1 / power)
    illuminant = np.where(illuminant < 1e-6, 1e-6, illuminant)
    arr = np.clip(arr * (illuminant.mean() / illuminant), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

class ImageProcessor:
    """[A] geometry -> FoV crop -> Shades-of-Gray -> views."""
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
            x0, y0, x1, y1 = _square_fov_box(valid)
            image = image.crop((x0, y0, x1, y1))
            valid = valid[y0:y1, x0:x1]
            image = shades_of_gray(image, valid)
        if self.use_fov:
            plane = valid if valid is not None else np.ones((image.height, image.width), dtype=bool)
            mask = torch.from_numpy(plane.astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.ones(1, image.height, image.width)
        hi = TF.resize(TF.to_tensor(image), [self.hi_size, self.hi_size], antialias=True)
        mask = F.interpolate(mask[None], size=(self.hi_size, self.hi_size), mode='area')[0]
        return hi, mask

class ImageManifest(Dataset):
    def __init__(self, manifest, data_root=None, classes=None, hi_size=448,
                 training=False, use_fov=True, front_end=True):
        self.manifest = Path(manifest).resolve()
        root = Path(data_root).resolve() if data_root else self.manifest.parent
        with self.manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'path', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain path,label columns')
            rows = list(reader)
        if not rows:
            raise ValueError(f'Empty manifest: {manifest}')
        if TASK == 'binary':
            self.classes = ['Normal', 'Abnormal']
            if classes is not None and list(classes) != self.classes:
                raise ValueError(f'Binary classes must be {self.classes}')
        else:
            self.classes = list(classes) if classes is not None else sorted({r['label'] for r in rows})
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('classes must be nonempty and unique')
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.samples = []
        for row in rows:
            if TASK == 'binary':
                raw_label = row['label'].strip().casefold()
                if not raw_label:
                    raise ValueError(f'Empty label in {manifest}')
                label_index = 0 if raw_label in {'normal', 'bình thường'} else 1
            else:
                if row['label'] not in self.class_to_idx:
                    raise ValueError(f"Unknown label {row['label']!r} in {manifest}")
                label_index = self.class_to_idx[row['label']]
            path = (root / row['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, label_index))
        if len({p for p, _ in self.samples}) != len(self.samples):
            raise ValueError(f'Duplicate image paths in {manifest}')
        self.hi_size, self.training, self.use_fov = hi_size, training, use_fov
        self.front_end = front_end
        self.processor = ImageProcessor(hi_size, training, use_fov, front_end)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        hi, mask = self.processor(image)
        return hi, mask, label, str(path)

    def resolution_stats(self):
        sizes = []
        for path, _ in self.samples:
            with Image.open(path) as image:
                sizes.append(image.size)
        a = np.asarray(sizes)
        return {'count': len(sizes), 'width_min_median_max': np.quantile(a[:, 0], [0, .5, 1]).tolist(),
                'height_min_median_max': np.quantile(a[:, 1], [0, .5, 1]).tolist(),
                'fraction_smaller_than_hi': float((a.min(1) < self.hi_size).mean())}

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def create_dataloader(dataset, batch_size=8, shuffle=False, workers=0,
                      device='cpu', seed=None, drop_last=False):
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if workers < 0:
        raise ValueError('workers must be >= 0')
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=str(device).startswith('cuda'), persistent_workers=workers > 0,
                      worker_init_fn=_seed_worker if workers > 0 else None,
                      generator=generator, drop_last=drop_last)


class LoRALinear(nn.Module):
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
    """Shared ViT-B/16 with class-aware, adaptive ROI selection (v2)."""
    def __init__(self, num_classes, fusion='gate', crop_ratio=.5, use_fov=True,
                 backbone='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                 gate_hidden=256, drop_rate=.1, region_method='adaptive',
                 attention_method='gradcam', alpha=1., pretrained=True,
                 tuning='lora', lora_rank=8, lora_alpha=16., adapter_hidden=64,
                 cosine_head=True, tau=16., learnable_tau=False,
                 branch_drop=.1, aux_weight=.3, view_embed=False,
                 suppress_outliers=True, roi_coverage=.45, legacy=False, s_init=.1):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if not math.isfinite(s_init):
            raise ValueError('s_init must be finite')
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
        self._mode = None
        self._rollout = None
        self._last = None
        self._attn = []
        self._feat = None  # pre-attention tokens from the final block (gradcam)
        for block in self.vit.blocks:
            block.attn.fused_attn = False
            block.attn.attn_drop.register_forward_pre_hook(self._attn_hook)
        self._apply_tuning(tuning, lora_rank, lora_alpha, adapter_hidden)
        last_block = self.vit.blocks[-1]
        while isinstance(last_block, Adapter):
            last_block = last_block.block
        # Capture tokens before the final self-attention so the classifier's
        # CLS score has gradients with respect to every patch token.
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
            self._attn.append(a)
            return
        am = a.detach().float().mean(1)
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
                        cam_l = (a * grad).clamp(min=0).mean(1)
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
        if self.suppress_outliers:
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
            elif self.region_method == 'adaptive':
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
        integral = F.pad(dense.cumsum(0).cumsum(1), (1, 0, 1, 0))
        sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
        index = int(sums.argmax())
        y, x = divmod(index, sums.shape[1])
        return x + k/2, y + k/2, float(k)

    def _adaptive_window(self, dense, height, width):
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
            raise ValueError('E2/local requires a frozen selector (frozen E1 or EMA copy)')
        if (self.training and self.attention_method in GRAD_METHODS
                and self.fusion != 'global' and selector is None):
            raise ValueError('training grad ROI needs a separate frozen selector (EMA/E1)')
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
        total = criterion(out['logits'], target)
        if out.get('aux_g') is not None and out.get('aux_l') is not None:
            total = total + self.aux_weight * (criterion(out['aux_g'], target)
                                               + criterion(out['aux_l'], target))
        return total


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def compute_metrics(labels, probabilities, num_classes, class_names=None):
    predictions = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), average='macro', zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    try:
        auc = float(roc_auc_score(labels, probabilities[:, 1]) if num_classes == 2 else
                    roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes))))
    except ValueError:
        auc = None
    report = {'accuracy': float(accuracy_score(labels, predictions)),
              'precision_macro': float(precision), 'macro_f1': float(f1),
              'roc_auc': auc, 'confusion_matrix': matrix.tolist()}
    if num_classes == 2:
        positive_index = (list(class_names).index('Abnormal')
                          if class_names is not None and 'Abnormal' in class_names else 1)
        negative_index = 1 - positive_index
        positive_count = matrix[positive_index].sum()
        negative_count = matrix[negative_index].sum()
        sensitivity = float(matrix[positive_index, positive_index] / positive_count) if positive_count else None
        specificity = float(matrix[negative_index, negative_index] / negative_count) if negative_count else None
        balanced_accuracy = ((sensitivity + specificity) / 2
                             if sensitivity is not None and specificity is not None else None)
        positive_name = (list(class_names)[positive_index]
                         if class_names is not None else f'class_{positive_index}')
        report.update(positive_class=positive_name, sensitivity=sensitivity,
                      specificity=specificity, balanced_accuracy=balanced_accuracy)
    else:
        tn = matrix.sum() - matrix.sum(0) - matrix.sum(1) + matrix.diagonal()
        fp = matrix.sum(0) - matrix.diagonal()
        specificity_macro = np.divide(
            tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0).mean()
        report.update(recall_macro=float(recall), specificity_macro=float(specificity_macro))
    return report

@torch.no_grad()
def evaluate_epoch(model, loader, device, selector=None):
    model.eval()
    labels, probabilities = [], []
    for hi, mask, target, _ in loader:
        output = model(hi.to(device), mask.to(device), selector)
        labels.extend(target.tolist())
        probabilities.extend(output['logits'].softmax(-1).cpu().tolist())
    return np.asarray(labels), np.asarray(probabilities)

def load_selector_v3(path, classes, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['model_config'].get('roi_version') != 'v3':
        raise ValueError('v3 local/E2 requires a v3 global (E1) selector checkpoint')
    selector = GlanceFocusViTV3.from_config(checkpoint['model_config'], pretrained=False)
    state = checkpoint['model_ema'] if checkpoint.get('model_ema') is not None else checkpoint['model']
    selector.load_state_dict(state)
    if selector.fusion != 'global' or checkpoint['classes'] != classes:
        raise ValueError('Selector checkpoint must be v3 E1/global with the same classes')
    return selector.to(device).eval().requires_grad_(False)


def main():
    if FUSION not in {'global', 'local', 'concat', 'gate'}:
        raise ValueError('FUSION must be global, local, concat, or gate')
    if FUSION == 'local' and not SELECTOR_CHECKPOINT:
        raise ValueError('E2/local requires SELECTOR_CHECKPOINT pointing to a frozen v3 E1 checkpoint')
    if ATTENTION_METHOD in GRAD_METHODS and not USE_EMA and FUSION != 'global' and not SELECTOR_CHECKPOINT:
        raise ValueError('grad ROI needs EMA (or SELECTOR_CHECKPOINT) as a separate frozen selector; keep USE_EMA=True')
    for csv_path in (TRAIN_CSV, VAL_CSV):
        if not Path(csv_path).is_file():
            raise FileNotFoundError(f'CSV not found: {csv_path}. Check TRAIN_CSV and VAL_CSV at the top of this cell.')
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f'OUTPUT_DIR already contains files: {OUTPUT_DIR}; change OUTPUT_DIR at the top.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = choose_device()
    print('Device:', device, '| ROI:', ATTENTION_METHOD, '| region:', REGION_METHOD)
    if device.type != 'cuda':
        print('No GPU detected. In Colab choose Runtime → Change runtime type → GPU.')

    train_data = ImageManifest(TRAIN_CSV, DATA_ROOT, hi_size=HI_SIZE, training=True,
                               use_fov=USE_FOV, front_end=FRONT_END)
    val_data = ImageManifest(VAL_CSV, DATA_ROOT, classes=train_data.classes,
                             hi_size=HI_SIZE, training=False, use_fov=USE_FOV, front_end=FRONT_END)
    if len(train_data.classes) < 2:
        raise ValueError('Training CSV must contain at least two classes')
    if {p for p, _ in train_data.samples} & {p for p, _ in val_data.samples}:
        raise ValueError('Train and validation CSVs contain overlapping images')
    print('Classes:', train_data.classes, '| images: train=%d valid=%d' % (len(train_data), len(val_data)))
    train_loader = create_dataloader(train_data, BATCH_SIZE, shuffle=True, workers=NUM_WORKERS, device=device, seed=SEED)
    val_loader = create_dataloader(val_data, BATCH_SIZE, shuffle=False, workers=NUM_WORKERS, device=device, seed=SEED + 1)

    model = GlanceFocusViTV3(len(train_data.classes), fusion=FUSION, crop_ratio=CROP_RATIO,
                             use_fov=USE_FOV, region_method=REGION_METHOD,
                             attention_method=ATTENTION_METHOD, alpha=ALPHA, tuning=TUNING,
                             lora_rank=LORA_RANK, cosine_head=COSINE_HEAD, branch_drop=BRANCH_DROP,
                             aux_weight=AUX_WEIGHT, suppress_outliers=SUPPRESS_OUTLIERS,
                             roi_coverage=ROI_COVERAGE, s_init=S_INIT, pretrained=PRETRAINED).to(device)
    external_selector = load_selector_v3(SELECTOR_CHECKPOINT, train_data.classes, device) if SELECTOR_CHECKPOINT else None
    if external_selector is not None:
        external_selector.attention_method = ATTENTION_METHOD
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
    if FUSION == 'local':
        train_selector = external_selector
    elif FUSION in {'concat', 'gate'}:
        train_selector = external_selector or (ema.module if ema is not None else None)
    else:
        train_selector = None
    infer_model = ema.module if ema is not None else model
    val_selector = external_selector if FUSION == 'local' else None

    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = np.bincount([label for _, label in train_data.samples], minlength=len(train_data.classes))
        if np.any(counts == 0):
            raise ValueError(f'Cannot compute class weights; missing training class: {counts.tolist()}')
        class_weights = torch.tensor(len(train_data) / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    trainable = [p for p in model.parameters() if p.requires_grad]
    layer_norm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    grouped_parameters = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith('vit.')
        base_lr = BACKBONE_LEARNING_RATE if is_backbone else FUSION_LEARNING_RATE
        no_decay = (name == 's' or name.endswith('.bias') or parameter.ndim <= 1
                    or id(parameter) in layer_norm_parameter_ids)
        decay = 0.0 if no_decay else WEIGHT_DECAY
        grouped_parameters.setdefault((base_lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW(
        [{'params': parameters, 'lr': base_lr, 'base_lr': base_lr, 'weight_decay': decay}
         for (base_lr, decay), parameters in grouped_parameters.items()]
    )
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_EPOCHS * len(train_loader), total_steps)
    best_f1, global_step, epochs_without_improvement = -1.0, 0, 0
    history_path = OUTPUT_DIR / 'history.jsonl'

    for epoch in range(EPOCHS):
        model.train()
        loss_total = 0.0
        processed_samples = 0
        epoch_started = time.perf_counter()
        log_interval = max(1, len(train_loader) // 10)
        for batch_index, (hi, mask, target, _) in enumerate(train_loader):
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(1, warmup_steps)
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * lr_scale
            optimizer.zero_grad(set_to_none=True)
            output = model(hi.to(device), mask.to(device), train_selector)
            loss = model.loss(output, target.to(device), criterion)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            global_step += 1
            loss_total += loss.item() * len(target)
            processed_samples += len(target)
            if ((batch_index + 1) % log_interval == 0
                    or batch_index + 1 == len(train_loader)):
                print(
                    f'Epoch {epoch + 1}/{EPOCHS} | '
                    f'batch {batch_index + 1}/{len(train_loader)} | '
                    f'loss {loss_total / processed_samples:.4f} | '
                    f'{time.perf_counter() - epoch_started:.1f}s',
                    flush=True,
                )

        labels, probabilities = evaluate_epoch(infer_model, val_loader, device, val_selector)
        report = compute_metrics(labels, probabilities, len(train_data.classes), class_names=train_data.classes)
        row = {'epoch': epoch + 1, 'train_loss': loss_total / len(train_data), **report}
        print(json.dumps(row), flush=True)
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        checkpoint = {'model': model.state_dict(),
                      'model_ema': ema.module.state_dict() if ema is not None else None,
                      'model_config': model.config,
                      'classes': train_data.classes, 'task': TASK, 'hi_size': HI_SIZE,
                      'front_end': FRONT_END, 'ema_decay': EMA_DECAY, 'epoch': epoch + 1,
                      'val_metrics': report, 'seed': SEED, 'optimizer': optimizer.state_dict(),
                      'train_config': {'fusion': FUSION, 'crop_ratio': CROP_RATIO,
                                       'use_fov': USE_FOV, 'hi_size': HI_SIZE,
                                       'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
                                       'fusion_lr': FUSION_LEARNING_RATE,
                                       'backbone_lr': BACKBONE_LEARNING_RATE,
                                       'weight_decay': WEIGHT_DECAY,
                                       'class_weights': USE_CLASS_WEIGHTS,
                                       'label_smoothing': LABEL_SMOOTHING,
                                       's_init': S_INIT, 'branch_drop': BRANCH_DROP,
                                       'aux_weight': AUX_WEIGHT, 'roi_coverage': ROI_COVERAGE,
                                       'warmup_epochs': WARMUP_EPOCHS,
                                       'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                                       'attention_method': ATTENTION_METHOD,
                                       'region_method': REGION_METHOD}}
        torch.save(checkpoint, OUTPUT_DIR / 'last.pt')
        if report['macro_f1'] > best_f1:
            best_f1 = report['macro_f1']
            epochs_without_improvement = 0
            torch.save(checkpoint, OUTPUT_DIR / 'best.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping: validation Macro-F1 did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.')
                break
    print(f'Best validation Macro-F1: {best_f1:.6f}')
    print('Best checkpoint:', OUTPUT_DIR / 'best.pt')


if __name__ == '__main__':
    main()
