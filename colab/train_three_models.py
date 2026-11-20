"""Train Swin-Tiny, ResNet-50 and DenseNet-121 for seeds 42, 46 and 49.

Colab cell workflow:
  1. Run/paste the cells for model_swin_tiny.py, model_resnet50.py and
     model_densenet121.py.
  2. Paste this whole file into the next cell and run it.

This script does not import this repository's train.py. It includes its own
CSV dataset, image preprocessing, metrics, training loop and checkpoint saves.
If run as a .py file, keep the three model_*.py files beside it.

CSV files must have path,label columns. Binary mapping treats Normal or
"bình thường" as Normal; every other label is Abnormal.
"""

# ---- Colab settings ----
from pathlib import Path

TRAIN_CSV = Path('/content/data/Datos/train.csv')
VAL_CSV = Path('/content/data/Datos/valid.csv')
DATA_ROOT = None
OUTPUT_DIR = Path('/content/runs/three_models_3seeds')

SEEDS = (42, 46, 49)
MODEL_NAMES = ('swin_tiny', 'resnet50', 'densenet121')
MODEL_TUNING = {
    'swin_tiny': 'lora',
    'resnet50': 'last_stage',
    'densenet121': 'last_stage',
}
PRETRAINED = True
USE_EMA = True
EMA_DECAY = 0.999

HI_SIZE = 448
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
WARMUP_EPOCHS = 5
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True
FRONT_END = True
USE_FOV = True
# -------------------------

import copy
import csv
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

_missing = [name for name in ('numpy', 'PIL', 'scipy', 'sklearn', 'timm')
            if importlib.util.find_spec(name) is None]
if _missing:
    _packages = {'numpy': 'numpy', 'PIL': 'pillow', 'scipy': 'scipy',
                 'sklearn': 'scikit-learn', 'timm': 'timm'}
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           *[_packages[name] for name in _missing]])

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


CLASS_NAMES = ['Normal', 'Abnormal']
MODEL_DEFINITIONS = {
    'swin_tiny': ('SwinTinyClassifier', 'model_swin_tiny.py'),
    'resnet50': ('ResNet50Classifier', 'model_resnet50.py'),
    'densenet121': ('DenseNet121Classifier', 'model_densenet121.py'),
}


def _load_model_classes():
    """Use model classes already defined in notebook cells, or load sibling files."""
    classes = {}
    script_path = globals().get('__file__')
    model_dir = Path(script_path).resolve().parent if script_path else Path.cwd().resolve()
    for model_name, (class_name, filename) in MODEL_DEFINITIONS.items():
        model_class = globals().get(class_name)
        if model_class is None:
            model_path = model_dir / filename
            if not model_path.is_file() and (model_dir / 'colab' / filename).is_file():
                model_path = model_dir / 'colab' / filename
            if not model_path.is_file():
                raise FileNotFoundError(
                    f'Missing {filename}. Run its model cell first, or keep the file '
                    'beside train_three_models.py.'
                )
            spec = importlib.util.spec_from_file_location(f'_baseline_{model_name}', model_path)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot load model class from {model_path}')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, class_name)
        classes[model_name] = model_class
    return classes


def _fov_valid(image):
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255
    dark = gray <= 0.05
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
    return valid


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


def _shades_of_gray(image, valid=None, power=6):
    arr = np.asarray(image, dtype=np.float32) / 255
    pixels = arr[valid] if valid is not None and valid.any() else arr.reshape(-1, 3)
    illuminant = np.mean(pixels ** power, axis=0) ** (1 / power)
    illuminant = np.maximum(illuminant, 1e-6)
    arr = np.clip(arr * (illuminant.mean() / illuminant), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


class ImageProcessor:
    """Geometry, FoV crop and Shades-of-Gray front end used by the project."""
    def __init__(self, hi_size=448, training=False, use_fov=True, front_end=True):
        self.hi_size = hi_size
        self.training = training
        self.use_fov = use_fov
        self.front_end = front_end

    def __call__(self, image):
        image = image.convert('RGB')
        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.uniform(-15, 15),
                              interpolation=InterpolationMode.BILINEAR)
        valid = _fov_valid(image) if (self.use_fov or self.front_end) else None
        if self.front_end:
            x0, y0, x1, y1 = _square_fov_box(valid)
            image = image.crop((x0, y0, x1, y1))
            valid = valid[y0:y1, x0:x1]
            image = _shades_of_gray(image, valid)
        if self.use_fov:
            plane = valid if valid is not None else np.ones((image.height, image.width), dtype=bool)
            mask = torch.from_numpy(plane.astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.ones(1, image.height, image.width)
        hi = TF.resize(TF.to_tensor(image), [self.hi_size, self.hi_size], antialias=True)
        mask = F.interpolate(mask[None], size=(self.hi_size, self.hi_size), mode='area')[0]
        return hi, mask


class ImageManifest(Dataset):
    """Binary CSV dataset; relative paths resolve from DATA_ROOT or CSV folder."""
    def __init__(self, manifest, data_root=None, hi_size=448, training=False,
                 use_fov=True, front_end=True):
        self.manifest = Path(manifest).expanduser().resolve()
        root = Path(data_root).expanduser().resolve() if data_root else self.manifest.parent
        with self.manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'path', 'label'} <= set(reader.fieldnames or []):
                raise ValueError(f'CSV must contain path,label columns: {self.manifest}')
            rows = list(reader)
        if not rows:
            raise ValueError(f'Empty CSV: {self.manifest}')
        self.samples = []
        for row in rows:
            raw_label = row['label'].strip()
            if not raw_label:
                raise ValueError(f'Empty label in {self.manifest}')
            label = 0 if raw_label.casefold() in {'normal', 'bình thường'} else 1
            image_path = (root / row['path']).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            self.samples.append((image_path, label))
        if len({path for path, _ in self.samples}) != len(self.samples):
            raise ValueError(f'Duplicate image paths in {self.manifest}')
        self.classes = CLASS_NAMES.copy()
        self.processor = ImageProcessor(hi_size, training, use_fov, front_end)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        hi, mask = self.processor(image)
        return hi, mask, label, str(image_path)

    def resolution_stats(self):
        sizes = []
        for path, _ in self.samples:
            with Image.open(path) as image:
                sizes.append(image.size)
        sizes = np.asarray(sizes)
        return {
            'count': len(sizes),
            'width_min_median_max': np.quantile(sizes[:, 0], [0, 0.5, 1]).tolist(),
            'height_min_median_max': np.quantile(sizes[:, 1], [0, 0.5, 1]).tolist(),
            'fraction_smaller_than_hi': float((sizes.min(1) < HI_SIZE).mean()),
        }


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_loader(dataset, batch_size, shuffle, workers, device, generator):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == 'cuda', persistent_workers=False,
        worker_init_fn=_seed_worker if workers > 0 else None, generator=generator,
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_metrics(labels, probabilities):
    predictions = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average='macro', zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    try:
        auc = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        auc = None
    normal_count = int(matrix[0].sum())
    abnormal_count = int(matrix[1].sum())
    specificity = float(matrix[0, 0] / normal_count) if normal_count else None
    sensitivity = float(matrix[1, 1] / abnormal_count) if abnormal_count else None
    balanced_accuracy = ((specificity + sensitivity) / 2
                         if specificity is not None and sensitivity is not None else None)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'macro_f1': float(f1),
        'roc_auc': auc,
        'positive_class': 'Abnormal',
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': matrix.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for hi, mask, target, _ in loader:
        output = model(hi.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        labels.extend(target.numpy().tolist())
        probabilities.extend(output['logits'].softmax(-1).cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


class WeightEMA:
    def __init__(self, model, decay=0.999):
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


def _capture_rng(train_generator):
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'loader_generator': train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, train_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    train_generator.set_state(state['loader_generator'])
    if torch.cuda.is_available() and state.get('torch_cuda') is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _checkpoint_base(model, ema, train_data, val_data, seed, epoch, val_metrics, args_record):
    return {
        'model': model.state_dict(),
        'model_ema': ema.module.state_dict() if ema is not None else None,
        'model_config': model.config,
        'model_type': model.config['model_type'],
        'classes': train_data.classes,
        'task': 'binary',
        'hi_size': HI_SIZE,
        'front_end': FRONT_END,
        'seed': seed,
        'epoch': epoch,
        'val_metrics': val_metrics,
        'train_args': args_record,
        'splits': {
            'train': [{'path': str(path), 'label': int(label)} for path, label in train_data.samples],
            'val': [{'path': str(path), 'label': int(label)} for path, label in val_data.samples],
        },
    }


def train_one(model_name, model_class, seed, train_data, val_data, device):
    run_dir = OUTPUT_DIR / model_name / f'seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = run_dir / 'completed.json'
    last_path = run_dir / 'last.pt'
    best_path = run_dir / 'best.pt'
    if completed_path.exists():
        print(f'Skip completed run: {model_name}, seed {seed}', flush=True)
        return

    seed_all(seed)
    tuning = MODEL_TUNING[model_name]
    model = model_class(num_classes=2, tuning=tuning, pretrained=PRETRAINED).to(device)
    ema = WeightEMA(model, EMA_DECAY) if USE_EMA else None
    infer_model = ema.module if ema is not None else model

    train_generator = torch.Generator().manual_seed(seed + 1)
    val_generator = torch.Generator().manual_seed(seed + 2)
    train_loader = _make_loader(train_data, BATCH_SIZE, True, NUM_WORKERS, device, train_generator)
    val_loader = _make_loader(val_data, BATCH_SIZE, False, NUM_WORKERS, device, val_generator)

    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = np.bincount([label for _, label in train_data.samples], minlength=2)
        if np.any(counts == 0):
            raise ValueError(f'Training class is missing for {model_name}/seed{seed}: {counts.tolist()}')
        class_weights = torch.tensor(len(train_data) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f'No trainable parameters in {model_name}')
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = min(WARMUP_EPOCHS * len(train_loader), total_steps)
    best_f1, global_step, epochs_without_improvement, start_epoch = -1.0, 0, 0, 0

    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state.get('model_type') != model_name or state.get('seed') != seed:
            raise ValueError(f'Existing checkpoint does not match {model_name}/seed{seed}: {last_path}')
        model.load_state_dict(state['model'])
        if ema is not None and state.get('model_ema') is not None:
            ema.module.load_state_dict(state['model_ema'])
        optimizer.load_state_dict(state['optimizer'])
        resume = state['resume_state']
        start_epoch = int(state['epoch'])
        best_f1 = float(resume['best_f1'])
        global_step = int(resume['global_step'])
        epochs_without_improvement = int(resume['epochs_without_improvement'])
        _restore_rng(resume['rng'], train_generator)
        print(f'Resuming {model_name}/seed{seed} at epoch {start_epoch + 1}', flush=True)

    args_record = {
        'model': model_name, 'tuning': tuning, 'seeds': list(SEEDS),
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'label_smoothing': LABEL_SMOOTHING,
        'class_weights': USE_CLASS_WEIGHTS, 'pretrained': PRETRAINED,
        'ema_decay': EMA_DECAY if USE_EMA else None,
    }
    print(f'\nRun {model_name}/seed{seed} on {device}; trainable parameters: '
          f'{sum(p.numel() for p in trainable):,}', flush=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        loss_total, sample_total = 0.0, 0
        epoch_started = time.perf_counter()
        log_interval = max(1, len(train_loader) // 10)
        for batch_index, (hi, mask, target, _) in enumerate(train_loader):
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(1, warmup_steps)
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                lr_scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group['lr'] = LEARNING_RATE * lr_scale
            optimizer.zero_grad(set_to_none=True)
            output = model(hi.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            loss = criterion(output['logits'], target.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss for {model_name}/seed{seed}')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            global_step += 1
            loss_total += loss.item() * len(target)
            sample_total += len(target)
            if ((batch_index + 1) % log_interval == 0
                    or batch_index + 1 == len(train_loader)):
                print(f'{model_name} seed{seed} | epoch {epoch + 1}/{EPOCHS} | '
                      f'batch {batch_index + 1}/{len(train_loader)} | '
                      f'loss {loss_total / sample_total:.4f} | '
                      f'{time.perf_counter() - epoch_started:.1f}s', flush=True)

        labels, probabilities = evaluate(infer_model, val_loader, device)
        report = compute_metrics(labels, probabilities)
        print(json.dumps({'model': model_name, 'seed': seed, 'epoch': epoch + 1,
                          'train_loss': loss_total / len(train_data), **report}), flush=True)
        with (run_dir / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'epoch': epoch + 1, 'train_loss': loss_total / len(train_data),
                                     **report}) + '\n')

        epochs_without_improvement += 1
        if report['macro_f1'] > best_f1:
            best_f1 = report['macro_f1']
            epochs_without_improvement = 0
            best_checkpoint = _checkpoint_base(model, ema, train_data, val_data, seed, epoch + 1,
                                               report, args_record)
            torch.save(best_checkpoint, best_path)

        last_checkpoint = _checkpoint_base(model, ema, train_data, val_data, seed, epoch + 1,
                                           report, args_record)
        last_checkpoint.update(
            optimizer=optimizer.state_dict(),
            resume_state={
                'best_f1': best_f1,
                'global_step': global_step,
                'epochs_without_improvement': epochs_without_improvement,
                'rng': _capture_rng(train_generator),
            },
        )
        torch.save(last_checkpoint, last_path)
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f'Early stop {model_name}/seed{seed}: validation Macro-F1 did not improve '
                  f'for {EARLY_STOPPING_PATIENCE} epochs.', flush=True)
            break

    if not best_path.exists():
        raise RuntimeError(f'No best checkpoint was written for {model_name}/seed{seed}')
    completed_path.write_text(json.dumps({
        'model': model_name, 'seed': seed, 'best_macro_f1': best_f1,
        'best_checkpoint': str(best_path.resolve()),
    }, indent=2))
    print(f'Finished {model_name}/seed{seed}; best Macro-F1={best_f1:.4f}; '
          f'checkpoint={best_path}', flush=True)


def write_summary():
    fields = ['model', 'seed', 'best_epoch', 'accuracy', 'roc_auc', 'sensitivity',
              'specificity', 'balanced_accuracy', 'macro_f1']
    rows = []
    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            run_dir = OUTPUT_DIR / model_name / f'seed{seed}'
            checkpoint_path = run_dir / 'best.pt'
            if not (run_dir / 'completed.json').exists() or not checkpoint_path.is_file():
                continue
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            metrics = checkpoint['val_metrics']
            rows.append({
                'model': model_name,
                'seed': seed,
                'best_epoch': checkpoint['epoch'],
                **{key: metrics.get(key) for key in fields[3:]},
            })
    summary_path = OUTPUT_DIR / 'summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Validation summary: {summary_path}', flush=True)


def main():
    if not MODEL_NAMES or set(MODEL_NAMES) - set(MODEL_DEFINITIONS):
        raise ValueError(f'MODEL_NAMES must be selected from {tuple(MODEL_DEFINITIONS)}')
    if len(set(SEEDS)) != len(SEEDS) or any(not isinstance(seed, int) for seed in SEEDS):
        raise ValueError('SEEDS must contain unique integers')
    if BATCH_SIZE < 1 or EPOCHS < 1 or NUM_WORKERS < 0 or HI_SIZE < 224:
        raise ValueError('Require BATCH_SIZE,EPOCHS >=1, NUM_WORKERS >=0 and HI_SIZE >=224')
    for csv_path in (TRAIN_CSV, VAL_CSV):
        if not Path(csv_path).is_file():
            raise FileNotFoundError(f'CSV not found: {csv_path}; update paths at the top of this cell')

    classes = _load_model_classes()
    device = choose_device()
    print('Device:', device)
    if device.type != 'cuda':
        print('No CUDA GPU detected. In Colab choose Runtime → Change runtime type → GPU.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = ImageManifest(TRAIN_CSV, DATA_ROOT, hi_size=HI_SIZE, training=True,
                               use_fov=USE_FOV, front_end=FRONT_END)
    val_data = ImageManifest(VAL_CSV, DATA_ROOT, hi_size=HI_SIZE, training=False,
                             use_fov=USE_FOV, front_end=FRONT_END)
    if {path for path, _ in train_data.samples} & {path for path, _ in val_data.samples}:
        raise ValueError('Train and validation CSVs contain overlapping images')
    train_counts = np.bincount([label for _, label in train_data.samples], minlength=2)
    val_counts = np.bincount([label for _, label in val_data.samples], minlength=2)
    print(json.dumps({
        'classes': CLASS_NAMES,
        'train_count': len(train_data), 'train_class_counts': dict(zip(CLASS_NAMES, train_counts.tolist())),
        'val_count': len(val_data), 'val_class_counts': dict(zip(CLASS_NAMES, val_counts.tolist())),
        'resolution_stats': {'train': train_data.resolution_stats(), 'val': val_data.resolution_stats()},
        'models': list(MODEL_NAMES), 'seeds': list(SEEDS), 'class_weights': USE_CLASS_WEIGHTS,
    }, ensure_ascii=False), flush=True)

    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            train_one(model_name, classes[model_name], seed, train_data, val_data, device)
    write_summary()
    print('\nAll model/seed runs complete.', flush=True)


if __name__ == '__main__':
    main()

