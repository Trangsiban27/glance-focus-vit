"""Train E1 using the model classes defined by the preceding model_e1.py cell.

In Colab, run the cell containing model_e1.py first, then run this cell. Edit
DEFAULT_TRAIN_CSV, DEFAULT_VAL_CSV, and DEFAULT_OUTPUT below for your runtime.

CSV files need path,label columns. Relative image paths resolve from the CSV
folder, or from --data-root when it is supplied. Binary mode maps Normal to
Normal and every other label to Abnormal.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path


# Defaults match the usual Colab project layout. Pass CLI options to override.
DEFAULT_TRAIN_CSV = Path('/content/data/Datos/train.csv')
DEFAULT_VAL_CSV = Path('/content/data/Datos/valid.csv')
DEFAULT_OUTPUT = Path('/content/runs/E1')


def _install_missing_packages():
    packages = {'numpy': 'numpy', 'PIL': 'pillow', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'timm': 'timm'}
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])


_install_missing_packages()

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

if 'GlanceFocusE1' not in globals() or 'ModelEMA' not in globals():
    raise RuntimeError(
        'Run the model_e1.py cell first. It must define GlanceFocusE1 and ModelEMA.'
    )


def fov_valid(image):
    """Return the largest connected non-dark field-of-view region."""
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
    illuminant = np.maximum(illuminant, 1e-6)
    arr = np.clip(arr * (illuminant.mean() / illuminant), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


class ImageProcessor:
    """E1 front end: geometry, FoV crop and Shades-of-Gray color correction."""
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
    """CSV image dataset returning high-resolution RGB, FoV mask, label, path."""
    def __init__(self, manifest, data_root=None, classes=None, task='binary', hi_size=448,
                 training=False, use_fov=True, front_end=True):
        self.manifest = Path(manifest).expanduser().resolve()
        root = Path(data_root).expanduser().resolve() if data_root else self.manifest.parent
        with self.manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'path', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain path,label columns')
            rows = list(reader)
        if not rows:
            raise ValueError(f'Empty manifest: {self.manifest}')
        if task == 'binary':
            self.classes = ['Normal', 'Abnormal']
            if classes is not None and list(classes) != self.classes:
                raise ValueError(f'Binary classes must be {self.classes}')
        else:
            self.classes = list(classes) if classes is not None else sorted({r['label'].strip() for r in rows})
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('classes must be nonempty and unique')
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.samples = []
        for row in rows:
            raw_label = row['label'].strip()
            if not raw_label:
                raise ValueError(f'Empty label in {self.manifest}')
            if task == 'binary':
                label_index = 0 if raw_label.casefold() in {'normal', 'bình thường'} else 1
            else:
                if raw_label not in self.class_to_idx:
                    raise ValueError(f'Unknown label {raw_label!r} in {self.manifest}')
                label_index = self.class_to_idx[raw_label]
            path = (root / row['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, label_index))
        if len({path for path, _ in self.samples}) != len(self.samples):
            raise ValueError(f'Duplicate image paths in {self.manifest}')
        self.hi_size = hi_size
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
        sizes = np.asarray(sizes)
        return {
            'count': len(sizes),
            'width_min_median_max': np.quantile(sizes[:, 0], [0, .5, 1]).tolist(),
            'height_min_median_max': np.quantile(sizes[:, 1], [0, .5, 1]).tolist(),
            'fraction_smaller_than_hi': float((sizes.min(1) < self.hi_size).mean()),
        }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(dataset, batch_size, shuffle, workers, device, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == 'cuda', persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None, generator=generator,
    )


def compute_metrics(labels, probabilities, num_classes, class_names):
    predictions = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), average='macro', zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    try:
        auc = float(roc_auc_score(labels, probabilities[:, 1]) if num_classes == 2 else
                    roc_auc_score(labels, probabilities, multi_class='ovr', average='macro',
                                  labels=list(range(num_classes))))
    except ValueError:
        auc = None
    report = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'macro_f1': float(f1),
        'roc_auc': auc,
        'confusion_matrix': matrix.tolist(),
    }
    if num_classes == 2:
        positive_index = class_names.index('Abnormal') if 'Abnormal' in class_names else 1
        negative_index = 1 - positive_index
        positive_count = int(matrix[positive_index].sum())
        negative_count = int(matrix[negative_index].sum())
        sensitivity = float(matrix[positive_index, positive_index] / positive_count) if positive_count else None
        specificity = float(matrix[negative_index, negative_index] / negative_count) if negative_count else None
        report.update(
            positive_class=class_names[positive_index], sensitivity=sensitivity,
            specificity=specificity,
            balanced_accuracy=((sensitivity + specificity) / 2
                               if sensitivity is not None and specificity is not None else None),
        )
    else:
        true_negative = matrix.sum() - matrix.sum(0) - matrix.sum(1) + np.diag(matrix)
        false_positive = matrix.sum(0) - np.diag(matrix)
        specificity = np.divide(true_negative, true_negative + false_positive,
                                out=np.zeros_like(true_negative, dtype=float),
                                where=(true_negative + false_positive) != 0)
        report['specificity_macro'] = float(specificity.mean())
    return report


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for hi, mask, target, _ in loader:
        output = model(hi.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        labels.extend(target.tolist())
        probabilities.extend(output['logits'].softmax(-1).cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-csv', default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument('--val-csv', default=str(DEFAULT_VAL_CSV))
    parser.add_argument('--data-root')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--task', choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--no-fov', action='store_true')
    parser.add_argument('--no-front-end', action='store_true')
    parser.add_argument('--tuning', choices=['lora', 'adapter', 'full', 'frozen'], default='lora')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--no-cosine-head', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--hi-size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--class-weights', dest='class_weights', action='store_true', default=True,
                        help='Use inverse-frequency class weights (enabled by default)')
    parser.add_argument('--no-class-weights', dest='class_weights', action='store_false',
                        help='Disable inverse-frequency class weights')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not download pretrained weights; mainly for offline checks')
    # Pasted notebook cells carry the kernel's -f argument; use the defaults
    # above instead. Run the code in this cell after the model_e1.py cell.
    if argv is None and '__file__' not in globals() and 'get_ipython' in globals():
        argv = []
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.hi_size < 224 or args.workers < 0
            or args.warmup_epochs < 0 or args.early_stopping_patience < 1):
        parser.error('Require epochs,batch-size >=1, hi-size >=224, workers,warmup-epochs >=0')
    if not 0 < args.ema_decay < 1 or args.lora_rank < 1:
        parser.error('Require 0<ema-decay<1 and lora-rank>=1')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        parser.error('Require lr>0, weight-decay>=0 and label-smoothing in [0,1)')

    seed_all(args.seed)
    device = device_for(args.device)
    front_end = not args.no_front_end
    use_fov = not args.no_fov
    train = ImageManifest(args.train_csv, args.data_root, task=args.task, hi_size=args.hi_size,
                          training=True, use_fov=use_fov, front_end=front_end)
    val = ImageManifest(args.val_csv, args.data_root, classes=train.classes, task=args.task,
                        hi_size=args.hi_size, use_fov=use_fov, front_end=front_end)
    if len(train.classes) < 2:
        raise ValueError('Training requires at least two classes')
    if {path for path, _ in train.samples} & {path for path, _ in val.samples}:
        raise ValueError('Train/validation image paths overlap')

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'history.jsonl').exists():
        raise FileExistsError(f'Output already contains a run: {output}; choose a new --output directory')
    stats = {'train': train.resolution_stats(), 'val': val.resolution_stats()}
    print(json.dumps({'device': str(device), 'model': 'E1', 'classes': train.classes,
                      'resolution_stats': stats}, ensure_ascii=False))
    (output / 'resolution_stats.json').write_text(json.dumps(stats, indent=2))
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    splits = {name: [{'path': str(path), 'label': label} for path, label in ds.samples]
              for name, ds in (('train', train), ('val', val))}
    (output / 'splits.json').write_text(json.dumps({'classes': train.classes, **splits}, indent=2))

    loaders = [
        create_dataloader(ds, args.batch_size, shuffle=(ds is train), workers=args.workers,
                          device=device, seed=args.seed + index)
        for index, ds in enumerate((train, val))
    ]
    model = GlanceFocusE1(len(train.classes), use_fov=use_fov, tuning=args.tuning,
                          lora_rank=args.lora_rank, cosine_head=not args.no_cosine_head,
                          pretrained=not args.no_pretrained).to(device)
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    infer_model = ema.module if ema is not None else model

    weights = None
    if args.class_weights:
        counts = np.bincount([label for _, label in train.samples], minlength=len(train.classes))
        if np.any(counts == 0):
            raise ValueError(f'Cannot compute class weights; a training class has no images: {counts.tolist()}')
        weights = torch.tensor(len(train) / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders[0])
    total_steps = args.epochs * steps_per_epoch
    warm_steps = min(args.warmup_epochs * steps_per_epoch, total_steps)
    best, step, epochs_without_improvement = -1., 0, 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        for hi, mask, target, _ in loaders[0]:
            if warm_steps and step < warm_steps:
                factor = (step + 1) / warm_steps
            else:
                decay_steps = max(1, total_steps - warm_steps)
                progress = min(1., max(0., (step - warm_steps) / decay_steps))
                factor = .5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(hi.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            loss = model.loss(output_batch, target.to(device, non_blocking=True), criterion)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step += 1
            loss_sum += loss.item() * len(target)

        labels, probabilities = predict(infer_model, loaders[1], device)
        report = compute_metrics(labels, probabilities, len(train.classes), train.classes)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / len(train), **report)
        print(json.dumps(row), flush=True)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        checkpoint = {
            'model': model.state_dict(),
            'model_ema': ema.module.state_dict() if ema is not None else None,
            'model_config': model.config,
            'classes': train.classes,
            'task': args.task,
            'hi_size': args.hi_size,
            'front_end': front_end,
            'ema_decay': args.ema_decay,
            'epoch': epoch + 1,
            'val_metrics': report,
            'seed': args.seed,
            'optimizer': optimizer.state_dict(),
            'train_args': vars(args),
            'splits': splits,
            'selector_checkpoint': None,
        }
        torch.save(checkpoint, output / 'last.pt')
        if report['macro_f1'] > best:
            best = report['macro_f1']
            epochs_without_improvement = 0
            torch.save(checkpoint, output / 'best.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print('Early stopping: validation Macro-F1 did not improve for '
                      f'{args.early_stopping_patience} consecutive epochs.')
                break
    print(f'Best validation Macro-F1: {best:.6f}; checkpoint: {output / "best.pt"}')


if __name__ == '__main__':
    main()
