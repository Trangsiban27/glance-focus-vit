"""Train Glance→Focus ViT. CSV manifests require path,label columns.

Example:
 python train.py --train-csv data/train.csv --val-csv data/val.csv --fusion global --output runs/E1
 python train.py --train-csv data/train.csv --val-csv data/val.csv --fusion local --selector runs/E1/best.pt --output runs/E2
 python train.py --train-csv data/train.csv --val-csv data/val.csv --fusion gate --output runs/E4
Paths in manifests are relative to --data-root (default: manifest directory).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model import GlanceFocusViT, ModelEMA
from dataset import ImageManifest
from dataloader import create_dataloader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')


def metrics(labels, probabilities, num_classes, predictions=None, class_names=None):
    pred = probabilities.argmax(1) if predictions is None else np.asarray(predictions)
    p, r, f, _ = precision_recall_fscore_support(labels, pred, labels=list(range(num_classes)),
                                               average='macro', zero_division=0)
    cm = confusion_matrix(labels, pred, labels=list(range(num_classes)))
    auc = None
    try:
        auc = float(roc_auc_score(labels, probabilities[:, 1]) if num_classes == 2 else
                    roc_auc_score(labels, probabilities, multi_class='ovr', average='macro', labels=list(range(num_classes))))
    except ValueError:
        pass  # AUC is undefined if the split lacks a class.
    report = dict(accuracy=float(accuracy_score(labels, pred)), precision_macro=float(p),
                  macro_f1=float(f), roc_auc=auc, confusion_matrix=cm.tolist())
    if num_classes == 2:
        # Binary label mapping is Normal=0, Abnormal=1. Report metrics for the
        # clinically relevant positive class instead of duplicate macro values.
        positive_index = (list(class_names).index('Abnormal')
                          if class_names is not None and 'Abnormal' in class_names else 1)
        negative_index = 1 - positive_index
        positive_count, negative_count = cm[positive_index].sum(), cm[negative_index].sum()
        sensitivity = float(cm[positive_index, positive_index] / positive_count) if positive_count else None
        specificity = float(cm[negative_index, negative_index] / negative_count) if negative_count else None
        balanced_accuracy = ((sensitivity + specificity) / 2
                             if sensitivity is not None and specificity is not None else None)
        positive_name = (list(class_names)[positive_index]
                         if class_names is not None else f'class_{positive_index}')
        report.update(positive_class=positive_name, sensitivity=sensitivity,
                      specificity=specificity, balanced_accuracy=balanced_accuracy)
    else:
        tn = cm.sum() - cm.sum(0) - cm.sum(1) + cm.diagonal()
        fp = cm.sum(0) - cm.diagonal()
        report.update(recall_macro=float(r), specificity_macro=float(
            np.divide(tn, tn+fp, out=np.zeros_like(tn, dtype=float), where=(tn+fp)!=0).mean()))
    return report


@torch.no_grad()
def predict(model, loader, device, selector=None):
    model.eval()
    if selector is not None:
        selector.eval()
    labels, probs, records = [], [], []
    for hi, mask, target, paths in loader:
        out = model(hi.to(device), mask.to(device), selector)
        prob = out['logits'].softmax(-1).cpu().numpy()
        labels.extend(target.tolist()); probs.extend(prob.tolist())
        boxes = out['boxes'].cpu().tolist() if out['boxes'] is not None else [None]*len(paths)
        gates = out['gate'].cpu().tolist() if out['gate'] is not None else [None]*len(paths)
        for path, y, pr, box, gate in zip(paths, target.tolist(), prob.tolist(), boxes, gates):
            records.append(dict(path=path, label=y, prediction=int(np.argmax(pr)), probabilities=pr,
                                box_hi_xyxy=box, gate=gate))
    return np.asarray(labels), np.asarray(probs), records


def load_checkpoint(path, device, prefer_ema=True):
    # Only load checkpoints produced by these scripts from trusted sources.
    # Inference uses the EMA weights when present (spec §5, "EMA trọng số").
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt['model_config']
    model_type = config.get('model_type', ckpt.get('model_type'))
    if model_type == 'swin_tiny':
        from colab.model_swin_tiny import SwinTinyClassifier
        model = SwinTinyClassifier.from_config(config, pretrained=False)
    elif model_type == 'resnet50':
        from colab.model_resnet50 import ResNet50Classifier
        model = ResNet50Classifier.from_config(config, pretrained=False)
    elif model_type == 'densenet121':
        from colab.model_densenet121 import DenseNet121Classifier
        model = DenseNet121Classifier.from_config(config, pretrained=False)
    elif config.get('roi_version') == 'v3':
        from model_v3 import GlanceFocusViTV3
        model = GlanceFocusViTV3.from_config(config, pretrained=False)
    elif config.get('roi_version') == 'v2':
        from model_v2 import GlanceFocusViTV2
        model = GlanceFocusViTV2.from_config(config, pretrained=False)
    else:
        model = GlanceFocusViT.from_config(config, pretrained=False)
    state = ckpt['model_ema'] if prefer_ema and ckpt.get('model_ema') is not None else ckpt['model']
    model.load_state_dict(state)
    return model.to(device), ckpt


def load_selector(path, classes, device):
    model, ckpt = load_checkpoint(path, device)
    if model.fusion != 'global' or ckpt['classes'] != classes:
        raise ValueError('Selector must be an E1/global checkpoint with the same class mapping')
    model.eval().requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='runs/E4')
    parser.add_argument('--task', choices=['binary', 'multiclass'], default='binary',
                        help='Default task: Normal vs Abnormal; all non-Normal CSV labels map to Abnormal')
    parser.add_argument('--fusion', choices=['global', 'local', 'concat', 'gate'], default='gate')
    parser.add_argument('--selector', help='Frozen E1 checkpoint; required for local/E2')
    parser.add_argument('--crop-ratio', type=float, default=.5)
    parser.add_argument('--no-fov', action='store_true')
    parser.add_argument('--region-method', choices=['window', 'threshold'], default='window')
    parser.add_argument('--attention-method', choices=['rollout', 'last'], default='rollout')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--tuning', choices=['lora', 'adapter', 'full', 'frozen'], default='lora',
                        help='[B] parameter-efficient tuning of the frozen ViT backbone')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--no-cosine-head', action='store_true', help='[E] use a plain Linear head instead of cosine')
    parser.add_argument('--branch-drop', type=float, default=.2,
                        help='[D] per-sample local-branch dropout probability during training')
    parser.add_argument('--aux-weight', type=float, default=.3, help='[D] weight of the per-branch auxiliary heads (0 disables)')
    parser.add_argument('--ema-decay', type=float, default=.999, help='EMA decay for the selector + inference weights')
    parser.add_argument('--no-ema', action='store_true', help='Disable the weight EMA (selector uses the online model)')
    parser.add_argument('--no-front-end', action='store_true', help='[A] disable FoV-crop + Shades-of-Gray front-end')
    parser.add_argument('--hi-size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Stop after this many epochs without validation Macro-F1 improvement')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-pretrained', action='store_true', help='For offline smoke checks; not for the pretrained experiments')
    args = parser.parse_args()
    if args.fusion == 'local' and not args.selector:
        parser.error('E2 requires --selector pointing to a frozen E1 checkpoint')
    if (args.epochs < 1 or args.batch_size < 1 or args.hi_size < 224 or args.workers < 0
            or args.warmup_epochs < 0 or args.early_stopping_patience < 1):
        parser.error('Require epochs,batch-size >=1, hi-size >=224, workers,warmup-epochs >=0')
    if (not 0 < args.ema_decay < 1 or args.lora_rank < 1 or args.aux_weight < 0
            or not 0 <= args.branch_drop < 1):
        parser.error('Require 0<ema-decay<1, lora-rank>=1, aux-weight>=0, 0<=branch-drop<1')
    seed_all(args.seed)
    device = device_for(args.device)
    front_end = not args.no_front_end
    train = ImageManifest(args.train_csv, args.data_root, hi_size=args.hi_size, training=True,
                          use_fov=not args.no_fov, task=args.task, front_end=front_end)
    val = ImageManifest(args.val_csv, args.data_root, classes=train.classes, hi_size=args.hi_size,
                        use_fov=not args.no_fov, task=args.task, front_end=front_end)
    if len(train.classes) < 2:
        raise ValueError('Training requires at least two classes')
    if {p for p, _ in train.samples} & {p for p, _ in val.samples}:
        raise ValueError('Train/validation image paths overlap')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output/'best.pt').exists() or (output/'history.jsonl').exists():
        raise FileExistsError('Output already contains a run; use a new --output directory')
    stats = {'train': train.resolution_stats(), 'val': val.resolution_stats()}
    print(json.dumps({'device': str(device), 'resolution_stats': stats}, ensure_ascii=False))
    (output/'resolution_stats.json').write_text(json.dumps(stats, indent=2))
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    # Persist actual memberships and mapping, independently of future CSV edits.
    splits = {name: [{'path': str(p), 'label': y} for p, y in ds.samples] for name, ds in [('train', train), ('val', val)]}
    (output/'splits.json').write_text(json.dumps({'classes': train.classes, **splits}, indent=2))
    loaders = [
        create_dataloader(ds, batch_size=args.batch_size, shuffle=(ds is train),
                          workers=args.workers, device=device, seed=args.seed + index)
        for index, ds in enumerate((train, val))
    ]
    model = GlanceFocusViT(len(train.classes), fusion=args.fusion, crop_ratio=args.crop_ratio,
                           use_fov=not args.no_fov, region_method=args.region_method,
                           attention_method=args.attention_method, alpha=args.alpha,
                           tuning=args.tuning, lora_rank=args.lora_rank,
                           cosine_head=not args.no_cosine_head,
                           branch_drop=args.branch_drop,
                           aux_weight=args.aux_weight,
                           pretrained=not args.no_pretrained).to(device)
    external_selector = load_selector(args.selector, train.classes, device) if args.selector else None
    if external_selector is not None:
        external_selector.attention_method = args.attention_method
    # [C] EMA weights drive both the region selector and the inference model.
    # gate/concat without an explicit E1 selector pick the region with the EMA
    # model; local/E2 keeps the frozen E1 selector.
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    if args.fusion == 'local':
        train_selector = external_selector
    elif args.fusion in {'concat', 'gate'}:
        train_selector = external_selector or (ema.module if ema is not None else None)
    else:
        train_selector = None
    infer_model = ema.module if ema is not None else model
    val_selector = external_selector if args.fusion == 'local' else None
    weights = None
    if args.class_weights:
        counts = np.bincount([y for _, y in train.samples], minlength=len(train.classes))
        weights = torch.tensor(len(train)/(len(counts)*counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs*len(loaders[0])
    warm_steps = min(args.warmup_epochs*len(loaders[0]), total_steps)
    best, step, epochs_without_improvement = -1., 0, 0
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        for hi, mask, target, _ in loaders[0]:
            factor = ((step+1)/warm_steps if step < warm_steps else
                      .5*(1+math.cos(math.pi*(step-warm_steps)/max(1, total_steps-warm_steps-1))))
            for group in optimizer.param_groups:
                group['lr'] = args.lr*factor
            optimizer.zero_grad(set_to_none=True)
            out = model(hi.to(device), mask.to(device), train_selector)
            loss = model.loss(out, target.to(device), criterion)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step += 1
            loss_sum += loss.item()*len(target)
        y, prob, _ = predict(infer_model, loaders[1], device, val_selector)
        report = metrics(y, prob, len(train.classes), class_names=train.classes)
        row = dict(epoch=epoch+1, train_loss=loss_sum/len(train), **report)
        print(json.dumps(row), flush=True)
        with (output/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        checkpoint = dict(model=model.state_dict(),
                          model_ema=ema.module.state_dict() if ema is not None else None,
                          model_config=model.config, classes=train.classes, task=args.task,
                          hi_size=args.hi_size, front_end=front_end, ema_decay=args.ema_decay,
                          epoch=epoch+1, val_metrics=report, seed=args.seed,
                          optimizer=optimizer.state_dict(), train_args=vars(args), splits=splits,
                          selector_checkpoint=str(Path(args.selector).resolve()) if args.selector else None)
        torch.save(checkpoint, output/'last.pt')
        if report['macro_f1'] > best:
            best = report['macro_f1']
            epochs_without_improvement = 0
            torch.save(checkpoint, output/'best.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping: validation Macro-F1 did not improve for {args.early_stopping_patience} consecutive epochs.")
                break
    print(f'Best validation Macro-F1: {best:.6f}; checkpoint: {output / "best.pt"}')


if __name__ == '__main__':
    main()
