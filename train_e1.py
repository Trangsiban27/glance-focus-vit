"""Train the E1 baseline (ViT-B/16 + LoRA + cosine head, global view only).

Dedicated E1 trainer, separate from train.py (which trains E4/gate and the other
fusion modes). Same "nền chung" recipe — [A] front-end, [B] LoRA, [E] cosine
head, EMA weights — but no local branch. The checkpoint it writes is loadable by
the shared evaluate.py, so the E1 and E4 comparison tables come from one
evaluator.

Example:
 python train_e1.py --train-csv data/train.csv --val-csv data/val.csv --output runs/E1
Then evaluate exactly like E4:
 python evaluate.py --checkpoint runs/E1/best.pt --csv data/test.csv --output results/E1
 python evaluate.py --checkpoint runs/E4/best.pt --csv data/test.csv --output results/E4 \
     --compare-predictions results/E1/predictions.json      # paired McNemar E1 vs E4
Paths in manifests are relative to --data-root (default: manifest directory).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model_e1 import GlanceFocusE1, ModelEMA
from dataset import ImageManifest
from dataloader import create_dataloader
from train import device_for, metrics, predict, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='runs/E1')
    parser.add_argument('--task', choices=['binary', 'multiclass'], default='binary',
                        help='Default task: Normal vs Abnormal; all non-Normal CSV labels map to Abnormal')
    parser.add_argument('--no-fov', action='store_true')
    parser.add_argument('--no-front-end', action='store_true', help='[A] disable FoV-crop + Shades-of-Gray front-end')
    parser.add_argument('--tuning', choices=['lora', 'adapter', 'full', 'frozen'], default='lora')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--no-cosine-head', action='store_true', help='[E] use a plain Linear head instead of cosine')
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--no-ema', action='store_true')
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
    parser.add_argument('--class-weights', dest='class_weights', action='store_true', default=True,
                        help='Use inverse-frequency class weights (enabled by default)')
    parser.add_argument('--no-class-weights', dest='class_weights', action='store_false',
                        help='Disable inverse-frequency class weights')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-pretrained', action='store_true', help='For offline smoke checks; not for the pretrained experiments')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.hi_size < 224 or args.workers < 0
            or args.warmup_epochs < 0 or args.early_stopping_patience < 1):
        parser.error('Require epochs,batch-size >=1, hi-size >=224, workers,warmup-epochs >=0')
    if not 0 < args.ema_decay < 1 or args.lora_rank < 1:
        parser.error('Require 0<ema-decay<1 and lora-rank>=1')
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
    print(json.dumps({'device': str(device), 'model': 'E1', 'resolution_stats': stats}, ensure_ascii=False))
    (output/'resolution_stats.json').write_text(json.dumps(stats, indent=2))
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    splits = {name: [{'path': str(p), 'label': y} for p, y in ds.samples] for name, ds in [('train', train), ('val', val)]}
    (output/'splits.json').write_text(json.dumps({'classes': train.classes, **splits}, indent=2))
    loaders = [
        create_dataloader(ds, batch_size=args.batch_size, shuffle=(ds is train),
                          workers=args.workers, device=device, seed=args.seed + index)
        for index, ds in enumerate((train, val))
    ]
    model = GlanceFocusE1(len(train.classes), use_fov=not args.no_fov, tuning=args.tuning,
                          lora_rank=args.lora_rank, cosine_head=not args.no_cosine_head,
                          pretrained=not args.no_pretrained).to(device)
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    infer_model = ema.module if ema is not None else model
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
            out = model(hi.to(device), mask.to(device))
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
        y, prob, _ = predict(infer_model, loaders[1], device)
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
                          selector_checkpoint=None)
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
