"""Train Glance→Focus ViT v2 (improved ROI selection) — gate/concat/global/local.

Same recipe as train.py (front-end [A], LoRA, cosine head, EMA, aux loss) but
uses the v2 model whose ROI selection is class-aware (Grad-CAM / grad-rollout),
winsorised against register-token noise, and adaptively scaled. Checkpoints carry
model_config['roi_version']='v2' so evaluate.py rebuilds the v2 model.

Example:
 python train_v2.py --train-csv data/train.csv --val-csv data/val.csv --fusion gate --output runs/E4v2
 python train_v2.py ... --attention-method gradrollout --region-method adaptive --roi-coverage 0.6
Then evaluate exactly like E4 (evaluate.py auto-detects the v2 checkpoint):
 python evaluate.py --checkpoint runs/E4v2/best.pt --csv data/test.csv --output results/E4v2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model_v2 import GlanceFocusViTV2, ModelEMA
from dataset import ImageManifest
from dataloader import create_dataloader
from train import device_for, metrics, predict, seed_all


def load_selector_v2(path, classes, device):
    """Load a frozen global v2 selector for E2/local (grad ROI needs region_map)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt['model_config'].get('roi_version') != 'v2':
        raise ValueError('v2 local/E2 requires a v2 global (E1) selector checkpoint')
    model = GlanceFocusViTV2.from_config(ckpt['model_config'], pretrained=False)
    state = ckpt['model_ema'] if ckpt.get('model_ema') is not None else ckpt['model']
    model.load_state_dict(state)
    if model.fusion != 'global' or ckpt['classes'] != classes:
        raise ValueError('Selector must be a global/E1 v2 checkpoint with the same classes')
    return model.to(device).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='runs/E4v2')
    parser.add_argument('--task', choices=['binary', 'multiclass'], default='binary')
    parser.add_argument('--fusion', choices=['global', 'local', 'concat', 'gate'], default='gate')
    parser.add_argument('--selector', help='Frozen v2 global/E1 checkpoint; required for local/E2')
    parser.add_argument('--crop-ratio', type=float, default=.5)
    parser.add_argument('--no-fov', action='store_true')
    parser.add_argument('--no-front-end', action='store_true', help='[A] disable FoV-crop + Shades-of-Gray')
    parser.add_argument('--attention-method', choices=['rollout', 'last', 'gradcam', 'gradrollout'],
                        default='gradcam', help='ROI map: gradcam/gradrollout are class-aware (v2)')
    parser.add_argument('--region-method', choices=['window', 'threshold', 'adaptive'], default='adaptive')
    parser.add_argument('--roi-coverage', type=float, default=.6, help='adaptive: target attention-mass fraction')
    parser.add_argument('--no-suppress-outliers', action='store_true', help='keep register-token peaks in the map')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--tuning', choices=['lora', 'adapter', 'full', 'frozen'], default='lora')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--no-cosine-head', action='store_true')
    parser.add_argument('--branch-drop', type=float, default=.2)
    parser.add_argument('--aux-weight', type=float, default=.3)
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--hi-size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-pretrained', action='store_true', help='For offline smoke checks only')
    args = parser.parse_args()
    if args.fusion == 'local' and not args.selector:
        parser.error('E2/local requires --selector pointing to a frozen v2 E1 checkpoint')
    if args.attention_method in {'gradcam', 'gradrollout'} and args.no_ema and args.fusion != 'global' and not args.selector:
        parser.error('grad ROI needs EMA (or --selector) as a separate frozen selector; do not pass --no-ema')
    if (args.epochs < 1 or args.batch_size < 1 or args.hi_size < 224 or args.workers < 0
            or args.warmup_epochs < 0 or args.early_stopping_patience < 1):
        parser.error('Require epochs,batch-size >=1, hi-size >=224, workers,warmup-epochs >=0')
    if not 0 < args.ema_decay < 1 or args.lora_rank < 1 or args.aux_weight < 0 or not 0 <= args.branch_drop < 1:
        parser.error('Require 0<ema-decay<1, lora-rank>=1, aux-weight>=0, 0<=branch-drop<1')
    if not 0 < args.roi_coverage <= 1:
        parser.error('roi-coverage must be in (0,1]')
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
    print(json.dumps({'device': str(device), 'model': 'v2', 'roi': args.attention_method,
                      'region': args.region_method, 'resolution_stats': stats}, ensure_ascii=False))
    (output/'resolution_stats.json').write_text(json.dumps(stats, indent=2))
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    splits = {name: [{'path': str(p), 'label': y} for p, y in ds.samples] for name, ds in [('train', train), ('val', val)]}
    (output/'splits.json').write_text(json.dumps({'classes': train.classes, **splits}, indent=2))
    loaders = [
        create_dataloader(ds, batch_size=args.batch_size, shuffle=(ds is train),
                          workers=args.workers, device=device, seed=args.seed + index)
        for index, ds in enumerate((train, val))
    ]
    model = GlanceFocusViTV2(len(train.classes), fusion=args.fusion, crop_ratio=args.crop_ratio,
                             use_fov=not args.no_fov, region_method=args.region_method,
                             attention_method=args.attention_method, alpha=args.alpha,
                             tuning=args.tuning, lora_rank=args.lora_rank,
                             cosine_head=not args.no_cosine_head, branch_drop=args.branch_drop,
                             aux_weight=args.aux_weight, suppress_outliers=not args.no_suppress_outliers,
                             roi_coverage=args.roi_coverage, pretrained=not args.no_pretrained).to(device)
    external_selector = load_selector_v2(args.selector, train.classes, device) if args.selector else None
    if external_selector is not None:
        external_selector.attention_method = args.attention_method
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
