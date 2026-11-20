"""Train GF-ViT Level 1 A1.1/A1.2/A1.3 and export validation diagnostics.

Examples (run from the project root):
    python train_a1.py --experiment a1.1 --seed 42
    python train_a1.py --experiment a1.2 --seed 42 \
        --selector-checkpoint experiments/gf_vit_level1/seed42/a1_1_global/selector_seed42.pt
    python train_a1.py --experiment a1.3 --seed 42

Repeat for seeds 46 and 49. A1.2 requires the frozen A1.1 selector from the
same seed. Test CSVs are intentionally not accepted by this training script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F

from dataloader import create_dataloader
from dataset import ImageManifest
from model_a1 import GFViTLevel1A1, ModelEMA
from train import device_for, metrics, seed_all


EXPERIMENTS = {
    'a1.1': ('global', 'a1_1_global'),
    'a1.2': ('local', 'a1_2_local'),
    'a1.3': ('gate', 'a1_3_gate_online'),
}
PRETRAINED_ID = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'
BRANCH_DROP = .2
AUX_WEIGHT = .3
GATE_HIDDEN = 256


def _git_commit():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_yaml(path, content):
    """Write readable YAML, falling back to JSON (valid YAML 1.2)."""
    content = _jsonable(content)
    try:
        import yaml
    except ImportError:
        Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        Path(path).write_text(
            yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding='utf-8')


def _write_json(path, content):
    Path(path).write_text(json.dumps(_jsonable(content), indent=2, ensure_ascii=False) + '\n',
                          encoding='utf-8')


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _metrics_with_pr_auc(labels, probabilities, class_names):
    report = metrics(labels, probabilities, len(class_names), class_names=class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, np.asarray(probabilities).argmax(1), labels=list(range(len(class_names))),
        average=None, zero_division=0)
    report['per_class'] = {
        name: {'precision': float(precision[index]), 'recall': float(recall[index]),
               'f1': float(f1[index]), 'support': int(support[index])}
        for index, name in enumerate(class_names)
    }
    if len(class_names) == 2 and len(np.unique(labels)) == 2:
        positive = class_names.index('Abnormal') if 'Abnormal' in class_names else 1
        binary_labels = (np.asarray(labels) == positive).astype(np.int64)
        report['pr_auc_abnormal'] = float(average_precision_score(
            binary_labels, np.asarray(probabilities)[:, positive]))
    else:
        report['pr_auc_abnormal'] = None
    return report


def _load_selector(path, classes, seed, device, args, front_end, use_fov):
    path = Path(path).expanduser().resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('model_config', {})
    if checkpoint.get('experiment') != 'a1.1':
        raise ValueError('A1.2 requires an A1.1 checkpoint produced by train_a1.py')
    if config.get('fusion') != 'global':
        raise ValueError('A1.2 selector checkpoint must be an A1.1/global model')
    if checkpoint.get('classes') != list(classes):
        raise ValueError('Selector class mapping must match the A1.2 training data')
    if checkpoint.get('seed') != seed:
        raise ValueError('A1.2 selector must come from the same seed')
    if config.get('attention_method') != args.attention_method:
        raise ValueError('Selector attention_method must match the A1.2 configuration')
    if config.get('region_method') != args.region_method:
        raise ValueError('Selector region_method must match the A1.2 configuration')
    if float(config.get('crop_ratio', -1)) != args.crop_ratio:
        raise ValueError('Selector crop_ratio must match the A1.2 configuration')
    if float(config.get('alpha', -1)) != args.alpha:
        raise ValueError('Selector alpha must match the A1.2 configuration')
    if config.get('backbone') != args.backbone:
        raise ValueError('Selector backbone must match the A1.2 configuration')
    if bool(config.get('use_fov')) != use_fov:
        raise ValueError('Selector FoV setting must match the A1.2 configuration')
    if int(checkpoint.get('hi_size', -1)) != args.hi_size:
        raise ValueError('Selector high-resolution input size must match A1.2')
    if bool(checkpoint.get('front_end')) != front_end:
        raise ValueError('Selector front-end preprocessing must match A1.2')
    selector = GFViTLevel1A1.from_config(config, pretrained=False)
    state = checkpoint.get('model_ema') or checkpoint['model']
    selector.load_state_dict(state)
    selector = selector.to(device).eval().requires_grad_(False)
    return selector, checkpoint, str(path)


def _build_model(args, num_classes, fusion, device):
    model = GFViTLevel1A1(
        num_classes=num_classes,
        fusion=fusion,
        crop_ratio=args.crop_ratio,
        use_fov=not args.no_fov,
        backbone=args.backbone,
        gate_hidden=GATE_HIDDEN,
        region_method=args.region_method,
        attention_method=args.attention_method,
        alpha=args.alpha,
        tuning=args.tuning,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_hidden=args.adapter_hidden,
        cosine_head=not args.no_cosine_head,
        tau=args.tau,
        learnable_tau=args.learnable_tau,
        branch_drop=BRANCH_DROP,
        aux_weight=AUX_WEIGHT,
        view_embed=False,
        pretrained=not args.no_pretrained,
    )
    return model.to(device)


@torch.no_grad()
def _predict_metrics(model, loader, device, selector, classes):
    model.eval()
    if selector is not None:
        selector.eval()
    labels, probabilities = [], []
    for hi, mask, target, _ in loader:
        output = model(hi.to(device), mask.to(device), selector)
        labels.extend(target.tolist())
        probabilities.extend(output['logits'].softmax(-1).cpu().tolist())
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return labels, probabilities, _metrics_with_pr_auc(labels, probabilities, classes)


def _learning_rate_factor(step, warm_steps, total_steps):
    if warm_steps and step < warm_steps:
        return (step + 1) / warm_steps
    decay_steps = max(1, total_steps - warm_steps)
    progress = min(1., max(0., (step - warm_steps) / decay_steps))
    return .5 * (1 + math.cos(math.pi * progress))


def _make_checkpoint(model, ema, optimizer, args, classes, epoch, val_report,
                     train_data, val_data, selector_path, model_id):
    return {
        'format_version': 1,
        'experiment': args.experiment,
        'model': model.state_dict(),
        'model_ema': ema.module.state_dict() if ema is not None else None,
        'model_config': dict(model.config),
        'classes': list(classes),
        'label_map': {name: index for index, name in enumerate(classes)},
        'task': 'binary',
        'seed': args.seed,
        'epoch': epoch,
        'val_metrics': val_report,
        'metric_on_validation': 'macro_f1',
        'tie_break': 'higher_validation_roc_auc_then_earlier_epoch',
        'pretrained_id': args.backbone,
        'selector_checkpoint': selector_path,
        'selector_id': Path(selector_path).stem if selector_path else None,
        'model_id': model_id,
        'use_ema': ema is not None,
        'ema_decay': args.ema_decay if ema is not None else None,
        'input_size': {'global': 224, 'high_resolution': args.hi_size},
        'hi_size': args.hi_size,
        'crop_ratio': args.crop_ratio,
        'front_end': not args.no_front_end,
        'optimizer': optimizer.state_dict(),
        'train_config': _jsonable(vars(args)),
        'splits': {
            'train': [{'path': str(path), 'label': int(label)}
                      for path, label in train_data.samples],
            'val': [{'path': str(path), 'label': int(label)}
                    for path, label in val_data.samples],
        },
        'git_commit': _git_commit(),
    }


def _write_environment(path, device):
    packages = {
        'torchvision': 'torchvision', 'numpy': 'numpy', 'Pillow': 'Pillow',
        'scipy': 'scipy', 'scikit-learn': 'scikit-learn', 'timm': 'timm',
    }
    lines = [
        f'python={platform.python_version()}',
        f'torch={torch.__version__}',
        *(f'{name}={_package_version(package)}' for name, package in packages.items()),
        f'device={device}',
        f'cuda={torch.version.cuda or "unavailable"}',
        f'git_commit={_git_commit() or "unavailable"}',
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return 'unavailable'


def _image_ids(dataset, data_root):
    root = Path(data_root).expanduser().resolve() if data_root else dataset.manifest.parent
    result = {}
    for path, _ in dataset.samples:
        try:
            image_id = Path(path).resolve().relative_to(root).as_posix()
        except ValueError:
            image_id = Path(path).resolve().as_posix()
        result[str(path)] = image_id
    return result


def _save_overlay(hi, box, attention, output_dir, image_id, y_true, y_pred):
    if box is None:
        return
    image = Image.fromarray(
        (hi.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8))
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(round(x1), 0), width - 1)
    y1 = min(max(round(y1), 0), height - 1)
    x2 = min(max(round(x2), x1 + 1), width)
    y2 = min(max(round(y2), y1 + 1), height)
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 40, 40), width=3)
    crop = image.crop((x1, y1, x2, y2)).resize((width, height), Image.Resampling.BILINEAR)
    if attention is not None:
        attention = attention.detach().float().cpu()
        if attention.ndim == 3:
            attention = attention[0]
        attention = F.interpolate(attention[None, None], (height, width), mode='bilinear',
                                  align_corners=False)[0, 0].numpy()
        attention -= attention.min()
        attention /= max(float(attention.max()), 1e-8)
        heat = np.zeros((height, width, 3), dtype=np.uint8)
        heat[..., 0] = (attention * 255).astype(np.uint8)
        heat[..., 1] = (attention * 48).astype(np.uint8)
        heat_overlay = Image.blend(image, Image.fromarray(heat), .42)
        ImageDraw.Draw(heat_overlay).rectangle((x1, y1, x2 - 1, y2 - 1),
                                               outline=(255, 255, 255), width=3)
    else:
        heat_overlay = image.copy()
        ImageDraw.Draw(heat_overlay).rectangle((x1, y1, x2 - 1, y2 - 1),
                                               outline=(255, 40, 40), width=3)
    panel = Image.new('RGB', (width * 3, height + 42), 'white')
    panel.paste(boxed, (0, 0))
    panel.paste(heat_overlay, (width, 0))
    panel.paste(crop, (width * 2, 0))
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((6, height + 6), 'Image + ROI', fill=(0, 0, 0))
    panel_draw.text((width + 6, height + 6), 'Attention + ROI', fill=(0, 0, 0))
    panel_draw.text((width * 2 + 6, height + 6), 'Selected crop', fill=(0, 0, 0))
    panel_draw.text((6, height + 23), f'{image_id} | true={y_true} pred={y_pred}', fill=(0, 0, 0))
    digest = hashlib.sha1(image_id.encode('utf-8')).hexdigest()[:8]
    safe_name = Path(image_id).stem.replace(' ', '_')[:48] or 'image'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.save(output_dir / f'{safe_name}_{digest}.jpg', quality=90)


def _binary_probabilities(logits):
    probabilities = logits.softmax(-1)
    return probabilities


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()


@torch.no_grad()
def _export_predictions(model, dataset, loader, selector, args, output_dir,
                        checkpoint_id, split='valid'):
    model.eval()
    if selector is not None:
        selector.eval()
    ids = _image_ids(dataset, args.data_root)
    rows = []
    labels, probs_fused = [], []
    probs_global, probs_local, probs_global_only = [], [], []
    fused_preds, global_preds, local_preds = [], [], []
    gate_vectors, gate_labels, gate_predictions, gate_ratios = [], [], [], []
    gate_image_ids = []
    abnormal_index = dataset.classes.index('Abnormal')
    overlay_dir = Path(output_dir) / 'overlays' / split
    model_forward_seconds = 0.
    for hi, mask, target, paths in loader:
        hi_device, mask_device = hi.to(args.device_obj), mask.to(args.device_obj)
        _synchronize(args.device_obj)
        forward_start = time.perf_counter()
        output = model.forward_diagnostics(hi_device, mask_device, selector)
        _synchronize(args.device_obj)
        model_forward_seconds += time.perf_counter() - forward_start
        fused_logits = output.get('logits_fused', output['logits'])
        fused_prob = _binary_probabilities(fused_logits).cpu().numpy()
        aux_global_logits = output.get('logits_aux_global')
        aux_local_logits = output.get('logits_aux_local')
        global_prob = (_binary_probabilities(aux_global_logits).cpu().numpy()
                       if aux_global_logits is not None else None)
        local_prob = (_binary_probabilities(aux_local_logits).cpu().numpy()
                      if aux_local_logits is not None else None)
        boxes = output['boxes'].cpu().tolist() if output['boxes'] is not None else [None] * len(paths)
        gates = output['gate'].cpu().numpy() if output['gate'] is not None else None
        attention = output['attention']
        local_ratios = (output['local_contrib_ratio'].cpu().tolist()
                        if output.get('local_contrib_ratio') is not None else None)
        delta_logits = output['logit_delta'].cpu().numpy() if output.get('logit_delta') is not None else None
        global_only_logits = output.get('logits_global_only')
        global_only_prob = (_binary_probabilities(global_only_logits).cpu().numpy()
                            if global_only_logits is not None else None)
        z_global = output['z_global'].cpu().numpy() if output.get('z_global') is not None else None
        z_local = output['z_local'].cpu().numpy() if output.get('z_local') is not None else None
        s_value = float(output['s'].cpu()) if output.get('s') is not None else None

        for i, (path, label) in enumerate(zip(paths, target.tolist())):
            image_id = ids[str(Path(path))]
            y_pred = int(np.argmax(fused_prob[i]))
            row = {
                'image_id': image_id,
                'split': split,
                'seed': args.seed,
                'model': args.experiment,
                'y_true': int(label),
                'prob_abnormal': float(fused_prob[i, abnormal_index]),
                'y_pred': y_pred,
                'correct': int(y_pred == label),
                'checkpoint_id': checkpoint_id,
            }
            if args.experiment == 'a1.1':
                row['p_global'] = float(fused_prob[i, abnormal_index])
            elif args.experiment == 'a1.2':
                row['p_local'] = float(fused_prob[i, abnormal_index])
            else:
                row['p_fused'] = float(fused_prob[i, abnormal_index])
            overlay_case = None
            if boxes[i] is not None:
                row.update(roi_x1=float(boxes[i][0]), roi_y1=float(boxes[i][1]),
                           roi_x2=float(boxes[i][2]), roi_y2=float(boxes[i][3]))
            if z_global is not None:
                row['z_global_norm'] = float(np.linalg.norm(z_global[i]))
            if z_local is not None:
                row['z_local_norm'] = float(np.linalg.norm(z_local[i]))
            if global_prob is not None and local_prob is not None:
                pred_global = int(np.argmax(global_prob[i]))
                pred_local = int(np.argmax(local_prob[i]))
                row.update(
                    logits_fused_normal=float(fused_logits[i, 0]),
                    logits_fused_abnormal=float(fused_logits[i, abnormal_index]),
                    p_global_aux=float(global_prob[i, abnormal_index]),
                    p_local_aux=float(local_prob[i, abnormal_index]),
                    prob_global_aux=float(global_prob[i, abnormal_index]),
                    prob_local_aux=float(local_prob[i, abnormal_index]),
                    pred_global_aux=pred_global,
                    pred_local_aux=pred_local,
                    logits_aux_global_normal=float(aux_global_logits[i, 0]),
                    logits_aux_global_abnormal=float(aux_global_logits[i, abnormal_index]),
                    logits_aux_local_normal=float(aux_local_logits[i, 0]),
                    logits_aux_local_abnormal=float(aux_local_logits[i, abnormal_index]),
                    gate_mean=float(gates[i].mean()),
                    gate_std=float(gates[i].std()),
                    s=s_value,
                    local_contrib_ratio=float(local_ratios[i]),
                    logit_delta_norm=float(np.linalg.norm(delta_logits[i])),
                    logit_delta_normal=float(delta_logits[i, 0]),
                    logit_delta_abnormal=float(delta_logits[i, abnormal_index]),
                )
                if global_only_prob is not None:
                    row['p_global_only_ablation'] = float(global_only_prob[i, abnormal_index])
                    probs_global_only.append(global_only_prob[i])
                probs_global.append(global_prob[i])
                probs_local.append(local_prob[i])
                global_preds.append(pred_global)
                local_preds.append(pred_local)
                fused_preds.append(y_pred)
                gate_vectors.append(gates[i].astype(np.float32, copy=True))
                gate_labels.append(int(label))
                gate_predictions.append(y_pred)
                gate_ratios.append(float(local_ratios[i]))
                gate_image_ids.append(image_id)
                global_ok = pred_global == int(label)
                local_ok = pred_local == int(label)
                fused_ok = y_pred == int(label)
                overlay_case = (
                    f'global_{"correct" if global_ok else "wrong"}_'
                    f'local_{"correct" if local_ok else "wrong"}_'
                    f'fusion_{"correct" if fused_ok else "wrong"}')
            if args.save_overlays:
                overlay_group = overlay_dir / overlay_case if overlay_case else overlay_dir
                attention_i = attention[i] if attention is not None else None
                _save_overlay(hi[i], boxes[i], attention_i, overlay_group,
                              image_id, int(label), y_pred)
            rows.append(row)
            labels.append(int(label))
            probs_fused.append(fused_prob[i])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f'predictions_{split}.csv', rows)
    labels = np.asarray(labels, dtype=np.int64)
    probs_fused = np.asarray(probs_fused, dtype=np.float64)
    report = _metrics_with_pr_auc(labels, probs_fused, dataset.classes)
    report.update(experiment=args.experiment, seed=args.seed, split=split,
                  count=len(dataset), checkpoint_id=checkpoint_id,
                  classes=dataset.classes, positive_class='Abnormal',
                  model_forward_elapsed_seconds=model_forward_seconds,
                  model_forward_ms_per_image=1000 * model_forward_seconds / len(dataset),
                  frozen_selector_included_in_latency=selector is not None,
                  inference_scope={
                      'a1.1': 'global ViT and classifier',
                      'a1.2': 'frozen A1.1 selector, ROI selection, local ViT and classifier',
                      'a1.3': 'online global attention and ROI selection, global/local ViT and gate',
                  }[args.experiment])
    _write_json(output_dir / f'metrics_{split}.json', report)

    box_rows = []
    for row in rows:
        if 'roi_x1' in row:
            box_rows.append({
                'image_id': row['image_id'], 'split': split, 'seed': args.seed,
                'model': args.experiment,
                'box_xyxy_hi': [row['roi_x1'], row['roi_y1'], row['roi_x2'], row['roi_y2']],
                'coordinate_space': 'hi_pixel_edges',
                'input_width': args.hi_size, 'input_height': args.hi_size,
                'selector_checkpoint_id': args.selector_id,
                'crop_ratio': args.crop_ratio,
                'fov_enabled': not args.no_fov,
                'rollout_method': args.attention_method,
                'roi_jitter_applied': False,
            })
    if box_rows:
        _write_json(output_dir / f'boxes_{split}.json', box_rows)

    if probs_global:
        probs_global = np.asarray(probs_global)
        probs_local = np.asarray(probs_local)
        global_report = _metrics_with_pr_auc(labels, probs_global, dataset.classes)
        local_report = _metrics_with_pr_auc(labels, probs_local, dataset.classes)
        diagnostic_report = {
            'fused': report, 'aux_global': global_report, 'aux_local': local_report,
        }
        if probs_global_only:
            ablation_report = _metrics_with_pr_auc(
                labels, np.asarray(probs_global_only, dtype=np.float64), dataset.classes)
            diagnostic_report['fusion_without_local_ablation'] = ablation_report
            diagnostic_report['ablation'] = (
                'A1.3 classifier evaluated on z_global, same checkpoint/head; no retraining')
        _write_json(output_dir / f'diagnostic_metrics_{split}.json', diagnostic_report)
        correctness = np.zeros((2, 2), dtype=np.int64)
        correctness_by_class = {
            name: np.zeros((2, 2), dtype=np.int64) for name in dataset.classes
        }
        fusion_counts_by_class = {
            name: {
                'fusion_rescue_success': 0, 'fusion_missed_rescue': 0,
                'fusion_rescue_denominator': 0, 'fusion_harmed_global': 0,
                'fusion_harm_denominator': 0, 'fusion_synergy': 0,
            } for name in dataset.classes
        }
        rescue_denominator = 0
        fusion_rescue_success = fusion_missed_rescue = 0
        harm_denominator = fusion_harmed_global = 0
        fusion_synergy = 0
        for y, pg, pl, pf in zip(labels, global_preds, local_preds, fused_preds):
            g_ok, l_ok, f_ok = pg == y, pl == y, pf == y
            class_counts = fusion_counts_by_class[dataset.classes[int(y)]]
            correctness[int(g_ok), int(l_ok)] += 1
            correctness_by_class[dataset.classes[int(y)]][int(g_ok), int(l_ok)] += 1
            if not g_ok and l_ok:
                rescue_denominator += 1
                class_counts['fusion_rescue_denominator'] += 1
                if f_ok:
                    fusion_rescue_success += 1
                    class_counts['fusion_rescue_success'] += 1
                else:
                    fusion_missed_rescue += 1
                    class_counts['fusion_missed_rescue'] += 1
            if g_ok and not l_ok:
                harm_denominator += 1
                class_counts['fusion_harm_denominator'] += 1
                if not f_ok:
                    fusion_harmed_global += 1
                    class_counts['fusion_harmed_global'] += 1
            if not g_ok and not l_ok and f_ok:
                fusion_synergy += 1
                class_counts['fusion_synergy'] += 1
        for class_counts in fusion_counts_by_class.values():
            rescue_n = class_counts['fusion_rescue_denominator']
            harm_n = class_counts['fusion_harm_denominator']
            class_counts['fusion_rescue_efficiency'] = (
                class_counts['fusion_rescue_success'] / rescue_n if rescue_n else None)
            class_counts['fusion_harm_rate'] = (
                class_counts['fusion_harmed_global'] / harm_n if harm_n else None)
        rescue_rate = (fusion_rescue_success / rescue_denominator
                       if rescue_denominator else None)
        harm_rate = (fusion_harmed_global / harm_denominator if harm_denominator else None)
        global_correct = np.asarray(global_preds) == labels
        fused_correct = np.asarray(fused_preds) == labels
        global_fused_table = np.zeros((2, 2), dtype=np.int64)
        for global_ok, fused_ok in zip(global_correct, fused_correct):
            global_fused_table[int(global_ok), int(fused_ok)] += 1
        global_fused_discordant = int(np.sum(global_correct != fused_correct))
        global_fused_wins = int(np.sum(~global_correct & fused_correct))
        _write_json(output_dir / f'complementarity_{split}.json', {
            'correctness_table_rows_global_cols_local': correctness.tolist(),
            'correctness_table_by_class_rows_global_cols_local': {
                name: table.tolist() for name, table in correctness_by_class.items()
            },
            'fusion_counts_by_class': fusion_counts_by_class,
            'correctness_table_rows_global_cols_fused': global_fused_table.tolist(),
            'counts': {
                'fusion_rescue_success': fusion_rescue_success,
                'fusion_missed_rescue': fusion_missed_rescue,
                'fusion_rescue_denominator': rescue_denominator,
                'fusion_rescue_efficiency': rescue_rate,
                'fusion_harmed_global': fusion_harmed_global,
                'fusion_harm_denominator': harm_denominator,
                'fusion_harm_rate': harm_rate,
                'fusion_synergy': fusion_synergy,
            },
            'mcnemar_exact_pvalue_aux_global_vs_fused': (
                float(binomtest(global_fused_wins, global_fused_discordant, .5).pvalue)
                if global_fused_discordant else 1.0),
        })
        gate_rows = [{key: row.get(key) for key in (
            'image_id', 'split', 'seed', 'y_true', 'pred_global_aux', 'pred_local_aux',
            'y_pred', 'correct', 'gate_mean', 'gate_std', 's', 'local_contrib_ratio',
            'logit_delta_norm', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2')}
            for row in rows]
        _write_csv(output_dir / f'gate_stats_{split}.csv', gate_rows)
        gate_array = np.asarray(gate_vectors, dtype=np.float32)
        gate_labels_array = np.asarray(gate_labels, dtype=np.int64)
        gate_predictions_array = np.asarray(gate_predictions, dtype=np.int64)
        gate_ratio_array = np.asarray(gate_ratios, dtype=np.float32)
        np.savez_compressed(
            output_dir / f'gate_vectors_{split}.npz',
            image_ids=np.asarray(gate_image_ids),
            gate=gate_array, labels=gate_labels_array, predictions=gate_predictions_array,
            local_contrib_ratio=gate_ratio_array,
        )
        gate_summary = {
            'count': int(gate_array.shape[0]),
            'dimension': int(gate_array.shape[1]),
            'gate_mean': float(gate_array.mean()),
            'gate_std': float(gate_array.std()),
            'gate_percentiles_5_50_95': np.quantile(
                gate_array, [.05, .5, .95]).tolist(),
            'fraction_gate_below_0.05': float((gate_array < .05).mean()),
            'fraction_gate_above_0.95': float((gate_array > .95).mean()),
            'local_contrib_ratio_mean': float(gate_ratio_array.mean()),
            'local_contrib_ratio_std': float(gate_ratio_array.std()),
            's': float(rows[0]['s']) if rows else None,
            'by_class_and_correctness': {},
        }
        for class_index, class_name in enumerate(dataset.classes):
            for correct in (False, True):
                selected = ((gate_labels_array == class_index)
                            & ((gate_predictions_array == gate_labels_array) == correct))
                values = gate_array[selected]
                ratios = gate_ratio_array[selected]
                key = f'{class_name}|correct={str(correct).lower()}'
                gate_summary['by_class_and_correctness'][key] = {
                    'count': int(selected.sum()),
                    'gate_mean': float(values.mean()) if values.size else None,
                    'gate_std': float(values.std()) if values.size else None,
                    'gate_percentiles_5_50_95': (
                        np.quantile(values, [.05, .5, .95]).tolist() if values.size else None),
                    'fraction_gate_below_0.05': float((values < .05).mean()) if values.size else None,
                    'fraction_gate_above_0.95': float((values > .95).mean()) if values.size else None,
                    'local_contrib_ratio_mean': float(ratios.mean()) if ratios.size else None,
                    'local_contrib_ratio_std': float(ratios.std()) if ratios.size else None,
                }
        _write_json(output_dir / f'gate_summary_{split}.json', gate_summary)

    return rows, report


@torch.no_grad()
def _export_train_boxes(model, dataset, selector, args, output_dir):
    """Export deterministic, unjittered train ROIs from the frozen selector."""
    if selector is None:
        return
    prior = dataset.processor.training
    dataset.processor.training = False
    model.eval()
    selector.eval()
    loader = create_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, workers=args.workers,
        device=args.device_obj, seed=args.seed)
    ids = _image_ids(dataset, args.data_root)
    rows = []
    try:
        for hi, mask, target, paths in loader:
            hi_device, mask_device = hi.to(args.device_obj), mask.to(args.device_obj)
            global_view = F.interpolate(
                hi_device, (224, 224), mode='bilinear', align_corners=False, antialias=True)
            global_view = (global_view - .5) / .5
            _, attention = selector.encode_map(global_view)
            boxes, _ = model.select_boxes(attention, mask_device, *hi_device.shape[-2:])
            for path, label, box in zip(paths, target.tolist(), boxes.cpu().tolist()):
                rows.append({
                    'image_id': ids[str(Path(path))], 'split': 'train', 'seed': args.seed,
                    'model': args.experiment, 'y_true': int(label),
                    'box_xyxy_hi': [float(value) for value in box],
                    'coordinate_space': 'hi_pixel_edges',
                    'input_width': args.hi_size, 'input_height': args.hi_size,
                    'selector_checkpoint_id': args.selector_id,
                    'crop_ratio': args.crop_ratio, 'fov_enabled': not args.no_fov,
                    'rollout_method': args.attention_method, 'roi_jitter_applied': False,
                })
    finally:
        dataset.processor.training = prior
    _write_json(Path(output_dir) / 'boxes_train.json', rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument('--train-csv', default='data/Datos/train.csv')
    parser.add_argument('--val-csv', default='data/Datos/valid.csv')
    parser.add_argument('--data-root')
    parser.add_argument('--output', help='Exact run directory; default follows the A1 folder layout')
    parser.add_argument('--selector-checkpoint', help='Required for A1.2; frozen A1.1 checkpoint from same seed')
    parser.add_argument('--task', choices=['binary'], default='binary')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hi-size', type=int, default=448)
    parser.add_argument('--crop-ratio', type=float, default=.5,
                        help='Lock using validation before test evaluation')
    parser.add_argument('--no-fov', action='store_true')
    parser.add_argument('--no-front-end', action='store_true')
    parser.add_argument('--region-method', choices=['window', 'threshold'], default='window')
    parser.add_argument('--attention-method', choices=['rollout', 'last'], default='rollout')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--backbone', default=PRETRAINED_ID)
    parser.add_argument('--tuning', choices=['lora', 'adapter', 'full', 'frozen'], default='lora')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.)
    parser.add_argument('--adapter-hidden', type=int, default=64)
    parser.add_argument('--no-cosine-head', action='store_true')
    parser.add_argument('--tau', type=float, default=16.)
    parser.add_argument('--learnable-tau', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=.999)
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--label-smoothing', type=float, default=.1)
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--no-overlays', dest='save_overlays', action='store_false', default=True)
    args = parser.parse_args(argv)
    if args.experiment == 'a1.2' and not args.selector_checkpoint:
        parser.error('A1.2 requires --selector-checkpoint from A1.1 with the same seed')
    if args.experiment != 'a1.2' and args.selector_checkpoint:
        parser.error('--selector-checkpoint is only used for A1.2')
    if (args.epochs < 1 or args.batch_size < 1 or args.hi_size < 224 or args.workers < 0
            or args.warmup_epochs < 0 or args.early_stopping_patience < 1):
        parser.error('Require epochs,batch-size >=1, hi-size >=224, workers,warmup-epochs >=0')
    if (args.lora_rank < 1 or args.lora_alpha <= 0 or args.adapter_hidden < 1
            or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1
            or not 0 < args.ema_decay < 1 or not 0 < args.crop_ratio <= 1
            or not math.isfinite(args.tau) or args.tau <= 0):
        parser.error('Invalid model or optimization hyperparameter')
    return args


def main(argv=None):
    args = _parse_args(argv)
    seed_all(args.seed)
    device = device_for(args.device)
    args.device_obj = device
    fusion, experiment_dirname = EXPERIMENTS[args.experiment]
    output_dir = (Path(args.output).expanduser().resolve() if args.output else
                  Path('experiments/gf_vit_level1') / f'seed{args.seed}' / experiment_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'best.pt').exists() or (output_dir / 'history.jsonl').exists():
        raise FileExistsError(f'Run directory already contains a training run: {output_dir}')

    front_end = not args.no_front_end
    use_fov = not args.no_fov
    train_data = ImageManifest(
        args.train_csv, args.data_root, task=args.task, hi_size=args.hi_size,
        training=True, use_fov=use_fov, front_end=front_end)
    val_data = ImageManifest(
        args.val_csv, args.data_root, classes=train_data.classes, task=args.task,
        hi_size=args.hi_size, training=False, use_fov=use_fov, front_end=front_end)
    if len(train_data.classes) != 2 or 'Abnormal' not in train_data.classes:
        raise ValueError('A1 requires binary classes Normal and Abnormal')
    train_paths = {str(path) for path, _ in train_data.samples}
    val_paths = {str(path) for path, _ in val_data.samples}
    if train_paths & val_paths:
        raise ValueError('Train/validation image paths overlap')

    selector = selector_ckpt = selector_path = None
    if args.experiment == 'a1.2':
        selector, _, selector_path = _load_selector(
            args.selector_checkpoint, train_data.classes, args.seed, device,
            args, front_end, use_fov)
    args.selector_id = Path(selector_path).stem if selector_path else None

    model = _build_model(args, len(train_data.classes), fusion, device)
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    infer_model = ema.module if ema is not None else model
    train_loader = create_dataloader(
        train_data, batch_size=args.batch_size, shuffle=True, workers=args.workers,
        device=device, seed=args.seed)
    val_loader = create_dataloader(
        val_data, batch_size=args.batch_size, shuffle=False, workers=args.workers,
        device=device, seed=args.seed + 1)

    weights = None
    if args.class_weights:
        counts = np.bincount([label for _, label in train_data.samples], minlength=2)
        if np.any(counts == 0):
            raise ValueError(f'Cannot compute class weights; a class has no train images: {counts.tolist()}')
        weights = torch.tensor(len(train_data) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warm_steps = min(args.warmup_epochs * steps_per_epoch, total_steps)

    model_config = dict(model.config)
    resolved_config = {
        **vars(args),
        'device_obj': str(device),
        'output_dir_resolved': str(output_dir.resolve()),
        'train_csv_resolved': str(Path(args.train_csv).expanduser().resolve()),
        'val_csv_resolved': str(Path(args.val_csv).expanduser().resolve()),
        'data_root_resolved': (str(Path(args.data_root).expanduser().resolve())
                               if args.data_root else None),
        'selector_checkpoint_resolved': selector_path,
        'experiment': args.experiment,
        'fusion': fusion,
        'pretrained_id': args.backbone,
        'model_config': model_config,
        'classes': train_data.classes,
        'label_map': {name: index for index, name in enumerate(train_data.classes)},
        'front_end': front_end,
        'use_fov': use_fov,
        'branch_drop': BRANCH_DROP,
        'aux_weight': AUX_WEIGHT,
        'gate_hidden': GATE_HIDDEN,
        'view_embed': False,
        'checkpoint_selection': 'validation macro_f1; tie: validation roc_auc; tie: earliest epoch',
        'prediction_threshold': .5,
        'positive_class': 'Abnormal',
        'git_commit': _git_commit(),
    }
    _write_yaml(output_dir / 'resolved_config.yaml', resolved_config)
    _write_environment(output_dir / 'environment.txt', device)
    _write_json(output_dir / 'splits.json', {
        'classes': train_data.classes,
        'train': [{'path': str(path), 'label': int(label)} for path, label in train_data.samples],
        'val': [{'path': str(path), 'label': int(label)} for path, label in val_data.samples],
    })
    print(json.dumps({
        'device': str(device), 'experiment': args.experiment,
        'classes': train_data.classes,
        'train_count': len(train_data), 'val_count': len(val_data),
        'selector': selector_path,
        'resolution_stats': {'train': train_data.resolution_stats(),
                             'val': val_data.resolution_stats()},
    }, ensure_ascii=False), flush=True)

    best_rank = (-float('inf'), -float('inf'))
    best_epoch = None
    step = 0
    epochs_without_improvement = 0
    history_path = output_dir / 'history.jsonl'
    model_id = f'{args.experiment}_seed{args.seed}'
    for epoch in range(args.epochs):
        model.train()
        if selector is not None:
            selector.eval()
        loss_total = 0.
        gate_epoch_values, local_ratio_epoch_values = [], []
        for hi, mask, target, _ in train_loader:
            factor = _learning_rate_factor(step, warm_steps, total_steps)
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            output = model(hi.to(device, non_blocking=True),
                           mask.to(device, non_blocking=True), selector)
            if output.get('gate') is not None:
                gate_epoch_values.extend(
                    output['gate'].detach().float().cpu().reshape(-1).tolist())
                if output.get('local_contrib_ratio') is not None:
                    local_ratio_epoch_values.extend(
                        output['local_contrib_ratio'].detach().float().cpu().tolist())
            loss = model.loss(output, target.to(device, non_blocking=True), criterion)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            step += 1
            loss_total += loss.item() * len(target)

        _, _, val_report = _predict_metrics(
            infer_model, val_loader, device, selector, train_data.classes)
        row = {
            'epoch': epoch + 1,
            'train_loss': loss_total / len(train_data),
            's': float(model.s.detach().cpu()) if fusion == 'gate' else None,
            'train_gate_mean': float(np.mean(gate_epoch_values)) if gate_epoch_values else None,
            'train_gate_std': float(np.std(gate_epoch_values)) if gate_epoch_values else None,
            'train_gate_percentiles_5_50_95': (
                np.quantile(gate_epoch_values, [.05, .5, .95]).tolist()
                if gate_epoch_values else None),
            'train_fraction_gate_below_0.05': (
                float(np.mean(np.asarray(gate_epoch_values) < .05)) if gate_epoch_values else None),
            'train_fraction_gate_above_0.95': (
                float(np.mean(np.asarray(gate_epoch_values) > .95)) if gate_epoch_values else None),
            'train_local_contrib_ratio_mean': (
                float(np.mean(local_ratio_epoch_values)) if local_ratio_epoch_values else None),
            'train_local_contrib_ratio_std': (
                float(np.std(local_ratio_epoch_values)) if local_ratio_epoch_values else None),
            **val_report,
        }
        print(json.dumps(row), flush=True)
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + '\n')

        auc = val_report.get('roc_auc')
        auc_rank = float(auc) if auc is not None else -float('inf')
        rank = (float(val_report['macro_f1']), auc_rank)
        checkpoint = _make_checkpoint(
            model, ema, optimizer, args, train_data.classes, epoch + 1,
            val_report, train_data, val_data, selector_path, model_id)
        torch.save(checkpoint, output_dir / 'last.pt')
        if rank > best_rank:  # strict > keeps the earlier epoch on exact ties
            best_rank = rank
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / 'best.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print('Early stopping: validation Macro-F1/AUC did not improve for '
                      f'{args.early_stopping_patience} consecutive epochs.', flush=True)
                break

    if best_epoch is None:
        raise RuntimeError('No best checkpoint was produced')
    print(f'Best validation Macro-F1: {best_rank[0]:.6f}; epoch={best_epoch}; '
          f'checkpoint={output_dir / "best.pt"}', flush=True)

    best_checkpoint = torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=False)
    best_model = GFViTLevel1A1.from_config(
        best_checkpoint['model_config'], pretrained=False)
    infer_state = (best_checkpoint.get('model_ema')
                   if not args.no_ema and best_checkpoint.get('model_ema') is not None
                   else best_checkpoint['model'])
    best_model.load_state_dict(infer_state)
    best_model = best_model.to(device).eval()
    if args.experiment == 'a1.1':
        selector_copy = output_dir / f'selector_seed{args.seed}.pt'
        torch.save(best_checkpoint, selector_copy)
        print(f'A1.1 frozen selector: {selector_copy}', flush=True)

    rows, final_report = _export_predictions(
        best_model, val_data, val_loader, selector, args, output_dir,
        checkpoint_id=model_id, split='valid')
    if args.experiment == 'a1.2':
        _export_train_boxes(best_model, train_data, selector, args, output_dir)
    print(json.dumps({'final_validation_metrics': final_report,
                      'output_dir': str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
