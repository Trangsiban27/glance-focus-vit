"""Evaluate a fixed checkpoint without tuning on test labels.

python evaluate.py --checkpoint runs/E4/best.pt --csv data/test.csv --output results/E4
Use --compare-predictions results/E1/predictions.json for paired analysis.
Boxes are expressed in the resized high-resolution view's pixel coordinates.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from dataset import ImageManifest
from dataloader import create_dataloader
from label_mapping import BINARY_CLASSES, map_class_indices, multiclass_to_binary
from train import device_for, load_checkpoint, load_selector, metrics, predict, seed_all


def bootstrap_f1(y, probabilities, classes, samples, seed):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    pred = probabilities.argmax(1)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[indices], pred[indices], labels=list(range(classes)),
                               average='macro', zero_division=0))
    return np.quantile(values, [.025, .975]).tolist()


def paired_analysis(records, other_path):
    other = json.loads(Path(other_path).read_text())
    by_path = {r['path']: r for r in other}
    if len(by_path) != len(other) or set(by_path) != {r['path'] for r in records}:
        raise ValueError('Paired predictions must contain exactly the same unique images')
    table = np.zeros((2, 2), dtype=int)
    for record in records:
        previous = by_path[record['path']]
        if record['label'] != previous['label']:
            raise ValueError('Paired predictions have different labels/class mappings')
        a = int(previous['prediction'] == previous['label'])
        b = int(record['prediction'] == record['label'])
        table[a, b] += 1
    discordant = int(table[0, 1] + table[1, 0])
    return {'correctness_table_rows_reference_cols_current': table.tolist(),
            'mcnemar_exact_pvalue': float(binomtest(int(table[0, 1]), discordant, .5).pvalue) if discordant else 1.}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='results/E4')
    parser.add_argument('--selector', help='Override frozen E1 checkpoint path recorded during training')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare-predictions')
    parser.add_argument('--binary', action='store_true', help='Also map predicted disease classes to Normal/Abnormal')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.bootstrap_samples < 1:
        parser.error('Require batch-size,bootstrap-samples >=1 and workers >=0')
    seed_all(args.seed)
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = ImageManifest(args.csv, args.data_root, classes=checkpoint['classes'],
                            hi_size=checkpoint['hi_size'], use_fov=model.use_fov)
    train_paths = {r['path'] for r in checkpoint.get('splits', {}).get('train', [])}
    if train_paths & {str(p) for p, _ in dataset.samples}:
        raise ValueError('Evaluation images overlap the training manifest')
    selector_path = args.selector or checkpoint.get('selector_checkpoint')
    if model.fusion == 'local' and not selector_path:
        parser.error('Local/E2 evaluation requires its frozen E1 selector')
    selector = load_selector(selector_path, dataset.classes, device) if selector_path else None
    if selector is not None:
        selector.attention_method = model.attention_method
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.json').exists():
        raise FileExistsError('Evaluation output exists; choose a new --output directory')
    loader = create_dataloader(dataset, batch_size=args.batch_size, workers=args.workers,
                               device=device, seed=args.seed)
    start = time.perf_counter()
    y, probabilities, records = predict(model, loader, device, selector)
    elapsed = time.perf_counter()-start
    report = metrics(y, probabilities, len(dataset.classes), class_names=dataset.classes)
    report.update(classes=dataset.classes, count=len(dataset), checkpoint=str(Path(args.checkpoint).resolve()),
                  resolution_stats=dataset.resolution_stats(),
                  macro_f1_ci95=bootstrap_f1(y, probabilities, len(dataset.classes), args.bootstrap_samples, args.seed),
                  parameters=sum(p.numel() for p in model.parameters()),
                  selector_parameters=sum(p.numel() for p in selector.parameters()) if selector else 0,
                  elapsed_seconds=elapsed, end_to_end_ms_per_image=1000*elapsed/len(dataset))
    if args.binary:
        binary_pred, binary_prob = multiclass_to_binary(probabilities, dataset.classes)
        binary_y = map_class_indices(y, dataset.classes)
        report['binary'] = metrics(binary_y, binary_prob, 2, predictions=binary_pred,
                                   class_names=BINARY_CLASSES)
        report['binary']['classes'] = list(BINARY_CLASSES)
        report['binary']['decision_rule'] = 'multiclass_argmax_then_binary_mapping'
        for record, target, pred, prob in zip(records, binary_y, binary_pred, binary_prob):
            record.update(binary_label=int(target), binary_prediction=int(pred),
                          binary_probabilities=prob.tolist())
    if hasattr(model, 's'):

        report['s'] = model.s.item()
    gates = [r for r in records if r['gate'] is not None]
    if gates:
        report['gate_summary'] = {}
        for label, name in enumerate(dataset.classes):
            for correct in (False, True):
                group = [np.mean(r['gate']) for r in gates if r['label'] == label and
                         (r['prediction'] == r['label']) == correct]
                report['gate_summary'][f'{name}|correct={correct}'] = {
                    'count': len(group), 'mean': float(np.mean(group)) if group else None}
    if args.compare_predictions:
        report['paired_comparison'] = paired_analysis(records, args.compare_predictions)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (output/'predictions.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
