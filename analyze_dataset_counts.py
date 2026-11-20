"""Summarize Normal/Abnormal counts for internal splits and external test.

Example (defaults include internal train/validation/test and external test):
    python analyze_dataset_counts.py \
        --internal-csv data/Datos/internal_test.csv \
        --external-csv data/eardrum/test_external.csv \
        --output-dir results/dataset_counts

Disease-specific labels are grouped as Abnormal using label_mapping.py.
Counts are taken from manifest rows; this script does not load image pixels.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from label_mapping import canonical_label


CLASSES = ('Normal', 'Abnormal')


def count_manifest(csv_path, dataset_name):
    manifest = Path(csv_path).expanduser().resolve()
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'label'} <= set(reader.fieldnames or []):
            raise ValueError(f'{manifest} must have path,label columns')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{manifest} has no image rows')

    paths = [row['path'].strip() for row in rows]
    if any(not path for path in paths):
        raise ValueError(f'{manifest} contains an empty image path')
    if len(paths) != len(set(paths)):
        raise ValueError(f'{manifest} contains duplicate image paths')

    raw_counts = Counter()
    binary_counts = Counter({name: 0 for name in CLASSES})
    for row in rows:
        raw_label = row['label'].strip()
        canonical = canonical_label(raw_label)
        raw_counts[raw_label] += 1
        binary_counts['Normal' if canonical == 'Normal' else 'Abnormal'] += 1

    return {
        'dataset': dataset_name,
        'manifest': str(manifest),
        'total': len(rows),
        'counts': {name: int(binary_counts[name]) for name in CLASSES},
        'percentages': {name: binary_counts[name] / len(rows) for name in CLASSES},
        'source_label_counts': dict(sorted(raw_counts.items())),
    }


def make_plot(results, output_path):
    names = [item['dataset'] for item in results]
    x = np.arange(len(names))
    width = .34
    colors = {'Normal': '#3B82F6', 'Abnormal': '#F9735B'}

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    for offset, class_name in ((-width / 2, 'Normal'), (width / 2, 'Abnormal')):
        values = [item['counts'][class_name] for item in results]
        bars = ax.bar(x + offset, values, width, label=class_name,
                      color=colors[class_name], edgecolor='white', linewidth=1.2)
        for bar, item, count in zip(bars, results, values):
            pct = item['percentages'][class_name] * 100
            ax.annotate(f'{count}\n{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    max_total = max(item['total'] for item in results)
    ax.set_xticks(x, [f"{item['dataset']}\nN={item['total']}" for item in results])
    ax.set_ylabel('Number of images')
    ax.set_title('Class counts across internal splits and external test')
    ax.set_ylim(0, max_total * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=.28)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc='upper right')
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', default='data/Datos/train.csv')
    parser.add_argument('--val-csv', default='data/Datos/valid.csv')
    parser.add_argument('--internal-test-csv', '--internal-csv', dest='internal_test_csv',
                        default='data/Datos/internal_test.csv')
    parser.add_argument('--external-test-csv', '--external-csv', dest='external_test_csv',
                        default='data/eardrum/test_external.csv')
    parser.add_argument('--output-dir', default='results/dataset_counts')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        count_manifest(args.train_csv, 'Internal — Train'),
        count_manifest(args.val_csv, 'Internal — Validation'),
        count_manifest(args.internal_test_csv, 'Internal — Test'),
        count_manifest(args.external_test_csv, 'External — Test'),
    ]

    report = {
        'class_mapping': 'Normal stays Normal; every recognized disease label maps to Abnormal',
        'count_source': 'CSV manifest rows; image pixels are not loaded',
        'datasets': results,
    }
    (output_dir / 'class_counts.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    with (output_dir / 'class_counts.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fields = ['dataset', 'manifest', 'total', 'normal_count', 'normal_percent',
                  'abnormal_count', 'abnormal_percent']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'dataset': item['dataset'],
                'manifest': item['manifest'],
                'total': item['total'],
                'normal_count': item['counts']['Normal'],
                'normal_percent': round(100 * item['percentages']['Normal'], 2),
                'abnormal_count': item['counts']['Abnormal'],
                'abnormal_percent': round(100 * item['percentages']['Abnormal'], 2),
            })

    plot_path = output_dir / 'class_counts.png'
    make_plot(results, plot_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Plot: {plot_path}')
    print(f'Table: {output_dir / "class_counts.csv"}')


if __name__ == '__main__':
    main()
