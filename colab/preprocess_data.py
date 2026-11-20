"""Offline preprocessing of CSV image datasets, without modifying source images.

Train (resize + deterministic augmentation):
 python preprocess_data.py --csv data/Datos/train.csv --output data/processed/train --augment
Validation/test (resize only):
 python preprocess_data.py --csv data/Datos/valid.csv --output data/processed/valid
From a notebook cell, import and call preprocess(csv, output, augment=True).

Output: manifest.csv with relative path,label columns, images/, metadata.json.
Use the generated manifest with train.py/evaluate.py. Keep original manifests
split BEFORE augmentation. Train-time processing.py remains a separate online
augmentation step; omit --augment here to avoid combining both strategies.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


def variants(image, augment=False, angle=10., brightness=.2, contrast=.2, flip=False):
    """Yield independent variants of the same image, not chained transforms."""
    yield 'original', image
    if not augment:
        return
    yield 'rotate_plus', image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    yield 'rotate_minus', image.rotate(-angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    yield 'brightness_plus', ImageEnhance.Brightness(image).enhance(1+brightness)
    yield 'brightness_minus', ImageEnhance.Brightness(image).enhance(1-brightness)
    yield 'contrast_plus', ImageEnhance.Contrast(image).enhance(1+contrast)
    yield 'contrast_minus', ImageEnhance.Contrast(image).enhance(1-contrast)
    if flip:
        yield 'horizontal_flip', ImageOps.mirror(image)


def preprocess(manifest, output, data_root=None, size=448, augment=False,
               angle=10., brightness=.2, contrast=.2, flip=False):
    if not isinstance(size, int) or size < 1:
        raise ValueError('size must be a positive integer')
    if not math.isfinite(angle) or not 0 < angle <= 180:
        raise ValueError('angle must be in (0,180]')
    if not all(math.isfinite(v) and 0 < v < 1 for v in (brightness, contrast)):
        raise ValueError('brightness and contrast must be in (0,1)')
    if flip and not augment:
        raise ValueError('--horizontal-flip requires --augment')
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    root = Path(data_root).resolve() if data_root else manifest.parent
    with manifest.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain path,label columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty input manifest')
    sources = []
    for row in rows:
        if not row['path'].strip() or not row['label'].strip():
            raise ValueError('Empty path or label')
        path = (root / row['path']).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        sources.append(path)
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate image paths in input CSV')
    # A dedicated fresh output folder prevents overwriting prior runs/sources.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    records, provenance = [], []
    for index, (row, source) in enumerate(zip(rows, sources)):
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
        for name, result in variants(image, augment, angle, brightness, contrast, flip):
            result = result.resize((size, size), Image.Resampling.LANCZOS)
            relative = f'images/{index:07d}_{name}.png'
            result.save(output / relative)
            records.append({'path': relative, 'label': row['label']})
            provenance.append({'path': relative, 'source': str(source), 'transform': name})
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label'])
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(source_manifest=str(manifest), input_images=len(rows),
                    output_images=len(records), size=[size, size], augment=augment,
                    angle=angle, brightness_delta=brightness, contrast_delta=contrast,
                    horizontal_flip=flip, rotation_fill='black',
                    class_counts=dict(Counter(r['label'] for r in records)),
                    provenance=provenance)
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return output / 'manifest.csv', metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', help='Input CSV with path,label columns')
    parser.add_argument('--output', help='New output directory; must not already exist')
    parser.add_argument('--data-root')
    parser.add_argument('--size', type=int, default=448, help='Output width and height; keep >=448 for local crops')
    parser.add_argument('--augment', action='store_true', help='Enable offline augmentation for TRAIN only')
    parser.add_argument('--angle', type=float, default=10., help='Rotation in both directions, default +/-10 degrees')
    parser.add_argument('--brightness', type=float, default=.2, help='Brightness delta, default +/-20%%')
    parser.add_argument('--contrast', type=float, default=.2, help='Contrast delta, default +/-20%%')
    parser.add_argument('--horizontal-flip', action='store_true', help='Also mirror horizontally with --augment')
    # parse_known_args ignores notebook kernel arguments such as `-f kernel.json`.
    args, _unknown = parser.parse_known_args(argv)
    if not args.csv or not args.output:
        parser.print_help()
        print('Notebook example: preprocess("/content/data/Datos/train.csv", '
              '"/content/processed_train", augment=True)')
        return None
    manifest, meta = preprocess(args.csv, args.output, args.data_root, args.size,
                                args.augment, args.angle, args.brightness,
                                args.contrast, args.horizontal_flip)
    print(f"Input: {meta['input_images']}; output: {meta['output_images']}; CSV: {manifest}")


if __name__ == '__main__':
    main()
