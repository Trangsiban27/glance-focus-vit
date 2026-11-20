"""CSV image dataset. Labels remain multiclass during training and inference.

CSV columns: path,label. Relative paths resolve against data_root or the CSV
folder. Pass checkpoint classes to preserve the training label order.
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from processing import ImageProcessor


class ImageManifest(Dataset):
    def __init__(self, manifest, data_root=None, classes=None, hi_size=448,
                 training=False, use_fov=True):
        self.manifest = Path(manifest).resolve()
        root = Path(data_root).resolve() if data_root else self.manifest.parent
        with self.manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'path', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain path,label columns')
            rows = list(reader)
        if not rows:
            raise ValueError(f'Empty manifest: {manifest}')
        self.classes = classes if classes is not None else sorted({r['label'] for r in rows})
        self.classes = list(self.classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('classes must be nonempty and unique')
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        mapping = self.class_to_idx
        self.samples = []
        for row in rows:
            if row['label'] not in mapping:
                raise ValueError(f"Unknown label {row['label']!r} in {manifest}")
            path = (root / row['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, mapping[row['label']]))
        if len({p for p, _ in self.samples}) != len(self.samples):
            raise ValueError(f'Duplicate image paths in {manifest}')
        self.hi_size, self.training, self.use_fov = hi_size, training, use_fov
        self.processor = ImageProcessor(hi_size, training, use_fov)

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

