"""Reusable PyTorch DataLoader factory for Glance→Focus datasets."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def _seed_worker(worker_id):
    """Give each worker deterministic but distinct Python/NumPy RNG streams."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(dataset, batch_size=8, shuffle=False, workers=0,
                      device='cpu', seed=None, drop_last=False):
    """Create batches of ``(image, fov_mask, label, path)`` from ImageManifest.

    A seeded generator makes shuffling and worker-side augmentation reproducible.
    Pin memory is enabled only for CUDA; persistent workers are enabled only
    when workers > 0, so this also works in Colab notebooks with workers=0.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    if workers < 0:
        raise ValueError('workers must be >= 0')
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=str(device).startswith('cuda'),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None,
        generator=generator,
        drop_last=drop_last,
    )
