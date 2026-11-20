"""Map disease predictions to binary AFTER multiclass inference.

Normal=0, Abnormal=1. Hard decisions map argmax disease -> binary; binary
probabilities sum disease probabilities. Their argmax can differ from the
mapped disease decision, so the two are returned separately.
"""
import numpy as np

BINARY_CLASSES = ('Normal', 'Abnormal')
DISEASE_ALIASES = {
    'Normal': 'Normal',
    'Chronic otitis media': 'Chronic otitis media',
    'CSOM': 'Chronic otitis media',
    'Earwax plug': 'Earwax plug',
    'Earwax': 'Earwax plug',
    'Myringosclerosis': 'Myringosclerosis',
    'Miringoskleroz': 'Myringosclerosis',
    'AOM': 'AOM',
    'OtitisEksterna': 'OtitisEksterna',
    'TympanostomyTubes': 'TympanostomyTubes',
}


def canonical_label(label):
    """Resolve known dataset aliases; reject typos and unrecognized labels."""
    lookup = {key.casefold(): value for key, value in DISEASE_ALIASES.items()}
    try:
        return lookup[label.strip().casefold()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f'Unknown disease label: {label!r}') from exc


def label_to_binary(label):
    return int(canonical_label(label) != 'Normal')


def binary_class_mapping(classes):
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('classes must be nonempty and unique')
    canonical = [canonical_label(label) for label in classes]
    if len(set(canonical)) != len(canonical):
        raise ValueError('classes contain aliases of the same disease')
    if 'Normal' not in canonical or len(canonical) < 2:
        raise ValueError('Binary evaluation requires Normal and at least one disease class')
    return np.asarray([label_to_binary(label) for label in classes], dtype=np.int64)


def map_class_indices(indices, classes):
    mapping = binary_class_mapping(classes)
    values = np.asarray(indices)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Class indices must be integers')
    if np.any(values < 0) or np.any(values >= len(mapping)):
        raise ValueError('Class index outside checkpoint classes')
    return mapping[values]


def multiclass_to_binary(probabilities, classes):
    """Return (mapped multiclass argmax, [P(Normal), P(Abnormal)])."""
    mapping = binary_class_mapping(classes)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(classes):
        raise ValueError('Expected probabilities with shape [N, len(classes)]')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError('Probabilities must sum to one')
    binary_p = np.column_stack([p[:, mapping == i].sum(1) for i in range(2)])
    return mapping[p.argmax(1)], binary_p
