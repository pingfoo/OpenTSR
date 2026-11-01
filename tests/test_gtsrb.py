import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsdb import DANGER, MANDATORY, PROHIBITORY, category_of  # noqa: E402
from opentsr.signs.datasets.gtsrb import CLASS_NAMES, NUM_CLASSES, SHORT_NAMES, check_official_split_sizes  # noqa: E402
from opentsr.signs.models.gtsrb_cnn import MODELS, build_model, random_affine_batch, random_photometric_batch  # noqa: E402


def test_class_lists():
    assert len(CLASS_NAMES) == NUM_CLASSES == len(SHORT_NAMES) == 43
    assert CLASS_NAMES[14] == "stop" and SHORT_NAMES[14] == "Stop"


def test_split_check_accepts_official_sizes():
    train = np.repeat(np.arange(43), 912)[:39209]
    test = np.repeat(np.arange(43), 294)[:12630]
    assert check_official_split_sizes(train, test) == []
    assert check_official_split_sizes(train[:-1], test)


def test_gtsdb_categories_are_disjoint_and_complete():
    assert not (PROHIBITORY & MANDATORY) and not (PROHIBITORY & DANGER) and not (MANDATORY & DANGER)
    cats = {category_of(c) for c in range(43)}
    assert cats == {"prohibitory", "mandatory", "danger", "other"}
    assert category_of(14) == "other" and category_of(1) == "prohibitory"


def test_models_forward_and_augmentations_keep_shape():
    x = torch.rand(4, 3, 32, 32)
    for name in MODELS:
        y = build_model(name)(x)
        assert y.shape == (4, 43)
    g = torch.Generator().manual_seed(0)
    xa = random_affine_batch(x, generator=g)
    xp = random_photometric_batch(x, generator=g)
    assert xa.shape == x.shape and xp.shape == x.shape
    assert float(xp.min()) >= 0.0 and float(xp.max()) <= 1.0
