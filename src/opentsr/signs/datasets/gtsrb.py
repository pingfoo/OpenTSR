"""GTSRB (German Traffic Sign Recognition Benchmark) loader.

Source of the files: the Hugging Face mirror ``bazyl/GTSRB`` (parquet export of the Kaggle
release of GTSRB, itself a PNG conversion of the official PPM archives). It reproduces the
official split exactly: 39,209 training images (GTSRB_Final_Training_Images) and 12,630 test
images (GTSRB_Final_Test_Images with GT-final_test.csv), 43 classes, per-image ROI columns.
The official host (sid.erda.dk) is not reachable from the cloud sessions, which is why the
mirror is used; ``scripts/prepare_gtsrb.py`` checks the split sizes and class histogram against
the official numbers before anything is cached.

Class names follow the official GTSRB class ids (0 to 42). The list below is the one used in
the OpenAI CLIP repository (data/prompts.md), which spells out every class in English.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

OFFICIAL_TRAIN_SIZE = 39_209
OFFICIAL_TEST_SIZE = 12_630
NUM_CLASSES = 43

# Official GTSRB class ids in order, English names as used by OpenAI CLIP (data/prompts.md).
CLASS_NAMES: list[str] = [
    "red and white circle 20 kph speed limit",
    "red and white circle 30 kph speed limit",
    "red and white circle 50 kph speed limit",
    "red and white circle 60 kph speed limit",
    "red and white circle 70 kph speed limit",
    "red and white circle 80 kph speed limit",
    "end / de-restriction of 80 kph speed limit",
    "red and white circle 100 kph speed limit",
    "red and white circle 120 kph speed limit",
    "red and white circle red car and black car no passing",
    "red and white circle red truck and black car no passing",
    "red and white triangle road intersection warning",
    "white and yellow diamond priority road",
    "red and white upside down triangle yield right-of-way",
    "stop",
    "empty red and white circle",
    "red and white circle no truck entry",
    "red circle with white horizonal stripe no entry",
    "red and white triangle with exclamation mark warning",
    "red and white triangle with black left curve approaching warning",
    "red and white triangle with black right curve approaching warning",
    "red and white triangle with black double curve approaching warning",
    "red and white triangle rough / bumpy road warning",
    "red and white triangle car skidding / slipping warning",
    "red and white triangle with merging / narrow lanes warning",
    "red and white triangle with person digging / construction / road work warning",
    "red and white triangle with traffic light approaching warning",
    "red and white triangle with person walking warning",
    "red and white triangle with child and person walking warning",
    "red and white triangle with bicyle warning",
    "red and white triangle with snowflake / ice warning",
    "red and white triangle with deer warning",
    "white circle with gray strike bar no speed limit",
    "blue circle with white right turn arrow mandatory",
    "blue circle with white left turn arrow mandatory",
    "blue circle with white forward arrow mandatory",
    "blue circle with white forward or right turn arrow mandatory",
    "blue circle with white forward or left turn arrow mandatory",
    "blue circle with white keep right arrow mandatory",
    "blue circle with white keep left arrow mandatory",
    "blue circle with white arrows indicating a traffic circle",
    "white circle with gray strike bar indicating no passing for cars has ended",
    "white circle with gray strike bar indicating no passing for trucks has ended",
]

# Short names (the common GTSRB sign names, as listed by most tutorials and by the
# Roboflow / keremberke GTSDB export), useful for compact tables and VLM prompts.
SHORT_NAMES: list[str] = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
    "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

assert len(CLASS_NAMES) == NUM_CLASSES and len(SHORT_NAMES) == NUM_CLASSES


@dataclass
class GTSRBSplit:
    images: list[Image.Image]  # RGB PIL images at native resolution
    labels: np.ndarray  # int64 class ids
    rois: np.ndarray  # (N, 4) x1, y1, x2, y2 in pixels (official ROI columns)
    paths: list[str]  # original relative paths from the Kaggle release, for traceability


def _parquet_path(root: Path, split: str) -> Path:
    cands = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not cands:
        raise FileNotFoundError(
            f"no {split} parquet under {root / 'data'}; run scripts/prepare_gtsrb.py --download first"
        )
    return cands[0]


def load_gtsrb(root: str | Path, split: str, decode: bool = True) -> GTSRBSplit:
    """Load a split ("train" or "test") from the parquet mirror.

    With ``decode=False`` the ``images`` list is empty, which is useful to read only labels.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    table = pq.read_table(_parquet_path(root, split))
    labels = np.asarray(table.column("ClassId").to_pylist(), dtype=np.int64)
    rois = np.stack(
        [np.asarray(table.column(c).to_pylist(), dtype=np.int64) for c in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")],
        axis=1,
    )
    col = table.column("Path").to_pylist()
    paths = [c.get("path") or "" for c in col]
    images: list[Image.Image] = []
    if decode:
        for c in col:
            images.append(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
    return GTSRBSplit(images=images, labels=labels, rois=rois, paths=paths)


def check_official_split_sizes(train_labels: np.ndarray, test_labels: np.ndarray) -> list[str]:
    """Return a list of problems (empty when the mirror matches the official split sizes)."""
    problems = []
    if len(train_labels) != OFFICIAL_TRAIN_SIZE:
        problems.append(f"train has {len(train_labels)} images, official is {OFFICIAL_TRAIN_SIZE}")
    if len(test_labels) != OFFICIAL_TEST_SIZE:
        problems.append(f"test has {len(test_labels)} images, official is {OFFICIAL_TEST_SIZE}")
    for name, lab in (("train", train_labels), ("test", test_labels)):
        if lab.min() != 0 or lab.max() != NUM_CLASSES - 1 or len(np.unique(lab)) != NUM_CLASSES:
            problems.append(f"{name} does not contain exactly the 43 class ids 0..42")
    return problems


def to_arrays(split: GTSRBSplit, size: int, use_roi: bool = False) -> np.ndarray:
    """Resize every image to ``size`` x ``size`` and return a uint8 array (N, size, size, 3).

    ``use_roi=True`` crops to the official ROI first (the sign without its margin). The classic
    GTSRB recipes (Sermanet 2011, the Udacity LeNet, poojahira's STN) resize the full image.
    """
    out = np.empty((len(split.images), size, size, 3), dtype=np.uint8)
    for i, img in enumerate(split.images):
        if use_roi:
            x1, y1, x2, y2 = split.rois[i].tolist()
            img = img.crop((x1, y1, x2, y2))
        out[i] = np.asarray(img.resize((size, size), Image.BILINEAR))
    return out
