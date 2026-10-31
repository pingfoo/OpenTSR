"""GTSDB (German Traffic Sign Detection Benchmark, IJCNN 2013) loader.

Official package: 900 images (1360x800 PPM), 1,213 signs, 43 class ids shared with GTSRB, split
into 600 training images (gt.txt with 852 boxes in 506 of the 600 images) and 300 test images
whose ground truth is only in the "FullIJCNN2013" archive. The archive mirrored on Hugging
Face as ``GlowBond/GTSDB`` contains TrainIJCNN2013 (images, gt.txt, per-class crops) and
TestIJCNN2013Download (300 images, no ground truth), so cloud replications evaluate on the
600 training images with the official gt.txt; that is fine for zero-shot models that never saw
GTSDB but is NOT the competition protocol. The license statement in the ReadMe is: "You are
free to use this data package for any purpose you like."

Competition categories (from the ReadMe):
    prohibitory = circular, white ground with red border
    mandatory   = circular, blue ground
    danger      = triangular, white ground with red border
Everything else is "other" (stop, yield, priority road, no entry, end-of-restriction signs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}
DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}


def category_of(class_id: int) -> str:
    if class_id in PROHIBITORY:
        return "prohibitory"
    if class_id in MANDATORY:
        return "mandatory"
    if class_id in DANGER:
        return "danger"
    return "other"


@dataclass
class GTSDBImage:
    image_id: str  # "00000"
    path: Path
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # x1, y1, x2, y2 (inclusive pixel coords)
    class_ids: list[int] = field(default_factory=list)


def parse_gt(gt_path: Path) -> dict[str, list[tuple[tuple[int, int, int, int], int]]]:
    out: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}
    for line in gt_path.read_text().strip().splitlines():
        parts = line.strip().split(";")
        if len(parts) < 6:
            continue
        img = parts[0].rsplit(".", 1)[0]
        x1, y1, x2, y2, cls = (int(p) for p in parts[1:6])
        out.setdefault(img, []).append(((x1, y1, x2, y2), cls))
    return out


def load_gtsdb_train(root: str | Path) -> list[GTSDBImage]:
    """Return the 600 training images (with empty box lists for the 94 negatives)."""
    root = Path(root)
    img_dir = root / "TrainIJCNN2013" / "TrainIJCNN2013"
    gt = parse_gt(img_dir / "gt.txt")
    images = []
    for p in sorted(img_dir.glob("*.ppm")):
        img_id = p.stem
        items = gt.get(img_id, [])
        images.append(GTSDBImage(img_id, p, [b for b, _ in items], [c for _, c in items]))
    return images


def load_gtsdb_test_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted((root / "TestIJCNN2013" / "TestIJCNN2013Download").glob("*.ppm"))
