"""Reader for S2TLD (SJTU Small Traffic Light Dataset), PASCAL VOC XML annotations.

Expected layout after ``traffic_lights/scripts/prepare_s2tld.py`` has extracted the two zips::

    <root>/S2TLD_720x1280/normal_1/{Annotations,JPEGImages}/
    <root>/S2TLD_720x1280/normal_2/{Annotations,JPEGImages}/
    <root>/S2TLD_1080x1920/{Annotations,JPEGImages}/

Image ids are ``<group>/<file stem>`` with groups ``720_normal_1``, ``720_normal_2`` and
``1080``; these are the ids stored in the split manifests under
``traffic_lights/configs/s2tld/splits/``.

This module was restored on 2026-10-31: the original was never committed because the
``datasets/`` pattern in ``.gitignore`` also matched this package directory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

S2TLD_CLASSES = ["red", "yellow", "green", "off", "wait_on"]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class S2TLDRecord:
    image_id: str
    group: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)


def _normalise_label(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_image(img_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        p = img_dir / (stem + suffix)
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {stem} in {img_dir}")


def _parse(xml_path: Path, image_id: str, group: str, img_dir: Path) -> S2TLDRecord:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        boxes.append(
            Box(
                _normalise_label(obj.findtext("name", "")),
                float(bb.findtext("xmin")),
                float(bb.findtext("ymin")),
                float(bb.findtext("xmax")),
                float(bb.findtext("ymax")),
            )
        )
    return S2TLDRecord(image_id, group, _find_image(img_dir, xml_path.stem), width, height, boxes)


def _group_name(subset_dir: Path, ann_dir: Path) -> str:
    resolution = "1080" if "1080" in subset_dir.name else "720"
    if ann_dir.parent == subset_dir:
        return resolution
    return f"{resolution}_{ann_dir.parent.name}"


def load_s2tld(root: Path | str) -> list[S2TLDRecord]:
    """Load every annotated image under ``root`` (the ``extracted`` directory), sorted by id."""
    root = Path(root)
    records: list[S2TLDRecord] = []
    for subset_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("S2TLD")):
        for ann_dir in sorted(subset_dir.rglob("Annotations")):
            img_dir = ann_dir.parent / "JPEGImages"
            group = _group_name(subset_dir, ann_dir)
            for xml_path in sorted(ann_dir.glob("*.xml")):
                records.append(_parse(xml_path, f"{group}/{xml_path.stem}", group, img_dir))
    if not records:
        raise FileNotFoundError(f"no S2TLD annotations found under {root}")
    records.sort(key=lambda r: r.image_id)
    return records
