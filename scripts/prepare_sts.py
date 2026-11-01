#!/usr/bin/env python
"""Fetch the Swedish Traffic Signs (STS) annotations, and optionally the annotated frames.

Sources (see src/opentsr/signs/datasets/sts.py and docs/third_party.md):
  CVL per-set files   https://www.cvl.isy.liu.se/research/trafficSigns/swedishSignsSummer/<Set>/annotations.txt
                      https://www.cvl.isy.liu.se/research/trafficSigns/swedishSignsSummer/<Set>/Set<N>Part0.zip
  Zenodo (CC BY 4.0)  https://zenodo.org/records/19666339  traffic-signs_dataset.zip (15.8 GB, everything)

Default: annotations only (two small text files), verified against the line counts seen on
2026-11-01 (Set1 1,970 lines, Set2 1,807 lines). ``--images`` also downloads Part0 of each set
(about 1.5 GB each, the annotated frames) and unzips it next to the annotations.

Usage: python scripts/prepare_sts.py --root data/sts [--images] [--sets Set1 Set2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.sts import load_annotations, sequences  # noqa: E402

BASE = "https://www.cvl.isy.liu.se/research/trafficSigns/swedishSignsSummer"
EXPECTED_LINES = {"Set1": 1970, "Set2": 1807}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"exists  {dst}")
        return
    print(f"get     {url}")
    subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(dst), url], check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sts")
    ap.add_argument("--sets", nargs="+", default=["Set1", "Set2"])
    ap.add_argument("--images", action="store_true", help="also fetch and unzip Part0 (annotated frames)")
    args = ap.parse_args()
    root = Path(args.root)
    summary = {}
    for s in args.sets:
        ann = root / s / "annotations.txt"
        download(f"{BASE}/{s}/annotations.txt", ann)
        n_lines = sum(1 for line in ann.read_text().splitlines() if line.strip())
        if s in EXPECTED_LINES and n_lines != EXPECTED_LINES[s]:
            print(f"warning: {s} has {n_lines} annotation lines, expected {EXPECTED_LINES[s]}")
        frames = load_annotations(ann, set_name=s)
        seqs = sequences(frames)
        n_boxes = sum(len(f.boxes) for f in frames)
        summary[s] = {"frames": len(frames), "boxes": n_boxes, "sequences": len(seqs), "sha256": sha256(ann),
                      "side_road_boxes": sum(b.status == "SIDE_ROAD" for f in frames for b in f.boxes)}
        print(f"{s}: {len(frames)} annotated frames, {n_boxes} boxes, {len(seqs)} sequences (gap > 10 s)")
        if args.images:
            n = s[-1]
            z = root / s / f"Set{n}Part0.zip"
            download(f"{BASE}/{s}/Set{n}Part0.zip", z)
            with zipfile.ZipFile(z) as zf:
                names = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
                print(f"unzip   {len(names)} images from {z.name}")
                zf.extractall(root / s)
            summary[s]["images_zip_sha256"] = sha256(z)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
