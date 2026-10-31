#!/usr/bin/env python
"""Download and extract the GTSDB archive mirrored on Hugging Face (GlowBond/GTSDB, 1.73 GB).

The mirror holds TrainIJCNN2013 (600 images, gt.txt, per-class crops) and TestIJCNN2013Download
(300 images without ground truth). The official host benchmark.ini.rub.de / sid.erda.dk is not
reachable from the cloud sessions. If you have the official FullIJCNN2013.zip, extract it next
to the mirror and pass ``--full-gt`` to the evaluation scripts to score the 300 test images.

Usage:
    python scripts/prepare_gtsdb.py --root data/gtsdb_hf --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsdb import category_of, load_gtsdb_train, load_gtsdb_test_images  # noqa: E402

REPO_ID = "GlowBond/GTSDB"
REVISION = "0d1c1a77fce322ba23b57a5e3261fae934f5afe3"  # 2026-03-18, checked 2026-10-31
ARCHIVE = "archive.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gtsdb_hf")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.download:
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO_ID, ARCHIVE, repo_type="dataset", revision=REVISION, local_dir=root)

    archive = root / ARCHIVE
    if not (root / "TrainIJCNN2013").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)

    train = load_gtsdb_train(root)
    n_boxes = sum(len(im.boxes) for im in train)
    cats = {}
    for im in train:
        for c in im.class_ids:
            cats[category_of(c)] = cats.get(category_of(c), 0) + 1
    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "archive_sha256": h.hexdigest(),
        "train_images": len(train),
        "train_images_with_signs": sum(1 for im in train if im.boxes),
        "train_boxes": n_boxes,
        "train_boxes_by_category": cats,
        "test_images": len(load_gtsdb_test_images(root)),
        "test_ground_truth_available": False,
    }
    (root / "VERIFIED.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = report["train_images"] == 600 and report["train_boxes"] == 852 and report["test_images"] == 300
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
