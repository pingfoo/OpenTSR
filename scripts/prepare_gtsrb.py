#!/usr/bin/env python
"""Download the GTSRB parquet mirror and check it against the official split sizes.

GTSRB's official host (https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/)
is not reachable from the cloud sessions, so the Hugging Face mirror ``bazyl/GTSRB`` is used.
It is a parquet export of the Kaggle release (PNG conversion of the official PPM files) and
carries the official ROI columns. This script pins the mirror revision, verifies the sha256 of
the two parquet files, and checks that the split sizes (39,209 / 12,630) and the class ids
(0 to 42) match the official benchmark before anything else uses the data.

Usage:
    python scripts/prepare_gtsrb.py --root data/gtsrb_hf --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import check_official_split_sizes, load_gtsrb  # noqa: E402

REPO_ID = "bazyl/GTSRB"
REVISION = "a8093a9c7757b59d64702f892002542e8f3a1fb0"  # 2022-10-25, checked 2026-10-31
SHA256 = {
    "data/train-00000-of-00001-dc762c064c221993.parquet": "6fbb17c156897e8d852e1a7a8304381dbc0d26cc628fdb46fa5bb2f457e7d588",
    "data/test-00000-of-00001-747a54d4a6461a97.parquet": "979ca6fe6db32b8f2721181c1cbc58417f2dd175fe28b0e1b8a0d3d32a718e46",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--download", action="store_true", help="fetch the parquet files from Hugging Face")
    args = ap.parse_args()
    root = Path(args.root)

    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=root,
            allow_patterns=["data/*.parquet", "Meta.csv", "Train.csv", "Test.csv", "README.md"],
        )

    problems: list[str] = []
    for rel, expected in SHA256.items():
        p = root / rel
        if not p.exists():
            problems.append(f"missing {p}")
            continue
        got = sha256_of(p)
        if got != expected:
            problems.append(f"sha256 mismatch for {rel}: {got}")

    train = load_gtsrb(root, "train", decode=False)
    test = load_gtsrb(root, "test", decode=False)
    problems += check_official_split_sizes(train.labels, test.labels)

    import numpy as np

    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "train_images": int(len(train.labels)),
        "test_images": int(len(test.labels)),
        "num_classes": int(len(np.unique(train.labels))),
        "test_class_histogram": np.bincount(test.labels, minlength=43).tolist(),
        "problems": problems,
    }
    (root / "VERIFIED.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "test_class_histogram"}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
