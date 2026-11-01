#!/usr/bin/env python
"""Score the released poojahira/gtsrb-pytorch checkpoint (model/model_40.pth) on the official GTSRB test split.

The repository reports 99.809 percent on the private leaderboard of the NYU CV Fall 2018 Kaggle
in-class competition, whose test images are the 12,630 official GTSRB test images with hidden
labels, using random test-time augmentation (which makes the number vary between runs by about
0.1 points per the README). Here the checkpoint is run without test-time augmentation on the same
12,630 images with the repository's preprocessing (resize to 32x32, per-channel normalisation
with the constants from data.py), so the result is a deterministic lower bound of the README
number. The checkpoint predates PyTorch 1.3, so ``--align-corners`` (default on) reproduces the
affine_grid / grid_sample convention it was trained with; the off setting is reported for
comparison.

Usage:
    curl -L -o weights/poojahira/model_40.pth https://raw.githubusercontent.com/poojahira/gtsrb-pytorch/master/model/model_40.pth
    python scripts/eval_poojahira_weights_gtsrb.py --weights weights/poojahira/model_40.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import SHORT_NAMES, load_gtsrb, to_arrays  # noqa: E402
from opentsr.signs.models.gtsrb_cnn import STNCNNPoojahira  # noqa: E402

EXPECTED_SHA256 = "157996e0f6baa5b5f0803dae03a74c61c31723c8768f7cdd0e6535df14bbae38"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/poojahira/model_40.pth")
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--out", default="runs/gtsrb/poojahira_model_40_official_test.json")
    args = ap.parse_args()

    sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    state = torch.load(args.weights, map_location="cpu")
    data = load_gtsrb(args.root, "test")
    x = torch.from_numpy(to_arrays(data, 32)).permute(0, 3, 1, 2).float() / 255.0
    y = data.labels
    results = {}
    for align in (True, False):
        model = STNCNNPoojahira(align_corners=align)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) <= {"mean", "std"}, (missing, unexpected)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[i : i + 512]).argmax(1) for i in range(0, len(x), 512)]).numpy()
        cm = np.zeros((43, 43), dtype=int)
        np.add.at(cm, (y, pred), 1)
        results[f"align_corners_{align}"] = {
            "top1_accuracy": float((pred == y).mean()),
            "errors": int((pred != y).sum()),
            "per_class_recall": {SHORT_NAMES[i]: float(cm[i, i] / cm[i].sum()) for i in range(43)},
        }
        print(f"align_corners={align}: {results[f'align_corners_{align}']['top1_accuracy'] * 100:.2f}% ({results[f'align_corners_{align}']['errors']} errors)")
    out = {
        "weights": args.weights,
        "sha256": sha,
        "sha256_matches_2026_10_31_download": sha == EXPECTED_SHA256,
        "protocol": "official GTSRB test split (12,630), resize 32x32, repo normalisation, no test-time augmentation",
        "reference": "README: 99.809% private / 99.746% public leaderboard with test-time augmentation",
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
