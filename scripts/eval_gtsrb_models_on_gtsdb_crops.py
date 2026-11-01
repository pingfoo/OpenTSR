#!/usr/bin/env python
"""Cross-dataset check: GTSRB-trained classifiers applied to sign crops cut from GTSDB frames.

GTSDB (600 training frames, 852 boxes, same 43 class ids as GTSRB, different camera and scenes)
is the closest public "same country, different capture" test for a GTSRB classifier. Crops are
cut from the full frames with the official gt.txt boxes plus a relative margin (``--margin``,
default 0.1 which mimics the roughly 10 percent border of GTSRB images), resized by each model's
own preprocessing, and scored as 43-way classification. Accuracy is also broken down by the
GTSDB competition categories and by crop size.

Models: ``--cnn-config CONFIG --cnn-weights PT`` (a run of ``scripts/train_gtsrb_classifier.py``)
and/or ``--hf-model NAME`` (a Hugging Face image-classification checkpoint, output index = class id).

Usage:
    python scripts/eval_gtsrb_models_on_gtsdb_crops.py --cnn-config configs/gtsrb/stn_cnn_poojahira_32.yaml \
        --cnn-weights runs/gtsrb/stn_cnn_poojahira_32.pt --out runs/gtsdb/crops_stn_cnn.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsdb import category_of, load_gtsdb_train  # noqa: E402
from opentsr.signs.datasets.gtsrb import SHORT_NAMES  # noqa: E402
from opentsr.signs.models.gtsrb_cnn import build_model  # noqa: E402


def cut_crops(root: str, margin: float):
    crops, labels, cats, sizes = [], [], [], []
    for im in load_gtsdb_train(root):
        if not im.boxes:
            continue
        frame = Image.open(im.path).convert("RGB")
        w, h = frame.size
        for (x1, y1, x2, y2), c in zip(im.boxes, im.class_ids):
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            mx, my = margin * bw, margin * bh
            box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(x2 + 1 + mx)), min(h, int(y2 + 1 + my)))
            crops.append(frame.crop(box))
            labels.append(c)
            cats.append(category_of(c))
            sizes.append(max(bw, bh))
    return crops, np.array(labels), cats, np.array(sizes)


def summarize(name: str, preds: np.ndarray, labels: np.ndarray, cats: list[str], sizes: np.ndarray) -> dict:
    ok = preds == labels
    by_cat = {c: float(ok[[x == c for x in cats]].mean()) for c in sorted(set(cats))}
    by_size = {
        "<32px": float(ok[sizes < 32].mean()),
        "32-64px": float(ok[(sizes >= 32) & (sizes < 64)].mean()),
        ">=64px": float(ok[sizes >= 64].mean()),
    }
    cm = np.zeros((43, 43), dtype=int)
    np.add.at(cm, (labels, np.clip(preds, 0, 42)), 1)
    return {
        "model": name,
        "accuracy": float(ok.mean()),
        "n": int(len(labels)),
        "accuracy_by_category": by_cat,
        "accuracy_by_size": by_size,
        "per_class_recall": {SHORT_NAMES[i]: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(43)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gtsdb_hf")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--cnn-config", default=None)
    ap.add_argument("--cnn-weights", default=None)
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crops, labels, cats, sizes = cut_crops(args.root, args.margin)
    results = []
    t0 = time.time()

    if args.cnn_config:
        cfg = yaml.safe_load(Path(args.cnn_config).read_text())
        model = build_model(cfg["model"])
        model.load_state_dict(torch.load(args.cnn_weights, map_location="cpu"))
        model.eval()
        size = int(cfg.get("image_size", 32))
        x = np.stack([np.asarray(c.resize((size, size), Image.BILINEAR)) for c in crops])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            preds = torch.cat([model(x[i : i + 256]).argmax(1) for i in range(0, len(x), 256)]).numpy()
        results.append(summarize(f"{cfg['model']} ({args.cnn_weights})", preds, labels, cats, sizes))

    if args.hf_model:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(args.hf_model)
        model = AutoModelForImageClassification.from_pretrained(args.hf_model).eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(crops), 32):
                inputs = processor(images=crops[i : i + 32], return_tensors="pt")
                preds.append(model(**inputs).logits.argmax(-1))
        preds = torch.cat(preds).numpy()
        results.append(summarize(args.hf_model, preds, labels, cats, sizes))

    out = {
        "protocol": {
            "source": "GTSDB TrainIJCNN2013 frames + official gt.txt",
            "crops": int(len(labels)),
            "margin": args.margin,
            "task": "43-way classification of ground-truth crops",
        },
        "results": results,
        "seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for r in results:
        print(f"{r['model']}: {r['accuracy'] * 100:.2f}% on {r['n']} crops; by category {r['accuracy_by_category']}; by size {r['accuracy_by_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
