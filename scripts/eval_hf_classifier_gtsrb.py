#!/usr/bin/env python
"""Score a Hugging Face image-classification checkpoint on the official GTSRB test split.

Public GTSRB checkpoints on the Hub report accuracies on "the evaluation set" without saying
which split that is, and several were trained on non-official splits (for example
``tanganke/gtsrb`` has 26,640 training images instead of 39,209). This script applies a
checkpoint to the 12,630 official test images with the checkpoint's own image processor and
reports top-1 accuracy, so that Hub claims can be compared on one protocol.

The checkpoint's output index ``i`` is assumed to be GTSRB class id ``i`` (``--label-order index``,
the default; the script prints ``id2label`` so that the assumption can be checked by eye).

Usage:
    python scripts/eval_hf_classifier_gtsrb.py --model bazyl/gtsrb-model
    python scripts/eval_hf_classifier_gtsrb.py --model FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsrb import SHORT_NAMES, load_gtsrb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--root", default="data/gtsrb_hf")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-roi", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForImageClassification.from_pretrained(args.model, revision=args.revision).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print("id2label (first 6):", {k: id2label[k] for k in sorted(id2label)[:6]}, "num_labels", len(id2label))

    data = load_gtsrb(args.root, args.split)
    n = len(data.images) if not args.limit else min(args.limit, len(data.images))
    preds = np.empty(n, dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            imgs = []
            for j in range(i, min(i + args.batch_size, n)):
                img = data.images[j]
                if args.use_roi:
                    img = img.crop(tuple(data.rois[j].tolist()))
                imgs.append(img)
            inputs = processor(images=imgs, return_tensors="pt")
            logits = model(**inputs).logits
            preds[i : i + len(imgs)] = logits.argmax(-1).numpy()
            if (i // args.batch_size) % 20 == 0:
                print(f"{i + len(imgs)}/{n}  {time.time() - t0:.0f}s", flush=True)
    labels = data.labels[:n]
    acc = float((preds == labels).mean())
    k = max(43, int(preds.max()) + 1)
    cm = np.zeros((43, k), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    per_class = (cm[:, :43].diagonal() / np.maximum(cm.sum(1), 1)).tolist()
    result = {
        "model": args.model,
        "revision": args.revision,
        "id2label": {str(i): id2label[i] for i in sorted(id2label)},
        "protocol": {"split": args.split, "images": n, "crop": "official ROI" if args.use_roi else "full image", "preprocessing": "checkpoint image processor"},
        "top1_accuracy": acc,
        "errors": int((preds != labels).sum()),
        "per_class_recall": {SHORT_NAMES[i]: r for i, r in enumerate(per_class)},
        "confusion_matrix": cm.tolist(),
        "seconds": time.time() - t0,
    }
    out = Path(args.out) if args.out else Path("runs/gtsrb") / f"hf_{args.model.replace('/', '__')}{'_roi' if args.use_roi else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.model}: top-1 {acc * 100:.2f}% ({result['errors']} errors of {n}) in {time.time() - t0:.0f}s")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
