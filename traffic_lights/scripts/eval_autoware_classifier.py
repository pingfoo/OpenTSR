#!/usr/bin/env python
"""Evaluate the Autoware MobileNetV2 traffic light classifier on public data.

Two data sources are supported:

``--folder DIR``
    An image-folder dataset with one sub-directory per class (red/green/yellow), for example
    the ``val`` split of the TIER IV sample dataset linked from the Autoware classifier README.

``--s2tld ROOT --manifest FILE --split NAME``
    Ground-truth crops from S2TLD (``ROOT/extracted``). Each annotated box is cropped (with an
    optional relative ``--margin``), classified, and the 11 Autoware car labels are collapsed to
    the S2TLD state vocabulary with ``LABEL_MAP`` below. ``wait_on`` has no Autoware counterpart
    and is reported separately.

Outputs a JSON file with accuracy, per-class recall, the collapsed confusion matrix and the raw
confusion matrix (dataset label x Autoware label).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from opentsr.traffic_lights.datasets.s2tld import load_s2tld  # noqa: E402
from opentsr.splits import read_manifest  # noqa: E402
from opentsr.metrics import accuracy, confusion_matrix  # noqa: E402
from opentsr.traffic_lights.models.autoware_onnx import AutowareTLClassifier  # noqa: E402

# Autoware car label -> coarse state used by S2TLD / BSTLD style datasets.
# Any label containing "red" is a red phase (red plus arrows), "unknown" is treated as "off".
LABEL_MAP = {
    "green": "green",
    "yellow": "yellow",
    "red": "red",
    "left,red": "red",
    "left,red,straight": "red",
    "red,right": "red",
    "red,straight": "red",
    "red,up_left": "red",
    "red,right,straight": "red",
    "red,up_right": "red",
    "unknown": "off",
}


def crop(img_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float, margin: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, np.floor(x1 - margin * bw)))
    y1 = int(max(0, np.floor(y1 - margin * bh)))
    x2 = int(min(w, np.ceil(x2 + margin * bw)))
    y2 = int(min(h, np.ceil(y2 + margin * bh)))
    return img_bgr[y1:y2, x1:x2]


def eval_folder(clf: AutowareTLClassifier, folder: Path) -> dict:
    y_true, y_pred, y_raw = [], [], []
    t0 = time.time()
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for img_path in sorted(cls_dir.iterdir()):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            label, _ = clf.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            y_true.append(cls_dir.name)
            y_raw.append(label)
            y_pred.append(LABEL_MAP.get(label, label))
    return summarize(y_true, y_pred, y_raw, clf.labels, time.time() - t0)


def eval_s2tld(clf: AutowareTLClassifier, root: Path, manifest: Path | None, split: str | None,
               margin: float, min_side: int) -> dict:
    records = load_s2tld(root / "extracted")
    if manifest is not None:
        ids = set(read_manifest(manifest)[split])
        records = [r for r in records if r.image_id in ids]
    y_true, y_pred, y_raw = [], [], []
    per_subset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    per_size: dict[str, list[tuple[str, str]]] = defaultdict(list)
    t0 = time.time()
    n_skipped = 0
    for r in records:
        img = cv2.imread(str(r.image_path))
        for b in r.boxes:
            if min(b.w, b.h) < min_side:
                n_skipped += 1
                continue
            roi = crop(img, b.x1, b.y1, b.x2, b.y2, margin)
            if roi.size == 0:
                n_skipped += 1
                continue
            label, _ = clf.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            mapped = LABEL_MAP.get(label, label)
            y_true.append(b.label)
            y_raw.append(label)
            y_pred.append(mapped)
            per_subset[r.group].append((b.label, mapped))
            bucket = "h<16" if b.h < 16 else "16<=h<32" if b.h < 32 else "32<=h<64" if b.h < 64 else "h>=64"
            per_size[bucket].append((b.label, mapped))
    out = summarize(y_true, y_pred, y_raw, clf.labels, time.time() - t0)
    out["n_skipped_small"] = n_skipped
    out["margin"] = margin
    out["per_subset"] = {
        k: {"n": len(v), "accuracy_excl_wait_on": _acc_excl(v)} for k, v in sorted(per_subset.items())
    }
    out["per_box_height"] = {
        k: {"n": len(v), "accuracy_excl_wait_on": _acc_excl(v)} for k, v in sorted(per_size.items())
    }
    return out


def _acc_excl(pairs: list[tuple[str, str]]) -> float | None:
    pairs = [(t, p) for t, p in pairs if t != "wait_on"]
    if not pairs:
        return None
    return accuracy([t for t, _ in pairs], [p for _, p in pairs])


def summarize(y_true, y_pred, y_raw, raw_labels, seconds: float) -> dict:
    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred) | set(true_labels))
    # Collapsed confusion (dataset label x mapped prediction).
    cm_labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(y_true, y_pred, cm_labels)
    # Raw confusion (dataset label x Autoware label).
    raw_cm = {t: dict(Counter(p for tt, p in zip(y_true, y_raw) if tt == t)) for t in true_labels}
    evaluable = [(t, p) for t, p in zip(y_true, y_pred) if t != "wait_on"]
    per_class = {}
    for t in true_labels:
        n = sum(1 for tt in y_true if tt == t)
        correct = sum(1 for tt, p in zip(y_true, y_pred) if tt == t and p == t)
        per_class[t] = {"n": n, "recall": correct / n if n else None}
    return {
        "n": len(y_true),
        "accuracy_excl_wait_on": accuracy([t for t, _ in evaluable], [p for _, p in evaluable]),
        "n_evaluable": len(evaluable),
        "per_class": per_class,
        "confusion_labels": cm_labels,
        "confusion": cm.tolist(),
        "raw_confusion": raw_cm,
        "seconds": seconds,
        "ms_per_crop": 1000.0 * seconds / max(len(y_true), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path("weights/traffic_lights/autoware/traffic_light_classifier_mobilenetv2_batch_1.onnx"))
    ap.add_argument("--labels", type=Path, default=Path("weights/traffic_lights/autoware/lamp_labels.txt"))
    ap.add_argument("--folder", type=Path, help="image-folder dataset (one directory per class)")
    ap.add_argument("--s2tld", type=Path, help="S2TLD root (containing extracted/)")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--margin", type=float, default=0.0, help="relative margin added around each GT box")
    ap.add_argument("--min-side", type=int, default=1, help="skip boxes whose shorter side is below this")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--preprocess", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox = Autoware runtime behaviour, stretch = mmpretrain training pipeline")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clf = AutowareTLClassifier(args.onnx, args.labels, args.threads, preprocess=args.preprocess)
    if args.folder:
        result = eval_folder(clf, args.folder)
        result["source"] = str(args.folder)
    elif args.s2tld:
        result = eval_s2tld(clf, args.s2tld, args.manifest, args.split, args.margin, args.min_side)
        result["source"] = f"S2TLD {args.manifest}:{args.split}" if args.manifest else "S2TLD all"
    else:
        ap.error("give --folder or --s2tld")
    result["model"] = str(args.onnx)
    result["preprocess"] = args.preprocess
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("source", "n", "accuracy_excl_wait_on", "per_class", "ms_per_crop")}, indent=1))
    print("confusion labels:", result["confusion_labels"])
    for lbl, row in zip(result["confusion_labels"], result["confusion"]):
        print(f"  {lbl:>8}: {row}")


if __name__ == "__main__":
    main()
