#!/usr/bin/env python
"""Class-agnostic traffic light localisation with COCO-pretrained Ultralytics YOLO weights.

COCO has a ``traffic light`` class (id 9) but no state labels, so a COCO detector can only be
scored on localisation. All S2TLD ground-truth boxes are merged into one class and the VOC
all-point AP at IoU 0.5 (and 0.3) is computed, plus recall per state and per box height.
This gives the "free" baseline every open detector starts from before any traffic-light
specific fine-tuning.

Weights are downloaded by Ultralytics from its GitHub releases (AGPL-3.0 code and weights,
see docs/third_party.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.datasets.s2tld import load_s2tld  # noqa: E402
from opentsr.datasets.splits import read_manifest  # noqa: E402
from opentsr.metrics import Detection, GroundTruth, iou_xyxy, voc_ap_per_class  # noqa: E402

COCO_TRAFFIC_LIGHT = 9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--s2tld", type=Path, default=Path("data/s2tld"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    records = load_s2tld(args.s2tld / "extracted")
    ids = set(read_manifest(args.manifest)[args.split])
    records = [r for r in records if r.image_id in ids]
    if args.limit:
        records = records[: args.limit]

    dets: list[Detection] = []
    gts: list[GroundTruth] = []
    t0 = time.time()
    for r in records:
        for b in r.boxes:
            gts.append(GroundTruth(r.image_id, "tl", (b.x1, b.y1, b.x2, b.y2)))
        res = model.predict(str(r.image_path), imgsz=args.imgsz, conf=args.conf,
                            classes=[COCO_TRAFFIC_LIGHT], verbose=False, device="cpu")[0]
        for xyxy, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            dets.append(Detection(r.image_id, "tl", float(conf), tuple(float(v) for v in xyxy)))
    seconds = time.time() - t0

    ap50 = voc_ap_per_class(dets, gts, ["tl"], iou_thr=0.5)["tl"]
    ap30 = voc_ap_per_class(dets, gts, ["tl"], iou_thr=0.3)["tl"]

    # Recall at conf >= 0.25 per state and per box height (best IoU >= 0.5).
    dets_by_img: dict[str, list[Detection]] = defaultdict(list)
    for d in dets:
        if d.score >= 0.25:
            dets_by_img[d.image_id].append(d)
    found_state: dict[str, list[int]] = defaultdict(list)
    found_size: dict[str, list[int]] = defaultdict(list)
    for r in records:
        dboxes = np.array([d.box for d in dets_by_img.get(r.image_id, [])]).reshape(-1, 4)
        for b in r.boxes:
            hit = 0
            if len(dboxes):
                hit = int(iou_xyxy(np.array(b.as_xyxy()), dboxes).max() >= 0.5)
            found_state[b.label].append(hit)
            bucket = "h<16" if b.h < 16 else "16<=h<32" if b.h < 32 else "32<=h<64" if b.h < 64 else "h>=64"
            found_size[bucket].append(hit)

    result = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "source": f"S2TLD {args.manifest}:{args.split}",
        "n_images": len(records),
        "n_gt": len(gts),
        "n_det": len(dets),
        "ap50_class_agnostic": ap50["ap"],
        "ap30_class_agnostic": ap30["ap"],
        "recall_at_conf0.25_iou0.5_per_state": {k: float(np.mean(v)) for k, v in sorted(found_state.items())},
        "recall_at_conf0.25_iou0.5_per_box_height": {k: {"n": len(v), "recall": float(np.mean(v))} for k, v in sorted(found_size.items())},
        "seconds": seconds,
        "ms_per_image": 1000.0 * seconds / max(len(records), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
