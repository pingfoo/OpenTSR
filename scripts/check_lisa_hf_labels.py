#!/usr/bin/env python
"""Check the label files shipped in the Hugging Face LISA export against its metadata.jsonl.

The export ``dronefreak/LISA-Traffic-Lights`` ships both ``metadata.jsonl`` (pixel boxes
``[x, y, w, h]``) and YOLO ``labels/*.txt`` (normalised). On 2026-10-31 the two disagreed: the
``metadata.jsonl`` boxes are expressed in a 640x640 frame, not in the 1280x960 image frame, while
the ``.txt`` files are correct (they put the boxes on the lights). This script quantifies the
disagreement (IoU between the two variants, with the metadata boxes used as-is) and evaluates a
detector against both variants with the in-repo VOC AP, so the effect of picking the wrong one
can be seen directly. See ``scripts/prepare_lisa_hf.py`` for the rescaling.

Usage::

    python scripts/check_lisa_hf_labels.py --weights weights/detectionbench/lisa-yolo26n_best.pt \
        --shard data/lisa_hf/data/images/test/shard_000 --out runs/replication/lisa_label_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.metrics import Detection, GroundTruth, iou_xyxy, voc_ap_per_class  # noqa: E402

NAMES = ["go", "goForward", "goLeft", "stop", "stopLeft", "warning", "warningLeft"]


def read_official(txt: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        out.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--shard", type=Path, required=True, help="an images/<split>/shard_xxx directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    split = args.shard.parent.name
    lbl_dir = args.shard.parents[2] / "labels" / split / args.shard.name
    records = [json.loads(l) for l in (args.shard / "metadata.jsonl").read_text().splitlines()]
    if args.limit:
        records = records[: args.limit]

    gts_meta: list[GroundTruth] = []
    gts_txt: list[GroundTruth] = []
    dets: list[Detection] = []
    mismatch = []
    sizes = set()
    t0 = time.time()
    for rec in records:
        img_path = args.shard / rec["file_name"]
        with Image.open(img_path) as im:
            w, h = im.size
        sizes.add((w, h))
        stem = Path(rec["file_name"]).stem
        meta_boxes = [(c, x, y, x + bw, y + bh) for (x, y, bw, bh), c in zip(rec["objects"]["bbox"], rec["objects"]["categories"])]
        txt_boxes = read_official(lbl_dir / (stem + ".txt"), w, h)
        for c, x1, y1, x2, y2 in meta_boxes:
            gts_meta.append(GroundTruth(stem, NAMES[c], (x1, y1, x2, y2)))
        for c, x1, y1, x2, y2 in txt_boxes:
            gts_txt.append(GroundTruth(stem, NAMES[c], (x1, y1, x2, y2)))
        if meta_boxes and txt_boxes and len(meta_boxes) == len(txt_boxes):
            a = np.array([b[1:] for b in meta_boxes])
            b = np.array([b[1:] for b in txt_boxes])
            mismatch.append(float(np.mean(np.diag(iou_xyxy(a, b)))))
        res = model.predict(str(img_path), imgsz=args.imgsz, conf=0.001, device="cpu", verbose=False)[0]
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            dets.append(Detection(stem, NAMES[int(cls)], float(conf), tuple(float(v) for v in xyxy)))
    seconds = time.time() - t0

    def summarise(gts):
        r = voc_ap_per_class(dets, gts, NAMES, iou_thr=0.5)
        present = [c for c in NAMES if r[c]["n_gt"] > 0]
        return {"map50_present_classes": float(np.mean([r[c]["ap"] for c in present])) if present else 0.0,
                "per_class_ap50": {c: r[c]["ap"] for c in NAMES}, "n_gt": {c: r[c]["n_gt"] for c in NAMES}}

    result = {
        "weights": args.weights,
        "shard": str(args.shard),
        "n_images": len(records),
        "image_sizes": sorted(sizes),
        "mean_iou_metadata_as_is_vs_shipped_txt": float(np.mean(mismatch)) if mismatch else None,
        "n_images_compared": len(mismatch),
        "n_detections_conf_ge_0.25": sum(1 for d in dets if d.score >= 0.25),
        "against_metadata_jsonl_as_is": summarise(gts_meta),
        "against_shipped_txt": summarise(gts_txt),
        "seconds": seconds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
