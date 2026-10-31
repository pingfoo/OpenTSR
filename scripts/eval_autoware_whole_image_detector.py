#!/usr/bin/env python
"""Class-agnostic localisation of the Autoware whole-image traffic light detector on S2TLD.

Model: ``AutowareFoundation/tensorrt_yolox`` tag v1.0,
``yolox_s_car_ped_tl_detector_960_960_batch_1.onnx`` (YOLOX-s, 960x960, classes BACKGROUND /
traffic_light / pedestrian_traffic_light), used by Autoware's ``yolox_traffic_light_detector``
launch (score 0.35, NMS 0.7 by default). The full frame is letterboxed to 960x960 exactly like
the ROI crops of the fine detector. Detections of both non-background classes are merged into one
class and scored against all S2TLD boxes with VOC AP at IoU 0.5 and 0.3, plus recall per state
and per box height at a confidence of 0.35.

For AP we run with a low score threshold (``--score-thresh 0.01``) so the precision-recall curve
is complete; the recall figures use Autoware's deployment threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.datasets.s2tld import load_s2tld  # noqa: E402
from opentsr.datasets.splits import read_manifest  # noqa: E402
from opentsr.metrics import Detection, GroundTruth, iou_xyxy, voc_ap_per_class  # noqa: E402
from opentsr.models.autoware_onnx import AutowareTLFineDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path("weights/autoware_whole/yolox_s_car_ped_tl_detector_960_960_batch_1.onnx"))
    ap.add_argument("--labels", type=Path, default=Path("weights/autoware_whole/car_ped_tl_detector_labels.txt"))
    ap.add_argument("--s2tld", type=Path, default=Path("data/s2tld"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thresh", type=float, default=0.01)
    ap.add_argument("--deploy-thresh", type=float, default=0.35, help="threshold for the recall figures")
    ap.add_argument("--nms-thresh", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    det = AutowareTLFineDetector(args.onnx, args.labels, args.threads, args.score_thresh, args.nms_thresh)
    positive = {i for i, l in enumerate(det.labels) if l != "BACKGROUND"}
    records = load_s2tld(args.s2tld / "extracted")
    ids = set(read_manifest(args.manifest)[args.split])
    records = [r for r in records if r.image_id in ids]
    if args.limit:
        records = records[: args.limit]

    dets: list[Detection] = []
    gts: list[GroundTruth] = []
    per_class_counts: dict[str, int] = defaultdict(int)
    t0 = time.time()
    for r in records:
        img = cv2.imread(str(r.image_path))
        for b in r.boxes:
            gts.append(GroundTruth(r.image_id, "tl", (b.x1, b.y1, b.x2, b.y2)))
        for d in det.detect_roi(img):  # the "ROI" is the whole frame
            if d.cls not in positive:
                continue
            per_class_counts[det.labels[d.cls]] += 1
            dets.append(Detection(r.image_id, "tl", d.score, (d.x1, d.y1, d.x2, d.y2)))
    seconds = time.time() - t0

    ap50 = voc_ap_per_class(dets, gts, ["tl"], iou_thr=0.5)["tl"]
    ap30 = voc_ap_per_class(dets, gts, ["tl"], iou_thr=0.3)["tl"]
    dets_by_img: dict[str, list[Detection]] = defaultdict(list)
    for d in dets:
        if d.score >= args.deploy_thresh:
            dets_by_img[d.image_id].append(d)
    found_state: dict[str, list[int]] = defaultdict(list)
    found_size: dict[str, list[int]] = defaultdict(list)
    n_fp = 0
    for r in records:
        dboxes = np.array([d.box for d in dets_by_img.get(r.image_id, [])]).reshape(-1, 4)
        gboxes = np.array([b.as_xyxy() for b in r.boxes]).reshape(-1, 4)
        if len(dboxes):
            if len(gboxes):
                n_fp += int((iou_xyxy(dboxes, gboxes).max(axis=1) < 0.5).sum())
            else:
                n_fp += len(dboxes)
        for b in r.boxes:
            hit = int(iou_xyxy(np.array(b.as_xyxy()), dboxes).max() >= 0.5) if len(dboxes) else 0
            found_state[b.label].append(hit)
            bucket = "h<16" if b.h < 16 else "16<=h<32" if b.h < 32 else "32<=h<64" if b.h < 64 else "h>=64"
            found_size[bucket].append(hit)

    result = {
        "model": str(args.onnx),
        "source": f"S2TLD {args.manifest}:{args.split}",
        "n_images": len(records),
        "n_gt": len(gts),
        "n_det_total": len(dets),
        "score_thresh_for_ap": args.score_thresh,
        "nms_thresh": args.nms_thresh,
        "ap50_class_agnostic": ap50["ap"],
        "ap30_class_agnostic": ap30["ap"],
        "detections_per_class_at_ap_thresh": dict(per_class_counts),
        "deploy_thresh": args.deploy_thresh,
        "recall_at_deploy_thresh_iou0.5_per_state": {k: float(np.mean(v)) for k, v in sorted(found_state.items())},
        "recall_at_deploy_thresh_iou0.5_per_box_height": {k: {"n": len(v), "recall": float(np.mean(v))} for k, v in sorted(found_size.items())},
        "false_positives_at_deploy_thresh": n_fp,
        "false_positives_per_image_at_deploy_thresh": n_fp / max(len(records), 1),
        "seconds": seconds,
        "ms_per_image": 1000.0 * seconds / max(len(records), 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
