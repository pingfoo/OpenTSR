#!/usr/bin/env python
"""Evaluate the Autoware YOLOX-s traffic light fine detector on S2TLD.

The Autoware fine detector does not scan whole frames. It receives coarse ROIs from the
map-based detector (projected from the HD map, with a generous margin) and returns a tight
box per ROI. We emulate that setting without a map:

* For every ground-truth box, a coarse ROI is built by scaling the box by ``--roi-scale`` around
  its centre (at least ``--roi-min`` pixels on each side), then shifting it by a seeded random
  offset of up to ``--jitter`` times the ROI size, to mimic projection error. The ROI is clipped
  to the frame.
* The detector runs on the ROI crop (letterbox to 416, Autoware preprocessing). The detection
  with the highest score of class ``traffic_light`` or ``pedestrian_traffic_light`` is compared
  with the ground truth: we report recall at IoU 0.5 and 0.3, mean IoU of the best detection,
  the fraction of ROIs with no detection, and the mean number of detections per ROI.
* As a negative control, ``--negatives N`` random ROIs that overlap no ground truth are sampled
  per image (same size distribution) and the rate of ROIs with at least one detection is
  reported (false alarms).

The state labels of S2TLD are not used here; the fine detector only localises lights.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from opentsr.traffic_lights.datasets.s2tld import load_s2tld  # noqa: E402
from opentsr.splits import read_manifest  # noqa: E402
from opentsr.metrics import iou_xyxy  # noqa: E402
from opentsr.traffic_lights.models.autoware_onnx import AutowareTLFineDetector  # noqa: E402


def make_roi(b, width, height, scale, min_side, jitter, rng):
    cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    rw, rh = max(b.w * scale, min_side), max(b.h * scale, min_side)
    dx = rng.uniform(-jitter, jitter) * rw
    dy = rng.uniform(-jitter, jitter) * rh
    x1 = int(np.clip(cx + dx - rw / 2, 0, width - 1))
    y1 = int(np.clip(cy + dy - rh / 2, 0, height - 1))
    x2 = int(np.clip(cx + dx + rw / 2, x1 + 1, width))
    y2 = int(np.clip(cy + dy + rh / 2, y1 + 1, height))
    return x1, y1, x2, y2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path("weights/traffic_lights/autoware/tlr_car_ped_yolox_s_batch_1.onnx"))
    ap.add_argument("--labels", type=Path, default=Path("weights/traffic_lights/autoware/tlr_labels.txt"))
    ap.add_argument("--s2tld", type=Path, default=Path("data/traffic_lights/s2tld"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--roi-scale", type=float, default=4.0)
    ap.add_argument("--roi-min", type=int, default=96)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--negatives", type=int, default=1)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--nms-thresh", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N images (smoke test)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    det = AutowareTLFineDetector(args.onnx, args.labels, args.threads, args.score_thresh, args.nms_thresh)
    positive_classes = {i for i, l in enumerate(det.labels) if l != "BACKGROUND"}
    records = load_s2tld(args.s2tld / "extracted")
    ids = set(read_manifest(args.manifest)[args.split])
    records = [r for r in records if r.image_id in ids]
    if args.limit:
        records = records[: args.limit]
    rng = random.Random(args.seed)

    ious: list[float] = []
    n_det_per_roi: list[int] = []
    per_group: dict[str, list[float]] = defaultdict(list)
    per_size: dict[str, list[float]] = defaultdict(list)
    cls_counts: dict[str, int] = defaultdict(int)
    neg_total = 0
    neg_fired = 0
    t0 = time.time()
    for r in records:
        img = cv2.imread(str(r.image_path))
        gt_boxes = np.array([b.as_xyxy() for b in r.boxes]).reshape(-1, 4)
        for b in r.boxes:
            x1, y1, x2, y2 = make_roi(b, r.width, r.height, args.roi_scale, args.roi_min, args.jitter, rng)
            dets = [d for d in det.detect_roi(img[y1:y2, x1:x2]) if d.cls in positive_classes]
            n_det_per_roi.append(len(dets))
            best_iou = 0.0
            if dets:
                best = max(dets, key=lambda d: d.score)
                cls_counts[det.labels[best.cls]] += 1
                box = np.array([best.x1 + x1, best.y1 + y1, best.x2 + x1, best.y2 + y1])
                best_iou = float(iou_xyxy(box, np.array(b.as_xyxy()))[0, 0])
            ious.append(best_iou)
            per_group[r.group].append(best_iou)
            bucket = "h<16" if b.h < 16 else "16<=h<32" if b.h < 32 else "32<=h<64" if b.h < 64 else "h>=64"
            per_size[bucket].append(best_iou)
        # negatives: random ROIs that do not touch any GT box
        for _ in range(args.negatives):
            ref = rng.choice(r.boxes) if r.boxes else None
            rw = max((ref.w if ref else 20) * args.roi_scale, args.roi_min)
            rh = max((ref.h if ref else 40) * args.roi_scale, args.roi_min)
            for _attempt in range(20):
                x1 = rng.randint(0, max(0, int(r.width - rw)))
                y1 = rng.randint(0, max(0, int(r.height - rh)))
                x2, y2 = int(min(r.width, x1 + rw)), int(min(r.height, y1 + rh))
                roi = np.array([x1, y1, x2, y2])
                if len(gt_boxes) and iou_xyxy(roi, gt_boxes).max() > 0:
                    continue
                # also reject ROIs that contain any GT centre
                if len(gt_boxes) and np.any(
                    ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 > x1) & ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 < x2)
                    & ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 > y1) & ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 < y2)
                ):
                    continue
                neg_total += 1
                dets = [d for d in det.detect_roi(img[y1:y2, x1:x2]) if d.cls in positive_classes]
                if dets:
                    neg_fired += 1
                break
    seconds = time.time() - t0
    ious_a = np.array(ious)

    def summ(v):
        v = np.array(v)
        return {
            "n": int(v.size),
            "recall@0.5": float((v >= 0.5).mean()) if v.size else None,
            "recall@0.3": float((v >= 0.3).mean()) if v.size else None,
            "mean_iou": float(v.mean()) if v.size else None,
            "no_detection": float((v == 0).mean()) if v.size else None,
        }

    result = {
        "model": str(args.onnx),
        "source": f"S2TLD {args.manifest}:{args.split}",
        "n_images": len(records),
        "settings": {k: getattr(args, k) for k in ("roi_scale", "roi_min", "jitter", "negatives", "score_thresh", "nms_thresh", "seed")},
        "positives": summ(ious_a),
        "mean_detections_per_positive_roi": float(np.mean(n_det_per_roi)) if n_det_per_roi else None,
        "best_detection_class_counts": dict(cls_counts),
        "negatives": {"n": neg_total, "false_alarm_rate": neg_fired / neg_total if neg_total else None},
        "per_subset": {k: summ(v) for k, v in sorted(per_group.items())},
        "per_box_height": {k: summ(v) for k, v in sorted(per_size.items())},
        "seconds": seconds,
        "ms_per_roi": 1000.0 * seconds / max(len(ious) + neg_total, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("source", "n_images", "settings", "positives", "negatives", "best_detection_class_counts", "ms_per_roi")}, indent=1))
    print("per box height:", json.dumps(result["per_box_height"]))


if __name__ == "__main__":
    main()
