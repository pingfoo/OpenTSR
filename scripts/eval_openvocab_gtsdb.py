#!/usr/bin/env python
"""Class-agnostic traffic sign detection on GTSDB with open-vocabulary detectors (zero-shot).

Models (``--detector``):
    owlv2        google/owlv2-base-patch16-ensemble via transformers (Apache-2.0)
    gdino        IDEA-Research/grounding-dino-tiny via transformers (Apache-2.0)
    yolo-world   Ultralytics YOLO-World (AGPL-3.0 code, used for a baseline number only)

Protocol: every GTSDB ground-truth box is mapped to one label "sign"; the detector is prompted
with a list of sign phrases (``--prompts``), all of its boxes are relabelled "sign", and
PASCAL VOC AP at IoU 0.5 (all-point interpolation, ``opentsr.metrics``) plus recall at a fixed
score threshold are reported. Recall is also broken down by competition category
(prohibitory / mandatory / danger / other) and by box size, which is what matters for a 16 to
128 px object. The 600 training images (852 boxes, 94 negatives) are used because the official
test ground truth is not in the Hugging Face mirror; pass ``--full-gt PATH --images DIR`` to score
the 300 official test images with the FullIJCNN2013 gt.txt instead.

Usage:
    python scripts/eval_openvocab_gtsdb.py --detector owlv2 --out runs/gtsdb/owlv2_train600.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.gtsdb import GTSDBImage, category_of, load_gtsdb_train, parse_gt  # noqa: E402
from opentsr.metrics import Detection, GroundTruth, iou_xyxy, voc_ap_per_class  # noqa: E402

DEFAULT_PROMPTS = [
    "traffic sign",
    "road sign",
    "speed limit sign",
    "round red and white prohibition sign",
    "triangular warning sign",
    "blue round mandatory sign",
    "stop sign",
    "yield sign",
]


def load_images(args) -> list[GTSDBImage]:
    if args.full_gt:
        gt = parse_gt(Path(args.full_gt))
        imgs = []
        for p in sorted(Path(args.images).glob("*.ppm")):
            items = gt.get(p.stem, [])
            imgs.append(GTSDBImage(p.stem, p, [b for b, _ in items], [c for _, c in items]))
        return imgs
    return load_gtsdb_train(args.root)


def run_owlv2(images, prompts, args):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    name = args.model or "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name).eval()
    texts = [prompts]
    t0 = time.time()
    for k, im in enumerate(images):
        pil = Image.open(im.path).convert("RGB")
        inputs = processor(text=texts, images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # OWLv2 pads the image to a square before resizing to 960; post_process expects the padded size.
        side = max(pil.size)
        res = processor.post_process_grounded_object_detection(
            outputs, threshold=args.min_score, target_sizes=torch.tensor([[side, side]]), text_labels=texts
        )[0]
        boxes = res["boxes"].numpy()
        scores = res["scores"].numpy()
        yield im, boxes, scores
        if k % 25 == 0:
            print(f"{k + 1}/{len(images)} {time.time() - t0:.0f}s", flush=True)


def run_gdino(images, prompts, args):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    name = args.model or "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).eval()
    text = ". ".join(prompts) + "."
    t0 = time.time()
    for k, im in enumerate(images):
        pil = Image.open(im.path).convert("RGB")
        inputs = processor(images=pil, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.min_score, text_threshold=args.min_score, target_sizes=[pil.size[::-1]]
        )[0]
        yield im, res["boxes"].numpy(), res["scores"].numpy()
        if k % 25 == 0:
            print(f"{k + 1}/{len(images)} {time.time() - t0:.0f}s", flush=True)


def run_yolo_world(images, prompts, args):
    from ultralytics import YOLO

    name = args.model or "yolov8s-worldv2.pt"
    model = YOLO(name)
    model.set_classes(prompts)
    t0 = time.time()
    for k, im in enumerate(images):
        pil = Image.open(im.path).convert("RGB")  # Ultralytics does not read PPM paths
        r = model.predict(pil, imgsz=args.imgsz, conf=args.min_score, verbose=False, device="cpu")[0]
        yield im, r.boxes.xyxy.numpy(), r.boxes.conf.numpy()
        if k % 25 == 0:
            print(f"{k + 1}/{len(images)} {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["owlv2", "gdino", "yolo-world"], required=True)
    ap.add_argument("--model", default=None, help="override the default checkpoint name")
    ap.add_argument("--root", default="data/gtsdb_hf")
    ap.add_argument("--full-gt", default=None, help="gt.txt of FullIJCNN2013 (scores the 300 test images)")
    ap.add_argument("--images", default=None, help="directory of the images matching --full-gt")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--min-score", type=float, default=0.05, help="score floor kept for the AP curve")
    ap.add_argument("--op-threshold", type=float, default=0.3, help="operating threshold for the recall/precision point")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO-World inference size")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    images = load_images(args)
    if args.limit:
        images = images[: args.limit]
    runner = {"owlv2": run_owlv2, "gdino": run_gdino, "yolo-world": run_yolo_world}[args.detector]

    gts: list[GroundTruth] = []
    dets: list[Detection] = []
    gt_meta = []  # (image_id, box, category, size)
    t0 = time.time()
    n_det_images = 0
    for im, boxes, scores in runner(images, prompts, args):
        for b, c in zip(im.boxes, im.class_ids):
            gts.append(GroundTruth(im.image_id, "sign", tuple(float(v) for v in b)))
            gt_meta.append((im.image_id, b, category_of(c), max(b[2] - b[0], b[3] - b[1])))
        for b, s in zip(boxes, scores):
            dets.append(Detection(im.image_id, "sign", float(s), tuple(float(v) for v in b)))
        n_det_images += 1

    per_class = voc_ap_per_class(dets, gts, ["sign"], iou_thr=0.5)
    per_class_30 = voc_ap_per_class(dets, gts, ["sign"], iou_thr=0.3)

    # Recall / precision at the operating threshold, with category and size breakdown.
    op = [d for d in dets if d.score >= args.op_threshold]
    by_img: dict[str, list[Detection]] = {}
    for d in op:
        by_img.setdefault(d.image_id, []).append(d)
    hit_by_cat: dict[str, list[int]] = {}
    hit_by_size: dict[str, list[int]] = {}
    matched_det = 0
    for img_id, box, cat, size in gt_meta:
        cand = by_img.get(img_id, [])
        hit = 0
        if cand:
            ious = iou_xyxy(np.array(box, dtype=float), np.array([d.box for d in cand]))[0]
            hit = int(ious.max() >= 0.5)
        hit_by_cat.setdefault(cat, []).append(hit)
        bucket = "<32px" if size < 32 else ("32-64px" if size < 64 else ">=64px")
        hit_by_size.setdefault(bucket, []).append(hit)
    # precision at threshold: detections overlapping any GT at IoU>=0.5
    gt_by_img: dict[str, list] = {}
    for img_id, box, _, _ in gt_meta:
        gt_by_img.setdefault(img_id, []).append(box)
    for d in op:
        g = gt_by_img.get(d.image_id, [])
        if g and iou_xyxy(np.array(d.box), np.array(g, dtype=float))[0].max() >= 0.5:
            matched_det += 1
    neg_images = [im.image_id for im in images if not im.boxes]
    fp_on_negatives = sum(len(by_img.get(i, [])) for i in neg_images)

    result = {
        "detector": args.detector,
        "model": args.model,
        "prompts": prompts,
        "protocol": {
            "images": len(images),
            "gt_boxes": len(gts),
            "source": "FullIJCNN2013 gt" if args.full_gt else "GTSDB TrainIJCNN2013 (600 images, official gt.txt)",
            "class_agnostic": True,
            "min_score_for_ap": args.min_score,
            "operating_threshold": args.op_threshold,
        },
        "ap50": per_class["sign"]["ap"],
        "ap30": per_class_30["sign"]["ap"],
        "max_recall_iou50": per_class["sign"]["recall"],
        "n_detections": len(dets),
        "at_operating_threshold": {
            "recall": float(np.mean([h for v in hit_by_cat.values() for h in v])) if gt_meta else 0.0,
            "precision": matched_det / max(len(op), 1),
            "n_detections": len(op),
            "recall_by_category": {k: float(np.mean(v)) for k, v in sorted(hit_by_cat.items())},
            "recall_by_size": {k: float(np.mean(v)) for k, v in sorted(hit_by_size.items())},
            "false_positives_on_negative_images": fp_on_negatives,
            "negative_images": len(neg_images),
        },
        "seconds": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("prompts",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
