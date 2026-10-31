#!/usr/bin/env python
"""Validate an Ultralytics checkpoint on the LISA test split exactly as DetectionBench does.

DetectionBench (https://github.com/dronefreak/DetectionBench) reports ``model.val(split="test")``
numbers on the Hugging Face export of LISA at 640 px. This script runs the same call and writes
the metrics next to the reported ones from the model card so the two can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data_abs.yaml written by traffic_lights/scripts/prepare_lisa_hf.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reported", default="map50=23.74,map50_95=10.57,precision=37.23,recall=25.7",
                    help="numbers from the model card, for the side-by-side in the output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(args.weights))
    res = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=2, plots=False, project=str(args.out.parent),
                    name=args.out.stem + "_val", exist_ok=True, verbose=True)
    per = {}
    for j, i in enumerate(res.box.ap_class_index):
        per[res.names[int(i)]] = {"ap50": float(res.box.ap50[j]), "ap50_95": float(res.box.ap[j]),
                                  "precision": float(res.box.p[j]), "recall": float(res.box.r[j])}
    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    out = {
        "weights": str(args.weights), "sha256": sha, "data": str(args.data), "split": args.split,
        "imgsz": args.imgsz, "map50": float(res.box.map50), "map50_95": float(res.box.map),
        "precision": float(res.box.mp), "recall": float(res.box.mr), "per_class": per,
        "reported": {k: float(v) for k, v in (kv.split("=") for kv in args.reported.split(","))},
        "seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
