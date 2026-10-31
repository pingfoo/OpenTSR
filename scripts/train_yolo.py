#!/usr/bin/env python
"""Train and evaluate an Ultralytics YOLO detector from an OpenTSR config.

The config (YAML) names the base weights, the YOLO ``data.yaml`` produced by
``scripts/prepare_s2tld.py``, the split manifest it came from, and the training arguments.
After training, the best checkpoint is evaluated on ``eval_split`` and the metrics are written
next to the run as ``metrics_<split>.json`` together with the config, the git commit and the
package versions, so every number in ``docs/`` maps back to a config and a commit.

Examples::

    # GPU box (lab server or Mac with MPS); device is chosen automatically
    python scripts/train_yolo.py --config configs/s2tld/yolo11s_1280_chunked.yaml

    # CPU smoke test: 1 epoch on 32 images at 640 px, to validate the pipeline only
    python scripts/train_yolo.py --config configs/s2tld/yolo11n_1280_chunked.yaml --smoke

    # Evaluate an existing checkpoint only
    python scripts/train_yolo.py --config ... --eval-only runs/s2tld_yolo11s_1280_chunked/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def versions() -> dict:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "platform": platform.platform(),
    }


def make_smoke_dataset(data_yaml: Path, n_images: int, out_dir: Path) -> Path:
    """Copy the first ``n_images`` of each split into a throwaway dataset for a CPU smoke test."""
    data = yaml.safe_load(data_yaml.read_text())
    base = Path(data["path"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    new = dict(data)
    new["path"] = str(out_dir.resolve())
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        src_imgs = base / data[split]
        src_lbls = src_imgs.parent / "labels"
        dst_imgs = out_dir / split / "images"
        dst_lbls = out_dir / split / "labels"
        dst_imgs.mkdir(parents=True)
        dst_lbls.mkdir(parents=True)
        for img in sorted(src_imgs.iterdir())[:n_images]:
            (dst_imgs / img.name).symlink_to(img.resolve())
            lbl = src_lbls / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dst_lbls / lbl.name)
        new[split] = f"{split}/images"
    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(new, sort_keys=False))
    return out


def evaluate(weights: Path, data_yaml: Path, split: str, imgsz: int, out_dir: Path, cfg: dict, extra: dict) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=8, plots=False,
                    project=str(out_dir), name=f"val_{split}", exist_ok=True, verbose=False)
    names = res.names
    per_class = {}
    for i, c in names.items():
        if i in getattr(res.box, "ap_class_index", []):
            j = list(res.box.ap_class_index).index(i)
            per_class[c] = {"ap50": float(res.box.ap50[j]), "ap50_95": float(res.box.ap[j]),
                            "precision": float(res.box.p[j]), "recall": float(res.box.r[j])}
    metrics = {
        "config": cfg,
        "weights": str(weights),
        "split": split,
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "per_class": per_class,
        "git_commit": git_commit(),
        "versions": versions(),
        **extra,
    }
    path = out_dir / f"metrics_{split}.json"
    path.write_text(json.dumps(metrics, indent=1) + "\n")
    print(json.dumps({k: metrics[k] for k in ("split", "map50", "map50_95", "per_class")}, indent=1))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps; default: Ultralytics auto")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a 32-image subset at 640 px (pipeline check)")
    ap.add_argument("--eval-only", type=Path, default=None, help="skip training, evaluate these weights")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_yaml = ROOT / cfg["data"]
    train_args = dict(cfg.get("train", {}))
    name = cfg["name"]
    extra = {"smoke": args.smoke}
    if args.smoke:
        name = name + "_smoke"
        data_yaml = make_smoke_dataset(data_yaml, 32, args.runs_dir / name / "smoke_data")
        train_args.update({"epochs": 1, "imgsz": 640, "batch": 4, "workers": 0})
        extra["note"] = "smoke test: numbers are meaningless"
    if args.device:
        train_args["device"] = args.device
    out_dir = args.runs_dir / name

    if args.eval_only is None:
        from ultralytics import YOLO

        model = YOLO(cfg["model"])
        model.train(data=str(data_yaml), project=str(args.runs_dir), name=name, exist_ok=True, **train_args)
        weights = out_dir / "weights" / "best.pt"
    else:
        weights = args.eval_only
    evaluate(weights, data_yaml, cfg.get("eval_split", "test"), train_args.get("imgsz", 640), out_dir, cfg, extra)


if __name__ == "__main__":
    main()
