#!/usr/bin/env python
"""Score the Panoramax German road-sign classifier on its own validation split (134 classes).

Panoramax/classify_de_road_signs (etalab-2.0) is an Ultralytics classification checkpoint trained
on Panoramax/classified_de_road_signs (CC BY-SA 4.0, crops of German signs from Panoramax street
imagery, "130+ classes" plus "bad" and "other" classes). The model card reports accuracy 0.986.

Safety note: Ultralytics ``.pt`` files are full pickles. This script does NOT call ``YOLO()`` or
``torch.load(weights_only=False)``. It first disassembles the pickle with ``pickletools`` and
refuses to continue if any global outside an allow-list of torch / torchvision / ultralytics
module classes is referenced, then loads with ``torch.load(weights_only=True)`` and the same
allow-list registered through ``torch.serialization.add_safe_globals``. Inference is plain PyTorch
(the checkpoint carries its own torchvision eval transform: resize, centre crop, normalise).

Usage:
    hf download Panoramax/classify_de_road_signs classify_de_road_signs.pt --local-dir weights/panoramax
    hf download Panoramax/classified_de_road_signs val.zip --repo-type dataset --local-dir data/panoramax_de
    (cd data/panoramax_de && unzip -q val.zip)
    python scripts/eval_panoramax_de_classifier.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ALLOWED_MODULES = ("torch", "torch._utils", "torch.nn.modules", "torchvision.transforms", "collections", "__builtin__", "builtins", "ultralytics.nn")


def pickle_globals(pt_path: Path) -> list[str]:
    with zipfile.ZipFile(pt_path) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        data = z.read(pkl)
    ops = list(pickletools.genops(data))
    found = set()
    for i, (op, arg, _) in enumerate(ops):
        if op.name == "GLOBAL":
            found.add(arg.replace(" ", "."))
        elif op.name == "STACK_GLOBAL":
            found.add(f"{ops[i - 2][1]}.{ops[i - 1][1]}")
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/panoramax/classify_de_road_signs.pt")
    ap.add_argument("--val-dir", default="data/panoramax_de/val")
    ap.add_argument("--imgsz", type=int, default=None, help="override the checkpoint's training image size")
    ap.add_argument("--out", default="runs/panoramax/classify_de_val.json")
    args = ap.parse_args()

    pt = Path(args.weights)
    globs = pickle_globals(pt)
    bad = [g for g in globs if not any(g.startswith(m) for m in ALLOWED_MODULES)]
    if bad:
        raise SystemExit(f"refusing to load {pt}: unexpected globals in pickle: {bad}")
    import importlib

    safe = []
    for g in globs:
        mod, _, name = g.rpartition(".")
        if mod in ("__builtin__", "builtins"):
            safe.append(getattr(__buil__ := __builtins__, name) if isinstance(__builtins__, dict) is False else __builtins__[name])
            continue
        safe.append(getattr(importlib.import_module(mod), name))
    torch.serialization.add_safe_globals(safe)
    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    model = ckpt.get("ema") or ckpt["model"]
    model = model.float().eval()
    names = model.names if hasattr(model, "names") else ckpt.get("names")
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    train_args = ckpt.get("train_args", {}) or {}
    imgsz = args.imgsz or int(train_args.get("imgsz", 224))
    # Ultralytics classification eval transform: resize short side to imgsz, centre crop, ToTensor, no mean/std
    # (ultralytics.data.augment.classify_transforms, default mean 0 / std 1). Reproduced here with torchvision.
    from torchvision import transforms as T

    tf = T.Compose([T.Resize(imgsz), T.CenterCrop(imgsz), T.ToTensor()])

    val = Path(args.val_dir)
    folders = sorted(p.name for p in val.iterdir() if p.is_dir())
    if folders != names:
        diff = [(a, b) for a, b in zip(folders, names) if a != b][:5]
        raise SystemExit(f"class list mismatch between model ({len(names)}) and val folders ({len(folders)}): {diff}")
    files = []
    for ci, f in enumerate(folders):
        for p in sorted((val / f).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((p, ci))
    preds = np.empty(len(files), dtype=np.int64)
    top5 = np.zeros(len(files), dtype=bool)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(files), 64):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in files[i : i + 64]])
            out = model(batch)
            out = out[0] if isinstance(out, (tuple, list)) else out
            if out.dim() == 2 and out.shape[1] == len(names):
                logits = out
            else:
                raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")
            preds[i : i + len(batch)] = logits.argmax(1).numpy()
            labels = torch.tensor([c for _, c in files[i : i + 64]])
            top5[i : i + len(batch)] = (logits.topk(5, dim=1).indices == labels[:, None]).any(1).numpy()
            if (i // 64) % 20 == 0:
                print(f"{i + len(batch)}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    labels = np.array([c for _, c in files])
    acc = float((preds == labels).mean())
    # accuracy over real sign classes only (drop the bad:* and other:* buckets)
    real = np.array([not (names[c].startswith("bad") or names[c].startswith("other")) for c in labels])
    per_class = {}
    for ci, n in enumerate(names):
        m = labels == ci
        if m.any():
            per_class[n] = {"n": int(m.sum()), "recall": float((preds[m] == ci).mean())}
    result = {
        "weights": str(pt),
        "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
        "pickle_globals": globs,
        "imgsz": imgsz,
        "train_args_subset": {k: train_args.get(k) for k in ("model", "imgsz", "epochs", "batch", "data") if k in train_args},
        "val_images": int(len(files)),
        "classes": len(names),
        "top1_accuracy": acc,
        "top5_accuracy": float(top5.mean()),
        "top1_accuracy_sign_classes_only": float((preds[real] == labels[real]).mean()),
        "sign_class_images": int(real.sum()),
        "reference": "model card accuracy 0.986",
        "per_class": per_class,
        "seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"top-1 {acc * 100:.2f}% top-5 {result['top5_accuracy'] * 100:.2f}% on {len(files)} val crops ({len(names)} classes, imgsz {imgsz})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
