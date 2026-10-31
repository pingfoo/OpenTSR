#!/usr/bin/env python
"""Download, extract, split and convert S2TLD to YOLO format.

Steps (each can be skipped with the matching flag):

1. ``--download``: fetch the two zip files from the Hugging Face mirror maintained by the
   dataset author (https://huggingface.co/datasets/yangxue/S2TLD) into ``<root>/raw``.
2. Extract into ``<root>/extracted`` and rename the full-width-parenthesis directories to
   ``S2TLD_720x1280`` and ``S2TLD_1080x1920``.
3. Write a split manifest (``--split-strategy`` random or chunked, seeded) to
   ``<manifest>``, or reuse an existing manifest with ``--manifest`` so that the exact split
   checked into ``configs/s2tld/splits/`` is used.
4. Write a YOLO dataset under ``<root>/yolo/<name>`` (images are symlinked, labels are
   normalised ``cls cx cy w h`` text files) plus ``data.yaml``.

Example (replicating the SCRDet++ random 6:4 protocol)::

    python scripts/prepare_s2tld.py --root data/s2tld --download \
        --manifest configs/s2tld/splits/random60_seed0.json --name random60

Example (OpenTSR chunked benchmark split, 60/10/30)::

    python scripts/prepare_s2tld.py --root data/s2tld \
        --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --name chunked
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.datasets.s2tld import S2TLD_CLASSES, load_s2tld  # noqa: E402
from opentsr.datasets.splits import make_split, read_manifest, write_manifest  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/yangxue/S2TLD/resolve/main/"
ZIPS = {
    # local name -> (remote name, sha256 measured on 2026-10-31)
    "S2TLD_720x1280.zip": (
        "S2TLD%EF%BC%88720x1280%EF%BC%89.zip",
        "303132bbb530dc877d82497941cad1ee005835196e8645a1e666edec0235bdb7",
    ),
    "S2TLD_1080x1920.zip": (
        "S2TLD%EF%BC%881080x1920%EF%BC%89.zip",
        "c81b968f8c365415c5f76f2fbfa5503b0bccdcb7926e75abf472cbc52b07a5de",
    ),
}
RENAMES = {
    "S2TLD（720x1280）": "S2TLD_720x1280",
    "S2TLD（1080x1920）": "S2TLD_1080x1920",
    "S2TLD#Uff08720x1280#Uff09": "S2TLD_720x1280",
    "S2TLD#Uff081080x1920#Uff09": "S2TLD_1080x1920",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for local, (remote, digest) in ZIPS.items():
        dst = raw_dir / local
        if dst.exists() and sha256(dst) == digest:
            print(f"[download] {local} present, checksum ok")
            continue
        print(f"[download] {local}")
        subprocess.run(["curl", "-sS", "-L", "-o", str(dst), HF_BASE + remote], check=True)
        got = sha256(dst)
        if got != digest:
            raise SystemExit(f"checksum mismatch for {local}: {got} != {digest}")


def extract(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for local in ZIPS:
        with zipfile.ZipFile(raw_dir / local) as zf:
            top = zf.namelist()[0].split("/")[0]
            target = out_dir / RENAMES.get(top, top)
            if target.is_dir() and any(target.iterdir()):
                print(f"[extract] {target} exists, skipping")
                continue
            print(f"[extract] {local} -> {out_dir}")
            zf.extractall(out_dir)
            src = out_dir / top
            if src != target:
                src.rename(target)
    # Some unzip tools decode the full-width parentheses differently; normalise.
    for p in list(out_dir.iterdir()):
        if p.name in RENAMES and not (out_dir / RENAMES[p.name]).exists():
            p.rename(out_dir / RENAMES[p.name])


def write_yolo(records, split: dict[str, list[str]], out_dir: Path, classes: list[str]) -> None:
    by_id = {r.image_id: r for r in records}
    cls_idx = {c: i for i, c in enumerate(classes)}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split_name, ids in split.items():
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for image_id in ids:
            r = by_id[image_id]
            stem = image_id.replace("/", "__").replace(" ", "_")
            link = img_dir / (stem + r.image_path.suffix)
            os.symlink(os.path.relpath(r.image_path, img_dir), link)
            lines = []
            for b in r.boxes:
                if b.label not in cls_idx:
                    raise ValueError(f"unknown label {b.label} in {image_id}")
                cx = (b.x1 + b.x2) / 2 / r.width
                cy = (b.y1 + b.y2) / 2 / r.height
                w = b.w / r.width
                h = b.h / r.height
                lines.append(f"{cls_idx[b.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    data = {
        "path": str(out_dir.resolve()),
        "names": {i: c for i, c in enumerate(classes)},
    }
    for split_name in split:
        data[split_name] = f"{split_name}/images"
    if "val" not in split and "test" in split:
        data["val"] = "test/images"  # ultralytics needs a val entry
    yaml_lines = [f"path: {data['path']}"]
    for k in [s for s in ("train", "val", "test") if s in data]:
        yaml_lines.append(f"{k}: {data[k]}")
    yaml_lines.append("names:")
    for i, c in enumerate(classes):
        yaml_lines.append(f"  {i}: {c}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/s2tld"))
    ap.add_argument("--download", action="store_true", help="download the zips if missing")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="split manifest to write (if missing) or reuse (if present)")
    ap.add_argument("--name", default="split", help="name of the YOLO dataset directory under <root>/yolo")
    ap.add_argument("--split-strategy", choices=["random", "chunked"], default="chunked")
    ap.add_argument("--fractions", default="train=0.6,val=0.1,test=0.3",
                    help="comma separated split=fraction list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--no-yolo", action="store_true", help="only write the manifest")
    args = ap.parse_args()

    raw_dir = args.root / "raw"
    ext_dir = args.root / "extracted"
    if args.download:
        download(raw_dir)
    if not (ext_dir / "S2TLD_720x1280").is_dir():
        extract(raw_dir, ext_dir)

    records = load_s2tld(ext_dir)
    n_boxes = sum(len(r.boxes) for r in records)
    print(f"[s2tld] {len(records)} images, {n_boxes} boxes")

    if args.manifest.exists():
        split = read_manifest(args.manifest)
        print(f"[split] reusing {args.manifest}")
    else:
        fractions = {k: float(v) for k, v in (kv.split("=") for kv in args.fractions.split(","))}
        split = make_split(records, fractions, args.split_strategy, args.seed, args.chunk_size)
        meta = {
            "dataset": "S2TLD",
            "strategy": args.split_strategy,
            "fractions": fractions,
            "seed": args.seed,
            "chunk_size": args.chunk_size if args.split_strategy == "chunked" else None,
            "classes": S2TLD_CLASSES,
        }
        write_manifest(args.manifest, split, meta)
        print(f"[split] wrote {args.manifest}")
    known = {r.image_id for r in records}
    for name, ids in split.items():
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"manifest lists unknown images in {name}: {missing[:3]}")
        print(f"[split] {name}: {len(ids)} images")

    if not args.no_yolo:
        out_dir = args.root / "yolo" / args.name
        write_yolo(records, split, out_dir, S2TLD_CLASSES)
        print(f"[yolo] wrote {out_dir / 'data.yaml'}")
    summary = {name: len(ids) for name, ids in split.items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
