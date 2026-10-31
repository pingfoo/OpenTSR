#!/usr/bin/env python
"""Make the Hugging Face LISA export (dronefreak/LISA-Traffic-Lights) usable with Ultralytics.

The export has ``data/images/<split>/shard_xxx/*.jpg`` (1280x960), one ``metadata.jsonl`` per
shard, and YOLO label files ``data/labels/<split>/shard_xxx/<stem>.txt`` (normalised
``cls cx cy w h``). Two things need care:

* Ultralytics requires an absolute ``path:`` and a ``val:`` key, so this script writes
  ``data/data_abs.yaml`` (``val`` points at ``test`` when the val split is not present).
* The pixel boxes in ``metadata.jsonl`` are NOT in the 1280x960 frame: they are in a 640x640
  frame (measured on 2026-10-31: rescaling them by ``W/640, H/640`` gives mean IoU 0.84 with the
  shipped label files, while using them as-is gives IoU 0). The shipped ``labels/*.txt`` files are
  the correct ones and are what DetectionBench trained and evaluated on. This script therefore
  only generates label files when they are missing, and then rescales from the 640x640 frame.

Usage::

    python scripts/prepare_lisa_hf.py --root data/lisa_hf

License reminder: LISA is CC BY-NC-SA 4.0 (non-commercial). Used here for evaluation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

NAMES = ["go", "goForward", "goLeft", "stop", "stopLeft", "warning", "warningLeft"]
METADATA_FRAME = 640.0  # metadata.jsonl boxes are expressed in a 640x640 frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/lisa_hf"))
    ap.add_argument("--splits", default="train,val,test", help="comma separated splits to process")
    args = ap.parse_args()
    wanted = set(args.splits.split(","))
    data_dir = args.root / "data"
    n_written = 0
    n_existing = 0
    n_images: dict[str, int] = {}
    n_boxes: dict[str, int] = {}
    for split_dir in sorted((data_dir / "images").iterdir()):
        split = split_dir.name
        if split not in wanted:
            continue
        n_images[split] = 0
        n_boxes[split] = 0
        for shard in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            meta = shard / "metadata.jsonl"
            if not meta.exists():
                raise SystemExit(f"missing {meta}; is the download complete?")
            lbl_dir = data_dir / "labels" / split / shard.name
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for line in meta.read_text().splitlines():
                rec = json.loads(line)
                n_images[split] += 1
                boxes = rec["objects"]["bbox"]
                cats = rec["objects"]["categories"]
                n_boxes[split] += len(boxes)
                out = lbl_dir / (Path(rec["file_name"]).stem + ".txt")
                if out.exists():
                    n_existing += 1
                    continue
                with Image.open(shard / rec["file_name"]) as im:
                    w, h = im.size
                lines = []
                for (x, y, bw, bh), c in zip(boxes, cats):
                    sx, sy = w / METADATA_FRAME, h / METADATA_FRAME
                    px, py, pw, ph = x * sx, y * sy, bw * sx, bh * sy
                    lines.append(f"{c} {(px + pw / 2) / w:.6f} {(py + ph / 2) / h:.6f} {pw / w:.6f} {ph / h:.6f}")
                out.write_text("\n".join(lines) + ("\n" if lines else ""))
                n_written += 1
    yaml_lines = [f"path: {data_dir.resolve()}"]
    for split in ("train", "val", "test"):
        if not (data_dir / "images" / split).is_dir():
            continue
        shards = sorted(p for p in (data_dir / "images" / split).iterdir() if p.is_dir())
        yaml_lines.append(f"{split}:")
        for s in shards:
            yaml_lines.append(f"  - images/{split}/{s.name}")
    if not (data_dir / "images" / "val").is_dir():
        # Ultralytics insists on a val key; point it at test when val is not downloaded yet.
        yaml_lines.append("val:")
        for s in sorted(p for p in (data_dir / "images" / "test").iterdir() if p.is_dir()):
            yaml_lines.append(f"  - images/test/{s.name}")
    yaml_lines.append(f"nc: {len(NAMES)}")
    yaml_lines.append("names:")
    for i, n in enumerate(NAMES):
        yaml_lines.append(f"  {i}: {n}")
    (data_dir / "data_abs.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(json.dumps({"labels_written": n_written, "labels_existing": n_existing, "images": n_images, "boxes": n_boxes}))
    print(f"wrote {data_dir / 'data_abs.yaml'}")


if __name__ == "__main__":
    main()
