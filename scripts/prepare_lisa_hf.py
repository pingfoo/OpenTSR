#!/usr/bin/env python
"""Make the Hugging Face LISA export (dronefreak/LISA-Traffic-Lights) usable with Ultralytics.

The export is an HF image-folder: ``data/images/<split>/shard_xxx/*.jpg`` plus one
``metadata.jsonl`` per shard with ``{"file_name", "objects": {"bbox": [[x, y, w, h], ...],
"categories": [...]}}`` in pixels. Ultralytics wants ``labels/<split>/shard_xxx/<stem>.txt``
with normalised ``cls cx cy w h``. This script writes those label files (skipping any that
already exist) and a ``data_abs.yaml`` with an absolute ``path`` so ``yolo val`` works from any
directory. Image sizes are read from the JPEG headers.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/lisa_hf"))
    args = ap.parse_args()
    data_dir = args.root / "data"
    n_written = 0
    n_images = {}
    n_boxes = {}
    for split_dir in sorted((data_dir / "images").iterdir()):
        split = split_dir.name
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
                img_path = shard / rec["file_name"]
                n_images[split] += 1
                boxes = rec["objects"]["bbox"]
                cats = rec["objects"]["categories"]
                n_boxes[split] += len(boxes)
                out = lbl_dir / (Path(rec["file_name"]).stem + ".txt")
                if out.exists():
                    continue
                if boxes:
                    with Image.open(img_path) as im:
                        w, h = im.size
                    lines = []
                    for (x, y, bw, bh), c in zip(boxes, cats):
                        lines.append(f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                    out.write_text("\n".join(lines) + "\n")
                else:
                    out.write_text("")
                n_written += 1
    yaml_lines = [f"path: {data_dir.resolve()}"]
    for split in ("train", "val", "test"):
        shards = sorted(p for p in (data_dir / "images" / split).iterdir() if p.is_dir())
        yaml_lines.append(f"{split}:")
        for s in shards:
            yaml_lines.append(f"  - images/{split}/{s.name}")
    yaml_lines.append(f"nc: {len(NAMES)}")
    yaml_lines.append("names:")
    for i, n in enumerate(NAMES):
        yaml_lines.append(f"  {i}: {n}")
    (data_dir / "data_abs.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(json.dumps({"labels_written": n_written, "images": n_images, "boxes": n_boxes}))
    print(f"wrote {data_dir / 'data_abs.yaml'}")


if __name__ == "__main__":
    main()
