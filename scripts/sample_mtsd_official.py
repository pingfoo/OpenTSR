#!/usr/bin/env python
"""Sample MTSD images together with their official annotation JSON from a Hugging Face mirror.

The Mapillary Traffic Sign Dataset (CC BY-NC-SA) is distributed by Mapillary behind a form. The
``NgTMDuc/cl_dataset`` mirror carries the official ``mtsd_v2_fully_annotated`` annotation files and
the training images under ``MTSD/``. This script lists the image keys, draws a seeded sample and
downloads ``<key>.json`` plus ``<key>.jpg`` for each key that has an annotation (test-split images
have none and are skipped). The result feeds ``scripts/make_dataset_catalog.py``; it is research
use of a non-commercial dataset and is not redistributed by OpenTSR.

Usage: python scripts/sample_mtsd_official.py --n 320 --out data/mtsd_official
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "OpenTSR/0.1 (research; https://github.com/pingfoo/OpenTSR)"}
REPO = "NgTMDuc/cl_dataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/MTSD/"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main/MTSD/mtsd_fully_annotated_train_images/images?limit=1000"


def list_keys(cache: Path, max_pages: int) -> list[str]:
    if cache.exists():
        return json.loads(cache.read_text())
    keys: list[str] = []
    url = TREE
    for page in range(max_pages):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                body, link = r.read(), r.headers.get("Link", "")
        except Exception as ex:  # noqa: BLE001  (the unauthenticated API is rate limited; keep what we have)
            print("listing stopped:", str(ex)[:80], flush=True)
            break
        keys += [x["path"].split("/")[-1][:-4] for x in json.loads(body) if x["path"].endswith(".jpg")]
        print("page", page, len(keys), flush=True)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not m:
            break
        url = m.group(1)
        time.sleep(0.5)
    cache.write_text(json.dumps(keys))
    return keys


def fetch(url: str, dest: Path, timeout: int) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        dest.write_bytes(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320, help="number of keys to try (about a quarter have no annotation)")
    ap.add_argument("--out", default="data/mtsd_official")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    keys = list_keys(out / "image_keys_all.json", args.max_pages)
    print("image keys listed:", len(keys), flush=True)
    rng = random.Random(args.seed)
    sample = rng.sample(keys, min(args.n, len(keys)))
    done: list[str] = []
    for i, k in enumerate(sample):
        jp, ip = out / "annotations" / f"{k}.json", out / "images" / f"{k}.jpg"
        ok = False
        for attempt in range(4):
            try:
                if not jp.exists():
                    fetch(BASE + f"annotations/mtsd_v2_fully_annotated/annotations/{k}.json", jp, 60)
                if not ip.exists():
                    fetch(BASE + f"mtsd_fully_annotated_train_images/images/{k}.jpg", ip, 120)
                ok = True
                break
            except Exception as ex:  # noqa: BLE001
                for p in (jp, ip):
                    if p.exists() and p.stat().st_size < 100:
                        p.unlink()
                if "404" in str(ex):
                    break  # no annotation for this image (test split)
                print("retry", attempt, k, str(ex)[:60], flush=True)
                time.sleep(15 * (attempt + 1))
        if ok:
            done.append(k)
        if i % 20 == 0:
            print(i, "tried,", len(done), "with annotations", flush=True)
        time.sleep(0.3)
    (out / "keys.json").write_text(json.dumps(sorted(done)))
    print("DONE", len(done), "images with annotations under", out)


if __name__ == "__main__":
    main()
