#!/usr/bin/env python
"""Cut the MLIT 道路標識一覧 (one A3 page PDF) into one PNG per sign, using the label positions.

The Japanese road sign catalogue published by the Ministry of Land, Infrastructure, Transport and
Tourism (https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf) lists every current sign of the
道路標識令 with its number and name, grouped into 案内 / 警戒 / 規制 / 指示 / 補助. The pictograms come
from the ordinance's annexed tables and are therefore outside copyright (著作権法 第 13 条).

This script renders the page at 300 dpi (pdftoppm), reads the label positions (pdftotext -bbox-layout),
assigns each label to a section, crops the sign under it (extending over the label grid for wide or
tall guide signs, trimming neighbours and the thin rules), and writes ``tiles/<number>.png`` plus a
JSON list (section, number, name, name_wiki, png). Pass ``--names-json`` with the list parsed from the
Japanese Wikipedia article to attach cleaner names by number.

Usage: python scripts/cut_mlit_sign_list.py --pdf data/japan/ichiran.pdf --out data/japan/mlit
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pdf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--names-json", default="", help="optional JSON list with section, number, name (Wikipedia parse)")
args = ap.parse_args()

ROOT = Path(args.out)
ROOT.mkdir(parents=True, exist_ok=True)
PDF = Path(args.pdf)
DPI = args.dpi
S = DPI / 72.0
TILES = ROOT / "tiles"
TILES.mkdir(exist_ok=True)
for old in TILES.glob("*.png"):
    old.unlink()

if not (ROOT / "page.png").exists():
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(PDF), str(ROOT / "page")], check=True)
if not (ROOT / "bbox.html").exists():
    subprocess.run(["pdftotext", "-bbox-layout", str(PDF), str(ROOT / "bbox.html")], check=True)
page = Image.open(ROOT / "page.png").convert("RGB")
W, H = page.size

bbox = (ROOT / "bbox.html").read_text()
words = [(float(a), float(b), float(c), float(d), unicodedata.normalize("NFKC", t))
         for a, b, c, d, t in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', bbox)]
headers = {}
for x1, y1, x2, y2, body in re.findall(r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', bbox, re.S):
    t = unicodedata.normalize("NFKC", "".join(re.findall(r"<word[^>]*>(.*?)</word>", body)))
    if t in ("案内標識", "警戒標識", "規制標識", "指示標識", "補助標識"):
        headers[t] = (float(x1), float(y1))
assert len(headers) == 5, headers

# labels: words starting with "(ddd"; split the rare word that holds two numbers
labels = []  # x1, y1, x2, y2, number, name, boxes(list of word boxes to whiten)
for x1, y1, x2, y2, t in words:
    parts = re.findall(r"\(([0-9]{3}[^)]*)\)([^(]*)", t)
    if not parts or not t.startswith("("):
        continue
    n = len(t)
    pos = 0
    for num, name in parts:
        seg = f"({num}){name}"
        start = t.index(seg, pos)
        pos = start + len(seg)
        sx1 = x1 + (x2 - x1) * start / n
        sx2 = x1 + (x2 - x1) * pos / n
        labels.append([sx1, y1, sx2, y2, num.replace("ノ", "の"), name.strip(), [(sx1, y1, sx2, y2)]])

# continuation lines of multi-line names: small-font words just below a label, left aligned with it
small = [w for w in words if (w[3] - w[1]) < 5.6 and not w[4].startswith("(")]
# same-line continuation to the right (pdftotext splits some labels into several words)
for lab in labels:
    for _ in range(3):
        cands = [w for w in small if abs(w[1] - lab[1]) < 1.5 and lab[2] - 0.5 <= w[0] <= lab[2] + 3]
        if not cands:
            break
        w = min(cands, key=lambda w: w[0])
        lab[5] = (lab[5] + w[4]).strip()
        lab[2] = w[2]
        lab[6].append((w[0], w[1], w[2], w[3]))
        small.remove(w)
for lab in labels:
    cur_y2 = lab[3]
    for _ in range(3):
        cands = [w for w in small if cur_y2 - 0.5 < w[1] < cur_y2 + 7 and lab[0] - 2 <= w[0] <= lab[0] + 40]
        if not cands:
            break
        w = min(cands, key=lambda w: w[1])
        lab[5] = (lab[5] + w[4]).strip()
        lab[6].append((w[0], w[1], w[2], w[3]))
        cur_y2 = w[3]
        small.remove(w)

y_bottom_band = min(y for t, (x, y) in headers.items() if t != "案内標識")
cols = sorted((x, t) for t, (x, y) in headers.items() if t != "案内標識")


def section_of(x, y):
    if y < y_bottom_band - 2:
        return "案内標識", (0, W / S)
    for i, (cx, t) in enumerate(cols):
        nx = cols[i + 1][0] if i + 1 < len(cols) else W / S
        if cx - 14 <= x < nx - 14:
            return t, (cx - 14, nx - 14)
    return None, None


# whiten every label word on a copy of the page so that overflowing neighbour labels never enter a tile
page_clean = page.copy()
dc = ImageDraw.Draw(page_clean)
for lab in labels:
    for wx1, wy1, wx2, wy2 in lab[6]:
        dc.rectangle([(wx1 - 1) * S, (wy1 - 1) * S, (wx2 + 1) * S, (wy2 + 1) * S], fill=(255, 255, 255))

rows = {}
for lab in labels:
    sec, xr = section_of(lab[0], lab[1])
    if sec is None:
        continue
    rows.setdefault((sec, xr), []).append(lab)

entries = []
for (sec, xr), labs_all in rows.items():
    # cluster labels into visual rows: a gap of more than 20 pt starts a new row
    labs_all = sorted(labs_all, key=lambda l: l[1])
    ordered = []
    for lab in labs_all:
        if ordered and lab[1] - ordered[-1][0] < 20:
            ordered[-1][1].append(lab)
        else:
            ordered.append([lab[1], [lab]])
    ordered = [(y, labs) for y, labs in ordered]
    ys = [min(l[1] for l in labs) for _, labs in ordered]
    for i, (_, labs) in enumerate(ordered):
        y0 = ys[i] - 1
        y1 = ys[i + 1] - 1.5 if i + 1 < len(ordered) else (y_bottom_band - 8 if sec == "案内標識" else H / S - 14)
        labs = sorted(labs, key=lambda l: l[0])
        for j, lab in enumerate(labs):
            x0 = max(xr[0], lab[0] - 4)
            nxt = [l[0] for l in labs if l[0] > lab[0] + 10]
            x1 = min(nxt) - 4 if nxt else xr[1]
            def occupancy(img):
                a = (255 - np.asarray(img.convert("RGB")).astype(int)).max(axis=2) > 90
                # a column counts as occupied only with more than 8 px of colour: ignores the thin rules between tiles
                return a.sum(axis=0) > 8, a

            def segments(cols, min_gap):
                segs, start, gap = [], None, 0
                for i, v in enumerate(cols):
                    if v:
                        if start is None:
                            start = i
                        gap = 0
                    else:
                        if start is not None:
                            gap += 1
                            if gap >= min_gap:
                                segs.append((start, i - gap + 1))
                                start, gap = None, 0
                if start is not None:
                    segs.append((start, len(cols)))
                return segs

            # extend to the right while the sign continues past the next label (wide guide signs)
            x1e = x1
            cols, _ = occupancy(page_clean.crop((int(x0 * S), int(y0 * S), int(x1e * S), int(y1 * S))))
            tries = 0
            while len(cols) and cols[-1] and x1e < xr[1] and tries < 8:
                x1e = min(xr[1], x1e + 5)
                cols, _ = occupancy(page_clean.crop((int(x0 * S), int(y0 * S), int(x1e * S), int(y1 * S))))
                tries += 1
            # extend downwards while the sign continues past the next row's label line (tall stacked guide signs)
            y1e = y1
            tries = 0
            while tries < 8:
                _, a_ = occupancy(page_clean.crop((int(x0 * S), int(y0 * S), int(x1e * S), int(y1e * S))))
                if not (a_[-4:].sum(axis=1) > 8).any():
                    break
                y1e = min(H / S - 2, y1e + 4)
                tries += 1
            box = (int(x0 * S), int(y0 * S), int(x1e * S), int(y1e * S))
            tile = page_clean.crop(box)
            cols, a = occupancy(tile)
            segs = [sg for sg in segments(cols, int(2.5 * S)) if sg[1] - sg[0] > 3 * S]
            if segs:
                # drop fragments of the right neighbour: segments starting beyond the next label (when there is one)
                limit = (x1 - x0 - 1) * S if nxt else tile.width
                keep = [sg for sg in segs if sg[0] < limit] or segs[:1]
                left, right = keep[0][0], keep[-1][1]
                rows_occ = a[:, left:right].sum(axis=1) > 8
                rsegs = [sg for sg in segments(rows_occ, int(2.5 * S)) if sg[1] - sg[0] > 3 * S]
                # keep the row segments that start above the original row limit (drop the next row's sign)
                rlimit = (y1 - y0 - 1) * S
                rkeep = [sg for sg in rsegs if sg[0] < rlimit] or rsegs[:1]
                top, bottom = (rkeep[0][0], rkeep[-1][1]) if rkeep else (0, tile.height)
                tile = tile.crop((max(0, left - 6), max(0, top - 6), min(tile.width, right + 6), min(tile.height, bottom + 6)))
            num = lab[4]
            base = re.sub(r"[^0-9A-Za-zの-]+", "_", num)
            fn = TILES / f"{base}.png"
            k = 2
            while fn.exists():
                fn = TILES / f"{base}_{k}.png"
                k += 1
            tile.save(fn)
            entries.append({"section": sec, "number": num, "name": lab[5], "png": str(fn), "x": lab[0], "y": lab[1],
                            "w_pt": x1 - x0, "h_pt": y1 - y0})

# names from the Japanese Wikipedia tables when the number matches (cleaner than the split PDF labels)
wiki_names = {}
if args.names_json:
    for e in json.loads(Path(args.names_json).read_text(encoding="utf-8")):
        if e.get("number") and (e["section"], e["number"]) not in wiki_names:
            wiki_names[(e["section"], e["number"])] = e["name"]
for e in entries:
    e["name_wiki"] = wiki_names.get((e["section"], e["number"]), "")

order = ["案内標識", "警戒標識", "規制標識", "指示標識", "補助標識"]
entries.sort(key=lambda e: (order.index(e["section"]), e["y"], e["x"]))
(ROOT / "signs.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
print(collections.Counter(e["section"] for e in entries))
print("DONE", len(entries))
