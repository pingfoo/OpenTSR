#!/usr/bin/env python
"""Parse the Japanese Wikipedia article 日本の道路標識 into a list of (section, number, name, file).

The article's galleries list every current sign with its 道路標識令 number and name and the Commons
file name of its pictogram. The list is used to attach clean names to the tiles cut from the MLIT
sign catalogue (scripts/cut_mlit_sign_list.py). Fetches the wikitext with ``action=raw`` unless
``--wikitext`` points to a saved copy.

Usage: python scripts/parse_ja_wikipedia_signs.py --out data/japan/wikipedia_signs.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

SECTIONS = ["案内標識", "警戒標識", "規制標識", "指示標識", "補助標識"]
URL = "https://ja.wikipedia.org/w/index.php?title=" + urllib.parse.quote("日本の道路標識") + "&action=raw"
UA = {"User-Agent": "OpenTSR/0.1 (research; https://github.com/pingfoo/OpenTSR)"}


def clean(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s)
    s = re.sub(r"\{\{efn\|.*?\}\}", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"'''?", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse(text: str) -> list[dict]:
    section = None
    subsection = ""
    in_gallery = False
    entries: list[dict] = []
    for ln in text.splitlines():
        m = re.match(r"^(={2,4})\s*(.+?)\s*\1\s*$", ln)
        if m:
            title = m.group(2).strip()
            if title in SECTIONS:
                section, subsection = title, ""
            elif title in ("廃止済", "法定外標識") or title.startswith("1"):
                section = None
            continue
        if section and ln.startswith("'''") and ln.endswith("'''"):
            subsection = clean(ln)
            continue
        if ln.startswith("<gallery"):
            in_gallery = True
            continue
        if ln.startswith("</gallery>"):
            in_gallery = False
            continue
        if in_gallery and section and "|" in ln:
            f, cap = ln.split("|", 1)
            f = re.sub(r"^(ファイル|File|画像|Image):", "", f.strip()).strip()
            if subsection == "ギャラリー" or not f.lower().endswith(".svg"):
                continue
            cap = clean(cap).replace("（", "(").replace("）", ")")
            mm = re.search(r"\(([0-9]{3}(?:の[0-9]+)?(?:-[A-Z0-9]+)*)\)", cap)
            num = mm.group(1) if mm else None
            name = re.sub(r"\s*\([0-9][^)]*\)\s*", " ", cap).strip()
            entries.append({"section": section, "subsection": subsection, "file": f, "caption": cap, "number": num, "name": name})
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wikitext", default="", help="saved wikitext instead of fetching")
    args = ap.parse_args()
    if args.wikitext:
        text = Path(args.wikitext).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=60) as r:
            text = r.read().decode("utf-8")
    entries = parse(text)
    Path(args.out).write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(entries), "entries:", dict(collections.Counter(e["section"] for e in entries)))


if __name__ == "__main__":
    main()
