"""Guards for the repository layout and the shared split utilities.

The layout rule (docs/decisions.md, 2026-10-31): traffic light material lives only under a
``traffic_lights`` path segment, and shared code and the sign track never import it.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opentsr.splits import make_split, read_manifest, write_manifest  # noqa: E402

LIGHT_WORDS = re.compile(r"traffic[_ -]?light|s2tld|bstld|lisa|dtld|autoware|\btlr\b", re.IGNORECASE)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_traffic_light_files_stay_under_traffic_lights():
    allowed_outside = {"README.md", "CLAUDE.md", "Makefile", ".gitignore", "pyproject.toml",
                       "docs/decisions.md", "docs/third_party.md", "docs/README.md",
                       "tests/test_layout.py", "src/opentsr/__init__.py"}
    offenders = [
        f for f in tracked_files()
        if "traffic_lights" not in f.split("/") and f not in allowed_outside and LIGHT_WORDS.search(f)
    ]
    assert offenders == []


def test_shared_and_sign_code_do_not_import_traffic_lights():
    pkg = ROOT / "src" / "opentsr"
    for py in pkg.rglob("*.py"):
        if "traffic_lights" in py.relative_to(pkg).parts:
            continue
        assert not re.search(r"(?m)^\s*(from|import)\s+opentsr\.traffic_lights", py.read_text()), py


@dataclass
class Rec:
    image_id: str


def _records():
    # two sequences of 250 and 130 frames
    return [Rec(f"a/{i:04d}") for i in range(250)] + [Rec(f"b/{i:04d}") for i in range(130)]


def test_chunked_split_keeps_chunks_together_and_covers_everything():
    split = make_split(_records(), {"train": 0.6, "val": 0.1, "test": 0.3}, "chunked", seed=0, chunk_size=50)
    all_ids = sorted(i for v in split.values() for i in v)
    assert all_ids == sorted(r.image_id for r in _records())
    owner = {i: name for name, ids in split.items() for i in ids}
    for group, n in (("a", 250), ("b", 130)):
        for start in range(0, n, 50):
            chunk = {owner[f"{group}/{i:04d}"] for i in range(start, min(start + 50, n))}
            assert len(chunk) == 1


def test_random_split_sizes_and_determinism():
    a = make_split(_records(), {"train": 0.6, "test": 0.4}, "random", seed=0)
    b = make_split(_records(), {"train": 0.6, "test": 0.4}, "random", seed=0)
    assert a == b
    assert len(a["train"]) == 228 and len(a["test"]) == 152


def test_manifest_round_trip(tmp_path):
    split = {"train": ["a/2", "a/1"], "test": ["b/1"]}
    p = tmp_path / "m.json"
    write_manifest(p, split, {"dataset": "toy"})
    assert read_manifest(p) == {"train": ["a/1", "a/2"], "test": ["b/1"]}
    assert json.loads(p.read_text())["meta"] == {"dataset": "toy"}


def test_checked_in_s2tld_manifests_are_readable():
    base = ROOT / "traffic_lights" / "configs" / "s2tld" / "splits"
    sizes = {name: {k: len(v) for k, v in read_manifest(base / name).items()}
             for name in ("chunked_60_10_30_seed0.json", "random60_seed0.json")}
    assert sizes["chunked_60_10_30_seed0.json"] == {"test": 1800, "train": 3386, "val": 600}
    assert sizes["random60_seed0.json"] == {"test": 2314, "train": 3472}


def test_every_sign_doc_has_a_japanese_translation():
    # CLAUDE.md: every Markdown file in docs/ has a translation under docs/ja/ with the same name.
    docs = ROOT / "docs"
    english = sorted(p.name for p in docs.glob("*.md"))
    japanese = sorted(p.name for p in (docs / "ja").glob("*.md"))
    assert english == japanese
