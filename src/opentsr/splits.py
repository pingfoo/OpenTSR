"""Split manifests shared by every dataset in OpenTSR.

A manifest is a JSON file ``{"meta": {...}, "splits": {"train": [ids], "val": [ids], ...}}``
whose ids are dataset-specific image identifiers of the form ``<group>/<stem>``. The group is
the camera sequence or subset the image belongs to; chunked splits never cut a group's
consecutive frames across splits except at chunk boundaries.

Checked-in manifests are the source of truth for every reported number. ``make_split`` creates
new manifests; it is a re-implementation written after the original generator was lost (it was
never committed), so it does not regenerate the two S2TLD manifests from 2026-10-31 bit for bit.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Protocol


class HasImageId(Protocol):
    image_id: str


def group_of(image_id: str) -> str:
    """Return the group part of a ``<group>/<stem>`` image id."""
    return image_id.split("/", 1)[0]


def _targets(fractions: dict[str, float], n: int) -> dict[str, int]:
    names = list(fractions)
    total = sum(fractions.values())
    counts = {k: round(n * fractions[k] / total) for k in names}
    counts[names[0]] += n - sum(counts.values())  # absorb rounding in the first split
    return counts


def make_split(
    records: Iterable[HasImageId],
    fractions: dict[str, float],
    strategy: str = "chunked",
    seed: int = 0,
    chunk_size: int = 100,
) -> dict[str, list[str]]:
    """Split image ids into named subsets.

    ``random``: shuffle all ids with ``random.Random(seed)`` and cut by ``fractions``.

    ``chunked``: within each group, sort ids and cut them into contiguous chunks of
    ``chunk_size``; shuffle the chunks with ``random.Random(seed)`` and assign each chunk to the
    split furthest below its target size. This keeps near-duplicate consecutive frames on one
    side of the split.
    """
    ids = sorted({r.image_id for r in records})
    if not ids:
        raise ValueError("no records to split")
    rng = random.Random(seed)
    targets = _targets(fractions, len(ids))
    out: dict[str, list[str]] = {k: [] for k in fractions}

    if strategy == "random":
        rng.shuffle(ids)
        start = 0
        for name, count in targets.items():
            out[name] = sorted(ids[start : start + count])
            start += count
        return out

    if strategy != "chunked":
        raise ValueError(f"unknown split strategy {strategy!r}")
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(group_of(i), []).append(i)
    chunks = [
        members[k : k + chunk_size]
        for g in sorted(groups)
        for members in [sorted(groups[g])]
        for k in range(0, len(members), chunk_size)
    ]
    rng.shuffle(chunks)
    for chunk in chunks:
        name = max(targets, key=lambda k: (targets[k] - len(out[k])) / max(targets[k], 1))
        out[name].extend(chunk)
    return {k: sorted(v) for k, v in out.items()}


def read_manifest(path: Path | str) -> dict[str, list[str]]:
    """Return ``{split name: [image ids]}`` from a manifest file."""
    with open(path) as f:
        data = json.load(f)
    return {k: list(v) for k, v in data["splits"].items()}


def write_manifest(path: Path | str, split: dict[str, list[str]], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "splits": {k: sorted(v) for k, v in split.items()}}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
