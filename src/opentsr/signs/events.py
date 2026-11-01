"""Sign-pass events: the unit of the L3 benchmark (``docs/benchmark.md`` section 2, L3).

An event is one physical sign instance passed by the ego vehicle. Sequence datasets give
per-frame boxes; some give track ids (ZOD), most do not (STS, CURE-TSD). ``link_tracks`` turns
per-frame boxes into tracks with a deterministic proximity linker, ``events_from_tracks`` turns
tracks into events, and ``events_to_json`` / ``events_from_json`` are the on-disk format that the
evaluator reads. The linker is intentionally simple and fully specified here so that anyone can
regenerate the event files from the original annotations.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


from opentsr.signs.linking import link_score
from opentsr.signs.pipeline.schema import Box, box_long_side


@dataclass
class GTBox:
    """One annotated sign in one frame."""

    frame_index: int
    box: Box
    fine: str
    coarse: str = "sign"
    relevant: bool | None = None  # None when the dataset has no relevance attribute
    status: str = ""  # dataset's own visibility flag (STS: VISIBLE, BLURRED, OCCLUDED, SIDE_ROAD)
    track_id: int | None = None


@dataclass
class Track:
    track_id: int
    boxes: list[GTBox] = field(default_factory=list)

    @property
    def first_index(self) -> int:
        return self.boxes[0].frame_index

    @property
    def last_index(self) -> int:
        return self.boxes[-1].frame_index


@dataclass
class Event:
    event_id: str
    sequence_id: str
    fine: str
    coarse: str
    relevant: bool | None
    first_index: int  # first frame in which the sign is at least ``min_px`` long
    pass_index: int  # last annotated frame in which the sign is visible
    first_time_s: float
    pass_time_s: float
    track_id: int
    n_frames: int
    boxes: dict[int, Box] = field(default_factory=dict)  # frame_index -> box, for matching system output
    limit_after: str | None = None  # speed limit in force after the pass, when annotated

    def to_dict(self) -> dict:
        d = asdict(self)
        d["boxes"] = {str(k): [round(float(x), 1) for x in v] for k, v in self.boxes.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        d = dict(d)
        d["boxes"] = {int(k): tuple(v) for k, v in d.get("boxes", {}).items()}
        return cls(**d)


def link_tracks(
    boxes: Iterable[GTBox],
    iou_thr: float = 0.3,
    max_gap: int = 1,
    center_factor: float = 1.5,
    first_factor: float = 5.0,
    scale_ratio: float = 2.0,
    same_class: bool = True,
) -> list[Track]:
    """Deterministic greedy linker over annotated frames (see ``linking.link_score`` for the rule).

    Frames are visited in index order; each box is matched to the live track with the best
    score (ties broken by track id). A track stays live for ``max_gap`` annotated frames without
    a match. With ``same_class`` the fine labels must agree, except that an unreadable / other
    label (``"URDBL"``, ``"OTHER"``) may join a track of any class. Boxes that already carry a
    ``track_id`` are grouped by it and never relinked.
    """
    by_frame: dict[int, list[GTBox]] = {}
    pre: dict[int, Track] = {}
    for b in boxes:
        if b.track_id is not None:
            pre.setdefault(b.track_id, Track(b.track_id)).boxes.append(b)
        else:
            by_frame.setdefault(b.frame_index, []).append(b)
    for t in pre.values():
        t.boxes.sort(key=lambda x: x.frame_index)
    tracks: list[Track] = []
    live: list[Track] = []
    next_id = (max(pre) + 1) if pre else 1
    wildcard = {"URDBL", "OTHER", "UNREADABLE"}
    for idx in sorted(by_frame):
        frame_boxes = sorted(by_frame[idx], key=lambda b: (b.box[0], b.box[1]))
        live = [t for t in live if idx - t.last_index <= max_gap]
        pairs = []
        for bi, b in enumerate(frame_boxes):
            for t in live:
                last = t.boxes[-1]
                if same_class and b.fine != last.fine and b.fine not in wildcard and last.fine not in wildcard:
                    continue
                s = link_score([x.box for x in t.boxes], b.box, iou_thr, center_factor, first_factor, scale_ratio)
                if s > 0:
                    pairs.append((-s, t.track_id, bi))
        used_b: set[int] = set()
        used_t: set[int] = set()
        for _, tid, bi in sorted(pairs):
            if bi in used_b or tid in used_t:
                continue
            t = next(t for t in live if t.track_id == tid)
            t.boxes.append(frame_boxes[bi])
            used_b.add(bi)
            used_t.add(tid)
        for bi, b in enumerate(frame_boxes):
            if bi not in used_b:
                t = Track(next_id, [b])
                next_id += 1
                tracks.append(t)
                live.append(t)
    all_tracks = sorted(pre.values(), key=lambda t: t.track_id) + tracks
    for t in all_tracks:
        for b in t.boxes:
            b.track_id = t.track_id
    return all_tracks


def _majority(values: list[str], ignore: set[str]) -> str:
    c = Counter(v for v in values if v not in ignore) or Counter(values)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def events_from_tracks(
    tracks: Iterable[Track],
    sequence_id: str,
    time_of_index: dict[int, float],
    min_px: float = 12.0,
    min_frames: int = 1,
) -> list[Event]:
    """One event per track. Fine and coarse class are the majority over the track (ignoring
    unreadable labels); relevance is the majority of the per-frame flags when present."""
    events: list[Event] = []
    for t in sorted(tracks, key=lambda t: (t.first_index, t.track_id)):
        if len(t.boxes) < min_frames:
            continue
        big = [b for b in t.boxes if box_long_side(b.box) >= min_px]
        if not big:
            continue
        fine = _majority([b.fine for b in t.boxes], {"URDBL", "UNREADABLE"})
        coarse = _majority([b.coarse for b in t.boxes], {"UNREADABLE"})
        flags = [b.relevant for b in t.boxes if b.relevant is not None]
        relevant = (sum(flags) * 2 >= len(flags)) if flags else None
        events.append(Event(
            event_id=f"{sequence_id}/{t.track_id}", sequence_id=sequence_id, fine=fine, coarse=coarse,
            relevant=relevant, first_index=big[0].frame_index, pass_index=t.last_index,
            first_time_s=time_of_index[big[0].frame_index], pass_time_s=time_of_index[t.last_index],
            track_id=t.track_id, n_frames=len(t.boxes), boxes={b.frame_index: b.box for b in t.boxes},
        ))
    return events


def events_to_json(path: str | Path, events: list[Event], meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta or {}, "events": [e.to_dict() for e in events]}, f, separators=(",", ":"))


def events_from_json(path: str | Path) -> list[Event]:
    with open(path) as f:
        return [Event.from_dict(d) for d in json.load(f)["events"]]
