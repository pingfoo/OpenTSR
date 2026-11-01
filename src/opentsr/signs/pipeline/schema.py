"""Data records exchanged between pipeline stages.

Every record is a plain dataclass that round-trips through ``to_dict`` / ``from_dict`` so that
the per-frame JSON Lines log (``runner.py``) can be replayed into any later stage or analysis
without rerunning the models. Boxes are ``(x1, y1, x2, y2)`` in pixels of the source frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

Box = tuple[float, float, float, float]


def box_long_side(box: Box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


@dataclass
class FrameMeta:
    """What the pipeline knows about a frame before looking at it."""

    frame_id: str  # unique within the sequence, for example "1277381619Image000001"
    index: int  # position in the sequence (0-based, consecutive even if frames are subsampled)
    time_s: float  # seconds since the start of the sequence
    sequence_id: str = ""
    distance_m: float | None = None  # odometer, when known (ISA distance-based metrics)
    speed_mps: float | None = None
    gnss: tuple[float, float] | None = None  # (lat, lon)
    width: int | None = None
    height: int | None = None


@dataclass
class Candidate:
    """Detector output: one box with a coarse class (or ``"sign"`` for class-agnostic detectors)."""

    box: Box
    score: float
    coarse: str = "sign"
    fine: str | None = None  # fine-class detectors fill this directly
    track_id: int | None = None  # filled by the tracker
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignState:
    """One tracked sign as the system currently believes it: the unit the decision stage votes on."""

    track_id: int
    box: Box
    coarse: str
    fine: str | None = None
    fine_score: float = 0.0
    value: str | None = None  # reader output, for example "50" for a speed limit
    relevant: bool | None = None  # None means the relevance stage is off
    first_index: int = 0
    last_index: int = 0
    n_frames: int = 1
    visible: bool = True  # False once memory keeps it after it left the field of view
    history: list[tuple[str | None, float]] = field(default_factory=list)  # (fine, score) per frame
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """What the system says to the driver / downstream: the stream that L3 evaluates."""

    index: int
    time_s: float
    track_id: int
    fine: str | None
    coarse: str
    confidence: float
    value: str | None = None
    state: str = "new"  # new, update, keep, end
    relevant: bool | None = None


@dataclass
class FrameRecord:
    """Everything produced for one frame, written as one JSON line."""

    meta: FrameMeta
    candidates: list[Candidate] = field(default_factory=list)
    signs: list[SignState] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    stage_times_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameRecord":
        meta = FrameMeta(**{**d["meta"], "gnss": tuple(d["meta"]["gnss"]) if d["meta"].get("gnss") else None})
        cands = [Candidate(**{**c, "box": tuple(c["box"])}) for c in d.get("candidates", [])]
        signs = [
            SignState(**{**s, "box": tuple(s["box"]), "history": [tuple(h) for h in s.get("history", [])]})
            for s in d.get("signs", [])
        ]
        msgs = [Message(**m) for m in d.get("messages", [])]
        return cls(meta=meta, candidates=cands, signs=signs, messages=msgs, stage_times_ms=d.get("stage_times_ms", {}))
