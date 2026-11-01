"""Stage interface, registry and the reference (model-free) stage implementations.

The pipeline is ``frame -> detector -> tracker -> memory -> classifier -> reader -> relevance
-> fusion -> decision -> defense`` (``docs/system_design.md`` section 2). A stage is an object
with ``reset()`` (start of a sequence) and ``__call__(ctx)`` that reads and mutates a
``FrameContext`` in place. A stage configured as ``null`` becomes ``PassThrough`` and the
``FrameContext`` flows through unchanged, which is what "switching a stage off" means in the
ablation driver.

The implementations here are the ones that need no learned model: precomputed detections and
classifications (so that any external detector or classifier can be evaluated at L3 by dumping
its per-frame outputs), an IoU tracker, a TTL memory, a geometric relevance rule, and the two
decision rules (per-frame argmax and vote over the track). Model-backed stages (ONNX detector,
crop classifier, OCR reader, OSM fusion, defenses) register themselves under the same names
when they are added, so configurations do not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from opentsr.metrics import iou_xyxy
from opentsr.signs.pipeline.schema import Box, Candidate, FrameMeta, Message, SignState, box_long_side

STAGE_ORDER = ["detector", "tracker", "memory", "classifier", "reader", "relevance", "fusion", "decision", "defense"]


@dataclass
class FrameContext:
    meta: FrameMeta
    image: Any = None  # numpy array or PIL image when a model-backed stage needs pixels; None in replay
    candidates: list[Candidate] = field(default_factory=list)
    signs: list[SignState] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


class Stage:
    """Base class. Subclasses set ``stage`` (the slot name) and ``name`` (the option name)."""

    stage: str = ""
    name: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def reset(self) -> None:  # start of a new sequence
        pass

    def __call__(self, ctx: FrameContext) -> None:
        raise NotImplementedError


class PassThrough(Stage):
    name = "null"

    def __call__(self, ctx: FrameContext) -> None:
        pass


_REGISTRY: dict[tuple[str, str], type[Stage]] = {}


def register(cls: type[Stage]) -> type[Stage]:
    assert cls.stage in STAGE_ORDER, cls
    _REGISTRY[(cls.stage, cls.name)] = cls
    return cls


def available(stage: str) -> list[str]:
    return sorted(n for (s, n) in _REGISTRY if s == stage)


def build_stage(stage: str, cfg: dict[str, Any] | None) -> Stage:
    if stage not in STAGE_ORDER:
        raise KeyError(f"unknown stage slot {stage!r}; slots are {STAGE_ORDER}")
    if cfg is None:
        return PassThrough()
    cfg = dict(cfg)
    try:
        kind = cfg.pop("type")
    except KeyError as e:
        raise KeyError(f"stage {stage!r}: configuration needs a 'type' key or null") from e
    try:
        cls = _REGISTRY[(stage, kind)]
    except KeyError as e:
        raise KeyError(f"stage {stage!r}: unknown type {kind!r}; available: {available(stage)}") from e
    return cls(**cfg)


# ----------------------------------------------------------------------------------------------
# Detector


@register
class PrecomputedDetector(Stage):
    """Candidates come from a mapping ``frame_id -> list[Candidate]`` (an external detector's dump)."""

    stage, name = "detector", "precomputed"

    def __init__(self, detections: dict[str, list[Candidate]] | None = None, min_score: float = 0.0, **params: Any):
        super().__init__(min_score=min_score, **params)
        self.detections = detections or {}
        self.min_score = min_score

    def __call__(self, ctx: FrameContext) -> None:
        ctx.candidates = [
            Candidate(**{**vars(c), "track_id": None}) for c in self.detections.get(ctx.meta.frame_id, [])
            if c.score >= self.min_score
        ]


# ----------------------------------------------------------------------------------------------
# Tracker


@register
class IoUTracker(Stage):
    """Greedy IoU association against the last box of each live track.

    Candidates that match no track start a new one. A track is dropped from the live set after
    ``max_gap`` consecutive frames without a match (memory, not the tracker, decides what happens
    to the sign afterwards). ``confirm_frames`` is the minimum number of matched frames before a
    track is passed on as a sign; shorter tracks are one-frame noise.
    """

    stage, name = "tracker", "iou"

    def __init__(self, iou_thr: float = 0.3, max_gap: int = 2, confirm_frames: int = 2, **params: Any):
        super().__init__(iou_thr=iou_thr, max_gap=max_gap, confirm_frames=confirm_frames, **params)
        self.iou_thr, self.max_gap, self.confirm_frames = iou_thr, max_gap, confirm_frames
        self.reset()

    def reset(self) -> None:
        self.next_id = 1
        self.live: dict[int, dict[str, Any]] = {}  # track_id -> {box, last_index, n}

    def __call__(self, ctx: FrameContext) -> None:
        idx = ctx.meta.index
        tracks = list(self.live.items())
        assigned: dict[int, int] = {}  # candidate position -> track id
        if tracks and ctx.candidates:
            ious = iou_xyxy(np.array([c.box for c in ctx.candidates]), np.array([t["box"] for _, t in tracks]))
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            used_t: set[int] = set()
            for ci, tj in order:
                if ious[ci, tj] < self.iou_thr:
                    break
                if ci in assigned or tj in used_t:
                    continue
                assigned[int(ci)] = tracks[tj][0]
                used_t.add(int(tj))
        for ci, cand in enumerate(ctx.candidates):
            tid = assigned.get(ci)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.live[tid] = {"box": cand.box, "last_index": idx, "n": 0}
            t = self.live[tid]
            t.update(box=cand.box, last_index=idx, n=t["n"] + 1)
            cand.track_id = tid
            cand.extra["track_len"] = t["n"]
        for tid in [tid for tid, t in self.live.items() if idx - t["last_index"] > self.max_gap]:
            del self.live[tid]
        ctx.candidates = [c for c in ctx.candidates if c.extra.get("track_len", 1) >= self.confirm_frames]


@register
class MotionTracker(IoUTracker):
    """Association with the sparse-frame rule of ``opentsr.signs.linking.link_score`` (IoU, else
    linear extrapolation of centre and size), for frame rates where IoU alone breaks. Same
    ``max_gap`` and ``confirm_frames`` semantics as the IoU tracker; ``same_class`` requires the
    detector's coarse class to agree."""

    stage, name = "tracker", "motion"

    def __init__(self, iou_thr: float = 0.3, center_factor: float = 1.5, first_factor: float = 5.0,
                 scale_ratio: float = 2.0, same_class: bool = True, max_gap: int = 2, confirm_frames: int = 2,
                 **params: Any):
        super().__init__(iou_thr=iou_thr, max_gap=max_gap, confirm_frames=confirm_frames, center_factor=center_factor,
                         first_factor=first_factor, scale_ratio=scale_ratio, same_class=same_class, **params)
        self.center_factor, self.first_factor, self.scale_ratio, self.same_class = (
            center_factor, first_factor, scale_ratio, same_class)

    def reset(self) -> None:
        super().reset()
        self.history: dict[int, list[Box]] = {}

    def __call__(self, ctx: FrameContext) -> None:
        from opentsr.signs.linking import link_score  # local import: linking imports the schema only

        idx = ctx.meta.index
        pairs = []
        for ci, cand in enumerate(ctx.candidates):
            for tid, t in self.live.items():
                if self.same_class and t.get("coarse") not in (None, cand.coarse):
                    continue
                s = link_score(self.history[tid], cand.box, self.iou_thr, self.center_factor, self.first_factor,
                               self.scale_ratio)
                if s > 0:
                    pairs.append((-s, tid, ci))
        assigned: dict[int, int] = {}
        used_t: set[int] = set()
        for _, tid, ci in sorted(pairs):
            if ci in assigned or tid in used_t:
                continue
            assigned[ci] = tid
            used_t.add(tid)
        for ci, cand in enumerate(ctx.candidates):
            tid = assigned.get(ci)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.live[tid] = {"box": cand.box, "last_index": idx, "n": 0, "coarse": cand.coarse}
                self.history[tid] = []
            t = self.live[tid]
            t.update(box=cand.box, last_index=idx, n=t["n"] + 1)
            self.history[tid].append(cand.box)
            cand.track_id = tid
            cand.extra["track_len"] = t["n"]
        for tid in [tid for tid, t in self.live.items() if idx - t["last_index"] > self.max_gap]:
            del self.live[tid]
            del self.history[tid]
        ctx.candidates = [c for c in ctx.candidates if c.extra.get("track_len", 1) >= self.confirm_frames]


# ----------------------------------------------------------------------------------------------
# Memory


@register
class TTLMemory(Stage):
    """Keep every tracked sign for ``ttl_s`` seconds after it was last seen.

    With the tracker off, candidates have no track id and each one becomes its own sign for the
    current frame only (so the stage is a no-op in the academic configuration).
    """

    stage, name = "memory", "ttl"

    def __init__(self, ttl_s: float = 3.0, **params: Any):
        super().__init__(ttl_s=ttl_s, **params)
        self.ttl_s = ttl_s
        self.reset()

    def reset(self) -> None:
        self.signs: dict[int, SignState] = {}
        self.last_seen_s: dict[int, float] = {}
        self._anon = 0

    def __call__(self, ctx: FrameContext) -> None:
        idx, t = ctx.meta.index, ctx.meta.time_s
        seen: set[int] = set()
        for c in ctx.candidates:
            tid = c.track_id
            if tid is None:
                self._anon -= 1
                tid = self._anon  # negative ids: untracked, this frame only
            s = self.signs.get(tid)
            if s is None:
                s = SignState(track_id=tid, box=c.box, coarse=c.coarse, first_index=idx, last_index=idx, n_frames=0)
                self.signs[tid] = s
            s.box, s.coarse, s.last_index, s.n_frames, s.visible = c.box, c.coarse, idx, s.n_frames + 1, True
            s.history.append((c.fine, c.score))
            if c.fine is not None:
                s.fine, s.fine_score = c.fine, c.score
            s.extra["candidate"] = c
            self.last_seen_s[tid] = t
            seen.add(tid)
        for tid in list(self.signs):
            if tid not in seen:
                self.signs[tid].visible = False
                self.signs[tid].extra.pop("candidate", None)
                if tid < 0 or t - self.last_seen_s[tid] > self.ttl_s:
                    del self.signs[tid]
                    del self.last_seen_s[tid]
        ctx.signs = list(self.signs.values())


# ----------------------------------------------------------------------------------------------
# Classifier


@register
class PrecomputedClassifier(Stage):
    """Fine class per candidate from ``candidate.extra["fine"]`` / ``["fine_score"]`` (an external dump)."""

    stage, name = "classifier", "precomputed"

    def __init__(self, reject_below: float = 0.0, **params: Any):
        super().__init__(reject_below=reject_below, **params)
        self.reject_below = reject_below

    def __call__(self, ctx: FrameContext) -> None:
        for s in ctx.signs:
            c = s.extra.get("candidate")
            if c is None:
                continue
            fine, score = c.extra.get("fine"), float(c.extra.get("fine_score", c.score))
            if fine is None or score < self.reject_below:
                continue
            if s.history:
                s.history[-1] = (fine, score)
            s.fine, s.fine_score = fine, score


@register
class FunctionClassifier(Stage):
    """Calls ``fn(image, box) -> (fine, score)``; the hook model-backed classifiers plug into."""

    stage, name = "classifier", "function"

    def __init__(self, fn: Callable[[Any, Box], tuple[str, float]] | None = None, reject_below: float = 0.0, **params: Any):
        super().__init__(reject_below=reject_below, **params)
        self.fn, self.reject_below = fn, reject_below

    def __call__(self, ctx: FrameContext) -> None:
        if self.fn is None:
            return
        for s in ctx.signs:
            if not s.visible:
                continue
            fine, score = self.fn(ctx.image, s.box)
            if score < self.reject_below:
                continue
            if s.history:
                s.history[-1] = (fine, score)
            s.fine, s.fine_score = fine, score


# ----------------------------------------------------------------------------------------------
# Relevance


@register
class GeometricRelevance(Stage):
    """A sign is relevant unless its centre is more than ``side_fraction`` of the half-width away from
    the image centre line. Needs ``meta.width``; without it every sign stays relevant."""

    stage, name = "relevance", "geometric"

    def __init__(self, side_fraction: float = 0.8, **params: Any):
        super().__init__(side_fraction=side_fraction, **params)
        self.side_fraction = side_fraction

    def __call__(self, ctx: FrameContext) -> None:
        w = ctx.meta.width
        for s in ctx.signs:
            if w is None:
                s.relevant = True
                continue
            cx = (s.box[0] + s.box[2]) / 2
            s.relevant = abs(cx - w / 2) <= self.side_fraction * w / 2


# ----------------------------------------------------------------------------------------------
# Decision


def _emit(ctx: FrameContext, s: SignState, fine: str | None, conf: float, state: str) -> None:
    ctx.messages.append(Message(index=ctx.meta.index, time_s=ctx.meta.time_s, track_id=s.track_id, fine=fine,
                                coarse=s.coarse, confidence=conf, value=s.value, state=state, relevant=s.relevant))


@register
class PerFrameArgmax(Stage):
    """The academic baseline: every visible sign emits its current fine class every frame."""

    stage, name = "decision", "per_frame_argmax"

    def __call__(self, ctx: FrameContext) -> None:
        for s in ctx.signs:
            if s.visible and s.relevant is not False:
                _emit(ctx, s, s.fine or s.coarse, s.fine_score, "new" if s.n_frames == 1 else "update")


@register
class VoteDecision(Stage):
    """Confidence-weighted vote over the track's history; emit after ``min_frames`` and change the
    emitted class only when the challenger leads by ``hysteresis`` (fraction of total weight)."""

    stage, name = "decision", "vote"

    def __init__(self, min_frames: int = 2, hysteresis: float = 0.2, weighted: bool = True, **params: Any):
        super().__init__(min_frames=min_frames, hysteresis=hysteresis, weighted=weighted, **params)
        self.min_frames, self.hysteresis, self.weighted = min_frames, hysteresis, weighted
        self.reset()

    def reset(self) -> None:
        self.emitted: dict[int, str] = {}

    def __call__(self, ctx: FrameContext) -> None:
        live = set()
        for s in ctx.signs:
            if s.relevant is False or s.n_frames < self.min_frames:
                continue
            live.add(s.track_id)
            weights: dict[str, float] = {}
            for fine, score in s.history:
                if fine is None:
                    continue
                weights[fine] = weights.get(fine, 0.0) + (score if self.weighted else 1.0)
            if not weights:  # no fine class anywhere on the track: emit the coarse class
                current = self.emitted.get(s.track_id)
                self.emitted[s.track_id] = s.coarse
                _emit(ctx, s, None, float(np.mean([sc for _, sc in s.history])) if s.history else 0.0,
                      "new" if current is None else "keep")
                continue
            total = sum(weights.values())
            best = max(weights, key=weights.get)  # type: ignore[arg-type]
            current = self.emitted.get(s.track_id)
            if current is None:
                self.emitted[s.track_id] = best
                _emit(ctx, s, best, weights[best] / total, "new")
            elif best != current and (weights[best] - weights.get(current, 0.0)) / total > self.hysteresis:
                self.emitted[s.track_id] = best
                _emit(ctx, s, best, weights[best] / total, "update")
            else:
                _emit(ctx, s, self.emitted[s.track_id], weights.get(self.emitted[s.track_id], 0.0) / total, "keep")
        for tid in [t for t in self.emitted if t not in live and t not in {s.track_id for s in ctx.signs}]:
            del self.emitted[tid]


__all__ = ["STAGE_ORDER", "FrameContext", "Stage", "PassThrough", "register", "build_stage", "available",
           "box_long_side"]
