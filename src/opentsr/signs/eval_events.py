"""L3 evaluator: score a system's message stream against sign-pass events.

Definitions follow ``docs/benchmark.md`` section 2 (L3 metrics). The inputs are the
``FrameRecord`` list a pipeline produced for one sequence (``opentsr.signs.pipeline``) and the
events of that sequence (``opentsr.signs.events``). Messages carry a track id; the box of that
track in the frame comes from ``FrameRecord.signs``, and a message is attributed to the event
whose annotated box in the same frame (or the nearest annotated frame within ``match_gap``
frames, for datasets that label every n-th frame) overlaps it with IoU at least ``iou_thr``.

Per event:

- hit: at least one attributed message with the correct fine class between the event's first
  frame and ``window_after_pass_s`` after the pass, and, from the first correct message on, no
  attributed message with a different class up to the pass (the ISA window and stability rule).
- latency_first_s: time of the first correct message minus ``first_time_s``.
- latency_final_s: time of the last class change minus ``pass_time_s`` (negative means the
  final answer was available before the pass).
- flips: number of class changes in the attributed message sequence, after reducing the
  messages of one frame to the single highest-confidence one (what a display would show).

Per sequence: event recall (relevant events only, and all events), event precision over
alerts (messages with state ``new`` or ``update``), false alerts split into relevance errors
(attributed to a non-relevant event) and phantoms (attributed to nothing), per km when the
frames carry ``distance_m``, and coarse-class versions of recall and precision.

``speed_limit_accuracy`` is the distance-based (or time-based when no odometer exists) share
of the route on which the system's current limit equals the limit in force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from opentsr.metrics import iou_xyxy
from opentsr.signs.events import Event
from opentsr.signs.pipeline.schema import FrameRecord, Message


@dataclass
class EventResult:
    event_id: str
    fine: str
    relevant: bool | None
    hit: bool
    coarse_hit: bool
    n_messages: int
    flips: int
    latency_first_s: float | None
    latency_final_s: float | None
    classes_emitted: list[str] = field(default_factory=list)


@dataclass
class SequenceResult:
    n_events: int
    n_relevant: int
    event_recall: float  # over relevant events (all events when relevance is unknown)
    event_recall_all: float
    coarse_event_recall: float
    event_precision: float  # alerts attributed to a relevant event with the right class / alerts
    coarse_event_precision: float
    n_alerts: int
    false_alerts_relevance: int
    false_alerts_phantom: int
    false_alerts_wrong_class: int
    km: float | None
    false_alerts_per_km: float | None
    mean_latency_first_s: float | None
    mean_latency_final_s: float | None
    mean_flips: float
    events: list[EventResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "events"}
        d["events"] = [vars(e) for e in self.events]
        return d


def _attribute(records: list[FrameRecord], events: list[Event], iou_thr: float, match_gap: int
               ) -> dict[tuple[int, int], str | None]:
    """Map (frame index, system track id) to an event id (or None) by box overlap."""
    gt_frames: dict[int, list[tuple[str, tuple]]] = {}
    for e in events:
        for idx, box in e.boxes.items():
            gt_frames.setdefault(idx, []).append((e.event_id, box))
    out: dict[tuple[int, int], str | None] = {}
    sticky: dict[int, str | None] = {}  # track id -> event while the sign is memorised (not visible)
    for r in records:
        idx = r.meta.index
        boxes = {s.track_id: s.box for s in r.signs}
        visible = {s.track_id: s.visible for s in r.signs}
        for m in r.messages:
            key = (idx, m.track_id)
            if key in out:
                continue
            box = boxes.get(m.track_id)
            if box is None:
                out[key] = None
                continue
            if not visible.get(m.track_id, True) and m.track_id in sticky:
                out[key] = sticky[m.track_id]  # a memorised sign keeps the event it had when last seen
                continue
            best, best_iou = None, 0.0
            for d in range(0, match_gap + 1):
                for j in {idx - d, idx + d}:
                    for eid, gbox in gt_frames.get(j, []):
                        iou = float(iou_xyxy(np.array(box), np.array(gbox))[0, 0])
                        if iou > best_iou:
                            best, best_iou = eid, iou
                if best is not None:
                    break
            out[key] = best if best_iou >= iou_thr else None
            sticky[m.track_id] = out[key]
    return out


def evaluate_sequence(
    records: list[FrameRecord],
    events: list[Event],
    window_after_pass_s: float = 2.0,
    iou_thr: float = 0.5,
    match_gap: int = 2,
    coarse_of: Callable[[str], str] | None = None,
    ignore_classes: frozenset[str] = frozenset({"URDBL", "UNREADABLE"}),
) -> SequenceResult:
    """``ignore_classes`` are emitted labels that mean "a sign, class unknown": they count neither
    as a class change nor as a wrong class (datasets label unreadable signs that way)."""
    coarse_of = coarse_of or (lambda fine: fine)
    attr = _attribute(records, events, iou_thr, match_gap)
    by_event: dict[str, list[Message]] = {e.event_id: [] for e in events}
    alerts_ok = alerts_coarse_ok = fa_rel = fa_phantom = fa_wrong = n_alerts = 0
    ev_by_id = {e.event_id: e for e in events}
    for r in records:
        for m in r.messages:
            eid = attr[(r.meta.index, m.track_id)]
            if eid is not None:
                by_event[eid].append(m)
            if m.state not in ("new", "update"):
                continue
            n_alerts += 1
            if eid is None:
                fa_phantom += 1
            elif ev_by_id[eid].relevant is False:
                fa_rel += 1
            elif m.fine == ev_by_id[eid].fine:
                alerts_ok += 1
                alerts_coarse_ok += 1
            elif m.fine in ignore_classes:
                alerts_coarse_ok += 1
            elif coarse_of(m.fine or "") == ev_by_id[eid].coarse:
                alerts_coarse_ok += 1
                fa_wrong += 1
            else:
                fa_wrong += 1

    results: list[EventResult] = []
    for e in events:
        # One output per frame per event: when several system tracks are attributed to the same
        # sign in one frame, the highest-confidence message is what the system would display.
        per_frame: dict[int, Message] = {}
        for m in by_event[e.event_id]:
            cur = per_frame.get(m.index)
            if cur is None or m.confidence > cur.confidence:
                per_frame[m.index] = m
        msgs = sorted(per_frame.values(), key=lambda m: (m.time_s, m.index))
        window_end = e.pass_time_s + window_after_pass_s
        in_window = [m for m in msgs if e.first_time_s <= m.time_s <= window_end]
        classes = [m.fine for m in msgs if m.fine is not None and m.fine not in ignore_classes]
        flips = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
        first_ok = next((m for m in in_window if m.fine == e.fine), None)
        hit = False
        if first_ok is not None:
            later = [m for m in msgs if first_ok.time_s <= m.time_s <= e.pass_time_s
                     and m.fine is not None and m.fine not in ignore_classes]
            hit = all(m.fine == e.fine for m in later)
        coarse_hit = any(coarse_of(m.fine or "") == e.coarse or m.coarse == e.coarse for m in in_window)
        timed = [m for m in msgs if m.fine is not None and m.fine not in ignore_classes]
        change_times = ([timed[0].time_s] + [b.time_s for a, b in zip(timed, timed[1:]) if a.fine != b.fine]) if timed else []
        results.append(EventResult(
            event_id=e.event_id, fine=e.fine, relevant=e.relevant, hit=hit, coarse_hit=coarse_hit,
            n_messages=len(msgs), flips=flips,
            latency_first_s=(first_ok.time_s - e.first_time_s) if first_ok else None,
            latency_final_s=(change_times[-1] - e.pass_time_s) if change_times else None,
            classes_emitted=classes,
        ))

    relevant = [r for r, e in zip(results, events) if e.relevant is not False]
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None
    dist = [r.meta.distance_m for r in records if r.meta.distance_m is not None]
    km = (max(dist) - min(dist)) / 1000.0 if len(dist) >= 2 else None
    n_fa = fa_rel + fa_phantom + fa_wrong
    return SequenceResult(
        n_events=len(events), n_relevant=len(relevant),
        event_recall=_rate(sum(r.hit for r in relevant), len(relevant)),
        event_recall_all=_rate(sum(r.hit for r in results), len(results)),
        coarse_event_recall=_rate(sum(r.coarse_hit for r in relevant), len(relevant)),
        event_precision=_rate(alerts_ok, n_alerts), coarse_event_precision=_rate(alerts_coarse_ok, n_alerts),
        n_alerts=n_alerts, false_alerts_relevance=fa_rel, false_alerts_phantom=fa_phantom,
        false_alerts_wrong_class=fa_wrong, km=km, false_alerts_per_km=(n_fa / km) if km else None,
        mean_latency_first_s=_mean([r.latency_first_s for r in relevant if r.hit]),
        mean_latency_final_s=_mean([r.latency_final_s for r in relevant if r.hit]),
        mean_flips=float(np.mean([r.flips for r in relevant])) if relevant else 0.0,
        events=results,
    )


def _rate(num: float, den: int) -> float:
    return float(num / den) if den else 0.0


def speed_limit_accuracy(
    records: list[FrameRecord],
    limit_in_force: Callable[[FrameRecord], str | None],
    limit_of_message: Callable[[Message], str | None],
    hold_s: float | None = None,
) -> dict:
    """Share of the route where the system's current limit equals the annotated one.

    The system's current limit is the value of the last limit-carrying message (``new`` or
    ``update``) whose event has not ended, held for ``hold_s`` seconds if given. Frames where
    ``limit_in_force`` returns None are excluded. Weighted by distance when ``distance_m`` is
    present on the frames, otherwise by time.
    """
    current: str | None = None
    current_t = -np.inf
    good = total = 0.0
    prev_t = prev_d = None
    for r in records:
        for m in r.messages:
            if m.state in ("new", "update"):
                v = limit_of_message(m)
                if v is not None:
                    current, current_t = v, m.time_s
        if hold_s is not None and current is not None and r.meta.time_s - current_t > hold_s:
            current = None
        truth = limit_in_force(r)
        if prev_t is not None and truth is not None:
            w = (r.meta.distance_m - prev_d) if (r.meta.distance_m is not None and prev_d is not None) else (r.meta.time_s - prev_t)
            total += w
            good += w if current == truth else 0.0
        prev_t, prev_d = r.meta.time_s, r.meta.distance_m
    return {"accuracy": _rate(good, 1) / total if total else 0.0, "weight": total,
            "unit": "m" if any(r.meta.distance_m is not None for r in records) else "s"}
