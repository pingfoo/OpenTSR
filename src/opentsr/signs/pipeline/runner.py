"""Run a system configuration over a sequence of frames and log every frame as JSON Lines.

``Pipeline.run(frames)`` yields one ``FrameRecord`` per frame; ``run_sequence`` also writes the
records to ``<log_dir>/<sequence_id>.jsonl`` so that L3 evaluation (``opentsr.signs.eval_events``)
and later analyses can read the message stream without rerunning the stages. ``read_log``
replays a file.

Memory switched off is implemented as a zero-TTL memory: tracked signs keep their per-track
history while they are visible and are forgotten the first frame they are not seen, so that the
tracker's benefit (voting over the track) is still measurable without memorisation. With the
tracker also off, every candidate is a sign for exactly one frame, which is the academic
per-frame setting. Decision switched off means per-frame argmax, and classifier switched off
means the detector's class (fine when the detector is fine-grained, else coarse) is emitted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from opentsr.signs.pipeline.config import SystemConfig, normalize
from opentsr.signs.pipeline.schema import FrameMeta, FrameRecord
from opentsr.signs.pipeline.stages import STAGE_ORDER, FrameContext, PassThrough, Stage, build_stage
from opentsr.signs.pipeline.stages import PerFrameArgmax, TTLMemory


class Pipeline:
    def __init__(self, cfg: SystemConfig, overrides: dict[str, Stage] | None = None):
        """``overrides`` maps a slot to a ready-made stage object (for detections or classifier
        functions that cannot be expressed in YAML, for example in tests)."""
        self.cfg = normalize(cfg)
        self.stages: dict[str, Stage] = {}
        for slot in STAGE_ORDER:
            if overrides and slot in overrides:
                self.stages[slot] = overrides[slot]
            elif slot == "memory" and self.cfg[slot] is None:
                self.stages[slot] = TTLMemory(ttl_s=0.0)
            elif slot == "decision" and self.cfg[slot] is None:
                self.stages[slot] = PerFrameArgmax()  # "decision off" means per-frame argmax (system_design.md)
            else:
                self.stages[slot] = build_stage(slot, self.cfg[slot])

    def reset(self) -> None:
        for s in self.stages.values():
            s.reset()

    def run(self, frames: Iterable[tuple[FrameMeta, Any]]) -> Iterator[FrameRecord]:
        self.reset()
        for meta, image in frames:
            ctx = FrameContext(meta=meta, image=image)
            times: dict[str, float] = {}
            for slot in STAGE_ORDER:
                stage = self.stages[slot]
                if isinstance(stage, PassThrough):
                    continue
                t0 = time.perf_counter()
                stage(ctx)
                times[slot] = (time.perf_counter() - t0) * 1000.0
            yield FrameRecord(meta=meta, candidates=ctx.candidates, signs=_strip(ctx.signs), messages=ctx.messages,
                              stage_times_ms=times)


def _strip(signs):
    out = []
    for s in signs:
        s2 = type(s)(**{**vars(s), "extra": {k: v for k, v in s.extra.items() if k != "candidate"}})
        out.append(s2)
    return out


def run_sequence(pipeline: Pipeline, sequence_id: str, frames: Iterable[tuple[FrameMeta, Any]],
                 log_dir: str | Path | None = None) -> list[FrameRecord]:
    records = list(pipeline.run(frames))
    if log_dir is not None:
        write_log(Path(log_dir) / f"{sequence_id}.jsonl", records, pipeline.cfg)
    return records


def write_log(path: Path, records: list[FrameRecord], cfg: SystemConfig | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if cfg is not None:
            f.write(json.dumps({"config": cfg}) + "\n")
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")


def read_log(path: str | Path) -> tuple[SystemConfig | None, list[FrameRecord]]:
    cfg, records = None, []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "config" in d and "meta" not in d:
                cfg = d["config"]
            else:
                records.append(FrameRecord.from_dict(d))
    return cfg, records
