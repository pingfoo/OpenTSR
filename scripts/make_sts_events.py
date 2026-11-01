#!/usr/bin/env python
"""Convert STS annotations into sign-pass events (the L3 ground truth), one file per set.

Pipeline: annotations.txt -> frames in time order -> sequences (gap > 10 s) -> tracks (the
deterministic linker in opentsr.signs.events, parameters recorded in the output) -> events.
The output JSON holds the events with their per-frame boxes, so the L3 evaluator needs nothing
else from the dataset except the frame ids. The generated files are checked in under
configs/sts/ because they are small and derived from CC BY 4.0 data; regenerate with

    python scripts/make_sts_events.py --root data/sts --out configs/sts
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.sts import load_annotations, sequences  # noqa: E402
from opentsr.signs.events import events_from_tracks, events_to_json, link_tracks  # noqa: E402
from opentsr.signs.taxonomy import Taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sts")
    ap.add_argument("--out", default="configs/sts")
    ap.add_argument("--sets", nargs="+", default=["Set1", "Set2"])
    ap.add_argument("--min-px", type=float, default=12.0)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--max-gap", type=int, default=1)
    ap.add_argument("--center-factor", type=float, default=1.5)
    ap.add_argument("--first-factor", type=float, default=5.0)
    ap.add_argument("--scale-ratio", type=float, default=2.0)
    args = ap.parse_args()
    mapping = Taxonomy.load().mapping("sts")
    for s in args.sets:
        frames = load_annotations(Path(args.root) / s / "annotations.txt", set_name=s)
        for f in frames:  # event coarse classes are coarse_v1 families, not the dataset's own types
            for b in f.boxes:
                b.coarse = mapping.coarse_of(b.fine)
        events = []
        for sid, frs in sequences(frames).items():
            tracks = link_tracks([b for f in frs for b in f.boxes], iou_thr=args.iou_thr, max_gap=args.max_gap,
                                 center_factor=args.center_factor, first_factor=args.first_factor,
                                 scale_ratio=args.scale_ratio)
            events += events_from_tracks(tracks, sid, {f.index: f.time_s for f in frs}, min_px=args.min_px)
        meta = {
            "dataset": "STS", "set": s, "source": "CVL annotations.txt (CC BY 4.0 via Zenodo 19666339)",
            "frames": len(frames), "sequences": len(sequences(frames)), "coarse_taxonomy": mapping.taxonomy,
            "linker": {k: getattr(args, k) for k in ("iou_thr", "max_gap", "center_factor", "first_factor", "scale_ratio")},
            "min_px": args.min_px,
            "frame_ids": [f.file for f in frames],  # index i of an event refers to frame_ids[i]
        }
        out = Path(args.out) / f"events_{s.lower()}.json"
        events_to_json(out, events, meta)
        n_frames = Counter(e.n_frames for e in events)
        print(f"{s}: {len(frames)} frames -> {len(events)} events; relevant {sum(bool(e.relevant) for e in events)}; "
              f"single-frame {n_frames[1]}; classes {dict(Counter(e.fine for e in events).most_common())}")
        print(f"  wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
