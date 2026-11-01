#!/usr/bin/env python
"""Ablation driver: run a system configuration and its one-off / one-on variants on sequences,
score every run at L3 and write one table.

Detections come from one of two sources:

  --detections DIR   per-frame JSON Lines dumped by a detector + classifier
                     (one object per line: {"frame_id": ..., "candidates": [{"box": [x1,y1,x2,y2],
                     "score": s, "coarse": c, "fine": f, "fine_score": p}, ...]})
  --simulate         detections derived from the STS ground truth with a seeded noise model
                     (misses, label flips, phantoms, score noise), to exercise the harness and the
                     temporal stages before any detector exists. Simulated numbers are not results.

Example:
  python scripts/run_ablation.py --simulate --sets Set1 --out runs/ablation/sts_sim

Writes <out>/<variant>.jsonl logs per sequence are not kept (they are large); per variant a
summary JSON and, at the end, <out>/table.md with the L3 metrics of every variant.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.signs.datasets.sts import IMAGE_HEIGHT, IMAGE_WIDTH, load_annotations, sequences  # noqa: E402
from opentsr.signs.eval_events import evaluate_sequence  # noqa: E402
from opentsr.signs.events import events_from_json  # noqa: E402
from opentsr.signs.pipeline import Candidate, FrameMeta, Pipeline, ablation_variants, load_config  # noqa: E402
from opentsr.signs.pipeline.stages import PrecomputedDetector  # noqa: E402
from opentsr.signs.taxonomy import Taxonomy  # noqa: E402

STS_CLASSES = ["30_SIGN", "50_SIGN", "60_SIGN", "70_SIGN", "80_SIGN", "90_SIGN", "100_SIGN", "110_SIGN", "120_SIGN",
               "NO_PARKING", "NO_STOPPING_NO_STANDING", "STOP", "GIVE_WAY", "PRIORITY_ROAD", "PEDESTRIAN_CROSSING",
               "PASS_RIGHT_SIDE", "PASS_LEFT_SIDE", "PASS_EITHER_SIDE", "OTHER"]


def simulate(frames, rng: random.Random, miss: float, miss_small: float, flip: float, phantom: float,
             coarse_of) -> dict[str, list[Candidate]]:
    out = {}
    for f in frames:
        cands = []
        for b in f.boxes:
            size = max(b.box[2] - b.box[0], b.box[3] - b.box[1])
            p_miss = miss + (miss_small if size < 16 else 0.0)
            if rng.random() < p_miss:
                continue
            fine = b.fine
            if rng.random() < flip:
                fine = rng.choice([c for c in STS_CLASSES if c != b.fine])
            score = min(1.0, max(0.05, rng.gauss(0.8 if fine == b.fine else 0.6, 0.12)))
            jitter = [rng.gauss(0, 0.04 * size) for _ in range(4)]
            box = tuple(v + j for v, j in zip(b.box, jitter))
            cands.append(Candidate(box=box, score=score, coarse=coarse_of(fine), extra={"fine": fine, "fine_score": score}))
        if rng.random() < phantom:
            s = rng.uniform(12, 40)
            x, y = rng.uniform(0, IMAGE_WIDTH - s), rng.uniform(0, IMAGE_HEIGHT - s)
            fine = rng.choice(STS_CLASSES)
            cands.append(Candidate(box=(x, y, x + s, y + s), score=rng.uniform(0.3, 0.7), coarse=coarse_of(fine),
                                   extra={"fine": fine, "fine_score": 0.5}))
        out[f.file] = cands
    return out


def read_dumps(path: Path, frames) -> dict[str, list[Candidate]]:
    out = {}
    for fp in sorted(path.glob("*.jsonl")):
        with open(fp) as fh:
            for line in fh:
                d = json.loads(line)
                out[d["frame_id"]] = [Candidate(box=tuple(c["box"]), score=c["score"], coarse=c.get("coarse", "sign"),
                                                extra={"fine": c.get("fine"), "fine_score": c.get("fine_score", c["score"])})
                                      for c in d["candidates"]]
    return out


def aggregate(results) -> dict:
    n_ev = sum(r.n_events for r in results)
    n_rel = sum(r.n_relevant for r in results)
    hits = sum(round(r.event_recall * r.n_relevant) for r in results)
    hits_all = sum(round(r.event_recall_all * r.n_events) for r in results)
    coarse_hits = sum(round(r.coarse_event_recall * r.n_relevant) for r in results)
    alerts = sum(r.n_alerts for r in results)
    ok = sum(round(r.event_precision * r.n_alerts) for r in results)
    fa_rel = sum(r.false_alerts_relevance for r in results)
    fa_ph = sum(r.false_alerts_phantom for r in results)
    fa_wc = sum(r.false_alerts_wrong_class for r in results)
    lat = [r.mean_latency_first_s for r in results if r.mean_latency_first_s is not None]
    flips = [r.mean_flips * r.n_relevant for r in results]
    return {
        "n_events": n_ev, "n_relevant": n_rel,
        "event_recall": hits / n_rel if n_rel else 0.0, "event_recall_all": hits_all / n_ev if n_ev else 0.0,
        "coarse_event_recall": coarse_hits / n_rel if n_rel else 0.0,
        "event_precision": ok / alerts if alerts else 0.0, "n_alerts": alerts,
        "false_alerts_relevance": fa_rel, "false_alerts_phantom": fa_ph, "false_alerts_wrong_class": fa_wc,
        "mean_latency_first_s": sum(lat) / len(lat) if lat else None,
        "mean_flips": sum(flips) / n_rel if n_rel else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/system/full.yaml")
    ap.add_argument("--base", default="configs/system/academic.yaml")
    ap.add_argument("--root", default="data/sts")
    ap.add_argument("--events-dir", default="configs/sts")
    ap.add_argument("--sets", nargs="+", default=["Set1"])
    ap.add_argument("--out", default="runs/ablation/sts")
    ap.add_argument("--detections", help="directory of per-frame JSONL detector dumps")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--miss", type=float, default=0.10)
    ap.add_argument("--miss-small", type=float, default=0.30, help="extra miss probability below 16 px")
    ap.add_argument("--flip", type=float, default=0.10)
    ap.add_argument("--phantom", type=float, default=0.05, help="per-frame probability of a spurious box")
    ap.add_argument("--only", nargs="*", help="variant names to run (default: all)")
    args = ap.parse_args()
    if not (args.simulate or args.detections):
        ap.error("give --detections DIR or --simulate")

    tax = Taxonomy.load()
    mapping = tax.mapping("sts", default="other")
    full = load_config(args.config)
    base = load_config(args.base)
    variants = ablation_variants(full, base)
    if args.only:
        variants = [v for v in variants if v["name"] in args.only]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for s in args.sets:
        frames = load_annotations(Path(args.root) / s / "annotations.txt", set_name=s)
        events = events_from_json(Path(args.events_dir) / f"events_{s.lower()}.json")
        rng = random.Random(f"{args.seed}:{s}")
        dets = simulate(frames, rng, args.miss, args.miss_small, args.flip, args.phantom, mapping.coarse_of) \
            if args.simulate else read_dumps(Path(args.detections), frames)
        data.append((s, frames, events, dets))

    rows = []
    for v in variants:
        results = []
        for s, frames, events, dets in data:
            ev_by_seq: dict[str, list] = {}
            for e in events:
                ev_by_seq.setdefault(e.sequence_id, []).append(e)
            for sid, frs in sequences(frames).items():
                pipe = Pipeline(v, overrides={"detector": PrecomputedDetector(detections=dets,
                                                                              min_score=(v["detector"] or {}).get("min_score", 0.0))})
                metas = [(FrameMeta(frame_id=f.file, index=f.index, time_s=f.time_s, sequence_id=sid,
                                    width=IMAGE_WIDTH, height=IMAGE_HEIGHT), None) for f in frs]
                recs = list(pipe.run(metas))
                results.append(evaluate_sequence(recs, ev_by_seq.get(sid, []), coarse_of=mapping.coarse_of))
        agg = aggregate(results)
        agg["name"] = v["name"]
        agg["stages_on"] = [k for k in ("tracker", "memory", "classifier", "reader", "relevance", "fusion", "decision", "defense")
                            if v.get(k) is not None]
        agg["config"] = v
        (out_dir / f"{v['name']}.json").write_text(json.dumps(agg, indent=1))
        rows.append(agg)
        print(f"{v['name']:<24} recall {agg['event_recall']:.3f}  precision {agg['event_precision']:.3f}  "
              f"alerts {agg['n_alerts']:5d}  FA rel/ph/wc {agg['false_alerts_relevance']}/{agg['false_alerts_phantom']}/"
              f"{agg['false_alerts_wrong_class']}  flips {agg['mean_flips']:.2f}")

    source = "simulated detections (seed %d, miss %.2f, miss<16px +%.2f, flip %.2f, phantom %.2f)" % (
        args.seed, args.miss, args.miss_small, args.flip, args.phantom) if args.simulate else f"detections from {args.detections}"
    lines = [f"# L3 ablation on STS {' '.join(args.sets)}", "", f"Source: {source}. Events: {rows[0]['n_events']} "
             f"({rows[0]['n_relevant']} relevant). Numbers from simulated detections exercise the harness only.", "",
             "| variant | event recall | coarse recall | event precision | alerts | FA relevance | FA phantom | FA wrong class | "
             "latency first (s) | flips / event |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lat = f"{r['mean_latency_first_s']:.2f}" if r["mean_latency_first_s"] is not None else "n/a"
        lines.append(f"| {r['name']} | {r['event_recall']:.3f} | {r['coarse_event_recall']:.3f} | {r['event_precision']:.3f} | "
                     f"{r['n_alerts']} | {r['false_alerts_relevance']} | {r['false_alerts_phantom']} | "
                     f"{r['false_alerts_wrong_class']} | {lat} | {r['mean_flips']:.2f} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'table.md'}")


if __name__ == "__main__":
    main()
