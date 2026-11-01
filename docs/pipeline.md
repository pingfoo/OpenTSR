# Pipeline skeleton, event ground truth and the L2 / L3 evaluators (2026-11-01)

Status: phase 0 deliverable of `system_design.md` section 8. Everything here runs on a CPU in
seconds and has unit tests (`tests/test_pipeline.py`, `tests/test_events.py`,
`tests/test_eval_frames.py`). No learned model is involved yet: the stages that exist are the
ones that need no weights, and detections enter the pipeline from a dump (or, for exercising
the harness, from a seeded corruption of the ground truth).

## 1. What exists

| Piece | Where | Notes |
|---|---|---|
| Stage interface, registry, `null` = pass-through | `src/opentsr/signs/pipeline/stages.py` | nine slots in fixed order: detector, tracker, memory, classifier, reader, relevance, fusion, decision, defense |
| Model-free stages | same file | `detector: precomputed`, `tracker: iou` and `tracker: motion`, `memory: ttl`, `classifier: precomputed` and `function`, `relevance: geometric`, `decision: per_frame_argmax` and `vote` |
| Configuration loading and ablation variants | `pipeline/config.py` | unknown keys and a null detector are rejected; `ablation_variants` gives all-on, one-off-at-a-time, one-on-at-a-time over a base, and named presets |
| Runner and JSON Lines log | `pipeline/runner.py` | one `FrameRecord` per frame with candidates, sign states, messages and per-stage milliseconds; `read_log` replays a file |
| System configurations | `configs/system/full.yaml`, `full_sparse.yaml`, `academic.yaml` | `full_sparse` is `full` with the frame-count parameters set for sequences labelled below 1 fps |
| Coarse taxonomy v1 and mapping tables | `configs/taxonomy/`, `src/opentsr/signs/taxonomy.py` | 13 shape-and-colour families; maps for GTSRB/GTSDB, STS and CURE-TSR, validated at load |
| Sign-pass events | `src/opentsr/signs/events.py`, `scripts/make_sts_events.py` | track linker for sparse labels, track-to-event converter, JSON format; STS event files checked in under `configs/sts/` |
| L3 evaluator | `src/opentsr/signs/eval_events.py` | event recall, precision, false alerts by cause, latency, flips, coarse versions, distance-based speed-limit accuracy |
| L2 evaluator | `src/opentsr/signs/eval_frames.py` | AP50 and AP50:95 per fine class, per coarse family and class-agnostic, recall by size bucket, on the in-repo VOC matcher |
| STS loader and fetch script | `src/opentsr/signs/datasets/sts.py`, `scripts/prepare_sts.py` | annotation format verified on the files; one coordinate typo in Set2 handled |
| Ablation driver | `scripts/run_ablation.py` | runs every variant over the STS sequences and writes `table.md`; `--simulate` for harness checks, `--detections DIR` for real dumps |

Semantics of a switched-off stage (what `null` means), as implemented:

- tracker off: every candidate is its own one-frame sign (negative track ids).
- memory off: a zero-TTL memory, so tracked signs keep their per-track history while visible
  and are forgotten the first frame they are missing. This keeps "tracking without memory"
  measurable.
- classifier off: the detector's class is emitted (fine if the detector is fine-grained,
  otherwise the coarse family, with `fine = null` in the message).
- decision off: per-frame argmax.
- reader, relevance, fusion, defense off: the stage is skipped (every sign relevant, camera
  only, no hardening).

## 2. STS as the first L3 set

- Zenodo record 19666339 (Larsson and Felsberg, uploaded 2026-04-20, `traffic-signs_dataset.zip`,
  15.8 GB) is licensed CC BY 4.0; verified through the Zenodo API on 2026-11-01. The CVL pages
  still carry the 2011 "no commercial use without approval" sentence; the Zenodo copy is the one
  OpenTSR uses and cites. The maintainer still intends to tell the authors.
- The annotation files alone (`annotations.txt`, 1,970 and 1,807 lines) are downloadable from
  CVL and are enough to build the event ground truth; the frames (Part0, about 1.5 GB per set)
  are needed only to run a detector.
- Format: `<burst>Image<n>.jpg:STATUS, x_max, y_max, x_min, y_min, TYPE, NAME;...`, status in
  VISIBLE / BLURRED / OCCLUDED / SIDE_ROAD, 20 names in 6 types. SIDE_ROAD is a relevance label
  and becomes `relevant = false` on the event when it is the majority status of the track.
- Timing: the burst prefix is a Unix second and every fifth raw frame is annotated; the raw
  frame rate is not documented, so `time_s = burst + (n - 1) * 0.4` for now (about 2.5 fps,
  consistent with a 132-frame burst over 58 s). Sequences are cut at gaps above 10 s: 100
  sequences in Set1, 51 in Set2.
- Events after linking: Set1 839 (752 relevant, 219 single-frame), Set2 866 (808 relevant, 221
  single-frame). The single-frame share is a property of the sparse labelling, not of the
  linker: a sign passed at 90 km/h is visible above 12 px for roughly one or two annotated frames.

The linker (`events.link_tracks`) is deterministic and fully parameterised in the event file's
`meta.linker`: IoU 0.3 links outright; otherwise a linear extrapolation of centre and size over
the last two boxes with a 1.5 x predicted-size radius; a one-box track accepts a box within 5
sizes; size ratio at most 2.0; fine labels must agree except for OTHER / URDBL; a track survives
one unlabelled annotated frame. Two variants were tried and dropped on the simulated runs (an
"outward from the vanishing point" rule, wrong because the STS camera is not centred on the
road, and a one-box fallback for extrapolation misses, which merges neighbouring signs of the
same family). The linker has not yet been checked against the frames themselves; that is the
first thing to do once Part0 is downloaded.

## 3. Evaluator conventions that were decided while writing the tests

- A message is attributed to the event whose annotated box overlaps the message's track box
  (IoU 0.5) in the same frame or within 2 annotated frames; a memorised (not visible) sign keeps
  the event it had when last seen.
- When several system tracks are attributed to one event in one frame, only the
  highest-confidence message counts (what a display would show). Without this, memorised
  one-frame phantoms attached to a sign tripled the flip count.
- Labels that mean "a sign, class unknown" (`URDBL`, `UNREADABLE`) are neither a class change
  nor a wrong class.
- Alerts are messages with state `new` or `update`; `keep` messages do not count as alerts, so a
  per-frame system (where every message is `new`) has many more alerts than a voting system.
  Precision is therefore reported with the absolute false-alert counts next to it.
- Event recall is over relevant events; `event_recall_all` includes side-road signs.

## 4. Simulated ablation (harness check only, not a result)

`make ablation-sim` runs every variant of `full_sparse.yaml` on both STS sets with detections
derived from the ground truth plus seeded noise (10 percent misses, 30 percent extra misses below
16 px, 10 percent label flips, a phantom box in 5 percent of frames). The table lands in
`runs/ablation/sts_sim/table.md`. What the run shows about the harness, with simulated numbers
that must not be quoted as results:

- Switching a stage off changes exactly the columns it should: no classifier gives zero fine
  recall and unchanged coarse recall; no decision equals the per-frame stream; no tracker and
  no decision coincide (voting needs tracks).
- The geometric relevance placeholder (a sign whose centre is in the outer 20 percent of the
  image is "not relevant") costs about 10 points of recall on STS because passing signs reach
  the image edge; a learned relevance head is the fix, as planned.
- Requiring two frames to confirm (the `confirm_2` preset, which is what `full.yaml` does at
  10 fps and above) halves recall on a set labelled every 2 s. Frame-count parameters must be
  set per frame rate; the next step is to express them in seconds.
- With one-frame confirmation the vote reduces alerts from 5,659 to 2,171 and flips from 0.43 to
  0.32 per event at equal fine recall (0.73), and absolute false alerts fall by about 40 percent.

## 5. Next steps (phase 0, remaining)

1. Download STS Part0, look at the linked tracks on the frames, fix the linker where it is
   wrong, then freeze the event files and the linker parameters.
2. Express confirmation, minimum frames and TTL in seconds, derived from frame timestamps.
3. Dump real detections: run OWLv2 (candidate boxes) plus a GTSRB-trained classifier mapped to
   the STS names where they exist, write the per-frame JSONL, and produce the first non-simulated
   L3 table.
4. Implement the L2 entry point on GTSDB with the official test ground truth (local) and the
   size buckets, so that `evaluate_frames` has its first real table.
5. The GPU items of `replication.md` section 10 and the baseline retraining list in
   `benchmark.md` section 4 (lab machine).
