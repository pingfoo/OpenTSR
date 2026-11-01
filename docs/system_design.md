# OpenTSR system design: a switchable production-style pipeline (draft v0, 2026-11-01)

Status: draft for discussion. Companion to `benchmark.md` (what is measured) and
`commercial_tsr.md` (what production systems are documented to do). This file says how OpenTSR
is built so that it can be evaluated as a whole system and as an ablation of its parts, and how
adversarial attacks are evaluated against it.

## 1. Design goals

1. Reproduce the structure of production TSR (detect, track, memorize, classify, read, judge
   relevance, fuse priors) with open code, open data and open weights.
2. Every stage can be switched off or replaced from a configuration file, so that one run of the
   ablation driver quantifies the contribution of each stage to accuracy, robustness and cost.
3. The same configuration mechanism is what the adversarial evaluation uses: attacks are run
   against a stage, against the per-frame pair (detector plus classifier) and against the full
   system, and defenses are stages like any other.
4. Everything that runs in the vehicle is a CNN or a small sequence model that exports to ONNX
   and INT8; transformers and VLMs are used offline (labelling, generation, candidate mining).

## 2. Pipeline and stages

```
frame ──► Detector ──► Tracker ──► Memory ──► Classifier ──► Reader ──► Relevance ──► Fusion ──► Decision ──► messages
            │             │          │            │             │           │            │
         boxes +       tracks     persisted    fine class +   digits /    applies to   map / road
         coarse class  (ids)      signs        confidence     text        ego lane?    type prior
```

Each stage is a Python class with one method and a declared input/output schema; `null` in the
configuration removes the stage and passes its input through unchanged. Intermediate outputs are
written as JSON lines per frame so that any later analysis can be done without rerunning.

| Stage | Purpose | Planned options (first entry is the default) | Off means |
|---|---|---|---|
| Detector | find sign candidates in the full frame | CNN one-stage with coarse classes (YOLOX family, Apache-2.0) at 1280 px; tiled 2048 px; RT-DETRv2 / D-FINE for comparison; OWLv2 offline for candidate mining; class granularity: coarse / single class / fine | n/a (always present) |
| Tracker | link candidates across frames, suppress one-frame detections | IoU tracker; ByteTrack (MIT); Kalman-predicted association; confirmation after k frames | every frame independent |
| Memory | keep a recognised sign until it is passed | position-based clearing using ego speed or visual odometry; time-based TTL; none | output exists only while the sign is visible |
| Classifier | fine class on the aligned crop | ConvNeXt-atto class CNN at 64 to 96 px with a fixed margin; STN-CNN; ViT (reference); multi-crop or TTA; ensemble agreement; unknown-class rejection by confidence or energy | detector's coarse class is the output |
| Reader | digits and text | OCR head (PARSeq-style, Apache-2.0) on speed, weight, height and supplementary plates; consistency check against the classifier; variable-message signs | classifier output only |
| Relevance | does the sign apply to the ego vehicle | geometric heuristics (lateral offset, size growth, lane position); learned head trained on LISA "on side road", MTSD "direction-or-information" and OpenTSR labels | every sign is relevant |
| Fusion | priors and implicit limits | OSM maxspeed via GNSS; road-type classifier for implicit limits; conflict rules (camera wins on explicit signs) | camera only |
| Decision | the message stream | voting over the track (majority, confidence-weighted, Bayesian); hysteresis; minimum evidence before emitting | per-frame argmax |
| Defense | optional hardening | input purification (JPEG, bit-depth, median); adversarially trained classifier; agreement checks between detector, classifier and reader; temporal consistency; patch detectors | none |

Configuration example (`configs/system/full.yaml`):

```yaml
detector:   {type: yolox_coarse, weights: weights/opentsr/det_v1.onnx, input: 1280, classes: coarse_v1}
tracker:    {type: bytetrack, confirm_frames: 3}
memory:     {type: position, clear_after_pass_m: 5}
classifier: {type: convnext_atto, weights: weights/opentsr/cls_de_v1.onnx, crop_margin: 0.15, reject_below: 0.6}
reader:     {type: parseq_digits, weights: weights/opentsr/ocr_v1.onnx, agree_with_classifier: true}
relevance:  {type: geometric, max_lateral_offset_m: 6.0}
fusion:     {type: osm_prior, source: data/osm/jp_maxspeed.pbf}
decision:   {type: weighted_vote, min_frames: 3, hysteresis: 0.2}
defense:    null
```

`configs/system/academic.yaml` is the baseline configuration: fine-class detector, every other stage
`null`, per-frame argmax. The ablation driver (`scripts/run_ablation.py`, to be written) takes a
base configuration and a list of stages, runs "all on", "one off at a time", "one on at a time"
and any named presets, and writes one table with the L1 to L3 metrics, robustness columns and
latency per configuration.

## 3. Models and taxonomy

- Coarse taxonomy (`configs/taxonomy/coarse_v1.yaml`, to be defined): shape and colour families
  that exist in every country, for example prohibitory round red, mandatory round blue, warning
  triangle, warning diamond, priority (yield, priority road), stop, information rectangle,
  supplementary plate, other. Every dataset ships with a mapping table from its own labels; the
  detector trains on the union of all detection sets mapped to this taxonomy.
- Fine taxonomy per region: GTSRB 43 for Germany, ZOD 156 for Europe, 道路標識令 numbers for Japan,
  MTSD's 400 labels as the global reference map. The classifier is one model per region or one
  model with region-specific heads; either way it is replaceable without touching the detector.
- Training data policy: redistributable sets train the released weights; non-commercial sets are
  used only for reference evaluations (`decisions.md`, benchmark tiers).
- Synthetic data: Synset Signset for Germany; the 221 MLIT pictograms rendered the same way for
  Japan; CARLA sequences for L3 ablations. Synthetic-only and synthetic-plus-real rows are reported
  separately.

## 4. Cost and deployment

- Export every in-vehicle stage to ONNX; INT8 with calibration or QAT; report accuracy after export
  next to the float number (as the Vitis AI reference does: 62.3 float vs 62.1 INT8 for YOLOX-m).
- Latency table per stage and for the whole pipeline on a Jetson Orin class device and on a CPU,
  at the benchmark input sizes; the budget to respect is the 5 to 20 TOPS front-camera class
  (`commercial_tsr.md` section B).
- The transformer detectors stay in the benchmark as comparison rows with the same export and
  latency columns, so that the cost argument is measured.

## 5. Adversarial and robustness evaluation

### 5.1 Threat models and levels

| Level | Target | Attacker knowledge | Typical attacks |
|---|---|---|---|
| Crop | classifier (and reader) | white-box | PGD / C&W on the crop; printable patches (RP2, MIT); EOT over crop transforms |
| Frame | detector + classifier | white-box with ROIAlign for gradient flow through the crop; grey-box with released OpenTSR weights | ShapeShifter (BSD-3) style patches on the sign; hiding (suppress detection), altering (wrong class), appearing (sign-like patch where none exists); SLAP (GPL-3.0, laser), ShadowAttack (MIT) as physically realisable perturbations |
| Event | full system including tracker, memory, relevance, fusion | black-box / transfer from academic surrogates (Faster R-CNN, YOLO, RT-DETR); expectation over the frames of a sequence | the same attacks rendered along a track; phantom appearing signs on billboards or screens; timing attacks that exploit memorization |

The event level is non-differentiable (tracking, voting, memory), so it is attacked by transfer and
by expectation-over-sequence optimisation, which is the setting of the NDSS 2025 black-box study
of production vehicles (`commercial_tsr.md` A.5). OpenTSR adds what that study could not do:
switch each stage off and measure how the attack success rate changes.

### 5.2 Rendering physical attacks

- Real images: REAP (realistic adversarial patch benchmark on Mapillary signs with geometric and
  lighting transforms; license to be confirmed) for crop and frame levels.
- Sequences: CARLA drives with adversarial textures applied to sign actors, with weather and
  distance sweeps, for the event level; the rendering pipeline is published so that attacks are
  reproducible.
- Physical confirmation: printed patches and a real drive on a closed course, only for the final
  transfer check of the strongest attacks.

### 5.3 Metrics

- Attack success rate at each level, separately for hiding, altering and appearing goals.
- Matrix: attack x level x system configuration (stages on or off) x surrogate.
- Clean-accuracy cost of every defense configuration, from the same ablation run.
- Attacker cost: patch size, number of frames the attack must survive, physical constraints.
- Natural robustness in the same tables (CURE-TSR levels, GTSRB corruptions, CURE-TSD and CARLA
  weather), so that adversarial and natural degradations are compared on one scale.

## 6. Claims the design is meant to support

1. On L1 and L2 the OpenTSR models are at least as good as the retrained academic baselines
   (no regression from the modular design or from the coarse detector).
2. On L3 the full system beats per-frame deployment of the same or better models, and the ablation
   table attributes the gain to tracking, memory, relevance and the reader.
3. Attack success measured on a single model overstates or misstates the risk to the system;
   memorization makes hiding harder and appearing easier, and the ablation shows by how much.
4. The released system meets a stated compute budget after INT8 export.

## 7. Risks and mitigations

- Sequence data with event ground truth is scarce and partly non-redistributable: start with STS
  and CARLA, publish the event converter, and make the Japan collection the long-term asset.
- Relevance labels are scarce: LISA and MTSD attributes bootstrap a weak head; the Japan
  annotation protocol includes relevance from the start.
- A coarse detector can hide fine-class failures: always report the fine-class L2 numbers of the
  detector-plus-classifier pair next to the fine-class detectors.
- CARLA results can flatter: every synthetic finding is confirmed on at least one real sequence set.

## 8. Phases

| Phase | Weeks | Deliverables |
|---|---|---|
| 0 | 1 to 2 | `benchmark.md` v1 frozen; sequence data acquired (STS, ZOD registration, license checks); pipeline skeleton with configuration loading, JSONL logging, the L2 and L3 evaluators with unit tests; two or three academic baselines retrained |
| 1 | 3 to 10 | detector v1 (coarse, union of detection sets), classifier v1 (union of crops plus Synset), reader v1, tracker and voting; first L3 results on STS and CARLA; ablation table v1; INT8 export and latency table |
| 2 | 10 to 22 | relevance head, OSM prior, position-based memory; Japan collection and annotation; synthetic Japan signs; cross-country experiments |
| 3 | from week 4, in parallel | attack and defense stages, REAP and CARLA integration, transfer matrix, physical confirmation if a vehicle is available |
| 4 | after phase 2 | benchmark release (OpenTSR-Bench), weights, system card with the ablation and attack matrices, papers (system-level security study; benchmark and method) |

Decisions to record in `decisions.md` early: training framework (Apache-2.0 family), coarse
taxonomy and mapping tables, tracker and reader choices, the event annotation protocol, and how
CARLA evidence is weighted against real sequences.

## Amendments

### 2026-11-01 (phase 0 implementation, see `pipeline.md`)

- The skeleton exists under `src/opentsr/signs/pipeline/` with the nine slots of section 2 and
  `null` = pass-through. What "off" means per stage is now fixed in code: memory off is a
  zero-TTL memory (tracks still vote while visible), classifier off emits the detector's class,
  decision off is per-frame argmax.
- A `tracker: motion` option (the same association rule as the ground-truth linker) was added
  next to `tracker: iou`, because IoU association fails on sequences labelled every 2 s.
  ByteTrack and Kalman prediction remain planned options for full frame rate.
- `configs/system/full_sparse.yaml` is the all-on configuration for sparsely labelled sets;
  `full.yaml` keeps the 10 fps and above settings. The ablation driver `scripts/run_ablation.py`
  exists and produces the one-table output of section 2 for L3; the L1 and L2 columns and the
  latency column are added when the model-backed stages exist.
