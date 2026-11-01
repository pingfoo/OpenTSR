# OpenTSR benchmark specification (draft v0, 2026-11-01)

Status: draft for discussion. Once agreed, this file is frozen before any model is tuned against it,
and later changes are recorded as dated amendments at the end. The goal of the benchmark is to let
OpenTSR make three claims with evidence: (1) its detector and classifier are at least as good as
the methods used in academic work on the standard per-image benchmarks, (2) the full system, built
the way production systems are built (detect, track, memorize, classify, read, judge relevance),
is better on the metrics that matter for the product (per sign-pass events), and (3) the system is
measurably more robust to natural degradation and adversarial attacks than single models, with the
contribution of every component quantified by switching it off.

## 1. Three layers

| Layer | Unit of evaluation | What it compares | Why it exists |
|---|---|---|---|
| L1 crop classification | one cropped sign | OpenTSR classifier vs academic classifiers | Same ground as GTSRB-era work; cheap; isolates the classifier |
| L2 single-frame detection and fine classification | one frame with boxes | OpenTSR detector + classifier (tracking off) vs academic detectors | Same ground as TT100K / MTSD / GTSDB work; proves there is no regression before system-level gains are claimed |
| L3 sign-pass events in sequences | one physical sign passed by the ego vehicle | full OpenTSR system vs the same models applied frame by frame | The product metric (EU ISA, Euro NCAP); where tracking, memorization, relevance and OCR show their value |

Robustness (natural and adversarial) is measured at all three layers with the same splits, so that
"attack success on the crop" and "attack success on the event" can be put side by side.

## 2. Data per layer

### L1 (crops)

| Set | Role | Split | License / access | Notes |
|---|---|---|---|---|
| GTSRB official | main classification benchmark | official train 39,209 / test 12,630 | "free to use" + citation | per-class and per-subset (Neural Networks 2012 subsets) accuracy |
| GTSRB corrupted copies (tanganke/gtsrb) | natural robustness | the 7 corrupted test copies (contrast, gaussian noise, impulse noise, jpeg, motion blur, pixelate, spatter) | mirror, no license tag; derived from GTSRB | report accuracy per corruption |
| CURE-TSR | challenge robustness | official 70/30 by sign, 12 challenge types x 5 levels, real and unreal | CC BY 4.0 (Zenodo) | 14 coarse classes; train on challenge-free real only unless stated |
| Synset Signset Germany (GTSRB subset) | synthetic-to-real | train on synthetic only, test on GTSRB official test | CC BY 4.0 | reproduces the 98.3 percent claim as a baseline row |
| Panoramax DE / FR, English roadsigns, Dognellaf RU, Vietnamese crops | cross-country transfer | fixed seeded splits published by OpenTSR | CC BY-SA 4.0 / etalab-2.0 / CC BY 4.0 / CC BY-SA 4.0 | each evaluated with its own taxonomy; measures how a shared backbone transfers |
| Japan crops (OpenTSR collection) | Japan track | OpenTSR split by prefecture | CC BY 4.0 (to be released) | taxonomy: 道路標識令 numbers (221 pictograms of the MLIT list) |

### L2 (frames)

| Set | Role | Split | License / access | Notes |
|---|---|---|---|---|
| GTSDB | detection benchmark (Germany) | official 600 / 300 with FullIJCNN2013 ground truth (local), plus 5-fold CV over 900 | "free to use" | 43 classes and the 3 competition categories; class-agnostic AP as well |
| TT100K 2016 | detection + fine classification (China) | official train / test, 45 benchmark classes | CC BY-NC | reference only (not redistributed); report per the paper's protocol |
| MTSD | detection + fine classification (global) | official train / val (test labels withheld) | CC BY-NC-SA | reference only; taxonomy groups and the 400 fine labels |
| ZOD Frames | detection + fine classification (14 EU countries) | official split, OpenTSR split files published | CC BY-SA 4.0 (registration) | the main redistributable fine-grained detection benchmark |
| DFG | detection (Slovenia) | official | CC BY-NC-SA 4.0 | reference only |
| LISA TS | detection (US) + relevance attribute | OpenTSR split by video | academic license | reference only; "on side road" attribute used for relevance labels |
| Japan frames (OpenTSR collection) | Japan track | by prefecture | CC BY 4.0 (to be released) | |

Metrics: VOC AP50 and COCO-style AP at IoU 0.5:0.95 per fine class, per coarse category, and
class-agnostic; recall by sign size bucket (<16, 16-31, 32-47, 48-63, 64-95, 96-127, 128+ px long
side, as in `dataset_catalog.md`); all computed by the in-repo evaluator (`src/opentsr/metrics.py`,
extended), never by a framework's own validator.

### L3 (sequences and sign-pass events)

| Set | Role | License / access | Status |
|---|---|---|---|
| Swedish Traffic Signs (STS) | real sequences, labels every fifth frame, 7 classes | CC BY 4.0 (Zenodo 2026 upload; confirm with CVL) | to download (15.8 GB) |
| ZOD Sequences / Drives | real sequences, 14 countries | CC BY-SA 4.0 (registration) | annotation coverage per frame to be checked |
| BelgiumTS detection sequences | real sequences from 8 cameras | citation request only | frame numbering and sequence structure to be checked |
| CURE-TSD | real (BelgiumTS-derived) and Unreal Engine video with per-frame boxes and 12 challenges | form; license to be confirmed | the only large sequence set with controlled degradations |
| CARLA synthetic drives | fully controlled sequences (sign placement, weather, adversarial textures) | MIT (simulator) | OpenTSR-generated; used for ablations and physical-attack simulation, never as the only evidence |
| OpenTSR Japan drives | dashcam sequences with GNSS, event ground truth | CC BY 4.0 (to be released) | the Japan track's main asset |

Event ground truth: one record per physical sign instance passed by the ego vehicle, with the
fine class, the OpenTSR coarse class, relevance (applies to the ego lane / road: yes, no, unknown),
the first frame in which the sign is at least 12 px, the pass frame (sign leaves the field of view
or the ego vehicle passes its position), and, for speed limits, the limit in force after the pass.
The annotation protocol and a conversion tool from per-frame boxes with track ids are part of the
repository (`scripts/events_from_tracks.py`, to be written); sets that only have per-frame boxes
without track ids get track ids from a simple IoU linker that is itself published and seeded.

Metrics (defined on the system's output stream of messages {time, class, confidence, state}):

- Event recall: fraction of relevant events for which the system emits the correct fine class
  between the first-visible frame and 2.0 s after the pass (the ISA window), and keeps it (no
  change of class) until the pass.
- Event precision and false alerts per km: emitted sign messages that match no event (per km of
  driving), separately for messages caused by non-relevant signs (relevance errors) and by nothing.
- Distance-based speed-limit accuracy: fraction of driven distance where the system's current
  limit equals the limit in force (ISA's 90 percent figure), computed where the route's limit is
  annotated.
- Decision latency: time from first-visible to the first correct stable output, and from pass to
  the final output.
- Stability: number of class flips per event.
- Coarse-category versions of recall and precision, for comparing systems with different fine
  taxonomies.

Academic baselines on L3 are the per-frame models with no tracking: the message stream is the
per-frame top detection per sign, which is exactly how a single-model paper would deploy it.

## 3. Robustness protocol

- Natural: GTSRB corrupted copies and CURE-TSR at L1; CURE-TSD challenge sequences and CARLA
  weather sweeps at L3; report the metric per challenge type and level.
- Adversarial: the same metrics under the attacks listed in `system_design.md` section 5, at the
  three layers, with the attacker's knowledge stated (white-box on a stage, grey-box with a
  published OpenTSR checkpoint as surrogate, black-box transfer from academic surrogates).
- Every robustness number is reported with the clean number next to it and with the system
  configuration (which stages were on) as a column, so that a defense's cost is visible.

## 4. Baselines to reimplement (same data, same splits, same evaluator)

Classification: Udacity LeNet-5, poojahira STN-CNN (already in `replication.md`), ConvNeXt V2 atto
(Hub checkpoint and retrained), ViT-B/16 (retrained with a sign-aware recipe, no flips), CLIP
ViT-B/16 fine-tune and linear probe.

Detection: YOLOX-s/m (Apache-2.0) and YOLO11n/s (AGPL, baseline only) with fine classes; Faster
R-CNN (torchvision) as the GTSDB-era reference; RT-DETRv2 and D-FINE (Apache-2.0) as the
transformer detectors; each at 640 and 1280 px, each also exported to INT8 with the accuracy
reported after export.

Zero-shot and open-vocabulary references: OWLv2, YOLO-World, CLIP zero-shot (already measured).

## 5. Reporting rules

- Fixed public splits, seeds and the commit hash in every table; three seeds for anything trained
  by OpenTSR, with mean and standard deviation.
- Numbers come from `scripts/` entry points that write JSON under `runs/`; tables are generated
  from those files, not typed.
- Latency and memory on at least two targets (an NVIDIA Jetson Orin class device and a CPU), with
  the export path (ONNX, TensorRT) and precision stated.
- Non-commercial datasets (TT100K, MTSD, DFG, LISA) appear as reference columns and are never part
  of what OpenTSR redistributes.

## 6. Open items before freezing v1

1. Confirm the STS Zenodo license with CVL and the CURE-TSD terms; decide whether BelgiumTS
   sequences can be used for a public leaderboard (citation-only terms).
2. Define the OpenTSR coarse taxonomy and the mapping tables from every dataset (`configs/taxonomy/`).
3. Write and test the event evaluator and the track-to-event converter on STS.
4. Decide the Japan event annotation tool and the first collection route.

## Amendments

### 2026-11-01 (phase 0 implementation, see `pipeline.md`)

- STS license confirmed: Zenodo 19666339 is CC BY 4.0 (checked through the Zenodo API). Event
  ground truth for Set1 and Set2 is generated by `scripts/make_sts_events.py` and checked in
  under `configs/sts/` with the linker parameters in the file's metadata (open item 3 done at
  the tool level; the linker still has to be checked against the frames, open item 1 for the
  authors' confirmation stays).
- Coarse taxonomy v1 and the mapping tables for GTSRB/GTSDB, STS and CURE-TSR are in
  `configs/taxonomy/` (open item 2 done for these three sets).
- L3 metric conventions fixed in code: attribution by IoU 0.5 within 2 annotated frames, sticky
  attribution for memorised signs, one message per frame per event (highest confidence), unknown
  labels ignored for flips and stability, alerts are `new` and `update` messages only.
- L2 size buckets and AP50:95 are implemented in `src/opentsr/signs/eval_frames.py`.
- The frame-count parameters of the system (track confirmation, minimum frames to vote, TTL)
  will be expressed in seconds, because a sequence set labelled every 2 s and a 25 fps drive
  need different frame counts for the same behaviour.
