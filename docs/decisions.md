# Decisions

Dated entries: what was decided, why, and the alternatives considered. Newest last.

## 2026-10-31: Survey first, then replicate; cloud sessions are inference-only

Decided: the first milestone is a written survey of public datasets and open-source
implementations (`docs/datasets.md`, `docs/oss_implementations.md`), followed by replication of
the implementations whose weights and evaluation data are reachable. Cloud (claude.ai) sessions
have 2 CPU cores and no GPU, so they run inference-only replications and pipeline smoke tests;
every training run is handed off to the maintainer's machine or a lab GPU with a config under
`configs/` and a one-line command.

Why: numbers from the literature are not comparable across protocols (split, metric, IoU
threshold). Measuring a few open models ourselves on one fixed protocol gives a baseline that
later OpenTSR models can be compared against.

Alternatives considered: starting with model training directly (rejected: no benchmark yet to
measure against); running training on CPU (rejected: a 60 epoch run at 1280 px on 2 cores would
take days and would not match a GPU run anyway).

## 2026-10-31: S2TLD is the first replication dataset and the seed of the fixed benchmark

Decided: use S2TLD (SJTU Small Traffic Light Dataset, 5,786 images, 14,130 boxes, five states)
as the first dataset for replication and as the first component of the fixed OpenTSR benchmark.
Two split manifests are checked in under `configs/s2tld/splits/`:

- `random60_seed0.json`: random 6:4 train/test, the SCRDet++ (TPAMI 2022) protocol, used only
  to compare with the paper's numbers.
- `chunked_60_10_30_seed0.json`: contiguous 100-frame chunks per camera sequence shuffled into
  train/val/test (60/10/30). Frames in S2TLD are consecutive video frames; a random split puts
  near-identical frames on both sides and inflates test scores, so the chunked split is the one
  OpenTSR reports on.

Why: S2TLD is small enough to download in a cloud session (1.4 GB), is hosted by its author on
Hugging Face with a permissive license in the repository (MIT in the GitHub LICENSE file,
`apache-2.0` tag on the Hugging Face mirror; no separate data license text), has housing-level
boxes with state labels including `off`, and was the dataset most other candidates failed on:
BSTLD is 18 GB and its license is research-oriented, LISA is CC BY-NC-SA 4.0 (non-commercial),
DTLD requires registration and forbids redistribution, BDD100K requires registration.

Open point: confirm with the S2TLD authors that the MIT license is meant to cover the images and
annotations, not only the repository, before redistributing any derived labels.

## 2026-10-31: Evaluation metrics live in the repository, not in a framework

Decided: `src/opentsr/metrics.py` implements PASCAL VOC AP (all-point interpolation by default,
VOC07 11-point optional) and simple classification metrics. Ultralytics `val` results
(COCO-style 101-point interpolation) are reported for YOLO baselines as a convenience but every
OpenTSR number will be recomputed with the in-repo evaluator so that models from different
frameworks are scored identically.

Why: SCRDet++ reports VOC-style AP at IoU 0.5; DTLD and BSTLD baselines use yet other
conventions. One evaluator removes that ambiguity.

## 2026-10-31: Ultralytics is for baselines only until the licensing question is settled

Decided: Ultralytics YOLO11 (AGPL-3.0, weights included) is used to establish quick baselines
on S2TLD because it is the fastest path to a trained detector. It is not yet the framework of
the model OpenTSR will release. Candidate Apache-2.0 or MIT alternatives to evaluate next:
YOLOX (Apache-2.0, also what Autoware's fine detector uses), RT-DETR / D-FINE / RF-DETR style
detectors with permissive licenses, and MMDetection (Apache-2.0).

Why: OpenTSR wants commercial use of code, data and weights without copyleft obligations.
Weights trained with AGPL-3.0 tooling are treated by Ultralytics as AGPL-3.0 as well, which
conflicts with an Apache-2.0 release of the weights.

## 2026-10-31: Autoware's released models are the reference production stack to measure first

Decided: the Autoware Foundation traffic light classifier (MobileNetV2, v4.0) and fine detector
(YOLOX-s, v3.0), republished on Hugging Face under Apache-2.0, are the first production-grade
open models we measure, and their preprocessing is re-implemented exactly in
`src/opentsr/models/autoware_onnx.py` (letterbox, normalisation, YOLOX decode and NMS as in
`autoware_universe`).

Why: they are the only open, deployed, permissively licensed traffic light models with a public
accuracy claim (99.81% classification accuracy on TIER IV's internal Japanese test set). Their
training data is private, so the claim cannot be replicated directly, but cross-domain accuracy
on public data tells us how far a production classifier transfers and what a map-prior ROI
pipeline costs in recall.
