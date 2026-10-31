# Decisions

Dated entries: what was decided, why, and the alternatives considered. Newest last.

Decisions that only concern the traffic light track are in `traffic_lights/docs/decisions.md`.

## 2026-10-31: Survey first, then replicate; cloud sessions are inference-only

Decided: the first milestone is a written survey of public datasets and open-source
implementations (for traffic lights: `traffic_lights/docs/datasets.md`, `traffic_lights/docs/oss_implementations.md`), followed by replication of
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

## 2026-10-31: The main target is traffic signs; traffic lights become a secondary track

Decided: OpenTSR's main target changes from traffic lights to traffic signs (road sign
detection and classification). The traffic light work done so far (dataset and implementation
surveys, S2TLD split manifests, Autoware, COCO-YOLO and DetectionBench replications) is kept as
a secondary track and is not deleted.

Why: decided by the maintainer. The traffic light results remain useful as a reference and
the shared parts (evaluator, split manifests, licensing policy) carry over to signs.

Alternatives considered: deleting the traffic light work (rejected: the measurements took real
effort and remain valid); a separate repository for traffic lights (rejected for now: the two
tracks share the evaluator and conventions, and a second repository doubles maintenance).

## 2026-10-31: Repository layout keeps the two tracks apart

Decided: the repository root (`docs/`, `configs/`, `scripts/`, `src/opentsr/signs/`) belongs to
traffic signs. Everything specific to traffic lights lives under a `traffic_lights` path
segment and nowhere else:

- `traffic_lights/` at the root: its README, docs, configs, scripts and Makefile
- `src/opentsr/traffic_lights/`: its Python package
- `data/traffic_lights/`, `weights/traffic_lights/`, `runs/traffic_lights/`: its local, gitignored files

Code shared by both tracks sits at the root of `src/opentsr/` (`metrics.py`, `splits.py`) and
must not import from `opentsr.traffic_lights`. Project-wide decisions stay in this file;
decisions that only concern traffic lights are in `traffic_lights/docs/decisions.md`.

Why: one rule ("traffic light material always sits under `traffic_lights`") keeps the tracks
from mixing as the sign work grows, and lets the secondary track be removed or split out later
with a single `git mv`.

Alternatives considered: symmetric `signs/` and `traffic_lights/` folders at the root
(rejected: it makes the main track as deep as the secondary one); tags or file name prefixes
inside shared folders (rejected: easy to get wrong, hard to grep).
