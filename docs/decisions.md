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

## 2026-10-31: GTSRB (official split) is the first sign classification benchmark; GTSDB for detection checks

Decided: the official GTSRB split (39,209 train / 12,630 test, 43 classes) is the first fixed
classification benchmark for signs, read from the Hugging Face mirror `bazyl/GTSRB` (parquet
export of the Kaggle release) with the revision and sha256 pinned in `scripts/prepare_gtsrb.py`
and the split sizes and class ids checked against the official numbers before use. The
`tanganke/gtsrb` mirror is not used for training because its 26,640-image train split is the
IJCNN 2011 Phase I subset, not the official training set; numbers trained on it are flagged as
such when compared. GTSDB is the detection check: in cloud sessions the 600 training frames with
the official gt.txt (852 boxes, 94 negatives) are used, because the 300 test frames' ground
truth only exists in the official FullIJCNN2013 archive hosted on sid.erda.dk, which the cloud
sessions cannot reach; the scripts take `--full-gt` so that the maintainer can score the real
test split locally.

Why: GTSRB is the one sign dataset whose terms ("The data is free to use", cite the paper) allow
redistribution of derived crops and labels, whose test labels are public, and for which the
literature gives dozens of comparable numbers from 2011 to 2026. GTSDB shares its class ids and
license statement ("free to use this data package for any purpose").

Alternatives considered: TT100K or MTSD as the first detection benchmark (rejected for now:
both non-commercial, both large; they stay as pointer-based evaluation sets, see
`docs/datasets.md` section 8); torchvision's GTSRB loader (rejected: downloads
from sid.erda.dk).

## 2026-10-31: Benchmark tiers for signs follow the traffic light policy

Decided: classification and detection benchmarks for signs are organised in the same tiers as
for traffic lights. Tier A (redistributable, OpenTSR ships fixed splits): GTSRB, GTSDB, Synset
Signset Germany (CC BY 4.0), CURE-TSR and STS from their Zenodo CC BY 4.0 copies, ZOD (CC BY-SA
4.0, share-alike on derived labels) and the Panoramax crop sets (CC BY-SA 4.0). Tier B
(non-commercial, test labels public, pointer distribution only): TT100K, MTSD, DFG, BTSD, RTSD,
CCTSDB. Tier C (registration): none needed yet. Japan has no public set in any tier, so a
Japanese benchmark has to be built from Mapillary imagery (CC BY-SA 4.0) or own recordings plus
synthetic renders of the public-domain 標識令 pictograms. Details and sizes in
`docs/datasets.md` section 8.

Why: same reasoning as for traffic lights: anything OpenTSR redistributes must be usable
commercially, and non-commercial sets can still be used to report numbers.

## 2026-10-31: Third-party checkpoints are loaded only through restricted unpicklers

Decided: Ultralytics-style `.pt` files from third parties (full pickles) are not loaded with
`YOLO()` or `torch.load(weights_only=False)` in cloud sessions. `scripts/eval_panoramax_de_classifier.py`
shows the pattern: disassemble the pickle with `pickletools`, refuse anything outside an
allow-list of torch / torchvision / ultralytics module classes, then load with
`torch.load(weights_only=True)` after registering exactly those classes. State-dict
checkpoints (for example poojahira's `model_40.pth`) load with the default `weights_only=True`.

Why: a pickle can execute arbitrary code at load time; a survey that pulls weights from dozens
of unknown accounts must not run them blindly. The restricted loader costs nothing for
well-formed checkpoints.

## 2026-10-31: CPU replications for signs define the comparison floor; GPU runs are queued, not improvised

Decided: the CPU-only replications in `docs/replication.md` (from-scratch CNNs on
GTSRB, released checkpoints on the official test split, CLIP zero-shot (SigLIP queued), small VLMs in
multiple choice, open-vocabulary detectors on GTSDB, Panoramax's German classifier on its own
validation set) are the reference points that the first OpenTSR models must beat. Training a
detector on GTSDB / ZOD / TT100K and fine-tuning a VLM are GPU tasks and are listed with
configs and expected cost in that file rather than run on 2 cores.

Why: the replicated numbers are either reproduced claims (poojahira 99.71 vs 99.81 claimed,
CLIP ViT-B/32 zero-shot vs 32.2 in the paper) or measurements no paper gives (open-vocabulary
AP on GTSDB, VLM multiple-choice on GTSRB); both are needed before a new model is trained.

## 2026-10-31: Dependencies and local files are split per track as well

Decided: the layout rule ("traffic light material only under a `traffic_lights` path segment")
also covers dependency pins, package extras and local files. `requirements.txt` at the root
pins the sign track; `traffic_lights/requirements.txt` pins the traffic light track.
`pyproject.toml` keeps only the shared base (`numpy`, `PyYAML`) as hard dependencies and
exposes `[signs]`, `[lights]` and `[train]` extras so that an install never mixes the two.
Local files follow the same split: signs use `data/`, `weights/`, `runs/` directly
(`data/gtsrb_hf`, `weights/poojahira`, `runs/gtsrb`, ...), traffic lights use
`data/traffic_lights/`, `weights/traffic_lights/`, `runs/traffic_lights/`. Sign documents under
`docs/` do not discuss traffic light material; the only shared files are `README.md`,
`CLAUDE.md`, `Makefile`, `pyproject.toml`, `docs/README.md`, `docs/decisions.md` and
`docs/third_party.md`.

Why: the maintainer asked that traffic light and sign data, code and documents never mix.
A root `requirements.txt` that pinned onnxruntime and OpenCV for the Autoware wrappers was the
last place where the two tracks shared a file by accident.
