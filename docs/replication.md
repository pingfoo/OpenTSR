# Replication of open implementations

What we ran, how, and what came out. Every number below was produced by a script in this
repository on 2026-10-31 in a claude.ai cloud session (2 vCPU, 7 GB RAM, no GPU; Python 3.13,
onnxruntime 1.31.1, Ultralytics 8.4.194, torch 2.15.0+cpu). Raw JSON outputs are in
`runs/replication/` (gitignored); the commands are in the `Makefile`. See
`docs/oss_implementations.md` for why these implementations were chosen and
`docs/datasets.md` for the data.

Status legend: **done** (measured here), **GPU** (prepared, needs a GPU run), **blocked**.

| # | Implementation | Data | Status | Headline result |
|---|---|---|---|---|
| 1 | Autoware traffic light classifier (MobileNetV2 car, v4.0) | S2TLD test crops | done | 96.95 % state accuracy (cross-domain, Japanese model on Chinese lights) |
| 2 | same | TIER IV sample crops (val, 232) | done | 61.2 % (letterbox) / 81.9 % (stretch); crops are tiny vertical lights, out of domain |
| 3 | Autoware fine detector (YOLOX-s 416, v3.0) on synthetic coarse ROIs | S2TLD test | done | recall@IoU0.5 39.6 % at the deployed threshold 0.3, 48.2 % at 0.1 |
| 4 | Autoware whole-image TL detector (YOLOX-s 960, v1.0) | S2TLD test | done | class-agnostic AP50 8.0 %, AP30 19.4 % |
| 5 | COCO-pretrained YOLO11n, `traffic light` class, 1280 px | S2TLD test | done | class-agnostic AP50 65.9 %, AP30 72.2 % |
| 6 | DetectionBench `lisa-yolo26n` (`yolo val`, test split) | LISA (HF export) | done / in progress, see section 6 | reported mAP@50 23.74 |
| 7 | YOLO11s / YOLO11n fine-tuned on S2TLD (`configs/s2tld/*.yaml`) | S2TLD, both splits | GPU | SCRDet++ reference: 93 to 96 mAP@0.5 on a random 6:4 split |
| 8 | TLoNY YOLO (3840 px val), ImVisible LytNet V2, OpenLenda | HF / Google Drive / GitHub | not started | see `docs/oss_implementations.md` F2, F3, F5 |

## 1. Protocol shared by the S2TLD experiments

- Data: S2TLD, both resolution subsets, downloaded from the author's Hugging Face mirror
  (checksums in `scripts/prepare_s2tld.py`), 5,786 images / 14,130 boxes.
- Split: `configs/s2tld/splits/chunked_60_10_30_seed0.json` (contiguous 100-frame chunks per
  camera sequence, seed 0): train 3,386 / val 600 / test 1,800 images. The test split holds
  4,497 boxes: red 2,334, green 1,855, off 215, yellow 59, wait_on 34 (counts from the
  classifier run). All numbers below are on this test split unless stated.
- Box heights in the test split: 75 boxes under 16 px, 977 in [16, 32), 2,458 in [32, 64),
  987 of 64 px or more (image height 720 or 1080).
- Detection metric: PASCAL VOC AP, all-point interpolation (`src/opentsr/metrics.py`, unit
  tests in `tests/test_metrics.py`). Class-agnostic means every S2TLD box counts as the single
  class `tl` regardless of state.

## 2. Autoware MobileNetV2 state classifier on ground-truth crops (done)

Script: `scripts/eval_autoware_classifier.py`. Model: `traffic_light_classifier_mobilenetv2_batch_1.onnx`
from `AutowareFoundation/traffic_light_classifier` tag `v4.0` (11 car labels). Each S2TLD box is
cropped with no margin, converted to RGB, preprocessed as the Autoware node does (letterbox to
224 with `scale = min(224/w, 224/h)`, bilinear, zero padding bottom/right, mean
`[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`), and the argmax label is mapped to
S2TLD's vocabulary: every label containing `red` (red plus arrows) to `red`, `unknown` to
`off`, `green` and `yellow` unchanged. `wait_on` has no counterpart and is excluded from the
accuracy (34 boxes), so 4,463 crops are scored.

| Preprocessing | Accuracy | red recall | yellow | green | off | ms/crop (CPU) |
|---|---|---|---|---|---|---|
| letterbox (Autoware runtime) | **96.95 %** | 96.8 % | 89.8 % | 97.1 % | 99.1 % | 10.9 |
| stretch to 224x224 (mmpretrain `Resize(scale=224)` as in the training recipe) | 66.3 % | 92.4 % | 59.3 % | 30.0 % | 97.7 % | 11.2 |

Confusion matrix, letterbox (rows: S2TLD label, columns: mapped prediction green / off / red / yellow):

| | green | off | red | yellow |
|---|---|---|---|---|
| green (1,855) | 1,802 | 36 | 14 | 3 |
| off (215) | 2 | 213 | 0 | 0 |
| red (2,334) | 4 | 49 | 2,259 | 22 |
| wait_on (34) | 10 | 11 | 13 | 0 |
| yellow (59) | 0 | 4 | 2 | 53 |

Breakdown (letterbox): 1080x1920 subset 95.6 %, 720 `normal_1` 92.9 %, 720 `normal_2` 98.4 %;
by box height: under 16 px 93.8 %, 16 to 32 px 97.1 %, 32 to 64 px 97.8 %, 64 px and above
94.8 %. The raw predictions on red crops were `red` 2,252, `unknown` 49, `yellow` 22, arrow
variants 7; `wait_on` crops came out as `left,red,straight` 13, `unknown` 11, `green` 10.

Reading: the published 99.81 % (TIER IV internal Japanese test set) cannot be reproduced because
the data is private, but the released model transfers to Chinese vertical lights at about 97 %
on clean ground-truth crops with the runtime preprocessing. The large gap between the two
preprocessing variants is worth remembering: the training recipe in the Autoware README
stretches crops while the C++ node letterboxes them; whichever TIER IV actually used, the model
is sensitive to it (on S2TLD green crops the stretched variant collapses to 30 % recall).

## 3. Autoware classifier on the TIER IV sample crops (done)

Same script with `--folder`. The "sample dataset" linked from the Autoware classifier README
(1,045 crops, train 813 / val 232, red / green / yellow) turned out to be vertical, mostly
non-Japanese-looking lights, many under 20 px wide, so it is a tutorial dataset rather than a
sample of the training distribution. Results on `val` (232 crops): letterbox 61.2 % (green 75.6,
red 62.0, yellow 43.7; most misses are `unknown`), stretch 81.9 % (green 86.6, red 84.8, yellow
73.2). By crop width (stretch): 20 to 40 px 100 % on all three classes; under 20 px 38 to 69 %;
40 px and above 62 to 90 %. Do not use this set as a benchmark.

## 4. Autoware YOLOX-s fine detector on synthetic coarse ROIs (done)

Script: `scripts/eval_autoware_fine_detector.py`. Model: `tlr_car_ped_yolox_s_batch_1.onnx`
(`AutowareFoundation/traffic_light_fine_detector` tag `v3.0`). In Autoware this model refines
coarse ROIs projected from the HD map. Without a map we build one coarse ROI per ground-truth
box: the box scaled by `roi_scale` around its centre (at least 96 px per side), shifted by a
seeded random offset of up to 20 % of the ROI size, clipped to the frame. The crop is
letterboxed to 416 (pad 114, raw BGR, no normalisation), decoded as YOLOX raw output
(`obj * cls`), class-agnostic NMS at 0.65, boxes mapped back by the letterbox scale, exactly as
`autoware_tensorrt_yolox` does. The best-scoring `traffic_light` or `pedestrian_traffic_light`
detection is compared with the ground truth. Negatives: one random ROI per image that touches no
box.

| roi_scale | score thr | recall@IoU0.5 | recall@IoU0.3 | mean IoU | ROIs with no detection | false alarm rate on negatives | detections per positive ROI |
|---|---|---|---|---|---|---|---|
| 4 | 0.3 (deployed default) | **39.6 %** | 47.8 % | 0.353 | 45.5 % | 4.0 % (1,800 negatives) | 0.86 |
| 4 | 0.1 | 48.2 % | 57.5 % | 0.430 | 32.3 % | 7.8 % | 1.43 |
| 2 | 0.3 | 37.9 % | 46.9 % | 0.352 | 41.6 % | 3.6 % | 0.92 |

Recall@0.5 (thr 0.3) by box height: under 16 px 8.0 %, 16 to 32 px 24.0 %, 32 to 64 px 42.4 %,
64 px and above 50.8 %; by subset: 1080 40.5 %, 720 `normal_1` 29.1 %, 720 `normal_2` 43.0 %.
The best detection was labelled `traffic_light` 1,619 times and `pedestrian_traffic_light`
1,440 times: Chinese two- and three-lamp vertical housings look like Japanese pedestrian signals
to this model. Runtime about 100 ms per 416x416 ROI on CPU.

The ROI size barely matters (scale 2 vs 4); the score threshold does. Recall@0.5 at scale 2:
under 16 px 6.7 %, 16 to 32 px 23.3 %, 32 to 64 px 40.1 %, 64 px and above 49.2 %.

Reading: TIER IV reports AP 0.915 / recall 0.971 for `traffic_light` on their own ROI-crop test
set (AWML `YOLOX_opt-S-TrafficLight` v1, private data). On S2TLD the same weights find fewer than
half of the lights at the deployed threshold, and the boxes they do find overlap S2TLD's
annotations loosely (mean IoU 0.35 to 0.43; S2TLD boxes are drawn around the whole housing while
the Autoware detector appears to prefer tighter boxes). A map-prior pipeline therefore does not
remove the need for region-specific detector training.

## 5. Whole-image detectors on S2TLD frames (done)

Scripts: `scripts/eval_autoware_whole_image_detector.py` and `scripts/eval_coco_pretrained_yolo.py`.

| Detector | Input | AP50 (class-agnostic) | AP30 | Recall@IoU0.5 at deployment threshold | FP per image | ms/image (CPU) |
|---|---|---|---|---|---|---|
| Autoware `yolox_s_car_ped_tl_detector_960_960` (v1.0), score 0.01 for AP, NMS 0.7 | letterbox 960 | 8.0 % | 19.4 % | 15 % overall at thr 0.35 (green 24 %, red 9 %, yellow 0 %) | 1.29 at thr 0.35 | 739 |
| Ultralytics YOLO11n COCO weights, class 9 only, conf 0.001 | 1280 | **65.9 %** | 72.2 % | at conf 0.25: green 65 %, red 65 %, yellow 73 %, off 36 %, wait_on 3 % | n/a | 238 |

COCO YOLO11n recall at conf 0.25 by box height: under 16 px 4 %, 16 to 32 px 28 %, 32 to 64 px
71 %, 64 px and above 85 %.

Reading: a generic COCO detector already localises two thirds of S2TLD lights at IoU 0.5 and is
the natural starting point for a detector that OpenTSR fine-tunes; its weakness is the small
end (under 32 px), which is where TLR datasets put most of their difficulty. The Autoware
whole-image model, trained on about 17 k Japanese frames at 960 px, does not transfer to this
camera and country at all (AP50 8 %), which quantifies the domain gap OpenTSR has to close
either with multi-region data or with generated data.

## 6. DetectionBench `lisa-yolo26n` on the LISA test split (in progress)

Target: mAP@50 23.74, mAP@50-95 10.57, precision 37.23, recall 25.7 on the `test` split of
`dronefreak/LISA-Traffic-Lights` (model card of `dronefreak/lisa-yolo26n`, AGPL-3.0,
`best.pt` sha256 `6eeca4dece2d433c629fcf14ef2b260c7cd249b527e31a75a50b93e112d47f66`).
Protocol: `scripts/prepare_lisa_hf.py` (writes YOLO label files from the export's
`metadata.jsonl`), then
`yolo val model=weights/detectionbench/lisa-yolo26n_best.pt data=data/lisa_hf/data/data_abs.yaml split=test imgsz=640 batch=8 device=cpu plots=False`.
The dataset is CC BY-NC-SA 4.0 and is used for evaluation only. Result: see the end of this
file; the 22,481-image test split takes about an hour on 2 CPU cores.

## 7. YOLO11 fine-tuning on S2TLD (GPU, prepared)

Prepared and smoke-tested on CPU (1 epoch, 32 images, 640 px: pipeline runs end to end and
writes `metrics_test.json`; the numbers of a smoke run are meaningless and are not reported).

| Config | Split | Expected run | Compare with |
|---|---|---|---|
| `configs/s2tld/yolo11s_1280_random60.yaml` | random 6:4 (`random60_seed0.json`) | 60 epochs at 1280 px, batch 8; roughly 1 to 2 h on an RTX 4090 class GPU, longer on Apple MPS | SCRDet++ Table 4: RetinaNet 93.25, FPN 94.93 (+InLD 95.94) mAP@0.5 on their own random 6:4 split (same leakage) |
| `configs/s2tld/yolo11s_1280_chunked.yaml` | chunked 60/10/30 | same | OpenTSR benchmark number (no published reference; expect lower than the random split) |
| `configs/s2tld/yolo11n_1280_chunked.yaml` | chunked | about half the time | edge-size reference |

Command on the GPU machine (from the repository root, after `make s2tld`):

```bash
python scripts/train_yolo.py --config configs/s2tld/yolo11s_1280_random60.yaml   # device auto
python scripts/train_yolo.py --config configs/s2tld/yolo11s_1280_chunked.yaml
python scripts/train_yolo.py --config configs/s2tld/yolo11n_1280_chunked.yaml
```

Each run writes `runs/<name>/metrics_test.json` with mAP50 / mAP50-95 per class, the config,
the git commit and package versions; copy the table into this file and commit. Ultralytics
reports COCO-style 101-point interpolated AP; for the SCRDet++ comparison recompute VOC AP from
the saved predictions with `src/opentsr/metrics.py` (follow-up).

## 8. What was blocked and why

- TIER IV / AWML evaluation data (T4dataset) requires WebAuto access: the published Autoware
  numbers (99.81 % classifier accuracy, fine detector mAP 0.792 on crops) are not reproducible.
- Autoware's original artifact host `awf.ml.dev.web.auto` is not reachable from the sandbox;
  the Hugging Face republication under `AutowareFoundation` is byte-identical in size and carries
  pinned tags, so it was used instead.
- `github.com` HTML and `api.github.com` are not reachable from the sandbox (raw files and
  `git clone` are), which is why repository metadata in the surveys comes from badges.
- BSTLD (18 GB), DTLD (registration), ATLAS (30 GB) were not downloaded in this session; they
  are the next evaluation sets to add once the GPU machine runs the baselines.
