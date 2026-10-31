# Third-party assets

Every dataset, model and codebase OpenTSR builds on or measures, with its license and source.
"Use in OpenTSR" states what we are allowed to do with it under the project's Apache-2.0
release policy (see `docs/decisions.md`). Licenses are quoted as written on the source page on
2026-10-31; re-check before any redistribution.

## Datasets

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| S2TLD (SJTU Small Traffic Light Dataset) | https://github.com/Thinklab-SJTU/S2TLD, mirror by the author at https://huggingface.co/datasets/yangxue/S2TLD (zip checksums in `scripts/prepare_s2tld.py`) | GitHub `LICENSE`: MIT ("Copyright (c) 2018 DetectionTeamUCAS"); Hugging Face mirror tag `apache-2.0`; no separate data-license text | Benchmark seed and replication target. Split manifests under `configs/s2tld/splits/` are ours (Apache-2.0). Confirm with the authors that MIT covers images and labels before re-hosting the images. |
| TIER IV traffic light classifier sample dataset (1,045 crops, red/green/yellow) | https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz (linked from the Autoware classifier README), sha256 `bba8279d524f404b904da9cce23b67ae666b9e3ac23b05c472edea6943d7af1b` | No license file in the archive and none stated on the linking page: unknown | Evaluation only (`docs/replication.md`). Not redistributed, not used for training. |
| BSTLD (Bosch Small Traffic Lights Dataset) | https://zenodo.org/records/12706046 (DOI 10.5281/zenodo.12706046); mirror https://huggingface.co/datasets/shpotes/bosch-small-traffic-lights-dataset | "LICENSE FOR NONCOMMERCIAL PURPOSES" (Robert Bosch LLC, 2017-02-02); code in bosch-ros-pkg/bstld is MIT | Not downloaded yet. Pointer-based test set only; derived labels would inherit the NC license. |
| LISA Traffic Light Dataset | Kaggle mbornoe/lisa-traffic-light-dataset; mirror https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights | "CC BY-NC-SA 4.0" | Not downloaded yet. Pointer-based test set only. |
| DTLD (DriveU Traffic Light Dataset) | https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/ (registration) | Research and teaching only; "Any commercial use or the disclosure of data to third parties (without charge or paid) is prohibited." | Private validation only, numbers may be reported; nothing derived from it can be shared. |
| COCO 2017 (traffic light class, used through COCO-pretrained YOLO weights) | https://cocodataset.org | annotations "CC BY 4.0"; image rights belong to Flickr users | Pretraining through third-party weights only. |

All other datasets from the survey (`docs/datasets.md`) are listed there with their licenses; they
are added to this table when we first download them.

## Models and weights

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| Autoware traffic light classifier (MobileNetV2, car and pedestrian, lamp recognizer), tag `v4.0` | https://huggingface.co/AutowareFoundation/traffic_light_classifier (original artifact `awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4`), checksums in `scripts/download_autoware_weights.sh` | "Apache-2.0" (model card) | Reference production model; evaluated in `docs/replication.md`. Could be redistributed and fine-tuned. Training data (TIER IV internal) is not public. |
| Autoware traffic light fine detector (YOLOX-s), tag `v3.0` | https://huggingface.co/AutowareFoundation/traffic_light_fine_detector (original artifact `awf.ml.dev.web.auto/perception/models/tlr_yolox_s/v3/`) | "Apache-2.0" (model card) | Same as above. |
| Ultralytics YOLO11 (code and COCO-pretrained weights `yolo11n.pt`, `yolo11s.pt`) | https://github.com/ultralytics/ultralytics, weights from the ultralytics/assets GitHub releases | "AGPL-3.0" (Ultralytics Enterprise License available commercially) | Baselines and replication only. Weights fine-tuned with Ultralytics are not release candidates for OpenTSR until the licensing decision in `docs/decisions.md` is revisited. |

## Code we re-implemented rather than copied

| Asset | Source | License | What we took |
|---|---|---|---|
| Autoware Universe `autoware_tensorrt_classifier`, `autoware_tensorrt_yolox`, `autoware_traffic_light_classifier`, `autoware_traffic_light_fine_detector` | https://github.com/autowarefoundation/autoware_universe | Apache-2.0 | Preprocessing conventions (letterbox, normalisation, padding value), YOLOX output decoding and NMS semantics, label decoding. Re-implemented in Python in `src/opentsr/models/autoware_onnx.py`; no source copied. |
| PASCAL VOC devkit AP computation (as used by SCRDet++ for S2TLD) | https://host.robots.ox.ac.uk/pascal/VOC/ | n/a (algorithm) | Greedy matching and all-point / 11-point interpolation, re-implemented in `src/opentsr/metrics.py`. |

## Python dependencies

Pinned in `requirements.txt`. Notable licenses: PyTorch (BSD-3-Clause), onnxruntime (MIT), OpenCV
(Apache-2.0), NumPy (BSD-3-Clause), Ultralytics (AGPL-3.0, see above).
