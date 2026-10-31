# Third-party assets

Every dataset, model and codebase OpenTSR builds on or measures, with its license and source.
"Use in OpenTSR" states what we are allowed to do with it under the project's Apache-2.0
release policy (see `docs/decisions.md`). Licenses are quoted as written on the source page on
the date given; re-check before any redistribution.

This file covers assets shared by both tracks and the traffic sign track (main). Assets used only
by the traffic light track are listed in `traffic_lights/docs/third_party.md`.

## Traffic sign datasets

None downloaded yet. Add a row the first time a sign dataset is downloaded or measured.

## Shared datasets

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| COCO 2017 (through COCO-pretrained YOLO weights) | https://cocodataset.org | annotations "CC BY 4.0"; image rights belong to Flickr users | Pretraining through third-party weights only. The traffic light track also uses its "traffic light" class directly (`traffic_lights/docs/replication.md`). |

## Shared models and tooling

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| Ultralytics YOLO11 (code and COCO-pretrained weights `yolo11n.pt`, `yolo11s.pt`) | https://github.com/ultralytics/ultralytics, weights from the ultralytics/assets GitHub releases | "AGPL-3.0" (Ultralytics Enterprise License available commercially) | Baselines and replication only (both tracks). Weights fine-tuned with Ultralytics are not release candidates for OpenTSR until the licensing decision in `docs/decisions.md` is revisited. |

## Code we re-implemented rather than copied

| Asset | Source | License | What we took |
|---|---|---|---|
| PASCAL VOC devkit AP computation (as used by SCRDet++ for S2TLD and by most sign benchmarks) | https://host.robots.ox.ac.uk/pascal/VOC/ | n/a (algorithm) | Greedy matching and all-point / 11-point interpolation, re-implemented in `src/opentsr/metrics.py`. |

## Python dependencies

Pinned in `requirements.txt`. Notable licenses: PyTorch (BSD-3-Clause), onnxruntime (MIT), OpenCV
(Apache-2.0), NumPy (BSD-3-Clause), Ultralytics (AGPL-3.0, see above).
