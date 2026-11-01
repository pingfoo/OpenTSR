# Traffic lights (secondary track)

OpenTSR started as a traffic light recognition project. On 2026-10-31 the main target changed to
traffic signs (see `docs/decisions.md`). The traffic light work is kept here as a secondary
track: it is maintained, but new effort goes to signs first.

Everything specific to traffic lights lives under a `traffic_lights` path segment:

| Path | Contents |
|---|---|
| `traffic_lights/docs/` | Dataset survey, OSS implementation survey, replication results, track decisions, third-party assets |
| `traffic_lights/configs/` | S2TLD split manifests and YOLO training configs |
| `traffic_lights/scripts/` | Data preparation, evaluation and training entry points |
| `traffic_lights/Makefile` | Targets, run from the repository root as `make lights-<target>` |
| `src/opentsr/traffic_lights/` | Python package: S2TLD reader, Autoware ONNX wrappers |
| `data/traffic_lights/`, `weights/traffic_lights/`, `runs/traffic_lights/` | Local files (gitignored) |

Shared code it relies on: `src/opentsr/metrics.py` (VOC AP) and `src/opentsr/splits.py` (split
manifests).

## Results so far

Measured on 2026-10-31 on CPU (details and protocol in `traffic_lights/docs/replication.md`):

| What | Result |
|---|---|
| Autoware state classifier (MobileNetV2 v4.0) on S2TLD test ground-truth crops | 96.95% accuracy |
| Autoware fine detector (YOLOX-s 416 v3.0) on simulated coarse ROIs | recall@IoU0.5 39.6% |
| Autoware whole-image detector (YOLOX-s 960 v1.0), class-agnostic | AP50 8.0% |
| COCO-pretrained YOLO11n (class 9, 1280 px), class-agnostic | AP50 65.9% |
| DetectionBench `lisa-yolo26n` on LISA test, reproduced | mAP@50 24.78 (published 23.74) |

Pending: YOLO11s/11n fine-tuning on S2TLD (GPU, `make lights-train-s2tld`).

## Quick start

From the repository root, with the virtualenv active:

```bash
pip install -r traffic_lights/requirements.txt   # the track has its own pins; signs use the root requirements.txt
make lights-s2tld               # download S2TLD (1.4 GB), build YOLO datasets from the checked-in splits
make lights-replicate-autoware  # CPU replication of the Autoware models
make lights-train-s2tld         # GPU training runs
```
