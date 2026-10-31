# Decisions: traffic light track

Dated entries that only concern traffic lights. Project-wide decisions (evaluator, licensing,
where training runs, the change of main target) are in `docs/decisions.md`. Newest last.

## 2026-10-31: S2TLD is the first replication dataset and the seed of the fixed benchmark

Decided: use S2TLD (SJTU Small Traffic Light Dataset, 5,786 images, 14,130 boxes, five states)
as the first dataset for replication and as the first component of the fixed OpenTSR benchmark.
Two split manifests are checked in under `traffic_lights/configs/s2tld/splits/`:

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

## 2026-10-31: Autoware's released models are the reference production stack to measure first

Decided: the Autoware Foundation traffic light classifier (MobileNetV2, v4.0) and fine detector
(YOLOX-s, v3.0), republished on Hugging Face under Apache-2.0, are the first production-grade
open models we measure, and their preprocessing is re-implemented exactly in
`src/opentsr/traffic_lights/models/autoware_onnx.py` (letterbox, normalisation, YOLOX decode and NMS as in
`autoware_universe`).

Why: they are the only open, deployed, permissively licensed traffic light models with a public
accuracy claim (99.81% classification accuracy on TIER IV's internal Japanese test set). Their
training data is private, so the claim cannot be replicated directly, but cross-domain accuracy
on public data tells us how far a production classifier transfers and what a map-prior ROI
pipeline costs in recall.
