# OpenTSR

Open-source, open-data traffic signal (traffic light) recognition.

## Goal

Build a traffic-light detection and state-recognition model of commercial-grade quality, and release everything in the open: code, data, and trained weights.

The approach:

- Start from open-source implementations and public datasets.
- Close the data gap with generation techniques such as vision-language models (VLMs) and world models.
- Evaluate against a fixed, public benchmark so that progress is measurable and reproducible.

## Status

Survey and first replication (October 2026):

- `docs/datasets.md`: survey of public traffic light datasets and the benchmark recommendation.
- `docs/oss_implementations.md`: survey of open-source implementations with a ranked replication list.
- `docs/replication.md`: measured numbers for the Autoware models and COCO-pretrained YOLO on S2TLD, a reproduced DetectionBench number on LISA, plus the GPU runs still to do.
- `configs/s2tld/splits/`: fixed, checked-in S2TLD split manifests.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install -r requirements.txt

make s2tld               # download S2TLD (1.4 GB) and build the split manifests + YOLO datasets
make autoware-weights    # Autoware TLR models from Hugging Face (Apache-2.0)
make replicate-autoware  # CPU evaluations reported in docs/replication.md
make replicate-coco      # COCO-pretrained YOLO11n localisation baseline on S2TLD
make replicate-lisa      # DetectionBench lisa-yolo26n on the LISA test split (5 GB download)
make smoke               # 1-epoch CPU check of the YOLO training pipeline
make train-s2tld         # GPU: YOLO11s baselines on both S2TLD splits
```

Layout: `src/opentsr/` (dataset readers, metrics, model wrappers), `scripts/` (entry points), `configs/` (every reported number maps to a config), `docs/` (surveys and decisions). `data/`, `weights/` and `runs/` are local only.

## 日本語

信号機を認識する機械学習モデルを、オープンソース・オープンデータとして開発するプロジェクトです。OSS 実装と公開データを土台に、VLM や世界モデルなどのデータ生成技術も活用し、商用レベルで使える品質を目指します。
