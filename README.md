# OpenTSR

Open-source, open-data traffic sign recognition.

## Goal

Build a traffic sign detection and classification model of commercial-grade quality, and release
everything in the open: code, data, and trained weights.

The approach:

- Start from open-source implementations and public datasets.
- Close the data gap, including Japanese road signs, with generation techniques such as
  vision-language models (VLMs) and world models.
- Evaluate against a fixed, public benchmark so that progress is measurable and reproducible.

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | Project-wide decisions and third-party asset list; traffic sign surveys and results |
| `configs/`, `scripts/` | Traffic sign configs (`configs/gtsrb/`) and entry points (data preparation, training, replications) |
| `src/opentsr/` | Shared code: `metrics.py` (VOC AP evaluator), `splits.py` (split manifests) |
| `src/opentsr/signs/` | Traffic sign package (main track): `datasets/` (GTSRB, GTSDB readers), `models/` (small CNNs) |
| `tests/` | Unit tests |
| `traffic_lights/`, `src/opentsr/traffic_lights/` | Traffic light track (secondary), see below |

## Traffic lights (secondary track)

The project began with traffic lights. That work (dataset and implementation surveys,
S2TLD benchmark splits, replication of Autoware and other open models) is kept, separated from
the sign work, under `traffic_lights/`. See [traffic_lights/README.md](traffic_lights/README.md).

## Status

October 2026: target changed from traffic lights to traffic signs. Survey and first replications
for signs are in (2026-10-31):

- `docs/datasets.md`: 40+ public sign datasets with licenses and mirrors, the Japan gap, synthetic
  and VLM-style sets, and a tiered benchmark proposal.
- `docs/oss_implementations.md`: open implementations from the 2011 GTSRB competition to 2026 Hub
  models, production stacks (Autoware, Apollo, openpilot forks), and ranked replication candidates.
- `docs/vlm_and_foundation.md`: CLIP-style zero-shot, large multimodal models, open-vocabulary
  detectors, and generative data for signs.
- `docs/requirements_and_robustness.md`: what "commercial-grade" means (EU ISA, Euro NCAP, Japan),
  robustness benchmarks, adversarial attack and defence code, and a draft target specification.
- `docs/replication.md`: measured numbers on the official GTSRB test split (from-scratch CNNs,
  released checkpoints, CLIP zero-shot, small VLMs), open-vocabulary detection on GTSDB, the
  Panoramax German classifier, and the GPU runs that remain.

## Quick start (signs)

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install -r requirements.txt

make gtsrb                      # GTSRB official split from the Hugging Face mirror (312 MB), verified
make gtsdb                      # GTSDB training package + test images (1.73 GB)
make replicate-gtsrb-cnn        # LeNet-5 and STN-CNN trained from scratch (minutes to ~45 min on 2 cores)
make replicate-gtsrb-released   # released checkpoints (poojahira, Hub models) on the official test split
make replicate-gtsrb-clip       # CLIP / SigLIP zero-shot with the OpenAI prompts
make replicate-gtsrb-vlm        # SmolVLM / Qwen2-VL multiple choice on a stratified subset
make replicate-gtsdb-openvocab  # OWLv2 and YOLO-World class-agnostic detection on GTSDB
make replicate-panoramax        # Panoramax German sign classifier on its CC BY-SA validation crops
make test                       # unit tests
```

## 日本語

道路標識を認識する機械学習モデルを、オープンソース・オープンデータとして開発するプロジェクトです。OSS 実装と公開データを土台に、VLM や世界モデルなどのデータ生成技術も活用し、商用レベルで使える品質を目指します。

2026-10-31 に標識の OSS 実装・公開データセット・VLM・要件の調査と CPU 再現実験を `docs/` に入れました。信号機認識の調査・再現実験の成果はおまけのトラックとして `traffic_lights/` 以下に分けて保持しています。
