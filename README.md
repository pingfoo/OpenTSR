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
| `configs/`, `scripts/` | Traffic sign configs and entry points (created as they are needed) |
| `src/opentsr/` | Shared code: `metrics.py` (VOC AP evaluator), `splits.py` (split manifests) |
| `src/opentsr/signs/` | Traffic sign package (main track) |
| `tests/` | Unit tests |
| `traffic_lights/`, `src/opentsr/traffic_lights/` | Traffic light track (secondary), see below |

## Traffic lights (secondary track)

The project began with traffic lights. That work (dataset and implementation surveys,
S2TLD benchmark splits, replication of Autoware and other open models) is kept, separated from
the sign work, under `traffic_lights/`. See [traffic_lights/README.md](traffic_lights/README.md).

## Status

October 2026: target changed from traffic lights to traffic signs. Traffic sign survey not
started yet.

## 日本語

道路標識を認識する機械学習モデルを、オープンソース・オープンデータとして開発するプロジェクトです。OSS 実装と公開データを土台に、VLM や世界モデルなどのデータ生成技術も活用し、商用レベルで使える品質を目指します。

信号機認識の調査・再現実験の成果はおまけのトラックとして `traffic_lights/` 以下に分けて保持しています。
