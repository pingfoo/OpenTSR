# docs/

Project-wide documents and the traffic sign track (main). Traffic light documents are in
`traffic_lights/docs/`.

| File | What it holds |
|---|---|
| `decisions.md` | Dated decision log for the whole project (evaluator, licensing, where training runs, target change, repository layout, sign benchmarks). |
| `third_party.md` | Shared and traffic sign datasets, models and codebases, with license and source URL. |
| `datasets.md` | Survey of public traffic sign datasets (classification and detection, synthetic, VLM-style), licenses, access, the Japan gap, benchmark tiers. |
| `oss_implementations.md` | Survey of open-source traffic sign recognition code and weights from 2011 to 2026 (classic, CNN, YOLO-era, production stacks, Hub models), with replication candidates. |
| `vlm_and_foundation.md` | CLIP-style zero-shot, large multimodal models, open-vocabulary detectors, self-supervised backbones and generative data for signs. |
| `requirements_and_robustness.md` | What commercial-grade TSR means (EU ISA, Euro NCAP, Japan), robustness benchmarks, adversarial attack and defence code, draft target specification. |
| `replication.md` | What we replicated on signs, the exact protocol, the measured numbers, and the GPU runs that remain. |
| `dataset_catalog.md` | The generated, illustrated catalog of the representative sign datasets (class definitions next to seeded samples) and how to rebuild it. |
| `commercial_tsr.md` | Survey memo on production TSR systems: documented evidence (patents, supplier papers, product descriptions, black-box tests, SoC toolchains), the inferred detect-track-classify architecture, and what it implies for OpenTSR. |
| `benchmark.md` | Draft benchmark specification: three layers (crops, frames, sign-pass events in sequences), data and licenses per layer, metrics, robustness protocol, baselines to reimplement, reporting rules. |
| `system_design.md` | Draft system design: the switchable production-style pipeline (detector, tracker, memory, classifier, reader, relevance, fusion, decision, defense), taxonomy, deployment cost, adversarial evaluation at crop, frame and event level, claims, risks and phases. |

Conventions: English, no em dashes, every number maps to a config under `configs/` and a commit.
