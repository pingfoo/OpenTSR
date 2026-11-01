# Foundation models, VLMs, zero-shot / open-vocabulary methods and generative data for traffic sign recognition

Survey compiled on 2026-10-31 from live web pages (claude.ai research session, pingfoo/OpenTSR).
Scope: CLIP-style zero-shot and linear-probe results on GTSRB, large multimodal models (LMMs / VLMs)
evaluated on traffic signs, open-vocabulary and promptable detectors applied to signs, self-supervised
and large-scale pretraining (including model merging papers that report GTSRB), and generative data
synthesis for signs. Classic CNN and YOLO implementations are covered by a separate document and are not
repeated here. Every factual claim carries the URL on which it was verified. Where a value could not be
verified the field says "unknown" or "not verified". Where sources disagree both values are given.
Section G lists pages that could not be fetched.

Conventions:
- "GTSRB" = German Traffic Sign Recognition Benchmark, 43 classes, 26,640 train / 12,630 test crops as
  used by the CLIP evaluation suite (https://arxiv.org/pdf/2103.00020, Table 9).
- "zero-shot" = no GTSRB training images used; class names only (CLIP-style) or a free-form / multiple
  choice question (VLM-style).
- "linear probe" = logistic regression on frozen features; "FT" = full fine-tuning.
- Accuracies are top-1 in percent unless noted. Numbers copied from CSV files are rounded to one decimal.
- "NC" = non-commercial restriction.

Quick orientation (details below):

| Question | Short answer | Where |
|---|---|---|
| How good is CLIP zero-shot on GTSRB? | 32.2 (ViT-B/32) to 52.3 (ViT-L/14-336) in the CLIP paper; best downloadable model 67.8 (DFN-5B ViT-H/14-378, apple-amlr license) and best permissive one 67.4 (EVA02-E-14-plus, MIT) in the open_clip results table | A.1, A.3 |
| How good is a CLIP linear probe or fine-tune on GTSRB? | linear probe 85.3 (ViT-B/32) to 92.5 (ViT-L/14); full FT 98.5 (B/32), 99.0 (B/16), 99.3 (L/14) | A.1, D.2 |
| Do VLMs beat specialised classifiers on fine-grained signs today? | No. Open VLMs reach 0.32 to 0.68 top-1 zero-shot on GTSRB-style tasks; GPT-4o 0.89; DINOv2 feature matching beats InternVL-3 and Gemma-3 on the MVV 11-class sign task; supervised CNNs remain at 99+ on GTSRB | B.1, B.3, F.2 |
| Which permissive foundation components are strong enough to build on? | SigLIP / SigLIP 2 (Apache-2.0), LAION / DataComp open_clip weights (MIT-style), DINOv2 (Apache-2.0), OWLv2 and Grounding DINO (Apache-2.0), Florence-2 (MIT), SAM / SAM 2 (Apache-2.0), Qwen2-VL / Qwen3-VL (Apache-2.0) | F.2 |
| Is there a measured downstream TSR gain from synthetic data? | Yes: Synset Signset Germany (CC BY 4.0) trained alone gives 98.3 on real GTSRB test; AMAP diffusion inpainting raises rare-class AP50 on TT100K by 1.23x to 7.40x | E.1, E.2 |

---

## A. CLIP-style zero-shot and linear-probe results on GTSRB

### A.1 OpenAI CLIP paper (Radford et al., 2021, arXiv 2103.00020)

- Dataset as evaluated: GTSRB, 43 classes, 26,640 train, 12,630 test, metric accuracy
  (https://arxiv.org/pdf/2103.00020, Table 9, page 38 of the PDF).
- Zero-shot accuracy, Table 11 ("Zero-shot performance of CLIP models over 27 datasets"), GTSRB column
  (https://arxiv.org/pdf/2103.00020, page 43):

| Backbone | RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14-336px |
|---|---|---|---|---|---|---|---|---|---|
| GTSRB zero-shot | 35.2 | 38.3 | 35.7 | 39.6 | 47.9 | 32.2 | 43.3 | 50.3 | 52.3 |

- Linear probe accuracy, Table 10 ("Linear probe performance of various pre-trained models over 27
  datasets"), GTSRB column (https://arxiv.org/pdf/2103.00020, page 40):

| Model | GTSRB linear probe |
|---|---|
| CLIP RN50 / RN101 / RN50x4 / RN50x16 / RN50x64 | 82.4 / 82.4 / 85.5 / 89.0 / 90.2 |
| CLIP ViT-B/32 / ViT-B/16 / ViT-L/14 / ViT-L/14-336px | 85.3 / 86.6 / 92.5 / 92.4 |
| LM RN50 (language model baseline) | 73.8 |
| EfficientNet B0 to B8 (ImageNet) | 80.0, 75.9, 76.4, 75.8, 78.3, 78.5, 78.1, 80.8, 80.4 |
| EfficientNet Noisy Student L2-475 / L2-800 | 73.4 / 77.7 |
| Instagram ResNeXt 32x48d / FixRes-v2 | 73.0 / 76.3 |
| BiT-M R152x4 (ImageNet-21k) | 68.2 |
| ViT (ImageNet-21k) B/32 / B/16 / L/16 / H/14 | 68.1 / 63.1 / 70.4 / 73.9 |
| SimCLRv2 R152x3 | 78.4 |
| BYOL 200x2 / MoCo v2 / VirTex | 77.1 / 75.7 / 69.5 |
| ResNet-50 / 101 / 152 (ImageNet supervised) | 70.6 / 72.5 / 70.1 |

- Context given in the paper: zero-shot CLIP is described as "quite weak on several specialized, complex,
  or abstract tasks such as ... self-driving related tasks such as German traffic sign recognition
  (GTSRB)" and the authors note that "non-expert humans can robustly perform several of these tasks, such
  as ... traffic sign recognition" (https://arxiv.org/pdf/2103.00020, Section 3.1.5). Figure 5 shows
  zero-shot CLIP 18.4 points below a linear probe on ImageNet ResNet-50 features for GTSRB
  (https://arxiv.org/pdf/2103.00020, Figure 5). Figure 7 states that a linear probe on CLIP features needs
  only 1.6 labelled examples per class to match zero-shot accuracy on GTSRB (among the lowest values in the
  figure; only CLEVRCounts 1.5, RESISC45 1.5, EuroSAT 0.9 and Flowers102 0.9 are lower, versus a median of
  5.4), i.e. zero-shot is barely better than a handful of labels
  (https://arxiv.org/pdf/2103.00020, Figure 7). In the linear-probe comparison the best CLIP model beats
  the Noisy Student EfficientNet-L2 by +14.7 on GTSRB, which the authors attribute to ImageNet-1k having
  "only a single label for all traffic and street signs" (https://arxiv.org/pdf/2103.00020, Figure 11 and
  Section 3.2). Table 12 (ResNet-50 trained on YFCC100M vs a same-size WIT subset): linear probe GTSRB
  66.8 vs 72.5, zero-shot 6.9 vs 7.0 (https://arxiv.org/pdf/2103.00020, Table 12).
- Prompt engineering notes relevant to signs: "GTSRB, don't appear to include this mapping [class id to
  name] at all in their released versions", so the authors wrote their own class descriptions; for OCR
  datasets "putting quotes around the text or number to be recognized improved performance"
  (https://arxiv.org/pdf/2103.00020, Section 3.1.4).
- Code and weights: MIT license (https://img.shields.io/github/license/openai/CLIP.json, value "MIT").

### A.2 Official GTSRB class names and templates (openai/CLIP, data/prompts.md)

Source: https://raw.githubusercontent.com/openai/CLIP/main/data/prompts.md (section "GTSRB"). The same
43 strings are used by CLIP_benchmark (https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/classnames.txt),
by fusion_bench (https://raw.githubusercontent.com/tanganke/fusion_bench/main/fusion_bench/tasks/clip_classification/gtsrb.py)
and as label names in the tanganke/gtsrb Hugging Face dataset (https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md).

Templates (3): `a zoomed in photo of a "{}" traffic sign.`, `a centered photo of a "{}" traffic sign.`,
`a close up photo of a "{}" traffic sign.`

Class names, in GTSRB label order 0 to 42:
`red and white circle 20 kph speed limit`, `red and white circle 30 kph speed limit`,
`red and white circle 50 kph speed limit`, `red and white circle 60 kph speed limit`,
`red and white circle 70 kph speed limit`, `red and white circle 80 kph speed limit`,
`end / de-restriction of 80 kph speed limit`, `red and white circle 100 kph speed limit`,
`red and white circle 120 kph speed limit`, `red and white circle red car and black car no passing`,
`red and white circle red truck and black car no passing`, `red and white triangle road intersection warning`,
`white and yellow diamond priority road`, `red and white upside down triangle yield right-of-way`, `stop`,
`empty red and white circle`, `red and white circle no truck entry`,
`red circle with white horizonal stripe no entry` (sic), `red and white triangle with exclamation mark warning`,
`red and white triangle with black left curve approaching warning`,
`red and white triangle with black right curve approaching warning`,
`red and white triangle with black double curve approaching warning`,
`red and white triangle rough / bumpy road warning`, `red and white triangle car skidding / slipping warning`,
`red and white triangle with merging / narrow lanes warning`,
`red and white triangle with person digging / construction / road work warning`,
`red and white triangle with traffic light approaching warning`,
`red and white triangle with person walking warning`,
`red and white triangle with child and person walking warning`, `red and white triangle with bicyle warning` (sic),
`red and white triangle with snowflake / ice warning`, `red and white triangle with deer warning`,
`white circle with gray strike bar no speed limit`, `blue circle with white right turn arrow mandatory`,
`blue circle with white left turn arrow mandatory`, `blue circle with white forward arrow mandatory`,
`blue circle with white forward or right turn arrow mandatory`,
`blue circle with white forward or left turn arrow mandatory`, `blue circle with white keep right arrow mandatory`,
`blue circle with white keep left arrow mandatory`, `blue circle with white arrows indicating a traffic circle`,
`white circle with gray strike bar indicating no passing for cars has ended`,
`white circle with gray strike bar indicating no passing for trucks has ended`
(https://raw.githubusercontent.com/openai/CLIP/main/data/prompts.md).

Note: 9 of the 43 names differ only by the number (speed limits 20 to 120), and 3 "white circle with gray
strike bar" classes differ only in the trailing clause, so the text side of the classifier is nearly
degenerate for a third of the classes.

### A.3 CLIP_benchmark / open_clip results (LAION, mlfoundations)

- CLIP_benchmark (https://github.com/LAION-AI/CLIP_benchmark, MIT per
  https://img.shields.io/github/license/LAION-AI/CLIP_benchmark.json, 817 stars per
  https://img.shields.io/github/stars/LAION-AI/CLIP_benchmark.json) evaluates GTSRB from the webdataset
  `clip-benchmark/wds_gtsrb` on Hugging Face: files `classnames.txt`, `zeroshot_classification_templates.txt`,
  `train/0..5.tar`, `test/0..2.tar`; last modified 2023-01-20; no license metadata on the card (cardData
  null) (https://huggingface.co/api/datasets/clip-benchmark/wds_gtsrb?full=true). Test shard 0 is
  41,308,160 bytes (https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/test/0.tar,
  x-linked-size header). Templates use `{c}` instead of `{}` but are otherwise the OpenAI templates
  (https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/zeroshot_classification_templates.txt).
  The README points to the open_clip results CSV for 38-dataset results
  (https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/README.md).
- The open_clip results CSV (https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv,
  121 model rows, column "GTSRB") gives zero-shot GTSRB accuracy with the CLIP_benchmark protocol. Selected
  rows (all values from that file; fraction converted to percent):

| open_clip model | pretrained tag | GTSRB zero-shot | ImageNet-1k | Weights license (HF card) |
|---|---|---|---|---|
| ViT-H-14-378-quickgelu | dfn5b | 67.8 | 84.4 | apple-amlr (https://huggingface.co/api/models/apple/DFN5B-CLIP-ViT-H-14-378) |
| ViT-H-14-quickgelu | dfn5b | 67.7 | 83.4 | apple-amlr |
| EVA02-E-14-plus | laion2b_s9b_b144k | 67.4 | 82.0 | MIT (https://huggingface.co/api/models/QuanSun/EVA-CLIP) |
| EVA02-E-14 | laion2b_s4b_b115k | 67.3 | 82.0 | MIT |
| EVA01-g-14-plus | merged2b_s11b_b114k | 66.4 | 79.3 | MIT |
| ViT-bigG-14-CLIPA | datacomp1b | 64.6 | 82.7 | not verified |
| ViT-SO400M-14-SigLIP-384 | webli | 64.3 | 83.1 | Apache-2.0 for the google SigLIP cards (https://huggingface.co/api/models/google/siglip-base-patch16-224) |
| ViT-L-16-SigLIP-384 | webli | 63.9 | 82.1 | Apache-2.0 |
| ViT-SO400M-14-SigLIP | webli | 63.8 | 82.0 | Apache-2.0 |
| ViT-H-14-quickgelu | metaclip_fullcc | 62.6 | 80.5 | cc-by-nc-4.0 (https://huggingface.co/api/models/facebook/metaclip-h14-fullcc2.5b) |
| ViT-bigG-14 | laion2b_s39b_b160k | 62.4 | 80.1 | MIT (laion cards, e.g. https://huggingface.co/api/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| ViT-L-14-quickgelu | dfn2b | 61.7 | 81.4 | apple-amlr (https://huggingface.co/api/models/apple/DFN2B-CLIP-ViT-L-14) |
| ViT-L-16-SigLIP-256 | webli | 61.2 | 80.5 | Apache-2.0 |
| ViT-L-14 | datacomp_xl_s13b_b90k | 58.5 | 79.2 | not verified |
| ViT-H-14 | laion2b_s32b_b79k | 58.3 | 78.0 | MIT |
| convnext_large_d_320 | laion2b_s29b_b131k_ft_soup | 58.2 | 76.9 | not verified |
| ViT-L-14 | laion2b_s32b_b82k | 56.2 | 75.3 | MIT |
| ViT-B-16 | datacomp_xl_s13b_b90k | 55.2 | 73.5 | not verified |
| ViT-L-14-336 | openai | 52.4 | 76.6 | MIT (github) |
| ViT-B-16-SigLIP-384 | webli | 52.0 | 78.5 | Apache-2.0 |
| ViT-B-32 | datacomp_xl_s13b_b90k | 51.9 | 69.2 | not verified |
| ViT-B-16-SigLIP | webli | 51.6 | 76.0 | Apache-2.0 |
| ViT-L-14 | openai | 50.6 | 75.5 | MIT (github) |
| ViT-B-32 | laion2b_s34b_b79k | 49.3 | 66.6 | MIT (https://huggingface.co/api/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) |
| ViT-B-16 | laion2b_s34b_b88k | 48.3 | 70.2 | MIT |
| RN50x64 | openai | 47.9 | 73.9 | MIT (github) |
| ViT-B-16 | dfn2b | 47.0 | 76.2 | apple-amlr |
| ViT-B-16 | openai | 43.4 | 68.3 | MIT (github) |
| ViT-B-32 | laion400m_e32 | 40.7 | 60.2 | MIT |
| RN50x16 / RN101 / RN50x4 / RN50 | openai | 39.8 / 37.5 / 36.2 / 35.1 | 70.7 / 62.3 / 66.3 / 59.8 | MIT (github) |
| ViT-B-32 | openai | 32.6 | 63.3 | MIT (github) |
| ViT-B-32 | commonpool_s_* (13M samples) | 4.1 to 7.3 | 3 to 5 | not verified |

  (all GTSRB and ImageNet values: https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv).
  Observations: (1) the two OpenAI protocols agree within 0.4 points (32.2 vs 32.6 for ViT-B/32, 50.3 vs
  50.6 for ViT-L/14); (2) no open CLIP-style model exceeds 68 percent zero-shot on GTSRB even at 1 to 5
  billion parameters, versus 99+ for supervised CNNs; (3) data curation matters more than size: DFN-5B
  ViT-H (67.8) beats LAION-2B ViT-bigG (62.4).
- open_clip code license: an MIT-style permission notice naming the authors
  (https://raw.githubusercontent.com/mlfoundations/open_clip/main/LICENSE; GitHub cannot auto-identify it,
  https://img.shields.io/github/license/mlfoundations/open_clip.json returns "not identifiable by github").

### A.4 Other foundation-model papers that report GTSRB

- Reproducible scaling laws for open_clip (Cherti et al., arXiv 2212.07143): Table 6 (zero-shot),
  Table 7 (joint fine-tuning of the vision encoder on 8 tasks with a frozen zero-shot head) and Table 8
  (joint patching). GTSRB column (https://arxiv.org/pdf/2212.07143, Appendix tables):

| Model (samples seen, data) | Zero-shot | Fine-tuned (joint) | Patched |
|---|---|---|---|
| ViT-B/32 (13B, CLIP-WIT) | 32.6 | 98.2 | 95.5 |
| ViT-B/32 (13B, LAION-400M) | 42.0 | 98.4 | 94.0 |
| ViT-B/32 (34B, LAION-2B) | 36.5 | 98.0 | 95.6 |
| ViT-B/16 (13B, CLIP-WIT) | 43.3 | 98.8 | 97.4 |
| ViT-B/16 (13B, LAION-400M) | 43.5 | 98.8 | 97.2 |
| ViT-L/14 (13B, CLIP-WIT) | 50.6 | 99.2 | 99.0 |
| ViT-L/14 (13B, LAION-400M) | 49.9 | 99.0 | 98.9 |
| ViT-L/14 (34B, LAION-2B) | 56.1 | 98.9 | 98.7 |
| ViT-H/14 (34B, LAION-2B) | 58.4 | 99.0 | 98.8 |
| ViT-g/14 (13B, LAION-2B) | 49.9 | 99.0 | 98.8 |

  The paper's own zero-shot table for LAION-2B models (page 34) lists GTSRB 49.5 (B/32, 34B), 48.2 (B/16),
  56.1 (L/14), 49.7 (g/14, 13B), 58.5 (H/14) (https://arxiv.org/pdf/2212.07143). The 36.5 vs 49.5 values
  for ViT-B/32 LAION-2B are two different checkpoints (laion2b_e16 = 36.6 and laion2b_s34b_b79k = 49.3 in
  the open_clip CSV, https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv).
- MetaCLIP (Xu et al., arXiv 2309.16671), Table 4 (MetaCLIP-400M vs CLIP vs OpenCLIP-LAION-400M, same
  training setup) and the scaling table, GTSRB column (https://arxiv.org/pdf/2309.16671, page 8):

| Backbone | CLIP (WIT-400M, their eval) | OpenCLIP (LAION-400M) | MetaCLIP 400M | MetaCLIP 1B | MetaCLIP 2.5B |
|---|---|---|---|---|---|
| ViT-B/32 | 32.6 | 42.0 | 35.8 | 40.7 | 39.2 |
| ViT-B/16 | 43.3 | 43.5 | 43.8 | 49.5 | 46.6 |
| ViT-L/14 | 50.6 | 49.9 | 47.5 | 58.7 | 56.3 |
| ViT-H/14 | n/a | n/a | n/a | n/a | 62.5 |

  MetaCLIP weights are CC-BY-NC 4.0 on Hugging Face (https://huggingface.co/api/models/facebook/metaclip-h14-fullcc2.5b),
  so they are not usable for OpenTSR's release.
- EVA-CLIP (Sun et al., arXiv 2303.15389), Table 2 zero-shot over 27 datasets, GTSRB column
  (https://arxiv.org/pdf/2303.15389, page 3): OpenAI CLIP-B/16 43.5, OpenCLIP-B/16 48.3, EVA-02-CLIP-B/16
  46.3; OpenAI CLIP-L/14 50.6 (and 52.4 for the 336 px variant), OpenCLIP-L/14 51.6, EVA-02-CLIP-L/14 57.1
  (57.6 for the 336 px variant); OpenCLIP-H/14 58.4; EVA-01-CLIP-g/14 49.1 and 66.5 (the second is the
  "plus" data variant); OpenCLIP-G/14 62.5; EVA-02-CLIP-E/14 67.1 and 67.7. EVA-CLIP weights are MIT on
  Hugging Face (https://huggingface.co/api/models/QuanSun/EVA-CLIP, https://huggingface.co/api/models/timm/eva02_large_patch14_clip_224.merged2b).
- DINOv3 (Simeoni et al., arXiv 2508.10104), Appendix B.4 "Classification on OCR-Heavy Datasets",
  Table 25, frozen-backbone classification at 512 px input: GTSRB DINOv2 ViT-g 78.2, PE-core ViT-G 94.8,
  DINOv3-7B 87.5 (https://arxiv.org/pdf/2508.10104, page 57). The paper calls these "notoriously hard
  datasets for SSL" and the gap to the weakly supervised PE-core model is 7.3 points. DINOv3 weights are
  under a custom "dinov3-license" with manual gating on Hugging Face
  (https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m); the GitHub license is
  "not identifiable by github" (https://img.shields.io/github/license/facebookresearch/dinov3.json).
  DINOv2 is Apache-2.0 (https://img.shields.io/github/license/facebookresearch/dinov2.json,
  https://huggingface.co/api/models/facebook/dinov2-base).
- Not reported: the SigLIP paper (https://arxiv.org/pdf/2303.15343), SigLIP 2 paper
  (https://arxiv.org/pdf/2502.14786), Data Filtering Networks paper (https://arxiv.org/pdf/2309.17425),
  AIMv2 paper (https://arxiv.org/pdf/2411.14402) and DINOv2 paper (https://arxiv.org/pdf/2304.07193)
  contain no GTSRB number (grep of the full PDF text, 0 hits each). SigLIP (v1) and DFN numbers exist only
  via the open_clip CSV above; no SigLIP 2 GTSRB number was found anywhere (the open_clip CSV has no SigLIP 2
  rows, https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv). The
  DataComp paper lists GTSRB (12,630 test images, 43 classes) as one of its 38 evaluation tasks but only
  plots it (https://arxiv.org/pdf/2304.14108, Table on page 41 and figure on page 56).

### A.5 The "cannot read the number" failure mode: what the sources actually say

- No fetched source isolates speed-limit digit confusion for CLIP with a per-class table. The evidence is
  indirect: (1) 9 of the 43 GTSRB classes differ only in the digits of the speed limit (A.2), and zero-shot
  accuracy stays between 32 and 68 while linear probes on the same features reach 85 to 93 (A.1, A.3),
  which shows the information is in the image embedding but is not aligned with the digit strings in the
  text embedding; (2) the CLIP paper reports that for OCR-type datasets "putting quotes around the text or
  number to be recognized improved performance" (https://arxiv.org/pdf/2103.00020, Section 3.1.4), which is
  why the GTSRB templates quote the class name; (3) the DINOv3 authors classify GTSRB as an "OCR-heavy"
  dataset that is "notoriously hard" for self-supervised features (https://arxiv.org/pdf/2508.10104,
  Appendix B.4); (4) on the generation side, the AMAP inpainting paper reports that general-purpose
  diffusion inpainting "distort[s] digits" on sign faces (https://arxiv.org/pdf/2609.02348, abstract);
  (5) a small 28-sample Indian textbook sign test found CLIP accuracy moving from 0.39 (bare label words)
  to 0.46 (appending "sign") to 0.54 (appending "symbol"), showing the strong prompt dependence
  (https://arxiv.org/pdf/2511.13909, Table 1 and discussion). A per-class confusion analysis of CLIP on
  GTSRB speed limits is proposed as replication experiment 1 in F.3.

### A.6 Prompt engineering and template / prototype based zero-shot TSR

- Prototypical Priors (Jetley et al., BMVC 2015, arXiv 1512.01192): uses the official pictogram template
  of each sign as side information; on GTSRB ("D1", 43 classes) classification improved from 97.48 to
  97.98 with prototypes, and in a zero-shot split of 33 seen / 10 unseen classes the method beats the
  attribute-based baseline by 5.48 points on average (absolute values only in figures)
  (https://arxiv.org/pdf/1512.01192, Table 2 and Section 5.2.2).
- Few-shot TSR with clustering inductive bias and random neural network (Zhou et al., Pattern Recognition
  2020) and zero-shot TSR by midlevel feature matching (Gan et al., Sensors 2023) are the two
  template-matching baselines used by the Hokkaido group; in the CdMT comparison Gan et al. 2023 reaches
  top-1 0.56 (GTSRB subset), 0.67 (BTSD), 0.12 (TT-100K), 0.42 (Sapporo), 0.27 (Yokohama)
  (https://arxiv.org/pdf/2409.01534, Table 1 and references [22], [23]).
- CLIP with prompt learning on signs: CoOp 0.32, MaPLe 0.28, CLIP-Adapter 0.37 top-1 on the GTSRB subset
  versus plain CLIP ViT-B/32 0.24 (same subset, https://arxiv.org/pdf/2409.01534, Table 1). TSCLIP
  (ICRA 2025, arXiv 2409.15077) introduces a "traffic sign prompt engineering scheme" (scenario plus
  traffic-rule description per class) and an adaptive dynamic weight ensemble with the zero-shot weights;
  CLIP zero-shot on the German (GTSRB-derived, 31 classes) part of its CRTS dataset is 27.75 percent,
  linear probe 30.56, full FT 84.58, Wise-FT 85.54, TSCLIP 87.08 when training on China plus Slovenia
  (https://arxiv.org/pdf/2409.15077, Tables I and II). The CLIP backbone used is not stated in the text
  we could parse (not verified). Code "will be released soon" and the CRTS dataset (over 190,000 images,
  46 unified categories, 10 regions) is a Google Drive link (https://raw.githubusercontent.com/guoyangzhao/TSCLIP/main/README.md);
  repo license MIT (https://img.shields.io/github/license/guoyangzhao/TSCLIP.json).
- LMM prompt engineering (context / characteristic / differential descriptions generated from pictogram
  templates) is the core of CdMT and of the few-shot in-context learning paper, see B.1 and B.2.

---

## B. Large multimodal models evaluated on traffic signs

### B.1 Cross-domain Multi-step Thinking, CdMT (Gan et al., Hokkaido University; arXiv 2409.01534)

- Title history: v1 (2024) "Think Twice Before Recognizing: Large Multimodal Models for General
  Fine-grained Traffic Sign Recognition" (http://export.arxiv.org/api/query?id_list=2409.01534v1); v2
  (23 Jul 2025) "Cross-domain Multi-step Thinking: Zero-shot Fine-grained Traffic Sign Recognition in the
  Wild" (https://arxiv.org/pdf/2409.01534).
- Method: traffic sign extraction (ViT-Adapter detector, SAM 2 compared as an alternative extractor),
  then three LMM "thinking" steps: context descriptions with centre-coordinate prompt optimisation and
  "prior traffic sign hypotheses", characteristic descriptions generated by in-context learning from clean
  pictogram templates, and differential descriptions between similar signs. No training
  (https://arxiv.org/pdf/2409.01534, abstract and Section 3).
- Datasets: GTSRB, BTSD (Belgium) and TT-100K public benchmarks, plus two private Japanese sets, Sapporo
  and Yokohama urban road datasets. Because of API rate limits the authors "randomly used the subsets of
  GTSRB, BTSD, and TT-100K validation data" with stratified sampling and all classes kept; the subset size
  is not stated in v2 (not verified; the companion paper B.2 used 100 GTSRB and 125 BTSD images)
  (https://arxiv.org/pdf/2409.01534, Section 4.1). Metric: top-k accuracy averaged over five trials.
- Table 1 top-1 (top-3) results (https://arxiv.org/pdf/2409.01534, Table 1):

| Method | GTSRB | BTSD | TT-100K | Sapporo | Yokohama |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 0.24 (0.35) | 0.20 | 0.29 | 0.27 | 0.14 |
| EVA-02 | 0.41 | 0.30 | 0.32 | 0.48 | 0.29 |
| LLaVA-1.5 | 0.32 (0.45) | 0.28 | 0.13 | 0.09 | 0.11 |
| LLaVA-NeXT | 0.39 (0.48) | 0.31 | 0.20 | 0.10 | 0.13 |
| VITA-1.5 | 0.45 | 0.39 | 0.25 | 0.18 | 0.20 |
| GPT-4V | 0.81 (0.85) | 0.70 | 0.72 | 0.32 | 0.22 |
| GPT-4o | 0.89 (0.89) | 0.83 | 0.74 | 0.57 | 0.49 |
| CdMT-LLaVA-1.5 | 0.55 | 0.48 | 0.45 | 0.35 | 0.36 |
| CdMT-LLaVA-NeXT | 0.60 | 0.51 | 0.48 | 0.37 | 0.41 |
| CdMT-VITA-1.5 | 0.68 | 0.65 | 0.52 | 0.45 | 0.61 |
| CdMT-GPT-4V | 0.91 | 0.89 | 0.90 | 0.77 | 0.83 |
| CdMT-GPT-4o | 0.93 (0.97) | 0.88 | 0.97 | 0.89 | 0.85 |

  Supervised models evaluated cross-domain without GTSRB training (DenseNet-121 0.31, EfficientNet-B0
  0.52, Li et al. 0.75, MobileViT 0.05, Swin V2 0.14, MAE 0.20, DeiT 0.27) are also in Table 1. Inference
  cost per sign: LLaVA-1.5 6.4 s, LLaVA-NeXT 6.0 s, GPT-4V 2.0 s, GPT-4o 1.6 s (https://arxiv.org/pdf/2409.01534, Table 6).
- Why it matters for OpenTSR: the only paper found that evaluates LMMs on Japanese signs; open LMMs
  without the multi-step prompting are at or below CLIP linear-probe level; the GPT-4o numbers are on
  small subsets. No code or data release was found in the paper (not verified).

### B.2 Cross-domain few-shot in-context learning (Gan et al., arXiv 2407.05814)

- Same group, GPT-4V and GPT-4o only, description texts generated from template signs. Subsets: 100 GTSRB
  images (43 classes), 125 BTSD images (62 classes), Sapporo 68 images (18 sign types), Yokohama 33 images
  (13 types) (https://arxiv.org/pdf/2407.05814, Section 4). Top-1 / top-3 / top-5
  (https://arxiv.org/pdf/2407.05814, Tables 1 and 2):

| Method | GTSRB | BTSD | Sapporo | Yokohama |
|---|---|---|---|---|
| Baseline GPT-4V (image only) | 0.80 / 0.86 / 0.87 | 0.61 / 0.70 / 0.75 | 0.28 / 0.43 / 0.54 | 0.46 / 0.70 / 0.76 |
| Ours GPT-4V | 0.88 / 0.97 / 1.00 | 0.91 / 0.98 / 1.00 | 0.75 / 0.84 / 0.91 | 0.88 / 0.94 / 1.00 |
| Baseline GPT-4o | 0.88 / 0.89 / 0.89 | 0.79 / 0.89 / 0.89 | 0.46 / 0.66 / 0.80 | 0.65 / 0.88 / 0.88 |
| Ours GPT-4o | 0.89 / 0.98 / 0.98 | 0.91 / 0.99 / 0.99 | 0.82 / 0.99 / 1.00 | 0.93 / 1.00 / 1.00 |

### B.3 Mapillary Vistas Validation for Traffic Signs, MVV (Garg and Aich, NEC Labs America; arXiv 2508.02047, ICCVW 2025)

- Dataset: 2,000 Mapillary Vistas validation images re-labelled by expert annotators into 11 sign
  categories (stop, speed limit, yield, do not enter, crosswalk, parking, no parking, roundabout, turn
  restriction, cycle lane, others), reusing the original instance masks, so boxes plus masks are
  available (https://arxiv.org/pdf/2508.02047, Sections 1 and 3). The repository README lists the same 11
  names (while saying "12 categories") (https://raw.githubusercontent.com/nec-labs-ma/relabeling/main/README.md).
  License: GitHub reports "not identifiable by github" (https://img.shields.io/github/license/nec-labs-ma/relabeling.json);
  4 stars (https://img.shields.io/github/stars/nec-labs-ma/relabeling.json). The underlying Mapillary
  Vistas images carry Mapillary's own research license (not re-verified here).
- Protocol: classification of each sign instance given its coarse label, three input modes (cropped
  image, full image with red bounding box, SAM-refined crop), zero-shot VLMs with default settings, runs
  repeated twice, accuracy metric; DINOv2 nearest-neighbour matching against a feature bank built from
  external datasets (Mapillary and BDD excluded) (https://arxiv.org/pdf/2508.02047, Sections 4.2, 4.3).
- Results on the traffic sign task T3 (https://arxiv.org/pdf/2508.02047, Table 3): InternVL-3 1B 0.085
  (crop), 2B 0.323, 8B 0.453, 9B 0.468 (best VLM, crop), 14B 0.408; Gemma-3 4B 0.276, 12B 0.428;
  supervised ResNet-50 0.275, ResNet-101 0.310; DINOv2 feature matching 0.484 (best). On vehicles (T1)
  DINOv2 reaches 0.935 vs best VLM 0.768, so the gap is general but signs are the hardest task.
- Takeaway: on an 11-way sign task with real street-level crops, zero-shot open VLMs up to 14B stay below
  0.5 accuracy, and a language-free DINOv2 prototype matcher is the strongest zero-shot method.

### B.4 TS-1M (Zhao et al., HKUST(GZ); arXiv 2603.23034, March 2026)

- Dataset: "TS-1M", 454 standardised categories aggregated from public datasets plus web images; paper
  Table I gives 1,033,947 train and 233,611 test crops (https://arxiv.org/pdf/2603.23034, Table I); the
  project page says "approximately 1.26 million images ... about 1.06M training images and 200K test
  images" (https://guoyangzhao.github.io/projects/ts1m). Download via Baidu, Google Drive and Kaggle,
  "publicly available for research and academic use"; no dataset license text found (the page footer
  licenses the website under CC BY-SA 4.0) (https://guoyangzhao.github.io/projects/ts1m,
  https://raw.githubusercontent.com/guoyangzhao/TS1M-Traffic-Sign/main/README.md). Code repo MIT, 28 stars
  (https://img.shields.io/github/license/guoyangzhao/TS1M-Traffic-Sign.json, https://img.shields.io/github/stars/guoyangzhao/TS1M-Traffic-Sign.json).
  Because TS-1M is built from GTSRB, TT100K, MTSD, RTSD, DFG, ARTS and others, the source licenses
  (several NC) propagate; treat as research-only.
- Protocol: all models are trained on TS-1M with a classification head (the VLMs are used as encoders:
  CLIP ViT-B/32 dual encoder, BLIP, BLIP-2, LLaVA-1.5-7B with a linear head, "without invoking multi-step
  reasoning") (https://arxiv.org/pdf/2603.23034, Sections IV.C, IV.D, Table III).
- Table IV overall top-1 / F1 (https://arxiv.org/pdf/2603.23034): ResNet50 0.9260 / 0.8570, ConvNeXt
  0.9428 / 0.9043, DINO ViT-B/16 0.9190 / 0.8595, MoCoV3 0.9385, CLIP 0.9533 / 0.9113 (best), BLIP 0.9448,
  BLIP-2 0.9432, LLaVA 0.9406 / 0.8776 (7,062.9 M parameters, 6,216.7 GFLOPs). Table V cross-region
  accuracy: CLIP GTSRB 0.9996, TT100K 0.9898, MTSD 0.9118, average 0.9578 vs ResNet50 0.9150. Table VI
  rare classes: CLIP 0.9291 vs ResNet50 baseline (+9.10 percent relative).
- Takeaway: with supervision, a CLIP ViT-B/32 encoder is the best backbone in their benchmark, and a 7B
  LLaVA used as a classifier (0.9406) is below a ConvNeXt (0.9428) that has 140x fewer parameters and about
  700x fewer FLOPs (https://arxiv.org/pdf/2603.23034, Table IV).

### B.5 VLADBench (Li et al., USTC and Huawei; arXiv 2503.21505, ICCV 2025 "Fine-Grained Evaluation of LVLMs in AD")

- 12K multiple-choice questions over 5 domains and 29 tasks; the "Traffic Sign" tertiary task covers
  type, status, meaning and OCR of signs (https://arxiv.org/pdf/2503.21505, Sections 3.1, 3.3, Table 2).
  Repo https://github.com/Depth2World/VLADBench, license "not specified" (https://img.shields.io/github/license/Depth2World/VLADBench.json), 43 stars.
- Traffic Sign task accuracy, Table 10 (https://arxiv.org/pdf/2503.21505): Qwen2-VL-72B 71.18,
  GPT-4o 68.96, Qwen2.5-VL-72B 65.11, Gemini-1.5 Pro 64.05, Qwen2.5-VL-7B 61.71, InternVL2-76B 53.92,
  LLaVA-Video-7B 49.73, LLaVA-OneVision-7B 49.27, InternVL2-8B 47.45, Qwen2-VL-7B 40.94, MiniCPM-V-2.6
  32.10, Qwen-VL-7B 33.58, InternVL2-4B 34.01; driving-specific DriveLM 44.74, DriveMM 47.05, Dolphins
  25.02; their InternVL2-4B fine-tuned on VLADBench 61.00.

### B.6 IDKB, "Can LVLMs Obtain a Driver's License?" (Lu et al., ShanghaiTech; arXiv 2409.02914)

- The "Driving Road Data" part is a CARLA-rendered traffic sign understanding set: 3D sign models from
  several countries placed in two CARLA maps, about 400,000 frames collected, 112,388 QA samples after
  filtering (https://arxiv.org/pdf/2409.02914, Section 3). Repo MIT (https://img.shields.io/github/license/4DVLab/IDKB.json).
- Driving Road Data multiple-choice accuracy, Table 2 (https://arxiv.org/pdf/2409.02914): GPT-4o 0.87,
  Gemini-1.5-flash 0.85, XComposer2 0.62, MiniCPM-Llama3-V2.5 0.58, Qwen-VL-chat 0.47, Monkey-chat 0.46,
  DeepSeek-VL-7B 0.45, ShareGPT4V-7B 0.44, CogVLM 0.40, LLaVA-1.5-7B 0.33, VisualGLM-6B 0.18, BLIP2 0.17.

### B.7 DriveQA (Wei, Liu, Ohn-Bar; arXiv 2508.21824, ICCV 2025)

- Text and vision benchmark covering US driving-manual knowledge; DriveQA-V renders MUTCD signs in CARLA
  with perspective, weather and scenario variation; project page https://driveqaiccv.github.io (license
  not verified). Off-the-shelf accuracy on the Signs part, Table 4 (https://arxiv.org/pdf/2508.21824):
  GPT-4o regulatory 93.75 / warning 94.02 / guide 95.11 / temporary 94.35; Mini-InternVL-2B 64.06 / 55.34
  / 65.82 / 45.04; LLaVA-1.6-mistral-7B 42.58 / 43.01 / 52.75 / 37.50; LLaVA-1.5-7B 23.51 / 26.61 / 22.31
  / 21.10; VILA-1.5-8B 25.32 / 23.33 / 27.78 / 21.46. After fine-tuning on DriveQA: Mini-InternVL-2B
  93.79 / 92.19 / 91.08 / 96.51. Hardest sign types across models include Playground, Trauma Center, Golf
  Carts, Ground Clearance, No Stopping, No Parking (Table 5).

### B.8 SignReasoner and TrafficSignEval (Wang et al., Huawei Noah's Ark; arXiv 2604.10436, April 2026)

- Target: complex Chinese guide signs (multi-lingual text, composite symbols); "Functional Structure
  Units" decomposition, two-stage post-training (Caption-FSU distillation and FSU-GRPO) on Qwen2.5-VL-7B
  and Pangu-MM-7B; 5K manually annotated signs (https://arxiv.org/pdf/2604.10436, abstract and Section 4).
- Zero-shot caption-protocol accuracy on TrafficSignEval, Table 1 (https://arxiv.org/pdf/2604.10436):
  InternVL3-8B 54.07, InternVL3-78B 68.02, Qwen2.5-VL-7B 48.26, Qwen2.5-VL-72B 65.12; SignReasoner
  (Qwen2.5-VL-7B) 72.67 (+24.41). Structure protocol: Qwen2.5-VL-7B 37.95 to 83.08 after SignReasoner.
  Figure 1 shows Qwen2.5-VL-72B, InternVL3-78B and Gemini 2.5 Pro all misreading a lane-guidance sign.
  Benchmark release URL not found in the PDF (not verified).

### B.9 SignRAG (Zhu et al., Ohio State and TRC; arXiv 2512.12885, Dec 2025)

- Zero-shot recognition of 303 Ohio MUTCD regulatory sign designs: Gemini 2.5 Flash describes the sign,
  the description is embedded (OpenAI 3072-d embeddings) and retrieves candidates from a vector database
  of reference designs, then an LLM reasons over the candidates (https://arxiv.org/pdf/2512.12885, Section III).
- Results: 95.58 percent generation accuracy on ideal reference images (top-1 retrieval 88.65), 82.45 on
  a real-world set of 181 instances / 20 types collected over 72 miles of Ohio roads (top-1 63.44, top-5
  96.45); direct zero-shot prompting of Gemini 2.5 Flash and GPT-5 on the 303 reference images gave "less
  than 10%" accuracy; average latency 3.99 s (https://arxiv.org/pdf/2512.12885, Section IV).

### B.10 Traffic sign interpretation (TSI) line: TSI-CN and SignEye

- "Traffic Sign Interpretation in Real Road Scene" (Yang et al., NWPU; arXiv 2311.10793): TSI-CN
  dataset of 2,682 images at 2160x3840 with 2,910 natural-language descriptions, 7 symbol and 155 guide
  panel categories, split 2,016 / 666; a multi-task TSI architecture (not an LMM) reaches symbol F 79 /
  Acc 72, text F 77 / Acc 63, panel F 85 / Acc 85, interpretation soft accuracy 48; code and data "will be
  publicly available after the revision process" (https://arxiv.org/pdf/2311.10793, Sections 4, 6, Table 5).
  No release verified.
- SignEye (Yang et al.; arXiv 2411.11507): Traffic-CN, 20k+ first-person-view 1920x1080 images from six
  Chinese provinces; SigLIP vision encoder plus Qwen2 decoder fine-tuned with LoRA; TSI BLEU-4 56.2 /
  CIDEr 4.70 vs Qwen2-VL 45.4 / 4.24, MiniCPM-2.6 48.6 / 4.05, LLaVA-1.5 53.9 / 4.20; traffic guidance
  assistance accuracy 87.0 vs MiniCPM-2.6 79.7 (https://arxiv.org/pdf/2411.11507, Tables 3 and 4).
  Whether the compared VLMs were fine-tuned is not stated explicitly in the extracted text (not verified).

### B.11 Qualitative and adjacent evaluations

- "On the Road with GPT-4V(ision)" (OpenGVLab; arXiv 2311.05332): qualitative only; GPT-4V "can
  recognize most road signs, including the nearby 'SLOW' and the distant height limit '4.5m', but it
  misrecognizes the 'Speed Bump' sign"; limitation noted for "non-English traffic signs"
  (https://arxiv.org/pdf/2311.05332, Section 2.1 and Section 7).
- VehicleSec 2024 WIP (Aldeen et al., Clemson): LMMs as detectors of adversarial ("Natural Denoising
  Diffusion") fake traffic signs generated from four GTSRB classes; binary "real sign or not" accuracy
  GPT-4V 84.06, Bard 85.42, LLaVA-7B 79.80, LLaVA-13B 80.81 vs GTSRB-trained ResNeXt 17.17, MobileNet
  18.67, YOLOv5 44.12, VGG16 71.05 (https://www.ndss-symposium.org/wp-content/uploads/vehiclesec2024-44-paper.pdf, Table II).
- "Mind the Gap" (Chalamalasetti, Potsdam; arXiv 2511.13909, AAAI 2026): 28 Indian textbook sign
  samples, zero-shot multiple choice: GPT-5 Image Mini 0.79, InternVL3-78B 0.71, Gemma-3-27B 0.71,
  Qwen3-VL-30B 0.68, CLIP 0.54 (https://arxiv.org/pdf/2511.13909, Table 1).
- Scenario understanding with LVLMs (Rivera et al., TUM; arXiv 2501.17131, WACVW 2025): yes / no tag
  "traffic sign for ego-vehicle" among scene tags; LLaVA-1.6-34B best overall 83.7 accuracy
  (https://arxiv.org/pdf/2501.17131, Tables 1 and 2); not a fine-grained sign test.
- Sah et al. (arXiv 2503.06313): TSR is done by ResNet-50 (99.8 on GTSRB) and the MLLM part is only used
  for lane reasoning (https://arxiv.org/pdf/2503.06313, abstract); listed to avoid confusion with the title.
- TLS-Assist (Schmidt et al., Esslingen and Freiburg; arXiv 2511.14391): not a VLM evaluation but a
  modular redundancy layer that injects YOLO11 traffic light and sign detections as text into LLM driving
  agents in CARLA; YOLO11l/x sign detection mAP50-95 0.974 on their CARLA sign set; code at
  https://github.com/iis-esslingen/TLS-Assist (https://arxiv.org/pdf/2511.14391, Table I).
- Work-zone speed VLM pipeline (Martinez-Sanchez et al., UC Merced; arXiv 2606.08860): detector plus OCR
  plus semantic verification for temporary speed limits on Jetson; frame-level state metrics only
  (https://arxiv.org/pdf/2606.08860).

### B.12 Driving-specific VLMs: do they treat signs explicitly?

Checked by searching the full PDF text of each paper for "traffic sign", "road sign", "speed limit" and
"stop sign" (counts in parentheses).

| Model / dataset | arXiv | Explicit sign handling | Evidence |
|---|---|---|---|
| DriveLM (graph VQA on nuScenes / CARLA) | 2312.14150 | none found (0 hits) | https://arxiv.org/pdf/2312.14150 |
| DriveGPT4 | 2310.01412 | none found (0 hits) | https://arxiv.org/pdf/2310.01412 |
| Dolphins | 2312.00438 | none found (0 hits) | https://arxiv.org/pdf/2312.00438 |
| LingoQA | 2312.14115 | none found (0 hits) | https://arxiv.org/pdf/2312.14115 |
| EM-VLM4AD | 2403.19838 | none found (0 hits) | https://arxiv.org/pdf/2403.19838 |
| OmniDrive | 2405.01533 | none found (0 hits) | https://arxiv.org/pdf/2405.01533 |
| Senna | 2410.22313 | "Traffic Signal Detection" task but "we primarily focus on the most critical one: traffic lights" (red / green / yellow / none) | https://arxiv.org/pdf/2410.22313, Section on perception tasks |
| DriveBench (Are VLMs Ready for AD?) | 2501.04003 | no sign task; signs appear only in example answers | https://arxiv.org/pdf/2501.04003 |
| VLADBench | 2503.21505 | yes, dedicated Traffic Sign task (B.5) | https://arxiv.org/pdf/2503.21505 |
| IDKB | 2409.02914 | yes, CARLA sign set (B.6) | https://arxiv.org/pdf/2409.02914 |
| DriveQA | 2508.21824 | yes, MUTCD signs in CARLA (B.7) | https://arxiv.org/pdf/2508.21824 |

Conclusion: the first-generation driving VLMs (2023 to mid 2024) do not model signs as a task; sign
understanding entered driving VLM benchmarks only through knowledge-test style datasets (IDKB, DriveQA,
VLADBench), all of which find open 7B-class models far below GPT-4o on signs.

### B.13 Fine-tuned VLM checkpoints for traffic signs on Hugging Face

Query https://huggingface.co/api/models?search=traffic%20sign&limit=200 plus variants ("traffic-sign",
"trafficsign", "road sign") returned 146 unique models on 2026-10-31, almost all YOLO detectors. VLM /
LoRA checkpoints found:

| Model | Base | What it is | Evidence |
|---|---|---|---|
| longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs | unsloth/Qwen3-VL-8B-Instruct | full merged Qwen3-VL-8B, Unsloth template card only, no dataset or metrics, apache-2.0, 9 downloads | https://huggingface.co/api/models/longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs?full=true |
| RidaKo/qwen3-vl-road-signs-lt-lora | unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | LoRA for Lithuanian road sign captioning, 161 training images, 50 test images, no license tag | https://huggingface.co/RidaKo/qwen3-vl-road-signs-lt-lora/raw/main/README.md |
| AKrasavcev/lora_gemma3_4b_lt_road_signs | google/gemma-3-4b-it | LoRA, Lithuanian road sign captioning, gemma license | https://huggingface.co/api/models/AKrasavcev/lora_gemma3_4b_lt_road_signs?full=true |
| Anakonkai/vietnamese-traffic-sign-vqa-checkpoints | Qwen2.5-VL-3B | QLoRA and DPO adapters for Vietnamese sign VQA (VQA acc 0.9379 on their own test set) plus a Stable Diffusion 1.5 LoRA for sign generation; apache-2.0 tag | https://huggingface.co/Anakonkai/vietnamese-traffic-sign-vqa-checkpoints/raw/main/README.md |

No checkpoint for Florence-2, PaliGemma, LLaVA, SmolVLM or Idefics fine-tuned on traffic signs was found
by these queries, and none of the found checkpoints reports GTSRB, TT100K or MTSD accuracy. There is
therefore no open VLM checkpoint whose sign accuracy is comparable to the literature above.

---

## C. Open-vocabulary and promptable detectors applied to signs

### C.1 Which generic detection benchmarks contain sign classes

| Benchmark | Sign-related classes | Evidence |
|---|---|---|
| COCO (80) | `traffic light` (id 9), `stop sign` (id 11); no generic traffic sign | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml |
| LVIS v1 (1203) | `stop_sign` (id 1019, frequency f), `street_sign` (1026, f), `signboard` (959, f), `traffic_light` (1112, f); no generic traffic sign | https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py |
| Objects365 (365) | `Traffic Sign` (89), `Stop Sign` (127), `Speed Limit Sign` (267), `Crosswalk Sign` (305), `Traffic Light` (40), `Traffic cone` (66) | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/Objects365.yaml |
| Open Images V7 (601) | `Traffic sign` (549), `Stop sign` (495), `Traffic light` (548) | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/open-images-v7.yaml |

Per-class AP for these classes is not published in the Grounding DINO, OWL-ViT, YOLO-World or Detic
papers that we checked (their tables report dataset-level AP); "not verified" for all four. Objects365 is
the only common pretraining set with speed limit and crosswalk sign classes, which is relevant because
Grounding DINO, YOLO-World, GLIP and Florence-2 all pretrain on Objects365.

### C.2 Reported numbers for open-vocabulary or promptable models on signs

- Grounding DINO zero-shot vs fine-tuned on a Wuhan roadside infrastructure dataset (Fu et al., Wuhan
  University; arXiv 2601.10551): category "Traffic sign" zero-shot mAP 28.4 / mAP@50 38.9; after
  open-vocabulary fine-tuning 63.8 / 77.7 (Shanghai set 63.9 / 85.5). Traffic light zero-shot 33.3 / 57.9.
  Qwen-VL with LoRA then reasons over attributes (type, shape, colour, damage)
  (https://arxiv.org/pdf/2601.10551, Tables on pages with the Wuhan / Shanghai results). This is the only
  quantitative Grounding DINO number on the "traffic sign" class we found; none on GTSDB, TT100K or MTSD.
- Grounding DINO 1.5 paper (https://arxiv.org/pdf/2405.10300), Grounding DINO paper
  (https://arxiv.org/pdf/2303.05499, listed by search), YOLO-World (https://arxiv.org/abs/2401.17270):
  no traffic sign specific numbers (0 hits for "traffic sign" in 2405.10300; the others not grepped, not
  verified).
- SAM for traffic sign segmentation (Henninger et al., Fraunhofer IPM; Journal of Imaging 2024, 10(9),
  220, CC BY): SAM ViT-H and TinySAM encoders with learned prompt embeddings or a convolutional decoder,
  trained on 2,808 images and evaluated on Cityscapes (3,281 sign images), A2D2, ApolloScape, IDDA and
  Mapillary Vistas with the class "traffic sign"; best average IoU 0.432 (SAM-Conv-ViTH) and Mask R-CNN
  still better on Cityscapes, ApolloScape and Mapillary; with 90 training images SAM-Conv-ViTH loses 0.15
  IoU vs 0.26 for Mask R-CNN; instance detection rate 32.04 percent (SAM-Conv-ViTH) vs 22.60 (Mask R-CNN);
  conclusion: SAM "does not clearly outperform" Mask R-CNN (https://www.mdpi.com/2313-433X/10/9/220).
- SAM 2 as a sign extractor inside CdMT: compared against a ViT-Adapter detector; ViT-Adapter gives top-1
  0.93 / 0.84 / 0.80 on TT-100K / Sapporo / Yokohama (https://arxiv.org/pdf/2409.01534, Table 5).
- NanoVerse YOLO + TSR-MCL (Lu et al.; arXiv 2507.23331): a YOLO-World style RepVL-PAN detector plus a
  ViT / BERT contrastive classifier; 78.4 mAP on TT100K all-class long-tail detection, 91.8 accuracy;
  the preprint still contains placeholder citations ("[please supplement specific references]")
  (https://arxiv.org/pdf/2507.23331, abstract and Section 1).
- No paper reporting OWL-ViT / OWLv2, YOLO-World, YOLOE, Florence-2, Detic, GLIP or DINO-X on GTSDB,
  TT100K or MTSD was found with the queries used (web search 2026-10-31); treat as an open question and see
  replication experiment 7 in F.3.

### C.3 Licenses of open-vocabulary detectors and segmenters (for OpenTSR)

| Model | Code license | Weights (HF card) | Evidence |
|---|---|---|---|
| Grounding DINO | Apache-2.0 | IDEA-Research/grounding-dino-tiny apache-2.0 | https://img.shields.io/github/license/IDEA-Research/GroundingDINO.json, https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny |
| Grounding DINO 1.5 / DINO-X | Apache-2.0 (API client only; weights closed) | n/a | https://img.shields.io/github/license/IDEA-Research/Grounding-DINO-1.5-API.json, https://img.shields.io/github/license/IDEA-Research/DINO-X-API.json |
| MM-Grounding-DINO (mmdetection) | Apache-2.0 | n/a | https://img.shields.io/github/license/open-mmlab/mmdetection.json |
| OWL-ViT / OWLv2 | Apache-2.0 (scenic) | google/owlv2-base-patch16-ensemble apache-2.0, google/owlvit-base-patch32 apache-2.0 | https://img.shields.io/github/license/google-research/scenic.json, https://huggingface.co/api/models/google/owlv2-base-patch16-ensemble |
| YOLO-World | GPL-3.0 | n/a | https://img.shields.io/github/license/AILab-CVC/YOLO-World.json |
| YOLOE | AGPL-3.0 | n/a | https://img.shields.io/github/license/THU-MIG/yoloe.json |
| Florence-2 | MIT (weights) | microsoft/Florence-2-base and -large mit | https://huggingface.co/api/models/microsoft/Florence-2-large |
| SAM / SAM 2 | Apache-2.0 | facebook/sam2-hiera-small apache-2.0 | https://img.shields.io/github/license/facebookresearch/segment-anything.json, https://img.shields.io/github/license/facebookresearch/sam2.json, https://huggingface.co/api/models/facebook/sam2-hiera-small |
| SAM 3 | "not identifiable by github"; HF weights "other", manual gating | | https://img.shields.io/github/license/facebookresearch/sam3.json, https://huggingface.co/api/models/facebook/sam3 |
| Detic | Apache-2.0 | n/a | https://img.shields.io/github/license/facebookresearch/Detic.json |
| GLIP | MIT | n/a | https://img.shields.io/github/license/microsoft/GLIP.json |

---

## D. Self-supervised and large-scale pretraining relevant to signs

### D.1 DINOv2 / DINOv3 and SSL encoders on signs

- DINOv2 nearest-neighbour feature matching is the best zero-shot method on the MVV 11-class sign task
  (0.484 vs 0.468 for InternVL-3 9B) (https://arxiv.org/pdf/2508.02047, Table 3), see B.3.
- DINOv3 Table 25 (frozen features, 512 px): GTSRB DINOv2 ViT-g 78.2, DINOv3-7B 87.5, PE-core ViT-G 94.8
  (https://arxiv.org/pdf/2508.10104, Appendix B.4). For comparison CLIP ViT-L/14 linear probe is 92.5 at
  224 px (https://arxiv.org/pdf/2103.00020, Table 10), so language-supervised encoders still win on this
  OCR-like task.
- TS-1M (supervised fine-tuning of the whole encoder): DINO ViT-B/16 0.9190, MAE 0.9160, SimMIM 0.9403,
  MoCoV3 0.9385, SimCLR 0.9364 vs CLIP 0.9533 and ConvNeXt 0.9428 (https://arxiv.org/pdf/2603.23034, Table IV);
  the authors conclude masked-image SSL "fall[s] below the supervised baseline" on signs.
- ImageNet-21k ViT and BiT-M linear probes on GTSRB are weak (63 to 74) compared with CLIP (85 to 93)
  (https://arxiv.org/pdf/2103.00020, Table 10), consistent with ImageNet collapsing all signs into one class.

### D.2 Model merging and task arithmetic papers (GTSRB fine-tuned CLIP numbers)

All use the OpenAI CLIP backbones with the zero-shot text head frozen, fine-tuned 2000 iterations, batch
128, lr 1e-5, cosine schedule, AdamW wd 0.1 (https://arxiv.org/pdf/2212.04089, Appendix B.1).

| Backbone | Zero-shot ("Pre-trained") | Individual fine-tuned | Source |
|---|---|---|---|
| CLIP ViT-B/32 | 32.5 | 98.5 | https://arxiv.org/pdf/2212.04089, Table 7 |
| CLIP ViT-B/16 | 43.4 | 99.0 | https://arxiv.org/pdf/2212.04089, Table 6 |
| CLIP ViT-L/14 | 50.6 | 99.3 | https://arxiv.org/pdf/2212.04089, Table 5 |
| CLIP ViT-B/32 (TIES / AdaMerging eval) | 32.6 | 98.7 | https://arxiv.org/pdf/2306.01708, Table 11; https://arxiv.org/pdf/2310.02575, Table 1 |
| CLIP ViT-L/14 (TIES / AdaMerging eval) | 50.5 | 99.2 | https://arxiv.org/pdf/2306.01708, Table 12; https://arxiv.org/pdf/2310.02575, Table 2 |

Negation (forgetting) results: subtracting the GTSRB task vector drops GTSRB accuracy from 50.6 to 6.45
(ViT-L/14) while ImageNet control stays 72.2 (https://arxiv.org/pdf/2212.04089, Table 5). Merged 8-task
models on GTSRB: ViT-B/32 Task Arithmetic 69.7, TIES 72.1, Layer-wise AdaMerging 91.9, AdaMerging++
89.0 (traditional multi-task 98.9); ViT-L/14 Task Arithmetic 86.7, TIES 83.3, AdaMerging 98.0, AdaMerging++
97.5 (https://arxiv.org/pdf/2310.02575, Tables 1 and 2; TIES values also in https://arxiv.org/pdf/2306.01708, Tables 11 and 12).
In other words GTSRB is the task that suffers most when merged, which is a warning for any plan to fold a
sign head into a general-purpose merged CLIP.

Checkpoints and data used by this literature:
- tanganke/gtsrb on Hugging Face: parquet, train 26,640, test 12,630, plus 7 corrupted test splits
  (contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter) of 12,630
  each; 16,418 downloads; class names are the OpenAI CLIP strings; no license field on the card
  (https://huggingface.co/api/datasets/tanganke/gtsrb?full=true, https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md).
- tanganke/clip-vit-base-patch32_gtsrb: vision encoder fine-tuned with Adam, lr 1e-5, 4000 steps, batch
  32; card reports pre-trained 0.3043 and fine-tuned 0.9881 (https://huggingface.co/tanganke/clip-vit-base-patch32_gtsrb/raw/main/README.md;
  19,362 downloads per https://huggingface.co/api/models/tanganke/clip-vit-base-patch32_gtsrb?full=true).
  tanganke/clip-vit-large-patch14_gtsrb: pre-trained 0.4382, fine-tuned 0.9932
  (https://huggingface.co/tanganke/clip-vit-large-patch14_gtsrb/raw/main/README.md). The tanganke
  pre-trained values (30.4, 43.8) are lower than the CLIP paper (32.2, 50.3) although the templates are
  the OpenAI ones (https://raw.githubusercontent.com/tanganke/fusion_bench/main/fusion_bench/tasks/clip_classification/gtsrb.py);
  the cause (HF preprocessing vs OpenAI preprocessing) is not verified. The B/16 card has no README
  (https://huggingface.co/tanganke/clip-vit-base-patch16_gtsrb/raw/main/README.md returns "Entry not found").
  No license tag on these model cards.
- fusion_bench: MIT, 244 stars, last commit within the week before 2026-10-31
  (https://img.shields.io/github/license/tanganke/fusion_bench.json, https://img.shields.io/github/stars/tanganke/fusion_bench.json,
  https://img.shields.io/github/last-commit/tanganke/fusion_bench.json). task_vectors repo: license "not
  specified", 552 stars (https://img.shields.io/github/license/mlfoundations/task_vectors.json).

### D.3 Robustness and corruptions

- AdaMerging Table 4 (ViT-B/32 merged on 4 tasks; GTSRB column): clean Task Arithmetic 75.1 / AdaMerging
  96.3; motion blur 64.2 / 94.1; impulse noise 40.4 / 77.5; gaussian noise 49.4 / 76.0; pixelate 66.6 /
  90.6; spatter 54.3 / 89.4; contrast 70.6 / 95.1; JPEG 60.0 / 90.1 (https://arxiv.org/pdf/2310.02575, Table 4).
  The corrupted splits are the ones shipped in tanganke/gtsrb (D.2).
- Synset Signset Germany ships per-image environment and imaging-effect parameters for sensitivity
  analysis and a separate "SynsetSignsetSensitivityAnalysisDataset" (1M to 10M size class, CC BY 4.0)
  (https://huggingface.co/api/datasets?search=synset&limit=50), see E.1.
- CURE-TSR (challenge-level corruptions, 2M+ images, synthetic bias) is listed by MVV as "Cls. only" with
  "Synthetic bias, no masks" (https://arxiv.org/pdf/2508.02047, Table 1); not re-verified here.

---

## E. Generative data for signs

### E.1 Synset Signset Germany (Fraunhofer IOSB / IPA, ITSC 2024; arXiv 2512.05936)

- What: physically based rendering pipeline (OCTAS platform; Cycles path tracer and OGRE rasteriser) with
  GAN-generated dirt and wear textures; 211 German sign classes including signs introduced in 2020, 500
  images per class, 105,500 images per renderer (211,000 total), with mask, segmentation image and full
  metadata (environment, lighting, imaging effects) per image (https://arxiv.org/pdf/2512.05936, abstract
  and Section III; https://synset.de/datasets/synset-signset-ger/).
- Downstream result (ConvNeXt-Small, top-1): train on Signset Cycles only, test on real GTSRB 98.3; train
  on Signset OGRE, test GTSRB 98.2; train GTSRB test GTSRB 99.9; train GTSRB test Signset Cycles 89.4;
  train CATERED (older synthetic set) test GTSRB 76.4; Signset trained models reach 84.4 on CATERED
  (https://arxiv.org/pdf/2512.05936, Table II). So synthetic-only training is 1.6 points below real
  training on the real test set, roughly 20 points better than the previous synthetic dataset.
- License: "This work is licensed under Creative Commons Attribution 4.0" with citation and attribution
  requested on redistribution (https://synset.de/datasets/synset-signset-ger/); Hugging Face mirrors
  FraunhoferIOSB/Synset-Signset-Germany (cc-by-4.0, cycles/ogre train and validation parquet) and
  FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset (cc-by-4.0, 43 GTSRB classes, 21,500 images, the
  "synthetic twin" of GTSRB; cycles/train.parquet 1,495,705,131 bytes)
  (https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany?full=true,
  https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md,
  https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/resolve/main/cycles/train.parquet).
  Related CC BY 4.0 sets by the same group: Synset-Boulevard, Synset-Background-Effect-Datasets
  (https://huggingface.co/api/datasets?search=synset&limit=50).
- Relevance: the only permissively licensed, large, measured synthetic sign dataset found; the pipeline
  is template-driven (sign shapes from the official German catalogue), so the same approach would work
  for the Japanese catalogue if OpenTSR renders its own templates, but the OCTAS pipeline itself is not
  released (not verified).

### E.2 Diffusion inpainting of signs into real scenes (AMAP / Alibaba; arXiv 2609.02348, Sep 2026)

- Method: Stable Diffusion 1.5 inpainting backbone (about 1.4B parameters) with three priors: a JSON text
  prompt (type and value), a front-view vector template filled with measured dominant colours via
  IP-Adapter, and the same template affine-warped onto the fitted sign silhouette via ControlNet, plus a
  physical-consistency loss (CIELAB chroma and Sobel gradient) (https://arxiv.org/pdf/2609.02348,
  abstract; https://raw.githubusercontent.com/52hz-whale/TrafficSignInpaint/main/README.md).
- Evaluation: trained on in-house AMAP data, evaluated zero-shot on 5,182 TT100K-2021 regions against
  seven inpainting baselines including FLUX.1 Fill [dev] (12B); first on all nine image metrics and 1/14
  of FLUX Fill's 123.55 s latency (https://arxiv.org/pdf/2609.02348, Table 2 and Section 4).
- Downstream TSR gain: with YOLO26 on TT100K rare classes, synthetic data raises group-pooled AP50 of rare
  classes "by 1.23x to 7.40x over a real-data-only baseline" at 10 to 100 percent data fractions; Copy-Paste
  "reaches most of that improvement"; diffusion wins mainly on the hardest symbol classes (mean AP50 over
  12 target classes 0.770 vs 0.606 hard paste vs 0.481 Poisson paste); head classes unchanged (0.966 /
  0.954 / 0.970 / 0.963) (https://arxiv.org/pdf/2609.02348, Section 4.4 and Table 6).
- Code: https://github.com/52hz-whale/TrafficSignInpaint, Apache-2.0, 0 stars on 2026-10-31
  (https://img.shields.io/github/license/52hz-whale/TrafficSignInpaint.json, https://img.shields.io/github/stars/52hz-whale/TrafficSignInpaint.json).
  Note the SD 1.5 base weights are CreativeML OpenRAIL-M (https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5).

### E.3 Older GAN and copy-paste work

- Konushin, Faizov, Shakhuro (MSU; arXiv 2101.04927): rare Russian signs (RTSD, 205 classes of which 99
  appear only in the test set) embedded into real road images using high-resolution icons; a VAE predicts
  plausible sign locations and three GAN-based appearance-consistency methods are compared with plain
  "Synt" pasting; synthetic plus real data "improves the accuracy of both classifier and detector" for rare
  classes (https://arxiv.org/pdf/2101.04927, abstract and Section 4). Exact gains are in Tables 3 to 8 of
  the PDF (not transcribed here).
- Copy-paste with templates remains the strongest cheap baseline: in the AMAP study hard and Poisson
  copy-paste recover most of the diffusion gain on pooled AP50 (https://arxiv.org/pdf/2609.02348, Section 4.4).

### E.4 Community LoRAs for sign generation on Hugging Face

From the same HF search (B.13): Nyanmaru/TS_LoRA_sd3_uneven-road-sign and TS_LoRA_sd3_slipper-road-
condition-sign (base stabilityai/stable-diffusion-3-medium-diffusers, tag openrail++), CiroN2022/road-sign
(SDXL base, license "other"), Muapi/road-sign-erector-flux (FLUX.1-dev base, openrail++ tag) and the
SD 1.5 LoRA inside Anakonkai/vietnamese-traffic-sign-vqa-checkpoints
(https://huggingface.co/api/models?search=road%20sign&limit=200). None reports a downstream TSR
measurement; the FLUX.1-dev base is non-commercial (E.6).

### E.5 World models and driving video generators: do they render signs, and under what license?

| Generator | Paper | Sign handling found | Code license | Weights |
|---|---|---|---|---|
| GAIA-1 (Wayve) | https://arxiv.org/pdf/2309.17080 | 0 hits for sign terms | not released | not released |
| GAIA-2 (Wayve) | https://arxiv.org/pdf/2503.20523 | conditions on metadata including "Speed limits", "pedestrian crossings, traffic lights and their states", one-way indicators (Section on metadata conditioning); no sign recognition evaluation | not released | not released |
| Cosmos (NVIDIA) | https://arxiv.org/pdf/2501.03575 | 0 hits for sign or traffic light terms | Apache-2.0 (https://img.shields.io/github/license/nvidia-cosmos/cosmos-predict1.json, cosmos-predict2, cosmos-transfer1) | "nvidia-open-model-license" (https://huggingface.co/api/models/nvidia/Cosmos-Predict2-2B-Video2World) |
| DriveDreamer | https://arxiv.org/pdf/2309.09777 | 0 hits | Apache-2.0 (https://img.shields.io/github/license/JeffWang987/DriveDreamer.json) | not verified |
| Vista (OpenDriveLab) | https://arxiv.org/pdf/2405.17398 | 0 hits | Apache-2.0 (https://img.shields.io/github/license/OpenDriveLab/Vista.json) | apache-2.0 (https://huggingface.co/api/models/OpenDriveLab/Vista) |
| GenAD / OpenDV (OpenDriveLab) | https://arxiv.org/pdf/2403.09630 | 0 hits | Apache-2.0 (https://img.shields.io/github/license/OpenDriveLab/DriveAGI.json) | not verified |
| MagicDrive | https://arxiv.org/pdf/2310.02601 | 0 hits | Apache-2.0 (https://img.shields.io/github/license/cure-lab/MagicDrive.json) | not verified |
| Panacea | https://arxiv.org/pdf/2311.16813 | 0 hits | Apache-2.0 (https://img.shields.io/github/license/wenyuqing/panacea.json) | not verified |
| DrivingDiffusion | https://arxiv.org/pdf/2310.07771 | 0 hits | MIT (https://img.shields.io/github/license/shalfun/DrivingDiffusion.json) | not verified |
| SimGen | https://arxiv.org/pdf/2406.09386 | one prompt example "A motorcyclist on a road with a speed limit sign" | "not specified" (https://img.shields.io/github/license/metadriverse/SimGen.json) | not verified |

Finding: none of the driving world model papers evaluates sign legibility or downstream TSR, and only
GAIA-2 (closed) conditions on speed limits. Layout-conditioned generators (MagicDrive, Panacea,
DrivingDiffusion) condition on nuScenes boxes and maps, which have no sign class, so signs appear only as
uncontrolled background. For OpenTSR, world models are currently useful at most for background and
weather diversity, not for controlled sign content; sign-level control exists only in inpainting (E.2)
and template rendering (E.1).

### E.6 Licenses of image generators that could be used for sign synthesis

| Generator | License (as tagged on HF) | Evidence |
|---|---|---|
| Stable Diffusion 1.5 | creativeml-openrail-m | https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5 |
| SDXL base 1.0 | openrail++ | https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0 |
| Stable Diffusion 3 Medium | "stabilityai-ai-community" (other), gated | https://huggingface.co/api/models/stabilityai/stable-diffusion-3-medium |
| FLUX.1-dev, FLUX.1-Fill-dev | "flux-1-dev-non-commercial-license", gated | https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev, https://huggingface.co/api/models/black-forest-labs/FLUX.1-Fill-dev |
| FLUX.1-schnell | apache-2.0 | https://huggingface.co/api/models/black-forest-labs/FLUX.1-schnell |
| Cosmos Predict 1 / 2 weights | nvidia-open-model-license | https://huggingface.co/api/models/nvidia/Cosmos-1.0-Diffusion-7B-Text2World |
| Stability generative-models code | MIT; black-forest-labs/flux code Apache-2.0 | https://img.shields.io/github/license/Stability-AI/generative-models.json, https://img.shields.io/github/license/black-forest-labs/flux.json |

OpenRAIL-M and OpenRAIL++ permit commercial use with use-based restrictions; whether generated images may
be redistributed under Apache-2.0 or CC BY depends on the model license and on local law, and is not
settled here (not verified). FLUX.1-dev outputs are restricted by its non-commercial license.

---

## F. Synthesis

### F.1 Summary table

| Model or paper | Dataset | Protocol | Metric and value | Open weights | License | Source |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | GTSRB test | zero-shot, 3 templates | 32.2 top-1 | yes | MIT | https://arxiv.org/pdf/2103.00020 Table 11 |
| CLIP ViT-L/14-336 | GTSRB test | zero-shot | 52.3 | yes | MIT | same |
| CLIP ViT-L/14 | GTSRB | linear probe | 92.5 | yes | MIT | same, Table 10 |
| DFN-5B ViT-H/14-378 | GTSRB test | zero-shot (CLIP_benchmark) | 67.8 | yes | apple-amlr | open_clip results CSV |
| EVA02-E-14-plus | GTSRB | zero-shot | 67.4 (CSV) / 67.7 (paper) | yes | MIT | CSV; https://arxiv.org/pdf/2303.15389 |
| SigLIP SO400M-384 | GTSRB | zero-shot | 64.3 | yes | Apache-2.0 | open_clip results CSV |
| SigLIP ViT-B/16 | GTSRB | zero-shot | 51.6 | yes | Apache-2.0 | open_clip results CSV |
| MetaCLIP ViT-H/14 2.5B | GTSRB | zero-shot | 62.5 | yes | CC-BY-NC 4.0 | https://arxiv.org/pdf/2309.16671 |
| LAION-2B ViT-H/14 | GTSRB | zero-shot / joint FT / patched | 58.4 / 99.0 / 98.8 | yes | MIT-style | https://arxiv.org/pdf/2212.07143 |
| CLIP ViT-B/32, B/16, L/14 fine-tuned | GTSRB | full FT, frozen text head | 98.5 / 99.0 / 99.3 | yes (tanganke) | none stated | https://arxiv.org/pdf/2212.04089 |
| DINOv2 ViT-g / DINOv3-7B / PE-core G | GTSRB | frozen features, 512 px | 78.2 / 87.5 / 94.8 | yes / gated / yes | Apache-2.0 / custom / not verified | https://arxiv.org/pdf/2508.10104 Table 25 |
| CdMT GPT-4o (Hokkaido) | GTSRB, BTSD, TT-100K subsets, Sapporo, Yokohama | zero-shot LMM with multi-step prompts | top-1 0.93 / 0.88 / 0.97 / 0.89 / 0.85 | no (API) | n/a | https://arxiv.org/pdf/2409.01534 |
| Plain GPT-4o / GPT-4V | same | zero-shot | 0.89 / 0.81 (GTSRB subset) | no | n/a | same |
| LLaVA-1.5 / LLaVA-NeXT / VITA-1.5 | GTSRB subset | zero-shot | 0.32 / 0.39 / 0.45 | yes | llama2 and others | same |
| InternVL-3 9B / Gemma-3 12B / DINOv2 kNN | MVV (11 classes) | zero-shot | 0.468 / 0.428 / 0.484 | yes | Apache-2.0 (InternVL3), gemma, Apache-2.0 | https://arxiv.org/pdf/2508.02047 |
| CLIP ViT-B/32 encoder + head | TS-1M (454 classes) | supervised | 0.9533 top-1 | yes | MIT | https://arxiv.org/pdf/2603.23034 |
| TSCLIP | CRTS (46 classes, 10 regions) | FT China+Slovenia, test Germany | 0.8708 (zero-shot 0.2775) | code pending | MIT (repo) | https://arxiv.org/pdf/2409.15077 |
| Qwen2-VL-72B / GPT-4o / Qwen2.5-VL-7B | VLADBench Traffic Sign | multiple choice | 71.18 / 68.96 / 61.71 | yes / no / yes | Qwen license / n/a / Apache-2.0 | https://arxiv.org/pdf/2503.21505 |
| GPT-4o / Gemini-1.5-flash / best open (XComposer2) | IDKB Driving Road Data (CARLA signs) | multiple choice | 0.87 / 0.85 / 0.62 | no / no / yes | n/a | https://arxiv.org/pdf/2409.02914 |
| GPT-4o / Mini-InternVL-2B zero-shot / fine-tuned | DriveQA-V regulatory signs (CARLA) | multiple choice | 93.75 / 64.06 / 93.79 | no / yes | n/a | https://arxiv.org/pdf/2508.21824 |
| InternVL3-78B / Qwen2.5-VL-72B / SignReasoner-7B | TrafficSignEval (complex CN signs) | caption protocol | 68.02 / 65.12 / 72.67 | yes / yes / not verified | | https://arxiv.org/pdf/2604.10436 |
| SignRAG (Gemini 2.5 Flash + RAG) | 303 Ohio MUTCD designs / 181 real | zero-shot | 95.58 / 82.45 | no | n/a | https://arxiv.org/pdf/2512.12885 |
| GPT-4V / LLaVA-13B vs ResNeXt | NDD adversarial signs (binary) | zero-shot | 84.06 / 80.81 vs 17.17 | | | VehicleSec 2024 PDF |
| Grounding DINO | Wuhan roadside, class traffic sign | zero-shot / fine-tuned | mAP@50 38.9 / 77.7 | yes | Apache-2.0 | https://arxiv.org/pdf/2601.10551 |
| SAM-Conv-ViTH vs Mask R-CNN | Cityscapes and 4 others, traffic sign | fine-tuned decoder | avg IoU 0.432; Mask R-CNN better on 3 of 5 | yes | Apache-2.0 | https://www.mdpi.com/2313-433X/10/9/220 |
| Synset Signset Germany (ConvNeXt-S) | train synthetic, test GTSRB | supervised | 98.3 top-1 | dataset yes | CC BY 4.0 | https://arxiv.org/pdf/2512.05936 |
| AMAP diffusion inpainting (YOLO26) | TT100K rare classes | synthetic augmentation | pooled AP50 x1.23 to x7.40 | code yes | Apache-2.0 code, SD1.5 weights OpenRAIL-M | https://arxiv.org/pdf/2609.02348 |

### F.2 What this means for OpenTSR

1. VLMs are not yet a replacement for a specialised classifier on fine-grained signs. Every zero-shot
   number for open VLMs on GTSRB-like tasks is between 0.32 and 0.68 top-1 (B.1, B.3, B.8), the best
   closed model (GPT-4o) reaches 0.89 to 0.93 only on small stratified subsets and with heavy prompt
   scaffolding (B.1, B.2), and supervised CNNs or fine-tuned CLIP encoders are at 98.5 to 99.9 (A.4, D.2,
   E.1). On the 11-class MVV street-level task a language-free DINOv2 matcher beats InternVL-3 and Gemma-3
   (B.3). The reasonable role for VLMs in OpenTSR is (a) semi-automatic labelling and relabelling with
   human review, as NEC did (B.3), (b) interpretation of text-heavy guide and supplementary plates
   (SignEye, SignReasoner, SignRAG; B.8 to B.10), and (c) adversarial or out-of-distribution sign
   screening, where GPT-4V-class models beat GTSRB-trained CNNs by a wide margin (B.11).
2. CLIP-family encoders are the best permissively licensed starting point for the classifier. CLIP
   ViT-B/32 fine-tuned reaches 98.5 to 98.8 on GTSRB (D.2), CLIP ViT-B/32 was the best encoder of 20 in
   TS-1M (B.4) and language-supervised features dominate SSL features on this OCR-like task (D.1).
   Permissive choices: OpenAI CLIP (MIT), LAION / DataComp open_clip weights (MIT), SigLIP and SigLIP 2
   (Apache-2.0), EVA-CLIP (MIT). Avoid MetaCLIP (CC-BY-NC 4.0) and treat DFN (apple-amlr) and DINOv3
   (custom gated license) as needing legal review before weights derived from them are released (A.3, A.4).
3. Zero-shot CLIP is useful only as a weak prior or for class discovery. With 32 to 68 percent on 43
   classes and near-degenerate text embeddings for speed limits (A.2, A.5), it cannot label speed values;
   the template-driven approaches (pictogram prototypes with DINOv2 or CLIP image embeddings, A.6, B.3)
   are the credible zero-shot path for new Japanese classes, and should be measured in OpenTSR before any
   VLM is.
4. Open-vocabulary detectors can give class-agnostic sign boxes but no fine-grained class. Grounding
   DINO zero-shot "traffic sign" mAP@50 is 38.9 on a street-level set and jumps to 77.7 after fine-tuning
   (C.2); Objects365 pretraining gives these detectors some notion of "Speed Limit Sign" and "Crosswalk
   Sign" (C.1). Grounding DINO, OWLv2 and Florence-2 are Apache-2.0 or MIT; YOLO-World (GPL-3.0) and YOLOE
   (AGPL-3.0) are not compatible with an Apache-2.0 release that bundles their code (C.3). SAM does not
   clearly beat Mask R-CNN for sign masks (C.2), so masks are not a free win.
5. Synthetic data is the place where OpenTSR can gain most. Synset Signset Germany (CC BY 4.0) shows that
   a template-rendered synthetic set alone trains a 98.3 percent GTSRB classifier, and its GTSRB-subset is
   a ready-made "synthetic twin" to study the domain gap (E.1). For Japan the same recipe (official
   pictograms from the 道路標識 catalogue, physically based rendering, GAN or diffusion wear textures) is
   the most defensible route to a redistributable dataset, because no photo license question arises. For
   inpainting signs into real scenes, the AMAP recipe (SD 1.5 inpainting + ControlNet geometry + IP-Adapter
   colour) is open under Apache-2.0 code with OpenRAIL-M weights and is 14x cheaper than FLUX Fill; its
   measured gain on rare classes is real but copy-paste recovers most of it (E.2). Driving world models
   do not control sign content and should not be expected to close sign class gaps today (E.5).
6. Model merging literature is a caution, not a tool: GTSRB is consistently the task that degrades most
   when a CLIP sign head is merged with other tasks (D.2).

### F.3 Replication experiments feasible on a 2-core, 7 GB RAM CPU machine with Hugging Face access

All use `clip-benchmark/wds_gtsrb` test shards (3 tar files, about 41 MB each, A.3) or `tanganke/gtsrb`
(test parquet about 105 MB, D.2). Rough CPU costs assume about 15 to 25 images per second for ViT-B/32 at
224 px and about 1 image per second for ViT-L/14 on two cores; these are estimates, not measurements.

1. CLIP ViT-B/32 zero-shot on the full GTSRB test set (12,630 images) with the 43 official class names and
   3 templates (A.2), OpenAI weights via open_clip or HF `openai/clip-vit-base-patch32`. Compare with 32.2
   (https://arxiv.org/pdf/2103.00020, Table 11), 32.6 (open_clip CSV), 32.5 (https://arxiv.org/pdf/2212.04089, Table 7)
   and 30.4 (tanganke card). Add a per-class confusion matrix for the 9 speed limit classes to quantify the
   digit failure mode (A.5), which no fetched paper reports. Estimated 15 to 20 minutes.
2. open_clip ViT-B-32 laion2b_s34b_b79k and ViT-B-16 datacomp_xl_s13b_b90k zero-shot on the same set.
   Compare with 49.3 and 55.2 (https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv).
   Estimated 20 to 40 minutes each.
3. SigLIP `google/siglip-base-patch16-224` zero-shot (sigmoid logits, same templates). Compare with 51.6 for
   ViT-B-16-SigLIP in the open_clip CSV. Then run `google/siglip2-base-patch16-224` with identical prompts:
   no published GTSRB number exists (A.4), so this would be a new data point. Estimated 30 to 60 minutes each.
4. Linear probe on frozen CLIP ViT-B/32 image features: extract features for 26,640 train and 12,630 test
   images, fit scikit-learn logistic regression. Compare with 85.3 (https://arxiv.org/pdf/2103.00020, Table 10).
   Repeat on `facebook/dinov2-small` or `dinov2-base` features for a DINOv2 point that the literature only
   reports for ViT-g at 512 px (78.2, https://arxiv.org/pdf/2508.10104, Table 25). Estimated 1 to 2 hours.
5. Evaluate the released fine-tuned encoder `tanganke/clip-vit-base-patch32_gtsrb` on the test split with
   the frozen zero-shot text head. Compare with 0.9881 on its card (https://huggingface.co/tanganke/clip-vit-base-patch32_gtsrb/raw/main/README.md)
   and 98.5 (https://arxiv.org/pdf/2212.04089, Table 7). Also evaluate on the 7 corrupted splits of
   tanganke/gtsrb and compare the clean-minus-corrupted drop with AdaMerging's single-task numbers
   (https://arxiv.org/pdf/2310.02575, Table 4; note those are merged models, so expect a smaller drop).
   Estimated 20 minutes clean plus about 2 hours for the corruptions.
6. Small VLM multiple-choice on a stratified GTSRB subset (10 images per class, 430 images): SmolVLM-256M
   (Apache-2.0), Moondream2 (Apache-2.0), Florence-2-base (MIT, use the caption or OCR task and match the
   text to class names) and Qwen2-VL-2B-Instruct (Apache-2.0, int8 or bf16 on CPU, about 10 to 30 s per
   image). Prompt: the 43 class names as options, ask for the letter. Compare with LLaVA-1.5 0.32, LLaVA-NeXT
   0.39, CLIP ViT-B/32 0.24 and GPT-4o 0.89 on the Hokkaido GTSRB subset (https://arxiv.org/pdf/2409.01534, Table 1).
   Only Qwen2-VL-2B is likely to need several hours; the others fit in under an hour.
7. Class-agnostic sign detection with OWLv2 (`google/owlv2-base-patch16-ensemble`, Apache-2.0) and
   Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`, Apache-2.0) on the GTSDB test images (300
   images, boxes available from the dataset survey) with the prompt "traffic sign"; report AP50 for the
   single merged class. No published GTSDB number exists for either model (C.2); the closest reference is
   Grounding DINO zero-shot mAP@50 38.9 on Wuhan street-level signs (https://arxiv.org/pdf/2601.10551).
   Estimated 1 to 2 hours (OWLv2 at 960 px is slow on CPU; downscale to 640 px if needed and report that).
8. Synthetic-to-real transfer with frozen features: load `FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset`
   (cycles validation parquet, or stream a few thousand rows of train), train the logistic-regression probe
   of experiment 4 on synthetic features only and test on the real GTSRB test set. Compare qualitatively
   with the paper's full-training result of 98.3 synthetic-only vs 99.9 real (https://arxiv.org/pdf/2512.05936, Table II);
   the frozen-feature variant will be lower but isolates the domain gap at almost no compute. Estimated
   1 to 2 hours including the 1.5 GB download.

---

## G. Pages that could not be fetched

- https://guoyangzhao.github.io/projects/ts1m via WebFetch (permission request not answered); the page was
  retrieved with curl instead, so TS-1M claims rest on the curl text and the GitHub README.
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12193857/ and its redirect
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12193857 ("SAM for Road Object Segmentation: Promising but
  Challenging"): permission request not answered; not used.
- https://doi.org/10.3390/ai7050153 ("Automated Synthetic Traffic Dataset Generation via Diffusion-Based
  Inpainting Pipeline", MDPI AI): not fetched for time reasons; not used.
- https://huggingface.co/tanganke/clip-vit-base-patch16_gtsrb/raw/main/README.md: "Entry not found" (no
  model card), so the B/16 pre-trained and fine-tuned values on that checkpoint are unknown.
- https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/raw/main/README.md: "Entry not found" (no card),
  so the dataset license is unknown.
- The TSCLIP paper text did not yield the CLIP backbone size (parsed PDF only); not verified.
- No code or data URL was found inside the CdMT (2409.01534), few-shot ICL (2407.05814), SignReasoner
  (2604.10436) or TSI-CN (2311.10793) PDFs; release status of their datasets is unknown.
- Per-class AP for stop sign / traffic sign in LVIS, Objects365 and Open Images for Grounding DINO,
  OWL-ViT, YOLO-World and Detic: not found in the respective papers; not verified.
- https://github.com/nec-labs-ma/relabeling: README fetched via raw.githubusercontent.com (truncated by the
  output limit after the "Getting Started" section); LICENSE file content not fetched, GitHub reports "not
  identifiable by github".
