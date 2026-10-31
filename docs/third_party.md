# Third-party assets

Every dataset, model and codebase OpenTSR builds on or measures, with its license and source.
"Use in OpenTSR" states what we are allowed to do with it under the project's Apache-2.0
release policy (see `docs/decisions.md`). Licenses are quoted as written on the source page on
the date given; re-check before any redistribution.

This file covers assets shared by both tracks and the traffic sign track (main). Assets used only
by the traffic light track are listed in `traffic_lights/docs/third_party.md`.

## Traffic sign datasets

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| GTSRB (German Traffic Sign Recognition Benchmark, IJCNN 2011) | Official: https://benchmark.ini.rub.de/gtsrb_dataset.html (files hosted on sid.erda.dk, unreachable from the cloud sessions). Used here: Hugging Face mirror https://huggingface.co/datasets/bazyl/GTSRB, revision `a8093a9c7757b59d64702f892002542e8f3a1fb0` (parquet export of the Kaggle release; sha256 of both parquet files in `scripts/prepare_gtsrb.py`), 39,209 train / 12,630 test, verified against the official split sizes | Official page: "The data is free to use. However, we cordially ask you to cite the following publication if you do" (Stallkamp et al., IJCNN 2011). The mirror's card carries a `gpl-3.0` tag that the mirror author added; it is not the benchmark's license | First classification benchmark for signs (`docs/replication.md`). Redistribution of derived labels and crops is allowed by the "free to use" statement; cite the paper. |
| GTSDB (German Traffic Sign Detection Benchmark, IJCNN 2013) | Official: https://benchmark.ini.rub.de/gtsdb_dataset.html. Used here: Hugging Face mirror https://huggingface.co/datasets/GlowBond/GTSDB, revision `0d1c1a77fce322ba23b57a5e3261fae934f5afe3`, `archive.zip` sha256 `6144b38827acf3989bd7cd000a424c52cfd46a49da07e586439e25cf52995547` (TrainIJCNN2013 with gt.txt and TestIJCNN2013Download without ground truth) | ReadMe.txt in the package: "You are free to use this data package for any purpose you like." No license text on the web page | Zero-shot detection baseline on the training frames; cross-capture classification check on its 852 crops. Test ground truth requires the official FullIJCNN2013.zip (download locally). |
| Panoramax/classified_de_road_signs (German sign crops, 134 classes, 20,000+ images) | https://huggingface.co/datasets/Panoramax/classified_de_road_signs, revision `b4856947ed7cb6312587258acc90e8cf88a4aa13`; only `val.zip` (29.7 MB, 6,944 crops) downloaded | "cc-by-sa-4.0" (dataset card); crops cut from Panoramax street-level imagery | Evaluation of the Panoramax classifier (`docs/replication.md`). Share-alike: derived labels would have to stay CC BY-SA. |
| CLIP_benchmark `clip-benchmark/wds_gtsrb` test shards | https://huggingface.co/datasets/clip-benchmark/wds_gtsrb (no license metadata; webdataset export of the official GTSRB test set, lossless WebP) | none stated on the card; the underlying data is GTSRB ("free to use") | Only used to check that the mirror images are pixel-identical to the CLIP_benchmark protocol; not redistributed. |
| Synset Signset Germany, GTSRB subset (Fraunhofer IOSB) | https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset; only `cycles/validation.parquet` (4,300 renders with mask and segmentation) downloaded | "CC BY 4.0" with the attribution notice the card requests ("This work is based on the Synset Signset Germany Dataset ... (c) 2024 Fraunhofer IOSB") | Synthetic twins of the GTSRB classes in the dataset catalog (`docs/dataset_catalog.md`); candidate training data. |
| TT100K (Tsinghua-Tencent 100K) via the `finedet/tt100k` mirror | https://huggingface.co/datasets/finedet/tt100k (parquet export with `categories.csv`, 232 category codes); three shards downloaded (test-00003, train-00005, train-00006; 1,050 frames) | Official page: "Creative Commons Attribution-NonCommercial (CC-BY-NC)"; mirror tag `cc-by-nc-4.0` | Catalog samples only (research use). Not a release candidate and not redistributed. |
| MTSD (Mapillary Traffic Sign Dataset) official annotation JSON and images, sampled from the `NgTMDuc/cl_dataset` mirror | https://huggingface.co/datasets/NgTMDuc/cl_dataset (`MTSD/annotations/mtsd_v2_fully_annotated`, `MTSD/mtsd_fully_annotated_train_images`); a seeded sample of a few hundred keys fetched by `scripts/sample_mtsd_official.py` | `LICENSE.txt` shipped with the annotations: CC BY-NC-SA (link to the CC BY-NC 4.0 legal code) | Catalog samples only (research use). The `ThankGod/*` parquet mirrors were abandoned: their integer categories do not match their `id2labels.txt` (checked visually and by matching boxes across the mirrors). |
| CURE-TSR (Zenodo copy) | https://zenodo.org/records/3903066 (`CURE-TSR.zip`, 4.9 GB) | Zenodo record: CC BY 4.0 (the GitHub README asks for a usage-conditions form instead) | Robustness catalog section (14 sign types x 12 challenges x 5 levels); candidate robustness test set. |
| 国土交通省 道路局「道路標識一覧」(MLIT Japanese road sign list, one A3 page) | https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf | The pictograms are the figures of the 道路標識令 annexed tables (outside copyright under 著作権法 第 13 条); the PDF itself is a public MLIT document | Official Japanese pictograms cut into 221 tiles by `scripts/cut_mlit_sign_list.py` for the catalog and as the class list for the Japan track. Names cross-checked with the Japanese Wikipedia article 日本の道路標識 (`scripts/parse_ja_wikipedia_signs.py`, CC BY-SA text, used only for the number-to-name table). |

All other sign datasets from the survey (`docs/datasets.md`) are listed there with their licenses;
they are added to this table when first downloaded.

## Traffic sign models and weights

| Asset | Source | License (as written) | Use in OpenTSR |
|---|---|---|---|
| poojahira/gtsrb-pytorch released checkpoint `model/model_40.pth` (sha256 `157996e0f6baa5b5f0803dae03a74c61c31723c8768f7cdd0e6535df14bbae38`) | https://github.com/poojahira/gtsrb-pytorch | no license file in the repository | Scored on the official GTSRB test split (`docs/replication.md`). Not redistributed; the architecture is re-implemented (below). |
| OpenAI CLIP ViT-B/32, ViT-B/16, ViT-L/14 (zero-shot weights, loaded through open_clip `pretrained="openai"` with the `-quickgelu` configs) | https://github.com/openai/CLIP (weights re-hosted on the Hugging Face Hub by timm) | "MIT" (CLIP repository, covers the released weights) | Zero-shot GTSRB baselines. Permissive; could be a backbone or teacher. |
| FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb (revision `50b332c27907`) | https://huggingface.co/FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb | "cc-by-4.0" (model card); base model facebook/convnextv2-atto-1k-224 is Apache-2.0 | Scored on the official GTSRB test split and on GTSDB crops. Trained on the tanganke/gtsrb split; CC BY 4.0 would allow redistribution with attribution. |
| bazyl/gtsrb-model (ViT-B/16 in21k fine-tuned on bazyl/GTSRB, revision `95f6c95c765e`) | https://huggingface.co/bazyl/gtsrb-model | "apache-2.0" (model card) | Queued for scoring on the official test split (`make replicate-gtsrb-released`). |
| google/owlv2-base-patch16-ensemble (revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`) | https://huggingface.co/google/owlv2-base-patch16-ensemble | "apache-2.0" (model card) | Zero-shot class-agnostic sign detection baseline on GTSDB. Permissive; could be used for auto-labelling. |
| HuggingFaceTB/SmolVLM-500M-Instruct (revision `a7da5b986cb5`), Qwen/Qwen2-VL-2B-Instruct (revision `895c3a49bc3f`) | https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct, https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct | "apache-2.0" (both model cards) | Multiple-choice VLM probes on a GTSRB subset. |
| Panoramax/classify_de_road_signs (`classify_de_road_signs.pt`, 26,282,865 bytes, revision `5360aa6f4ef6c7b1998044b18d00b4d0b1a5a790`) | https://huggingface.co/Panoramax/classify_de_road_signs | "etalab-2.0" (model card); Ultralytics checkpoint format | Reproduction of the card's 0.986 on its validation crops; loaded only through the restricted unpickler in `scripts/eval_panoramax_de_classifier.py`. Not a release candidate (AGPL-3.0 toolchain). |
| Ultralytics YOLO-World v2 (`yolov8s-worldv2.pt`) | https://github.com/ultralytics/ultralytics (weights from the ultralytics/assets releases) | "AGPL-3.0" (same as YOLO11 above) | Open-vocabulary detection baseline on GTSDB only; not a release candidate. |

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
| poojahira/gtsrb-pytorch (STN + 3 conv CNN for GTSRB, 99.809 percent private leaderboard) | https://github.com/poojahira/gtsrb-pytorch | no license file in the repository | Architecture (layer sizes, normalisation constants, augmentation list) re-implemented from the README and `model.py` in `src/opentsr/signs/models/gtsrb_cnn.py`; no source copied. |
| Udacity Self-Driving Car Nanodegree "Traffic Sign Classifier" LeNet-5 recipe | hundreds of public student repositories (see `docs/oss_implementations.md`) | various (MIT in most) | LeNet-5 topology and grayscale normalisation re-implemented in `src/opentsr/signs/models/gtsrb_cnn.py`. |
| OpenAI CLIP GTSRB prompts and class names | https://github.com/openai/CLIP/blob/main/data/prompts.md | MIT | The 43 class descriptions and 3 templates are copied verbatim into `src/opentsr/signs/datasets/gtsrb.py` and `scripts/eval_clip_zeroshot_gtsrb.py` (MIT permits this; attribution kept in the docstrings). |
| Ultralytics classification eval transform (resize, centre crop, ToTensor) | https://github.com/ultralytics/ultralytics (`ultralytics/data/augment.py`) | AGPL-3.0 | The three-step preprocessing is re-created with torchvision in `scripts/eval_panoramax_de_classifier.py`; no source copied. |

## Python dependencies

Pinned in `requirements.txt` (traffic sign track, root) and `traffic_lights/requirements.txt` (traffic light track).
Notable licenses: PyTorch (BSD-3-Clause), transformers (Apache-2.0), open_clip (MIT-style notice),
onnxruntime (MIT), OpenCV (Apache-2.0), NumPy (BSD-3-Clause), Ultralytics (AGPL-3.0, see above).
