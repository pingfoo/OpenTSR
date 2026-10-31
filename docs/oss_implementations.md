# Open-source implementations of traffic light detection and state recognition

Survey compiled on 2026-10-31 (claude.ai research session, pingfoo/OpenTSR). Everything below
was checked against a live page, a raw file, an API response or a shallow clone on that date;
the URL that supports each claim is given next to it. "unknown" means the fact could not be
verified. Star counts come from shields.io badges and are approximate; last-commit dates come
from `git log` of shallow clones or badges.

How to read this file: section 0 is the summary, A covers production stacks (Autoware, Apollo,
openpilot), B the tooling shipped with datasets, C paper code with weights, D community
repositories and Hugging Face models, E leaderboards and surveys, and F ranks the replication
candidates. What we actually replicated, and the numbers we measured, is in
`docs/replication.md`; the datasets themselves are in `docs/datasets.md`.

Notes on verification limits: github.com HTML and api.github.com were not reachable from the
research sandbox (git clone and raw.githubusercontent.com were), paperswithcode.com is offline
(it redirects to huggingface.co/papers/trending, and no Wayback snapshot of the traffic light
task page exists), apollo-pkg-beta.cdn.bcebos.com was unreliable, and the KIT share hosting the
TLD-READY weights refused automated downloads.

## 0. Key takeaways

1. The only production-grade, actively maintained, permissively licensed TLR stack with downloadable weights is Autoware Universe. Since 2026 its models live on Hugging Face under `AutowareFoundation` (Apache-2.0): a YOLOX-s fine detector (416x416), a 960x960 YOLOX-s whole-image traffic light detector, MobileNetV2 car/pedestrian state classifiers (224x224, 11 and 3 classes), and a new "lamp recognizer". Training data (TIER IV T4dataset) is private; evaluation numbers are published in TIER IV's AWML repository but are on private data.
2. Baidu Apollo ships an equivalent pipeline (HD-map projection, YOLOX detector at 384x384, EfficientNet recognizer at 96x96, temporal reviser) under Apache-2.0, but its model zip files are on a Baidu CDN with no public accuracy numbers. openpilot has no traffic light perception at all.
3. The best "paper with weights and a reproducible number" candidates are: DetectionBench's LISA models on Hugging Face (YOLO26/YOLO11/YOLOv8/RF-DETR with test-split mAP and a plain `yolo val` protocol), the Traffic Lights of New York (TLoNY) YOLO models and dataset on Hugging Face, ImVisible/LytNet for pedestrian lights (weights in repo, 739-image test split), Turing's OpenLenda for Japanese arrow lights (weights on GitHub releases, but private evaluation data), and Possatti et al. 2019 (Darknet YOLOv3 weights on Google Drive).
4. Classic academic code (Behrendt 2017 scripts, Müller and Dietmayer TL-SSD, KIT TLD-READY) is either Caffe/TF1 era, lacks weights, lacks an evaluation script, or depends on registration-only datasets (BSTLD, DTLD).
5. The 2023 survey by Pavlitska et al. states that "open-sourcing the code of the TLR models is still rare" and flags only three works with source code (Müller and Dietmayer 2018, Yudin et al. 2018, Possatti et al. 2019); see https://arxiv.org/pdf/2309.02158.

## A. Production-grade autonomous driving stacks

### A1. Autoware Universe (Autoware Foundation / TIER IV)

Repository and status
- Repo: https://github.com/autowarefoundation/autoware_universe (default branch `main`, HEAD 1a411ff68d3344e973d584528137d6babf690a47 dated 2026-10-30, from `git log` of a shallow clone). Stars: 1.8k, last commit "yesterday" per https://img.shields.io/github/stars/autowarefoundation/autoware_universe.json and .../last-commit/... . The umbrella repo https://github.com/autowarefoundation/autoware shows 12k stars.
- License: `Apache License 2.0` in every TLR `package.xml` (9 packages) and the repo `LICENSE` is the Apache License, Version 2.0 text (checked in the clone).
- Framework: ROS 2 C++ nodes, ONNX models executed with TensorRT (FP16 default, FP32/INT8 selectable). No CPU runtime in the nodes themselves, but the ONNX files load fine in onnxruntime CPU (verified below).

Pipeline (packages under `perception/` in the clone; READMEs read from the clone)
- `autoware_traffic_light_map_based_detector`: projects Lanelet2 HD-map traffic lights into the image, inflating the ROI by calibration/vibration error parameters; outputs `~/output/rois` (with offset) and `~/expect/rois` (offset-free). Optional route filtering; otherwise `max_detection_range` and angle limits (README).
- `autoware_traffic_light_fine_detector`: YOLOX-s refines each coarse ROI to a tight box; if nothing is found the ROI becomes 0x0. Params: `precision: fp16`, `score_thresh: 0.3`, `nms_thresh: 0.65` (config/traffic_light_fine_detector.param.yaml). Default model `tlr_car_ped_yolox_s_batch_6.onnx`, labels `tlr_labels.txt` (launch file).
- `autoware_tensorrt_yolox` with `yolox_traffic_light_detector.launch.xml`: an alternative "whole image" detector that finds car and pedestrian traffic lights in the full frame (labels `BACKGROUND`, `traffic_light`, `pedestrian_traffic_light`), `score_threshold: 0.35`, `nms_threshold: 0.7` (config/yolox_traffic_light_detector.param.yaml).
- `autoware_traffic_light_selector`: matches whole-image detections to expected/rough ROIs and assigns traffic_light_id (README).
- `autoware_traffic_light_classifier`: three classifier types, selected by `classifier_type`: 0 = HSV rule-based, 1 = CNN (default), 2 = LampRecognizer (anchor-based per-lamp model). Params include `mean [123.675,116.28,103.53]`, `std [58.395,57.12,57.375]`, over/under exposure thresholds 0.85 / -0.83 (backlit ROIs become UNKNOWN with confidence 0.0), `score_threshold 0.2`, `nms_threshold 0.2`, `max_batch_size 64` (config/*.param.yaml). Labels follow the Vienna Convention naming, e.g. `red,right`.
- `autoware_traffic_light_category_merger` (merge car and pedestrian results), `autoware_traffic_light_occlusion_predictor` (LiDAR-based occlusion ratio, occluded ROIs become UNKNOWN), `autoware_traffic_light_multi_camera_fusion` (per-ROI best view selection, then Bayesian group fusion), `autoware_traffic_light_arbiter` (fuses perception with external/V2X signals, with a signal-match validator table and source-priority modes), `autoware_traffic_light_visualization`, `autoware_crosswalk_traffic_light_estimator`, plus `common/autoware_traffic_light_utils`, `common/autoware_traffic_light_compliance_checker` and `common/autoware_traffic_light_recognition_marker_publisher`.
- Launch layer: https://github.com/autowarefoundation/autoware_launch has `autoware_launch/launch/components/component_traffic_light.launch.xml` with `traffic_light_recognition/use_high_accuracy_detection` (true = map-based + fine detector pipeline `traffic_light_recognition_fine.launch.xml`; otherwise the whole-image pipeline `traffic_light_recognition_whole_image.launch.xml`), and `classifier_type` arguments documented as `0=HSVFilter, 1=CNN, 2=LampRecognizer` (checked in the clone of autoware_launch).

Model artifacts (all on Hugging Face now)
- The artifacts ansible role states "Every model is hosted on Hugging Face, under the AutowareFoundation organization. Each download is pinned to a tag": https://raw.githubusercontent.com/autowarefoundation/autoware/main/ansible/roles/artifacts/README.md ; the task list pins `traffic_light_fine_detector` to `v3.0`, `traffic_light_classifier` to `v4.0` and `tensorrt_yolox` to `v1.0` ("Also holds the whole_image_traffic_light_detector models"): https://raw.githubusercontent.com/autowarefoundation/autoware/main/ansible/roles/artifacts/tasks/main.yaml . Default install dir `~/autoware_data/ml_models`.
- https://huggingface.co/AutowareFoundation/traffic_light_fine_detector (license apache-2.0 on the card; created 2026-07-20; 486 downloads). Files: `tlr_car_ped_yolox_s_batch_{1,4,6}.onnx` each 35,794,092 bytes, `tlr_labels.txt` (50 bytes), `deploy_metadata.yaml`. Card: "YOLOX-s detectors fine-tuned by TIER IV", "Fine-tuning: performed by TIER IV on around 17,000 internal images of Japanese traffic lights", base weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth, "Further training details (epochs, hyperparameters, evaluation metrics) are not publicly documented", original source `https://awf.ml.dev.web.auto/perception/models/tlr_yolox_s/v3/`. Input verified with onnxruntime: `images [1,3,416,416]` float32, output `[1,3549,8]` (4 box + obj + 3 classes).
- https://huggingface.co/AutowareFoundation/traffic_light_classifier (apache-2.0; 734 downloads). Files: `traffic_light_classifier_mobilenetv2_batch_{1,4,6}.onnx` (8,926,508 / 8,926,648 bytes), `ped_traffic_light_classifier_mobilenetv2_batch_{1,4,6}.onnx` (8,885,654 bytes), `traffic_light_lamp_recognizer_comlops.onnx` (7,775,783 bytes), `lamp_labels.txt` (11 classes: green, left,red, left,red,straight, red, red,right, red,straight, unknown, yellow, red,up_left, red,right,straight, red,up_right), `lamp_labels_ped.txt` (red, green, unknown), `lamp_recognizer_ml.param.yaml`. Input verified: `input [1,3,224,224]`, output `[1,11]`. Card: vehicular classifier "fine-tuned on 83400 TIER IV internal images of Japanese traffic lights (58600 train, 14800 eval, 10000 test); the MobileNet-v2 model reaches 99.81% test accuracy at 224 x 224"; pedestrian: "21199 ... (17860 train, 2114 eval, 1225 test) ... 99.10% test accuracy". The lamp recognizer's "training data and procedure ... are not publicly documented". Original source `https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4`.
- Package README table (https://raw.githubusercontent.com/autowarefoundation/autoware_universe/main/perception/autoware_traffic_light_classifier/README.md): EfficientNet-b1 128x128 99.76% / MobileNet-v2 224x224 99.81% (vehicular); EfficientNet-b1 97.89% / MobileNet-v2 99.10% (pedestrian). The README also documents the full mmpretrain + mmdeploy retraining recipe and a 1045-image sample dataset https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz (1,621,492 bytes; verified: 1,045 JPEG crops, train 299 green / 306 red / 208 yellow, val 82 / 79 / 71; crops are tiny, for example 20x27 px).
- https://huggingface.co/AutowareFoundation/tensorrt_yolox (apache-2.0; 256 downloads): `yolox_s_car_ped_tl_detector_960_960_batch_1.onnx` (35,794,200 bytes) plus an `EntropyV2-calibration.table` for INT8 and `car_ped_tl_detector_labels.txt`; original source `https://awf.ml.dev.web.auto/perception/models/tl_detector_yolox_s/v1/`; "The training datasets, schedules, and evaluation metrics of the T4 finetuned variants are not publicly documented." Input verified: `images [1,3,960,960]`, output `[1,18900,8]`.
- Weight license: apache-2.0 per the model cards (the cards are the only license statement; no separate dataset license applies to the users of the weights).

Evaluation script: none in autoware_universe (the ROS nodes only infer). Reproducible evaluation exists only in AWML (below), against private T4dataset splits.

Measured during the survey (onnxruntime CPU, 2 threads, random input, mean of 5 runs): fine detector 416x416 60.6 ms/img; MobileNetV2 classifier 224x224 5.1 ms/img; whole-image detector 960x960 333.9 ms/img. Accuracy measured on public data (TIER IV sample crops and S2TLD) is in `docs/replication.md`; the published 99.81% is on TIER IV internal test data and cannot be reproduced.

### A2. TIER IV AWML (training pipelines for the Autoware TLR models)

- Repo: https://github.com/tier4/AWML (HEAD 9e44a974aae03a196003950898dc909cd0c680ad, 2026-07-13; 92 stars; license Apache-2.0 per shields badge and the `LICENSE` file in the clone).
- Framework: mmdetection / mmpretrain / mmdeploy (Python), with Autoware ONNX deploy scripts. Projects relevant to TLR: `projects/YOLOX_opt` (ROS package target `traffic_light_fine_detector`) and `projects/MobileNetv2` (target `autoware_traffic_light_classifier`), both "Supported dataset: T4dataset" only (project READMEs in the clone).
- Training data: TIER IV `DB TLR v1.0 ... v7.0` T4dataset splits (`autoware_ml/configs/t4dataset/db_tlr_v{1..7}.yaml`). The tutorial says "Request access to the T4dataset in WebAuto" (`docs/tutorial/tutorial_installation.md`), so the data is not public.
- Fine detector results (`projects/YOLOX_opt/docs/YOLOX_opt-S-TrafficLight/v1/base.md`): base/1.2 trained on DB TLR v1.0 to 7.0, config `yolox_s_tlr_416x416_pedcar_t4dataset.py`, "train time: (A100 * 1) * 1 days". On eval DB TLR v1.0, 4.0, 5.0, 6.0, 7.0: mAP 0.3432, AP50 0.4620, AP60 0.4520, AP70 0.4240, AP80 0.3110, AP90 0.0670 (whole-image metric). On the ROI-crop test set `tlr_infos_test_crops.json` at IoU 0.5: traffic_light AP 0.915 (recall 0.971, 19437 GT), pedestrian_traffic_light AP 0.669 (recall 0.809, 2181 GT), mAP 0.792; IoU 0.7: mAP 0.710; IoU 0.9: mAP 0.084. Model zoo links (reachable from the sandbox, HTTP 200 with sizes from HEAD): https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/yolox-opt/yolox-opt-s-trafficlight/t4base/v1.2/deployment.zip (99,677,954 bytes), .../best_mAP_epoch_300.pth , .../tlr_infos_test_crops.json (2,133,982 bytes; a dict of 13,205 crop keys of the form `<scene>/0/data/CAM_TRAFFIC_LIGHT_{FAR,NEAR}/<frame>.jpg_<box>` over 11 T4dataset scenes mapped to crop rectangles, so the annotations are public but the images are not).
- Classifier results (`projects/MobileNetv2/docs/MobileNetv2-CarTrafficLight/v1/base.md`): base/1.3 (DB TLR v1.0 to 7.0) per-class F1 on 11,355 eval crops: green 99.88, red 99.59, yellow 96.85, left,red 94.62, left,red,straight 99.49, red,right 97.97, red,straight 100.00, unknown 63.64; the headline `f1-score_top1 68.37` is a macro average that includes three zero-support classes. Pedestrian base/1.0: crosswalk_red F1 95.17, crosswalk_green 90.54, crosswalk_unknown 21.11 on 1,555 crops; "train time: (A40 * 1) * 5 hours". Checkpoints and ONNX: https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/mobilenet-v2/car_traffic_light/t4base/v1.3/deployment.zip (49,478,471 bytes, HTTP 200), .../v1.2/traffic_light_classifier_mobilenetv2_batch_1.onnx , pedestrian .../pedestrian_traffic_light/t4base/v1.0/ped_traffic_light_classifier_mobilenetv2_batch_1.onnx (8,885,529 bytes, HTTP 200).
- Latency tables (project READMEs): MobileNetV2 224x224 batch 6 as a ROS 2 node on RTX 3090 1.89 +- 1.32 ms, TensorRT FP16 on A100 0.224 ms; YOLOX-s fine detector batch 6 TensorRT FP16 on A100 1.45 ms, PyTorch 42.19 ms.
- Evaluation commands exist (`python3 tools/detection2d/test.py <config> <ckpt>`, `python3 tools/classification2d/test.py ...`) but require the T4dataset info files.

### A3. Autoware.AI legacy traffic light recognizers (ROS 1, archived)

- Repo: https://github.com/autowarefoundation/autoware_ai_perception (HEAD 98b49bb8841cc9f33ad3632a8d159ad0c6666a2f, 2023-03-10; 427 stars; license `Apache 2` in `trafficlight_recognizer/package.xml`, Apache-2.0 badge).
- Package `trafficlight_recognizer` nodes: `feat_proj` / `feat_proj_lanelet2` (project map traffic lights, optionally per-bulb `light_bulbs`, into the image; publishes `/roi_signal`), `region_tlr` (HSV hue/saturation/value threshold classifier on projected lamp positions; `region_tlr.cpp` builds a `thresholdSet` of hue ranges), `region_tlr_ssd` (Caffe SSD 300x300, "Autoware_tlr_SSD_300x300_iter_60000.caffemodel" + `deploy.prototxt`), `region_tlr_mxnet` (MXNet v1.0.0 classifier, user supplies `mxnet-network.json/params`), `roi_extractor`, `label_maker`, `tl_switch`, `tlr_tuner`.
- Weights: the README link https://autoware-ai.s3.us-east-2.amazonaws.com/region_tlr_ssd_trained_weights.tar.bz2 now returns HTTP 404 (checked with curl -I). No accuracy numbers anywhere in the package. Not a starting point; listed for completeness.

### A4. Baidu Apollo perception traffic light modules

- Repo: https://github.com/ApolloAuto/apollo (HEAD d53aa3da47a06a08e6d0cd175d5623a34fa0d6aa, 2026-04-16; 27k stars; Apache-2.0 badge; `LICENSE` is Apache License text, "Copyright (c) 2018 Apollo Authors").
- Framework: Cyber RT C++ components, models in ONNX -> TensorRT (new) or Caffe -> TensorRT (legacy), PaddlePaddle listed for `tl_detection`/`tl_recognition` in the `amodel` listing (`modules/tools/amodel/README.md`).
- Pipeline (READMEs under `modules/perception/`):
  - `traffic_light_region_proposal`: queries localization and HD map signal boundaries (4 3-D points per light), projects them into each camera, selects the camera with the longest focal length that sees all lights; inputs `/apollo/sensor/camera/front_6mm/image` and `front_12mm` (README). The design doc `docs/06_Perception/traffic_light.md` describes the two-camera (25 mm telephoto, 6 mm wide) design, the five output states (Red, Yellow, Green, Black, Unknown) and the Rectify / Recognize / Revise stages.
  - `traffic_light_detection`: `TrafficLightTLDetectorYolox`, `yolox.onnx`, input `images [1,3,384,384]`, outputs `bbox [1,3024,4]`, `conf`, `cls`, `max_batch_size 3`, `cls_th 0.24`, `min_crop_size 400` (`data/detection_yolox.pb.txt`, README). Legacy `detection_caffe.pb.txt`: `deploy.prototxt` + `baidu_iter_140000.caffemodel`, input 270x270 crop, `crop_scale 2.5`.
  - `traffic_light_recognition`: `EfficientNetRecognition`, `efficient_net.onnx`, input `inputs [1,3,96,96]`, outputs `outputs_cls [1,4]` (black, red, yellow, green) and `outputs_status [1,1]`, `classify_threshold 0.5`, scale 1/255 (`data/efficient_net.pb.txt`). Legacy Caffe recognizer uses three shape-specific nets (`vertical_caffe`, `quadrate_caffe`, `horizontal_caffe`; `baidu_iter_250000.caffemodel`, input 96x32 for vertical) (`data/recognition.pb.txt`).
  - `traffic_light_tracking`: `ReviseBySemantic` majority vote over multiple lights, hysteresis for black/unknown, and the rule that yellow after red before green is forced to red (README).
- Model hosting: `DOWNLOAD_REPOSITORY_URL = "https://apollo-pkg-beta.cdn.bcebos.com/perception_model"` in `modules/tools/amodel/amodel/model_manage.py`; the Chinese doc `docs/应用实践/开发调试教程/Apollo感知实践/感知模块综述.md` lists `tl_detection_caffe.zip`, `horizontal_torch.zip`, `quadrate_torch.zip`, `vertical_torch.zip`, `horizontal_caffe.zip`, `quadrate_caffe.zip`, `vertical_caffe.zip` under that CDN. File sizes unknown (HEAD requests were cut off from the sandbox). Weight license: unknown (not stated separately from the repo license).
- Accuracy: no public accuracy numbers found in the repo or docs. Training data: `dataset: "L4"` and `"KITTI"`/`"apollo"` strings in the model meta files, nothing more. Evaluation script: none found.
- Suitability: good architectural reference (projection, camera selection, reviser) but the models are opaque and the CDN is unreliable here.

### A5. comma.ai openpilot

- Repo: https://github.com/commaai/openpilot (64k stars, MIT, active daily per shields badges).
- Traffic light handling: none. `docs/LIMITATIONS.md` lists "In presence of traffic signs or stop lights, which are not detected by openpilot at this time." (https://raw.githubusercontent.com/commaai/openpilot/master/docs/LIMITATIONS.md). `cereal/log.capnp` contains no traffic light fields (grep on https://raw.githubusercontent.com/commaai/openpilot/master/cereal/log.capnp returned nothing for "traffic").

## B. Dataset-provided baselines and tooling

### B1. bosch-ros-pkg/bstld (Bosch Small Traffic Lights Dataset scripts)
- https://github.com/bosch-ros-pkg/bstld : 183 stars, MIT ("Copyright (c) 2018 Robert Bosch LLC, Karsten Behrendt"), last commit 2020-08-08 (HEAD 8156c10b).
- Contents (clone): `bosch_to_pascal.py`, `read_label_file.py`, `dataset_stats.py`, `show_label_images.py`, `label_files/{train,test,additional_train}.yaml` plus `non-commercial_license.docx`, `evaluation/evaluate.py` (precision-recall and mAP using `object_detection.utils.object_detection_evaluation` from the TF models repo; "NOTE This skips empty frames!"; class ids 0 off, 1 green, 2 yellow, 3 red), and `tf_object_detection/` (TF1 Object Detection API: `to_tfrecords.py`, configs `ssd_mobilenet_v1.config`, `faster_rcnn_nas.config` and `_no_classes` variants, `model_freezer.py`, `inference.py`, `evaluation.py`).
- Pretrained weights: none shipped. Reported numbers (README table, "Values are self-reported", test set without empty frames): Baseline (Behrendt et al. ICRA 2017) weighted mAP 0.36, <100 ms; Hierarchical Deep Architecture 0.53, ~150 ms (https://arxiv.org/abs/1806.07987); SSD Mobilenet V1 weighted mAP 0.60, mAP 0.41 (off 0.00, green 0.68, yellow 0.41, red 0.55), 38 ms; Faster RCNN NAS-A weighted mAP 0.65, mAP 0.43 (0.00/0.71/0.33/0.66), ~1560 s. Source: https://raw.githubusercontent.com/bosch-ros-pkg/bstld/master/README.md
- Dataset download: image zips at http://5.9.71.146/goijaskmo_bstld/ (label_files/README.md, "about 1.9 GB each"), registration page https://hci.iwr.uni-heidelberg.de/node/6132 (answered HTTP 500 on 2026-10-31). Unofficial HF mirror https://huggingface.co/datasets/shpotes/bosch-small-traffic-lights-dataset (license tag `other`, `train.tar.gz` 6.16 GB, `test.tar.gz` 12.15 GB, last modified 2022-03-10).
- Eval script: yes but TF1-only. Suitability: the evaluation protocol (skip empty frames, weighted mAP) is useful to copy; the code itself is obsolete.

### B2. julimueller/dtld_parsing (DriveU Traffic Light Dataset tools)
- https://github.com/julimueller/dtld_parsing : 61 stars, license "not specified" (no LICENSE file in the clone), last commit 2021-12-24 (HEAD f9575de4).
- Contents: Python package `dtld_parsing` (`driveu_dataset.py`, `calibration.py`, `vehicle_data.py`, `three_dimensional_position.py`), `python/load_dtld.py`, C++ library, deprecated MATLAB reader, calibration YAMLs, `doc/driveu_dataset_doc.pdf`. Label format changed to JSON in 05/2020 (attributes: direction, relevance, occlusion, orientation, aspects, state, pictogram, reflection). No models, no baselines, no evaluation code.
- Dataset: http://www.traffic-light-data.com/ with registration at Ulm University (page https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/ answered HTTP 200). Images are 2048x1024 16-bit TIFF (per the KIT and Korean repos below).

### B3. Thinklab-SJTU/S2TLD
- https://github.com/Thinklab-SJTU/S2TLD : 78 stars, MIT ("Copyright (c) 2018 DetectionTeamUCAS"), last commit 2024-06-26. Repo holds only README, LICENSE, a sample image; no code.
- Dataset: 5,786 images at 1080x1920 and 720x1280, 5 classes (red, yellow, green, off, wait on), 14,130 instances (README). Mirrors: Baidu, OneDrive, and https://huggingface.co/datasets/yangxue/S2TLD (license tag apache-2.0; `S2TLD（1080x1920）.zip` 304,993,489 bytes, `S2TLD（720x1280）.zip` 1,124,547,772 bytes; HEAD on the 1080p zip returned 200).
- Baselines: SCRDet++ paper (TPAMI 2022, https://thinklab.sjtu.edu.cn/SCRDet++.html); the survey lists "Zhao et al. 2022, S2TLD, mAP 71.24%" (https://arxiv.org/pdf/2309.02158). The ACCV 2022 workshop benchmark (B5) reports mAP@0.5 on S2TLD: Faster R-CNN 64.72, SSD 57.90, YOLOv3 71.00, RetinaNet 66.38, CenterNet 69.30, Cascade R-CNN 73.01.

### B4. LISA Traffic Light Dataset tooling
- Original: Kaggle https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset (page HTTP 200; needs a Kaggle account to download); papers Jensen et al. T-ITS 2016 (doi 10.1109/TITS.2015.2509509) and Philipsen et al. ITSC 2015. No official detector code.
- YOLO export on Hugging Face: https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights (license cc-by-nc-sa-4.0; 18,511 downloads; 86,053 files, 5,057,565,122 bytes). Card: BOX-annotation variant only, clip-level splits, train 17,082 images / 44,657 boxes, val 3,454 / 7,169, test 22,481 / 57,649; 1280x960 JPEG; 7 classes (go, goForward, goLeft, stop, stopLeft, warning, warningLeft); `data/data.yaml` uses `path: data` with shard sub-folders (fetched from the repo).
- Companion benchmark: DetectionBench (section C10) trains/evaluates YOLO and RF-DETR on it and publishes weights.

### B5. OPPO "Evaluating and Bench-marking Object Detection Models for Traffic Sign and Traffic Light Datasets" (ACCV 2022 Workshop DLSOD)
- Paper: https://openaccess.thecvf.com/content/ACCV2022W/DLSOD/papers/Mishra_Evaluating_and_Bench-marking_Object_Detection_Models_for_Traffic_Sign_and_ACCVW_2022_paper.pdf (Mishra et al., OPPO R&D Hyderabad). Detectron2/MMDetection, batch 4, lr 0.00025. mAP@0.5 (FRCNN / SSD / YOLOv3 / RetinaNet / CenterNet / Cascade R-CNN): WPI 58.98 / 88.60 / 79.80 / 28.12 / 84.70 / 79.15; LISA TL 34.09 / 48.50 / 29.00 / 37.23 / 50.10 / 49.09; BOSCH 37.23 / 36.50 / 46.60 / 31.28 / 52.70 / 52.57; S2TLD 64.72 / 57.90 / 71.00 / 66.38 / 69.30 / 73.01. CPU seconds per image on BOSCH: 14.04 / 3.11 / 8.75 / 4.62 / 5.74 / 5.69.
- Code/weights: the paper says they are at https://github.com/OppoResearchIndia/DLSOD-ACCVW, but the repository contains only a 234-byte README (clone, HEAD 6b5af7f6 2022-10-10; 0 stars). Not usable.

## C. Paper code with public weights (2017 to 2026)

### C1. Behrendt, Novak, Botros, "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification" (ICRA 2017)
- Code: the bstld repo (B1). The paper's adapted YOLO v1 detector and the tracker/classifier are not released; only the TF OD API baselines and the evaluation script are. Weights: none. Reported: weighted mAP 0.36 on the BSTLD test set (README table). Survey value: F1 about 80% on BSTLD (https://arxiv.org/pdf/2309.02158).

### C2. Müller and Dietmayer, "Detecting Traffic Lights by Single Shot Detection" (ITSC 2018), TL-SSD
- Repo: https://github.com/julimueller/tl_ssd (48 stars, no license file, HEAD be5ae7f7 2020-03-26). Caffe (weiliu89 SSD fork) with a modified PriorBox layer (small min_sizes 7..70 px, offsets) and a per-box state head (`do_state_prediction`, `num_states 6`, logistic loss).
- Weights: `caffemodel/SSD_DTLD_iter_90000.caffemodel` (36,241,373 bytes) and `prototxt/deploy.prototxt` in the repo; input 2048x512 crop (`SSD_INPUT_IMAGE_WIDTH/HEIGHT` in `test_on_dtld/ssd_dtld_test.py`). License of weights: unknown (no license).
- Evaluation: `ssd_dtld_test.py` only runs the net over DTLD and draws boxes; it computes no metric (checked the script). Reported: recall 95% on DTLD per the survey table. Suitability: Caffe-only, DTLD registration required; low.

### C3. Possatti et al., "Traffic Light Recognition Using Deep Learning and Prior Maps for Autonomous Cars" (IJCNN 2019)
- Code: `src/traffic_light_yolo` in https://github.com/LCAD-UFES/carmen_lcad (69 stars, license not specified, active in 2026 per badge). README: https://raw.githubusercontent.com/LCAD-UFES/carmen_lcad/master/src/traffic_light_yolo/README.md . Approach: Darknet YOLOv3 detector (classes: red or yellow merged, green) + prior map of relevant lights + optional OpenCV Random Forest, threshold 0.2.
- Training data: DTLD + LISA. Weights: Google Drive `yolov3-nrgr-10000-const-lr-1e-4_15000.weights` (235 MB, https://drive.google.com/open?id=1CCbc-6FgCtXD2g_1jQfGdm0cWqfIwwbC , verified alive), cfg https://drive.google.com/open?id=1dx6rKLPxnxu_YXzmsqr3D8IT5UTL_3Gu , names https://drive.google.com/open?id=1ZgUceNvhHm9XST0H_M_hi8hqk4wx9mI6 . Weight license: unknown.
- Reported: mAP 85.62% on DTLD and 50.59% on LISA per the survey (https://arxiv.org/pdf/2309.02158); the README gives no numbers and no evaluation script (inference command only: `./darknet detector test dummy_nrg.data <cfg> <weights> image.png -thresh 0.2`).

### C4. Weber et al. DeepTLR (IV 2016) and HDTLR (ITSC 2018)
- No official code or weights found (search across GitHub/web; the survey marks them without source code). Reported: DeepTLR F1 93.5% on private German data; HDTLR F1 85.8% on BSTLD and 88.8% on private data (survey Table II, https://arxiv.org/pdf/2309.02158). The arXiv version of HDTLR is https://arxiv.org/pdf/1805.02523 (no code link found).

### C5. Fernández, Guindel, Salscheider, Stiller, "A Deep Analysis of the Existing Datasets for Traffic Light State Recognition" (ITSC 2018)
- doi 10.1109/ITSC.2018.8569914. Dataset analysis paper; no code or weights found (search results only return publication listings).

### C6. KIT / FZI: TLD-READY (ITSC 2024) and ATLAS (IV 2025)
- Repo: https://github.com/KASTEL-MobilityLab/traffic-light-detection (19 stars, AGPL-3.0 per LICENSE and badge, last commit January 2025). Framework: Docker, Ultralytics YOLOv8 (a modified `ultralytics_modified` with a second class head for pictograms), YOLOv7, YOLOv9, ROS inference, relevance estimation via road markings and gradient boosting.
- Weights: `model_weights/download_model_weights.sh` (the raw URL returned 404 in our fetch) and the manual link https://bwsyncandshare.kit.edu/s/iabWB5k3q3LKzRz (share page HTTP 200; `/download` redirects to a WebDAV endpoint that answered 401 to curl, so it probably needs a browser session). Weight license: AGPL-3.0 presumably (repo license); unknown for sure.
- Training data: DTLD (registration). Reported: README "on filtered data": YOLOv8 XL precision 0.87, recall 0.74, mAP50 0.82. Paper https://arxiv.org/html/2409.07284v1 (Polley, Pavlitska et al.): splits BSTLD 4,329/764/8,334 at 1280x720 (4 classes), LISA 10,220/2,555/3,348 at 1280x960 (7 classes), DTLD 22,820/5,705/12,453 at 2048x1024 (20 state x pictogram classes); mAP50 per model: BSTLD YOLOv7 0.62, YOLOv8m 0.61, YOLOv8x 0.58, RT-DETR-L 0.62, RT-DETR-X 0.52; LISA YOLOv7 0.65, YOLOv8x 0.59, RT-DETR-L 0.65; DTLD YOLOv7x 0.60, YOLOv8x 0.61, RT-DETR-X 0.26; speeds on a GTX 2080 Ti (YOLOv8x 50.4 ms on DTLD); deployed at 52 ms end to end on an RTX A6000.
- Evaluation: `yolo val model=... data=... batch=4 imgsz=2048 conf=...` (README). Reproducible only with DTLD access and the KIT weights.
- ATLAS (https://arxiv.org/html/2504.19722v1, Polley et al., IV 2025): 33,044 images, 72,998 boxes, 25 pictogram-state classes, Karlsruhe; YOLO11x trained on DTLD+ATLAS reaches DTLD test 0.66 / 0.49 and ATLAS test 0.72 / 0.54 (mAP50 / mAP50-95); dataset at https://url.fzi.de/ATLAS ; no code or weights mentioned.

### C7. Turing Inc. OpenLenda (Japanese traffic lights with arrow indications)
- Repo: https://github.com/turingmotors/openlenda (52 stars, Apache-2.0, HEAD ab49eb47 2024-09-02). Framework: PyTorch, YOLOX fork. Approach: single-stage detector with multi-label outputs; classes `("red", "green", "yellow", "empty", "straight", "left", "right", "other")` (`yolox/data/datasets/coco_classes.py`), post-processing keeps every class whose score passes the threshold so one box can carry colour and arrow labels (`yolox/utils/boxes.py`, `detections_multi`). `exps/openlenda_s.py`: depth 0.33, width 0.50, `num_classes 8`, `input_size (1280, 1280)` for training, `flip_prob 0`, 30 epochs; demo uses `--tsize 640`.
- Weights (GitHub releases, verified sizes via HEAD): `openlenda_nano.pth` 7,590,937 bytes, `openlenda_tiny.pth` 40,593,541, `openlenda_s.pth` 71,861,829, `openlenda_x.pth` 792,901,075, at https://github.com/turingmotors/openlenda/releases/latest/download/<name>.pth (release v0.1.0). License Apache-2.0 (README).
- Reported (README, "private validation set"): nano 416x416 AP50 0.941 AR50 0.830; tiny 416x416 0.944 / 0.807; s 640x640 0.964 / 0.946; x 640x640 0.950 / 0.961. Dataset private, "About 44,000 images"; training/eval code "not possible to use ... as-is". Background paper: Shiotsuka et al., "Camera-based Context-aware Traffic Light Detection for Self-Driving Vehicles", ML4AD 2023 (https://ml4ad.github.io/files/papers2023/Camera-based%20Context-aware%20Traffic%20Light%20Detection%20for%20Self-Driving%20Vehicles.pdf): Tokyo data, 1928x1208, 20 Hz, 44,000 images with 19,509 labelled signals, YOLOX-X plus attribute head, BCE loss, 640x840 input; no numeric tables, no release.
- ONNX inference sample: https://github.com/Kazuhito00/openlenda-onnx-inference-sample (6 stars, Apache-2.0; onnxruntime, `--score_th 0.5 --nms_th 0.01 --grouping_th 0.8`).
- Suitability: the only open model for Japanese arrow lights; good inference baseline, not reproducible for accuracy.

### C8. CSDETR (improved RT-DETR, Journal of Real-Time Image Processing 2026)
- Paper: https://link.springer.com/article/10.1007/s11554-026-01864-6 (Chen, Liu, Shen, Zhang, Southeast University; published online 2026-03-02). BSTLD; abstract reports mAP50 +2.4% over the RT-DETR baseline with 13.1% fewer parameters at 345.9 FPS; absolute numbers are behind the paywall.
- Code: https://github.com/NuChen0602/CSDETR (1 star, no license, last commit December 2025; README has only a directory tree; no weights).

### C9. Zero-shot / VLM work on traffic lights
- Theodoridis et al., "Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception" (https://arxiv.org/html/2510.08352v1, Univ. of Limerick and Valeo, 2025): nine sub-4B VLMs on a CARLA-synthetic benchmark (DTP-Synthetic, 658 light-colour samples at 10 to 50 m). Light-colour accuracy: Qwen2-VL-2B 82.4%, InternVL2.5-2B-MPO 80.1%, Qwen2.5-VL-3B 78.2%, InternVL2.5-2B 77.2%, Ovis2-1B 77.0%, Aquila-VL-2B 75.5%, SAIL-VL-2B 75.0%, Ovis2-2B 74.7%, DeepSeek-VL2-Tiny 72.7%; InternVL3-78B 83.7%; humans 93.9%. Accuracy degrades roughly linearly with distance. No code or data URL in the paper.
- Zhang et al., "Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving" (https://arxiv.org/pdf/2608.18035, Tsinghua AIR and Bosch, Aug 2026): a YOLOv8 then YOLO26 traffic-element detector trained on OpenLane-V2 (4 light states + 9 sign classes) reaching mAP50 0.908 / mAP50-95 0.725 on 100 manually annotated NAVSIM frames; only a project page https://zzongzheng0918.github.io/TE-Aware-E2E-AD/ is given, release status unknown.
- TLS-Assist (Schmidt et al., https://huggingface.co/papers/2511.14391, Nov 2025) injects detector outputs into an LLM driving agent; no GitHub repo linked on the HF paper page.
- No paper was found that benchmarks OWL-ViT, Grounding DINO, Florence-2 or PaliGemma zero-shot on BSTLD/DTLD/LISA with released code; open-vocabulary detectors are only used as general tools in blog posts (search results such as https://pyimagesearch.com/2025/12/08/grounding-dino-open-vocabulary-object-detection-on-videos/).

### C10. DetectionBench LISA models (dronefreak, 2026)
- Framework repo: https://github.com/dronefreak/DetectionBench (10 stars, Apache-2.0, HEAD 18dc0ab9 2026-10-25). The YOLO evaluation path is plain Ultralytics `model.val(data=..., split=...)` and reports `box.map50`, `box.map`, `box.mp`, `box.mr` (`src/detectionbench/scripts/evaluate_yolo.py`); the LISA dataset config is `configs/dataset/lisa.yaml` with `eval_split: test`, `num_classes: 7`.
- Models on Hugging Face (all created 2026-08-01, updated 2026-10-01; test split of dronefreak/LISA-Traffic-Lights, imgsz 640, batch 32, seed 0, early stopping): https://huggingface.co/dronefreak/lisa-yolo26n (AGPL-3.0; `best.pt` 5,397,509 bytes; mAP@50 23.74, mAP@50-95 10.57, precision 37.23, recall 25.7; per class mAP@50 go 59.94, goForward 0.0, goLeft 6.2, stop 52.54, stopLeft 10.25, warning 37.13, warningLeft 0.14; 140 epochs trained), https://huggingface.co/dronefreak/lisa-yolov8m (AGPL-3.0; `best.pt` 52,053,650 bytes; 25.07 / 11.91 / 38.16 / 25.15), https://huggingface.co/dronefreak/lisa-rfdetr-nano (Apache-2.0; `checkpoint_best_total.pth` 120,899,707 bytes; 27.47 / 12.16 / 74.96 / 52.57, evaluated with Roboflow Supervision), plus lisa-yolo26s/m/l/x, lisa-yolo11x, lisa-rfdetr-small/medium (listed via https://huggingface.co/api/models?author=dronefreak).
- Leaderboard (LEADERBOARDS.md in the clone): RF-DETR Medium 33.01 / 14.12, RF-DETR Small 32.7 / 15.11, YOLO26m 29.08 / 13.68, RF-DETR Nano 27.47 / 12.16, YOLO26s 26.91 / 12.82, YOLO11x 26.4 / 13.09, YOLOv8m 25.07 / 11.91 (mAP@50 / mAP@50-95).
- Each model repo ships `args.yaml`, `results.csv`, PR/F1 curves and confusion matrices.

### C11. Traffic Lights of New York (TLoNY), Mehmet Kerem Turkcan (Columbia, 2025)
- Dataset: https://huggingface.co/datasets/mehmetkeremturkcan/traffic-lights-of-new-york (MIT; `tlony.zip` 487,995,061 bytes; YOLO format with train/val and `data.yaml`; 8 classes: red, green, yellow, red+yellow, unknown, pedred, pedgreen, pedunknown; images from Unsplash; size category n<1K; doi 10.57967/hf/4331).
- Models: https://huggingface.co/mehmetkeremturkcan/traffic-lights-of-new-york (AGPL-3.0; `tlony-yolov8n.zip` 18,972,492 bytes, `tlony-yolov8x-p2.zip` 269,549,693 bytes). Card metrics: yolov8x-p2 recall 0.50405, mAP50 0.56544, mAP50-95 0.30823; yolov8n 0.43263 / 0.50172 / 0.27866. Inside `tlony-yolov8n.zip`: `weights/best.pt` (6,460,243 bytes), `results.csv` whose final (100th) epoch row reproduces exactly those numbers, and `args.yaml` that says `model: yolo11n.pt`, `imgsz: 3840`, `batch: 1`, `epochs: 100` (so the "yolov8n" name is loose; the training resolution is 3840 px).
- GitHub mirror https://github.com/mkturkcan/traffic-lights-of-new-york (0 stars, MIT, Jan 2025).

### C12. ImVisible / LytNet (pedestrian traffic lights, CAIP 2019)
- Repo: https://github.com/samuelyu2002/ImVisible (76 stars, MIT, HEAD 5e4992e0 2024-04-18). PyTorch; LytNet is a MobileNetV2-derived classifier at 768x576 that outputs the pedestrian light class (Red, Green, Countdown Green, Countdown Blank, None) plus zebra-crossing endpoints.
- Dataset PTL: 5,059 images (3,456 train / 864 val / 739 test), class counts Red 1477, Green 1303, Countdown Green 963, Countdown Blank 904, None 412 (README); annotations in `Annotations/{training,validation,testing}_file.csv` (739 test rows); images `PTL_Dataset_876x657.zip` 776 MB on Google Drive https://drive.google.com/file/d/1KhKT3mfcYcdb9Zwq5jZS1bkAJyogHZbZ/view (verified alive) and a 4032x3024 version.
- Weights in repo: `Model/LytNetV1_weights` (7,699,886 bytes), `Model/LytNetV2_weights` (14,397,005 bytes). Reported precision / recall (Red, Green, Countdown Green, Countdown Blank): V1 P 0.97/0.94/0.99/0.86, R 0.96/0.94/0.96/0.92; V2 P 0.98/0.95/0.99/0.93, R 0.96/0.96/0.97/0.97. `Model/testing.py` computes per-class precision/recall and endpoint errors (paths must be edited at the top of the file).

### C13. Other recent academic repos (2024 to 2026) checked
- https://github.com/Rhhannn/ACK_2026_09_22_YOLO-Based-Traffic-Light-Detection (1 star, no license, Korean ACK 2026 paper "Temporal Robustness Evaluation of YOLO-Based Traffic Light Detection under Synthetic Degradations"): DTLD converted to 8-bit PNG, route-level split 32,699 / 4,035 / 4,244 images, classes Red/Yellow/Green/Red-Yellow; clean test mAP50 / mAP50-95: YOLOv8n 0.750 / 0.478, YOLO11n 0.752 / 0.481, YOLO12n 0.752 / 0.479; no weights in the repo.
- https://github.com/OmarElalaily/UGV-Vehicle-Traffic_light (0 stars, MIT, Oct 2026): YOLOv8 from scratch on BSTLD at 1280 px, 4 states; validation mAP@50 0.751, mAP@50-95 0.489 (from scratch), 0.746 / 0.487 after fine-tuning; no weights linked.
- https://github.com/alishavakhi/traffic-sign-light-detection (0 stars, MIT, Oct 2026): YOLO11 on a merged sign+light dataset; "No dataset and no trained checkpoint are included".
- https://github.com/POSTECH-IMLAB/traffic_light_recognition (35 stars, LGPL-3.0, Jan 2021): Caffe YOLOv2 + ResNet-101 deconv with focal regression loss (2018 paper); prototxt only, the `.caffemodel` named in the README is not in the repo listing we could see; no metrics in README.

## D. Community repositories and Hugging Face models

### D1. GitHub community repos (stars, last commit and license from shields.io; details from raw READMEs)

| Repo | Stars | Last commit | License | Framework / approach | Data | Weights | Reported numbers |
|---|---|---|---|---|---|---|---|
| https://github.com/davidbrai/deep-learning-traffic-lights | 482 | 2017-04 | BSD-2-Clause | Caffe SqueezeNet ensemble (3 models), 256x256 full-frame classification (Nexar challenge winner) | Nexar challenge images | caffemodels in `/model` | none in README |
| https://github.com/bosch-ros-pkg/bstld | 183 | 2020-08 | MIT | TF1 OD API SSD/Faster R-CNN configs + eval | BSTLD | none | see B1 |
| https://github.com/berktepebag/Traffic-light-detection-with-YOLOv3-BOSCH-traffic-light-dataset | 133 | 2021-10 | not specified | Darknet YOLOv3-tiny tutorial, 8 BSTLD classes, Carla + Jetson Nano | BSTLD | user trains (backup weights not published) | none |
| https://github.com/alex-lechner/Traffic-Light-Classification | 126 | 2018-04 | MIT | TF1 OD API SSD Inception v2 / Faster R-CNN, Udacity simulator + real images | own Dropbox dataset, BSTLD, LaRA | frozen graphs described | none |
| https://github.com/Thinklab-SJTU/S2TLD | 78 | 2024-06 | MIT | dataset only | S2TLD | none | see B3 |
| https://github.com/sovit-123/Traffic-Light-Detection-Using-YOLOv3 | 76 | 2020-09 | GPL-3.0 | Ultralytics YOLOv3-SPP (PyTorch), 608x608, 6 LISA classes | LISA (Kaggle) | Google Drive folder https://drive.google.com/drive/folders/1nGRGqw5KP6js9UbXDL5G99j_jYdKgdXl | "Current mAP is 0.919" (own 90/10 split, `best_model_12.pt`) |
| https://github.com/samuelyu2002/ImVisible | 76 | 2024-04 | MIT | PyTorch LytNet | PTL (Google Drive) | in repo | see C12 |
| https://github.com/LCAD-UFES/carmen_lcad (traffic_light_yolo) | 69 | 2026-08 | not specified | Darknet YOLOv3 + prior maps | DTLD + LISA | Google Drive | see C3 |
| https://github.com/JunshengFu/traffic-light-detector | 62 | 2017-12 | not specified | TF OD API detector + colour heuristics | COCO model | none | none |
| https://github.com/turingmotors/openlenda | 52 | 2024-09 | Apache-2.0 | YOLOX multi-label | private (44k JP images) | GitHub releases | see C7 |
| https://github.com/julimueller/tl_ssd | 48 | 2020-03 | not specified | Caffe SSD | DTLD | in repo | see C2 |
| https://github.com/affinis-lab/traffic-light-detection-module | 43 | 2023-10 | not specified | Keras YOLOv2, CARLA 0.8.4 (deprecated; points to a PyTorch successor) | LISA (~5800) then CARLA (~1800) | Google Drive | none |
| https://github.com/POSTECH-IMLAB/traffic_light_recognition | 35 | 2021-01 | LGPL-3.0 | Caffe YOLOv2 ResNet-101, focal regression loss | unknown | not found | none |
| https://github.com/KASTEL-MobilityLab/traffic-light-detection | 19 | 2025-01 | AGPL-3.0 | Ultralytics YOLOv7/8/9, modified two-head YOLOv8 | DTLD | KIT share | see C6 |
| https://github.com/marcomarasca/SDCND-Traffic-Light-Detection | 18 | 2019-07 | not specified | TF OD API (Udacity capstone) | Udacity | unknown | none |
| https://github.com/Syazvinski/Traffic-Light-Detection-Color-Classification | 12 | 2023-09 | not specified | Ultralytics YOLOv8 n/s/m | unknown | `.pt` files in repo | only CPU latency (M1 Max: nano 43.7 ms, small 72.1, medium 130.2) |
| https://github.com/maheshpaulj/Traffic-Light-Detection-YOLOv8 | 11 | 2024-08 | MIT | YOLOv8/v9, 1,713 images | own | unknown | only plots |
| https://github.com/AbangLZU/tl_detector | 11 | 2019-03 | not specified | TF OD API, LISA, TPU | LISA | unknown | none |
| https://github.com/dronefreak/DetectionBench | 10 | 2026-10 | Apache-2.0 | Ultralytics + RF-DETR benchmark | LISA (HF) | HF | see C10 |
| https://github.com/Satish-Vennapu/Traffic-Light-Detection-using-YOLOv8 | 6 | 2023-05 | MIT | YOLOv8 | unknown | unknown | none checked |
| https://github.com/Kazuhito00/openlenda-onnx-inference-sample | 6 | 2023-10 | Apache-2.0 | onnxruntime sample for OpenLenda | n/a | user exports | n/a |
| https://github.com/evanchien/LISA_TL_DETECTION | 5 | 2018-09 | not specified | Faster R-CNN (TF) on LISA | LISA | unknown | none |
| https://github.com/FridtjofS/TrafficLightDetection | 4 | 2024-03 | not specified | unknown | unknown | unknown | none |
| https://github.com/NuChen0602/CSDETR | 1 | 2025-12 | not specified | RT-DETR variant | BSTLD | none | see C8 |
| https://github.com/Rhhannn/ACK_2026_09_22_YOLO-Based-Traffic-Light-Detection | 1 | 2026-10 | not specified | Ultralytics YOLOv8n/11n/12n | DTLD | none | see C13 |
| https://github.com/OmarElalaily/UGV-Vehicle-Traffic_light | 0 | 2026-10 | MIT | YOLOv8 from scratch | BSTLD | none | see C13 |
| https://github.com/mkturkcan/traffic-lights-of-new-york | 0 | 2025-01 | MIT | Ultralytics YOLO | TLoNY | HF | see C11 |
| https://github.com/OppoResearchIndia/DLSOD-ACCVW | 0 | 2022-10 | not specified | README only | n/a | none | see B5 |

GitHub topic page https://github.com/topics/traffic-light-detection lists 47 repositories (fetched 2026-10-31); the ones with the most stars there are alex-lechner (126), sovit-123 (76), JunshengFu (62), ser94mor/self-driving-car-using-ros (59) and POSTECH-IMLAB (35).

### D2. Hugging Face Hub models (from https://huggingface.co/api/models?search=traffic+light and related queries, 2026-10-31)

| Model | Downloads | Updated | License | Weights | Evaluation numbers |
|---|---|---|---|---|---|
| AutowareFoundation/traffic_light_classifier | 734 | 2026-07-20 | apache-2.0 | 7 ONNX files (see A1) | 99.81% / 99.10% test acc on private data (card) |
| AutowareFoundation/traffic_light_fine_detector | 486 | 2026-07-20 | apache-2.0 | 3 ONNX | none on card (AWML has them) |
| AutowareFoundation/tensorrt_yolox | 256 | 2026-07-20 | apache-2.0 | whole-image TL detector ONNX | none |
| dronefreak/lisa-yolo26n and siblings (yolo26s/m/l/x, yolo11x, yolov8m, rfdetr-nano/small/medium) | 38 to 72 each | 2026-10-01 | agpl-3.0 (YOLO), apache-2.0 (RF-DETR) | `best.pt` / `.pth` | LISA test mAP@50 23.74 to 33.01 (see C10) |
| poorvika4566/YOLOv8-Traffic-Sign-Light-OCR | 68 | 2026-10-08 | apache-2.0 | `best.pt` 6.2 MB (YOLOv8n, 640, 10 epochs) | none |
| Aleton/trafficlight | 55 | 2026-05-31 | mit | `traficlight.pt` 6.3 MB (YOLOv8n, 2 classes, pedestrian lights for mobile robots, Russian card) | none |
| Xanadom1/AutoDrive_TrafficLight_Baseline_V1 | 43 | 2026-10-20 | agpl-3.0 | `weights/best.pt` 3.0 MB (yolov8n-cls, 128x128, 3 classes, Ohio State Buckeye AutoDrive) | val top-1 96.0%; dataset https://huggingface.co/datasets/Xanadom1/auto_drive_traffice_light (MIT, 27.9 MB) |
| mehmetkeremturkcan/traffic-lights-of-new-york | 14 | 2026-09-29 | agpl-3.0 | two zips (see C11) | mAP50 0.565 / 0.502 |
| Sawanparuthipqr/trafficlights_detection_models | 0 | 2026-06-18 | none stated | 123 files: Ultralytics YOLOv3 to YOLO12 and RT-DETR `tl-*.pt` checkpoints plus TorchVision Faster R-CNN/RetinaNet/FCOS `.pth`; no README | none |
| subhodeepmoitra/Traffic_signals_detection_YOLOv8m | 0 | 2026-07-13 | none stated | `best.pt` 52 MB, `best.onnx` | 9-class road dataset, val mAP50 0.878, mAP50-95 0.665 (not TL-only) |
| SiegenLi/TrafficLightMobileNetV3 | 0 | 2026-06-07 | mit | `traffic_light_classifier.onnx` 0.65 MB (CARLA crops, 64x32, 5 classes; dataset Zenodo record 20575485, CC-BY-4.0, 84 MB, 18,201 images) | none |
| Aniket200325/traffic_light_classification_mobilenetv3 | 0 | 2026-04-05 | mit | `.onnx`, `.pth`, `.pt` (6 MB) | none |
| OleFranz/TrafficLightDetectionAI | 0 | 2024-12-08 | gpl-3.0 | TorchScript 4-class classifier 26 MB plus ETS2/ATS game datasets (truck simulator screenshots) | none |
| Nughm3/yolov8-traffic-lights, ojaskandy/traffic-light-detection-yolo, YYdream/traffic_light_weights, AAI3001-P20/*, jojeong/detr_traffic_light, Krishu987/TrafficLightClassification, Ushan123/traffic-light-cnn | 0 | 2023 to 2025 | none | weights only, no card | none |

Takeaway: apart from Autoware, DetectionBench and TLoNY, HF traffic light models have no documented evaluation; most are course projects.

## E. Leaderboards and surveys

- Papers with Code: offline. paperswithcode.com/task/traffic-light-detection and paperswithcode.com/ redirect to https://huggingface.co/papers/trending (curl follow-redirect check, 2026-10-31); no Wayback snapshot exists for the traffic-light-detection task page; the traffic-sign-recognition page has a 2025-03-21 snapshot at http://web.archive.org/web/20250321025853/https://paperswithcode.com/task/traffic-sign-recognition .
- Survey: Pavlitska, Lambing, Bangaru, Zöllner, "Traffic Light Recognition using Convolutional Neural Networks: A Survey" (https://arxiv.org/pdf/2309.02158, KIT/FZI, ITSC 2023). Table II (selected): Behrendt 2017 BSTLD F1 about 80%; Weber HDTLR 2018 BSTLD F1 85.8%; Müller and Dietmayer 2018 DTLD recall 95%; Pon et al. 2018 BSTLD mAP 53%; Bach et al. 2018 DTLD mAP 83%; Kim et al. 2018 BSTLD mAP 38.48%; Lu et al. 2018 LISA mAP 91.1%; Possatti 2019 DTLD mAP 85.62%, LISA 50.59%; Gokul et al. 2020 BSTLD mAP 48.64%; Wang et al. 2022 LISA mAP 82.15%, LaRa 79.97%; Zhao et al. 2022 S2TLD mAP 71.24%; Mostafa et al. 2022 LISA mAP 92.16%; Liu and Li 2023 BSTLD mAP 81.5%. Metrics, splits and class sets differ between rows, so the table is not a leaderboard. Code availability flagged for only three works.
- Cross-dataset tables with a fixed protocol: TLD-READY (C6, BSTLD/LISA/HDTLR/DTLD with YOLOv7/v8 and RT-DETR), ACCV 2022 DLSOD workshop (B5, WPI/LISA/BOSCH/S2TLD with six detectors), DetectionBench LEADERBOARDS.md (C10, LISA), ATLAS (C6, DTLD/ATLAS with YOLOv8 to YOLO11).
- A Hugging Face Hub search for "traffic light" datasets (2026-10-31) returned 21 entries; the ones that matter are shpotes/bosch-small-traffic-lights-dataset, dronefreak/LISA-Traffic-Lights, mehmetkeremturkcan/traffic-lights-of-new-york and yangxue/S2TLD (see `docs/datasets.md`, section 9).

## F. Replication candidates (ranked)

Ranking criteria: weights downloadable without registration from a host reachable in the sandbox, evaluation data obtainable, evaluation script present, a published number to compare against, CPU-only feasibility (2 CPUs, 7 GB RAM).

### F1. DetectionBench LISA models (dronefreak/lisa-yolo26n first, then lisa-yolov8m, lisa-rfdetr-nano)
- Why first: weights on HF (5.4 MB), the exact test split on HF (5.06 GB, CC BY-NC-SA 4.0), the evaluation is plain Ultralytics `val` on `split=test`, and the targets are documented (YOLO26n mAP@50 23.74, mAP@50-95 10.57, P 37.23, R 25.7; YOLOv8m 25.07 / 11.91).
- CPU feasibility: 22,481 test images at 640 px; YOLO26n is about 6 GFLOPs so expect roughly 0.1 to 0.2 s per image on 2 cores, about 1 hour total; YOLOv8m (78.9 GFLOPs) would take several hours. Disk: 5.1 GB. The `data/data.yaml` has `path: data`, so run from the download root or rewrite `path` to an absolute path.
- Steps:
```
python3 -m venv ~/venv-tl && . ~/venv-tl/bin/activate
pip install ultralytics huggingface_hub[cli]     # pulls CPU torch
hf download dronefreak/LISA-Traffic-Lights --repo-type dataset --local-dir ~/lisa_hf
hf download dronefreak/lisa-yolo26n best.pt --local-dir ~/lisa_yolo26n
sed -i 's#^path: data#path: '"$HOME"'/lisa_hf/data#' ~/lisa_hf/data/data.yaml
yolo val model=~/lisa_yolo26n/best.pt data=~/lisa_hf/data/data.yaml split=test imgsz=640 batch=8 device=cpu workers=2 plots=False
# compare metrics/mAP50(B) with 0.2374 and mAP50-95(B) with 0.1057
```
- Note on licenses: AGPL-3.0 weights (Ultralytics) and a non-commercial dataset; fine for a benchmark, not for a commercial OpenTSR release. The RF-DETR nano checkpoint is Apache-2.0 but needs `pip install rfdetr` and a COCO conversion (`detectionbench-convert-yolo-to-coco`).

### F2. TLoNY YOLO (mehmetkeremturkcan/traffic-lights-of-new-york)
- Why: MIT dataset (488 MB) with the exact val split and `data.yaml`, weights on HF (19 MB zip), target mAP50 0.50172 / mAP50-95 0.27866 for the small model and 0.56544 / 0.30823 for yolov8x-p2, and `results.csv` in the zip for a sanity check.
- Caveat: training used `imgsz 3840` and `batch 1` (`args.yaml`), so validation must also run at 3840 px to match; the val set is small (dataset size tag n<1K), so even at about 5 to 15 s per image on CPU this is under an hour for the n model. Vehicular and pedestrian lights are mixed in 8 classes.
```
hf download mehmetkeremturkcan/traffic-lights-of-new-york --repo-type dataset --local-dir ~/tlony_ds && unzip -q ~/tlony_ds/tlony.zip -d ~/tlony_ds
hf download mehmetkeremturkcan/traffic-lights-of-new-york tlony-yolov8n.zip --local-dir ~/tlony_m && unzip -q ~/tlony_m/tlony-yolov8n.zip -d ~/tlony_m
# edit ~/tlony_ds/tlony/data.yaml so that path/train/val point at the extracted folders
yolo val model=~/tlony_m/tlony-yolov8n/weights/best.pt data=~/tlony_ds/tlony/data.yaml split=val imgsz=3840 batch=1 device=cpu workers=2
# compare with mAP50 0.50172, mAP50-95 0.27866
```

### F3. ImVisible LytNet V2 (pedestrian traffic lights)
- Why: weights in the repo (14 MB), MIT, PyTorch, a fixed 739-image test split with CSV annotations in the repo, a test script that prints per-class precision and recall, and published targets (V2 precision 0.98/0.95/0.99/0.93, recall 0.96/0.96/0.97/0.97). Image zip (776 MB) is on Google Drive, which is reachable (use `gdown`).
- CPU feasibility: 739 images through a MobileNetV2-sized net at 768x576, a few minutes.
```
pip install torch torchvision pandas opencv-python-headless gdown
git clone https://github.com/samuelyu2002/ImVisible.git ~/ImVisible
gdown 1KhKT3mfcYcdb9Zwq5jZS1bkAJyogHZbZ -O ~/PTL_Dataset_876x657.zip && unzip -q ~/PTL_Dataset_876x657.zip -d ~/ptl
# in ~/ImVisible/Model/testing.py set test_file_loc=~/ImVisible/Annotations/testing_file.csv,
# test_image_directory=<extracted 876x657 folder>, MODEL_PATH=~/ImVisible/Model/LytNetV2_weights, and use LYTNetV2()
cd ~/ImVisible/Model && python3 testing.py
```

### F4. Autoware Universe TLR models (AutowareFoundation on Hugging Face)
- Why: production models, Apache-2.0, ONNX that runs on CPU (measured 61 ms detector / 5 ms classifier / 334 ms whole-image detector per image), pinned tags (`--revision v3.0`, `v4.0`, `v1.0`). What is missing is public evaluation data: the published 99.81% / 99.10% (classifier) and AWML mAP 0.792 at IoU 0.5 on ROI crops (fine detector) are on TIER IV internal data. A faithful replication is impossible; the substitute implemented in this repository (`scripts/eval_autoware_*.py`, results in `docs/replication.md`) is (a) the colour-only check on the Autoware sample crops and (b) cross-domain accuracy on S2TLD: classifier accuracy on ground-truth crops, fine-detector recall on synthetic coarse ROIs, and class-agnostic AP of the 960x960 whole-image detector. This benchmarks a Japanese-trained model on another country and gives OpenTSR a baseline to beat.
```
hf download AutowareFoundation/traffic_light_classifier --revision v4.0 --local-dir ~/aw_cls
hf download AutowareFoundation/traffic_light_fine_detector --revision v3.0 --local-dir ~/aw_det
hf download AutowareFoundation/tensorrt_yolox --revision v1.0 --include "yolox_s_car_ped_tl_detector_960_960_batch_1.onnx" "car_ped_tl_detector_labels.txt" --local-dir ~/aw_whole
curl -O https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz && tar xzf traffic_light_sample_dataset.tar.gz
python scripts/eval_autoware_classifier.py --folder traffic_light_sample_dataset/val --out runs/replication/sample_val.json
```
  For the detectors, decode the `[N,8]` YOLOX output as (cx, cy, w, h, obj, 3 class scores) on strides 8/16/32 grids (416 px: 52x52+26x26+13x13 = 3549 cells; 960 px: 120x120+60x60+30x30 = 18900), multiply obj by class score, apply NMS 0.65 (fine) or 0.7 (whole image) with score thresholds 0.3 / 0.35, then compute AP with pycocotools against S2TLD or LISA boxes collapsed to one class.

### F5. Turing OpenLenda (Japanese arrow lights)
- Why: the only open model that outputs Japanese arrow indications, Apache-2.0, weights on GitHub releases (nano 7.6 MB, s 72 MB). What is missing: the validation set is private, so the AP50 0.941 to 0.964 numbers cannot be reproduced; and the repo's training/eval code "cannot be used as-is". Useful as a qualitative baseline on Japanese dashcam footage and as a label-space reference (red, green, yellow, empty, straight, left, right, other).
```
git clone https://github.com/turingmotors/openlenda.git ~/openlenda && cd ~/openlenda
pip install torch torchvision loguru opencv-python-headless thop tabulate pycocotools ninja  # or: poetry install
wget https://github.com/turingmotors/openlenda/releases/download/v0.1.0/openlenda_s.pth -P models
python tools/demo.py image -f exps/openlenda_s.py -c models/openlenda_s.pth --path <jp_image.jpg> --conf 0.5 --nms 0.01 --tsize 640 --save_result   # no --device gpu on CPU
python tools/export_onnx.py --output-name openlenda_s.onnx -f exps/openlenda_s.py -c models/openlenda_s.pth   # then use Kazuhito00/openlenda-onnx-inference-sample
```

### Lower-ranked
- Possatti et al. YOLOv3 (Google Drive weights alive, 235 MB; needs Darknet build; LISA from HF is reachable but the paper's evaluation protocol is not in the repo and DTLD needs registration).
- KIT TLD-READY (weights behind a KIT share that refused automated download; DTLD registration; GPU-era resolutions of 1920 to 2048 px). Good target for the GPU machine once DTLD access is granted: `yolo val model=<kit weights> data=<dtld yaml> imgsz=2048` should give about mAP50 0.61 for YOLOv8x (paper) or 0.82 on the "filtered" README split.
- TL-SSD (Caffe, DTLD, no metric script), Autoware.AI region_tlr_ssd (weights 404), Apollo (CDN unreliable, no numbers), CSDETR and DLSOD-ACCVW (no weights).
