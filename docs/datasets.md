# Public datasets for traffic light detection and state recognition

Survey compiled on 2026-10-31 from live web pages (claude.ai research session, pingfoo/OpenTSR).
Every factual claim carries the URL on which it was verified. Where a page could not be fetched
or did not state a value, the field says "unknown" or "not verified". Numbers that disagree
between sources are reported with both values. Section 11 gives the recommendation for the
OpenTSR benchmark; section 12 lists pages that could not be fetched. The S2TLD section also
records what we measured ourselves after downloading the data (see `docs/replication.md`).

Conventions:
- "state" = lamp colour state (red / yellow / green / off and variants)
- "pictogram" = lamp symbol (circle, arrow with direction, pedestrian, bike, tram)
- "NC" = non-commercial restriction
- "public test labels" = the labels of the official test split can be downloaded

Quick orientation (details below):

| If you need | Use | Why |
|---|---|---|
| A redistributable benchmark seed, downloadable in minutes | S2TLD (1.4 GB, MIT) | permissive, housing-level boxes with `off`, hosted by the author on Hugging Face |
| The standard US small-light benchmark with public test labels | BSTLD (18 GB, Bosch NC license) | most-cited baseline numbers; test set is 8,334 frames |
| The richest labels (arrows, red-yellow, relevance, track IDs) | DTLD (registration, research-only, no redistribution) | 2 MP stereo, 11 German cities, 292k boxes |
| US day/night sequences | LISA (5 GB, CC BY-NC-SA 4.0) | 43k frames, arrows in class names |
| A 2025 dataset with pictograms and rain | ATLAS (30 GB, CC BY-NC-SA 4.0) | 33k images, 25 pictogram-state classes |
| Japanese horizontal lights with arrow lamps | nothing public | must be collected by OpenTSR (section 6) |

---

## 1. Dedicated traffic light datasets with state labels

### 1.1 LISA Traffic Light Dataset (USA)

- Origin: UC San Diego LISA lab with Aalborg University. Authors Jensen, Philipsen, Møgelmose, Moeslund, Trivedi. Paper: "Vision for Looking at Traffic Lights: Issues, Survey, and Perspectives", IEEE T-ITS 17(7), pp. 1800-1815, 2016, DOI 10.1109/TITS.2015.2509509 (https://vbn.aau.dk/en/publications/vision-for-looking-at-traffic-lights-issues-survey-and-perspectiv). Companion paper Philipsen et al., IEEE ITSC 2015 (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights).
- Geography: San Diego, California, USA (https://datasetninja.com/lisa-traffic-light).
- Camera: Point Grey Bumblebee XB3 stereo (BBX3-13S2C-60), 66 deg FOV, 24 cm baseline, roof mounted; day shutter 1/5000 s gain 0, night shutter 1/100 s gain 8; 1280x960 per lens; only the left frame is annotated (https://datasetninja.com/lisa-traffic-light).
- Frame rate: the TLR survey (Pavlitska et al.) reports 10 Hz in text and 16 Hz in its table, an internal inconsistency (https://arxiv.org/pdf/2309.02158).
- Size: 43,007 frames and 119,231 annotations per the survey table (https://arxiv.org/pdf/2309.02158); Dataset Ninja gives 43,016 images and cites 113,888 annotated lights in its intro while its object table shows 100,848 objects (https://datasetninja.com/lisa-traffic-light); the Hugging Face YOLO export contains 43,017 images and 109,475 boxes (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights). The counts are not consistent across sources.
- Sequences: 4 continuous test sequences (2 day, 2 night) totalling 23 min 25 s, plus 18 shorter clips; day 24,988 images, night 18,028 images; test 22,481 images, train 20,535 images (https://datasetninja.com/lisa-traffic-light).
- Annotation: bounding boxes with state. Dataset Ninja describes the boxes as drawn "around the illuminated lamp, with its state" (https://datasetninja.com/lisa-traffic-light). 7 classes: go, goForward, goLeft, stop, stopLeft, warning, warningLeft (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights). Arrow direction is encoded in the class (goLeft, stopLeft, warningLeft, goForward). No occlusion flag, no track ID verified. Class balance (Dataset Ninja): go 47,237; stop 44,574; stop left 12,750; go left 2,490; warning 2,750; warning left and go forward counts not shown (https://datasetninja.com/lisa-traffic-light).
- Splits: Kaggle release has day/night train and test clips; the survey notes a "training subset plus private test subset" historically, but the Kaggle/HF releases ship test labels (test split 22,481 images with 57,649 boxes on HF) (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights). HF YOLO export: train 17,082 / val 3,454 / test 22,481 images.
- License: "CC BY-NC-SA 4.0" (https://datasetninja.com/lisa-traffic-light); HF card: "cc-by-nc-sa-4.0", expanded as "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)" (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights). Non-commercial; derived labels must be shared under the same NC-SA terms.
- Access: original on Kaggle, kaggle.com/mbornoe/lisa-traffic-light-dataset (listed by the Aalborg publication record https://vbn.aau.dk/en/publications/vision-for-looking-at-traffic-lights-issues-survey-and-perspectiv; the Kaggle page itself could not be fetched in this session). Mirrors: Hugging Face dronefreak/LISA-Traffic-Lights, YOLO box export, 5.06 GB, 993 downloads last month, "box-annotation-only subset" (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights); Dataset Ninja Supervisely format 5.12 GB (https://datasetninja.com/lisa-traffic-light); multiple Roboflow re-uploads such as leoli/lisa-traffic-light-data-set, which claims "CC BY 4.0" and has no images displayed, conflicting with the upstream NC license (https://universe.roboflow.com/leoli/lisa-traffic-light-data-set).
- Baselines: survey Table II: Lu et al. (Faster R-CNN attention) 91.1% mAP; Mostafa and Ghantous (YOLOv4) 92.16% mAP; Jensen et al. (modified YOLOv2) 90.49% AUC; Wang et al. (improved YOLOv4) 82.15% mAP (https://arxiv.org/pdf/2309.02158). TigAug (Lu et al., Fudan): mAP@[.50,.95] on their 4:1:1 resplit after dedup: YOLOv5 0.615, YOLOX 0.624, Faster R-CNN 0.459, SSD 0.315 (https://arxiv.org/pdf/2507.05932). TLD-READY uses 7 classes with 10,220 / 2,555 / 3,348 images (https://arxiv.org/pdf/2409.07284).
- Known problems: ATLAS authors state LISA "has unlabeled traffic lights and boxes that often do not fit the lights" (https://arxiv.org/html/2504.19722v1); heavy frame duplication (TigAug reduced 43,007 to 36,265 images after removing duplicates, https://arxiv.org/pdf/2507.05932); inconsistent published counts; most annotations in the upper-right of frames (https://datasetninja.com/lisa-traffic-light).

### 1.2 Bosch Small Traffic Lights Dataset (BSTLD, USA)

- Origin: Bosch North America Research (Palo Alto) and Heidelberg Collaboratory for Image Processing (HCI). Behrendt, Novak, Botros, "A deep learning approach to traffic lights: Detection, tracking, and classification", IEEE ICRA 2017, pp. 1370-1377 (https://www.github.com/bosch-ros-pkg/bstld; https://zenodo.org/records/12706046).
- Geography: Palo Alto, California, USA (https://ml4ad.github.io/files/papers2023/Camera-based Context-aware Traffic Light Detection for Self-Driving Vehicles.pdf); "California daytime images" (https://github.com/Kairos-Automotive/TL-detection-segmentation).
- Camera: 1280x720; raw 12-bit HDR with red-clear-clear-blue filter ("riib"), 8-bit RGB reconstructed; RGB conversion "can introduce artifacts" (https://zenodo.org/records/12706046).
- Size: 13,427 images, about 24,000 annotated lights (24,242 per survey https://arxiv.org/pdf/2309.02158; 24,241 per https://datasetninja.com/bstld). Train: 5,093 images annotated about every 2 s, 10,756 lights, 15 labels. Test: 8,334 consecutive images at about 15 fps, 13,486 lights (Zenodo) or 13,493 (Dataset Ninja), 4 labels: red, yellow, green, off (https://zenodo.org/records/12706046; https://datasetninja.com/bstld).
- Annotation: bounding box of the whole housing with state of the active light; occluded flag (170 partially occluded in train, 2,088 in test) (https://hci.iwr.uni-heidelberg.de/node/6132). Training label types and counts (train/test): Red 3,057/5,321; RedStraight 9/0; RedStraightLeft 1/0; RedLeft 1,092/0; RedRight 5/0; Yellow 444/154; Green 5,207/7,569; GreenStraight 20/0; GreenStraightLeft 1/0; GreenStraightRight 3/0; GreenLeft 178/0; GreenRight 13/0; off 726/449 (https://datasetninja.com/bstld). Arrow information exists only in train. No track IDs verified.
- Splits: official train and test; test labels are public (test.yaml; dataset_test_rgb.zip 12.2 GB on Zenodo). README asks not to train on the test set (https://www.github.com/bosch-ros-pkg/bstld).
- License: Zenodo record license titled "LICENSE FOR NONCOMMERCIAL PURPOSES"; description: "released explicitly for non-commercial use only" (https://zenodo.org/records/12706046). The license PDF: "Licensor means Robert Bosch LLC"; permits to "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only" and to "produce, reproduce, and Share Adapted Material for NonCommercial purposes only"; attribution and license notice required (https://graphmod.iwr.uni-heidelberg.de/system/files/private/datasets/1639193981/non-commercial_license.pdf). Code in bosch-ros-pkg/bstld: MIT license (https://www.github.com/bosch-ros-pkg/bstld). Note: the TLR survey lists BSTLD's license as "MIT", which refers to the code, not the data (https://arxiv.org/pdf/2309.02158).
- Access: Zenodo DOI 10.5281/zenodo.12706046, 37.7 GB total (train rgb 6.2 GB, train riib 6.9 GB, test rgb 12.2 GB, test riib 11.9 GB, additional 276.8 MB + 293.9 MB), no password (https://zenodo.org/records/12706046; https://hci.iwr.uni-heidelberg.de/node/6132). Mirrors: Hugging Face shpotes/bosch-small-traffic-lights-dataset hosts train.tar.gz 6.16 GB and test.tar.gz 12.2 GB with license tag "other" and loader string "non-commercial use only" (https://huggingface.co/datasets/shpotes/bosch-small-traffic-lights-dataset/tree/main; https://huggingface.co/datasets/shpotes/bosch-small-traffic-lights-dataset/blob/main/bosch-small-traffic-lights-dataset.py). Roboflow re-upload trafficlights-ds/bosch-small-traffic-lights-dd claims "CC BY 4.0" with no images displayed, conflicting with the upstream license (https://universe.roboflow.com/trafficlights-ds/bosch-small-traffic-lights-dd). Dataset Ninja page shows a placeholder "ADD CUSTOM LICENSE MANUALLY" (https://datasetninja.com/bstld).
- Baselines (official README, self reported, test set without empty frames): SSD Mobilenet V1 weighted mAP 0.60, mAP 0.41 (off 0.00, green 0.68, yellow 0.41, red 0.55); Faster RCNN NAS-A weighted mAP 0.65, mAP 0.43; Hierarchical Deep Architecture weighted mAP 0.53; Baseline 0.36 (https://www.github.com/bosch-ros-pkg/bstld). Sanitz, Wilms, Frintrop: proposed 0.572 mAP / 0.710 mAP_weighted (https://arxiv.org/pdf/2307.15191). TigAug mAP@[.50,.95]: YOLOv5 0.283, YOLOX 0.220, Faster R-CNN 0.155, SSD 0.057 (https://arxiv.org/pdf/2507.05932). Survey: Kim et al. 95.10% F1; Weber et al. HDTLR 85.8% F1; Liu and Li 81.5% mAP (https://arxiv.org/pdf/2309.02158). TLD-READY split: 4 classes, 4,329 / 764 / 8,334 (https://arxiv.org/pdf/2409.07284).
- Known problems: 13 to 15 label types in train vs 4 in test; test split larger than train; strong imbalance (yellow 598, off 1,168 total); lights labelled "off" when they only appeared off due to camera sampling vs signal refresh (https://datasetninja.com/bstld; https://arxiv.org/pdf/2409.07284). Original Heidelberg download was replaced by Zenodo in July 2024 (https://zenodo.org/records/12706046).

### 1.3 DriveU Traffic Light Dataset (DTLD, Germany)

- Origin: Ulm University, Institute of Measurement, Control and Microtechnology (driveU project). Fregin, Müller, Kreßel, Dietmayer, "The DriveU Traffic Light Dataset: Introduction and Comparison with Existing Datasets", IEEE ICRA 2018, pp. 3376-3383, DOI 10.1109/ICRA.2018.8460737 (https://github.com/julimueller/dtld_parsing; https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/).
- Geography: 11 German cities: Berlin, Bochum, Bremen, Dortmund, Duesseldorf, Essen, Frankfurt, Fulda, Hannover, Kassel, Koeln (https://github.com/julimueller/dtld_parsing).
- Camera: stereo, 2 MP, 15 Hz, .tiff images, disparity images and calibration provided; per-image timestamp, GPS, velocity, yaw rate (https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/). Resolution 2048x1024 (https://arxiv.org/pdf/2309.02158).
- Size: 40,979 images; 232,039 annotations in v1.0 and 292,245 in v2.0 (https://arxiv.org/pdf/2309.02158); the website says "more than 230 000 annotated traffic lights" (https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/). TL-SSD paper: train 28,526 images, 1,478 sequences, 159,902 objects; evaluation 12,453 images, 632 sequences, 72,137 objects (https://arxiv.org/pdf/1805.02523). v2.0 changed the label format only; "the amount of labels and images did not change" (https://github.com/julimueller/dtld_parsing).
- Annotation: bounding box (top-left, width, height) with attributes: direction/viewpoint (front, back, left, right), relevance (applies to ego route), occlusion (occluded and truncated), orientation (horizontal/vertical), aspects (1, 2 or 4 lamps), state (including the red-yellow transition), pictogram (circular, arrow with direction, pedestrian, bike, tram), reflection (added in v2.0); track identity per instance through a sequence; labels down to about 5 px width (https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/; https://github.com/julimueller/dtld_parsing). Some v2.0 labels have "unknown" for aspects, state or pictogram (https://github.com/julimueller/dtld_parsing). Housing-level boxes with per-pictogram attributes. v1.0 used a six-digit class identity in YAML; v2.0 uses a JSON attribute dictionary.
- Splits: train and evaluation files are provided (see TL-SSD numbers above). Test labels are available to registered users (TLD-READY and ATLAS evaluate on the DTLD test split: https://arxiv.org/pdf/2409.07284; https://arxiv.org/html/2504.19722v1).
- License (Terms of Use, quoted): "the licensee has the right to use the DriveU Traffic Light Dataset (DTLD) permanent, royalty-free and non-exclusive for own research and teaching purposes." "Any commercial use or the disclosure of data to third parties (without charge or paid) is prohibited." Publications must cite DOI 10.1109/ICRA.2018.8460737 and must blur people and license plates. No warranty (https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/).
- Access: registration form (first/last name, email, organisation type Company / University / Research Institution, address, accept terms); download behind a login (https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/). Also referenced as www.traffic-light-data.com (https://github.com/julimueller/dtld_parsing). Parsing tools: julimueller/dtld_parsing (C++ and Python; MATLAB deprecated). No Hugging Face or Kaggle mirror was found, consistent with the no-disclosure clause.
- Baselines: TLD-READY (20 classes, 22,820 / 5,705 / 12,453): YOLOv8x mAP50 0.61, P 0.75, R 0.54, 3-state mAP 0.72; YOLOv7x mAP50 0.60; YOLOv8m 0.58; RT-DETR-X 0.26 (https://arxiv.org/pdf/2409.07284). ATLAS paper on DTLD test: YOLO11x mAP50 0.64 / mAP50-95 0.46 trained on DTLD only; 0.66 / 0.49 trained on DTLD+ATLAS (https://arxiv.org/html/2504.19722v1). Sanitz et al.: 0.552 mAP vs TL-SSD 0.329 (https://arxiv.org/pdf/2307.15191). Survey: Possatti et al. YOLOv3 with prior maps 85.62% mAP; Bach et al. 83% mAP; Müller and Dietmayer 95% recall (https://arxiv.org/pdf/2309.02158). TL-SSD LAMR at IoU 0.5 about 0.015 to 0.02 (https://arxiv.org/pdf/1805.02523). Relevance classifier (gradient boosting on arrows) P 0.96 R 0.96 (https://arxiv.org/pdf/2409.07284).
- Known problems: 423 (v1.0) / 620 (v2.0) unique attribute combinations; v2.0 partly labelled by a neural network (semi-automatic) (https://arxiv.org/pdf/2309.02158); "varying label rules", turned lights annotated inconsistently (https://arxiv.org/pdf/1805.02523); no straight-left / straight-right pictogram examples (https://arxiv.org/pdf/2409.07284); redistribution of any data, including derived labels, is prohibited.

### 1.4 S2TLD (SJTU Small Traffic Light Dataset, China)

- Origin: Thinklab, Shanghai Jiao Tong University (Xue Yang, Junchi Yan et al.) with Anhui University; the README asks to cite SCRDet++ (Yang et al., IEEE TPAMI 2022, arXiv 2004.13316) (https://github.com/Thinklab-SJTU/S2TLD; https://datasetninja.com/s2tld).
- Geography: not stated on the README or Dataset Ninja; institutions are in China (https://datasetninja.com/s2tld).
- Camera: unknown. Two resolution subsets: 1080x1920 (1,222 images) and 720x1280 (4,564 images) (https://datasetninja.com/s2tld).
- Size: 5,786 images; 14,130 instances (README writes "1,4130 instances", a typo) (https://github.com/Thinklab-SJTU/S2TLD; https://datasetninja.com/s2tld).
- Annotation: bounding boxes with 5 classes: red 7,708; green 5,344; off 507; wait on 306; yellow 265 (https://datasetninja.com/s2tld). No arrow or pictogram labels, no occlusion, no track IDs. 3 images unlabelled; minimum box height 1 px (https://datasetninja.com/s2tld).
- Splits: none predefined (https://datasetninja.com/s2tld). Third-party resplit: "Repartition of S2TLD dataset" by Gang Liu, 7:2:1 train/val/test in COCO format, coco-s2tld.zip 1.4 GB, license "Creative Commons Attribution 4.0 International", DOI 10.5281/zenodo.7352960 (https://zenodo.org/records/7352960).
- License: GitHub shows "MIT license" with a LICENSE file (https://github.com/Thinklab-SJTU/S2TLD); Dataset Ninja: "under MIT license" (https://datasetninja.com/s2tld); survey table: MIT (https://arxiv.org/pdf/2309.02158). The Hugging Face mirror yangxue/S2TLD is tagged apache-2.0, which does not match the GitHub MIT license (https://huggingface.co/datasets/yangxue/S2TLD/tree/main).
- Access: Baidu Drive, OneDrive and Hugging Face links in the README (https://github.com/Thinklab-SJTU/S2TLD). HF yangxue/S2TLD: S2TLD(1080x1920).zip 305 MB and S2TLD(720x1280).zip 1.12 GB, both subsets present (https://huggingface.co/datasets/yangxue/S2TLD/tree/main).
- Baselines: Zhao et al. (YOLOv4 with ShuffleNetv2) 71.24% mAP (https://arxiv.org/pdf/2309.02158). No official baseline in the README.
- Known problems: small, strongly imbalanced (yellow and wait_on rare), no official split, cited paper is about rotated object detection rather than this dataset.
- Measured locally (2026-10-31, after downloading from the Hugging Face mirror; `scripts/prepare_s2tld.py`): 5,786 images and 14,130 boxes, matching the paper. Per subset: 720x1280 `normal_1` 779 frames / 1,978 boxes, 720x1280 `normal_2` 3,785 frames / 9,716 boxes, 1080x1920 1,222 frames / 2,436 boxes. Class counts: red 7,708, green 5,344, off 507, wait_on 306 (1080 subset only), yellow 265. Boxes are PASCAL VOC XML at housing level; frames are consecutive video frames (the 1080 subset carries timestamps about 0.2 s apart), so random splits leak near-duplicates. The GitHub LICENSE file is the MIT license text with the copyright line "Copyright (c) 2018 DetectionTeamUCAS", i.e. copied from the authors' detection code repository; the README has no separate data-license statement. The SCRDet++ paper evaluates S2TLD with a random 6:4 train/test split and reports VOC-style AP (%) at IoU 0.5 for RetinaNet (mAP 93.25, with their InLD module 94.11) and FPN (94.93, with InLD 95.94); per-class AP for FPN without InLD: red 97.98, yellow 87.55, green 97.42, off 93.42, wait on 98.31 (Table 4 of https://arxiv.org/pdf/2004.13316).

### 1.5 WPI Traffic Light Dataset (USA)

- Origin: Worcester Polytechnic Institute. Chen and Huang, "Accurate and Reliable Detection of Traffic Lights Using Multiclass Learning and Multiobject Tracking", IEEE Intelligent Transportation Systems Magazine 8(4), pp. 28-42, 2016; dataset stated at http://computing.wpi.edu/Dataset.html (https://trid.trb.org/View/1428826).
- Geography: USA (survey); city not stated on fetched pages.
- Size: 3,456 images, 6,766 annotations, 1920x1080, "21 and 2" classes, pictograms yes, license N/A (survey Table I, https://arxiv.org/pdf/2309.02158). Includes circular and arrow lights, green and red (https://trid.trb.org/View/1428826).
- Files: Test.zip 23.51 GB, Samples.zip 35.64 GB, Frames_GT_Lights.zip 25.04 GB, Frames_GT_wHolder.zip 35.56 GB; sequences 8, 9, 10 contain no traffic lights; two GT variants (with holder vs lights only) (https://hyper.ai/en/datasets/16682).
- Splits: unknown. License: none stated (https://hyper.ai/en/datasets/16682). Access: computing.wpi.edu/dataset.html (not fetched in this session) or HyperAI mirror.
- Baselines: Ouyang et al. 99.7% accuracy shared across WPI, LISA and a private set (https://arxiv.org/pdf/2309.02158).
- Known problems: small, no license, original host availability uncertain, huge file sizes relative to image count (raw video frames).

### 1.6 LaRA Traffic Lights Recognition benchmark (France)

- Origin: Robotics Centre of Mines ParisTech and Inria IMARA; Raoul de Charette and Fawzi Nashashibi; related papers IEEE IV 2009 and IROS 2009 (https://topic.alibabacloud.com/a/traffic-lights-identify-open-font-classtopic-s-color00c1dedatasetsfont-traffic-lights-recognition-tlr-public-benchmarks_8_8_10269246.html).
- Geography: Paris, France, dense urban (same URL).
- Camera: Marlin F-046C, 25 Hz, 12 mm lens, behind the rear-view mirror of the C3 vehicle; 640x480 RGB 8 bit; 11,179 frames, 8 min 49 s (same URL).
- Annotation: 9,168 hand-labelled lights (3,381 green, 5,280 red, 449 ambiguous; orange by subtraction about 58); classes green/orange/red/ambiguous; only circular lights labelled; minimum 5 px width (same URL). Survey: 4 classes, no pictograms (https://arxiv.org/pdf/2309.02158).
- Splits: none stated. License: none named; "available publicly and for free", "read the copyrights information before any use" (alibabacloud reprint, same URL).
- Access: the original Mines ParisTech page URL was not recoverable from fetched pages; the above is a 2018 reprint. Treat the official link as unverified.
- Baselines: Wang et al. improved YOLOv4 79.97% mAP; Bali et al. SqueezeNet with YOLOv2 84% mAP (https://arxiv.org/pdf/2309.02158).
- Known problems: low resolution, 2009 vintage, no arrow signals, unclear license, possibly dead official link.

### 1.7 ATLAS: Applied Traffic Light Annotation Set (Germany, 2025)

- Origin: FZI Research Center for Information Technology and Karlsruhe Institute of Technology. Rupert Polley, Nikolai Polley, Heid, Heinrich, Ochs, Zöllner, "The ATLAS of Traffic Lights: A Reliable Perception Framework for Autonomous Driving", IEEE Intelligent Vehicles Symposium 2025, pp. 2215-2222, DOI 10.1109/IV64158.2025.11097347 (https://publikationen.bibliothek.kit.edu/1000183940); arXiv 2504.19722 (https://arxiv.org/html/2504.19722v1).
- Geography: Karlsruhe, Germany (https://zenodo.org/records/14846709).
- Camera: CoCar NextGen, three synchronized cameras at 20 Hz, annotated at 2 Hz: Front-Medium 61x39 deg, 1920x1200, 25,158 images; Front-Tele 31x20 deg, 1920x1200, 5,109 images; Front-Wide 106x92 deg, 2592x2048, 2,777 images; 2,776 images in medium to heavy rain (https://arxiv.org/html/2504.19722v1; https://zenodo.org/records/14846709).
- Size: 33,044 images; 72,998 bounding boxes (https://zenodo.org/records/14846709).
- Annotation: states green, red, yellow, red-yellow; pictograms circle, off, straight, left, right, straight-left, straight-right; 25 pictogram-state classes; labelled from 2 to 4 px width; iterative model-assisted labelling with a second-person review; faces and plates blurred (https://arxiv.org/html/2504.19722v1). YOLO format txt labels; class names in ATLAS_classes.yaml (https://zenodo.org/records/14846709). No occlusion flag or track ID described.
- Splits: train and test folders (train has front_medium, front_tele, front_wide subfolders); sizes not stated; no validation split mentioned (https://zenodo.org/records/14846709). Test labels appear to be included in the archive (the paper evaluates on "both test splits").
- License: "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)"; "may be used exclusively for non-commercial purposes" (https://zenodo.org/records/14846709).
- Access: Zenodo DOI 10.5281/zenodo.14846709, version 1.0 (published 2025-02-10), 29.7 GB in ATLAS.z01 to z05 (5.4 GB each) plus ATLAS.zip 2.6 GB, ATLAS_sample.zip 25.3 MB, preview mp4 186 MB (https://zenodo.org/records/14846709); older v0.1 record 14775869 holds only the sample (https://zenodo.org/records/14775869). Short link url.fzi.de/ATLAS (https://publikationen.bibliothek.kit.edu/1000182830). No HF or Kaggle mirror found.
- Baselines (mAP50 / mAP50-95 on ATLAS test): YOLOv8x trained on DTLD only 0.53 / 0.34; YOLOv8x DTLD+ATLAS 0.70 / 0.52; YOLOv9c DTLD+ATLAS 0.72 / 0.53; YOLO11x DTLD+ATLAS 0.72 / 0.54 (https://arxiv.org/html/2504.19722v1).
- Known problems: large class imbalance across the 25 classes; split sizes undocumented; NC license.

### 1.8 aiMotive 3D Traffic Light and Traffic Sign Dataset (USA + Hungary, 2024)

- Origin: aiMotive, Budapest. Kunsági-Máté, Pethő, Seres, Matuszka, "Accurate Automatic 3D Annotation of Traffic Lights and Signs for Autonomous Driving", ECCV 2024 Workshop on Vision-Centric Autonomous Driving, arXiv 2409.12620 (https://uithub.com/aimotive/aimotive_tl_ts_dataset; https://arxiv.org/html/2409.12620v3).
- Geography: California, USA and Hungary; urban and highway (https://arxiv.org/html/2409.12620v3).
- Camera: four Sony IMX490 cameras (front long range, front mid range, left and right cross traffic), 2896x1876, 30 to 40 Hz; Novatel PwrPak7 GNSS/INS (https://uithub.com/aimotive/aimotive_tl_ts_dataset).
- Size: about 50,000 auto-annotated frames from 220 sequences of 15 s (55 min); about 320,000 traffic light 3D boxes and 550,000 traffic sign boxes; ODD folders highway, night, rainy, urban (https://arxiv.org/html/2409.12620v3; https://uithub.com/aimotive/aimotive_tl_ts_dataset).
- Annotation: 3D boxes up to 200 m, traffic light colour state and mask type (pictogram, for example forward arrow), occlusion; generated automatically from 2D detections plus GNSS/INS, without LiDAR (https://arxiv.org/html/2409.12620v3; https://uithub.com/aimotive/aimotive_tl_ts_dataset).
- Splits: unknown.
- License: "Attribution-NonCommercial-ShareAlike 4.0 International"; "made freely available for non-commercial research purposes only" (https://uithub.com/aimotive/aimotive_tl_ts_dataset).
- Access: https://aimotive-dataset.s3.amazonaws.com/aimotive_tl_ts_dataset.zip (about 107 GB), also "from Kaggle" per the README (Kaggle page not verified) (https://uithub.com/aimotive/aimotive_tl_ts_dataset).
- Baselines: no detector baseline; annotation-method evaluation on a Palo Alto route: association precision 91.13%, recall 95.87%, colour state accuracy 94% (https://arxiv.org/html/2409.12620v3).
- Known problems: labels are automatic; ATLAS authors report they "contain many errors, especially for states" (https://arxiv.org/html/2504.19722v1).

### 1.9 VZC-TLD: Verizon Connect Traffic Light Dataset (USA, 2025)

- Origin: Verizon Connect Research (Florence) and University of Florence. Trinci, Magistri, Bianconcini, Taccari, Sarti, Sambo, "Color Is Not Enough: Dataset and Method for Identifying Relevant Traffic Lights in Driving Scenes", IEEE T-ITS vol. 27 no. 1, pp. 1116-1125, DOI 10.1109/TITS.2025.3626165 (https://api.openalex.org/works/doi:10.1109%2FTITS.2025.3626165).
- Geography: diverse US locations, day and night (https://huggingface.co/datasets/vzc-research-chapter/vzc-traffic-light-dataset).
- Size: about 3,000 images from 2,856 intersections; 25,624 traffic light annotations; COCO format; 2.59 GB (same URL).
- Annotation: bounding boxes; 8 categories combining relevance (relevant / not relevant) with state (unknown, green, yellow, red); no arrow annotations mentioned (same URL).
- Splits: unknown. License: "cc-by-nc-4.0", "Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)" (same URL). Gated on Hugging Face (must accept conditions and share contact info).
- Access: https://huggingface.co/datasets/vzc-research-chapter/vzc-traffic-light-dataset; project page https://leotac.github.io/vzc-tld (which links a different HF path, florence-research-chapter, that may be stale).
- Baselines: only qualitative claims in the abstract; no public numbers found (https://api.openalex.org/works/doi:10.1109%2FTITS.2025.3626165).

### 1.10 LAVA Salient Lights Dataset (USA)

- Origin: UCSD LISA. Greer, Gopalkrishnan, Landgren, Rakla, Gopalan, Trivedi, "Robust Traffic Light Detection Using Salience-Sensitive Loss", arXiv 2305.04516, 2023; "first US traffic light dataset with an annotated salience property" (https://arxiv.org/abs/2305.04516v1).
- Size: 30,566 signals of which 9,051 important, per a third-party summary (https://ml4ad.github.io/files/papers2023/Camera-based Context-aware Traffic Light Detection for Self-Driving Vehicles.pdf).
- License, download URL, splits: unknown (the TIB record page returned a bot check: https://service.tib.eu/ldmservice/dataset/lava-salient-lights-dataset). Model: Deformable DETR with salience-sensitive focal loss; no numeric results in the abstract.

### 1.11 aUToLights (Canada / USA)

- Origin: aUToronto, University of Toronto Institute for Aerospace Studies. Wu, Amenta, Zhou, Papais, Kelly, "aUToLights: A Robust Multi-Camera Traffic Light Detection and Tracking System", arXiv 2305.08673 (https://arxiv.org/pdf/2305.08673; https://www.autodrive.utoronto.ca/datasets-autolights).
- Geography: UTIAS closed track (Toronto) and MCity (Ann Arbor, Michigan); lights follow Michigan DOT standards (https://arxiv.org/pdf/2305.08673; https://www.autodrive.utoronto.ca/datasets-autolights).
- Camera: two front Lucid Vision cameras, 7.1 MP, 16 mm and 8 mm lenses (https://www.autodrive.utoronto.ca/datasets-autolights).
- Size: about 47,550 labelled instances (train about 22,027, test about 25,525); test set about 10,000 images (https://arxiv.org/pdf/2305.08673).
- Annotation: 13 classes: 3-green, 3-red, 3-yellow, 4-gleft, 4-off, 4-rleft, 4-yleft1, 4-yleft2, 5dh-green, 5dh-red, 5dh-red-gleft, 5dh-red-yleft, 5dh-yellow (arrow states included); human labelled 2D boxes (https://arxiv.org/pdf/2305.08673; https://www.autodrive.utoronto.ca/datasets-autolights).
- License: not stated. Access: request form on the aUToronto page (https://www.autodrive.utoronto.ca/datasets-autolights).
- Baselines: YOLOv5n mAP 0.77 / AP50 0.93 / AP75 0.89 at 163 FPS; YOLOv5s 0.77 / 0.93 / 0.86; YOLOv5m 0.75 / 0.92 / 0.86 (https://arxiv.org/pdf/2305.08673).
- Known problems: test set has no 4-off labels; test larger than train; closed-course data.

### 1.12 DualCam (2022/2023)

- Origin: Jayarathne, Samarakoon, Koralege, Divisekara, Rodrigo, Jayasekara, "DualCam: A Novel Benchmark Dataset for Fine-Grained Real-Time Traffic Light Detection", ICMLA 2023, pp. 1778-1783, DOI 10.1109/ICMLA58977.2023.00270 (https://github.com/harinduravin/DualCam). Country not stated on the fetched pages (https://arxiv.org/abs/2209.01357v1).
- Camera: two synchronized Basler daA1920-30uc cameras at 30 fps, narrow (6 mm) and wide (Theia SY125) lenses; 1920x1080 (https://github.com/harinduravin/DualCam).
- Size: 2,250 annotated images, 8,321 instances; train 1,032 images, test set 1 335 pairs, test set 2 478 pairs, 28 test video pairs (https://github.com/harinduravin/DualCam).
- Annotation: 10 classes: Green, Red, Green-up, Empty-count-down, Count-down, Yellow, Empty, Green-right, Green-left, Red-yellow (https://github.com/harinduravin/DualCam).
- License: not stated. Access: Google Drive links in the README. Baselines: none on the README.

### 1.13 Cityscapes TL++ (Germany, 2022)

- Origin: Johannes Janosovits (KIT MRT), "Cityscapes TL++: Semantic Traffic Light Annotations for the Cityscapes Dataset", ICRA 2022, pp. 2569-2575, DOI 10.1109/ICRA46639.2022.9812144 (https://www.mrt.kit.edu/english/alumni_janosovits.php).
- Size and labels: 5,000 Cityscapes images; 6 state classes red, red-yellow, yellow, green, off, unknown; 2048x1024; license listed as "LGPL-2.1" in the survey table (https://arxiv.org/pdf/2309.02158). Sanitz et al. use the states stop, warning, go, unknown and report Faster R-CNN+FPN per-state AP: stop 0.645, warning 0.381, go 0.626, unknown 0.332 (https://arxiv.org/pdf/2307.15191).
- Access: download URL not found on KIT pages (https://www.mrt.kit.edu/english/datasets.php lists no TL dataset). Depends on Cityscapes images (non-commercial, registration). Treat as unknown availability.

### 1.14 IRD: Indian Roads Dataset for Supported and Suspended Traffic Lights (India)

- Origin: Sarita Gautam and Anuj Kumar, Panjab University, Chandigarh; arXiv 2209.04203 (https://arxiv.org/pdf/2209.04203).
- Geography: New Delhi (suspended lights) and Chandigarh (supported lights), about 10 km routes each; day and night.
- Camera: 64 MP phone camera in a car; 3840x2160; frame rate given as 20 fps and 3 fps in different tables.
- Size: about 3,050 images (3,334 incl. thumbnails); 8,520 annotations (also "8000+"); class count 13 to 16 depending on section; includes red, yellow, green, green-arrow-light, pedestrian green/red light, timer, plus vehicles and boards.
- Splits, license: not stated. Access: https://sites.google.com/view/ird-dataset/home. Baseline: SSD Inception_v3 97.23% test accuracy (no mAP). Known problems: inconsistent counts, phone camera, small.

### 1.15 Open-source dataset for traffic light and countdown display detection (China, 2025)

- Origin: Jilin University (Wan, Ding, Zhang, Hu) and Tongji University (Hong Chen). "Open-source dataset for traffic light and countdown display detection in urban environment", Science China Information Sciences 68, 157201 (2025), DOI 10.1007/s11432-024-4313-x (https://link.springer.com/article/10.1007/s11432-024-4313-x?code=35b071b2-9982-48c4-9d68-27a8d0f555de&error=cookies_not_supported).
- Geography: Changchun and Wuhan; data collected 2019 to early 2025; 1920x1080 smartphone frames, four seasons (https://lacuna.tiptreesystems.com/work/open-source-dataset-for-traffic-light-and-countdown-display-detection-in-urban/wrk_ae5eaa1444afa7dcf0f37ba67e8e8c77).
- Size: subset A traffic lights 20,295 images with 16 state classes including arrows (left, straight, right, U-turn), lamp off, red, amber, green; subset B 1,000 countdown display images (digits 0-9); subset C 479 bicycle and pedestrian signals; total 21,774; two label sets (primary/secondary vs third to fifth intersections, including extremely small signals); labelled per GB14886-2016 and GA/T 508-2014 (same URL).
- License, download URL, splits, baselines: unknown (the publisher PDF was blocked by robots.txt: http://scis.scichina.com/en/2025/157201.pdf).

### 1.16 COCO Traffic (derived)

- Origin: daved01/cocoTraffic (David Kirchhoff, Philip Hoang). COCO 2017 traffic light boxes refined into traffic_light_red, traffic_light_green, traffic_light_na; mislabelled COCO lights kept as class 10; LISA images labelled with makesense.ai after DETR pre-labelling (https://github.com/daved01/cocoTraffic).
- License: repository shows "MIT license"; annotation files on Google Drive; images from images.cocodataset.org. Image count unknown. Known problems: no arrows, LISA-derived part inherits CC BY-NC-SA.

### 1.17 Udacity Self Driving Car dataset (Roboflow re-release, USA)

- 15,000 images (1,720 null), 97,942 boxes, 11 classes, originals 1920x1200 (3.1 GB) plus 512x512 version, "MIT License", April 2020 (https://public.roboflow.com/object-detection/self-driving-car). The class list (which is known to include traffic light colour classes) was not shown on the fetched page; treat class names as unverified.

### 1.18 Indoor smart traffic testbed (Zenodo, 2022)

- CUHK / CityU HK; 3,507 images from an indoor F1TENTH testbed with toy lights, boxes and state 0 green / 1 red; "Creative Commons Attribution 4.0 International"; 244.6 MB; DOI 10.5281/zenodo.7181314 (https://zenodo.org/record/7181314). Toy domain, not useful for road TL.

---

## 2. General driving datasets with traffic light boxes AND colour state

### 2.1 BDD100K (USA)

- Origin: UC Berkeley (Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning", CVPR 2020); curated by ETH VIS group at github.com/bdd100k/bdd100k (https://huggingface.co/datasets/dgural/bdd100k/blob/main/README.md?code=true; https://www.github.com/bdd100k/bdd100k).
- Labels: box2d categories bike, bus, car, motor, person, rider, traffic light, traffic sign, train, truck; traffic light attribute trafficLightColor in {"red", "green", "yellow", "none"}; occluded and truncated booleans; frame attributes weather, scene, timeofday (https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md). No arrow/pictogram labels, housing-level boxes, no track IDs in the image set.
- Size: 100K videos; frames sampled at the 10th second (https://huggingface.co/datasets/dgural/bdd100k/blob/main/README.md?code=true). Resolution 1280x720 at 30 fps (https://arxiv.org/pdf/2209.04203). Official split sizes (70K/10K/20K) were not verified on any page fetched here.
- Test labels: not verified (the detection test labels are held out for the benchmark server; treat as unknown).
- License: "BDD100K License" with Regents of the University of California copyright; permits educational, research and not-for-profit use without fee; commercial use only for BDD and BAIR Commons members and affiliates (summary from the HF card, https://huggingface.co/datasets/dgural/bdd100k/blob/main/README.md?code=true). Code: BSD-3-Clause (https://www.github.com/bdd100k/bdd100k). The official license page itself was not fetched.
- Access: bdd-data.berkeley.edu registration (not fetched). Mirrors: HF dgural/bdd100k (10,000 samples, tagged "bsd", detections plus weather/time/scene; traffic light colour not mentioned) (https://huggingface.co/datasets/dgural/bdd100k/blob/main/README.md?code=true); HF ShantyCam/bdd100k-det (20,001 rows, 695 MB) (https://huggingface.co/datasets/ShantyCam/bdd100k-det). Neither is confirmed complete.
- Baselines (survey): Yan et al. YOLOv5 63.3% AP; Cai et al. SSDLite 33.84% mAP; Wang et al. 2022 84.7% F1; Wang et al. 98% accuracy (classification) (https://arxiv.org/pdf/2309.02158).
- Known problems: traffic lights are tiny at 720p; "none" colour is frequent; the survey lists it among datasets with state but without orientation/pictogram/relevance (https://arxiv.org/pdf/2309.02158).

---

## 3. Datasets with a traffic light class but NO state label (pretraining or localisation only)

| Dataset | TL class evidence | Images | License (as written) | Access | Source |
|---|---|---|---|---|---|
| Mapillary Vistas v1.2 / v2.0 | "traffic light - general (upright)" 49,431 objects; macro "traffic-light" 79,512 objects; "traffic light - pedestrians" named; no state | 25,000 (18,000 / 2,000 / 5,000; test labels withheld); at least 1920x1080, up to 22 MP; worldwide | v2.0 "under CC BY-NC-SA 4.0 license" (Dataset Ninja); "Non-Commercial" (HyperAI); research edition requested from mapillary.com/dataset/vistas | v1.2.zip 23.86 GB, v2.0.zip 51.14 GB | https://cdn.datasetninja.com/mapillary-vistas-dataset ; https://hyper.ai/en/datasets/17342 ; https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf ; https://github.com/NVIDIA/semantic-segmentation/blob/master/PREPARE_DATASETS.md |
| Cityscapes | "traffic light" class in the object group; no state | 5,000 fine + 20,000 coarse; 50 cities; 2048x1024 | "made freely available to academic and non-academic entities for non-commercial purposes"; "intended for research purposes only"; registration | cityscapes-dataset.com | https://cityscapes-dataset.com/ ; https://www.cityscapes-dataset.com/?p=464 ; https://www.cityscapes-dataset.com/dataset-overview/ |
| COCO 2017 | "traffic light" is category index 10 (refined by COCO Traffic) | 330K images (>200K labelled), 80 categories | HF card tag "cc-by-4.0" (annotations); image terms not verified | cocodataset.org | https://github.com/daved01/cocoTraffic ; https://huggingface.co/datasets/HuggingFaceM4/COCO/blob/main/README.md |
| Open Images V7 | "Traffic light" is class 548 of 601 boxable classes | about 9M images, about 16M boxes | "CC BY 4.0" shown on the Ultralytics page (image-level licence not verified) | storage.googleapis.com/openimages | https://docs.ultralytics.com/datasets/detect/open-images-v7 |
| Objects365 | "Traffic Light" is class 40 of 365 | 1,742,289 train + 80,000 val; about 30M boxes | "CC BY 4.0" (Ultralytics page) | objects365.org | https://docs.ultralytics.com/datasets/detect/objects365/ |
| LVIS | not verified in this session | unknown | unknown | unknown | none |
| ApolloScape | "traffic light" semantic label ID 14 (infrastructure group); no state | 143,906 pixel-labelled frames; 3384x2710 | "For academic purposes only" (CAD catalogue); HyperAI lists "Other" | apolloscape.auto (downloads marked not available on HyperAI) | https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Huang_The_ApolloScape_Dataset_CVPR_2018_paper.pdf ; https://www.connectedautomateddriving.eu/data-sharing/data-catalogue/ ; https://beta.hyper.ai/en/datasets/15988 |
| Audi A2D2 | semantic classes include "Traffic signal 1-3" (undefined in paper); no state | 41,277 semantic frames; 1928x1208; Gaimersheim, Ingolstadt, Munich | abstract: "CC BY-ND 4.0 license, permitting commercial use subject to the terms of the license"; the arXiv page badge links CC BY-NC-SA 4.0 (conflict) | a2d2.audi | https://arxiv.org/pdf/2004.06320v1.pdf ; https://arxiv.org/abs/2004.06320 |
| KITTI-360 | TRAFFIC_LIGHT = 12 in the box detection labels; no state | over 320k images; Karlsruhe | "Creative Commons Attribution-NonCommercial-ShareAlike 3.0"; registration with stated purpose (GDPR) | cvlibs.net | https://www.cvlibs.net/datasets/kitti-360 ; https://autonomousvision.github.io/py123d/datasets/kitti-360 |
| Zenseact Open Dataset (ZOD) | 2D class "Traffic Signal" with attributes Unclear, OcclusionRatio, IsForEgoRoad, IsForEgoLane etc.; no colour/state attribute | ZOD Frames 100k; 14 European countries; 8 MP 3848x2168 | "CC BY-SA 4.0" (paper); dataset notice must be kept; devkit MIT; access by application for a download link | zod.zenseact.com; HF Zenseact/ZOD | https://arxiv.org/pdf/2305.02008 ; https://openaccess.thecvf.com/content/ICCV2023/supplemental/Alibeigi_Zenseact_Open_Dataset_ICCV_2023_supplemental.pdf ; https://pypi.org/project/zod |

---

## 4. Datasets checked that have NO traffic light image labels

- nuScenes / nuImages: 25 classes, no traffic light class; 93,476 images; Boston and Singapore; "CC BY-NC-SA 4.0" (https://datasetninja.com/nu-images).
- Waymo Open Dataset Perception: box classes TYPE_UNKNOWN, TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_SIGN, TYPE_CYCLIST; traffic lights "n/a"; license "Waymo Dataset License Agreement for Non-Commercial Use" (https://autonomousvision.github.io/py123d/datasets/wod_perception). Current terms: "Waymo Dataset License Agreement for Non-Commercial Use (March 2025)"; redistribution only to registered users; trained models and derived datasets may be published with notice (https://waymo.com/open/terms/).
- Waymo Open Motion Dataset: "Traffic signal states and the lanes they control are included" (https://arxiv.org/pdf/2104.10133); 103,354 segments of 20 s at 10 Hz, 70/15/15 split; raw camera images are NOT included, only image tokens and VQ-GAN embeddings for the first second (https://waymo.com/intl/en-ca/open/data/motion/). Useful for lane-level signal semantics, not for image training. Derived trajectory dataset: "Interaction Dataset of Autonomous Vehicles with Traffic Lights and Signs" (Tongji / UW-Madison / UGA, Communications in Transportation Research 2025), about 37,000 TL interaction trajectories with state codes 0-8, CSV only, https://github.com/CATS-Lab/Data-AV-Traffic-Light-Stop-Sign (https://arxiv.org/pdf/2501.12536).
- Argoverse 2 Sensor: 30 classes, only TRAFFIC_LIGHT_TRAILER (a vehicle), traffic lights "n/a"; 1000 logs; "CC BY-NC-SA 4.0" (https://autonomousvision.github.io/py123d/datasets/av2).
- ONCE (Huawei): 3D classes Car, Bus, Truck, Pedestrian, Cyclist; 7M images; no license on the page (https://once-for-auto-driving.github.io).
- Lyft Level 5 / Woven Planet prediction dataset: semantic map holds traffic lights with "position, lane association" and each scene captures "traffic lights state"; no raw camera images in the release; 1,118 hours, Palo Alto; license not stated on the fetched page (https://ar5iv.arxiv.org/html/2006.14480).
- SHIFT (CARLA synthetic): 2D boxes for car, truck, bus, bicycle, motorcycle, pedestrian only; panoptic labels use the 23 Cityscapes classes; about 2.5M frames (https://arxiv.org/pdf/2206.08367).

---

## 5. Pedestrian datasets that annotate traffic lights

### 5.1 PIE (Pedestrian Intention Estimation, Toronto)

- York University (Rasouli, Kotseruba, Kunic, Tsotsos, ICCV 2019). 909,480 frames (293,437 annotated), over 6 hours, 1920x1080 at 30 fps, Waylens Horizon camera (157 deg), downtown Toronto, daytime; 1,842 pedestrians with 738,970 boxes; 2,353,983 traffic object boxes in total (traffic lights, signs, zebra crossings; per-type counts not stated); pedestrian samples split 50% train / 40% test / 10% val over 6 sets (https://data.nvision2.eecs.yorku.ca/PIE_dataset; https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf).
- License: "MIT License, covering videos and annotations" (https://data.nvision2.eecs.yorku.ca/PIE_dataset); repo "licensed under the MIT License" (https://github.com/aras62/PIE). Videos about 74 GB from the YorkU server or Google Drive.
- Traffic light type (regular/transit/pedestrian) and state (red/yellow/green) fields are part of the PIE annotation schema per the project documentation, but the annotations README could not be fetched in this session, so these fields are listed as not verified here.

### 5.2 JAAD (Joint Attention in Autonomous Driving)

- 346 clips; per-frame "Traffic" annotations covering signs and traffic lights (state/box detail not described on the README); video-level weather, time of day, location attributes. License: code and annotations "licensed under the MIT License"; "The video clips are licensed under Creative Commons Attribution 4.0 International License"; about 3.1 GB (https://github.com/ykotseruba/JAAD).

---

## 6. Japan: what exists publicly (important gap for OpenTSR)

Japanese signals are horizontal with arrow lamps (left / straight / right, and up-left / up-right variants used in Autoware's label set). Findings:

- TIER IV / Autoware (internal, NOT public): vehicular classifier trained on 83,400 TIER IV internal images (58,600 / 14,800 / 10,000), pedestrian classifier on 21,199 (17,860 / 2,114 / 1,225); car classes: green, yellow, red, unknown, "left,red", "left,red,straight", "red,right", "red,straight", "red,up_left", "red,right,straight", "red,up_right"; pedestrian: red, green, unknown; MobileNetV2 accuracy 99.81% (vehicular) and 99.10% (pedestrian); models released under "Apache-2.0" (https://huggingface.co/AutowareFoundation/traffic_light_classifier). YOLOX-s fine detector fine-tuned on about 17,000 internal Japanese images (https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_traffic_light_fine_detector). Public samples only: a 1,045-image red/green/yellow classifier sample (https://huggingface.co/AutowareFoundation/traffic_light_classifier) and a 1,062-image cropped YOLOX sample with Pascal VOC labels, 3 MB, from an Autoware S3 bucket (https://docs.autoware.org/main/tutorials/training-machine-learning-models/training-models/). No "T4 dataset" traffic light release is named on these pages.
- Turing Inc. (Tokyo): 44,000 images, 19,509 labelled signals, about 1,000 hours of footage, primarily Tokyo; YOLOX-X; dataset URL not given, not released (https://ml4ad.github.io/files/papers2023/Camera-based Context-aware Traffic Light Detection for Self-Driving Vehicles.pdf). JSAI 2023 paper by the same group: about 15,000 annotated images from over 900 hours; public release not stated (https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2M6GS1002/_article/-char/en).
- Nagoya University (Murase lab, bachelor thesis): 480 position-annotated images and 15,195 colour-classifier images, not released (https://rum.vislab.is.i.nagoya-u.ac.jp/~murase/pdf/1577-pdf.pdf).
- Roboflow Universe Japanese-titled projects: "信号機のアノテーション 最終" (1 class, "CC BY 4.0", 3 views, 0 downloads) (https://universe.roboflow.com/project-yubh1/-bzquf); yuki-saigo/traffic_light (1 class, "CC BY 4.0", 2 views) (https://universe.roboflow.com/yuki-saigo/traffic_light-txkjh). Neither states image count or country.
- Roboflow arrow sets (country not stated): "Traffic light with arrow detection" (classes Green arrow, Green circle, Traffic light, red traffic light; "CC BY 4.0"; 0 downloads) (https://universe.roboflow.com/traffic-light-and-arrow/traffic-light-with-arrow-detection); l0veyuk1no/traffic-light (classes light green arrow, light green circle, light green walk, light red arrow, light red circle, light red walk; "Public Domain"; 2 downloads) (https://universe.roboflow.com/l0veyuk1no/traffic-light-1jant); cherry-bnrg6/traffic-light-arrow-detection v4 (304 images, license not stated) (https://universe.roboflow.com/cherry-bnrg6/traffic-light-arrow-detection/dataset/4).
- Conclusion: no public, licensed Japanese traffic light detection dataset with arrow labels was found on Kaggle, Roboflow, Hugging Face, Zenodo or university pages. OpenTSR would need to collect and label its own Japanese data.

---

## 7. China, Korea, India

### China
- S2TLD (section 1.4), Jilin/Tongji 2025 (section 1.15).
- PTL (Pedestrian Traffic Light, LYTNet): over 5,000 photos at hundreds of Shanghai intersections; states and license not stated (https://paperswithcode.com/dataset/ptl).
- Chengdu roadside camera set (10,000 images): "cannot be released publicly" (https://www.techscience.com/CMES/v147n1/67131/html).
- Kaggle "Chinese Traffic Signs in the Natural Scenes" (9,898 images) is signs only, no traffic lights (https://hyper.ai/en/datasets/37148).

### Korea
- AI Hub "신호등/도로표지판 인지 영상(수도권)" and "(수도권 외)": 1,915,672 images in total; 400,592 images contain vehicle traffic lights; annotations: bounding box, light count, and on/off attributes for red, yellow, green, left arrow and other arrow; derived 5-class set (stop, warning, go, go&left, left) and 8-class set by 3-lamp vs 4-lamp housing; YOLOv5-l 80.5 mAP, YOLOv5-n 77.1, YOLOv4 65.4 on class type 1 (https://koreascience.kr/article/JAKO202219559499578.pdf). Full set about 1 TB (https://koreascience.kr/article/JAKO202411462432202.pdf). Access and license: AI Hub requires a Korean account; terms were not verified in this session (the papers only cite https://aihub.or.kr/).
- ETRI AI 나눔 "ETRI 신호등 데이터 셋": 96,307 images at 2048x1536 (https://koreascience.kr/article/JAKO202411462432202.pdf); the AI Nanum listing page shows only top downloads and no TL entry (https://nanum.etri.re.kr/share/list). License unknown.
- Kakao Mobility Korea-specific dataset (Feb 2025): 150,000 instances across 10 categories incl. 2D traffic lights; released on ETRI's AI Nanum; "anyone can use it for autonomous driving R&D"; state labelling not stated (https://www.asiae.co.kr/en/article/2025022810532944143).
- DGIST (2024) describes a TL dataset construction process but releases no dataset (https://scholar.dgist.ac.kr/handle/20.500.11750/57171).
- Roboflow "ETRI traffic light" (classes Traffic Light_Green, Green Left, Red, Yellow, traffic light; "CC BY 4.0"; 0 downloads; provenance unclear) (https://universe.roboflow.com/project-x9thl/etri-traffic-light).

### India
- IRD (section 1.14).

---

## 8. Synthetic datasets

- CARLA Traffic Light Dataset (University of Siegen, Dongchen Li, June 2026): 18,201 crops of 64x32 px from CARLA Map10HD, 5 classes (red, yellow, green, unknown, not_traffic_light), label.csv, "Creative Commons Attribution 4.0 International", 84.1 MB, DOI 10.5281/zenodo.20575485 (https://zenodo.org/records/20575485). Classification only, no boxes.
- Synthetic-context overlay (Jpvmello): synthetic foregrounds blended onto arbitrary backgrounds, states red/yellow/green, Processing + Python generators, "GPL-3.0 license" (code), foregrounds on Google Drive; SIBGRAPI 2020 paper DOI 10.1016/j.cag.2020.09.012 claims mAP and F1 "nearly 4 p.p. higher" than the real-data reference (https://www.github.com/Jpvmello/traffic-light-detection-synthetic-context).
- Roboflow "traffic light carla world shuffled": 1 class, "CC BY 4.0", 5 downloads (https://universe.roboflow.com/bachelor-thesis-sru9q/traffic-light-carla-world-shuffled).
- Virtual KITTI 2: TrafficLight segmentation class (15 categories); "Creative Commons Attribution-NonCommercial-ShareAlike 3.0"; no state; archives at download.europe.naverlabs.com/virtual_kitti_2.0.3/ (https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/).
- Synscapes (7D Labs): 25,000 images at 1440x720 (and 2048x1024), 19 Cityscapes classes incl. traffic light for segmentation only; boxes only for dynamic classes; state not addressed (https://arxiv.org/pdf/1810.08705). Terms: "You may use the dataset for non-commercial purposes only"; "You are prohibited from distributing, reproducing, displaying, editing, altering, modifying" (no derivatives); 337 GB (https://synscapes.on.liu.se/download.html).
- SHIFT: no TL boxes (section 4).
- The 2023 synthetic-dataset survey lists no synthetic dataset with traffic light state labels (https://arxiv.org/pdf/2304.12205).

---

## 9. Community datasets (Kaggle, Roboflow Universe, Hugging Face)

Hugging Face Hub (search for "traffic light"):
- dronefreak/LISA-Traffic-Lights: 43,017 images, 109,475 boxes, YOLO, 5.06 GB, cc-by-nc-sa-4.0, 993 downloads last month, 10 models trained (https://huggingface.co/datasets/dronefreak/LISA-Traffic-Lights).
- shpotes/bosch-small-traffic-lights-dataset: full train+test archives 18.3 GB, license tag "other" (https://huggingface.co/datasets/shpotes/bosch-small-traffic-lights-dataset/tree/main).
- yangxue/S2TLD: both subsets, 1.43 GB, tagged apache-2.0 (https://huggingface.co/datasets/yangxue/S2TLD/tree/main).
- vzc-research-chapter/vzc-traffic-light-dataset: 3,000 images, 25,624 boxes, cc-by-nc-4.0, gated (https://huggingface.co/datasets/vzc-research-chapter/vzc-traffic-light-dataset).
- mehmetkeremturkcan/traffic-lights-of-new-york: single tlony.zip about 488 MB, under 1K items, license "mit", no card, annotations unknown (https://huggingface.co/datasets/mehmetkeremturkcan/traffic-lights-of-new-york/commit/93bf0c5a1bbf17e1795b47470634680679ec1b97).
- BDD100K partial mirrors dgural/bdd100k and ShantyCam/bdd100k-det (section 2.1).
- Zenseact/ZOD loader (section 3).

Kaggle (pages could not be fetched directly; details from secondary pages):
- mbornoe/lisa-traffic-light-dataset: the canonical LISA host (https://vbn.aau.dk/en/publications/vision-for-looking-at-traffic-lights-issues-survey-and-perspectiv).
- wjybuqi/traffic-light-detection-dataset: about 2,600 annotated images, 9 colour categories, "CC0: Public Domain" (https://www.innovatiana.com/en/datasets/traffic-light-detection-dataset).
- farukece/traffic-light-detection-image-set: binary classification, counts and license not stated (https://www.kaggle.com/datasets/farukece/traffic-light-detection-image-set via WebFetch summary).
- aiMotive TL/TS dataset is also listed on Kaggle per its README (https://uithub.com/aimotive/aimotive_tl_ts_dataset).

Roboflow Universe: licenses vary per project and are self-declared by uploaders; several re-uploads of NC datasets (LISA, Bosch) declare "CC BY 4.0", which cannot override the upstream license. Examples verified: traffic-light-detection-15v3q v5 (6,255 images: 5,466 / 527 / 262; mixed labels; license not shown) (https://universe.roboflow.com/traffic-light-detection-15v3q/traffic-light-detection-0qhhh/dataset/5); Udacity self-driving (MIT) (https://public.roboflow.com/object-detection/self-driving-car); Japanese/Korean/arrow projects in sections 6 and 7.

---

## 10. Summary table

| Dataset | Year | Country | Images | TL boxes | States | Arrows / pictograms | License (as written) | Test labels public | Access | Mirror |
|---|---|---|---|---|---|---|---|---|---|---|
| LISA | 2016 | USA (San Diego) | 43,007 | 113,888 to 119,231 (109,475 in HF export) | go/stop/warning | yes (left, forward in class names) | CC BY-NC-SA 4.0 | yes (Kaggle/HF) | Kaggle | HF dronefreak (full YOLO), Dataset Ninja |
| BSTLD | 2017 | USA (Palo Alto) | 13,427 | about 24,000 | red/yellow/green/off | train only (left/right/straight) | "LICENSE FOR NONCOMMERCIAL PURPOSES" (Robert Bosch LLC) | yes | Zenodo | HF shpotes (full) |
| DTLD v1/v2 | 2018/2021 | Germany (11 cities) | 40,979 | 232,039 / 292,245 | red/yellow/green/red-yellow/off | yes (arrow dir, ped, bike, tram) + relevance, orientation, occlusion, track ID | research and teaching only; no commercial use; no disclosure to third parties | yes, to registrants | registration form | none |
| S2TLD | 2022 | China | 5,786 | 14,130 | red/yellow/green/off/wait_on | no | MIT (GitHub) | no official split | GitHub links | HF yangxue (full), Zenodo resplit CC BY 4.0 |
| WPI | 2016 | USA | 3,456 | 6,766 | red/green | yes | unknown | unknown | WPI site / HyperAI | HyperAI |
| LaRA | 2009/2015 | France (Paris) | 11,179 | 9,168 | green/orange/red/ambiguous | no (circular only) | unknown, "free" | n/a | official page unverified | none |
| Cityscapes TL++ | 2022 | Germany | 5,000 | unknown | red/red-yellow/yellow/green/off/unknown | no | LGPL-2.1 (survey) over Cityscapes NC | unknown | unknown | none |
| ATLAS | 2025 | Germany (Karlsruhe) | 33,044 | 72,998 | green/red/yellow/red-yellow | yes (7 pictograms, 25 classes) | CC BY-NC-SA 4.0 | yes (test folder) | Zenodo 29.7 GB | none |
| aiMotive TL/TS | 2024 | USA + Hungary | about 50,000 | about 320,000 (3D, auto) | colour state | mask type | CC BY-NC-SA 4.0 | unknown | S3 107 GB, Kaggle | Kaggle |
| VZC-TLD | 2025 | USA | about 3,000 | 25,624 | red/yellow/green/unknown + relevance | no | CC BY-NC 4.0 | unknown | HF (gated) | HF |
| LAVA Salient Lights | 2023 | USA | unknown | 30,566 (9,051 salient) | unknown | unknown | unknown | unknown | unknown | none |
| aUToLights | 2023 | Canada/USA (MCity) | about 10,000 test + train | about 47,550 | red/yellow/green/off | yes (left arrows, doghouse) | not stated | n/a | request form | none |
| DualCam | 2023 | unknown | 2,250 | 8,321 | red/yellow/green/red-yellow/empty/countdown | green-left/right/up | not stated | yes | Google Drive | none |
| IRD | 2022 | India | about 3,050 | 8,520 (all classes) | red/yellow/green | green-arrow | not stated | unknown | Google Sites | none |
| Jilin/Tongji TL+countdown | 2025 | China | 21,774 | unknown | 16 classes incl. off | yes (left/straight/right/U-turn) | unknown | unknown | unknown | none |
| BDD100K | 2020 | USA | 100K | unknown | red/green/yellow/none | no | BDD100K License (research, education, non-profit) | detection test held out (not verified) | registration | partial HF mirrors |
| Udacity (Roboflow) | 2020 | USA | 15,000 | 97,942 (all classes) | yes (not verified) | unknown | MIT License | yes | Roboflow | Roboflow |
| COCO Traffic | 2021 | mixed | unknown | unknown | red/green/na | no | MIT (repo) | n/a | GitHub + Google Drive | none |
| Mapillary Vistas | 2017/2021 | worldwide | 25,000 | 79,512 TL objects | no | class-level only (pedestrian, upright) | CC BY-NC-SA 4.0 | no (5,000 test withheld) | request | Dataset Ninja |
| Cityscapes | 2016 | Germany | 5,000 + 20,000 | unknown | no | no | non-commercial, research only | no | registration | none |
| ZOD | 2023 | 14 EU countries | 100k | unknown | no | no | CC BY-SA 4.0 | unknown | application | HF Zenseact/ZOD |
| COCO / Open Images / Objects365 | 2017 / 2022 / 2019 | worldwide | 330K / 9M / 1.8M | unknown | no | no | CC BY 4.0 (annotations) | n/a | direct | many |
| PIE | 2019 | Canada (Toronto) | 909,480 frames | 2,353,983 traffic objects (all types) | yes (not verified) | n/a | MIT License | yes | YorkU / Google Drive | none |
| JAAD | 2017 | mixed | 346 clips | unknown | per-frame TL info | n/a | MIT (annotations), CC BY 4.0 (videos) | yes | YorkU / Google Drive | none |
| CARLA TL (Siegen) | 2026 | synthetic | 18,201 crops | n/a | red/yellow/green/unknown/not-TL | no | CC BY 4.0 | n/a | Zenodo 84 MB | none |

---

## 11. Recommendations for a fixed public benchmark (Apache-2.0 open-data project)

1. Redistributable under permissive terms (data can be re-hosted and relabelled by OpenTSR):
   - S2TLD (MIT per GitHub and Dataset Ninja). No official split; adopt the Zenodo CC BY 4.0 repartition (DOI 10.5281/zenodo.7352960) or publish a fixed split. Note the HF mirror's apache-2.0 tag disagrees with GitHub; cite the GitHub LICENSE file.
   - Udacity Self Driving Car (MIT via Roboflow public). Verify the class list before use.
   - COCO Traffic annotations (MIT) on COCO images (annotations CC BY 4.0). Only red/green/na, no arrows.
   - PIE and JAAD (MIT; JAAD videos CC BY 4.0). Useful for pedestrian-signal and far-field cases; traffic light type/state fields should be checked in the annotation README before relying on them.
   - CARLA TL crops (CC BY 4.0) for classifier pretraining only.
   - Zenseact Open Dataset (CC BY-SA 4.0): has Traffic Signal boxes without state. OpenTSR could add state labels and release them, but ShareAlike means the derived labels must stay CC BY-SA 4.0, not Apache-2.0.
   - COCO, Open Images, Objects365 traffic light boxes (CC BY 4.0 annotations) for detector pretraining; no state.
2. Research-only or NC; usable for evaluation by pointer, not redistributable under Apache-2.0:
   - LISA (CC BY-NC-SA 4.0), BSTLD (Bosch NC license: Share and Adapted Material allowed for NonCommercial purposes only), ATLAS (CC BY-NC-SA 4.0), aiMotive (CC BY-NC-SA 4.0), VZC-TLD (CC BY-NC 4.0), Mapillary Vistas (CC BY-NC-SA 4.0), Cityscapes (non-commercial), BDD100K (research/non-profit), nuScenes, Argoverse 2, KITTI-360, Virtual KITTI 2 (all NC), Waymo (NC license, redistribution only to registered users), Synscapes (NC and no derivatives at all).
   - For these, OpenTSR can publish conversion scripts and evaluation code, and report numbers, but any corrected or extended label files derived from LISA/BSTLD/ATLAS must carry the same NC(-SA) terms and cannot be folded into an Apache-2.0 release. BSTLD and LISA both ship public test labels, so they work as "bring your own copy" test sets.
3. DTLD: the richest label schema (pictograms, red-yellow, relevance, track IDs, 2 MP stereo) and test labels are available after registration, but "disclosure of data to third parties" is prohibited, so not even derived labels can be shared. Use it as a private validation set and report numbers.
4. Gaps: no public Japanese dataset with horizontal housings and arrow lamps exists; TIER IV's labels (83,400 vehicular, 21,199 pedestrian images) are internal and only 1,045/1,062-image samples are public. For a Japan-focused world-class model, OpenTSR must collect and label its own data (own CC BY 4.0 or CC0 release) and could use ATLAS/DTLD arrow classes only for transfer, not redistribution.
5. Suggested benchmark structure: Tier A (fully redistributable, fixed splits, hosted by OpenTSR): S2TLD resplit + Udacity + COCO Traffic + the new Japanese set. Tier B (pointer-based, NC, public test labels): LISA test clips, BSTLD test, ATLAS test, with hashes and converters. Tier C (registration): DTLD test. Report mAP50 and mAP50-95 per state and per pictogram, plus small-object (under 10 px) breakdowns, since BSTLD/DTLD/ATLAS labels go down to 2 to 5 px.

---

## 12. Pages that could not be fetched (claims depending on them are marked not verified)

- https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset and https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset (provenance block)
- https://www.kaggle.com/datasets/wjybuqi/traffic-light-detection-dataset (provenance block)
- https://www.mdpi.com/2624-8921/8/4/90 (Vehicles 2026, YOLOv8 to YOLOv12 TL comparison; provenance block after redirect)
- https://scispace.com/pdf/vision-for-looking-at-traffic-lights-issues-survey-and-4k3ze3791o.pdf (LISA paper PDF; provenance block)
- https://projekter.aau.dk/projekter/files/213565236/main.pdf (Aalborg thesis; client error)
- https://github.com/aras62/PIE/tree/master/annotations (provenance block)
- http://scis.scichina.com/en/2025/157201.pdf (robots disallowed)
- https://service.tib.eu/ldmservice/dataset/lava-salient-lights-dataset (bot check page)
- https://www.researchgate.net/publication/403710089_... (client error)
- https://universe.roboflow.com/ksta/udacity_traffic_light and https://universe.roboflow.com/computer-vision-kmvf7/traffic-light-jvuxd (client error)
- https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset, https://www.uni-ulm.de/en/in/driveu/projects/driveu-traffic-light-dataset/ and https://github.com/Thinklab-SJTU/S2TLD (first attempts blocked; equivalent content obtained from https://hci.iwr.uni-heidelberg.de/node/6132, the uni-ulm data-sets page and a later successful fetch of the S2TLD README)
- Official license pages not fetched: bdd100k.com license, cityscapes-dataset.com/license, mapillary.com/dataset/vistas, aihub.or.kr dataset pages.
