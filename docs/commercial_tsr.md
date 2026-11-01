# Commercial traffic sign recognition systems: what is documented and what is inferred

Survey memo, retrieval date 2026-11-01. Written to answer two questions from the maintainer: which
architecture production TSR systems most likely use (cost, accuracy, speed taken together), and what
evidence exists that they are built as a detect, track, classify pipeline. Every claim in section A
is followed by the source it was read from; section C is inference and is labelled as such.

## Summary

- The documented skeleton of a production TSR system is: single-frame detection of sign candidates,
  tracking of the candidates over frames, classification of the aligned crop, and a decision taken
  over the whole track. It is spelled out in Mobileye's 2006 patent, in the Siemens VDO / Continental
  papers of 2005 and 2008, and in the 2012 IEEE T-ITS survey; the 2023 Mobileye ISA description adds
  OCR for text, a road-type network for implicit limits and a relevance step (exit and side-road
  signs); the 2025 NDSS black-box study adds long "spatial memorization" of recognised signs.
- The detector in current products is a CNN. Mobileye's accelerator description names convolutional
  networks, Tesla's AI Day stack uses a RegNet backbone with YOLO-like one-stage heads and reserves
  its transformer for the multi-camera BEV fusion, and the automotive SoCs that front cameras run on
  (5 to 34 TOPS, a few watts) have mature CNN toolchains and only partial transformer support.
- Whether detection and fine classification are one multi-task network or two models is not
  published for any current product. The two-stage design is attested for the 2006 to 2012
  generation and for the open stacks (Autoware traffic lights, Panoramax); for Tesla only the
  detection heads are described.

## A. Documented evidence

### A.1 Patents

- Mobileye, US 8,064,643 B2 "Detecting and recognizing traffic signs" (priority 2006-12-06, filed
  2007-12-06, granted 2011-11-22; inventors Stein, Shachar, Taieb, Wolfovitz). Pipeline in the
  patent: frames are partitioned between TSR and the other driver-assistance functions; detection
  finds candidate signs in a single frame ("a candidate image may include a circle and detection may
  be performed based on a circle detection algorithm"; at night "detects light spots in the image,
  typically in a low resolution image"); candidates are matched across frames into a "multi-frame
  structure" that "is verified over multiple frames" and aligned for vehicle motion; classification
  "is preferably a multi-level classification that constructs a gating tree", "the classifier is
  preferably a non-binary classifier"; "a decision is made (step 370) based on the output of the
  classification (step 350) for each image in the multiple frame structure", and the structure ends
  when the sign leaves the field of view or is occluded. Speed limits are classified, not read with
  OCR: "classification proceeds (step 440) by the number of digits", two-digit candidates against
  70, 80, 90 and three-digit candidates against 110, 120, 130 (https://patents.google.com/patent/US8064643B2/en).

### A.2 Supplier research papers (first production generation)

- Bahlmann, Zhu, Ramesh (Siemens Corporate Research) and Pellkofer, Koehler (Siemens VDO
  Automotive, Regensburg), "A system for traffic sign detection, tracking, and recognition using
  color, shape, and motion information", IEEE IV 2005. Two components: detection and tracking
  (AdaBoost over Haar wavelets with colour channels as a wavelet parameter, five scales, simple motion
  model, per-frame detections fused over time) and classification (crop to a circle, grey, histogram
  equalisation, fixed size; Bayesian classifier with one Gaussian per class on a 25-dimensional LDA
  feature; confidence fused across frames). 23 German speed and no-passing classes; detection false
  negatives 1.4 percent, false positives 0.03 percent, patch classification error 6 percent, video
  recognition error 15 percent, 10 fps on a 2.8 GHz Xeon at 384x288; "many signs are smaller than
  20 pixels in diameter" (https://lmbweb.informatik.uni-freiburg.de/people/bahlmann/data/ba_zh_ra_pe_ko_iv2005.pdf).
- Keller, Sprunk (Freiburg), Bahlmann (Siemens CR), Giebel, Baratoff (VDO Automotive AG, Lindau,
  Continental addresses), "Real-time recognition of U.S. speed signs", IEEE IV 2008. Gradient-voting
  rectangle detector plus AdaBoost rejection, alignment (rotation by histogram voting, rectangle
  re-detection), LDA features with a Mahalanobis classifier, and majority voting over the track,
  which "proved to be very robust against accidental misclassifications with high confidence".
  Detection 98.75 percent, classification 97.5 percent, recognition 96.25 percent on 80 sign
  instances in 673 s of video, one false alarm per 42 s, 11 fps with tracking on a 2.16 GHz laptop
  (https://lmb.informatik.uni-freiburg.de/people/bahlmann/data/ke_sp_ba_ba_gi_iv08.pdf).
  Both papers present research prototypes, but the authors sit in the supplier organisations that
  shipped the first speed-limit cameras, and the structure matches the Mobileye patent.

### A.3 Survey

- Møgelmose, Trivedi, Moeslund, "Vision-based traffic sign detection and analysis for intelligent
  driver assistance systems: perspectives and survey", IEEE T-ITS 2012. Defines TSR as detection
  ("concerned with locating signs in input images"; segmentation, feature extraction, detection) and
  classification, with tracking as a temporal layer ("accept sign candidates as signs only if they
  have shown up on a number of consecutive frames", or a Kalman predictor). On products: "speed limit
  detection systems have been included in top-of-the-line models from various manufacturers", while
  "a more general sign detection solution and an integration into other vehicle systems have not yet
  materialized"; systems use neither driver state nor driver input, and detected signs are often not
  linked to the road the car is on (https://vbn.aau.dk/ws/files/72574480/survey.pdf).

### A.4 Current product descriptions (functional level)

- Mobileye, camera-only Intelligent Speed Assist (blog, 2023; press release 2023-07-18): runs on
  EyeQ4 and EyeQ6 by software update; sign detection and classification with "signature-based
  classification" so that new sign types are added through updates; OCR for text that shape cannot
  convey (school hours, weight classes, city-entrance names); a deep network for road-type
  classification that estimates the limit where signs are missing, weathered or occluded; implicit
  limits (construction zones, school zones, highway entries and exits); temporary and variable
  digital signs; lane relevance ("offramp speed limits that shouldn't affect vehicles staying on the
  highway"); "tens of millions of video clips" of training data; certified by six labs in five EU
  countries against the GSR; no accuracy figure published
  (https://www.mobileye.com/blog/intelligent-speed-assist-isa-computer-vision-adas-solution/,
  https://ir.mobileye.com/news-releases/news-release-details/mobileye-launches-first-camera-only-intelligent-speed-assist).
- Mobileye EyeQ page: EyeQ6 Lite 5 TOPS (int8, 7 nm) for ADAS front cameras, EyeQ6 High 34 TOPS;
  the XNN engine is "the main source of horse power for convolutional neural networks", the VMP and
  PMA target "short integral types common in computer vision and deep learning"; no mention of
  transformers (https://www.mobileye.com/technology/eyeq-chip/).
- Bosch "Road sign information" (commercial vehicles): detects and classifies "round, triangular
  and rectangular" signs and reads the numbers on them; relies on the multi-purpose camera
  (https://bosch-mobility.com/en/solutions/assistance-systems/road-sign-information-cv).
- Tesla AI Day 2021 (secondary summary): eight cameras at 1280x960, 12-bit HDR, 36 Hz; RegNet
  backbone, BiFPN multi-scale fusion, HydraNet with task heads described as "one-stage, YOLO-like
  detectors"; transformers are used only to map image features into the bird's-eye-view space and
  fuse the cameras ("one or more multi-head self-attention blocks"); traffic signs appear as content
  of that vector space; nothing on a separate classifier or on latency figures
  (https://kimbochen.github.io/blog/2021/11/09/tesla-ai-day-vision.html).

### A.5 Black-box study of 2023 to 2024 production vehicles

- Wang et al., "Revisiting physical-world adversarial attack on traffic sign recognition: a
  commercial systems perspective", NDSS 2025 (UC Irvine). Four anonymised 2023 or 2024 vehicles from
  top-15 US brands; front-windshield camera data "processed by DNN-based object detectors"; the
  recognised sign is "persistently memorized until the sign's reaction task is finished"
  (clearing by position, not by time; three of four vehicles kept a detection for the full 60 s test
  window, against under 3 s for ordinary tracking); speed-limit results appeared about halfway past
  the sign; no evidence of map or GPS fusion for the tested signs; black-box transfer of published
  physical attacks 6.67 percent over 30 combinations (papers claimed 51.6 percent), which the
  authors attribute to the memorization making hiding attacks hard and appearing attacks easy
  (https://arxiv.org/html/2409.09860v1).

### A.6 Open stacks with readable internals

- Autoware Universe (TIER IV, traffic lights, not signs): HD-map projection to a region of interest,
  YOLOX-s fine detector at 416x416, MobileNetV2 state classifier at 224x224 (11 classes incl.
  arrows), plus a whole-image YOLOX-s at 960x960; all models on Hugging Face since 2026 (see
  `traffic_lights/docs/oss_implementations.md`).
- Panoramax: YOLO11l sign detector and per-country classifiers (DE 134 classes) published as
  separate models with their training crops (see `oss_implementations.md`).
- openpilot: no camera TSR ("openpilot does not detect speed limits at this time"); the FrogPilot
  fork ships an MNN speed-limit reader fused with map data (see `oss_implementations.md` E3).

### A.7 Embedded hardware and toolchains

- TI TDA4 (TIDL): SDK 9.1 "supports basic transformer (classification) ViT, DeiT (matmul,
  layernorm, softmax) and partial support of SwinT"; DETR-style detectors not listed
  (https://e2e.ti.com/f/791/t/1329460).
- Renesas 4th-generation R-Car: V4M up to 17 TOPS, "9 TOPS/W", about 5 W in a typical 8 MP smart
  camera, mass production from Q1 2026; the page names "dedicated deep learning and computer vision
  IP" without network types
  (https://www.renesas.com/en/about/newsroom/renesas-leads-adas-innovation-power-efficient-4th-generation-r-car-automotive-socs).
- Hailo-8: a CNN-oriented dataflow accelerator; the vendor forum thread on attention-based
  detectors exists but could not be read in full
  (https://community.hailo.ai/t/hailo-8-for-attention-based-object-detection-models/16105).
- AMD Vitis AI 2.5 TT100K YOLOX-m: AP 62.3 float versus 62.1 INT8 (QAT), i.e. INT8 costs a CNN
  detector 0.2 points (https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/model-list/pt_yolox_TT100K_640_640_73G_2.5/model.yaml).
- Ultralytics comparison (COCO 640, T4 TensorRT 10): YOLO11n 39.5 mAP, 1.5 ms, 2.6 M params,
  6.5 GFLOPs; YOLO11s 47.0, 2.5 ms; YOLO11l 53.4, 6.2 ms, 87 GFLOPs; RT-DETRv2-s 48.1, 5.03 ms,
  20 M, 60 GFLOPs; RT-DETRv2-l 53.4, 9.76 ms, 136 GFLOPs; the page notes the higher training memory
  and slower convergence of RT-DETR and calls it "less accessible" for constrained edge devices
  (https://docs.ultralytics.com/compare/rtdetr-vs-yolo11).

## B. Numbers that frame the design

- GTSRB classification is saturated for both families: CNNs 99.46 to 99.81 percent, fine-tuned ViT
  and CLIP encoders 98.5 to 99.3 percent in the literature (`vlm_and_foundation.md` A.1, D.2); our
  own runs: ConvNeXt V2 atto 99.81 percent with 3.7 M parameters, bazyl ViT-B/16 97.00 percent, a
  recipe effect (`replication.md` section 4).
- The only transformer-versus-CNN detection comparison we ran was zero-shot: OWLv2 (ViT-B/16)
  class-agnostic AP50 77.1 on GTSDB frames against YOLO-World-s 54.2 (`replication.md` section 7),
  so nothing we measured says transformers detect signs worse; they cost more.
- Compute budget: a 5 TOPS front-camera SoC at 30 to 50 percent utilisation and 30 fps leaves
  roughly 50 to 80 GOPs per frame for every function. A nano or small CNN detector at 1280 px
  (about 25 to 90 GFLOPs) fits; an RT-DETR-s at 640 px (60 GFLOPs) and the resolution signs need
  (16 to 48 px objects, see `dataset_catalog.md`) does not leave room for the rest.

## C. Inferred architecture (not documented for any current product)

1. Detector: a CNN one-stage network at high resolution with coarse sign classes (shape and colour
   families, or a single "sign" class), sharing a backbone with the other camera functions
   (Tesla's HydraNet is the documented example of the sharing).
2. Tracking and spatial memorization: candidates confirmed over frames, the recognised sign held
   with its position until the vehicle has passed it (NDSS 2025), decisions by voting over the track
   (Mobileye patent, Keller 2008).
3. Classification on the crop: a small network per region or country, replaceable without
   touching the detector (Mobileye's "signature-based classification" updates), plus OCR for text
   and variable signs (Mobileye ISA, Bosch).
4. Relevance and fusion: lane and exit assignment, implicit limits from road type, optional map
   fusion (Mobileye ISA; FrogPilot for the open equivalent; the NDSS vehicles showed no fusion).
5. Transformers appear, where they appear, in the multi-camera BEV fusion of full self-driving
   stacks and in sequence decoders for OCR; offline they serve auto-labelling and data generation.

Why this is the likely shape: the published components are all CNNs; the accelerators and
toolchains of the 5 to 34 TOPS class are CNN-first with INT8 at negligible accuracy cost; the
accuracy ceiling on the classification task is reached by small CNNs; the functional descriptions
(update-able classification, OCR, relevance) presuppose a modular pipeline; and the long-tail
requirements (EU ISA, Euro NCAP conditional and implicit limits) are addressed by data and system
logic rather than by backbone choice.

## D. What this means for OpenTSR

- Train and release a CNN one-stage detector with coarse classes at 1280 px or tiled 2048 px,
  evaluated after INT8 quantisation, and a crop classifier under 5 M parameters with an OCR head
  for numbers; keep the two models separately replaceable.
- Evaluate on sign-pass events with tracking and relevance, not only on crops (`requirements_and_robustness.md`
  D.1); the ISA 90 percent distance-based figure is an event-level metric.
- Use transformers and VLMs offline: OWLv2 for candidate boxes, VLMs for digit reading and
  labelling, generative data for the long tail (`vlm_and_foundation.md`).
- Keep RT-DETR / D-FINE baselines in the benchmark with INT8 accuracy and embedded latency columns,
  so that the cost argument is measured rather than assumed.

## E. Open questions

- No current product documents whether fine classification is a separate model; the stacked
  evidence is from 2006 to 2012 and from open stacks.
- EyeQ6 High and comparable 30 TOPS-class SoCs may run attention blocks in the fusion stage;
  nothing public ties that to sign recognition.
- The NDSS study saw no map fusion on four US vehicles; European ISA implementations (Mobileye's
  is "without relying on a third-party map") may differ by OEM.
