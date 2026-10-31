# Open-source implementations of traffic sign recognition (classification and detection)

Draft survey compiled on 2026-10-31 (claude.ai research session, pingfoo/OpenTSR). Every claim below
was checked on that date against a live page, a raw file, an API response, a shallow git clone or a
remote zip listing; the URL that supports each claim follows it in parentheses. "unknown" means the
fact could not be verified. Star counts and "last commit" values come from shields.io badges
(https://img.shields.io/github/stars/OWNER/REPO.json, .../last-commit/..., .../license/...) and are
approximate; exact HEAD dates come from `git log -1` of shallow clones made on 2026-10-31.

Scope: traffic SIGNS only (the traffic light survey is in `traffic_lights/docs/oss_implementations.md`). Vision
language models, CLIP-style zero-shot classifiers and open-vocabulary detectors are covered by a
separate document; here they appear only as pointers (section C4 and G).

Verification limits in this session: github.com HTML pages could be read only when their URL had
first appeared in a web search result; api.github.com was not available; sid.erda.dk (the host of
the official GTSRB and GTSDB archives) is not reachable from the sandbox; universe.roboflow.com,
doaj.org and pmc.ncbi.nlm.nih.gov refused or blocked automated requests; Kaggle notebook pages were
not reachable. Papers with Code is offline and is not cited.

## 0. Key takeaways

1. The GTSRB leaderboard is frozen since 2018. The official results page lists 99.71% for the
   "CNN with 3 Spatial Transformers" of Arcos-Garcia et al. (DeepKnowledge Seville), 99.46% for the
   IDSIA committee of CNNs, 98.84% for humans, 98.31% for Sermanet's multi-scale CNN, 96.14% for
   random forests and 95.68% for LDA on HOG (https://benchmark.ini.rub.de/gtsrb_results.html).
   Only the 99.71% entry has official code with downloadable weights, and it is Torch7/Lua with a
   mandatory CUDA spatial transformer module (https://github.com/aarcosg/tsr-torch). The IDSIA,
   Sermanet, HOG/LDA and random forest entries have no official code; the best reimplementation of
   the IDSIA-style network is Moodstocks/gtsrb.torch (99.67% with two STNs, Torch7).
2. For GTSRB classification in a modern framework with weights in hand, the practical choices are
   poojahira/gtsrb-pytorch (STN + CNN, 3.4 MB `model_40.pth` in the repo, 99.809% on the NYU Kaggle
   private leaderboard), hello2all/GTSRB_Keras_STN (Keras, 13.9 MB `weights.hdf5`, 99.59% on the
   Udacity pickled test set) and ViT fine-tunes on Hugging Face (kelvinandreas 98.46%, bazyl 99.93%
   on its own evaluation split). The Udacity pickled GTSRB (train 34,799 / valid 4,410 / test
   12,630 at 32x32) is still downloadable from CloudFront, and the official 12,630-image test set is
   mirrored on Hugging Face (tanganke/gtsrb), which matters because torchvision's `datasets.GTSRB`
   downloads from sid.erda.dk.
3. Detection with weights and a published number: aarcosg/traffic-sign-detection (TF1 Object
   Detection API, 8 model zips on Google Drive, GTSDB 3-superclass mAP 95.77 for Faster R-CNN
   Inception ResNet V2 down to 61.64 for SSD MobileNet), skokec/detectron-traffic-signs (Caffe2
   Detectron, 644 MB DFG Mask R-CNN weights, mAP50 95.5 on 200 DFG classes), the official TT100K
   Caffe model (467 MB `model.caffemodel` inside an 801 MB `code.zip`, accuracy 0.877 / recall
   0.906 at IoU 0.5 on 45 classes) and the AMD/Xilinx Vitis AI 2.5 YOLOX-m trained on TT100K
   (594 MB public zip with PyTorch float, QAT and INT8 xmodel, AP 62.3 on TT100K val). All four are
   old-framework or GPU-bound; none is a 2024 to 2026 YOLO with a clean `yolo val` protocol.
4. Two 2024 to 2026 releases are directly useful for OpenTSR: (a) Panoramax (the French open
   street-level imagery project) publishes under Etalab-2.0 / CC BY-SA 4.0 a YOLO11l detector of
   signs, plates and faces (sign AP50 0.898 at imgsz 2048) plus YOLO-cls classifiers of French,
   German (130+ classes, accuracy 0.986), Belgian and Dutch signs with their 20k+ image training
   sets on Hugging Face; (b) Infra Intelligence (Japan) publishes `road-sign-yolox-tiny`, an
   Apache-2.0 YOLOX-Tiny that detects Japanese sign plates in 5 Road Sign Ordinance categories
   (guide/warning/regulatory/indication/supplementary), with ONNX and PyTorch weights, a Japanese
   model card, held-out test AP50 0.902 and class-agnostic recall 0.857, trained on Mapillary
   images of Japan. Its evaluation images are not released.
5. No production stack opens vision TSR: Autoware Universe and Baidu Apollo have no traffic sign
   perception module (only traffic lights and HD-map stop/yield signs); openpilot states "openpilot
   does not detect speed limits at this time". The one exception is the openpilot fork FrogPilot,
   which ships a 12.6 MB MNN model `speed_limit_reader.mnn` that reads speed limit signs from the
   road camera on CPU and feeds a "Speed Limit Controller" and a crowd-sourced "Speed Limit Filler";
   the model's architecture and training data are undocumented. sunnypilot uses only the car's
   built-in TSR (CAN) and OpenStreetMap data.

## A. Pre-deep-learning and early deep learning (2011 to 2015)

### A1. IJCNN 2011 GTSRB competition (final results page)

- Official table (team, method, correct recognition rate on the final test set):
  DeepKnowledge Seville, CNN with 3 Spatial Transformers, 99.71; IDSIA, Committee of CNNs, 99.46;
  COSFIRE, color-blob-based COSFIRE filters, 98.97; INI-RTCV, Human Performance, 98.84; sermanet,
  Multi-Scale CNNs, 98.31; CAOR, Random Forests, 96.14; INI-RTCV, LDA on HOG 2, 95.68; LDA on HOG 1,
  93.18; LDA on HOG 3, 92.34 (https://benchmark.ini.rub.de/gtsrb_results.html). The page gives
  references only, no code links (same URL). The test set has 12,630 images (the count is stated in
  community repositories that evaluate on it, e.g. "a total of 12,630 images that we used for
  testing", https://raw.githubusercontent.com/navoshta/traffic-signs/master/README.md).
- Ciresan, Meier, Masci, Schmidhuber (IDSIA), "Multi-column deep neural network for traffic sign
  classification" (Neural Networks 2012; the MCDNN arXiv version is https://arxiv.org/abs/1202.2745).
  Official code: none found (the arXiv paper describes in-house GPU code; no repository link in the
  PDF, https://arxiv.org/pdf/1202.2745). Best-known reimplementation: Moodstocks/gtsrb.torch
  (`idsia_net.lua`), which reports 98.85% for a single IDSIA-like network, 99.46% with one spatial
  transformer and 99.67% with two (https://raw.githubusercontent.com/Moodstocks/gtsrb.torch/master/README.md).
  Repo facts: MIT, 177 stars, last commit May 2016, Torch7/LuaJIT, `--no_cuda` works only without
  the STN module (same README; https://img.shields.io/github/stars/Moodstocks/gtsrb.torch.json).
- Sermanet and LeCun, "Traffic Sign Recognition with Multi-Scale Convolutional Networks" (IJCNN
  2011). The paper states the network "was implemented using the EBLearn C++" library
  (http://eblearn.sf.net) and reports 98.97% in phase I and 99.17% after enlarging the network
  (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf, pages 1 and 5). Whether the GTSRB
  demo is still inside the EBLearn SourceForge tree was not verified. The architecture (two conv
  stages, multi-scale features fed to the classifier, grayscale input) is what most Udacity-era
  repositories reimplemented in TensorFlow (section B).
- INI-RTCV baselines (LDA on HOG 1/2/3, 92.34 to 95.68%): no code published; the HOG features
  themselves were shipped with the dataset as precomputed files (GTSRB_Final_Training_HOG.zip etc.,
  listed under https://benchmark.ini.rub.de/gtsrb_dataset.html; the host sid.erda.dk was not
  reachable so the file list is not verified).
- Zaklouta and Stanciulescu (CAOR, Mines ParisTech), random forests 96.14% and later K-d trees /
  random forests on HOG: no official code found (not verified beyond the results page).
- COSFIRE (Gecer, Azzopardi, Petkov), 98.97%: the authors' COSFIRE MATLAB code exists in general but
  a GTSRB-specific release was not verified.

### A2. IJCNN 2013 GTSDB competition

- The complete results table (about 280 rows, metric "Area under Curve" per category plus "Average
  overlap") is at https://benchmark.ini.rub.de/gtsdb_results_all.html. Top entries: prohibitory
  AUC 100% for wgy@HIT501 (ELL_SVM2, overlap 90.08%), visics (boosted_intChn_ratio_scales, 88.22%),
  LITS1 (intColorShapeAppearance, 86.91%), glc12125 (robok, 85.57%), DSL_clusters
  (ToneMap+AcfChunks, 83.56%); mandatory AUC 100% for glc12125 and DSL_clusters and wgy@HIT501
  (HOG_LDA_SVM2); danger AUC 100% for visics, glc12125, s_trafficsign (DL2, "danger 2, 9k aug TR")
  and DSL_clusters; qcom VSSD3 99.16% mandatory / 99.93% danger (same URL). The page contains no
  code links or protocol details (same URL). The competition paper is Houben et al., "Detection of
  Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark"
  (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf).
- Official code for the entries: none found. The visics entry (Mathias, Timofte, Benenson, Van Gool,
  "Traffic sign recognition, how far are we from the solution?", IJCNN 2013) used integral channel
  features in the style of Dollar's toolbox; a traffic-sign specific release was not verified.
- Later reimplementations that evaluate on GTSDB with code and weights are in section D (aarcosg
  2018, Signfeld 2020, hellojialee 2018).

### A3. Haloi 2015 (spatial transformer plus inception)

- Mrinal Haloi, "Traffic Sign Classification Using Deep Inception Based Convolutional Networks",
  arXiv 1511.02992 (2015), reports 99.81% on GTSRB (https://arxiv.org/pdf/1511.02992v2.pdf, page 1).
  The PDF contains no code link (checked by grepping the extracted text for "github" and "code").
  No widely used reimplementation was found (web search on 2026-10-31 returned only paper
  aggregators). Later work with spatial transformers that does publish code: Moodstocks/gtsrb.torch
  (2015 to 2016, A1), aarcosg/tsr-torch (2018, C1), hello2all/GTSRB_Keras_STN (2018, C2) and
  poojahira/gtsrb-pytorch (2018, C2).

## B. Udacity Self-Driving Car Nanodegree "Traffic Sign Classifier" era (2016 to 2018)

Common protocol: the pickled 32x32 GTSRB released by Udacity, "train.p / valid.p / test.p" with
34,799 training, 4,410 validation and 12,630 test images in 43 classes
(https://raw.githubusercontent.com/hello2all/GTSRB_Keras_STN/master/capstone_report.md). The zip is
still online: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
(HTTP 200, Content-Length 123,524,425 bytes on 2026-10-31). Accuracies are on the 12,630 test
images unless stated; almost all code is TensorFlow 0.12 to 1.x and will not run on current
Python without rewriting.

| Repo | Stars / last commit / license | Method and framework | Reported test accuracy | Weights |
|---|---|---|---|---|
| udacity/CarND-Traffic-Sign-Classifier-Project (template) | 372, Nov 2021, MIT (https://img.shields.io/github/stars/udacity/CarND-Traffic-Sign-Classifier-Project.json) | Starter notebook, LeNet lab | none (template) | no |
| navoshta/traffic-signs | 148, Dec 2019 (HEAD 2019-12-23), Apache-2.0 (shields; clone) | TensorFlow; 3 conv (5x5, 32/64/128) + 1 FC 1024 with skip connections, dropout, L2, augmentation, two-stage pretraining on an extended set (https://raw.githubusercontent.com/navoshta/traffic-signs/master/README.md) | "99.33% accuracy on the test set", "12,630 images", "85 examples" wrong (same README) | no (notebook only; file list of clone) |
| kenshiro-o/CarND-Traffic-Sign-Classifier-Project | 145, Aug 2017 (HEAD 2017-08-26), no license | TensorFlow, 3x3 conv model, grayscale, dropout p-conv 0.6 / p-fc 0.5, augmentation with imgaug (https://raw.githubusercontent.com/kenshiro-o/CarND-Traffic-Sign-Classifier-Project/master/writeup_template.md) | "test accuracy of 97.75%" (same writeup) | no |
| mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow | 73, Dec 2019, MIT | TensorFlow 0.12.1, LeNet and VGG-style nets, grayscale (https://raw.githubusercontent.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow/master/README.md) | "+99% validation accuracy, and a 97.6% testing accuracy" (same README) | no |
| vxy10/p2-TrafficSigns (Vivek Yadav) | 67, Feb 2017 (HEAD 2017-02-09), no license | TensorFlow notebook; README table lists "Our model 98.8%" next to Haloi 99.81% and "Committee of CNNs 99.47%" (https://raw.githubusercontent.com/vxy10/p2-TrafficSigns/master/README.md) | 98.8% | no |
| jeremy-shannon/CarND-Traffic-Sign-Classifier-Project | 59, Jan 2017 (HEAD 2017-01-16), no license | TensorFlow, Sermanet/LeCun-style multi-scale model, grayscale, augmentation to 800 per class (https://raw.githubusercontent.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/master/Traffic_Sign_Classifier.md) | validation accuracy 0.987 in the log; the test number is not in the markdown export (same file) | yes: TF 0.12 checkpoint `lenet.data-00000-of-00001`, `lenet.index`, `lenet.meta` committed to the repo (file list of the clone) |
| chsasank/Traffic-Sign-Classification.keras | 68, Sep 2018, MIT | Keras CNN, blog tutorial https://chsasank.github.io/keras-tutorial.html (https://raw.githubusercontent.com/chsasank/Traffic-Sign-Classification.keras/master/README.md) | not stated in README (the blog claims about 98%, not verified) | no |
| MehdiSv/TrafficSignsRecognition | 58, Dec 2016, no license, no README file fetched | unknown | unknown | unknown |
| vamsiramakrishnan/TrafficSignRecognition | 46, May 2017, no license | TensorFlow, CLAHE normalisation, 10x augmentation (https://raw.githubusercontent.com/vamsiramakrishnan/TrafficSignRecognition/master/README.md) | "98.5% accuracy over the test set" (same README) | no |
| pierluigiferrari/traffic_sign_classifier | 17, Nov 2017, no license | TensorFlow 0.1x CNN; pickled data on Google Drive (https://raw.githubusercontent.com/pierluigiferrari/traffic_sign_classifier/master/README.md) | "97.9% on the official GTSRB test dataset" (same README) | no |
| surmenok/GTSRB | 128, Nov 2019, MIT | fastai + PyTorch, ResNet-34 pretrained on ImageNet (https://raw.githubusercontent.com/surmenok/GTSRB/master/README.md) | "99.22% accuracy on the validation set (random 20% subset of the training dataset)", not the test set (same README) | no |
| magnusja/GTSRB-caffe-model | 15, Oct 2019, no license | Caffe AlexNet 256x256 with "fill" resizing (https://raw.githubusercontent.com/magnusja/GTSRB-caffe-model/master/README.md) | "roughly 97%" (same README) | caffemodel via git-lfs or a Nextcloud link (same README; not downloaded) |
| ndrplz/self-driving-car (all Udacity projects) | 2.9k, Oct 2020, no license (shields) | TensorFlow notebooks, project 2 is the sign classifier | not checked | no |

Runnability today: all TensorFlow 0.12 / 1.x code is "hard" (TF1 has no wheels for Python 3.12+;
`tf.compat.v1` can load frozen graphs but these repos are notebooks, not SavedModels). The data is
"easy" thanks to the CloudFront zip. The jeremy-shannon checkpoint is the only committed weight
file of this group, and loading a TF 0.12 checkpoint needs TF 1.x.

## C. GTSRB classifiers in PyTorch and Keras with public weights

### C1. Arcos-Garcia, Alvarez-Garcia, Soria-Morillo: aarcosg/tsr-torch (GTSRB leaderboard #1)

- Paper: "Deep neural network for traffic sign recognition systems: An analysis of spatial
  transformers and stochastic optimisation methods", Neural Networks 99 (2018) 158-165; CNN with
  three spatial transformers, "accuracy of 99.71%" (https://raw.githubusercontent.com/aarcosg/tsr-torch/master/README.md).
- Repo: MIT, 41 stars, last commit Oct 2018 (HEAD f21a9c1 2018-10-18)
  (https://img.shields.io/github/stars/aarcosg/tsr-torch.json; clone). Torch7 (Lua) with
  torch/nn, image, cutorch, cunn, cudnn, qassemoquab/stnbhwd and nninit (README). Pretrained model
  `gtsrb_cnn3st_pretrained.zip` on Google Drive id 1iqX1TZBxJSEmaEoReK2ZDko03JHqw1bp (README).
  Training command example: `th main.lua -data GTSRB ... -netType cnn3st -globalNorm -localNorm
  -cNormConv` (README). Based on fb.resnet.torch and Moodstocks/gtsrb.torch (README).
- Runnable today: hard (Torch7 is unmaintained; cutorch/cunn/cudnn and stnbhwd need an old CUDA;
  no CPU path for the STN).

### C2. PyTorch and Keras repositories with weights committed

- poojahira/gtsrb-pytorch: "CNN with 3 layers, 2 fully connected layers and 1 spatial transformer
  network layer", feature maps 3-100-150-250-350, filters [5,3,3]; heavy train- and test-time
  augmentation; "392,090 images for training"; "99.809% on the Private Leaderboard and 99.746% on
  the Public Leaderboard" of the NYU Kaggle InClass challenge (https://www.kaggle.com/c/nyu-cv-fall-2018),
  model "trained on combination of training + validation sets", repeated predictions vary between
  99.699% and 99.746% because of random test-time augmentation
  (https://raw.githubusercontent.com/poojahira/gtsrb-pytorch/master/README.md). Weights
  `model/model_40.pth` (3,430,539 bytes) and `model/pred.csv` are in the repo (git ls-tree of the
  clone). 60 stars, license not specified, HEAD b4c44b0 2026-02-19
  (https://img.shields.io/github/stars/poojahira/gtsrb-pytorch.json; clone). The Kaggle test set is
  the GTSRB test set with hidden labels, split into public/private parts by Kaggle; the exact
  split is unknown. Template repo: soumith/traffic-sign-detection-homework (50 stars, Oct 2018,
  https://raw.githubusercontent.com/soumith/traffic-sign-detection-homework/master/README.md).
  Runnable today: easy (plain PyTorch, torchvision transforms, CPU is fine).
- hello2all/GTSRB_Keras_STN: "99.59% Accuracy on German Traffic Sign Recognition without
  Preprocessing nor Augmentation"; Keras with a custom spatial transformer layer, trained 150
  epochs on the Udacity pickle (34,799 / 4,410 / 12,630), per-category accuracies 99.60 / 99.53 /
  100 / 99.81 / 99.87 / 99.69 / 99.17 vs human 99.72 / 98.67 / 98.89 / 98.00 / 99.93 / 97.63 / 100
  (https://raw.githubusercontent.com/hello2all/GTSRB_Keras_STN/master/capstone_report.md). Weights
  `output/weights.hdf5` (13,904,448 bytes) in the repo (clone). MIT, 51 stars, HEAD 4c40f72
  2020-02-07 (https://img.shields.io/github/stars/hello2all/GTSRB_Keras_STN.json; clone).
  Runnable today: medium (Keras 2 HDF5 with a custom layer; needs `tf.keras` legacy loading or
  Keras 2.x).
- tomlawrenceuk/GTSRB-Dataloader: PyTorch 1.0 `Dataset` wrapper; data as a OneDrive zip; 20 stars,
  Apr 2019, no license, no model (https://raw.githubusercontent.com/tomlawrenceuk/GTSRB-Dataloader/master/README.md).
- hellojialee/Traffic_Sign_Recognition_Efficient_CNNs: companion code of Li and Wang, "Real-time
  traffic sign recognition based on efficient CNNs in the wild", IEEE T-ITS 20(3) 975-984;
  Keras notebooks for a classifier and a MobileNet Faster R-CNN detector with GTSDB sample images
  committed; MIT, 154 stars, HEAD 6325032 2020-04-06
  (https://raw.githubusercontent.com/hellojialee/Traffic_Sign_Recognition_Efficient_CNNs/master/README.md;
  clone). No accuracy numbers in the README; weights: `config.pickle` only, no model file seen in
  the tree (clone, first 40 entries).
- dineshresearch/Novel-Deep-Learning-Model-for-Traffic-Sign-Detection-Using-Capsule-Networks:
  Keras capsule network, "state-of-the-art accuracy of 97.6% on GTSRB" (arXiv 1805.04424), MIT,
  126 stars, Jul 2018, no weights found in README
  (https://raw.githubusercontent.com/dineshresearch/Novel-Deep-Learning-Model-for-Traffic-Sign-Detection-Using-Capsule-Networks/master/README.md).
- frogermcs/GTSRB-TensorFlow-Lite: MobileNet v1 transfer learning to TFLite, Colab notebook, MIT,
  15 stars, Sep 2020 (https://raw.githubusercontent.com/frogermcs/GTSRB-TensorFlow-Lite/master/README.md).
- Kaggle notebooks: not surveyed. Kaggle code pages could not be fetched from the sandbox and web
  search surfaced no notebook URLs; the common Kaggle mirror of GTSRB is a user upload, so numbers
  posted there are on the same 12,630 test images but with unverifiable protocols.

### C3. Hugging Face Hub models trained on GTSRB (API queries `search=gtsrb` and `search=traffic sign`, 2026-10-31)

The two queries returned 100 and 89 models (254 unique together with `traffic-sign`, `tt100k`,
`gtsdb`, `road sign`, `speed limit` and `mapillary`). About 60 of the hits are `YusufBayram-Personal/
shufflenet_GTSRB_krum_round_*` federated learning checkpoints and about 25 are text-to-image LoRAs
or unrelated models. Relevant classifier entries (downloads are the 30-day counts shown by the API):

| Model | Type / base | Claimed accuracy and protocol | Downloads | License (card) |
|---|---|---|---|---|
| bazyl/gtsrb-model | transformers ViT, fine-tuned google/vit-base-patch16-224-in21k on bazyl/GTSRB (https://huggingface.co/bazyl/gtsrb-model/raw/main/README.md) | model-index accuracy 0.99932 "on the evaluation set" after 10 epochs; the card does not say the evaluation set is the official test split (same README) | 87 | apache-2.0 |
| kelvinandreas/vit-traffic-sign-GTSRB | ViT base patch16 224 in21k, safetensors 343,350,100 bytes (https://huggingface.co/api/models/kelvinandreas/vit-traffic-sign-GTSRB?blobs=true) | "Accuracy: 0.9846, F1 Score: 0.9846" on GTSRB, split not stated (https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB/raw/main/README.md) | 171 | mit |
| cvetanovskaa/vit-base-patch16-224-in21k-gtsrb-tuned | ViT, dataset bazyl/GTSRB | no number in README | 48 | apache-2.0 |
| dima806/traffic_sign_detection | ViT in21k classifier (not a detector) | "about 94% accuracy", classification report accuracy 0.9388 on 9,252 images (https://huggingface.co/dima806/traffic_sign_detection/raw/main/README.md); the dataset is not GTSRB-named | 15 | apache-2.0 |
| xpatox08/gtsrb-traffic-signs-v1, gtsrb-vit, gtsrb-vit-model | ViT base patch16 224 | no numbers | 12, 3, 3 | apache-2.0 |
| tzhao3/vit-GTSRB, vit-L-GTSRB, DeiT-GTSRB, DeiT-S-GTSRB | ViT / DeiT | no README | 10, 7, 6, 16 | none |
| sai-prakash-c/swinv2-tiny-patch4-window16-256-gtsrb-ft | SwinV2 tiny | no README (results json only) | 6 | none |
| Hydrogenium9/gtsrb-traffic-sign-models | Keras: best_convnexttiny.keras, best_custom_cnn.keras, best_efficientnetv2b0.keras | no README | 28 | none |
| jack-perlo/MobileNet-Gtsrb, MobileNetV2-Gtsrb, ResNet18-Gtsrb, ResNet34-Gtsrb | Keras .keras plus int8/uint8 .tflite | no README | 21, 19, 11, 5 | none |
| finnff/GTSRB_trained | torchvision EfficientNet / MaxViT / RegNet .pth, 10 epochs | "Validation Accuracy: 99.83%" (EfficientNet), split not stated (https://huggingface.co/finnff/GTSRB_trained/raw/main/README.md) | 0 | mit |
| xieyuy/traffic-sign-adverse-weather | timm ConvNeXtV2-base, EfficientNetV2-M, SwinV2-B | "Macro-F1 = 0.945 (Evaluated on 2-Core CPU without GPU)", custom adverse weather set (https://huggingface.co/xieyuy/traffic-sign-adverse-weather/raw/main/README.md) | 0 | mit |
| tanganke/clip-vit-base-patch32_gtsrb, clip-vit-large-patch14_gtsrb, clip-vit-base-patch16_gtsrb (+ LoRA variants), crisostomi/ViT-*-GTSRB, gladia/ViT-B-16__*-gtsrb, hoffman-lab/KnOTS-*_gtsrb, yongxianwei/clip-base-32_gtsrb, EvaristeL/vitb32-GTSRB | CLIP fine-tunes from model-merging / task-arithmetic research | see the VLM document; the most downloaded GTSRB models on the Hub (19,362 / 13,333 / 11,497) | | none stated |

Datasets used by these models (for replication): tanganke/gtsrb (train 26,640 / test 12,630 plus
corruption splits, parquet, 16,418 downloads, no license tag,
https://huggingface.co/api/datasets/tanganke/gtsrb?full=true); bazyl/GTSRB (parquet train/test,
license tag gpl-3.0, 2,295 downloads, https://huggingface.co/api/datasets/bazyl/GTSRB?full=true);
FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset (synthetic, cc-by-4.0, 152 downloads; paper
arXiv 2512.05936, 105,500 images of 211 German classes, https://arxiv.org/abs/2512.05936).
torchvision `datasets.GTSRB` downloads from
`https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/` with md5 checks
(https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py), which is
not reachable from this sandbox.

### C4. Pointer: VLM / CLIP / open-vocabulary

CLIP fine-tunes (tanganke/*, crisostomi/*, gladia/*, hoffman-lab/KnOTS-*), Qwen3-VL and Gemma-3
LoRAs for sign understanding (longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs,
RidaKo/qwen3-vl-road-signs-lt-lora, AKrasavcev/lora_gemma3_4b_lt_road_signs,
GuilhermeRatti/qwen2-7b-instruct-trl-sft-GTSDB_Detect) and the NEC Labs relabelled Mapillary
validation set for VLM evaluation (arXiv 2508.02047, https://github.com/nec-labs-ma/relabeling,
2,000 images, 12 categories, images at https://mapillary-signs.s3.us-west-2.amazonaws.com/images.zip
per https://raw.githubusercontent.com/nec-labs-ma/relabeling/master/README.md) are covered in the
separate VLM document.

## D. Detection on GTSDB, TT100K, MTSD, DFG, CCTSDB, STS (2016 to 2026)

### D1. aarcosg/traffic-sign-detection (GTSDB, TensorFlow Object Detection API, 2018)

- Paper: Arcos-Garcia, Alvarez-Garcia, Soria-Morillo, "Evaluation of deep neural networks for
  traffic sign detection systems", Neurocomputing 316 (2018) 332-344
  (https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/README.md).
- Repo: MIT, 335 stars, last commit Feb 2020 (HEAD 9949b3d 2020-02-13); TensorFlow Object
  Detection API (TF1) plus darkflow for YOLOv2 (shields; clone; README). Files: `Evaluation.ipynb`,
  `Run_models_on_new_images.ipynb`, `scripts/create_gtsdb_tf_record.py`,
  `scripts/create_gtsdb_3classes_tf_record.py`, label maps (clone).
- Classes: 3 superclasses `prohibitory`, `mandatory`, `danger`
  (https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/scripts/gtsdb3_label_map.pbtxt).
- Weights and data: GTSDB TFRecords on Google Drive (id 1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq) and 8
  model zips on Google Drive: faster_rcnn_inception_resnet_v2_atrous, faster_rcnn_inception_v2,
  faster_rcnn_resnet_101, faster_rcnn_resnet50, rfcn_resnet101, ssd_inception_v2, ssd_mobilenet_v1,
  yolo_v2 (README; the Drive links were not downloaded in this session).
- Results (README table; AP computed per class at IoU 0.5 with the VOC 2010+ area method in
  `Evaluation.ipynb`, which also has a COCO-style 0.5:0.95 function):
  Faster R-CNN Resnet 50 mAP 91.52; Resnet 101 95.08; Inception V2 90.62; Inception Resnet V2 95.77;
  R-FCN Resnet 101 95.15; SSD Mobilenet 61.64; SSD Inception V2 66.10; YOLO V2 78.83. Small /
  medium / large sign mAP, e.g. Faster R-CNN Resnet 101 70.89 / 94.17 / 88.87 and SSD Mobilenet
  22.13 / 55.32 / 82.06 (same README; https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/Evaluation.ipynb).
- Runnable today: medium. TF1 frozen graphs can be run with `tf.compat.v1` in TF 2.x; the
  notebooks target TF1 APIs. The GTSDB TFRecords from Drive avoid the unreachable sid.erda.dk.

### D2. Tabernik and Skocaj, skokec/detectron-traffic-signs (DFG, Mask R-CNN, 2019)

- Paper: "Deep Learning for Large-Scale Traffic-Sign Detection and Recognition", IEEE T-ITS 2019,
  arXiv 1904.00649: Mask R-CNN with OHEM, even small/large ROI sampling and size-weighted losses,
  200 categories, DFG split 5,254 training / 1,703 test images, annotations of at least 30 px;
  Table III: Faster R-CNN mAP50 92.4 / mAP50:95 80.4 / max recall 93.8; Mask R-CNN ResNet-50
  without adaptations 93.0 / 82.3 / 94.6; with adaptations 95.2 / 82.0 / 96.5; with adaptations and
  data augmentation 95.5 / 84.4 / 96.5 (https://arxiv.org/pdf/1904.00649, section VI.D and Table III).
- Repo: Apache-2.0 (Detectron license), 38 stars, last commit Apr 2026 (HEAD 0c11316 2026-04-21 on
  the default branch `villard`); Caffe2 Detectron fork with a custom
  `WeightedSigmoidCrossEntropyLoss` op in `caffe2-modules/`
  (https://raw.githubusercontent.com/skokec/detectron-traffic-signs/villard/README.md; clone).
- Weights: `https://data.vicos.si/skokec/villard/dfg_mask_rcnn_weights.tar.gz` (HTTP 200,
  643,851,795 bytes) and YAML configs `detectron-model-yaml-DFG-dataset.zip` (1,842 bytes) and the
  augmented variant (README; curl -I on 2026-10-31). README warns: "Due to slight update of the DFG
  dataset the resulty may vary from the ITS 2019 paper".
- Runnable today: hard (Caffe2 Detectron, custom CUDA op, GPU only).

### D3. Official TT100K code (Zhu et al., CVPR 2016)

- Site: https://cg.cs.tsinghua.edu.cn/traffic-sign/ , paper "Traffic-Sign Detection and
  Classification in the Wild". Download links are direct: `data_model_code/code.zip`
  (801,430,483 bytes), `cvpr_fastrcnn.zip` (370,022,700 bytes, "Fast R-CNN Model and Results"),
  `data_model_code/data.zip` (19,152,969,603 bytes), `tt100k_2021.zip` (19,205,284,669 bytes) and
  five "no sign" parts of about 17 GB each (HEAD requests on 2026-10-31). The page states
  "Tsinghua-Tencent 100K dataset is under Creative Commons Attribution-NonCommercial (CC-BY-NC)
  license" and "The 2021 version ... could NOT be used independently" and asks for name, email and
  organisation before download, although the links work without the form (same page).
- Contents of `code.zip` (central directory read remotely): `code/caffe/` (a Caffe fork with its
  git history), `code/model/model.caffemodel` (467,396,710 bytes), `model.prototxt`,
  `solver.prototxt`, `train_val.prototxt`, `code/python/anno_func.py`, `data_show.ipynb`,
  `deployer.ipynb`, `eval_check.ipynb`, `code/script/1_convert_lmdb.sh`, `2_calc_img_mean.sh`, and
  `code/results/` with `ours_result.json`, `fastrcnn_result.json` and reports.
  `code/README.md`: "caffe: caffe code that we used; model: our caffe model file; python: some
  python utils; results: some results in our experiments".
- Reported numbers (`code/results/report_ours.txt`): "iou:0.5, size:[0,400), types:[... total
  45 ...], accuracy:0.877, recall:0.906"; by size [0,32) 0.822 / 0.867, [32,96) 0.913 / 0.941,
  [96,400) 0.905 / 0.875. Fast R-CNN baseline (`report_fastrcnn.txt`): accuracy 0.501, recall 0.555
  overall, recall 0.238 for signs under 32 px. ("accuracy" here is precision of detections at
  IoU 0.5 over the 45 classes listed in the report; the paper's rule for selecting these 45
  classes was not re-checked here.)
- Runnable today: hard (BVLC Caffe fork, LMDB conversion scripts, GPU, 19 GB data).

### D4. MTSD baselines (Mapillary, 2020)

- Paper: Ertler et al., "The Mapillary Traffic Sign Dataset for Detection and Classification on a
  Global Scale", arXiv 1909.04422 (ECCV 2020). Splits 36,589 train / 5,320 val / 10,544 test images
  plus 47,547 partially annotated; "the annotations for the test set will not be released";
  baseline: Faster R-CNN FPN binary detector trained on 1000x1000 crops at full resolution, plus a
  separate classifier; "Our classifier in combination with ResNet101 binary detector reaches 83.4
  mAP over all 313 classes; the ResNet50 variant is only about 2 points lower"
  (https://arxiv.org/pdf/1909.04422, sections 2.5 and 5). No code repository is named in the PDF
  (grep for github.com found nothing); dataset at https://www.mapillary.com/dataset/trafficsign
  (free for academic research per the abstract).
- Community mirrors of MTSD crops exist on the Hub (ThankGod/mapillary_traffic_sign_dataset and
  variants, 647 / 117 / 63 downloads, https://huggingface.co/api/datasets?search=mapillary%20traffic)
  but no MTSD-trained detector weights were found on GitHub or the Hub.

### D5. CCTSDB and CCTSDB2021 (CSUST)

- csust7zhangjm/CCTSDB2021: dataset release only (Google Drive and Baidu links), GPL-3.0 LICENSE
  file, 102 stars, HEAD de484cd 2026-07-25. "16356 training set images", "positive sample test set
  has 1500 images", 3 classes mandatory / prohibitory / warning, 500 negative samples; the README
  lists the group's detectors (HQ-YOLO, Neurocomputing 2026; the anchor-free one-level detector,
  IEEE TETCI 2024; ReYOLO 2022; cascaded R-CNN 2020) but links no code or weights
  (https://raw.githubusercontent.com/csust7zhangjm/CCTSDB2021/master/README.md; clone).
- csust7zhangjm/CCTSDB (2017 version): annotation text only, 271 stars, HEAD 2024-07-12 (clone).
- Genius-Society/yolov4_tt100k (Hugging Face): MIT, README only, "Testing on the CCTSDB 2021
  dataset showed ... mAP of up to 92.77%", "81 frames per second"; the weights are on the
  ModelScope mirror https://www.modelscope.cn/models/Genius-Society/yolov4_tt100k, not on the Hub
  (file list has only `.gitattributes` and `README.md`,
  https://huggingface.co/api/models/Genius-Society/yolov4_tt100k?full=true). Not verified beyond
  the card.
- HaoqinHong/CCSPNet-Joint (IJCNN 2024, joint denoising + YOLOv5l with EfficientViT-b2 backbone,
  CCTSDB-AUG): GPL-3.0, 28 stars, last commit Jan 2026; weights on Kaggle and Baidu
  (https://raw.githubusercontent.com/HaoqinHong/CCSPNet-Joint/master/README.md).

### D6. Swedish Traffic Signs (STS / STSD)

- Li and Wang 2018 (T-ITS, repo hellojialee, C2) and "Efficient Traffic-Sign Recognition with
  Scale-aware CNN" (arXiv 1805.12289, "99.88% precision and 96.61% recall on the Swedish Traffic
  Signs Dataset", https://arxiv.org/abs/1805.12289) are the main STS results; the latter has no
  code link on its abstract page. No STS-trained weights were found in this survey.

### D7. YOLO-family traffic sign detectors with weights (GitHub and Hugging Face)

- keremberke: no traffic sign models exist (author query returned 86 models, none sign-related,
  https://huggingface.co/api/models?author=keremberke&limit=200); the related Hub entry is a dataset,
  keremberke/german-traffic-sign-detection (GTSDB re-export from Roboflow Universe, 383 train /
  108 valid / 54 test images, 43 labels, "License: CC BY 4.0",
  https://huggingface.co/datasets/keremberke/german-traffic-sign-detection/raw/main/README.md).
- Hub detectors with weights and a stated metric (all are Ultralytics-format `.pt` checkpoints;
  the Ultralytics training and inference code is AGPL-3.0, while the license below is the one the
  uploader declared on the card for the checkpoint):
  - cvtechniques/TrafficSignDetection: YOLO11n on the LISA traffic sign dataset, "mAP@0.5 0.994,
    mAP@0.5-0.95 ~0.89, Best F1 0.99" on its validation split, mit, 153 downloads
    (https://huggingface.co/cvtechniques/TrafficSignDetection/raw/main/README.md).
  - neitpm/traffic-sign-yolo11-benchmark: YOLO11n baseline "70.27% mAP@[0.5:0.95]" on a custom
    5,668 / 1,259 / 1,151 split, plus UIB-nano and distilled variants, Dockerfile, mit, 93 downloads
    (https://huggingface.co/neitpm/traffic-sign-yolo11-benchmark/raw/main/README.md).
  - star092304/traffic-sign-detection-vietnam-yolo: Vietnamese signs, dataset on the Hub (CC BY 4.0,
    size n<1K), "mAP50 98.06%, mAP50-95 83.57%", apache-2.0, 93 downloads
    (https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo/raw/main/README.md).
  - OrDinaD/belarus-road-signs-yolov8: "mAP50 0.7252, mAP50-95 0.6106", mit, 30 downloads
    (https://huggingface.co/OrDinaD/belarus-road-signs-yolov8/raw/main/README.md).
  - ankitjha07/Traffic-Sign-Detection-YOLOv8 (mit, 128 downloads, results.csv but no number in the
    card), yahyagul/traffic-sign-yolov8 (Roboflow data, mit, 86), JakobJFL/yolov8-dk-Traffic-Signs
    (Danish, apache-2.0), Nughm3/yolov8-traffic-signs, Erpix3lt/traffic-sign-detection (YOLO26 on a
    Roboflow set, .pt and .onnx, mit), jcasdvasj/yolov12-traffic-sign-detection (zips only, mit),
    jkot/pova-traffic-sign-recognition-models (YOLO binary detectors at 1280/1920 plus
    classifiers, no card) (https://huggingface.co/api/models?search=traffic%20sign&limit=100).
  - youssefhazem/yolov8s-tt100k: empty repository (only `.gitattributes`,
    https://huggingface.co/api/models/youssefhazem/yolov8s-tt100k?full=true).
- GitHub (from the `traffic-sign-detection` and `traffic-sign-recognition` topic pages, 102 and
  192 repositories, https://github.com/topics/traffic-sign-detection,
  https://github.com/topics/traffic-sign-recognition):
  - MDhamani/Traffic-Sign-Recognition-Using-YOLO: YOLOv5 on GTSRB crops, ">93% mAP", weights on
    mega.nz, 81 stars, Jan 2023, no license
    (https://raw.githubusercontent.com/MDhamani/Traffic-Sign-Recognition-Using-YOLO/master/README.md).
  - AmeyaWagh/traffic_sign_detection: darkflow tiny-YOLO on LISA, pb/meta on Google Drive, MIT, 75
    stars, Jan 2020 (https://raw.githubusercontent.com/AmeyaWagh/traffic_sign_detection/master/README.md).
  - wyfsh/YOLO-LISA: Darknet YOLOv2 on LISA-TS, 26 stars, Jun 2017
    (https://raw.githubusercontent.com/wyfsh/YOLO-LISA/master/README.md).
  - fredotran/traffic-sign-detector-yolov4: YOLOv4 with OpenCV DNN on the 877-image Kaggle
    "road-sign-detection" set (4 classes), 32 stars, Nov 2025
    (https://raw.githubusercontent.com/fredotran/traffic-sign-detector-yolov4/master/README.md).
  - bhaskrr/traffic-sign-detection-using-yolov11: YOLO11n on a Roboflow "Self-Driving Cars" set
    (4,969 images, 15 classes, 416 px), 34 stars, Oct 2024
    (https://raw.githubusercontent.com/bhaskrr/traffic-sign-detection-using-yolov11/master/README.md).
  - MehmetOKUYAR/yolov7_tensorrt_test: YOLOv7 + TensorRT on a custom set, MIT, 19 stars
    (https://github.com/topics/traffic-sign-dataset).
  - Savatage7/Yolo5_TT100K (0 stars, Jun 2022) and lunian/Improved-YOLOv8 (1 star, Mar 2025) are
    the only TT100K YOLO repos surfaced by search; neither README states weights
    (https://raw.githubusercontent.com/Savatage7/Yolo5_TT100K/master/README.md,
    https://raw.githubusercontent.com/lunian/Improved-YOLOv8/master/README.md).
- Improved-YOLO papers on TT100K/CCTSDB 2021 to 2026 with code: the Hub paper search for "traffic
  sign detection" returns YOLO-TS (arXiv 2410.17144, TT100K and CCTSDB2021) and EMDFNet (arXiv
  2408.14189); neither PDF contains a GitHub link (https://arxiv.org/pdf/2410.17144,
  https://arxiv.org/pdf/2408.14189; https://huggingface.co/api/papers/search?q=traffic%20sign%20detection).
  YOLO-CCA (arXiv 2412.04289) and ETSR-YOLO (PLOS ONE 2023) were surfaced by web search but not
  checked for code. Conclusion: for TT100K the only verified downloadable modern weights are the
  Vitis AI YOLOX-m (D9) and the ModelScope-hosted yolov4_tt100k.

### D8. RT-DETR, D-FINE, DEIM, RF-DETR and DETR-family traffic sign detectors

- merve/rfdetr-roadsign-agree1/agree2 (and large/medium, noaug variants): RF-DETR fine-tunes on an
  unnamed road sign set, apache-2.0, "Map 0.2815, Map 50 0.3324" for agree2, 7 to 25 downloads
  (https://huggingface.co/merve/rfdetr-roadsign-agree2/raw/main/README.md).
- Jesse020202/detr_finetuned_roadsigns (+ poisoned / comparison variants): Conditional DETR
  ResNet-50 fine-tunes, apache-2.0, no metric in the card (API listing).
- iamrizwan077/detr-finetuned-road-signs-v1, maier-s/detr_fine_tuned_traffic_signs: no cards.
- No RT-DETR, D-FINE or DEIM traffic sign checkpoint was found on the Hub or in the topic pages.
  CSDETR (improved RT-DETR) exists for traffic lights, not signs (see the traffic light survey).

### D9. Vendor and framework configs: Vitis AI, MMDetection, Detectron2

- AMD/Xilinx Vitis AI 2.5 model zoo entry 21 "ADAS Traffic sign Detection,
  pt_yolox_TT100K_640_640_73G_2.5, 0.623 float / 0.621 quantized, 640*640, 73G"
  (https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/README.md). The model yaml
  gives the public download https://www.xilinx.com/bin/public/openDownload?filename=pt_yolox_TT100K_640_640_73G_2.5.zip
  (redirects to https://download.amd.com/opendownload/xlnx/pt_yolox_TT100K_640_640_73G_2.5.zip,
  HTTP 200, 594,249,291 bytes) and board xmodels `tsd_yolox_pt-*`, license pointer to the Vitis AI
  LICENSE (Apache-2.0) (https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/model-list/pt_yolox_TT100K_640_640_73G_2.5/model.yaml).
  Inside the zip (central directory read remotely): `float/best_ckpt.pth` (202,928,501 bytes),
  `qat/epoch_58_ckpt.pth`, `quantized/YOLOX_0_int.xmodel`, a YOLOX code tree with
  `exps/example/custom/yolox_m_tt100k_float.py`, `run_train.sh` (4 GPUs, batch 64, fp16; eval with
  `tools/eval.py ... --conf 0.001`) and `readme.md`: "performance is evaluated on TT100k val",
  "YOLOx-m-modified, 640x640, 72.96G FLOPs, AP 62.3; QAT INT8 AP 62.1", preprocessing "RGB(0~255),
  mean 0, std 1". The AP definition (COCO 0.5:0.95 as in YOLOX, or AP50) is not stated. The entry
  is absent from the Vitis AI 3.0 and 3.5 model zoo READMEs (grep of
  https://raw.githubusercontent.com/Xilinx/Vitis-AI/v3.5/model_zoo/README.md returned no "yolox").
- MMDetection / mmpretrain: no traffic sign config in open-mmlab/mmpretrain (tree grep of the clone
  at HEAD ee7f2e8 2024-11-01 for gtsrb/traffic found nothing). TIER IV's AWML (mmdet/mmpretrain
  based) trains only traffic light models (see the traffic light survey).
- Detectron2: moabitcoin/signfeld ("Signfeld in PyTorch", synthetic training from sign templates
  pasted on neutral images): RetinaNet ResNet-50 "Calzone", "GTSDB mAP 67.23", weights
  `DE-signs-169.pth` on GitHub releases (HTTP 200, 317,883,321 bytes), 169 German sign classes,
  MIT, 13 stars, May 2020 (https://raw.githubusercontent.com/moabitcoin/signfeld/master/README.md;
  curl -I https://github.com/moabitcoin/Signfeld/releases/download/v1.0/DE-signs-169.pth).
  Related: LCAD-UFES/publications-tabelini-ijcnn-2019 (training Faster R-CNN from templates and
  arbitrary natural images, IJCNN 2019, uses endernewton/tf-faster-rcnn, generated GTSDB-style data
  on Google Drive, 18 stars, Jul 2020,
  https://raw.githubusercontent.com/LCAD-UFES/publications-tabelini-ijcnn-2019/master/README.md).
- sovit-123/fasterrcnn-pytorch-training-pipeline (torchvision Faster R-CNN trainer, MIT, 275 stars,
  Oct 2025) is a generic pipeline often used with GTSDB in tutorials but ships no sign weights
  (https://raw.githubusercontent.com/sovit-123/fasterrcnn-pytorch-training-pipeline/master/README.md).

### D10. Roboflow Universe

- Dozens of "Traffic Sign Detection" projects exist (for example
  https://universe.roboflow.com/traffic-signs-ryni7/traffic-signs-detection-auxjz: 7 classes, no
  image count or metrics shown, "LICENSE: CC BY 4.0", "Hosted Inference API and other deployment
  methods"). Licenses are chosen by each uploader (CC BY 4.0 is the default on the pages checked)
  and bind the dataset, while Roboflow-trained model weights are served through Roboflow's hosted
  or self-hosted inference with an API key rather than as downloadable files (the page checked
  offers no weight download; this statement is otherwise not verified). Datasets can be exported
  in YOLO/COCO formats after login. The GTSDB re-export on Roboflow
  (mohamed-traore-2ekkp/gtsdb---german-traffic-sign-detection-benchmark) is what
  keremberke/german-traffic-sign-detection mirrors (D7).

### D11. Panoramax (France): open detector and per-country classifiers (2024 to 2026)

- Panoramax/detect_face_plate_sign: "trained for the Panoramax project in order to detect people
  face to blur them, licence plates to blur them, road signs to classify them with other models";
  YOLO11l at imgsz 2048, 300 epochs; validation on 329 images: all mAP50 0.815 / mAP50-95 0.412,
  sign P 0.879 R 0.836 mAP50 0.898 mAP50-95 0.561 (507 instances), plate 0.889 / 0.438, face
  0.657 / 0.237; files `yolo11l_panoramax.pt` and `yolov8s_panoramax.pt`; license etalab-2.0
  (https://huggingface.co/Panoramax/detect_face_plate_sign/raw/main/README.md).
- Panoramax/classify_de_road_signs: "fine tuned version of classify_de_road_signs classification
  model (based on YOLOv26)", "130+ types of road signs", model-index accuracy 0.986,
  `classify_de_road_signs.pt` (26,282,865 bytes), etalab-2.0, created 2026-03-29
  (https://huggingface.co/Panoramax/classify_de_road_signs/raw/main/README.md;
  https://huggingface.co/api/models/Panoramax/classify_de_road_signs?blobs=true). Training set
  Panoramax/classified_de_road_signs: "20000+ photos of DE road signs in 130+ classes", crops made
  with detect_face_plate_sign from Panoramax pictures of Germany, extra "bad"/"other" classes,
  CC BY-SA 4.0, `train.zip` and `val.zip`
  (https://huggingface.co/datasets/Panoramax/classified_de_road_signs/raw/main/README.md).
- Panoramax/classify_fr_road_signs (accuracy 0.98717, 2024-06-23), classify_be_road_signs
  (2026-03-02), classify_nl_road_signs (2026-01-04), detect_fr_road_signs_subsigns (2024),
  detect_nl_subsigns and classify_nl_subsigns (2026-09, CC BY-SA 4.0): all etalab-2.0 unless noted,
  Ultralytics `.pt` files (https://huggingface.co/api/models?author=Panoramax&limit=50;
  https://huggingface.co/api/models/Panoramax/classify_fr_road_signs?full=true). Downloads shown by
  the API are 0 for all, so these are little known.
- Why it matters: this is the only open, maintained, permissively licensed detector + fine-grained
  classifier chain for European signs found in this survey, with training data released. Note the
  Ultralytics training code is AGPL-3.0; the `.pt` files are Ultralytics checkpoints.

### D12. Japan: Infra Intelligence road-sign-yolox-tiny (2026)

- Hub model infra-intelligence/road-sign-yolox-tiny (created 2026-10-22, 34 downloads, license
  apache-2.0 for weights and code, tags ja/en/japan/traffic-sign)
  (https://huggingface.co/api/models/infra-intelligence/road-sign-yolox-tiny?full=true). Model card
  (Japanese with an English summary, https://huggingface.co/infra-intelligence/road-sign-yolox-tiny/raw/main/README.md):
  - Task: detect Japanese road sign plates from car-mounted smartphone or dashcam images and sort
    them into 5 classes by shape and colour following 道路標識令: `guide` (案内), `warning` (警戒),
    `regulatory` (規制), `indication` (指示), `supplementary` (補助), plus a 6th `hard_negative`
    score column (backs of plates, shop signs, traffic lights, delineators, message boards) that is
    not output as a sign. It does not identify the sign type ("一時停止" or "最高速度").
  - Architecture: YOLOX-Tiny, about 5.03 M parameters, trained from scratch on Apple Silicon (MPS),
    batch 8, AdamW 1e-3 cosine, SimOTA, mosaic; input `[1,3,736,1280]` RGB 0 to 255 (normalisation
    inside the model); outputs `boxes [200,4]` and `scores [200,6]`, no NMS inside; recommended
    post-processing: argmax over 6 columns, class-wise NMS IoU 0.65, threshold 0.5.
  - Files: `road_sign_yolox_tiny_v9.pth` (20,373,347 bytes), `road_sign_yolox_tiny_v9.onnx`
    (20,426,324 bytes, opset 17, fixed 1280x736), `model.py` (torch + torchvision + Pillow only),
    `examples/predict_onnx.py`, `config.json`, `MODEL_CONTRACT.md`, `eval/evaluation.json`,
    `eval/onnx_parity.json`, `LICENSE`, `NOTICE` (blobs listing).
  - Data: Mapillary images from Japan (CC BY-SA 4.0), 4,585 training and 992 validation images
    from 15 areas; drafts from Mapillary's automatic detections, boxes checked and redrawn by an
    "AI assistant" under written rules, not reviewed image by image; the dataset is not released.
  - Results: held-out test of 596 images from 7 other areas (988 boxes, 117 images without signs)
    at threshold 0.5: recall 0.857 (847/988, class-agnostic), precision 0.877, class accuracy on
    matched plates 0.980, 12.0 false alarms per 100 sign-free images, AP50 0.902, AP50-95 0.700;
    per class AP50 guide 0.594, warning 0.958 (9 boxes), regulatory 0.937, indication 0.893,
    supplementary 0.748; by size (width at 2048 px long side) AP50 0.759 (under 32 px), 0.861,
    0.878, 0.642 (128 px and above). Validation (992 images): AP50 0.925, recall 0.869, precision
    0.906. The author re-ran the published `model.py` and weights on CPU and got the same val AP50
    0.925 ("torch 2.13.0 で動くことを確かめています").
  - Caveats stated by the author: test labels were also AI-generated; weak on large near plates,
    small intersection-name and supplementary plates; night, rain and snow not evaluated; Japan
    only; not for safety-critical use.
- Runnable today: easy (ONNX Runtime CPU or plain PyTorch). The number cannot be reproduced because
  the evaluation images are private, but the model is the natural qualitative baseline and label
  space reference for OpenTSR's Japanese detector stage.

### D13. Other dataset-specific repos seen

- ituracingdriverless/TTVS (Turkish traffic sign dataset, 80 stars, May 2026, no license badge).
- ptran1203/traffic_sign_detection (Zalo AI 2020, RetinaNet with tiling, 27 stars).
- lynguyenminh/CS114.M11.Vietnam-traffic-sign-detection (47 stars), Hoang42 and Minh124689 Hub
  models (Vietnam), aresad/russian-road-signs, OrDinaD (Belarus), JakobJFL (Denmark),
  eugeneyan84 (Singapore, SSD MobileNet V2 FPNLite, 32-class classification set released, detection
  set withheld for PDPA reasons, GPL-3.0, https://raw.githubusercontent.com/eugeneyan84/Classification-and-Detection-of-Singapore-Road-Traffic-Signs/master/README.md).
  (Star counts from https://github.com/topics/traffic-sign-detection and
  https://github.com/topics/traffic-sign-recognition.)

## E. Production and industry stacks

### E1. Autoware Universe

- autowarefoundation/autoware_universe, 1.8k stars, last commit "yesterday" (HEAD 1a411ff
  2026-10-30), Apache-2.0 (shields; clone). A tree search of the clone for `traffic_sign`,
  `trafficsign`, `road_sign`, `speed_limit`, `speedlimit`, `tsr` finds only
  `perception/autoware_map_based_prediction/.../traffic_signal.*` and
  `planning/autoware_diffusion_planner/.../traffic_signals.*`, i.e. traffic lights. There is no
  traffic sign detection or classification node; speed limits come from the Lanelet2 map
  (regulatory elements), not from the camera. Autoware's Hugging Face organisation publishes only
  traffic light and object detection models (see the traffic light survey).

### E2. Baidu Apollo

- ApolloAuto/apollo, 27k stars, HEAD d53aa3d 2026-04-16, Apache-2.0 (shields; clone). The
  `modules/perception/` directory at HEAD contains barrier_recognition, camera_detection_*,
  lane_detection, lidar_*, radar_*, traffic_light_detection / recognition / region_proposal /
  tracking and no traffic sign module (git ls-tree of the clone). Signs exist only as HD-map
  elements: `modules/common_msgs/map_msgs/map_stop_sign.proto`, `map_yield_sign.proto`,
  `map_speed_control.proto`, `map_speed_bump.proto`, and Dreamview renders them
  (`modules/dreamview/frontend/src/renderer/traffic_controls/traffic_signs.js`) (same clone).

### E3. comma.ai openpilot and its forks

- commaai/openpilot, 64k stars, MIT, HEAD afedebc 2026-10-30 (shields; clone).
  `docs/LIMITATIONS.md`: "In presence of traffic signs or stop lights, which are not detected by
  openpilot at this time." and "When the posted speed limit is below the user selected set speed.
  openpilot does not detect speed limits at this time."
  (https://raw.githubusercontent.com/commaai/openpilot/master/docs/LIMITATIONS.md). No sign model
  in the tree (grep of the clone for sign / speed_limit / mapd found nothing).
- sunnypilot/sunnypilot, 2.2k stars, MIT, HEAD 6f7a2c9 2026-10-29 (shields; clone). "Speed Limit
  Assist" has two sources: "Car State: Speed limit information provided by the vehicle's built-in
  sensors (e.g., Traffic Sign Recognition cameras)" and "Map Data: Speed limits from downloaded
  OpenStreetMap data" (https://community.sunnypilot.ai/t/249). The resolver confirms this:
  `SpeedLimitSource.car` reads `sm['carStateSP'].speedLimit` and `SpeedLimitSource.map` reads
  `sm['liveMapDataSP']`; policies `car_state_only`, `map_data_only`, `car_state_priority`,
  `map_data_priority`, `combined`
  (https://raw.githubusercontent.com/sunnypilot/sunnypilot/master/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py).
  No camera-based sign recognition of its own.
- FrogAi/FrogPilot, 568 stars, MIT, HEAD 51e1121 2026-10-31 on branch `FrogPilot-Staging`
  (shields; clone). It does ship vision TSR:
  - README: "It reads speed limit signs with the road camera and, on supported Ford, Genesis,
    Hyundai, Kia, Lexus, and Toyota models, your car's dashboard. It also uses maps from
    OpenStreetMap and the speed limits along your Navigate on openpilot route, and, with your own
    Mapbox key, fills gaps with Mapbox's online data." Source priority is user-configurable
    ("Speed Limit Source Priority"). "Speed Limit Filler ... learns the limit from the signs the
    camera reads, saves it for that road and direction ... keeps a cropped photo of the sign"; with
    "Share FrogPilot Data" on, learned limits and photos are uploaded to FrogPilot
    (https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/README.md).
  - Code: `frogpilot/system/speed_limit_reader.py` loads `speed_limit_reader.mnn` with the MNN
    runtime on CPU (`{"backend": "CPU", "numThread": 1}`), subscribes to the road camera
    (`VisionIpcClient("camerad", VISION_STREAM_ROAD)`), feeds the raw NV12 buffer plus text and
    numeric context (map match, GPS, pose) and publishes `frogpilotSignReading` with fields such
    as `speedLimit`, `heldSignConfidence` and `heldSignLeft/Top/Right/Bottom`; crops of confident
    signs are saved as JPEG with bearing/latitude/longitude JSON
    (https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/system/speed_limit_reader.py).
    `frogpilot/system/speed_limit_reader_models.py` downloads newer models from
    `FROGPILOT_API + /v1/speed-limit-reader/<version>/<file>` with a bearer token;
    `SPEED_LIMIT_READER_MODEL_FILES = ("speed_limit_reader.mnn",)`
    (https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/system/speed_limit_reader_models.py;
    https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/common/frogpilot_variables.py).
  - Weights: `frogpilot/assets/models/speed_limit_reader.mnn`, 12,601,448 bytes, committed to the
    repo (git ls-tree -l of the clone). Architecture, training data, accuracy and the model's own
    license are not documented anywhere found (the repo is MIT). Runnable outside openpilot: hard
    (MNN graph with openpilot-specific inputs).
- Tesla and Mobileye: nothing open-sourced for TSR (no repositories exist; not further checked).

### E4. Library and framework support

- torchvision: `torchvision.datasets.GTSRB(root, split="train"|"test", download=...)` reads
  `GTSRB/Training` and `GTSRB/Final_Test/Images` with `GT-final_test.csv`; downloads from
  `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/` with md5
  513f3c79a4c5141765e10e952eaa2478 / c7e4e6327067d32654124b0fe9e82185 / fe31e9c9270bbcd7b84b7f21a9d9d9e5
  (https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py). No
  pretrained sign model in torchvision.
- TensorFlow Datasets: GTSRB appears only inside `visual_domain_decathlon` ("gtsrb", "German
  Traffic Signs", 43 classes, images resized by the Decathlon authors, hosted at
  robots.ox.ac.uk / image-net.org)
  (https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image_classification/visual_domain_decathlon.py);
  no standalone GTSRB or GTSDB builder (tree grep of the clone).
- timm (huggingface/pytorch-image-models, HEAD 03b76b6 2026-10-30), keras-io (HEAD 5a1293e
  2026-10-28) and mmpretrain (HEAD ee7f2e8 2024-11-01): no GTSRB or traffic sign example or config
  (tree grep of the clones for gtsrb / traffic_sign / tt100k / gtsdb).

## F. Embedded and edge deployments

- Classical OpenCV pipeline with code: hoanglehaithanh/Traffic-Sign-Detection (292 stars, MIT, Dec
  2018): colour segmentation and contour-based candidate extraction on video, SVM classifier
  (`classification.py`, `data_svm.dat`) over 12 crop folders of a self-made dataset, dense optical
  flow tracking, OpenCV 3; "A new version using SSD will be released" never appeared
  (https://raw.githubusercontent.com/hoanglehaithanh/Traffic-Sign-Detection/master/README.md).
  Vicondrus/Roadster (classical extraction + CNN, 23 stars) and
  Praveenterax/Traffic-sign-Detection-Recognition-Matlab-RandomForest (13 stars, MATLAB) are
  similar (https://github.com/topics/traffic-sign-detection). No accuracy numbers on a public
  benchmark in any of them.
- NVIDIA Jetson / TensorRT: dusty-nv/jetson-inference (HEAD 45da40a 2025-10-16) has no traffic sign
  model or example (tree grep). MehmetOKUYAR/yolov7_tensorrt_test (YOLOv7 TensorRT, custom data,
  MIT) and freicar-2022-1/freicar_project_sign_detect (ROS YOLOv7) are community projects
  (https://github.com/topics/traffic-sign-detection). No NVIDIA TAO or NGC traffic sign model was
  found (not exhaustively verified).
- Raspberry Pi: the GTSDB fine-tuning paper arXiv 2003.03256 tested Faster R-CNN Inception v2 and
  Tiny-YOLOv2 "on the host PC as well as Raspberry Pi 3 Model B+" but has no code link
  (https://arxiv.org/abs/2003.03256). frogermcs/GTSRB-TensorFlow-Lite (MobileNet v1 TFLite, Colab)
  and jack-perlo/*-Gtsrb (int8/uint8 .tflite on the Hub) are the TFLite options found.
- FPGA: Vitis AI 2.5 `tsd_yolox_pt` xmodels for ZCU102/ZCU104/KV260, VCK190, VCK5000, U50LV/U55C
  (D9); LaErre9/Zynq_Ultrascale_Vitis_AI_CNN_ZCU102 (student project, LeNet/VGG16 GTSRB classifier
  deployed to ZCU102 DPU, GPL-3.0, 21 stars,
  https://raw.githubusercontent.com/LaErre9/Zynq_Ultrascale_Vitis_AI_CNN_ZCU102/master/README.md).
- Vendor sample applications checked and found empty for traffic signs (tree grep of shallow clones
  on 2026-10-31 for traffic sign / tsd / gtsrb / tt100k / road sign / speed limit):
  STMicroelectronics/stm32ai-modelzoo (HEAD 1423c78 2026-04-21; zoo folders are image
  classification, object detection, pose, segmentation and audio, no sign task),
  renesas-rz/rzv_ai_sdk (HEAD ce4aed7 2026-10-08; applications Q01 to Q13 and R01, none for signs),
  ARM-software/ML-zoo (HEAD fec0bb5 2023-03-31), quic/ai-hub-models (HEAD f4758c3 2026-10-31),
  TexasInstruments/edgeai-modelzoo (HEAD 99f1bd6 2026-08-12), openvinotoolkit/open_model_zoo (HEAD
  2c9b5c5 2026-10-15), nxp-imx/nxp-nnstreamer-examples (HEAD 37d3d86 2026-06-12). NXP's eIQ and
  Qualcomm's AI Hub web catalogues were not fetched (not verified).

## G. Leaderboards, surveys and benchmarks

- Live leaderboards: only the INI pages. GTSRB final results (A1) and GTSDB results (A2) have not
  changed since the 2010s; both pages state no code links. TT100K, MTSD, DFG and CCTSDB have no
  public leaderboard (TT100K site lists only the authors' results; the MTSD paper withholds test
  labels; DFG results are in the paper). Papers with Code is gone; the Hugging Face papers index
  has no task leaderboard for traffic signs (https://huggingface.co/api/papers/search?q=TT100K
  returns 0 and the "traffic sign detection" query returns papers without metrics).
- Surveys with results tables:
  - Lim, Lee, Lim, Ong, Alqahtani, Ali, "Recent Advances in Traffic Sign Recognition: Approaches
    and Datasets", Sensors 23(10) 4674, 2023, doi 10.3390/s23104674; reviews GTSRB (51,839
    images, 43 classes), BTSD (7,095+, 62) and TSRD (6,164, 58); quotes GTSRB accuracies up to
    99.99% (Li, Song and Wang 2022, colour histogram + HOG + PCA), 99.94% (Shustanov and Yakimov
    2017), 99.93% (Mamatkulovich 2022), 99.72% (DeepThin, Haque 2021); "Data sharing not
    applicable", no code links (https://www.mdpi.com/1424-8220/23/10/4674). Treat the 99.9x%
    entries as unverified self-reports.
  - Pavlitska, Lambing, Zoellner, "Adversarial Attacks on Traffic Sign Recognition: A Survey",
    arXiv 2307.08278 (2023): surveys digital and physical attacks on sign classifiers and detectors
    (https://arxiv.org/abs/2307.08278). Directly relevant to the maintainer's research.
  - Zhao, Qi, Zhang, Bi et al., "Road Infrastructure Perception for Autonomous Driving: Taxonomy,
    Representations, Tasks, and Challenges", arXiv 2610.33758 (2026-10-24): covers traffic sign
    recognition under "regulatory-semantic infrastructure" with benchmarks and evaluation
    (https://arxiv.org/abs/2610.33758). Not read in full.
  - Roman et al., "One-Stage Object Detectors in Autonomous Driving", arXiv 2608.19014
    (2026-08-19): survey of YOLO/SSD/RetinaNet/FCOS/CenterNet/YOLOv10 with reported benchmark
    performance including traffic signs (https://arxiv.org/abs/2608.19014). Not read in full.
  - "Review. Machine learning techniques for traffic sign detection", arXiv 1712.04391 (2017)
    (https://export.arxiv.org/api/query?search_query=all:%22traffic%20sign%22%20AND%20(all:survey%20OR%20all:review)).
  - Wali et al. 2019 (Sensors) and Saadna and Behloul 2017 are often cited but were not fetched
    (not verified).
- Benchmark papers with code relevant to protocol design: aarcosg 2018 (D1, 8 detectors, one
  protocol), Tabernik 2019 (D2, mAP50 and mAP50:95 with min 30 px), MTSD 2020 (D4, 313-class mAP
  with a separate detector and classifier), OPPO ACCV 2022 workshop benchmark "Evaluating and
  Bench-marking Object Detection Models for Traffic Sign and Traffic Light Datasets" (see the
  traffic light survey, section B5).

## H1. Summary table

Runnable today: easy = modern PyTorch/ONNX/Ultralytics on CPU; medium = TF1 frozen graphs via
compat, Keras 2 HDF5, or GPU-only but modern; hard = Torch7, Caffe, Caffe2 Detectron, MNN or
unavailable data. Stars and last commit from shields.io unless a clone HEAD date is given.

| Name | Year | Method | Framework | Dataset | Reported metric | Weights | License | Stars | Last commit | Runnable |
|---|---|---|---|---|---|---|---|---|---|---|
| IDSIA committee of CNNs (no official code) | 2011 | MCDNN | in-house GPU code | GTSRB | 99.46% | no | n/a | n/a | n/a | n/a |
| Sermanet multi-scale CNN (EBLearn) | 2011 | 2-stage ConvNet, multi-scale | EBLearn C++ | GTSRB | 98.31% final (99.17% in paper) | unknown | unknown | n/a | n/a | hard |
| INI LDA on HOG / CAOR random forests | 2011 | HOG + LDA / RF | none | GTSRB | 95.68% / 96.14% | no | n/a | n/a | n/a | n/a |
| Moodstocks/gtsrb.torch | 2015 | IDSIA-like CNN + 1 to 2 STN | Torch7 | GTSRB | 98.85 / 99.46 / 99.67% | no (training recipe) | MIT | 177 | May 2016 | hard |
| Haloi 2015 | 2015 | STN + inception | none | GTSRB | 99.81% | no | n/a | n/a | n/a | n/a |
| aarcosg/tsr-torch | 2018 | CNN + 3 STN | Torch7 + CUDA | GTSRB | 99.71% (leaderboard #1) | Google Drive zip | MIT | 41 | 2018-10-18 | hard |
| navoshta/traffic-signs | 2016 | 3 conv + FC, skip connections | TF 1.x | GTSRB (Udacity pickle) | 99.33% test | no | Apache-2.0 | 148 | 2019-12-23 | hard |
| kenshiro-o/CarND-... | 2017 | CNN 3x3 | TF 1.x | GTSRB (Udacity) | 97.75% test | no | none | 145 | 2017-08-26 | hard |
| jeremy-shannon/CarND-... | 2017 | Sermanet-style | TF 0.12 | GTSRB (Udacity) | val 0.987 | TF checkpoint in repo | none | 59 | 2017-01-16 | hard |
| vxy10/p2-TrafficSigns | 2017 | CNN | TF 1.x | GTSRB (Udacity) | 98.8% | no | none | 67 | 2017-02-09 | hard |
| mohamedameen93/German-Traffic-Sign-... | 2017 | LeNet / VGG-style | TF 0.12.1 | GTSRB (Udacity) | 97.6% test | no | MIT | 73 | Dec 2019 | hard |
| hello2all/GTSRB_Keras_STN | 2018 | CNN + STN | Keras 2 | GTSRB (Udacity) | 99.59% test | weights.hdf5 in repo (13.9 MB) | MIT | 51 | 2020-02-07 | medium |
| poojahira/gtsrb-pytorch | 2018 | CNN + STN, TTA | PyTorch | GTSRB (NYU Kaggle) | 99.809% private LB | model_40.pth in repo (3.4 MB) | none | 60 | 2026-02-19 | easy |
| surmenok/GTSRB | 2018 | ResNet-34 fine-tune | fastai | GTSRB | 99.22% on 20% val of train | no | MIT | 128 | Nov 2019 | medium |
| hellojialee/Traffic_Sign_Recognition_Efficient_CNNs | 2018 | efficient CNN + MobileNet FRCNN | Keras | GTSRB, GTSDB, TT100K | none in README | no | MIT | 154 | 2020-04-06 | hard |
| bazyl/gtsrb-model (HF) | 2022 | ViT-B/16 in21k | transformers | GTSRB (bazyl split) | 0.9993 eval | yes | apache-2.0 | n/a | 2023-05-05 | easy |
| kelvinandreas/vit-traffic-sign-GTSRB (HF) | 2024 | ViT-B/16 in21k | transformers | GTSRB | 0.9846 | yes | mit | n/a | 2025-01-02 | easy |
| aarcosg/traffic-sign-detection | 2018 | Faster R-CNN, R-FCN, SSD, YOLOv2 | TF1 OD API, darkflow | GTSDB (3 superclasses) | mAP 61.64 to 95.77 | Google Drive (8 zips) | MIT | 335 | 2020-02-13 | medium |
| moabitcoin/signfeld | 2020 | RetinaNet R50, synthetic training | Detectron2 | GTSDB (169 DE classes) | GTSDB mAP 67.23 | GitHub release (318 MB) | MIT | 13 | May 2020 | medium |
| skokec/detectron-traffic-signs | 2019 | Mask R-CNN + OHEM | Caffe2 Detectron | DFG (200 classes) | mAP50 95.5, mAP50:95 84.4 | data.vicos.si (644 MB) | Apache-2.0 | 38 | 2026-04-21 | hard |
| TT100K official (Zhu et al.) | 2016 | multi-task FCN + classifier | Caffe fork | TT100K (45 classes) | acc 0.877 / recall 0.906 at IoU 0.5 | code.zip (467 MB caffemodel) | CC BY-NC (dataset); code license unknown | n/a | 2016 | hard |
| Vitis AI 2.5 pt_yolox_TT100K | 2022 | YOLOX-m modified | PyTorch (YOLOX fork) + Vitis AI | TT100K val | AP 62.3 float / 62.1 INT8 | AMD zip (594 MB) | Apache-2.0 | 1.8k (Vitis-AI) | model removed after 2.5 | medium (GPU) |
| MTSD baseline (Mapillary) | 2020 | Faster R-CNN FPN + classifier | unknown | MTSD (313 classes) | 83.4 mAP | no | n/a (dataset research-only) | n/a | n/a | n/a |
| Genius-Society/yolov4_tt100k (HF/ModelScope) | 2024 | YOLOv4 + attention | unknown | CCTSDB2021 | mAP 92.77 (self-reported) | ModelScope only | MIT | n/a | 2025-10-30 | unknown |
| HaoqinHong/CCSPNet-Joint | 2024 | YOLOv5l + EfficientViT + denoiser | PyTorch | CCTSDB-AUG | in paper | Kaggle / Baidu | GPL-3.0 | 28 | Jan 2026 | medium |
| cvtechniques/TrafficSignDetection (HF) | 2026 | YOLO11n | Ultralytics | LISA TS | mAP50 0.994 (val) | best.pt | mit (card) | n/a | 2026-03-16 | easy |
| neitpm/traffic-sign-yolo11-benchmark (HF) | 2026 | YOLO11n + UIB variants | Ultralytics | custom (5,668/1,259/1,151) | mAP50-95 70.27 | 3 .pt | mit | n/a | 2026-10-16 | easy |
| star092304/traffic-sign-detection-vietnam-yolo (HF) | 2026 | YOLO | Ultralytics | Vietnam (HF dataset) | mAP50 0.9806 | best.pt/.onnx | apache-2.0 | n/a | 2026-06-08 | easy |
| Panoramax/detect_face_plate_sign (HF) | 2024 | YOLO11l @2048 | Ultralytics | Panoramax imagery | sign mAP50 0.898 (val 329 imgs) | yolo11l_panoramax.pt | etalab-2.0 | n/a | 2024-11-16 | easy |
| Panoramax/classify_de_road_signs (HF) | 2026 | YOLO26-cls | Ultralytics | Panoramax DE crops (130+ classes) | accuracy 0.986 | .pt (26 MB) | etalab-2.0 | n/a | 2026-06-13 | easy |
| infra-intelligence/road-sign-yolox-tiny (HF) | 2026 | YOLOX-Tiny from scratch | PyTorch / ONNX | Mapillary Japan (private labels) | AP50 0.902, recall 0.857 (test) | .pth + .onnx (20 MB) | Apache-2.0 | n/a | 2026-10-22 | easy |
| FrogPilot speed_limit_reader | 2026 | unknown (MNN graph) | MNN | unknown | none published | .mnn in repo (12.6 MB) | MIT (repo) | 568 | 2026-10-31 | hard |
| hoanglehaithanh/Traffic-Sign-Detection | 2017 | colour + shape + SVM | OpenCV 3 | self-made 12 classes | none | data_svm.dat | MIT | 292 | Dec 2018 | medium |

## H2. Replication candidates (ranked)

Constraints: CPU-only cloud session with 2 cores and 7 GB RAM, Hugging Face and GitHub release
downloads reachable, sid.erda.dk not reachable (so no official GTSRB/GTSDB archives and no
torchvision download). The number to compare against is given for each.

### CPU-only session

1. poojahira/gtsrb-pytorch (STN + CNN) on the official 12,630-image test set.
   Why: weights in the repo (3.4 MB), plain PyTorch, labels available through tanganke/gtsrb.
   Target: 99.809% (Kaggle private) / 99.746% (public); the model was trained on train+val, so
   expect a value in the 99.7 to 99.8 range with test-time augmentation and a little less without.
   ```
   python3 -m venv ~/venv-ts && . ~/venv-ts/bin/activate
   pip install torch torchvision datasets pillow numpy
   git clone https://github.com/poojahira/gtsrb-pytorch.git ~/gtsrb-pytorch
   python - <<'EOF'
   from datasets import load_dataset
   ds = load_dataset("tanganke/gtsrb", split="test")     # 12,630 images with labels
   ds.save_to_disk("~/gtsrb_test")
   EOF
   # write a 30-line evaluate script: import model.py (Net), load model/model_40.pth with
   # torch.load(map_location="cpu"), apply the same transforms as data.py (resize 32x32, normalize,
   # optionally the TTA transforms), argmax, compare with the label column; report accuracy.
   ```
   Check the class index order before trusting the number: whether tanganke/gtsrb's label column
   uses the GTSRB class ids 0 to 42 in the original order, and whether the repo's training folders
   (`data/train_images/000XX` in the NYU homework layout) map to the same ids, was not verified in
   this session; compare a few predictions on known classes (speed limit 30 = class 1) first.

2. Hugging Face ViT classifiers on the same test set (kelvinandreas/vit-traffic-sign-GTSRB, then
   bazyl/gtsrb-model and cvetanovskaa/vit-base-patch16-224-in21k-gtsrb-tuned).
   Why: `transformers` pipeline, no code to port. Cost: ViT-B/16 at 224 px on 2 cores is roughly
   0.2 to 0.4 s per image, about 1 to 1.5 hours for 12,630 images; run on a 2,000-image random
   subset first. Targets: 0.9846 (kelvinandreas, split unstated), 0.9993 (bazyl, own evaluation
   split; expect lower on the official test set).
   ```
   pip install transformers datasets torch pillow
   python - <<'EOF'
   from transformers import pipeline; from datasets import load_dataset
   ds = load_dataset("tanganke/gtsrb", split="test")
   clf = pipeline("image-classification", model="kelvinandreas/vit-traffic-sign-GTSRB", device=-1)
   # map id2label of the model to GTSRB ids (check config.json), loop, count argmax == label
   EOF
   ```

3. Panoramax/classify_de_road_signs (YOLO26-cls) on the Panoramax/classified_de_road_signs val split.
   Why: weights (26 MB) and the exact validation zip are both on the Hub; Ultralytics `val` gives
   top-1 directly; the label space (130+ German sign types) is close to what OpenTSR needs for
   Europe. Target: accuracy 0.986 (model-index). Licenses: Etalab-2.0 model, CC BY-SA 4.0 data.
   ```
   pip install ultralytics huggingface_hub
   hf download Panoramax/classify_de_road_signs classify_de_road_signs.pt --local-dir ~/pano_de
   hf download Panoramax/classified_de_road_signs --repo-type dataset --local-dir ~/pano_de_data
   cd ~/pano_de_data && unzip -q train.zip && unzip -q val.zip   # check folder names, expect train/ and val/
   yolo classify val model=~/pano_de/classify_de_road_signs.pt data=~/pano_de_data device=cpu workers=2
   # compare metrics/accuracy_top1 with 0.986
   ```

4. hello2all/GTSRB_Keras_STN with the Udacity pickled test set.
   Why: weights in repo (13.9 MB), test.p is the 12,630-image set at 32x32, target 99.59%.
   Cost: minutes on CPU. Risk: Keras 2 HDF5 with the custom `SpatialTransformer` layer; use
   `pip install "tensorflow<2.16" "keras<3"` or `tf_keras` and load with `custom_objects`.
   ```
   git clone https://github.com/hello2all/GTSRB_Keras_STN.git ~/stn_keras
   curl -L -o ~/traffic-signs-data.zip https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
   unzip -q ~/traffic-signs-data.zip -d ~/udacity_gtsrb          # train.p valid.p test.p
   pip install "tensorflow-cpu<2.16" "keras<3" numpy
   # adapt test.py: pickle.load test.p, build conv_model.py's model, model.load_weights("output/weights.hdf5"), evaluate
   ```

5. infra-intelligence/road-sign-yolox-tiny on Japanese dashcam frames (qualitative, plus a self-made
   mini test set). Why: the only open Japanese sign detector; ONNX Runtime CPU; 20 MB. No public
   number can be reproduced (evaluation images are private), so the deliverable is a run on a few
   hundred Japanese frames (for example Mapillary images of Saitama) with manual inspection, and
   later AP50 on OpenTSR's own labelled Japanese frames against the author's 0.902 (test) / 0.925 (val).
   ```
   pip install onnxruntime numpy pillow huggingface_hub
   hf download infra-intelligence/road-sign-yolox-tiny --local-dir ~/jp_yolox
   python ~/jp_yolox/examples/predict_onnx.py ~/jp_yolox/road_sign_yolox_tiny_v9.onnx frame.jpg
   # 16:9 frames are letterboxed to 1280x736; threshold 0.5; class order guide, warning, regulatory, indication, supplementary, hard_negative
   ```

6. aarcosg/traffic-sign-detection, SSD MobileNet v1 and Faster R-CNN ResNet 101 on the GTSDB test
   TFRecords. Why: the repo's Google Drive TFRecords replace the unreachable GTSDB archive; targets
   mAP 61.64 (SSD MobileNet) and 95.08 (FRCNN R101) at IoU 0.5 over 3 superclasses. Cost: 300 test
   images; SSD is seconds, Faster R-CNN R101 at 1360x800 is about 5 to 10 s per image on 2 cores.
   Risk: TF1 frozen graphs need `tensorflow` 2.x with `tf.compat.v1.GraphDef` import and a
   reimplementation of the notebook's VOC AP code (short).
   ```
   pip install gdown tensorflow-cpu numpy
   gdown 1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq -O ~/gtsdb_tfrecords.zip      # TFRecords
   gdown 1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg -O ~/ssd_mobilenet_v1.zip     # model zip with frozen graph and detections
   # the zips also contain the authors' detection result files, so the mAP can be recomputed without inference first
   ```

7. star092304/traffic-sign-detection-vietnam-yolo. Why: tiny dataset (n<1K) and weights on the Hub,
   `yolo val` protocol, target mAP50 0.9806 / mAP50-95 0.8357. Off-geography but a 10-minute
   sanity check of the Ultralytics path on CPU.

8. cvtechniques/TrafficSignDetection (YOLO11n, LISA traffic signs). Target mAP50 0.994 on its val
   split; needs the LISA TS dataset from UCSD and the author's split, which were not located (not
   verified), so rank last.

### GPU machine

1. Vitis AI 2.5 `pt_yolox_TT100K_640_640_73G_2.5` on TT100K.
   Why: public 594 MB zip with the float checkpoint, the YOLOX fork and eval script; target AP 62.3
   on "TT100k val" (the exp file `exps/example/custom/yolox_m_tt100k_float.py` defines the split;
   check whether it is the 2016 test split or a custom split; the split size was not verified
   here). Data: TT100K `data.zip` (19.2 GB) from
   cg.cs.tsinghua.edu.cn (direct link works) or the Hub mirrors finedet/tt100k and
   Genius-Society/tt100k (CC BY-NC 4.0). Steps:
   ```
   curl -L -o pt_yolox_tt100k.zip https://download.amd.com/opendownload/xlnx/pt_yolox_TT100K_640_640_73G_2.5.zip && unzip -q pt_yolox_tt100k.zip
   cd pt_yolox_TT100K_640_640_73G_2.5/code && pip install -r ../requirements.txt && pip install -v -e .
   curl -L -o data.zip https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip && unzip -q data.zip -d datasets/   # 19 GB
   bash datasets/convert_label.sh
   python tools/eval.py -f exps/example/custom/yolox_m_tt100k_float.py -c ../float/best_ckpt.pth -b 32 -d 1 --conf 0.001
   # compare AP with 62.3
   ```
   Note the CC BY-NC dataset license: fine for a benchmark, not for an OpenTSR release trained on it.

2. skokec/detectron-traffic-signs on DFG. Target mAP50 95.5 / mAP50:95 84.4 (Mask R-CNN R50 with
   adaptations and augmentation; README warns results may differ after a dataset update). Needs a
   Caffe2 build with the custom op copied into `modules/detectron`, CUDA, and the DFG dataset from
   https://www.vicos.si/Downloads/DFGTSD (301 redirect to the download page; terms not checked).
   Weights: https://data.vicos.si/skokec/villard/dfg_mask_rcnn_weights.tar.gz (644 MB). Effort:
   days; a Detectron2 re-implementation of the three adaptations is probably cheaper than
   building Caffe2.

3. Official TT100K Caffe model. Target: precision 0.877 / recall 0.906 at IoU 0.5 on the 45 classes
   with at least 100 instances (code/results/report_ours.txt). Steps: download code.zip (801 MB),
   build `code/caffe` (BVLC Caffe fork, Python 2 era), run `code/python/deployer.ipynb` and
   `eval_check.ipynb` with `model/model.caffemodel`. The `results/ours_result.json` file allows the
   evaluation code to be checked before any inference.

4. moabitcoin/signfeld RetinaNet on GTSDB. Target GTSDB mAP 67.23; Detectron2 on GPU is
   straightforward; weights from GitHub releases (318 MB); GTSDB images needed (use the Roboflow
   re-export keremberke/german-traffic-sign-detection only for smoke tests, because its split and
   labels differ from the 300-image official test set).

5. aarcosg/tsr-torch (99.71%) and Moodstocks/gtsrb.torch (99.67%): only inside a Torch7 Docker
   image with CUDA 8 to 10 era drivers; the pretrained zip is on Google Drive. Low priority, but
   this is the only path to the leaderboard-topping number.

6. Training from scratch to reproduce published recipes (navoshta 99.33%, hello2all 99.59%,
   poojahira 99.8%) on one GPU takes hours each and gives OpenTSR a modern PyTorch baseline with a
   clean protocol (train 39,209 / test 12,630). Suggested first: a torchvision ResNet-18 or
   ConvNeXt-T fine-tune at 64x64 with the poojahira augmentation list, evaluated with the script
   from CPU candidate 1.

## Pages that could not be fetched

- https://benchmark.ini.rub.de/gtsrb_dataset.html and the sid.erda.dk archive directory
  (https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/): host not reachable from
  the sandbox; GTSRB/GTSDB file lists and the HOG feature files are therefore unverified.
- https://github.com/topics/traffic-sign-detection?o=desc&s=stars (sorted variant) and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10223536/ : WebFetch permission not granted in time;
  the unsorted topic page and the MDPI page were used instead.
- https://doaj.org/article/b286c95037b941c6acfbc56c05543989 and three other DOAJ records (recent
  survey and YOLO papers): Cloudflare challenge page returned.
- https://universe.roboflow.com/* via curl: HTTP 403; one page was read through WebFetch.
- Kaggle notebook and dataset pages (www.kaggle.com/code, www.kaggle.com/datasets): not surfaced by
  web search and not fetched; Kaggle API requires credentials.
- Google Drive weight files of aarcosg/traffic-sign-detection, aarcosg/tsr-torch and
  MDhamani (mega.nz): link existence taken from READMEs; the files were not downloaded.
- https://www.vicos.si/Downloads/DFGTSD/ (DFG terms page): only the 301 redirect was observed.
- https://www.mapillary.com/dataset/trafficsign (MTSD terms): not fetched.
- NXP eIQ, Qualcomm AI Hub and NVIDIA NGC web catalogues: not fetched; only GitHub trees were
  checked.
- Papers with Code: offline; no citations used.
