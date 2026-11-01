# Replication of open traffic sign recognition implementations

What we re-ran ourselves, on which data, with which protocol, and what came out. All runs in
this file were done on 2026-10-31 in a claude.ai cloud session (2 CPU cores of an Intel Xeon with
AVX-512 and AMX, 7 GB RAM, no GPU, Python 3.13, PyTorch 2.15.0+cpu, transformers 5.22.1,
open_clip 3.3.0, ultralytics 8.4.194 installed only for YOLO-World; pins in
`requirements.txt`). Every number maps to a script under `scripts/`, a config under
`configs/gtsrb/` where training is involved, and a JSON file under `runs/` (gitignored;
regenerate with the `make` targets listed in section 10). See
`docs/oss_implementations.md` for why these implementations were chosen,
`docs/vlm_and_foundation.md` for the foundation-model reference numbers and
`docs/datasets.md` for the data.

Status table:

| # | What | Data | Status | Headline result |
|---|---|---|---|---|
| 2 | Released poojahira/gtsrb-pytorch checkpoint `model_40.pth` | GTSRB official test (12,630) | done | 99.71 % vs 99.809 % claimed |
| 3 | From-scratch CNNs: Udacity LeNet-5 (2 variants), poojahira STN-CNN architecture at reduced budget | GTSRB official split | done | 91.84 / 89.14 / 99.13 % |
| 4 | Hub checkpoints scored on the official test split: ConvNeXt V2 atto (Fraunhofer IOSB), ViT-B/16 (bazyl) | GTSRB official test | done | 99.81 % (ConvNeXt V2 atto, card 99.8); 97.00 % (ViT-B/16, card 99.93 on its own validation split; 200 of its 379 errors are left/right mirror confusions) |
| 5 | CLIP and SigLIP zero-shot with the OpenAI prompts | GTSRB official test | done (B/32), queued (B/16, SigLIP, L/14) | 32.56 % (ViT-B/32, paper 32.2, open_clip 32.6); 26.57 % with the open_clip GELU mismatch pitfall |
| 6 | Small VLMs in 43-way multiple choice | GTSRB test subset (stratified) | done | 2.33 % (SmolVLM-500M, 215 images), 23.26 % (Qwen2-VL-2B, 215 images) |
| 7 | Open-vocabulary detectors, class-agnostic | GTSDB training package (official gt.txt) | done | OWLv2 AP50 77.1 on 150 frames, YOLO-World-s AP50 54.2 |
| 8 | GTSRB-trained classifiers on GTSDB crops (cross-capture check) | GTSDB 852 crops | done | STN-CNN 99.53 %, ConvNeXt V2 atto 99.53 %, LeNet 98.36 % |
| 9 | Panoramax German sign classifier (134 classes) on its own validation crops | Panoramax/classified_de_road_signs val (6,944) | done | 98.68 % vs 0.986 claimed |
| 10 | Detector training on GTSDB / ZOD / TT100K, CLIP linear probe, synthetic-to-real, VLM fine-tuning | GPU | prepared, not run | section 10 |

Reading guide (numbers are section numbers): 2 and 9 are reproductions of a published number
from released weights; 4 and 5 check Hub and paper claims on one fixed protocol; 3 shows what
the classic recipes give on the official split when trained here; 6, 7 and 8 are measurements
that no paper reports in this form.

---

## 1. Data and protocol shared by the GTSRB experiments

- Data: `bazyl/GTSRB` on Hugging Face, revision `a8093a9c7757b59d64702f892002542e8f3a1fb0`
  (parquet export of the Kaggle release of GTSRB, PNG conversion of the official PPM files).
  `scripts/prepare_gtsrb.py` pins the revision, checks the sha256 of both parquet files, and
  verifies 39,209 training and 12,630 test images with class ids 0 to 42 before anything else
  runs (`data/gtsrb_hf/VERIFIED.json`, `problems: []`). The official host sid.erda.dk is not
  reachable from the cloud session; the mirror's `gpl-3.0` tag is the uploader's, not the
  benchmark's ("The data is free to use", https://benchmark.ini.rub.de/gtsrb_dataset.html).
- Why not `tanganke/gtsrb`: its training split has 26,640 images, the IJCNN 2011 Phase I subset
  (`docs/datasets.md` section 1.1), so models trained on it are not comparable to
  the official 39,209-image protocol. Its test split is the same 12,630 images.
- Metric: top-1 accuracy over all 12,630 test images (the only metric the official leaderboard
  used), plus per-class recall and the full confusion matrix in each JSON. No model selection
  on the test set: for trained models the weights after the last epoch are scored once.
- Images are fed as the full GTSRB image (sign plus its roughly 10 percent border), which is
  what every recipe compared against does; `--use-roi` exists in the scripts for ablations.
- Class names: the 43 descriptions of the OpenAI CLIP repository (`data/prompts.md`) for the
  text-based models, and the usual short names (Speed limit (20km/h), ..., End of no passing by
  vehicles over 3.5 metric tons) for tables and VLM prompts; both in `src/opentsr/signs/datasets/gtsrb.py`.

## 2. Released poojahira/gtsrb-pytorch checkpoint on the official test split (done)

- Source: https://github.com/poojahira/gtsrb-pytorch, file `model/model_40.pth` (3,430,539
  bytes, sha256 `157996e0f6baa5b5f0803dae03a74c61c31723c8768f7cdd0e6535df14bbae38`), README
  claim 99.809 percent on the private and 99.746 percent on the public leaderboard of the NYU CV
  Fall 2018 Kaggle in-class competition (whose test images are the official GTSRB test set with
  hidden labels), obtained with random test-time augmentation that the README says moves the
  score by about 0.1 point between runs. Training: 40 epochs, Adam 1e-4, batch 64, a 10x
  offline-augmented training set (392,090 images, including horizontal and vertical flips),
  trained on training plus validation images.
- What we did: re-implemented the network from the README and `model.py` in
  `src/opentsr/signs/models/gtsrb_cnn.py` (STN with conv7x7(8), conv5x5(10), FC32; CNN conv5x5(100),
  conv3x3(150), conv3x3(250), FC350, FC43; BatchNorm, Dropout2d, leaky ReLU), loaded the
  state dict with the default `weights_only=True`, resized every test image to 32x32 and applied
  the repository's normalisation constants, no test-time augmentation. The checkpoint predates
  PyTorch 1.3, whose `affine_grid` / `grid_sample` default changed from `align_corners=True` to
  `False`; both settings were scored. Script: `scripts/eval_poojahira_weights_gtsrb.py`.
- Result: **99.71 percent** (36 errors of 12,630) with `align_corners=True`, 99.71 percent
  (37 errors) with `False`. The README's 99.809 percent is reproduced within 0.1 point without
  its test-time augmentation; the remaining gap is the size the README itself attributes to TTA
  randomness. This is the best public PyTorch GTSRB checkpoint we found, and it beats the
  official leaderboard's 99.71 percent (Arcos-Garcia, Torch7) on the same images.
- Per-class recall below 97 percent: Beware of ice/snow 96.7.

## 3. From-scratch CNNs on the official split (done)

Configs under `configs/gtsrb/`, script `scripts/train_gtsrb_classifier.py`; data cached as
32x32 uint8 arrays, per-sample online augmentation on the batch (random affine: rotation 15
degrees, translation 10 percent, scale 0.9 to 1.1, shear 2 degrees; brightness and contrast
jitter 40 percent) where the config enables it, Adam with cosine decay, seed 0, last-epoch
weights scored once.

| Config | Network (parameters) | Epochs, batch, lr | Augmentation | Test top-1 | Errors | Wall clock (2 cores) |
|---|---|---|---|---|---|---|
| `lenet5_gray_32_noaug` | LeNet-5, grayscale 32x32 (64,511) | 30, 128, 1e-3 cosine | none | **91.84 %** | 1030 | 3.4 min |
| `lenet5_gray_32` | same | same | affine + photometric | **89.14 %** | 1371 | 3.5 min |
| `stn_cnn_poojahira_32` | poojahira STN-CNN, RGB 32x32 (855,487) | 20, 64, 1e-3 cosine | affine + photometric | **99.13 %** | 110 | 44 min |

- LeNet-5 lands where the Udacity-era repositories land (91 to 97 percent, see
  `docs/oss_implementations.md` section B). The augmented variant is lower than the
  plain one: with 64k parameters and grayscale input the network underfits under 40 percent
  brightness jitter (the worst classes are the dark, rare warning signs: Pedestrians
  0.0 percent recall, Beware of ice/snow 32.0 percent). The classic recipes fixed this
  with contrast normalisation (Sermanet's YUV + local contrast normalisation, Udacity's
  histogram equalisation), which was deliberately not added here so that the number stays the
  plain-recipe baseline.
- The STN-CNN reaches 99.13 percent after 20 epochs on the unexpanded training set, 0.59 points
  below the released checkpoint of the same architecture (section 2) that used 10x offline
  augmentation, flips, 40 epochs and TTA. Epoch curve: epoch 1 93.70, epoch 2 96.89, epoch 5 98.19, epoch 10 98.54, epoch 15 99.02, epoch 20 99.13. The run took 44 minutes on
  2 cores, so a faithful 40-epoch run on the 10x set is a GPU task (section 10).

## 4. Hub checkpoints on the official test split (done)

Script `scripts/eval_hf_classifier_gtsrb.py`: each checkpoint's own image processor, output
index i taken as GTSRB class i (both checkpoints' `id2label` follow the official order; printed
in the JSON).

| Checkpoint | Base model, training data as stated | Card claim | Official test top-1 | Errors | CPU time |
|---|---|---|---|---|---|
| `FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb` (CC BY 4.0) | ConvNeXt V2 atto, ImageNet-1k, fine-tuned on `tanganke/gtsrb` (26,640 train) with the first two stages frozen | "99.8% on the GTSRB validation set" | **99.81 %** | 24 | 4 min |
| `bazyl/gtsrb-model` (Apache-2.0) | ViT-B/16 ImageNet-21k, fine-tuned 10 epochs on `bazyl/GTSRB` (official train) | 0.9993 "on the evaluation set" | **97.00 %** | 379 | 45 min |

- The Fraunhofer IOSB ConvNeXt V2 atto (3.7 M parameters, ImageNet-1k pretraining, fine-tuned on the 26,640-image mirror split with the first two stages frozen) reproduces its card's 99.8 percent on the official test images and beats the released poojahira checkpoint (99.71 percent) and the frozen 2018 leaderboard (99.71 percent) with fewer errors than any model trained in this session. Its weakest class is Beware of ice/snow (97.3 percent recall). Pretraining plus 224 px input is worth more on GTSRB than any of the classic 32 px recipes. The bazyl ViT-B/16 fine-tune (86 M parameters, ImageNet-21k pretraining) scores 97.00 percent on the official test, 2.9 points below its card; the card's 99.93 percent was measured on a held-out part of the training set (its log shows 4,166 steps of batch 8 per epoch, i.e. about 33,300 of the 39,209 training images, so the evaluation set is the remaining 15 percent, which shares its tracks and physical signs with the training images). Its confusion matrix is dominated by mirror pairs: Turn right ahead read as Turn left ahead (70 images), Keep left as Keep right (48), Dangerous curve to the left as the right curve (29), Go straight or right and Go straight or left swapped (27 and 26). These 200 of the 379 errors are consistent with a random horizontal flip in the training augmentation (the card is auto-generated and does not list the transforms; the Hugging Face `run_image_classification.py` example flips by default), which turns left-pointing signs into right-pointing ones; without them the checkpoint would sit at about 98.6 percent. The remaining errors are the usual tiny speed-limit digits (120 read as 80 or 50, 60 as 50). So the gap to the CNN checkpoints is a recipe effect (flip augmentation, no ROI crop, 10 epochs), not an architecture effect: the fine-tuned CLIP ViT numbers in `vlm_and_foundation.md` (98.5 to 99.3 percent) show what a ViT reaches with a sign-aware recipe.

## 5. CLIP and SigLIP zero-shot with the OpenAI prompts (done)

Script `scripts/eval_clip_zeroshot_gtsrb.py`, protocol of the CLIP paper: the 43 class
descriptions and the 3 templates from `openai/CLIP/data/prompts.md`, text embeddings averaged
over the templates, cosine argmax, top-1 on the 12,630 test images, each image resized and
centre-cropped by the model's own open_clip transform. Reference values: CLIP paper Table 11
(https://arxiv.org/pdf/2103.00020) and the open_clip results CSV
(https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv),
both collected in `docs/vlm_and_foundation.md` section A.

| open_clip model | pretrained | Measured top-1 | CLIP paper | open_clip CSV | CPU time |
|---|---|---|---|---|---|
| ViT-B-32 | openai | **32.56 %** | 32.2 | 32.6 | 10 min |
| ViT-B-16 | openai | not run in the session (about 50 min on 2 cores; `make replicate-gtsrb-clip`) | 43.3 | 43.4 | |
| ViT-B-16-SigLIP | webli | not run in the session (about 50 min; same target) | n/a | 51.6 | |
| ViT-L-14 | openai | not run in the session (about 1.5 h on 2 cores; `make replicate-gtsrb-clip`) | 50.3 | 50.6 | |

- ViT-B/32 reproduces both reference numbers once the architecture matches the weights. The first run of this session scored 26.57 percent because open_clip 3.3.0 instantiates `ViT-B-32` with a standard GELU when asked for `pretrained="openai"` and only emits a `QuickGELU mismatch` warning; the OpenAI checkpoints were trained with QuickGELU, so the correct model name is `ViT-B-32-quickgelu` (the script now appends the suffix automatically, `--keep-gelu-mismatch` reproduces the pitfall, JSON kept as `clip_zeroshot_ViT-B-32_openai_gelu_mismatch.json`). Preprocessing is not the cause: on a 1,500-image subset the open_clip transform, squashing to 224x224, cropping to the official ROI and bilinear resizing all land between 26.1 and 27.0 percent with the mismatched model, and the CLIP_benchmark `wds_gtsrb` test shards are pixel-identical to the mirror (lossless WebP), so the 6-point gap is entirely the activation function. Zero-shot CLIP recognises shape-and-colour classes well (No entry 90.8 percent, Roundabout 90.0, Stop 88.9) and fails on classes whose name differs only by a number or a pictogram detail (every curve, no-passing and end-of-limit class is at 0 percent).
- Speed limit confusion (the failure mode the literature describes but never quantifies): with
  ViT-B/32, the 8 numeric speed-limit classes (20 to 120 km/h, 4,170 test images) are recognised as
  *some* speed limit 68.3 percent of the time but as the *correct* value only 44.4 percent of the
  time; the per-class recalls are (20km/h) 65.0, (30km/h) 10.0, (50km/h) 76.0, (60km/h) 4.0, (70km/h) 61.7, (80km/h) 27.3, (100km/h) 64.7, (120km/h) 62.4. The text side of the classifier cannot
  separate the digits, as `vlm_and_foundation.md` section A.5 predicts.

## 6. Small VLMs in 43-way multiple choice (done)

Script `scripts/eval_vlm_gtsrb_mc.py`: stratified subset of the official test split (5 images
per class, seed 0, 215 images; the same subset for both models), each crop upscaled to 224
px, one user turn with the image, the numbered list of the 43 short class names and "Answer
with the number of the correct option only", greedy decoding, first integer in the reply taken
as the prediction. This is the closed-vocabulary zero-shot setting of the LMM traffic sign
papers (`vlm_and_foundation.md` section B.1 reports LLaVA-1.5 0.32, LLaVA-NeXT 0.39, CLIP
ViT-B/32 0.24 and GPT-4o 0.89 on a GTSRB subset under a similar protocol).

| Model (license) | Parameters | Subset | Top-1 | Unparsable replies | Seconds per image (2 cores) |
|---|---|---|---|---|---|
| `HuggingFaceTB/SmolVLM-500M-Instruct` (Apache-2.0), single 512 px tile | 0.5 B | 215 | **2.33 %** | 0 | 3.7 |
| `Qwen/Qwen2-VL-2B-Instruct` (Apache-2.0), bf16 | 2.2 B | 215 | **23.26 %** | 0 | 5.3 |

- SmolVLM-500M answers "10" or "9" for almost every image (212 of 215 replies) and lands at chance level (1 of 43 = 2.3 percent); a 0.5 B model cannot use a 43-option list at all, so its number says nothing about sign recognition. Qwen2-VL-2B does read the image: it is right on 7 of the 8 speed-limit classes at 60 to 100 percent (mean 70 percent over the speed classes, 100 percent on 30 km/h and Stop), which is exactly what CLIP cannot do, but it collapses to option 1 (Speed limit 30) for 147 of the 215 replies whenever the sign is not a number, so pictogram classes score near zero and the overall top-1 stays below CLIP ViT-B/32 zero-shot. The CdMT paper's LLaVA-1.5 (0.32) and GPT-4o (0.89) numbers are on a different subset and prompt, so only the ordering is comparable. Qwen2-VL-2B runs at 2.7 s per image on this CPU alone (5.3 s contended) thanks to AMX bf16, so the 7 B class (Qwen2.5-VL-7B, InternVL 3) and option-order shuffling are GPU follow-ups listed in section 10. No open VLM measured here is within 60 points of the supervised classifiers, which matches the survey's conclusion that VLMs are labelling and interpretation aids, not the classifier; their digit reading is the part worth keeping.

## 7. Open-vocabulary detectors on GTSDB, class-agnostic (done)

- Data: GTSDB training package from the Hugging Face mirror `GlowBond/GTSDB` (revision
  `0d1c1a77fce322ba23b57a5e3261fae934f5afe3`, `archive.zip` sha256
  `6144b38827acf3989bd7cd000a424c52cfd46a49da07e586439e25cf52995547`): 600 frames of 1360x800,
  official `gt.txt` with 852 boxes in 506 frames (94 frames without signs), sign side 16 to 128
  px (median 38; 297 boxes under 32 px, 420 between 32 and 64, 135 above). The 300 test frames
  ship without ground truth; the official FullIJCNN2013 archive is needed for them (section 10).
  Category counts: prohibitory 396, danger 156, mandatory 114, other 186.
- Protocol (`scripts/eval_openvocab_gtsdb.py`): prompts "traffic sign", "road sign", "speed
  limit sign", "round red and white prohibition sign", "triangular warning sign", "blue round
  mandatory sign", "stop sign", "yield sign"; all detections relabelled to one class "sign";
  PASCAL VOC AP at IoU 0.5 and 0.3 with all-point interpolation (`src/opentsr/metrics.py`) over
  all detections with score at least 0.05; recall, precision, per-category and per-size recall
  and false positives on the negative frames (94 in the package, 3 in the 150-frame subset used
  here) at an operating threshold of 0.3. Zero-shot
  models never saw GTSDB, so evaluating on the training frames is legitimate, but the numbers
  are not the IJCNN 2013 competition protocol (which scored per category with a 0.6 IoU on the
  300 test frames).

| Detector (license) | Input | Frames | AP50 | AP30 | Recall at 0.3 | Precision at 0.3 | Recall by category (proh / mand / dang / other) | Recall by size (<32 / 32-64 / >=64 px) | FP on negative frames | Seconds per frame |
|---|---|---|---|---|---|---|---|---|---|---|
| OWLv2 base patch16 ensemble (Apache-2.0) | 960x960 padded | 150 (238 boxes) | **77.1** | 77.2 | 95.4 | 39.0 | 99 / 90 / 98 / 92 | 90 / 98 / 98 | 3 / 3 | 6.6 |
| YOLO-World v2 s (Ultralytics, AGPL-3.0) | 1280 | 150 (238 boxes) | **54.2** | 54.5 | 64.7 | 57.5 | 87 / 88 / 45 / 33 | 70 / 53 / 82 | 2 / 3 | 1.6 |

- OWLv2 localises nearly every sign (max recall 99.6 percent at IoU 0.5 over all detections, 95.4 percent at the 0.3 operating threshold, still 90.5 percent for signs under 32 px) but at low precision (39.0 percent at threshold 0.3, 618 boxes for 238 signs; it also fires on the backs of signs, posts and shop signs), hence AP50 of 77.1. AP at IoU 0.3 is the same as at 0.5, so localisation is tight when it fires. Mandatory (blue) signs are the weakest category (90.2 percent recall) with the sign-oriented prompts used. Both runs cover the first 150 training frames (00000 to 00149: 238 boxes, 3 negative frames) for time reasons; `make replicate-gtsdb-openvocab` runs all 600. YOLO-World v2 s at 1280 px is 4 times faster but misses most danger and other signs with these prompts (recall 44.7 and 32.8 percent) and ends at AP50 54.2; Ultralytics also refuses PPM paths, so the script hands it PIL images. Practical reading for OpenTSR: an Apache-2.0 open-vocabulary detector is good enough to propose candidate boxes for auto-labelling if a sign classifier filters them, but not as a detector on its own.

## 8. GTSRB-trained classifiers on GTSDB crops (done)

Script `scripts/eval_gtsrb_models_on_gtsdb_crops.py`: the 852 ground-truth boxes of the GTSDB
training frames cut with a 10 percent margin (mimicking the GTSRB border), classified 43-way by
models that only saw GTSRB. Caveat: GTSDB was recorded by the same group with the same camera
set-up as GTSRB (Houben et al., IJCNN 2013), so this is a different capture of the same domain,
not a cross-country test.

| Model | GTSRB test top-1 | GTSDB crops top-1 (852) | By category (proh / mand / dang / other) | By size (<32 / 32-64 / >=64 px) |
|---|---|---|---|---|
| LeNet-5 no-aug (section 3) | 91.84 | **98.36** | 98.5 / 96.5 / 97.4 / 100.0 | 97.5 / 98.6 / 99.3 |
| STN-CNN reduced budget (section 3) | 99.13 | **99.53** | 99.5 / 100.0 / 98.7 / 100.0 | 99.6 / 99.3 / 100.0 |
| ConvNeXt V2 atto Hub checkpoint (section 4) | 99.81 | **99.53** | 99.5 / 100.0 / 98.7 / 100.0 | 99.6 / 99.3 / 100.0 |

- Both the reduced-budget STN-CNN and the ConvNeXt V2 atto checkpoint fail on exactly the same four crops (indices 50, 224, 232 and 718 in gt.txt order): an overexposed triangle whose pictogram is bleached out, a No vehicles sign defaced by a sticker, a 19 px Speed limit 50 and one Children crossing sign; three of the four are degraded signs rather than model weaknesses, which is the kind of case the robustness benchmarks in `requirements_and_robustness.md` section B are built around. LeNet-5 makes 14 errors. Accuracy on GTSDB crops is higher than on the GTSRB test set for every model because the GTSDB frames are daytime recordings with mostly well-exposed signs, while the GTSRB test set keeps the dark and blurred tracks; the crops are therefore an easier same-domain check, not a harder one.

## 9. Panoramax German sign classifier on its validation crops (done)

- Source: `Panoramax/classify_de_road_signs` (etalab-2.0, `classify_de_road_signs.pt`
  26,282,865 bytes) and `Panoramax/classified_de_road_signs` (CC BY-SA 4.0, `val.zip` 29.7 MB,
  6,944 crops in 134 class folders from 1 to 247 images; crops of German signs cut from Panoramax
  street-level imagery by the project's YOLO11l detector, with "bad:*" and "other:*" buckets for
  non-signs). Model card: accuracy 0.986, "based on YOLOv26" (the pickle actually references
  YOLO11-style C3k2 / C2PSA blocks).
- Loading: the checkpoint is a full Ultralytics pickle, so `scripts/eval_panoramax_de_classifier.py`
  disassembles it with `pickletools` first (globals found: torch storages and modules,
  torchvision transforms, `ultralytics.nn.modules.{block,conv,head}` and
  `ultralytics.nn.tasks.ClassificationModel`, nothing else) and then loads it with
  `torch.load(weights_only=True)` after registering exactly those classes. Inference is plain
  PyTorch with the Ultralytics classification eval transform (resize short side to the training
  size, centre crop, ToTensor).
- Result: **98.68 percent top-1** (top-5 99.77 percent) on all 6,944 validation crops at
  imgsz 224; 98.77 percent on the 6,316 crops of real sign classes (bad/other
  excluded). The card's 0.986 is reproduced (98.68 vs 98.6); the remaining errors concentrate in the catch-all buckets (other:multi, other) and in classes with few validation crops (oneway, n=23). It is the only open European classifier found whose training crops are released (CC BY-SA 4.0), which makes it the natural reference for OpenTSR's fine-grained European head. Weakest classes with at least 20 validation crops: oneway 60.9 (n=23), other:multi 69.7 (n=33), no:other:dog 87.0 (n=23), hazard:other 92.0 (n=25), arrow_right 92.3 (n=39).

## 10. Prepared GPU runs and open items

| Run | Why | Command or config | Expected cost | Reference to beat |
|---|---|---|---|---|
| Official GTSDB test split for sections 7 and 8 | the mirror lacks the test ground truth | download FullIJCNN2013.zip from sid.erda.dk locally, then `python scripts/eval_openvocab_gtsdb.py --detector owlv2 --full-gt <dir>/gt.txt --images <dir> --out runs/gtsdb/owlv2_test300.json` | 1 to 2 h CPU, minutes on GPU | none published for OWLv2 |
| Faithful poojahira training | 40 epochs on the 10x augmented set with flips and TTA | `configs/gtsrb/stn_cnn_poojahira_32.yaml` with `epochs: 40` plus the offline 10x expansion (not implemented; the online augmentation is the CPU stand-in) | 1 to 2 h on one GPU | 99.809 (README), 99.71 (section 2) |
| CLIP linear probe | the paper's 85.3 (ViT-B/32) and 92.5 (ViT-L/14) linear-probe numbers | extract open_clip features for 39,209 + 12,630 images, scikit-learn logistic regression | 1 h CPU (B/32) | 85.3 / 92.5 (CLIP paper Table 10) |
| Synthetic-to-real | does training on Synset Signset Germany alone give 98.3 on real GTSRB (Fraunhofer IOSB claim) | ConvNeXt or the STN-CNN trained on `FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset` (CC BY 4.0, 21,500 images), scored with `scripts/eval_hf_classifier_gtsrb.py` style code on the official test | 1 to 3 h on one GPU | 98.3 synthetic-only vs 99.9 real (arXiv 2512.05936) |
| Detector fine-tuning on GTSDB (43 classes, 5-fold) and ZOD signs | first OpenTSR detector baselines | YOLOX / RT-DETR / D-FINE configs to be added under `configs/gtsdb/`, `configs/zod/` after the framework decision in `docs/decisions.md` | hours per run on one GPU | GTSDB 3-category mAP 95.77 (aarcosg Faster R-CNN), ZOD paper numbers |
| Larger VLMs | Qwen2.5-VL 7B / InternVL 3 / Gemma 3 in the multiple-choice protocol of section 6 | `scripts/eval_vlm_gtsrb_mc.py --model <id> --dtype bfloat16` on a GPU | minutes | GPT-4o 0.89, LLaVA-NeXT 0.39 (CdMT paper) |
| Japanese baseline | run `infra-intelligence/road-sign-yolox-tiny` (Apache-2.0, ONNX) on own Japanese footage | its `examples/predict_onnx.py` | minutes | AP50 0.902 on the author's private test set (qualitative only) |

Open items found on the way:

- `bazyl/GTSRB` has 44 output labels in the companion `bazyl/gtsrb-model` (index 43 unused);
  harmless for scoring.
- The GTSDB ReadMe lists 852 boxes for the training package while the IJCNN 2013 paper
  reports 1,213 (or 1,206) for all 900 images; the test package therefore has about 360 boxes.

Make targets: `make gtsrb`, `make gtsdb`, `make replicate-gtsrb-cnn`, `make replicate-gtsrb-released`,
`make replicate-gtsrb-clip`, `make replicate-gtsrb-vlm`, `make replicate-gtsdb-openvocab`,
`make replicate-panoramax`.
