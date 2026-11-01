# OpenTSR convenience targets. Run from the repository root with the virtualenv active.
#
# Traffic sign targets (main track) are added here as they appear.
# Traffic light targets (secondary track) live in traffic_lights/Makefile and are reached with a
# "lights-" prefix, for example `make lights-s2tld` or `make lights-replicate-autoware`.
PY ?= python

.PHONY: test help gtsrb gtsdb replicate-gtsrb-cnn replicate-gtsrb-released replicate-gtsrb-clip replicate-gtsrb-vlm replicate-gtsdb-openvocab replicate-panoramax catalog-data catalog

help:
	@echo "make test                       run the unit tests"
	@echo "make gtsrb / gtsdb              download and verify the GTSRB (312 MB) and GTSDB (1.73 GB) mirrors"
	@echo "make replicate-gtsrb-cnn        LeNet-5 and STN-CNN trained from scratch on the official GTSRB split"
	@echo "make replicate-gtsrb-released   released checkpoints (poojahira, Hub models) on the official test split"
	@echo "make replicate-gtsrb-clip       CLIP / SigLIP zero-shot with the OpenAI prompts"
	@echo "make replicate-gtsrb-vlm        SmolVLM / Qwen2-VL multiple choice on a stratified subset"
	@echo "make replicate-gtsdb-openvocab  OWLv2 and YOLO-World class-agnostic detection on GTSDB, GTSDB-crop check"
	@echo "make replicate-panoramax        Panoramax German sign classifier on its CC BY-SA validation crops"
	@echo "make catalog-data               extra downloads for the dataset catalog (Synset, TT100K shards, MTSD sample, CURE-TSR, MLIT list)"
	@echo "make catalog                    build the illustrated dataset catalog (runs/catalog/catalog.html and .pdf)"
	@echo "make lights-<target>            run a target from traffic_lights/Makefile (see that file)"

test:
	$(PY) -m pytest -q tests

lights-%:
	$(MAKE) -f traffic_lights/Makefile PY="$(PY)" $*

# Dataset catalog (docs/dataset_catalog.md). Needs `make gtsrb gtsdb replicate-panoramax` data first; the PDF step
# needs playwright with a Chromium build (`pip install playwright && playwright install chromium`).
catalog-data:
	hf download FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset cycles/validation.parquet --repo-type dataset --local-dir data/synset_signset
	hf download finedet/tt100k categories.csv data/test-00003-of-00004.parquet data/train-00005-of-00007.parquet data/train-00006-of-00007.parquet --repo-type dataset --local-dir data/tt100k_hf
	$(PY) scripts/sample_mtsd_official.py --n 320 --out data/mtsd_official
	$(PY) scripts/sample_mtsd_official.py --n 900 --seed 1 --out data/mtsd_official
	mkdir -p data/cure_tsr data/japan && curl -sS -L -C - -o data/cure_tsr/CURE-TSR.zip "https://zenodo.org/records/3903066/files/CURE-TSR.zip?download=1"
	curl -sS -L -o data/japan/ichiran.pdf https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf
	$(PY) scripts/parse_ja_wikipedia_signs.py --out data/japan/wikipedia_signs.json
	$(PY) scripts/cut_mlit_sign_list.py --pdf data/japan/ichiran.pdf --out data/japan/mlit --names-json data/japan/wikipedia_signs.json

catalog:
	$(PY) scripts/make_dataset_catalog.py --out runs/catalog --seed 0 --pdf --japan-json data/japan/mlit/signs.json --japan-overview data/japan/mlit/page.png

# ---------------------------------------------------------------------------------------------
# Traffic sign replications (docs/replication.md). Needs: pip install -r requirements.txt

# GTSRB official split from the Hugging Face parquet mirror (312 MB), verified against the official sizes.
gtsrb:
	$(PY) scripts/prepare_gtsrb.py --root data/gtsrb_hf --download

# GTSDB training package (600 images + gt.txt) and the 300 test images without ground truth (1.73 GB).
gtsdb:
	$(PY) scripts/prepare_gtsdb.py --root data/gtsdb_hf --download

# From-scratch CNN baselines on the official split (about 4 min for LeNet, 45 min for the STN net on 2 CPU cores).
replicate-gtsrb-cnn:
	$(PY) scripts/train_gtsrb_classifier.py --config configs/gtsrb/lenet5_gray_32_noaug.yaml
	$(PY) scripts/train_gtsrb_classifier.py --config configs/gtsrb/lenet5_gray_32.yaml
	$(PY) scripts/train_gtsrb_classifier.py --config configs/gtsrb/stn_cnn_poojahira_32.yaml

# Released checkpoints scored on the official test split: poojahira's model_40.pth (GitHub) and Hub models
# with their own preprocessing.
replicate-gtsrb-released:
	mkdir -p weights/poojahira && curl -sS -L -o weights/poojahira/model_40.pth https://raw.githubusercontent.com/poojahira/gtsrb-pytorch/master/model/model_40.pth
	$(PY) scripts/eval_poojahira_weights_gtsrb.py --weights weights/poojahira/model_40.pth
	$(PY) scripts/eval_hf_classifier_gtsrb.py --model FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb
	$(PY) scripts/eval_hf_classifier_gtsrb.py --model bazyl/gtsrb-model

# CLIP / SigLIP zero-shot with the OpenAI prompts (compare with CLIP paper Table 11). The script switches OpenAI
# weights to the -quickgelu configs; open_clip 3.x otherwise loads them with plain GELU and loses about 6 points.
replicate-gtsrb-clip:
	$(PY) scripts/eval_clip_zeroshot_gtsrb.py --model ViT-B-32 --pretrained openai
	$(PY) scripts/eval_clip_zeroshot_gtsrb.py --model ViT-B-16 --pretrained openai
	$(PY) scripts/eval_clip_zeroshot_gtsrb.py --model ViT-B-16-SigLIP --pretrained webli
	$(PY) scripts/eval_clip_zeroshot_gtsrb.py --model ViT-L-14 --pretrained openai

# Small instruction-tuned VLMs in 43-way multiple choice on a stratified test subset.
replicate-gtsrb-vlm:
	$(PY) scripts/eval_vlm_gtsrb_mc.py --model HuggingFaceTB/SmolVLM-500M-Instruct --per-class 5 --no-image-splitting
	$(PY) scripts/eval_vlm_gtsrb_mc.py --model Qwen/Qwen2-VL-2B-Instruct --per-class 5 --dtype bfloat16

# Open-vocabulary detectors, class-agnostic, on the 600 GTSDB training images (official gt.txt), and the
# cross-capture check of GTSRB-trained classifiers on the 852 GTSDB crops. YOLO-World needs ultralytics (AGPL-3.0).
replicate-gtsdb-openvocab:
	$(PY) scripts/eval_openvocab_gtsdb.py --detector owlv2 --out runs/gtsdb/owlv2_train600.json
	$(PY) scripts/eval_openvocab_gtsdb.py --detector yolo-world --out runs/gtsdb/yolo_world_s_train600.json
	$(PY) scripts/eval_gtsrb_models_on_gtsdb_crops.py --cnn-config configs/gtsrb/stn_cnn_poojahira_32.yaml --cnn-weights runs/gtsrb/stn_cnn_poojahira_32.pt --hf-model FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb --out runs/gtsdb/crops_stn_and_convnext.json

# Panoramax German sign classifier (etalab-2.0) on its own validation crops (CC BY-SA 4.0, 30 MB), loaded
# through the restricted unpickler in the script (needs the ultralytics package for the class definitions).
replicate-panoramax:
	hf download Panoramax/classify_de_road_signs classify_de_road_signs.pt --local-dir weights/panoramax
	hf download Panoramax/classified_de_road_signs val.zip --repo-type dataset --local-dir data/panoramax_de
	cd data/panoramax_de && unzip -q -o val.zip
	$(PY) scripts/eval_panoramax_de_classifier.py
