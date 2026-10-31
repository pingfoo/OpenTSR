# Convenience targets. Every target assumes the virtualenv is active (see docs/replication.md).
PY ?= python

.PHONY: s2tld autoware-weights replicate-autoware replicate-coco replicate-lisa train-s2tld smoke

# Download S2TLD (1.4 GB) and build both split manifests / YOLO datasets.
s2tld:
	$(PY) scripts/prepare_s2tld.py --root data/s2tld --download --manifest configs/s2tld/splits/random60_seed0.json --name random60
	$(PY) scripts/prepare_s2tld.py --root data/s2tld --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --name chunked

# Autoware TLR models (Apache-2.0) from Hugging Face, pinned to the tags used in docs/.
autoware-weights:
	bash scripts/download_autoware_weights.sh

# CPU-only evaluations reported in docs/replication.md.
replicate-autoware: autoware-weights
	$(PY) scripts/eval_autoware_classifier.py --preprocess letterbox --s2tld data/s2tld --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --split test --out runs/replication/autoware_classifier_s2tld_test_letterbox.json
	$(PY) scripts/eval_autoware_classifier.py --preprocess stretch --s2tld data/s2tld --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --split test --out runs/replication/autoware_classifier_s2tld_test_stretch.json
	$(PY) scripts/eval_autoware_fine_detector.py --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --split test --roi-scale 4 --out runs/replication/autoware_fine_detector_s2tld_test_scale4.json

replicate-coco:
	$(PY) scripts/eval_coco_pretrained_yolo.py --weights yolo11n.pt --imgsz 1280 --manifest configs/s2tld/splits/chunked_60_10_30_seed0.json --split test --out runs/replication/coco_yolo11n_1280_s2tld_test.json

# LISA (CC BY-NC-SA 4.0, evaluation only): HF export (5 GB, 86k files; slow without HF_TOKEN) and the
# DetectionBench yolo26n weights (AGPL-3.0), then plain Ultralytics validation on the test split (about 35 min on 2 CPUs).
replicate-lisa:
	hf download dronefreak/LISA-Traffic-Lights --repo-type dataset --local-dir data/lisa_hf
	mkdir -p weights/detectionbench && curl -sS -L -o weights/detectionbench/lisa-yolo26n_best.pt https://huggingface.co/dronefreak/lisa-yolo26n/resolve/main/best.pt
	$(PY) scripts/prepare_lisa_hf.py --root data/lisa_hf
	$(PY) scripts/eval_lisa_yolo.py --weights weights/detectionbench/lisa-yolo26n_best.pt --data data/lisa_hf/data/data_abs.yaml --out runs/replication/lisa_yolo26n_test.json

# GPU training runs (see docs/replication.md for expected duration).
train-s2tld:
	$(PY) scripts/train_yolo.py --config configs/s2tld/yolo11s_1280_chunked.yaml
	$(PY) scripts/train_yolo.py --config configs/s2tld/yolo11s_1280_random60.yaml

smoke:
	$(PY) scripts/train_yolo.py --config configs/s2tld/yolo11n_1280_chunked.yaml --smoke --device cpu
