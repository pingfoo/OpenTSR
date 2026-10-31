#!/usr/bin/env bash
# Download the Autoware traffic light models (Apache-2.0) from Hugging Face, pinned to the
# tags evaluated in traffic_lights/docs/replication.md, and verify their checksums.
set -euo pipefail
dst="${1:-weights/traffic_lights/autoware}"
mkdir -p "$dst"
clf="https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve/v4.0"
det="https://huggingface.co/AutowareFoundation/traffic_light_fine_detector/resolve/v3.0"
fetch() { curl -sS -L -o "$dst/$2" "$1/$2"; }
fetch "$clf" traffic_light_classifier_mobilenetv2_batch_1.onnx
fetch "$clf" ped_traffic_light_classifier_mobilenetv2_batch_1.onnx
fetch "$clf" traffic_light_lamp_recognizer_comlops.onnx
fetch "$clf" lamp_labels.txt
fetch "$clf" lamp_labels_ped.txt
fetch "$clf" lamp_recognizer_ml.param.yaml
fetch "$det" tlr_car_ped_yolox_s_batch_1.onnx
fetch "$det" tlr_labels.txt
cat > "$dst/SHA256SUMS" <<'SUMS'
455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703  traffic_light_classifier_mobilenetv2_batch_1.onnx
b52632fee96d1bc99922e743335ebfd49d6a0645c8a04e615f156e38661add24  ped_traffic_light_classifier_mobilenetv2_batch_1.onnx
300af04a5893961195a9b7ffd9d80364280f328a6dc549f9b1e05d428d96d2e5  traffic_light_lamp_recognizer_comlops.onnx
1ad633066a1195006f4709f8fa07800dd65a74a814b3efb4c99bcc5a1a7962f6  tlr_car_ped_yolox_s_batch_1.onnx
SUMS
(cd "$dst" && sha256sum -c SHA256SUMS)
