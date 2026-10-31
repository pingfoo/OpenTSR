"""ONNX Runtime wrappers for the Autoware traffic light models published on Hugging Face.

Models (Apache-2.0, trained by TIER IV on Japanese traffic lights):

* ``AutowareFoundation/traffic_light_classifier`` (tag v4.0): MobileNetV2 state classifier,
  224x224 input, 11 car labels or 3 pedestrian labels, softmax output.
* ``AutowareFoundation/traffic_light_fine_detector`` (tag v3.0): YOLOX-s detector, 416x416
  input, classes ``BACKGROUND``, ``traffic_light``, ``pedestrian_traffic_light``; raw YOLOX
  head output ``[N, 3549, 8]`` (``decode_in_inference=False``).

The preprocessing below mirrors the Autoware Universe C++ nodes so that numbers measured here
are comparable to what the ROS 2 pipeline would produce:

* Classifier (``autoware_tensorrt_classifier`` ``preprocess_opt``): the ROI crop, in RGB, is
  letterboxed with ``scale = min(224/w, 224/h)`` using bilinear resize, padded at the bottom
  and right with zeros, then normalised per channel with mean ``[123.675, 116.28, 103.53]``
  and std ``[58.395, 57.12, 57.375]``.
* Fine detector (``autoware_tensorrt_yolox`` ``multiScalePreprocessGpu`` and
  ``decodeOutputs``): each coarse ROI is cropped from the BGR image, letterboxed to 416x416
  with ``scale = min(416/w, 416/h)``, padded with 114, fed without normalisation
  (``norm_factor = 1``); proposals use ``objectness * class_score`` with ``score_thresh``
  (Autoware default 0.3) and NMS at ``nms_thresh`` (default 0.65), boxes are mapped back by
  dividing by ``scale`` and clipped to the crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSIFIER_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
CLASSIFIER_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _session(path: Path, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def letterbox(img: np.ndarray, size: int, pad_value: int) -> tuple[np.ndarray, float]:
    """Resize so the longer side fits ``size`` (bilinear), pad bottom/right. Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype)
    out[:new_h, :new_w] = resized
    return out, scale


class AutowareTLClassifier:
    """MobileNetV2 traffic light state classifier (car or pedestrian variant)."""

    def __init__(self, onnx_path: Path, labels_path: Path, threads: int = 0,
                 preprocess: str = "letterbox"):
        """``preprocess``: ``"letterbox"`` reproduces the Autoware runtime (pad bottom/right);
        ``"stretch"`` reproduces the mmpretrain training pipeline (``Resize(scale=224)``,
        aspect ratio not kept)."""
        if preprocess not in ("letterbox", "stretch"):
            raise ValueError(preprocess)
        self.preprocess_mode = preprocess
        self.session = _session(Path(onnx_path), threads)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.batch = int(shape[0]) if isinstance(shape[0], int) else 1
        self.size = int(shape[2])
        self.labels = [l.strip() for l in Path(labels_path).read_text().splitlines() if l.strip()]
        n_out = int(self.session.get_outputs()[0].shape[1])
        if n_out != len(self.labels):
            raise ValueError(f"model has {n_out} outputs but label file has {len(self.labels)} labels")

    def preprocess(self, roi_rgb: np.ndarray) -> np.ndarray:
        if self.preprocess_mode == "letterbox":
            img, _ = letterbox(roi_rgb, self.size, 0)
        else:
            img = cv2.resize(roi_rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) - CLASSIFIER_MEAN) / CLASSIFIER_STD
        return x.transpose(2, 0, 1)[None]  # NCHW

    def predict_proba(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Softmax probabilities over ``self.labels`` for one RGB ROI crop."""
        x = self.preprocess(roi_rgb)
        if self.batch > 1:
            x = np.repeat(x, self.batch, axis=0)
        out = self.session.run(None, {self.input_name: x})[0]
        return out[0]

    def predict(self, roi_rgb: np.ndarray) -> tuple[str, float]:
        p = self.predict_proba(roi_rgb)
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-12)
        order = rest[iou <= thr]
    return keep


class AutowareTLFineDetector:
    """YOLOX-s traffic light detector operating on coarse ROI crops."""

    def __init__(self, onnx_path: Path, labels_path: Path, threads: int = 0,
                 score_thresh: float = 0.3, nms_thresh: float = 0.65):
        self.session = _session(Path(onnx_path), threads)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.batch = int(shape[0]) if isinstance(shape[0], int) else 1
        self.size = int(shape[2])
        self.labels = [l.strip() for l in Path(labels_path).read_text().splitlines() if l.strip()]
        self.num_classes = len(self.labels)
        self.strides = [8, 16, 32]
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        grids = []
        for s in self.strides:
            g = self.size // s
            ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack([xs.ravel(), ys.ravel(), np.full(g * g, s)], axis=1))
        self.grid_strides = np.concatenate(grids, axis=0).astype(np.float32)  # (3549, 3)

    def preprocess(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        img, scale = letterbox(roi_bgr, self.size, 114)
        x = img.astype(np.float32).transpose(2, 0, 1)[None]  # raw 0..255 BGR, NCHW
        return x, scale

    def detect_roi(self, roi_bgr: np.ndarray) -> list[Det]:
        """Run the detector on one coarse ROI crop. Boxes are returned in crop coordinates."""
        x, scale = self.preprocess(roi_bgr)
        if self.batch > 1:
            x = np.repeat(x, self.batch, axis=0)
        out = self.session.run(None, {self.input_name: x})[0][0]  # (3549, 5 + C)
        if out.shape[0] != self.grid_strides.shape[0]:
            raise RuntimeError(f"unexpected anchor count {out.shape[0]}")
        gx, gy, st = self.grid_strides[:, 0], self.grid_strides[:, 1], self.grid_strides[:, 2]
        cx = (out[:, 0] + gx) * st
        cy = (out[:, 1] + gy) * st
        w = np.exp(out[:, 2]) * st
        h = np.exp(out[:, 3]) * st
        obj = out[:, 4]
        cls_scores = out[:, 5:5 + self.num_classes]
        dets: list[Det] = []
        boxes_all = []
        scores_all = []
        cls_all = []
        for c in range(self.num_classes):
            prob = obj * cls_scores[:, c]
            keep = prob > self.score_thresh
            if not np.any(keep):
                continue
            b = np.stack([cx[keep] - w[keep] / 2, cy[keep] - h[keep] / 2,
                          cx[keep] + w[keep] / 2, cy[keep] + h[keep] / 2], axis=1)
            boxes_all.append(b)
            scores_all.append(prob[keep])
            cls_all.append(np.full(b.shape[0], c))
        if not boxes_all:
            return dets
        boxes = np.concatenate(boxes_all)
        scores = np.concatenate(scores_all)
        clss = np.concatenate(cls_all)
        # Autoware applies NMS over all proposals regardless of class (nmsSortedBboxes).
        keep = _nms(boxes, scores, self.nms_thresh)
        crop_h, crop_w = roi_bgr.shape[:2]
        for i in keep:
            x1, y1, x2, y2 = boxes[i] / scale
            x1 = float(np.clip(x1, 0, crop_w - 1))
            y1 = float(np.clip(y1, 0, crop_h - 1))
            x2 = float(np.clip(x2, 0, crop_w - 1))
            y2 = float(np.clip(y2, 0, crop_h - 1))
            dets.append(Det(x1, y1, x2, y2, float(scores[i]), int(clss[i])))
        return dets
