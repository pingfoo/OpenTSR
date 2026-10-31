"""Evaluation metrics: IoU, PASCAL VOC average precision, and simple classification scores.

The AP implementation follows the VOC devkit: detections are sorted by score, each is
matched greedily to the best-overlapping unmatched ground truth of the same class with
IoU >= ``iou_thr``; duplicates count as false positives. ``use_07_metric`` selects the
11-point interpolation of VOC2007, otherwise the all-point interpolation of VOC2010+.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes ``a`` (N,4) and ``b`` (M,4) in xyxy format."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class Detection:
    image_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class GroundTruth:
    image_id: str
    label: str
    box: tuple[float, float, float, float]
    difficult: bool = False


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray, use_07_metric: bool) -> float:
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0.0
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def voc_ap_per_class(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    classes: list[str],
    iou_thr: float = 0.5,
    use_07_metric: bool = False,
) -> dict[str, dict]:
    """Per-class AP. Returns ``{class: {"ap", "n_gt", "n_det", "recall", "precision"}}``.

    ``recall``/``precision`` are the values at the end of the ranked list (all detections used).
    """
    gts_by_class_img: dict[str, dict[str, list[GroundTruth]]] = defaultdict(lambda: defaultdict(list))
    for gt in ground_truths:
        gts_by_class_img[gt.label][gt.image_id].append(gt)
    dets_by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        dets_by_class[d.label].append(d)

    results: dict[str, dict] = {}
    for cls in classes:
        gt_imgs = gts_by_class_img.get(cls, {})
        n_pos = sum(1 for lst in gt_imgs.values() for g in lst if not g.difficult)
        dets = sorted(dets_by_class.get(cls, []), key=lambda d: -d.score)
        matched: dict[str, np.ndarray] = {
            img: np.zeros(len(lst), dtype=bool) for img, lst in gt_imgs.items()
        }
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for i, d in enumerate(dets):
            gts = gt_imgs.get(d.image_id, [])
            if not gts:
                fp[i] = 1
                continue
            ious = iou_xyxy(np.array(d.box), np.array([g.box for g in gts]))[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                if gts[j].difficult:
                    continue  # neither TP nor FP, as in the VOC devkit
                if not matched[d.image_id][j]:
                    tp[i] = 1
                    matched[d.image_id][j] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / max(n_pos, 1)
        prec = tp_c / np.maximum(tp_c + fp_c, np.finfo(np.float64).eps)
        ap = _ap_from_pr(rec, prec, use_07_metric) if n_pos > 0 and len(dets) > 0 else 0.0
        results[cls] = {
            "ap": ap,
            "n_gt": int(n_pos),
            "n_det": len(dets),
            "recall": float(rec[-1]) if len(rec) else 0.0,
            "precision": float(prec[-1]) if len(prec) else 0.0,
        }
    return results


def mean_ap(per_class: dict[str, dict], classes: list[str] | None = None) -> float:
    classes = classes or [c for c, v in per_class.items() if v["n_gt"] > 0]
    vals = [per_class[c]["ap"] for c in classes if per_class[c]["n_gt"] > 0]
    return float(np.mean(vals)) if vals else 0.0


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))
