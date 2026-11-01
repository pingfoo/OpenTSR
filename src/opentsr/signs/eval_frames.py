"""L2 evaluator: single-frame detection with fine classes (``docs/benchmark.md`` section 2, L2).

Built on the VOC AP of ``opentsr.metrics`` and adds what the benchmark asks for: AP averaged
over IoU 0.50:0.95 (COCO style, on the same greedy matching), the same numbers per coarse
category and class-agnostic (every label mapped to ``"sign"``), and recall by size bucket of
the ground truth's long side. Every number is computed here, never by a framework's validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from opentsr.metrics import Detection, GroundTruth, iou_xyxy, mean_ap, voc_ap_per_class

SIZE_BUCKETS: list[tuple[str, float, float]] = [
    ("<16", 0, 16), ("16-31", 16, 32), ("32-47", 32, 48), ("48-63", 48, 64),
    ("64-95", 64, 96), ("96-127", 96, 128), ("128+", 128, float("inf")),
]
COCO_IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def long_side(box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def bucket_of(size: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= size < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def relabel(items, fn: Callable[[str], str]):
    return [type(x)(**{**vars(x), "label": fn(x.label)}) for x in items]


@dataclass
class DetectionResult:
    map50: float
    map50_95: float
    per_class: dict[str, dict]  # label -> {"ap50", "ap50_95", "n_gt", "recall", "precision"}
    recall_by_size: dict[str, dict]  # bucket -> {"n_gt", "recall"} at IoU 0.5, all detections
    n_gt: int
    n_det: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return vars(self)


def _recall_by_size(dets: list[Detection], gts: list[GroundTruth], iou_thr: float, class_aware: bool) -> dict[str, dict]:
    """Greedy, score-ordered matching once over everything; recall per ground-truth size bucket."""
    gt_by_img: dict[str, list[GroundTruth]] = {}
    for g in gts:
        gt_by_img.setdefault(g.image_id, []).append(g)
    matched: dict[str, np.ndarray] = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}
    for d in sorted(dets, key=lambda d: -d.score):
        cand = gt_by_img.get(d.image_id, [])
        if not cand:
            continue
        ious = iou_xyxy(np.array(d.box), np.array([g.box for g in cand]))[0]
        if class_aware:
            ious = np.where([g.label == d.label for g in cand], ious, 0.0)
        ious = np.where(matched[d.image_id], 0.0, ious)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            matched[d.image_id][j] = True
    out = {name: {"n_gt": 0, "n_hit": 0} for name, _, _ in SIZE_BUCKETS}
    for img, cand in gt_by_img.items():
        for j, g in enumerate(cand):
            if g.difficult:
                continue
            b = out[bucket_of(long_side(g.box))]
            b["n_gt"] += 1
            b["n_hit"] += int(matched[img][j])
    return {k: {"n_gt": v["n_gt"], "recall": (v["n_hit"] / v["n_gt"]) if v["n_gt"] else None} for k, v in out.items()}


def evaluate_detections(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    classes: list[str] | None = None,
    ious: list[float] = COCO_IOUS,
) -> DetectionResult:
    classes = classes or sorted({g.label for g in ground_truths})
    per_iou = {t: voc_ap_per_class(detections, ground_truths, classes, iou_thr=t) for t in ious}
    base = per_iou[0.5] if 0.5 in per_iou else per_iou[ious[0]]
    per_class = {}
    for c in classes:
        per_class[c] = {
            "ap50": base[c]["ap"],
            "ap50_95": float(np.mean([per_iou[t][c]["ap"] for t in ious])),
            "n_gt": base[c]["n_gt"], "n_det": base[c]["n_det"],
            "recall": base[c]["recall"], "precision": base[c]["precision"],
        }
    present = [c for c in classes if per_class[c]["n_gt"] > 0]
    return DetectionResult(
        map50=mean_ap(base, present),
        map50_95=float(np.mean([per_class[c]["ap50_95"] for c in present])) if present else 0.0,
        per_class=per_class,
        recall_by_size=_recall_by_size(detections, ground_truths, 0.5, class_aware=True),
        n_gt=sum(1 for g in ground_truths if not g.difficult), n_det=len(detections),
    )


def evaluate_frames(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    fine_classes: list[str] | None = None,
    coarse_of: Callable[[str], str] | None = None,
) -> dict[str, DetectionResult]:
    """Fine, coarse (when ``coarse_of`` is given) and class-agnostic views of the same detections."""
    out = {"fine": evaluate_detections(detections, ground_truths, fine_classes)}
    if coarse_of is not None:
        out["coarse"] = evaluate_detections(relabel(detections, coarse_of), relabel(ground_truths, coarse_of))
    out["agnostic"] = evaluate_detections(relabel(detections, lambda _: "sign"), relabel(ground_truths, lambda _: "sign"),
                                          ["sign"])
    out["agnostic"].recall_by_size = _recall_by_size(detections, ground_truths, 0.5, class_aware=False)
    return out
