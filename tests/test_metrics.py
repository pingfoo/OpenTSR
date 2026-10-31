"""Hand-checked cases for the VOC AP implementation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentsr.metrics import Detection, GroundTruth, iou_xyxy, mean_ap, voc_ap_per_class  # noqa: E402


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 5, 15, 15], [0, 0, 10, 10], [20, 20, 30, 30]])
    iou = iou_xyxy(a, b)[0]
    assert abs(iou[0] - 25 / 175) < 1e-9
    assert iou[1] == 1.0
    assert iou[2] == 0.0


def test_perfect_detections():
    gts = [GroundTruth("i1", "red", (0, 0, 10, 10)), GroundTruth("i2", "red", (0, 0, 10, 10))]
    dets = [Detection("i1", "red", 0.9, (0, 0, 10, 10)), Detection("i2", "red", 0.8, (1, 1, 10, 10))]
    res = voc_ap_per_class(dets, gts, ["red"])
    assert res["red"]["ap"] == 1.0
    assert res["red"]["recall"] == 1.0


def test_half_recall_with_false_positive():
    # One GT found at rank 1, one FP at rank 2, one GT never found:
    # precision/recall curve: (1.0, 0.5) then (0.5, 0.5) -> all-point AP = 0.5
    gts = [GroundTruth("i1", "red", (0, 0, 10, 10)), GroundTruth("i2", "red", (0, 0, 10, 10))]
    dets = [Detection("i1", "red", 0.9, (0, 0, 10, 10)), Detection("i2", "red", 0.8, (50, 50, 60, 60))]
    res = voc_ap_per_class(dets, gts, ["red"])
    assert abs(res["red"]["ap"] - 0.5) < 1e-9
    assert abs(voc_ap_per_class(dets, gts, ["red"], use_07_metric=True)["red"]["ap"] - 6 / 11) < 1e-9


def test_duplicate_is_false_positive_and_fp_first_lowers_ap():
    gts = [GroundTruth("i1", "red", (0, 0, 10, 10))]
    dets = [
        Detection("i1", "red", 0.9, (40, 40, 50, 50)),  # FP ranked first
        Detection("i1", "red", 0.8, (0, 0, 10, 10)),    # TP
        Detection("i1", "red", 0.7, (0, 0, 10, 10)),    # duplicate -> FP
    ]
    res = voc_ap_per_class(dets, gts, ["red"])
    # curve: rank1 p=0 r=0, rank2 p=0.5 r=1, rank3 p=1/3 r=1 -> AP = 0.5
    assert abs(res["red"]["ap"] - 0.5) < 1e-9


def test_mean_ap_ignores_classes_without_gt():
    gts = [GroundTruth("i1", "red", (0, 0, 10, 10))]
    dets = [Detection("i1", "red", 0.9, (0, 0, 10, 10)), Detection("i1", "green", 0.9, (0, 0, 10, 10))]
    res = voc_ap_per_class(dets, gts, ["red", "green"])
    assert mean_ap(res, ["red", "green"]) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
