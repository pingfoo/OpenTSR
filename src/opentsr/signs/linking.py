"""Box association across sparsely sampled frames, shared by the ground-truth track linker
(``events.py``) and the pipeline's motion tracker (``pipeline/stages.py``).

The rule is written for labels or detections that can be seconds apart (STS annotates every
fifth raw frame): IoU links outright; otherwise a linear extrapolation of centre and size over
the last two boxes, or a distance limit in units of sign size for a track with one box.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from opentsr.metrics import iou_xyxy
from opentsr.signs.pipeline.schema import Box, box_long_side


def _center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def link_score(history: Sequence[Box], b: Box, iou_thr: float, center_factor: float, first_factor: float,
                scale_ratio: float) -> float:
    """Score for appending box ``b`` to a track whose boxes are ``history``; 0 means the pair must not be linked.

    Sparse labels (STS: every fifth raw frame, about 2 s apart) make IoU useless for a sign that
    is being approached, so the rule is: IoU above ``iou_thr`` links outright; otherwise, with
    two or more boxes in the track, extrapolate centre and size linearly and accept a box within
    ``center_factor`` predicted sizes of the predicted centre; with one box, accept a box within
    ``first_factor`` sizes of the last centre. The size ratio must stay within ``scale_ratio`` in
    both cases. (Two variants were tried and dropped on the simulated STS runs: an "outward from
    the vanishing point" rule, because the STS camera is not centred on the road, and a
    one-box fallback when the extrapolation misses, because it merges neighbouring signs of the
    same family and doubles the class flips.) (An "outward from the vanishing point" rule was tried and dropped: in STS the
    camera is not centred on the road, so approaching signs can move towards the image centre.)
    """
    last = history[-1]
    iou = float(iou_xyxy(np.array(last), np.array(b))[0, 0])
    if iou >= iou_thr:
        return 2.0 + iou
    la, lb = box_long_side(last), box_long_side(b)
    if la <= 0 or lb <= 0:
        return 0.0
    cl, cb = _center(last), _center(b)
    if len(history) >= 2:
        prev = history[-2]
        cp, lp = _center(prev), box_long_side(prev)
        pred_c = (2 * cl[0] - cp[0], 2 * cl[1] - cp[1])
        pred_l = max(la + (la - lp), 1.0)
        ratio = max(pred_l, lb) / min(pred_l, lb)
        if ratio > scale_ratio:
            return 0.0
        limit = center_factor * pred_l
        dist = float(np.hypot(cb[0] - pred_c[0], cb[1] - pred_c[1]))
    else:
        ratio = max(la, lb) / min(la, lb)
        if ratio > scale_ratio:
            return 0.0
        limit = first_factor * max(la, lb)
        dist = float(np.hypot(cb[0] - cl[0], cb[1] - cl[1]))
    return 0.0 if dist > limit else 1.0 - dist / limit
