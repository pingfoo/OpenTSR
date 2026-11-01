"""L2 evaluator (COCO-style AP, size buckets, coarse and agnostic views) and the taxonomy tables."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opentsr.metrics import Detection, GroundTruth  # noqa: E402
from opentsr.signs.datasets.gtsdb import category_of  # noqa: E402
from opentsr.signs.datasets.gtsrb import NUM_CLASSES  # noqa: E402
from opentsr.signs.eval_frames import SIZE_BUCKETS, bucket_of, evaluate_detections, evaluate_frames  # noqa: E402
from opentsr.signs.taxonomy import Taxonomy  # noqa: E402


def _toy():
    gts = [
        GroundTruth("a", "1", (10, 10, 40, 40)),  # 30 px
        GroundTruth("a", "13", (100, 100, 220, 220)),  # 120 px
        GroundTruth("b", "1", (0, 0, 12, 12)),  # 12 px
        GroundTruth("b", "35", (50, 50, 110, 110)),  # 60 px
    ]
    dets = [
        Detection("a", "1", 0.9, (11, 11, 41, 41)),  # good
        Detection("a", "13", 0.8, (100, 100, 200, 220)),  # IoU 0.83: TP at 0.5, FP at 0.9
        Detection("b", "1", 0.7, (30, 30, 42, 42)),  # miss
        Detection("b", "35", 0.6, (50, 50, 110, 110)),  # exact
        Detection("a", "35", 0.5, (300, 300, 330, 330)),  # phantom
    ]
    return dets, gts


def test_buckets_cover_the_documented_ranges():
    assert [b[0] for b in SIZE_BUCKETS] == ["<16", "16-31", "32-47", "48-63", "64-95", "96-127", "128+"]
    assert bucket_of(15.9) == "<16" and bucket_of(16) == "16-31" and bucket_of(500) == "128+"


def test_evaluate_detections_ap_values_and_size_recall():
    dets, gts = _toy()
    res = evaluate_detections(dets, gts, classes=["1", "13", "35"])
    assert res.n_gt == 4 and res.n_det == 5
    assert res.per_class["13"]["ap50"] == 1.0 and res.per_class["13"]["ap50_95"] < 1.0
    assert res.per_class["35"]["ap50"] == 1.0 and res.per_class["35"]["ap50_95"] == 1.0  # exact box, phantom ranked lower
    assert res.per_class["1"]["ap50"] == pytest.approx(0.5)
    assert res.map50 == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert res.recall_by_size["<16"] == {"n_gt": 1, "recall": 0.0}
    assert res.recall_by_size["16-31"] == {"n_gt": 1, "recall": 1.0}
    assert res.recall_by_size["48-63"]["recall"] == 1.0 and res.recall_by_size["96-127"]["recall"] == 1.0
    assert res.recall_by_size["128+"]["n_gt"] == 0 and res.recall_by_size["128+"]["recall"] is None


def test_evaluate_frames_gives_fine_coarse_and_agnostic_views():
    dets, gts = _toy()
    coarse = Taxonomy.load().mapping("gtsrb").coarse_of
    views = evaluate_frames(dets, gts, coarse_of=coarse)
    assert set(views) == {"fine", "coarse", "agnostic"}
    assert views["agnostic"].map50 == pytest.approx(0.6875)  # one ranked list over all 4 boxes
    assert views["coarse"].map50 == pytest.approx((0.5 + 1.0 + 1.0) / 3)  # same as fine here: no cross-class confusion
    assert set(views["coarse"].per_class) == {"prohibitory_round_red", "yield", "mandatory_round_blue"}
    # class-agnostic size recall counts the phantom's bucket for nothing and the 12 px miss stays a miss
    assert views["agnostic"].recall_by_size["<16"]["recall"] == 0.0


def test_coarse_taxonomy_and_mappings_are_consistent():
    tax = Taxonomy.load()
    assert tax.version == 1 and "other" in tax.families and len(tax.families) == 13
    g = tax.mapping("gtsrb")
    assert sorted(int(k) for k in g.table) == list(range(NUM_CLASSES))
    # the GTSDB competition categories are refinements of coarse_v1, never the other way round
    for cid in range(NUM_CLASSES):
        cat = category_of(cid)
        fam = g.coarse_of(cid)
        if cat == "prohibitory":
            assert fam == "prohibitory_round_red"
        elif cat == "mandatory":
            assert fam == "mandatory_round_blue"
        elif cat == "danger":
            assert fam == "warning_triangle"
    for name in ("sts", "cure_tsr"):
        m = tax.mapping(name)
        assert set(m.families_used) <= set(tax.families)
    with pytest.raises(KeyError):
        g.coarse_of("not-a-class")
    assert tax.mapping("gtsrb", default="other").coarse_of("not-a-class") == "other"
    with pytest.raises(ValueError):
        tax.mapping("gtsrb", default="no-such-family")
