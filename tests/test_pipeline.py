"""Pipeline skeleton: configuration, null stages, tracker, memory, decisions, JSONL round trip."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opentsr.signs.pipeline import (  # noqa: E402
    STAGE_ORDER, Candidate, FrameMeta, PassThrough, Pipeline, ablation_variants, build_stage, load_config,
    normalize, read_log, run_sequence, stages_on,
)
from opentsr.signs.pipeline.stages import PrecomputedDetector  # noqa: E402


def _frames(n=8, w=1280, h=960, dt=0.5):
    return [(FrameMeta(frame_id=f"f{i:03d}", index=i, time_s=i * dt, sequence_id="toy", width=w, height=h), None)
            for i in range(n)]


def _growing_box(i, x0=900.0, size0=24.0, growth=8.0, dx=5.0):
    s = size0 + growth * i
    x = x0 + dx * i
    return (x, 400.0, x + s, 400.0 + s)


def _detections(fine_per_frame, n=8, score=0.9, extra_noise=None):
    """One growing sign with a fine label per frame, plus optional one-frame noise boxes."""
    out = {}
    for i in range(n):
        cands = [Candidate(box=_growing_box(i), score=score, coarse="prohibitory_round_red",
                           extra={"fine": fine_per_frame[i], "fine_score": score})]
        if extra_noise and i in extra_noise:
            cands.append(Candidate(box=(100.0, 100.0, 130.0, 130.0), score=0.5, coarse="other", extra={"fine": "OTHER"}))
        out[f"f{i:03d}"] = cands
    return out


def test_configs_load_and_validate():
    full = load_config(ROOT / "configs/system/full.yaml")
    academic = load_config(ROOT / "configs/system/academic.yaml")
    assert set(STAGE_ORDER) <= set(full)
    assert stages_on(academic) == ["detector", "classifier", "decision"]
    assert "reader" in STAGE_ORDER and full["reader"] is None
    with pytest.raises(KeyError):
        normalize({"detector": {"type": "precomputed"}, "trakcer": None})
    with pytest.raises(ValueError):
        normalize({"detector": None})
    with pytest.raises(KeyError):
        build_stage("tracker", {"type": "does-not-exist"})
    assert isinstance(build_stage("fusion", None), PassThrough)


def test_ablation_variants_cover_one_off_and_one_on():
    full = load_config(ROOT / "configs/system/full.yaml")
    base = load_config(ROOT / "configs/system/academic.yaml")
    names = [v["name"] for v in ablation_variants(full, base)]
    assert names[0] == "full"
    assert "full-minus-tracker" in names and "full-minus-memory" in names
    assert "base-plus-tracker" in names and "no_temporal" in names
    v = next(v for v in ablation_variants(full, base) if v["name"] == "full-minus-tracker")
    assert v["tracker"] is None and v["memory"] is not None


def _run(cfg_name, dets, n=8, overrides=None):
    cfg = load_config(ROOT / f"configs/system/{cfg_name}.yaml")
    ov = {"detector": PrecomputedDetector(detections=dets)}
    ov.update(overrides or {})
    return list(Pipeline(cfg, overrides=ov).run(_frames(n)))


def test_academic_config_emits_every_frame_and_follows_the_frame_label():
    labels = ["50", "50", "70", "50", "50", "50", "70", "50"]
    recs = _run("academic", _detections(labels))
    assert all(len(r.messages) == 1 for r in recs)
    assert [r.messages[0].fine for r in recs] == labels
    assert all(m.track_id < 0 for r in recs for m in r.messages)  # untracked, one frame only


def test_full_config_votes_and_suppresses_flips_and_one_frame_noise():
    labels = ["50", "50", "70", "50", "50", "50", "70", "50"]
    recs = _run("full", _detections(labels, extra_noise={3}))
    msgs = [m for r in recs for m in r.messages]
    assert msgs, "the vote should emit after min_frames"
    assert {m.fine for m in msgs} == {"50"}  # the two 70 frames never win the vote
    assert len({m.track_id for m in msgs}) == 1  # the noise box is dropped by confirm_frames
    assert msgs[0].state == "new" and all(m.state == "keep" for m in msgs[1:])


def test_memory_keeps_a_sign_after_it_disappears_and_null_memory_does_not():
    labels = ["50"] * 5
    dets = _detections(labels, n=5)
    recs = _run("full", dets, n=8)  # frames 5 to 7 have no detections, TTL is 3 s, dt is 0.5 s
    kept = [r for r in recs if r.meta.index >= 5]
    assert all(len(r.signs) == 1 and not r.signs[0].visible for r in kept)
    assert all(len(r.messages) == 1 and r.messages[0].state == "keep" for r in kept)
    cfg = load_config(ROOT / "configs/system/full.yaml")
    cfg["memory"] = None
    recs2 = list(Pipeline(cfg, overrides={"detector": PrecomputedDetector(detections=dets)}).run(_frames(8)))
    assert all(len(r.signs) == 0 and len(r.messages) == 0 for r in recs2 if r.meta.index >= 5)
    assert any(r.messages for r in recs2 if r.meta.index < 5)


def test_relevance_drops_side_signs():
    dets = {f"f{i:03d}": [Candidate(box=(10.0 + i, 400.0, 60.0 + i, 450.0), score=0.9, extra={"fine": "50"})]
            for i in range(4)}
    recs = _run("full", dets, n=4)
    assert all(not r.messages for r in recs)
    assert all(s.relevant is False for r in recs for s in r.signs)


def test_jsonl_round_trip(tmp_path):
    labels = ["50"] * 4
    cfg = load_config(ROOT / "configs/system/full.yaml")
    pipe = Pipeline(cfg, overrides={"detector": PrecomputedDetector(detections=_detections(labels, n=4))})
    recs = run_sequence(pipe, "toy", _frames(4), log_dir=tmp_path)
    cfg2, recs2 = read_log(tmp_path / "toy.jsonl")
    assert cfg2["name"] == "full"
    assert [r.to_dict() for r in recs] == [r.to_dict() for r in recs2]
    assert all("tracker" in r.stage_times_ms for r in recs2)
