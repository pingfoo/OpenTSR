"""STS loader, track linker, event converter and the L3 event evaluator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opentsr.signs.datasets.sts import load_annotations, parse_line, sequences  # noqa: E402
from opentsr.signs.eval_events import evaluate_sequence, speed_limit_accuracy  # noqa: E402
from opentsr.signs.events import Event, GTBox, events_from_json, events_from_tracks, events_to_json, link_tracks  # noqa: E402
from opentsr.signs.pipeline import Candidate, FrameMeta, Pipeline, load_config  # noqa: E402
from opentsr.signs.pipeline.stages import PrecomputedDetector  # noqa: E402
from opentsr.signs.taxonomy import Taxonomy  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/sts/annotations_set1_excerpt.txt"


def test_sts_parse_line_fields_and_relevance():
    fr = parse_line("1277388031Image000001.jpg:VISIBLE, 1182.0, 800.3, 1147.0, 766.6, PROHIBITORY, 70_SIGN;"
                    "SIDE_ROAD, 1222.3, 832.2, 1201.0, 812.1, OTHER, OTHER;MISC_SIGNS;")
    assert (fr.burst, fr.number, fr.misc_signs) == (1277388031, 1, True)
    assert len(fr.boxes) == 2
    assert fr.boxes[0].box == (1147.0, 766.6, 1182.0, 800.3)
    assert fr.boxes[0].fine == "70_SIGN" and fr.boxes[0].coarse == "PROHIBITORY" and fr.boxes[0].relevant
    assert fr.boxes[1].relevant is False and fr.boxes[1].status == "SIDE_ROAD"
    empty = parse_line("1277381619Image000001.jpg:")
    assert empty.boxes == [] and not empty.misc_signs


def test_sts_excerpt_orders_frames_and_cuts_sequences():
    frames = load_annotations(FIXTURE, set_name="Set1")
    assert len(frames) == 40
    assert [f.index for f in frames] == list(range(40))
    assert all(a.time_s <= b.time_s for a, b in zip(frames, frames[1:]))
    seqs = sequences(frames)
    assert list(seqs) == ["Set1/1277387858", "Set1/1277387921", "Set1/1277388031"]
    assert sum(len(v) for v in seqs.values()) == 40
    mapping = Taxonomy.load().mapping("sts")
    assert all(mapping.coarse_of(b.fine) for f in frames for b in f.boxes)  # every name is mapped


def _excerpt_events():
    frames = load_annotations(FIXTURE, set_name="Set1")
    out = {}
    for sid, frs in sequences(frames).items():
        tracks = link_tracks([b for f in frs for b in f.boxes])
        out[sid] = (frs, events_from_tracks(tracks, sid, {f.index: f.time_s for f in frs}))
    return out


def test_linker_follows_an_approaching_sign_across_sparse_frames():
    frs, events = _excerpt_events()["Set1/1277388031"]
    e = max(events, key=lambda e: e.n_frames)
    assert e.fine == "90_SIGN" and e.relevant and e.n_frames == 8
    sizes = [max(b[2] - b[0], b[3] - b[1]) for _, b in sorted(e.boxes.items())]
    assert sizes == sorted(sizes)  # grows monotonically while approaching
    assert e.first_index == 27 and e.pass_index == 34
    assert all(b.track_id is not None for f in frs for b in f.boxes)


def test_linker_never_merges_different_classes_and_respects_pre_assigned_ids():
    boxes = [GTBox(0, (10, 10, 30, 30), "A"), GTBox(1, (12, 12, 33, 33), "B"), GTBox(1, (11, 11, 31, 31), "A")]
    tracks = link_tracks(boxes)
    assert sorted(len(t.boxes) for t in tracks) == [1, 2]
    pre = [GTBox(0, (10, 10, 30, 30), "A", track_id=7), GTBox(5, (900, 10, 930, 30), "A", track_id=7)]
    assert [t.track_id for t in link_tracks(pre)] == [7] and len(link_tracks(pre)[0].boxes) == 2


def test_events_json_round_trip(tmp_path):
    _, events = _excerpt_events()["Set1/1277387921"]
    events_to_json(tmp_path / "e.json", events, {"source": "test"})
    back = events_from_json(tmp_path / "e.json")
    assert [e.to_dict() for e in back] == [e.to_dict() for e in events]
    assert {e.fine for e in events} == {"GIVE_WAY", "90_SIGN", "50_SIGN"}
    assert [e.relevant for e in events] == [False, True, False]


def _oracle_detections(frs, flip_at=None, phantom_at=None, score=0.9):
    """Detections copied from the annotation, with an optional wrong label on one frame and an
    optional box that corresponds to no sign."""
    dets = {}
    for f in frs:
        cands = []
        for b in f.boxes:
            fine = "30_SIGN" if (flip_at is not None and f.index == flip_at) else b.fine
            cands.append(Candidate(box=b.box, score=score, coarse=b.coarse, extra={"fine": fine, "fine_score": score}))
        if phantom_at is not None and f.index == phantom_at:
            cands.append(Candidate(box=(50.0, 50.0, 90.0, 90.0), score=0.8, extra={"fine": "STOP", "fine_score": 0.8}))
        dets[f.file] = cands
    return dets


def _run(cfg_name, frs, dets, cfg_edit=None):
    cfg = load_config(ROOT / f"configs/system/{cfg_name}.yaml")
    cfg["relevance"] = None  # the oracle test is about temporal behaviour, not geometry
    if cfg_edit:
        cfg_edit(cfg)
    pipe = Pipeline(cfg, overrides={"detector": PrecomputedDetector(detections=dets)})
    frames = [(FrameMeta(frame_id=f.file, index=f.index, time_s=f.time_s, sequence_id=f.sequence_id, width=1280,
                         height=960), None) for f in frs]
    return list(pipe.run(frames))


def test_event_evaluator_scores_oracle_detections_perfectly_under_the_academic_config():
    frs, events = _excerpt_events()["Set1/1277388031"]
    recs = _run("academic", frs, _oracle_detections(frs))
    res = evaluate_sequence(recs, events, coarse_of=Taxonomy.load().mapping("sts").coarse_of)
    assert res.n_events == 7 and res.n_relevant == 3
    assert res.event_recall == 1.0 and res.event_recall_all == 1.0
    assert res.false_alerts_phantom == 0 and res.false_alerts_wrong_class == 0
    # the academic stream alerts on non-relevant signs too, and those count as relevance errors
    assert res.false_alerts_relevance > 0 and res.event_precision < 1.0
    assert res.mean_flips == 0.0 and res.mean_latency_first_s == 0.0


def test_voting_absorbs_a_single_wrong_frame_that_breaks_the_per_frame_stream():
    frs, events = _excerpt_events()["Set1/1277388031"]
    big = max(events, key=lambda e: e.n_frames)  # 90_SIGN over frames 27 to 34
    dets = _oracle_detections(frs, flip_at=31)
    coarse_of = Taxonomy.load().mapping("sts").coarse_of
    per_frame = evaluate_sequence(_run("academic", frs, dets), events, coarse_of=coarse_of)
    r = next(x for x in per_frame.events if x.event_id == big.event_id)
    assert not r.hit and r.flips == 2 and "30_SIGN" in r.classes_emitted
    voted = evaluate_sequence(_run("full", frs, dets), events, coarse_of=coarse_of)
    r2 = next(x for x in voted.events if x.event_id == big.event_id)
    assert r2.hit and r2.flips == 0 and r2.classes_emitted and set(r2.classes_emitted) == {"90_SIGN"}
    assert voted.false_alerts_wrong_class == 0 and per_frame.false_alerts_wrong_class == 1


def test_phantom_detection_is_a_false_alert_per_frame_but_filtered_by_track_confirmation():
    frs, events = _excerpt_events()["Set1/1277388031"]
    dets = _oracle_detections(frs, phantom_at=30)
    per_frame = evaluate_sequence(_run("academic", frs, dets), events)
    assert per_frame.false_alerts_phantom == 1
    full = evaluate_sequence(_run("full", frs, dets), events)
    assert full.false_alerts_phantom == 0


def test_speed_limit_accuracy_is_distance_weighted():
    frs, events = _excerpt_events()["Set1/1277388031"]
    recs = _run("academic", frs, _oracle_detections(frs, flip_at=31))
    for i, r in enumerate(recs):
        r.meta.distance_m = 20.0 * i  # 20 m per annotated frame
    limit_of = lambda m: m.fine.split("_")[0] if m.fine and m.fine.endswith("_SIGN") else None  # noqa: E731
    truth = lambda r: "90"  # noqa: E731
    out = speed_limit_accuracy(recs, truth, limit_of)
    assert out["unit"] == "m" and out["weight"] == 20.0 * (len(recs) - 1)
    # the wrong 30 on frame 31 costs exactly one 20 m interval out of 12
    assert abs(out["accuracy"] - 11 / 12) < 1e-9


def test_event_dataclass_defaults():
    e = Event("s/1", "s", "50_SIGN", "PROHIBITORY", True, 0, 3, 0.0, 6.0, 1, 4)
    assert e.limit_after is None and e.boxes == {}
