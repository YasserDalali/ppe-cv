import numpy as np
import pytest

from ppe.evaluate import _iou
from ppe.predict import Boxes
from ppe.scoring import (ImageMatch, ap_from, iou_matrix, map50_fast, match_image,
                         recording_bootstrap, summarize)


def boxes(rows):
    if not rows:
        return Boxes.empty()
    return Boxes(np.array([r[:4] for r in rows], dtype=float),
                 [r[4] for r in rows], np.array([r[5] for r in rows], dtype=float))


def test_iou_matrix_matches_scalar_iou():
    rng = np.random.default_rng(0)
    a = np.sort(rng.uniform(0, 100, (5, 4)).reshape(5, 2, 2), axis=1).transpose(0, 2, 1).reshape(5, 4)
    a = a[:, [0, 2, 1, 3]]
    b = a[::-1] + 3
    m = iou_matrix(a, b)
    for i in range(5):
        for j in range(5):
            assert m[i, j] == pytest.approx(_iou(a[i], b[j]))


def test_perfect_predictions_score_one():
    gt = boxes([(0, 0, 10, 10, "Person", 1), (20, 20, 30, 30, "helmet", 1)])
    m = match_image("x.jpg", gt, boxes([(0, 0, 10, 10, "person", 0.9),
                                        (20, 20, 30, 30, "helmet", 0.8)]), ["Person", "helmet"])
    s = summarize([m], ["Person", "helmet"], 0.25)
    assert s["map50"] == pytest.approx(1.0) and s["map50_95"] == pytest.approx(1.0)
    assert s["P"] == 1.0 and s["R"] == 1.0


def test_hand_computed_ap_and_operating_point():
    gt = boxes([(0, 0, 10, 10, "helmet", 1), (20, 20, 30, 30, "helmet", 1)])
    pred = boxes([(0, 0, 10, 10, "helmet", 0.9),     # TP
                  (50, 50, 60, 60, "helmet", 0.8),   # FP
                  (20, 20, 30, 30, "helmet", 0.1)])  # TP below the operating threshold
    s = summarize([match_image("x.jpg", gt, pred, ["helmet"])], ["helmet"], 0.25)
    # recall 0.5,0.5,1.0 / precision envelope 1,2/3,2/3 -> 51 points at 1, 50 at 2/3
    assert s["map50"] == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert s["P"] == pytest.approx(0.5) and s["R"] == pytest.approx(0.5)


def test_wrong_class_is_never_a_match_and_absent_classes_are_excluded():
    gt = boxes([(0, 0, 10, 10, "helmet", 1)])
    pred = boxes([(0, 0, 10, 10, "no_helmet", 0.9)])
    s = summarize([match_image("x.jpg", gt, pred, ["helmet", "no_helmet"])],
                  ["helmet", "no_helmet"], 0.25)
    assert s["classes"] == ["helmet"]               # no_helmet has no ground truth here
    assert s["map50"] == 0.0 and s["P"] == 0.0
    assert np.isnan(s["per_class"]["no_helmet"]["ap50"])


def test_out_of_scope_predictions_are_ignored():
    gt = boxes([(0, 0, 10, 10, "helmet", 1)])
    pred = boxes([(0, 0, 10, 10, "helmet", 0.9), (40, 40, 50, 50, "gas mask", 0.9)])
    s = summarize([match_image("x.jpg", gt, pred, ["helmet"])], ["helmet"], 0.25)
    assert s["P"] == 1.0


def test_ap_agrees_with_supervision_on_random_data():
    # supervision and pycocotools agree; ours matched pycocotools to 1e-6
    # on valid boxes when this was written (2026-10-27).
    sv = pytest.importorskip("supervision")
    from supervision.metrics import MeanAveragePrecision

    rng = np.random.default_rng(1)
    names = ["a", "b"]
    metric = MeanAveragePrecision()
    matches = []
    for i in range(40):
        n = rng.integers(1, 5)
        xy = rng.uniform(0, 200, (n, 2))
        wh = rng.uniform(10, 60, (n, 2))
        g = np.hstack([xy, xy + wh])
        gcls = rng.integers(0, 2, n)
        p = np.vstack([g + rng.normal(0, 6, g.shape), g[:1] + 80])
        p = np.hstack([np.minimum(p[:, :2], p[:, 2:]), np.maximum(p[:, :2], p[:, 2:]) + 1])
        pcls = np.concatenate([gcls, [0]])
        pconf = rng.uniform(0.01, 1, len(p))
        gt_b = Boxes(g, [names[c] for c in gcls], np.ones(n))
        pr_b = Boxes(p, [names[c] for c in pcls], pconf)
        matches.append(match_image(f"{i}.jpg", gt_b, pr_b, names))
        metric.update(sv.Detections(xyxy=p, class_id=pcls, confidence=pconf),
                      sv.Detections(xyxy=g, class_id=gcls))
    ours = summarize(matches, names, 0.25)
    theirs = metric.compute()
    assert ours["map50"] == pytest.approx(float(theirs.map50), abs=1e-6)
    assert ours["map50_95"] == pytest.approx(float(theirs.map50_95), abs=1e-6)


def _match(name, n_gt, hits):
    """Synthetic ImageMatch for class 'a': hits = [(conf, is_tp)]."""
    conf = np.array([c for c, _ in hits], dtype=float)
    tp = np.zeros((len(hits), 10), dtype=bool)
    tp[:, 0] = [t for _, t in hits]
    return ImageMatch(name, {"a": n_gt}, np.array(["a"] * len(hits), dtype=object), conf, tp)


def test_recording_bootstrap_is_reproducible_and_pairs_runs():
    good = [_match(f"VID_{r}_1_f{k}.jpg", 1, [(0.9, True)]) for r in range(6) for k in range(3)]
    bad = [_match(m.name, 1, [(0.9, False)]) for m in good]
    models = {"A": [good], "S": [bad, bad]}
    r1 = recording_bootstrap(models, ["a"], 50, seed=3, pairs=[("A", "S")])
    r2 = recording_bootstrap(models, ["a"], 50, seed=3, pairs=[("A", "S")])
    assert r1 == r2
    assert r1["n_recordings"] == 6 and r1["n_images"] == 18
    assert r1["models"]["A"]["map50"] == pytest.approx(1.0)
    assert r1["pairs"]["A-S"]["diff"] == pytest.approx(1.0)
    assert r1["pairs"]["A-S"]["p_le_0"] == 0.0


def test_recording_bootstrap_requires_identical_image_sets():
    a = [_match("VID_1_1_f0.jpg", 1, [(0.9, True)])]
    b = [_match("VID_2_2_f0.jpg", 1, [(0.9, True)])]
    with pytest.raises(ValueError):
        recording_bootstrap({"A": [a], "S": [b]}, ["a"], 10)


def test_map50_fast_matches_summarize():
    ms = [_match(f"VID_{i}_1_f0.jpg", 2, [(0.9, True), (0.5, False), (0.3, True)]) for i in range(4)]
    assert map50_fast(ms, ["a"]) == pytest.approx(summarize(ms, ["a"], 0.25)["map50"])


def test_ap_edge_cases():
    assert np.isnan(ap_from(np.zeros(0), np.zeros(0, dtype=bool), 0))
    assert ap_from(np.zeros(0), np.zeros(0, dtype=bool), 3) == 0.0
