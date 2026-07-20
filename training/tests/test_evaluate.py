import numpy as np
import yaml
from PIL import Image

from conftest import write_png
from ppe.config import CANONICAL_CLASSES, Config
from ppe.evaluate import (
    SHARED_CANDIDATES,
    cameras_at_fps,
    compare_all,
    evaluate_predictor,
    load_eval_set,
    match_counts,
    model_a_val,
    shared_class_list,
)
from ppe.predict import Boxes


def boxes(rows):
    """rows: list of (x1,y1,x2,y2,name,conf)."""
    if not rows:
        return Boxes.empty()
    return Boxes(np.array([r[:4] for r in rows], dtype=float),
                 [r[4] for r in rows],
                 np.array([r[5] for r in rows], dtype=float))


def make_eval_set(root, images: dict):
    """images: {stem: [(cls_id, cx, cy, w, h), ...]} on 32x32 canvases."""
    for stem, bxs in images.items():
        write_png(root / "images" / f"{stem}.png")
        (root / "labels").mkdir(parents=True, exist_ok=True)
        lines = [f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in bxs]
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump(
        {"nc": len(CANONICAL_CLASSES), "names": CANONICAL_CLASSES}))
    return root


class StubPredictor:
    def __init__(self, class_names, mapping):
        self.class_names = class_names
        self.mapping = mapping  # {image stem: Boxes}

    def predict(self, image_path):
        return self.mapping.get(image_path.stem, Boxes.empty())


def test_shared_class_list_case_insensitive_ordered():
    ours = ["Person", "helmet", "vest", "gloves", "boots", "goggles",
            "no_helmet", "no_gloves", "no_goggle", "gas mask"]
    theirs = ["gloves", "helmet", "no-gloves", "no-helmet", "no-shoes",
              "no-vest", "person", "shoes", "vest"]
    assert shared_class_list(ours, theirs) == \
        ["person", "helmet", "vest", "gloves", "no_helmet", "no_gloves"]
    assert shared_class_list(["boots"], ["shoes"]) == []
    assert SHARED_CANDIDATES[0] == "person"


def test_match_counts_hand_computed():
    gt = boxes([(0, 0, 10, 10, "Person", 1.0), (20, 20, 30, 30, "helmet", 1.0)])
    pred = boxes([(0, 0, 10, 11, "person", 0.9),      # IoU ~0.91 -> TP
                  (50, 50, 60, 60, "person", 0.8)])   # FP
    tp, fp, fn = match_counts(gt, pred, iou_thr=0.5)
    assert (tp, fp, fn) == (1, 1, 1)


def test_match_counts_is_class_aware():
    gt = boxes([(0, 0, 10, 10, "Person", 1.0)])
    pred = boxes([(0, 0, 10, 10, "helmet", 0.9)])
    assert match_counts(gt, pred, 0.5) == (0, 1, 1)


def test_match_counts_greedy_one_match_per_gt():
    gt = boxes([(0, 0, 10, 10, "Person", 1.0)])
    pred = boxes([(0, 0, 10, 10, "person", 0.6), (1, 0, 11, 10, "person", 0.9)])
    assert match_counts(gt, pred, 0.5) == (1, 1, 0)


def test_cameras_at_fps():
    assert cameras_at_fps(20.0, 5) == 10
    assert cameras_at_fps(33.3, 5) == 6


def test_load_eval_set_denormalizes(tmp_path):
    d = make_eval_set(tmp_path / "es", {"a": [(0, 0.5, 0.5, 0.5, 0.5)]})
    items = load_eval_set(d)
    assert len(items) == 1
    img, gt = items[0]
    assert img.stem == "a"
    assert gt.xyxy.tolist() == [[8.0, 8.0, 24.0, 24.0]]
    assert gt.class_names == ["Person"]


def test_load_eval_set_respects_exif_orientation(tmp_path):
    """Regression test: camera/CCTV photos (e.g. OCP) can carry an EXIF
    orientation tag where the stored pixel grid is portrait but the tag says
    'rotate 90 deg to display' (landscape). Annotation tools label the
    displayed (rotated) image, so ground-truth boxes are normalized against
    the rotated dimensions — load_eval_set must denormalize against those
    same rotated dimensions, not the raw on-disk pixel grid."""
    es = tmp_path / "es"
    (es / "images").mkdir(parents=True)
    (es / "labels").mkdir(parents=True)

    # Stored 20x40 (portrait); orientation tag 6 means "rotate 90 CW to
    # display", so the correctly-oriented (annotated) image is 40x20.
    im = Image.new("RGB", (20, 40), (127, 127, 127))
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(es / "images" / "a.jpg", exif=exif)

    # Box centered at the right-hand edge of the 40x20 displayed image.
    (es / "labels" / "a.txt").write_text("0 0.9 0.5 0.2 0.5\n")
    (es / "data.yaml").write_text(yaml.safe_dump(
        {"nc": len(CANONICAL_CLASSES), "names": CANONICAL_CLASSES}))

    items = load_eval_set(es)
    assert len(items) == 1
    _, gt = items[0]
    # Denormalized against the 40x20 displayed frame: x in [32, 40], y in [5, 15].
    # (Denormalizing against the raw, un-rotated 20x40 pixel grid instead
    # would give the wrong box [16, 10, 20, 30] — this is what the bug did.)
    assert gt.xyxy.tolist() == [[32.0, 5.0, 40.0, 15.0]]


def test_evaluate_perfect_and_empty_predictor(tmp_path):
    d = make_eval_set(tmp_path / "es", {
        "a": [(0, 0.5, 0.5, 0.5, 0.5), (1, 0.25, 0.25, 0.2, 0.2)],
        "b": [(4, 0.5, 0.5, 0.4, 0.4)],   # boots: NOT in shared -> ignored entirely
    })
    shared = ["person", "helmet"]
    perfect = StubPredictor(CANONICAL_CLASSES, {
        stem: Boxes(gt.xyxy, gt.class_names, np.full(len(gt.class_names), 0.9))
        for stem, (img, gt) in ((i.stem, (i, g)) for i, g in load_eval_set(d))
    })
    res = evaluate_predictor(perfect, d, shared, conf=0.25, iou_thr=0.5)
    assert res.precision == 1.0 and res.recall == 1.0
    assert res.map50 > 0.99
    assert res.n_images == 2

    empty = StubPredictor(CANONICAL_CLASSES, {})
    res0 = evaluate_predictor(empty, d, shared, conf=0.25, iou_thr=0.5)
    assert res0.precision == 0.0 and res0.recall == 0.0 and res0.map50 == 0.0


def test_compare_all_six_rows(tmp_path):
    sets = {
        name: make_eval_set(tmp_path / name, {f"{name}_a": [(0, 0.5, 0.5, 0.5, 0.5)]})
        for name in ("fused_val", "industrial_proxy", "ocp_test")
    }
    gt_boxes = {}
    for d in sets.values():
        for img, gt in load_eval_set(d):
            gt_boxes[img.stem] = Boxes(gt.xyxy, gt.class_names,
                                       np.full(len(gt.class_names), 0.9))
    ours = StubPredictor(CANONICAL_CLASSES, gt_boxes)
    generic = StubPredictor(["person", "helmet", "vest", "gloves"], {})
    cfg = Config(drive_root=tmp_path / "dr", work_root=tmp_path / "wk")
    out = compare_all(cfg, ours, generic, sets)
    rows = out["rows"]
    assert len(rows) == 6
    assert rows[0] == {"model": "Generic", "eval_set": "Own val (construction)",
                       "P": 58.1, "R": 64.4, "mAP50": 59.1}
    assert [r["model"] for r in rows] == ["Generic"] * 3 + ["Ours"] * 3
    ours_val = next(r for r in rows if r["model"] == "Ours" and r["eval_set"] == "Fused val")
    assert ours_val["P"] == 100.0 and ours_val["R"] == 100.0
    assert out["shared_classes"] == ["person", "helmet", "vest", "gloves"]


class FakeBoxMetrics:
    def __init__(self, n):
        self.p = np.full(n, 0.8)
        self.r = np.full(n, 0.7)
        self.ap50 = np.full(n, 0.75)
        self.ap_class_index = np.arange(n)
        self.mp, self.mr, self.map50 = 0.8, 0.7, 0.75


class FakeValMetrics:
    def __init__(self, save_dir, n):
        self.box = FakeBoxMetrics(n)
        self.speed = {"preprocess": 1.0, "inference": 15.0, "postprocess": 4.0}
        self.save_dir = str(save_dir)


class FakeValModel:
    def __init__(self, save_dir):
        self.names = dict(enumerate(CANONICAL_CLASSES))
        self.save_dir = save_dir

    def val(self, **kwargs):
        save = self.save_dir
        save.mkdir(parents=True, exist_ok=True)
        (save / "confusion_matrix_normalized.png").write_bytes(b"png")
        return FakeValMetrics(save, 3)  # only 3 classes had val instances


def test_model_a_val_extras(tmp_path, monkeypatch):
    dataset = tmp_path / "fused"
    make_eval_set(dataset / "val", {"v1": [(0, 0.5, 0.5, 0.5, 0.5), (1, 0.2, 0.2, 0.1, 0.1)],
                                    "v2": [(0, 0.3, 0.3, 0.2, 0.2)]})
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset), "train": "train/images", "val": "val/images",
        "test": "test/images", "nc": 10, "names": CANONICAL_CLASSES}))
    monkeypatch.setattr("ppe.evaluate._load_yolo_model",
                        lambda w: FakeValModel(tmp_path / "valrun"))
    cfg = Config(drive_root=tmp_path / "dr", work_root=tmp_path / "wk")
    out = model_a_val(cfg, tmp_path / "best.pt", data_yaml)
    assert len(out["per_class"]) == 11  # 10 classes + ALL
    by_class = {r["class"]: r for r in out["per_class"]}
    assert by_class["Person"]["P"] == 80.0 and by_class["Person"]["images"] == 2
    assert by_class["helmet"]["images"] == 1
    assert by_class["gas mask"]["mAP50"] == 0.0   # absent from val -> zero row
    assert by_class["ALL"]["mAP50"] == 75.0 and by_class["ALL"]["images"] == 2
    assert out["speed"]["total"] == 20.0
    assert out["cameras"] == 10 and "1000" in out["camera_formula"]
    assert out["confusion_matrix_png"].endswith("confusion_matrix_normalized.png")
