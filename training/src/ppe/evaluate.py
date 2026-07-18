"""Step 6 — one shared evaluation harness for BOTH models.

For each (model, eval set) pair the same code runs the model's predictor with
identical conf/IoU, restricts predictions AND ground truth to the shared
classes (case-insensitive name intersection, recorded in metrics.md), and
computes P / R (greedy IoU matching, hand-verified in tests) and mAP50 (the
`supervision` library). Model A extras (per-class table, confusion matrix,
speed) come from Ultralytics model.val().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from ppe.config import Config
from ppe.predict import Boxes, _load_yolo_model
from ppe.remap import normalize_name

# candidate shared classes per the requirements: person, helmet, vest,
# gloves + the no_* variants; intersected at runtime with both models' names
SHARED_CANDIDATES = ["person", "helmet", "vest", "gloves",
                     "no_helmet", "no_gloves", "no_goggle"]


@dataclass
class EvalResult:
    precision: float
    recall: float
    map50: float
    n_images: int
    shared_classes: list


def shared_class_list(names_a: list, names_b: list) -> list:
    a = {normalize_name(n) for n in names_a}
    b = {normalize_name(n) for n in names_b}
    return [c for c in SHARED_CANDIDATES if c in a and c in b]


def load_eval_set(eval_set_dir: Path) -> list:
    """[(image_path, Boxes gt)] with absolute-pixel xyxy and class names."""
    eval_set_dir = Path(eval_set_dir)
    names = yaml.safe_load((eval_set_dir / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    items = []
    for img_path in sorted((eval_set_dir / "images").iterdir()):
        w, h = Image.open(img_path).size
        label = eval_set_dir / "labels" / f"{img_path.stem}.txt"
        rows, cls = [], []
        if label.is_file():
            for line in label.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h])
                cls.append(str(names[cid]))
        gt = Boxes(np.array(rows, dtype=float).reshape(-1, 4), cls,
                   np.ones(len(cls)))
        items.append((img_path, gt))
    return items


def _filter(boxes: Boxes, shared_norm: list) -> Boxes:
    keep = [i for i, n in enumerate(boxes.class_names)
            if normalize_name(n) in shared_norm]
    return Boxes(boxes.xyxy[keep].reshape(-1, 4),
                 [boxes.class_names[i] for i in keep],
                 boxes.confidence[keep])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_counts(gt: Boxes, pred: Boxes, iou_thr: float) -> tuple:
    """(tp, fp, fn): greedy confidence-descending, class-aware matching."""
    order = np.argsort(-pred.confidence) if len(pred.class_names) else []
    matched_gt: set = set()
    tp = fp = 0
    for pi in order:
        p_cls = normalize_name(pred.class_names[pi])
        best_iou, best_gi = 0.0, None
        for gi in range(len(gt.class_names)):
            if gi in matched_gt or normalize_name(gt.class_names[gi]) != p_cls:
                continue
            iou = _iou(pred.xyxy[pi], gt.xyxy[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi is not None and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            tp += 1
        else:
            fp += 1
    fn = len(gt.class_names) - len(matched_gt)
    return tp, fp, fn


def _to_sv(boxes: Boxes, shared_norm: list, with_conf: bool):
    import supervision as sv

    if not boxes.class_names:
        return sv.Detections.empty()
    ids = np.array([shared_norm.index(normalize_name(n)) for n in boxes.class_names])
    return sv.Detections(
        xyxy=boxes.xyxy.astype(float),
        class_id=ids,
        confidence=boxes.confidence.astype(float) if with_conf else None,
    )


def evaluate_predictor(predictor, eval_set_dir: Path, shared: list,
                       conf: float, iou_thr: float) -> EvalResult:
    from supervision.metrics import MeanAveragePrecision

    shared_norm = [normalize_name(s) for s in shared]
    items = load_eval_set(eval_set_dir)
    tp = fp = fn = 0
    map_metric = MeanAveragePrecision()
    for img_path, gt in tqdm(items, desc=f"[eval] {Path(eval_set_dir).name}", unit="img"):
        gt_f = _filter(gt, shared_norm)
        pred_f = _filter(predictor.predict(img_path), shared_norm)
        t, f, n = match_counts(gt_f, pred_f, iou_thr)
        tp, fp, fn = tp + t, fp + f, fn + n
        map_metric.update(_to_sv(pred_f, shared_norm, True),
                          _to_sv(gt_f, shared_norm, False))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    map50 = max(float(map_metric.compute().map50), 0.0)
    return EvalResult(precision, recall, map50, len(items), list(shared))


def _pct(x: float) -> float:
    return round(100.0 * x, 1)


def compare_all(cfg: Config, predictor_ours, predictor_generic,
                eval_sets: dict) -> dict:
    """The 6-row comparison table. eval_sets keys: fused_val,
    industrial_proxy, ocp_test."""
    shared = shared_class_list(predictor_ours.class_names,
                               predictor_generic.class_names)
    rows = [{"model": "Generic", "eval_set": "Own val (construction)",
             "P": cfg.generic_published["P"], "R": cfg.generic_published["R"],
             "mAP50": cfg.generic_published["mAP50"]}]
    plan = [
        (predictor_generic, "Generic", "Industrial proxy", "industrial_proxy"),
        (predictor_generic, "Generic", "OCP site", "ocp_test"),
        (predictor_ours, "Ours", "Fused val", "fused_val"),
        (predictor_ours, "Ours", "Industrial proxy", "industrial_proxy"),
        (predictor_ours, "Ours", "OCP site", "ocp_test"),
    ]
    for predictor, model, label, key in plan:
        print(f"[eval] {model} on {label} ({key})…")
        r = evaluate_predictor(predictor, eval_sets[key], shared,
                               cfg.eval_conf, cfg.eval_iou)
        rows.append({"model": model, "eval_set": label, "P": _pct(r.precision),
                     "R": _pct(r.recall), "mAP50": _pct(r.map50)})
    return {"rows": rows, "shared_classes": shared,
            "conf": cfg.eval_conf, "iou": cfg.eval_iou}


def cameras_at_fps(total_ms_per_frame: float, fps: int) -> int:
    return math.floor((1000.0 / total_ms_per_frame) / fps)


def _images_per_class(data_yaml: Path) -> dict:
    data = yaml.safe_load(Path(data_yaml).read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    root = Path(data.get("path", Path(data_yaml).parent))
    labels_dir = root / str(data["val"]).replace("images", "labels")
    counts: dict = {}
    total = 0
    for lf in sorted(labels_dir.glob("*.txt")):
        total += 1
        present = {int(line.split()[0]) for line in lf.read_text().splitlines() if line.strip()}
        for cid in present:
            counts[str(names[cid])] = counts.get(str(names[cid]), 0) + 1
    counts["__total__"] = total
    return counts


def model_a_val(cfg: Config, best_pt: Path, data_yaml: Path) -> dict:
    """Ultralytics val extras: per-class table, speed block, confusion matrix."""
    model = _load_yolo_model(best_pt)
    metrics = model.val(data=str(data_yaml), split="val", plots=True)
    names = model.names
    box = metrics.box
    img_counts = _images_per_class(data_yaml)

    per_class = []
    seen = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        seen[str(names[int(cls_idx)])] = i
    for cid in sorted(names):
        cname = str(names[cid])
        if cname in seen:
            i = seen[cname]
            row = {"class": cname, "P": _pct(float(box.p[i])),
                   "R": _pct(float(box.r[i])), "mAP50": _pct(float(box.ap50[i]))}
        else:
            row = {"class": cname, "P": 0.0, "R": 0.0, "mAP50": 0.0}
        row["images"] = img_counts.get(cname, 0)
        per_class.append(row)
    per_class.append({"class": "ALL", "P": _pct(float(box.mp)),
                      "R": _pct(float(box.mr)), "mAP50": _pct(float(box.map50)),
                      "images": img_counts.get("__total__", 0)})

    speed = {k: round(float(metrics.speed[k]), 2)
             for k in ("preprocess", "inference", "postprocess")}
    speed["total"] = round(sum(speed.values()), 2)
    return {
        "per_class": per_class,
        "speed": speed,
        "cameras": cameras_at_fps(speed["total"], cfg.camera_fps),
        "camera_fps": cfg.camera_fps,
        "camera_formula": "cameras = floor((1000 / total_ms_per_frame) / fps)",
        "confusion_matrix_png": str(Path(metrics.save_dir) / "confusion_matrix_normalized.png"),
    }
