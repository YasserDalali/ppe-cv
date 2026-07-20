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
from PIL import Image, ImageOps
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
        # exif_transpose: plain .size reads raw sensor pixel dimensions, which
        # disagree with the label file (annotated on the EXIF-rotated, on-
        # screen orientation) for any source with camera/CCTV EXIF orientation
        # tags — e.g. OCP. Training and model.val() already correct for this
        # via Ultralytics' own loaders; this was the one path that didn't.
        with Image.open(img_path) as im:
            w, h = ImageOps.exif_transpose(im).size
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


@dataclass
class ImageDiagnostic:
    """Per-image forensics for one predictor on one eval set: how many boxes
    the model actually produced (before any class filtering) versus how many
    ground-truth boxes exist, the best spatial overlap ignoring class (to
    tell 'wrong class' apart from 'no spatial overlap at all'), and how long
    the predict() call took (a silently-failing/short-circuiting predictor
    tends to be suspiciously fast)."""
    image: str
    width: int
    height: int
    exif_rotated: bool
    n_gt_shared: int
    n_pred_raw: int
    n_pred_shared: int
    pred_classes_seen: list
    best_iou_ignore_class: float
    predict_seconds: float
    error: str | None = None


def diagnose_eval_set(predictor, eval_set_dir: Path, shared: list,
                      iou_thr: float, max_images: int | None = None) -> list:
    """Cheap, single-predictor, single-eval-set forensic dump — meant to run
    on a small set (e.g. ocp_test, ~50 images, seconds) before committing to
    the full multi-set compare_all() sweep (minutes). Surfaces the three
    failure modes that all look like 'near-zero score' from the outside but
    need different fixes: (1) the model predicts nothing at all on these
    images, (2) it predicts plenty but nothing spatially overlaps the ground
    truth (a coordinate/orientation/scale bug), (3) boxes do overlap but the
    predicted class never matches the ground-truth class (a class-mapping
    bug)."""
    import time

    shared_norm = [normalize_name(s) for s in shared]
    items = load_eval_set(eval_set_dir)
    if max_images is not None:
        items = items[:max_images]
    out = []
    for img_path, gt in items:
        with Image.open(img_path) as im:
            raw_size = im.size
            exif_size = ImageOps.exif_transpose(im).size
        # report the corrected size: that's what the GT boxes above were
        # denormalized against (load_eval_set), so it's the one that should
        # agree with what the predictor's own coordinates are in.
        width, height = exif_size
        gt_f = _filter(gt, shared_norm)
        t0 = time.perf_counter()
        err = None
        try:
            pred = predictor.predict(img_path)
        except Exception as exc:  # noqa: BLE001 — diagnostic path, report don't crash
            pred = Boxes.empty()
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        pred_f = _filter(pred, shared_norm)
        best_iou = 0.0
        for g in gt_f.xyxy:
            for p in pred_f.xyxy:
                best_iou = max(best_iou, _iou(g, p))
        out.append(ImageDiagnostic(
            image=img_path.name, width=width, height=height,
            exif_rotated=raw_size != exif_size,
            n_gt_shared=len(gt_f.class_names), n_pred_raw=len(pred.class_names),
            n_pred_shared=len(pred_f.class_names),
            pred_classes_seen=sorted(set(pred.class_names)),
            best_iou_ignore_class=round(best_iou, 3),
            predict_seconds=round(dt, 4), error=err,
        ))
    return out


def print_diagnostics(infos: list) -> None:
    """Human-readable dump of diagnose_eval_set() output, plus an aggregate
    verdict pointing at which failure mode (if any) is in play."""
    for d in infos:
        flags = []
        if d.error:
            flags.append("ERROR")
        if d.n_gt_shared and d.n_pred_raw == 0:
            flags.append("NO PREDICTIONS AT ALL")
        elif d.n_gt_shared and d.best_iou_ignore_class < 0.1 and d.n_pred_raw > 0:
            flags.append("PREDICTIONS EXIST BUT DON'T OVERLAP GT")
        flag_str = f"  <-- {', '.join(flags)}" if flags else ""
        print(f"{d.image:45s} {d.width:4d}x{d.height:<4d} exif_rotated={str(d.exif_rotated):5s} "
              f"gt={d.n_gt_shared:2d} pred_raw={d.n_pred_raw:3d} pred_shared={d.n_pred_shared:2d} "
              f"best_iou={d.best_iou_ignore_class:.3f} {d.predict_seconds*1000:6.1f}ms{flag_str}")
        if d.error:
            print(f"    ERROR: {d.error}")

    n = len(infos)
    n_with_gt = sum(1 for d in infos if d.n_gt_shared)
    n_zero_pred = sum(1 for d in infos if d.n_pred_raw == 0)
    n_no_overlap = sum(1 for d in infos if d.n_gt_shared and d.n_pred_raw > 0
                       and d.best_iou_ignore_class < 0.1)
    n_errors = sum(1 for d in infos if d.error)
    avg_ms = sum(d.predict_seconds for d in infos) / n * 1000 if n else 0.0
    n_exif = sum(1 for d in infos if d.exif_rotated)
    print(f"\n{n} images ({n_with_gt} with ground truth in the shared classes), "
          f"avg predict() time {avg_ms:.1f}ms")
    print(f"  {n_exif} images had an EXIF orientation tag requiring correction")
    print(f"  {n_zero_pred} images produced ZERO raw predictions from the model")
    print(f"  {n_no_overlap} images had predictions that never spatially overlap "
          f"any ground-truth box (best IoU < 0.1) — points at a coordinate/scale/"
          f"orientation bug rather than a classification bug")
    print(f"  {n_errors} images raised an exception during predict()")


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
