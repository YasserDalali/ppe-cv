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
import re
import statistics
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
                coords = [float(v) for v in parts[1:]]
                if len(coords) == 4:
                    cx, cy, bw, bh = coords
                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2
                else:
                    # YOLO segmentation polygon (Roboflow instance-seg
                    # exports, e.g. OCP: 'cls x1 y1 x2 y2 ... xn yn').
                    # Bounding box = min/max over vertices, same conversion
                    # Ultralytics' own loader applies via segments2boxes()
                    # for train/val — otherwise eval GT is built from just
                    # the first two polygon vertices misread as (cx,cy,w,h).
                    xs, ys = coords[0::2], coords[1::2]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                rows.append([x1 * w, y1 * h, x2 * w, y2 * h])
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


def best_matches(predictor, eval_set_dir: Path, shared: list, image_names: list | None = None,
                 max_images: int = 4) -> list:
    """For up to max_images images (optionally restricted to specific
    filenames), return (img_path, gt, pred, matches) where matches is, for
    each ground-truth box, the best-matching predicted box index (or None)
    and the IoU between them, ignoring class — the raw material for a visual
    GT-vs-prediction sanity check (draw_gt_vs_pred)."""
    shared_norm = [normalize_name(s) for s in shared]
    items = load_eval_set(eval_set_dir)
    if image_names is not None:
        wanted = set(image_names)
        items = [(p, g) for p, g in items if p.name in wanted]
        # preserve the caller's requested order rather than eval-set order
        items.sort(key=lambda pg: image_names.index(pg[0].name))
    items = items[:max_images]

    out = []
    for img_path, gt in items:
        gt_f = _filter(gt, shared_norm)
        pred_f = _filter(predictor.predict(img_path), shared_norm)
        matches = []
        for gi in range(len(gt_f.class_names)):
            best_iou, best_pi = 0.0, None
            for pi in range(len(pred_f.class_names)):
                iou = _iou(gt_f.xyxy[gi], pred_f.xyxy[pi])
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            matches.append((gi, best_pi, round(best_iou, 3)))
        out.append((img_path, gt_f, pred_f, matches))
    return out


def draw_gt_vs_pred(img_path: Path, gt: Boxes, pred: Boxes, matches: list):
    """PIL image with ground-truth boxes in green (labeled with class and
    its best-matching IoU) and every filtered prediction in red (labeled
    with class and confidence) — a visual check for whether low scores come
    from a coordinate/scale bug (boxes nowhere near each other) or genuine
    localization imprecision (boxes overlapping but not tightly)."""
    from PIL import ImageDraw

    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    draw = ImageDraw.Draw(im)
    iou_by_gt = {gi: iou for gi, _, iou in matches}
    for gi in range(len(gt.class_names)):
        x1, y1, x2, y2 = gt.xyxy[gi]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        draw.text((x1, max(0, y1 - 12)),
                  f"GT:{gt.class_names[gi]} iou={iou_by_gt.get(gi, 0.0):.2f}", fill=(0, 200, 0))
    for pi in range(len(pred.class_names)):
        x1, y1, x2, y2 = pred.xyxy[pi]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        draw.text((x1, y2 + 2), f"pred:{pred.class_names[pi]} {pred.confidence[pi]:.2f}",
                  fill=(220, 0, 0))
    return im


_VIDEO_ID_RE = re.compile(r"(VID_\d+_\d+)_f\d+")


def _video_id(image_name: str) -> str:
    """Best-effort video/session id parsed from an OCP frame filename like
    'ocp__VID_20260716_163549_f00211_jpg.rf.<hash>.jpg' ->
    'VID_20260716_163549'. Falls back to the full filename for anything that
    doesn't match (non-OCP sources, or a naming scheme change), so those
    still group sensibly (each alone) instead of erroring."""
    m = _VIDEO_ID_RE.search(image_name)
    return m.group(1) if m else image_name


def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(acx - bcx, acy - bcy)


@dataclass
class AlignmentRecord:
    """The geometric relationship between one ground-truth box and its
    closest prediction (by IoU, ties broken by center distance), ignoring
    class. Offsets are normalized by image width/height so they're
    comparable across images of different resolutions; scale is the simple
    predicted/ground-truth size ratio per axis."""
    image: str
    video: str
    gt_class: str
    iou: float
    dx_norm: float
    dy_norm: float
    scale_w: float
    scale_h: float
    exif_orientation: int | None


def alignment_stats(predictor, eval_set_dir: Path, shared: list) -> list:
    """Per-GT-box offset/scale relative to the closest prediction on the
    same image, across an eval set — the raw material for telling apart two
    stories behind a "boxes overlap but never tightly" pattern:

    - GENUINE localization imprecision (e.g. compressed/blurry video
      frames): offsets and scale ratios scatter with no consistent
      direction or magnitude, including frame-to-frame within the same
      video.
    - A SOURCE-SIDE coordinate bug (annotations exported at a different
      resolution/crop than the images now being scored): offsets/scale
      cluster tightly around a roughly constant, often non-zero value —
      because the same misregistration applies to every box on a
      misregistered image, or every image from one export batch.

    Images with zero raw predictions are skipped (nothing to compare
    against); see diagnose_eval_set for that failure mode instead.
    """
    shared_norm = [normalize_name(s) for s in shared]
    items = load_eval_set(eval_set_dir)
    out = []
    for img_path, gt in items:
        with Image.open(img_path) as im:
            width, height = ImageOps.exif_transpose(im).size
            orientation = im.getexif().get(0x0112)
        gt_f = _filter(gt, shared_norm)
        pred_f = _filter(predictor.predict(img_path), shared_norm)
        if not len(pred_f.class_names):
            continue
        for gi in range(len(gt_f.class_names)):
            g = gt_f.xyxy[gi]
            best_iou, best_pi, best_dist = 0.0, None, math.inf
            for pi in range(len(pred_f.class_names)):
                p = pred_f.xyxy[pi]
                iou = _iou(g, p)
                dist = _center_dist(g, p)
                if best_pi is None or iou > best_iou or (iou == best_iou and dist < best_dist):
                    best_iou, best_pi, best_dist = iou, pi, dist
            p = pred_f.xyxy[best_pi]
            gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            gw, gh = g[2] - g[0], g[3] - g[1]
            pcx, pcy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            pw, ph = p[2] - p[0], p[3] - p[1]
            out.append(AlignmentRecord(
                image=img_path.name, video=_video_id(img_path.name),
                gt_class=gt_f.class_names[gi], iou=round(best_iou, 3),
                dx_norm=round((pcx - gcx) / width, 4),
                dy_norm=round((pcy - gcy) / height, 4),
                scale_w=round(pw / gw, 3) if gw > 0 else float("nan"),
                scale_h=round(ph / gh, 3) if gh > 0 else float("nan"),
                exif_orientation=orientation,
            ))
    return out


def _mean_std(xs: list) -> tuple:
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), statistics.pstdev(xs)


def print_alignment_summary(records: list) -> None:
    """Aggregate + per-video breakdown of alignment_stats() output, with a
    verdict aimed at the genuine-imprecision vs. source-side-coordinate-bug
    question: if the offset is tight WITHIN a video but differs ACROSS
    videos, that points at something constant per export/recording session
    (source-side) rather than per-frame noise (localization imprecision)."""
    if not records:
        print("No alignment records (no images had both GT and predictions).")
        return

    fields = ("dx_norm", "dy_norm", "scale_w", "scale_h")
    print(f"=== alignment summary over {len(records)} matched GT boxes ===")
    overall = {f: _mean_std([getattr(r, f) for r in records]) for f in fields}
    for f in fields:
        mean, std = overall[f]
        print(f"  overall {f}: mean={mean:+.4f}  std={std:.4f}")

    by_video: dict = {}
    for r in records:
        by_video.setdefault(r.video, []).append(r)
    multi = {v: rs for v, rs in by_video.items() if len(rs) >= 2}
    print(f"\n{len(by_video)} distinct video/session groups "
          f"({len(multi)} with >=2 matched boxes)")
    for video, rs in sorted(by_video.items()):
        stats = {f: _mean_std([getattr(r, f) for r in rs]) for f in fields}
        summary = "  ".join(f"{f}={stats[f][0]:+.3f}±{stats[f][1]:.3f}" for f in fields)
        print(f"  {video:30s} n={len(rs):3d}  {summary}")

    if multi:
        within_std = {f: statistics.fmean([_mean_std([getattr(r, f) for r in rs])[1]
                                           for rs in multi.values()]) for f in fields}
        print("\nmean WITHIN-video std vs. OVERALL (cross-video) std "
              "(within << overall -> per-video/export constant, points at a "
              "source-side bug; within ~= overall -> per-frame noise, points "
              "at genuine localization imprecision):")
        for f in fields:
            print(f"  {f}: within={within_std[f]:.4f}  overall={overall[f][1]:.4f}")


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
