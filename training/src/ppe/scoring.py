"""2026-10 revision — ONE scoring harness for every number in the paper.

Each (model, eval set) gets one prediction pass at a low confidence floor
(cfg.conf_floor), cached as JSON. Everything is then computed from that
cache with the same code:

- AP / mAP50 / mAP50-95: COCO-style 101-point interpolation, class-aware
  greedy matching (highest-IoU unmatched ground truth at each IoU threshold),
  classes without ground truth in the set excluded from the mean.
- P / R / F1 at the operating threshold (cfg.op_conf), pooled over classes.
- A precision-recall sweep and the best-F1 threshold (R1-4).
- A bootstrap over recordings, not frames (R4-5).

Fixes the old mismatch where Table 2 (conf 0.25, supervision) and Table 3
(conf 0.001, Ultralytics) scored the same model differently (R4-M2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ppe.config import Config
from ppe.evaluate import _filter, load_eval_set
from ppe.pool import recording_id
from ppe.predict import Boxes, UltralyticsPredictor
from ppe.remap import normalize_name

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
SWEEP = np.round(np.arange(0.05, 0.951, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------- caching

def cache_predictions(weights: Path, eval_dir: Path, out_json: Path, cfg: Config) -> dict:
    """One predict pass over an eval set at the confidence floor."""
    predictor = UltralyticsPredictor(weights, conf=cfg.conf_floor, iou=cfg.nms_iou,
                                     imgsz=cfg.imgsz)
    out = {"weights": str(weights), "conf": cfg.conf_floor, "nms_iou": cfg.nms_iou,
           "imgsz": cfg.imgsz, "images": {}}
    images = sorted((Path(eval_dir) / "images").iterdir())
    for img in tqdm(images, desc=f"[predict] {Path(eval_dir).name}", unit="img"):
        b = predictor.predict(img)
        out["images"][img.name] = {
            "xyxy": np.round(b.xyxy, 2).tolist(),
            "cls": list(b.class_names),
            "conf": [round(float(c), 5) for c in b.confidence],
        }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_name(out_json.name + ".partial")
    tmp.write_text(json.dumps(out))
    tmp.replace(out_json)
    return out


def boxes_from_cache(entry: dict) -> Boxes:
    if not entry["cls"]:
        return Boxes.empty()
    return Boxes(np.array(entry["xyxy"], dtype=float).reshape(-1, 4), list(entry["cls"]),
                 np.array(entry["conf"], dtype=float))


# ---------------------------------------------------------------- matching

@dataclass
class ImageMatch:
    name: str
    gt: dict              # normalized class -> number of ground-truth boxes
    cls: np.ndarray       # (P,) normalized predicted class, confidence-descending
    conf: np.ndarray      # (P,)
    tp: np.ndarray        # (P, len(IOU_THRESHOLDS)) bool


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(-1, 4), b.reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_image(name: str, gt: Boxes, pred: Boxes, scope: list[str]) -> ImageMatch:
    scope_norm = [normalize_name(c) for c in scope]
    gt_f, pred_f = _filter(gt, scope_norm), _filter(pred, scope_norm)
    gcls = np.array([normalize_name(c) for c in gt_f.class_names], dtype=object)
    order = np.argsort(-pred_f.confidence, kind="stable")
    pcls = np.array([normalize_name(pred_f.class_names[i]) for i in order], dtype=object)
    conf = pred_f.confidence[order].astype(float) if len(order) else np.zeros(0)
    tp = np.zeros((len(order), len(IOU_THRESHOLDS)), dtype=bool)
    if len(order) and len(gcls):
        ious = iou_matrix(pred_f.xyxy[order], gt_f.xyxy)
        same = pcls[:, None] == gcls[None, :]
        for ti, t in enumerate(IOU_THRESHOLDS):
            matched = np.zeros(len(gcls), dtype=bool)
            for k in range(len(order)):
                cand = np.flatnonzero(same[k] & ~matched & (ious[k] >= t))
                if len(cand):
                    g = cand[np.argmax(ious[k, cand])]
                    matched[g] = True
                    tp[k, ti] = True
    gt_count: dict = {}
    for c in gcls:
        gt_count[c] = gt_count.get(c, 0) + 1
    return ImageMatch(name, gt_count, pcls, conf, tp)


def match_cache(items: list, cache: dict, scope: list[str]) -> list[ImageMatch]:
    """items: load_eval_set() output; cache: cache_predictions() output (or a
    merged dict with an 'images' map covering every item)."""
    out = []
    for img_path, gt in items:
        entry = cache["images"].get(img_path.name)
        if entry is None:
            raise KeyError(f"no cached prediction for {img_path.name}")
        out.append(match_image(img_path.name, gt, boxes_from_cache(entry), scope))
    return out


# ---------------------------------------------------------------- metrics

def ap_from(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP for one class at one IoU threshold."""
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-conf, kind="mergesort")
    tps = tp[order].astype(float)
    tpc, fpc = np.cumsum(tps), np.cumsum(1.0 - tps)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.zeros(len(RECALL_POINTS))
    ok = idx < len(precision)
    q[ok] = precision[idx[ok]]
    return float(q.mean())


def _gather(matches: list[ImageMatch], cls: str) -> tuple[np.ndarray, np.ndarray, int]:
    confs, tps, n_gt = [], [], 0
    for m in matches:
        n_gt += m.gt.get(cls, 0)
        sel = m.cls == cls
        if sel.any():
            confs.append(m.conf[sel])
            tps.append(m.tp[sel])
    conf = np.concatenate(confs) if confs else np.zeros(0)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    return conf, tp, n_gt


def _prf(tp_n: float, n_pred: float, n_gt: float) -> tuple[float, float, float]:
    p = tp_n / n_pred if n_pred else 0.0
    r = tp_n / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def summarize(matches: list[ImageMatch], scope: list[str], op_conf: float) -> dict:
    """All metrics for one (model, eval set, class scope). Classes in scope
    with no ground truth in this set are listed but excluded from means."""
    scope_norm = [normalize_name(c) for c in scope]
    per_class = {}
    all_conf, all_tp50, total_gt = [], [], 0
    for c in scope_norm:
        conf, tp, n_gt = _gather(matches, c)
        aps = [ap_from(conf, tp[:, ti], n_gt) for ti in range(len(IOU_THRESHOLDS))]
        sel = conf >= op_conf
        p, r, f1 = _prf(float(tp[sel, 0].sum()), float(sel.sum()), n_gt)
        per_class[c] = {"n_gt": n_gt, "n_pred_at_op": int(sel.sum()),
                        "ap50": aps[0], "ap50_95": float(np.mean(aps)) if n_gt else float("nan"),
                        "P": p, "R": r, "F1": f1}
        all_conf.append(conf)
        all_tp50.append(tp[:, 0])
        total_gt += n_gt
    present = [c for c in scope_norm if per_class[c]["n_gt"] > 0]
    conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
    tp50 = np.concatenate(all_tp50) if all_tp50 else np.zeros(0, dtype=bool)

    def at(t: float) -> dict:
        sel = conf >= t
        p, r, f1 = _prf(float(tp50[sel].sum()), float(sel.sum()), total_gt)
        return {"conf": float(t), "P": p, "R": r, "F1": f1}

    curve = [at(t) for t in SWEEP]
    best = max(curve, key=lambda row: row["F1"]) if curve else at(op_conf)
    op = at(op_conf)
    return {
        "scope": scope_norm, "classes": present, "n_images": len(matches),
        "n_recordings": len({recording_id(m.name) for m in matches}),
        "map50": float(np.mean([per_class[c]["ap50"] for c in present])) if present else float("nan"),
        "map50_95": float(np.mean([per_class[c]["ap50_95"] for c in present])) if present else float("nan"),
        "P": op["P"], "R": op["R"], "F1": op["F1"], "op_conf": op_conf,
        "per_class": per_class, "pr_curve": curve, "best_f1": best,
    }


def map50_fast(matches: list[ImageMatch], classes: list[str]) -> float:
    aps = []
    for c in classes:
        conf, tp, n_gt = _gather(matches, c)
        if n_gt:
            aps.append(ap_from(conf, tp[:, 0], n_gt))
    return float(np.mean(aps)) if aps else float("nan")


def recording_bootstrap(models: dict, classes: list[str], n_boot: int, seed: int = 0,
                        pairs: list | None = None) -> dict:
    """models: name -> list of runs, each run a list[ImageMatch] over the SAME
    images. Resamples whole recordings with replacement; a model's value per
    resample is the mean over its runs (seeds). Returns point estimates, 95%
    CIs, and paired-difference CIs for ``pairs`` [(a, b), ...]."""
    names = None
    for runs in models.values():
        for run in runs:
            run_names = sorted(m.name for m in run)
            if names is None:
                names = run_names
            elif run_names != names:
                raise ValueError("bootstrap needs every run scored on the same images")
    by_name = {k: [{m.name: m for m in run} for run in runs] for k, runs in models.items()}
    groups: dict = {}
    for n in names:
        groups.setdefault(recording_id(n), []).append(n)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    point = {k: float(np.mean([map50_fast(run, classes) for run in runs]))
             for k, runs in models.items()}
    samples = {k: np.empty(n_boot) for k in models}
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        chosen = [n for i in pick for n in groups[keys[i]]]
        for k, runs in by_name.items():
            samples[k][b] = np.nanmean([map50_fast([run[n] for n in chosen], classes)
                                        for run in runs])
    out = {"classes": classes, "n_boot": n_boot, "n_recordings": len(keys),
           "n_images": len(names), "models": {}, "pairs": {}}
    for k in models:
        lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
        out["models"][k] = {"map50": point[k], "ci95": [float(lo), float(hi)]}
    for a, b in pairs or []:
        if a in models and b in models:
            diff = samples[a] - samples[b]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            out["pairs"][f"{a}-{b}"] = {"diff": point[a] - point[b], "ci95": [float(lo), float(hi)],
                                        "p_le_0": float(np.mean(diff <= 0))}
    return out


def confusion_matrix(items: list, cache: dict, classes: list[str], conf: float,
                     iou_thr: float) -> dict:
    """Class confusion at an operating threshold. Rows = true class, columns =
    predicted class; the extra last row/column is background (last row:
    detections matching no object = false alarms; last column: objects with
    no detection = misses). Matching is one-to-one by highest IoU regardless
    of class, so a wrong-class detection counts once, as a confusion."""
    names = [normalize_name(c) for c in classes]
    idx = {c: i for i, c in enumerate(names)}
    bg = len(names)
    m = np.zeros((bg + 1, bg + 1), dtype=int)
    for img_path, gt in items:
        gt_f = _filter(gt, names)
        pred = _filter(boxes_from_cache(cache["images"][img_path.name]), names)
        keep = pred.confidence >= conf
        p_xyxy = pred.xyxy[keep]
        p_cls = [idx[normalize_name(c)] for c, k in zip(pred.class_names, keep) if k]
        g_cls = [idx[normalize_name(c)] for c in gt_f.class_names]
        matched_g, matched_p = set(), set()
        if g_cls and p_cls:
            ious = iou_matrix(gt_f.xyxy, p_xyxy)
            pairs = sorted(((ious[g, p], g, p) for g, p in zip(*np.nonzero(ious >= iou_thr))),
                           reverse=True)
            for _, g, p in pairs:
                if g in matched_g or p in matched_p:
                    continue
                matched_g.add(g)
                matched_p.add(p)
                m[g_cls[g], p_cls[p]] += 1
        for g, c in enumerate(g_cls):
            if g not in matched_g:
                m[c, bg] += 1
        for p, c in enumerate(p_cls):
            if p not in matched_p:
                m[bg, c] += 1
    return {"labels": names + ["background"], "matrix": m.tolist(), "conf": conf,
            "iou": iou_thr, "rows": "true class", "cols": "predicted class"}


def score_eval_set(cache: dict, eval_dir: Path, scopes: dict, op_conf: float) -> dict:
    items = load_eval_set(eval_dir)
    return {name: summarize(match_cache(items, cache, scope), scope, op_conf)
            for name, scope in scopes.items()}


# ---------------------------------------------------------------- speed

def measure_latency(weights: Path, images: list, cfg: Config, n_warmup: int = 10) -> dict:
    """Batch-1, per-frame model time on the current device (R2-10, R4-6):
    pre-processing + inference + NMS post-processing, as Ultralytics reports
    it. Excludes video decoding, network transfer and alerting."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = [str(p) for p in images]
    for p in images[:n_warmup]:
        model.predict(p, imgsz=cfg.imgsz, conf=cfg.op_conf, iou=cfg.nms_iou, verbose=False)
    speeds = [model.predict(p, imgsz=cfg.imgsz, conf=cfg.op_conf, iou=cfg.nms_iou,
                            verbose=False)[0].speed for p in images]
    out = {k: round(float(np.mean([s[k] for s in speeds])), 2)
           for k in ("preprocess", "inference", "postprocess")}
    out["total"] = round(out["preprocess"] + out["inference"] + out["postprocess"], 2)
    out.update({
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "batch": 1, "imgsz": cfg.imgsz, "n_frames": len(images), "n_warmup": n_warmup,
        "scope": "pre-processing + model + NMS per frame; excludes decoding, network, alerts",
    })
    return out
