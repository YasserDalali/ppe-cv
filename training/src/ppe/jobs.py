"""2026-10 revision — what each Colab notebook runs.

    job0     build prepared/pool.zip (+ re-score the old model, if present)
    job1-4   for each run id in JOBS[job]: assemble -> train -> predict -> score
    collect  aggregate every finished run into results/revision/final/

Every step skips work that is already on Drive, so "Run all" after a
disconnect resumes where it stopped.
"""
from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

from ppe.config import CANONICAL_CLASSES, JOBS, Config, parse_run_id
from ppe.pool import assemble_run, ensure_pool
from ppe.scoring import (cache_predictions, match_cache, measure_latency,
                         score_eval_set, summarize)
from ppe.train import train_run


def results_root(cfg: Config) -> Path:
    return cfg.results_dir / "revision"


def protocol(cfg: Config) -> dict:
    return {
        "checkpoint": "best.pt (Ultralytics fitness on the shared public val split)",
        "conf_floor": cfg.conf_floor, "op_conf": cfg.op_conf,
        "match_iou": cfg.match_iou, "nms_iou": cfg.nms_iou, "imgsz": cfg.imgsz,
        "ap": "COCO 101-point interpolation; classes without ground truth in a set excluded",
        "P_R": f"pooled over classes at conf >= {cfg.op_conf}, IoU >= {cfg.match_iou}",
    }


def scopes_for(counts: dict) -> dict:
    """'common' = classes the construction baseline B was trained on (so all
    four models can be compared); 'all' = the 9 canonical classes."""
    b_train = counts["per_source"]["roboflow_ppe"]["train"]["boxes"]
    return {"common": [c for c in CANONICAL_CLASSES if b_train.get(c)],
            "all": list(CANONICAL_CLASSES)}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def run_one(cfg: Config, run_id: str, pool_dir: Path) -> dict:
    res = results_root(cfg) / run_id
    if (res / "scores.json").is_file():
        print(f"[job] {run_id}: scores.json on Drive — skipping")
        return json.loads((res / "scores.json").read_text())
    data = assemble_run(cfg, run_id, pool_dir)
    _, seed, _ = parse_run_id(run_id)
    best = train_run(cfg, run_id, data.data_yaml, seed)
    scopes = scopes_for(json.loads((Path(pool_dir) / "counts.json").read_text()))
    sets = {}
    for name, d in data.eval_sets.items():
        cache_path = res / f"preds_{name}.json"
        cache = (json.loads(cache_path.read_text()) if cache_path.is_file()
                 else cache_predictions(best, d, cache_path, cfg))
        sets[name] = score_eval_set(cache, d, scopes, cfg.op_conf)
    run_dir = cfg.runs_dir / run_id
    for f in ("args.yaml", "results.csv"):
        if (run_dir / f).is_file():
            shutil.copy2(run_dir / f, res / f)
    scores = {"run_id": run_id, "meta": data.meta, "weights": str(best),
              "protocol": protocol(cfg), "scopes": scopes, "sets": sets}
    _write_json(res / "scores.json", scores)
    ocp = sets["ocp"]
    print(f"[job] {run_id}: OCP mAP50 common={ocp['common']['map50']:.3f} "
          f"all={ocp['all']['map50']:.3f} ({ocp['all']['n_images']} frames)")
    return scores


def latency(cfg: Config, pool_dir: Path, run_id: str = "A_f0", n_frames: int = 200) -> dict:
    out = results_root(cfg) / "latency.json"
    if out.is_file():
        return json.loads(out.read_text())
    frames_dir = cfg.work_root / "runs" / run_id / "eval" / "ocp" / "images"
    if not frames_dir.is_dir():  # run was skipped after a reconnect: rebuild locally
        assemble_run(cfg, run_id, pool_dir)
    frames = sorted(frames_dir.iterdir())
    frames = (frames * math.ceil(n_frames / len(frames)))[:n_frames]
    best = cfg.runs_dir / run_id / "weights" / "best.pt"
    result = {"run_id": run_id, "frames": "OCP 1920x1080 test frames", **measure_latency(best, frames, cfg)}
    _write_json(out, result)
    print(f"[latency] {result}")
    return result


OLD_TABLE2_CLASSES = ["helmet", "vest", "gloves", "no_helmet", "no_gloves"]


def rescore_old_model(cfg: Config) -> dict | None:
    """R4-3: the submitted model's fused-val score under the old Table-2
    protocol (only boxes >= 0.25 enter AP) and under the new one (floor
    0.001), same 5 shared classes — explains 48.6 vs the Table-3 average."""
    old_zip = cfg.prepared_dir / "fused.zip"
    old_pt = cfg.runs_dir / cfg.run_name / "weights" / "best.pt"
    out = results_root(cfg) / "old_model_rescore.json"
    if out.is_file():
        return json.loads(out.read_text())
    if not (old_zip.is_file() and old_pt.is_file()):
        print(f"[old-model] skipped: need {old_zip} and {old_pt}")
        return None
    dest = cfg.work_root / "old"
    with zipfile.ZipFile(old_zip) as zf:
        zf.extractall(dest, [m for m in zf.namelist() if m.startswith("fused_val/")])
    eval_dir = dest / "fused_val"
    from ppe.evaluate import load_eval_set

    cache = cache_predictions(old_pt, eval_dir, results_root(cfg) / "old_model" / "preds_fused_val.json", cfg)
    items = load_eval_set(eval_dir)
    new = summarize(match_cache(items, cache, OLD_TABLE2_CLASSES), OLD_TABLE2_CLASSES, cfg.op_conf)
    thresholded = {"images": {
        k: {kk: [v for v, c in zip(e[kk], e["conf"]) if c >= cfg.op_conf] for kk in ("xyxy", "cls", "conf")}
        for k, e in cache["images"].items()}}
    old = summarize(match_cache(items, thresholded, OLD_TABLE2_CLASSES), OLD_TABLE2_CLASSES, cfg.op_conf)
    result = {
        "classes": OLD_TABLE2_CLASSES, "n_images": new["n_images"],
        "map50_old_protocol_conf_ge_0.25": old["map50"],
        "map50_new_protocol_conf_floor_0.001": new["map50"],
        "per_class_ap50_new": {c: v["ap50"] for c, v in new["per_class"].items()},
        "paper_submitted": {"table2_fused_val_map50": 0.486, "table3_mean_of_these_5": 0.562},
        "note": "fused_val of the submitted run; its OCP part had the augmentation leak",
    }
    _write_json(out, result)
    print(f"[old-model] {result}")
    return result


def run_job(cfg: Config, job: str, secrets: dict | None = None) -> None:
    if job not in JOBS:
        raise ValueError(f"unknown JOB {job!r}; expected one of {sorted(JOBS)}")
    if job == "job0":
        pool = ensure_pool(cfg, secrets)
        counts = json.loads((pool / "counts.json").read_text())
        for f, info in counts["ocp_folds"].items():
            print(f"[job0] OCP fold {f}: {info['images']} frames, recordings {info['recordings']}")
        rescore_old_model(cfg)
        return
    pool = ensure_pool(cfg, secrets)
    if job == "collect":
        from ppe.collect import collect

        collect(cfg, pool)
        return
    for run_id in JOBS[job]:
        run_one(cfg, run_id, pool)
        if run_id == "A_f0":
            latency(cfg, pool, run_id)
        # each local run copy is ~7 GB with the .npy cache; scores are on Drive
        shutil.rmtree(cfg.work_root / "runs" / run_id, ignore_errors=True)
    print(f"[job] {job} done: {JOBS[job]}")
