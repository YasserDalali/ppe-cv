"""End-to-end: every job + collect, with training and inference faked.

Fake models: A predicts the ground truth on every frame, S on every other
frame, P on every third, B never — so the expected ordering is known."""
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from conftest import OCP_RECORDINGS, revision_cfg
from ppe.config import JOBS, all_run_ids
from ppe.evaluate import load_eval_set
from ppe.jobs import run_job
from ppe.pool import recording_id

KEEP_EVERY = {"A": 1, "S": 2, "P": 3}


def fake_train_run(cfg, run_id, data_yaml, seed):
    run_dir = cfg.runs_dir / run_id
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_text("fake")
    (run_dir / "args.yaml").write_text(yaml.safe_dump(
        {"epochs": cfg.epochs, "imgsz": cfg.imgsz, "batch": cfg.batch, "seed": seed,
         "data": str(data_yaml), "optimizer": "SGD", "lr0": 0.01}))
    return run_dir / "weights" / "best.pt"


def fake_cache(weights, eval_dir, out_json, cfg):
    every = KEEP_EVERY.get(Path(weights).parts[-3][0])
    images = {}
    for i, (img, gt) in enumerate(load_eval_set(eval_dir)):
        hit = every is not None and i % every == 0
        images[img.name] = {"xyxy": gt.xyxy.tolist() if hit else [],
                            "cls": list(gt.class_names) if hit else [],
                            "conf": [0.9] * len(gt.class_names) if hit else []}
    out = {"images": images}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out))
    return out


@pytest.fixture
def ran(tmp_path, revision_raw, monkeypatch):
    cfg = revision_cfg(tmp_path)
    cfg.ocp_upload_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cfg.ocp_upload_zip, "w") as zf:
        for p in revision_raw["ocp"].rglob("*"):
            if p.is_file():
                zf.write(p, str(p.relative_to(revision_raw["ocp"])))
    monkeypatch.setattr("ppe.download.ensure_source", lambda c, s, name: revision_raw[name])
    monkeypatch.setattr("ppe.jobs.train_run", fake_train_run)
    monkeypatch.setattr("ppe.jobs.cache_predictions", fake_cache)
    monkeypatch.setattr("ppe.jobs.measure_latency",
                        lambda w, frames, c: {"total": 9.0, "n_frames": len(frames)})
    run_job(cfg, "job0", {"ROBOFLOW_API_KEY": "k"})
    for job in ("job1", "job2", "job3", "job4"):
        run_job(cfg, job)
    run_job(cfg, "collect")
    return cfg


def test_every_run_is_scored_once(ran):
    root = ran.results_dir / "revision"
    assert sorted(p.parent.name for p in root.glob("*/scores.json")) == sorted(all_run_ids())
    assert sorted(r for j in JOBS.values() for r in j) == sorted(all_run_ids())
    a0 = json.loads((root / "A_f0" / "scores.json").read_text())
    assert a0["sets"]["ocp"]["all"]["map50"] == pytest.approx(1.0)
    assert a0["protocol"]["conf_floor"] == 0.001 and a0["protocol"]["op_conf"] == 0.25
    assert (root / "latency.json").is_file()
    assert not any((ran.work_root / "runs").glob("*")), "local run copies must be removed"


def test_rerun_skips_finished_runs(ran, monkeypatch):
    monkeypatch.setattr("ppe.jobs.train_run", lambda *a: pytest.fail("retrained a finished run"))
    run_job(ran, "job2")


def test_collect_pools_ocp_and_orders_models(ran):
    final = ran.results_dir / "revision" / "final"
    summary = json.loads((final / "summary.json").read_text())
    assert summary["missing_runs"] == []
    pooled = summary["ocp_pooled"]
    assert pooled["n_images"] == sum(OCP_RECORDINGS.values())
    assert sorted(pooled["a_folds"]) == [f"A_f{i}" for i in range(5)]
    common = pooled["scopes"]["common"]
    m = {e: v["map50"] for e, v in common["models"].items()}
    assert m["A"] == pytest.approx(1.0) and m["B"] == 0.0
    assert m["A"] > m["S"] > m["P"] > m["B"]
    assert common["bootstrap"]["n_recordings"] == len(OCP_RECORDINGS)
    assert common["bootstrap"]["pairs"]["A-S"]["diff"] > 0
    assert "B" not in pooled["scopes"]["all"]["models"]
    for name in ("ablation.tex", "ocp_pooled.tex", "per_class.tex", "protocol.tex",
                 "counts.tex", "ocp_folds.tex", "settings.tex", "summary.md",
                 "confusion_matrix.png"):
        assert (final / name).is_file(), name
    assert "SGD" in (final / "settings.tex").read_text()
    cm = pooled["confusion_A"]["matrix"]  # A predicts the ground truth: all on the diagonal
    assert sum(cm[i][i] for i in range(len(cm) - 1)) == sum(map(sum, cm)) > 0


def test_A_folds_never_score_a_recording_they_trained_on(ran):
    root = ran.results_dir / "revision"
    for f in range(5):
        meta = json.loads((root / f"A_f{f}" / "scores.json").read_text())["meta"]
        preds = json.loads((root / f"A_f{f}" / "preds_ocp.json").read_text())
        assert {recording_id(n) for n in preds["images"]} == set(meta["ocp_test_recordings"])
