"""Opt-in real-training smoke test (excluded by default; run with -m slow).

Proves the actual Ultralytics train call accepts our exact kwargs and that
the auto-resume bookkeeping works against a real run directory. Re-run after
any Ultralytics upgrade.
"""
import pytest

pytest.importorskip("ultralytics")

from ppe.config import Config
from ppe.merge import merge_and_split
from ppe.remap import remap_all
from ppe.train import decide_train_action, train_model_a


@pytest.mark.slow
def test_real_cpu_smoke_train(fixture_sources, tmp_path):
    cfg = Config(
        drive_root=tmp_path / "drive",
        work_root=tmp_path / "work",
        epochs=1,
        imgsz=64,
        batch=2,
    )
    remapped_root = tmp_path / "remapped"
    remap_all(cfg, fixture_sources, remapped_root)
    res = merge_and_split(cfg, {n: remapped_root / n for n in fixture_sources},
                          tmp_path / "out")
    best = train_model_a(cfg, res.dataset_dir / "data.yaml")
    assert best.is_file()
    run_dir = cfg.runs_dir / cfg.run_name
    assert (run_dir / ".finished").is_file()
    assert (run_dir / "results.csv").is_file()
    assert decide_train_action(run_dir) == "skip"
    # second call must skip instantly (idempotency)
    assert train_model_a(cfg, res.dataset_dir / "data.yaml") == best


@pytest.mark.slow
def test_real_cpu_revision_run(revision_raw, tmp_path):
    """Real 1-epoch training of a revision run, then the real prediction
    cache, scoring and latency paths against the trained weights."""
    from conftest import revision_cfg
    from ppe.pool import assemble_run, build_pool_dir
    from ppe.scoring import cache_predictions, measure_latency, score_eval_set
    from ppe.train import train_run

    cfg = revision_cfg(tmp_path, epochs=1, imgsz=64, batch=2)
    pool_dir = build_pool_dir(cfg, revision_raw, tmp_path / "pool")
    run = assemble_run(cfg, "A_f0", pool_dir)
    best = train_run(cfg, "A_f0", run.data_yaml, seed=0)
    assert best.is_file() and not list(best.parent.glob("epoch*.pt"))  # save_period=-1
    cache = cache_predictions(best, run.eval_sets["ocp"], tmp_path / "preds.json", cfg)
    assert set(cache["images"]) == {p.name for p in (run.eval_sets["ocp"] / "images").iterdir()}
    scores = score_eval_set(cache, run.eval_sets["ocp"], {"all": ["Person", "helmet"]}, cfg.op_conf)
    assert 0.0 <= scores["all"]["map50"] <= 1.0
    lat = measure_latency(best, sorted((run.eval_sets["ocp"] / "images").iterdir()), cfg, n_warmup=1)
    assert lat["total"] > 0 and lat["batch"] == 1
