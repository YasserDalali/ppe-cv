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
