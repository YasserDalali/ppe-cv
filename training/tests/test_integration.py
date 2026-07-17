"""Full pipeline over synthetic sources, heavy calls mocked — proves the
wiring and that every artifact appears exactly where the notebook expects."""
import json
import shutil

import numpy as np
import pytest
import yaml

from test_evaluate import FakeValModel
from ppe.config import CANONICAL_CLASSES, Config
from ppe.evaluate import compare_all, load_eval_set, model_a_val
from ppe.merge import ensure_prepared
from ppe.predict import Boxes
from ppe.report import build_handoff, build_metrics_md, build_readme, check_no_placeholders
from ppe.train import train_model_a

SECRETS = {"ROBOFLOW_API_KEY": "k", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "kk"}


class FakeYOLO:
    """Fabricates the artifacts a real Ultralytics run leaves behind."""

    def __init__(self, source):
        self.source = source

    def train(self, **kwargs):
        from pathlib import Path

        run = Path(kwargs["project"]) / kwargs["name"]
        (run / "weights").mkdir(parents=True, exist_ok=True)
        (run / "weights" / "best.pt").write_bytes(b"weights")
        (run / "weights" / "last.pt").write_bytes(b"weights")
        (run / "results.csv").write_text("epoch,loss\n1,0.5\n")
        (run / "results.png").write_bytes(b"png")
        (run / "confusion_matrix_normalized.png").write_bytes(b"png")


class StubPredictor:
    def __init__(self, class_names, mapping):
        self.class_names = class_names
        self.mapping = mapping

    def predict(self, image_path):
        return self.mapping.get(image_path.stem, Boxes.empty())


def test_full_pipeline_synthetic(fixture_sources, tmp_path, monkeypatch):
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")

    # raw sources pre-cached on "Drive"; any real download attempt = bug
    for name, src in fixture_sources.items():
        dest = cfg.raw_dir / name
        shutil.copytree(src, dest)
        (dest / ".complete").touch()
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda *a, **k: pytest.fail("network download attempted"))
    monkeypatch.setattr("ppe.download._download_kaggle",
                        lambda *a, **k: pytest.fail("network download attempted"))

    # Steps 1-3
    local = ensure_prepared(cfg, SECRETS)
    assert (cfg.prepared_dir / "fused.zip").is_file()
    assert local == cfg.local_dataset_dir and (local / "data.yaml").is_file()
    data = yaml.safe_load((local / "data.yaml").read_text())
    assert data["names"] == CANONICAL_CLASSES and data["path"] == str(local)

    # disconnect simulation: local disk wiped, Drive intact -> restore, no rebuild
    shutil.rmtree(cfg.work_root)
    local = ensure_prepared(cfg, SECRETS)
    assert (local / "train" / "images").is_dir()
    work = cfg.work_root
    for eval_set in ("fused_val", "ocp_test", "industrial_proxy"):
        assert (work / eval_set / "data.yaml").is_file(), eval_set
    assert (work / "split_summary.json").is_file()
    assert (work / "remap.json").is_file()

    # Step 4 (mocked YOLO)
    monkeypatch.setattr("ppe.train._load_yolo_class", lambda: FakeYOLO)
    best = train_model_a(cfg, local / "data.yaml")
    run_dir = cfg.runs_dir / "model_a"
    assert best == run_dir / "weights" / "best.pt" and best.is_file()

    # Steps 5-6: perfect "Ours" stub, silent "Generic" stub, shared harness
    eval_sets = {name: work / name for name in ("fused_val", "industrial_proxy", "ocp_test")}
    gt_map = {}
    for d in eval_sets.values():
        for img, gt in load_eval_set(d):
            gt_map[img.stem] = Boxes(gt.xyxy, gt.class_names,
                                     np.full(len(gt.class_names), 0.95))
    ours = StubPredictor(CANONICAL_CLASSES, gt_map)
    generic = StubPredictor(
        ["gloves", "helmet", "no-gloves", "no-helmet", "no-shoes", "no-vest",
         "person", "shoes", "vest"], {})
    comparison = compare_all(cfg, ours, generic, eval_sets)
    assert len(comparison["rows"]) == 6
    assert comparison["shared_classes"] == ["person", "helmet", "vest", "gloves",
                                            "no_helmet", "no_gloves"]
    for row in comparison["rows"][1:3]:
        assert row["model"] == "Generic" and row["P"] == 0.0
    for row in comparison["rows"][3:]:
        assert row["model"] == "Ours"
        assert row["P"] == 100.0 and row["R"] == 100.0 and row["mAP50"] > 99.0

    monkeypatch.setattr("ppe.evaluate._load_yolo_model",
                        lambda w: FakeValModel(tmp_path / "valrun"))
    extras = model_a_val(cfg, best, local / "data.yaml")
    assert len(extras["per_class"]) == 11

    # Report + handoff
    training = {"epochs": cfg.epochs, "ultralytics_version": "8.4.95",
                "gpu": "Tesla T4", "headline_split": "val"}
    metrics_md = build_metrics_md(comparison, extras, training)
    check_no_placeholders(metrics_md)
    summary = json.loads((work / "split_summary.json").read_text())
    readme = build_readme(cfg, {
        "gasmask_images": 10, "headline_split": "val", "epochs": cfg.epochs,
        "gpu": "Tesla T4",
        "low_image_classes": [c for c in CANONICAL_CLASSES
                              if summary["per_class_boxes"]["train"].get(c, 0) < 200],
    })
    handoff = build_handoff(cfg, {
        "best_pt": best,
        "remap_json": work / "remap.json",
        "data_yaml": local / "data.yaml",
        "results_csv": run_dir / "results.csv",
        "split_summary": work / "split_summary.json",
        "dashboard_png": run_dir / "results.png",
        "confusion_png": extras["confusion_matrix_png"],
        "metrics_md": metrics_md,
        "readme_md": readme,
    })
    for rel in ["best.pt", "remap.json", "data.yaml", "metrics.md", "results.csv",
                "split_summary.json", "figures/dashboard.png",
                "figures/confusion_matrix.png", "README.md"]:
        assert (handoff / rel).is_file(), rel
    assert list(cfg.results_dir.glob("handoff-*.zip"))
