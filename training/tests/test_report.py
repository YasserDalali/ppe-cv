import json
import zipfile

import pytest

from ppe.config import CANONICAL_CLASSES, Config
from ppe.report import (
    ReportError,
    build_handoff,
    build_metrics_md,
    build_readme,
    check_no_placeholders,
    low_image_classes,
)


def fake_comparison():
    rows = [{"model": "Generic", "eval_set": "Own val (construction)",
             "P": 58.1, "R": 64.4, "mAP50": 59.1}]
    for model, es in [("Generic", "Industrial proxy"), ("Generic", "OCP site"),
                      ("Ours", "Fused val"), ("Ours", "Industrial proxy"),
                      ("Ours", "OCP site")]:
        rows.append({"model": model, "eval_set": es, "P": 70.0, "R": 60.0, "mAP50": 65.5})
    return {"rows": rows, "shared_classes": ["person", "helmet", "vest", "gloves"],
            "conf": 0.25, "iou": 0.5}


def fake_extras():
    per_class = [{"class": c, "P": 80.0, "R": 70.0, "mAP50": 75.0, "images": 12}
                 for c in CANONICAL_CLASSES]
    per_class.append({"class": "ALL", "P": 80.0, "R": 70.0, "mAP50": 75.0, "images": 120})
    return {"per_class": per_class,
            "speed": {"preprocess": 1.0, "inference": 15.0, "postprocess": 4.0, "total": 20.0},
            "cameras": 10, "camera_fps": 5,
            "camera_formula": "cameras = floor((1000 / total_ms_per_frame) / fps)",
            "confusion_matrix_png": "x/confusion_matrix_normalized.png"}


def fake_training():
    return {"epochs": 50, "ultralytics_version": "8.4.95", "gpu": "Tesla T4",
            "headline_split": "val"}


def test_metrics_md_fully_filled():
    md = build_metrics_md(fake_comparison(), fake_extras(), fake_training())
    check_no_placeholders(md)  # must not raise
    assert "58.1" in md and "Own val (construction)" in md
    for c in CANONICAL_CLASSES:
        assert f"| {c} " in md
    assert "| ALL " in md
    assert "person, helmet, vest, gloves" in md
    assert "conf=0.25" in md and "IoU=0.5" in md
    assert "~10 cameras at 5 FPS" in md
    assert "val" in md and "8.4.95" in md and "Tesla T4" in md


def test_check_no_placeholders_catches_blanks():
    with pytest.raises(ReportError):
        check_no_placeholders("| Generic | [ ] |")
    with pytest.raises(ReportError):
        check_no_placeholders("- epochs: __\n")
    check_no_placeholders("stem__dup1 is fine, so is a__b")


def test_low_image_classes():
    summary = {"per_class_boxes": {"train": {"Person": 5000, "helmet": 300,
                                             "no_goggle": 12, "gas mask": 150}}}
    low = low_image_classes(summary, threshold=200)
    # every canonical class not reaching 200 train boxes, in canonical order
    assert low == ["vest", "gloves", "boots", "goggles", "no_helmet",
                   "no_gloves", "no_goggle", "gas mask"]


def test_readme_answers_all_five_questions(tmp_path):
    cfg = Config(drive_root=tmp_path)
    text = build_readme(cfg, {
        "gasmask_images": 384, "headline_split": "val", "epochs": 50,
        "gpu": "Tesla T4", "low_image_classes": ["no_goggle", "gas mask"],
    })
    assert "384" in text and "daniil-yarmov/gas-masks/dataset/1" in text
    assert "val" in text
    assert "50" in text and "Tesla T4" in text
    assert "no_goggle, gas mask" in text
    assert "designed only" in text
    assert "no generic.pt" in text
    assert "Step 7" in text and "skipped" in text.lower()


def test_build_handoff_layout_and_zip(tmp_path):
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")
    src = tmp_path / "src"
    src.mkdir()
    files = {}
    for key, name in [("best_pt", "best.pt"), ("remap_json", "remap.json"),
                      ("data_yaml", "data.yaml"), ("results_csv", "results.csv"),
                      ("split_summary", "split_summary.json"),
                      ("dashboard_png", "results.png"),
                      ("confusion_png", "confusion_matrix_normalized.png")]:
        p = src / name
        p.write_text(json.dumps({"k": key}) if name.endswith(".json") else key)
        files[key] = p
    handoff = build_handoff(cfg, {**files, "metrics_md": "# metrics",
                                  "readme_md": "# readme"})
    for rel in ["best.pt", "remap.json", "data.yaml", "metrics.md", "results.csv",
                "split_summary.json", "figures/dashboard.png",
                "figures/confusion_matrix.png", "README.md"]:
        assert (handoff / rel).is_file(), rel
    zips = list(cfg.results_dir.glob("handoff-*.zip"))
    assert len(zips) == 1
    with zipfile.ZipFile(zips[0]) as zf:
        assert "metrics.md" in zf.namelist() and "figures/dashboard.png" in zf.namelist()
