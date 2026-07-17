"""Report — metrics.md autofill, README answers, handoff zip.

metrics.md is generated programmatically; check_no_placeholders() guarantees
zero unfilled "[ ]" / "__" blanks survive (tested).
"""
from __future__ import annotations

import datetime as _dt
import re
import shutil
from pathlib import Path

from ppe.config import CANONICAL_CLASSES, Config
from ppe.merge import zip_dir


class ReportError(RuntimeError):
    pass


def check_no_placeholders(text: str) -> None:
    problems = []
    if "[ ]" in text:
        problems.append('unfilled "[ ]"')
    if re.search(r"(^|\s)__($|\s)", text):
        problems.append('unfilled "__"')
    if problems:
        raise ReportError(f"metrics.md still has {', '.join(problems)} — refusing to ship")


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def build_metrics_md(comparison: dict, extras: dict, training: dict) -> str:
    lines = ["## Comparison (THE important table)", ""]
    lines.append(_row(["Model", "Eval set", "P %", "R %", "mAP50 %"]))
    lines.append(_row(["-------"] * 5))
    for r in comparison["rows"]:
        lines.append(_row([r["model"], r["eval_set"], r["P"], r["R"], r["mAP50"]]))
    lines += [
        "",
        f"Shared classes scored for both models: "
        f"{', '.join(comparison['shared_classes'])} "
        f"(conf={comparison['conf']}, IoU={comparison['iou']}; P/R at this fixed "
        "confidence, same harness for both models).",
        "",
        f"## Per-class (Ours, on {training['headline_split']})",
        "",
        _row(["Class", "P %", "R %", "mAP50 %", "# images"]),
        _row(["-------"] * 5),
    ]
    for r in extras["per_class"]:
        lines.append(_row([r["class"], r["P"], r["R"], r["mAP50"], r["images"]]))
    speed = extras["speed"]
    lines += [
        "",
        "## Runtime (same GPU as training)",
        "",
        f"- preprocess: {speed['preprocess']} ms",
        f"- inference: {speed['inference']} ms",
        f"- postprocess: {speed['postprocess']} ms",
        f"- total: {speed['total']} ms / frame",
        f"- ~{extras['cameras']} cameras at {extras['camera_fps']} FPS each",
        f"- formula: {extras['camera_formula']}",
        "",
        "## Training",
        "",
        f"- epochs: {training['epochs']}",
        f"- ultralytics version: {training['ultralytics_version']}",
        f"- GPU: {training['gpu']}",
        f"- split used for headline numbers: {training['headline_split']}",
        "",
    ]
    text = "\n".join(lines)
    check_no_placeholders(text)
    return text


def low_image_classes(split_summary: dict, threshold: int = 200) -> list:
    train_boxes = split_summary.get("per_class_boxes", {}).get("train", {})
    return [c for c in CANONICAL_CLASSES if train_boxes.get(c, 0) < threshold]


def build_readme(cfg: Config, answers: dict) -> str:
    low = answers.get("low_image_classes") or []
    low_txt = ", ".join(low) if low else "none"
    return f"""# Handoff README

Answers to the 5 required questions:

1. **Gas-mask dataset:** confirmed — Roboflow
   https://universe.roboflow.com/daniil-yarmov/gas-masks/dataset/1 (v1),
   {answers['gasmask_images']} images, downloaded and verified by the pipeline.
2. **Headline numbers from val or test?** {answers['headline_split']}
   (comparison-table rows state their eval set explicitly).
3. **Epochs + GPU:** {answers['epochs']} epochs on {answers['gpu']}.
4. **Classes with almost no images:** {low_txt}
   (threshold: <200 train boxes; exact counts in split_summary.json).
5. **Alert system:** designed only — not built.

Notes:

- Model B (generic) is the hosted Roboflow model `{cfg.generic_model_id}`;
  it was never trained by us, so there is **no generic.pt** in this handoff.
  Its rows in metrics.md come from the same shared evaluation harness as
  Model A; its published own-val numbers are copied as-is.
- Step 7 (LLM judge) was skipped in this run — out of scope for this handoff.
"""


def build_handoff(cfg: Config, artifacts: dict) -> Path:
    """Assemble results/handoff-<date>/ and zip it next to it on Drive."""
    date = _dt.date.today().isoformat()
    handoff = cfg.results_dir / f"handoff-{date}"
    if handoff.exists():
        shutil.rmtree(handoff)
    (handoff / "figures").mkdir(parents=True)

    copies = {
        "best_pt": "best.pt",
        "remap_json": "remap.json",
        "data_yaml": "data.yaml",
        "results_csv": "results.csv",
        "split_summary": "split_summary.json",
        "dashboard_png": "figures/dashboard.png",
        "confusion_png": "figures/confusion_matrix.png",
    }
    for key, rel in copies.items():
        src = Path(artifacts[key])
        if not src.is_file():
            raise ReportError(f"handoff artifact {key} missing at {src}")
        shutil.copy2(src, handoff / rel)
    (handoff / "metrics.md").write_text(artifacts["metrics_md"])
    (handoff / "README.md").write_text(artifacts["readme_md"])
    zip_dir(handoff, cfg.results_dir / f"handoff-{date}.zip")
    return handoff
