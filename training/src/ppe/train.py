"""Step 4 — Model A training with disconnect-proof auto-resume.

Checkpoints go straight to Drive (project=cfg.runs_dir, save_period=1), the
dataset is read from local disk (data_yaml under work_root). Recovery from a
Colab disconnect is "Run all" again: a finished run is skipped, an unfinished
one resumes from last.pt. Maximum loss: the current epoch.
"""
from __future__ import annotations

from pathlib import Path

from ppe.config import Config

FINISHED_MARKER = ".finished"


def _load_yolo_class():
    from ultralytics import YOLO  # lazy: runtime-only dependency

    return YOLO


def decide_train_action(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    if (run_dir / FINISHED_MARKER).is_file() and (run_dir / "weights" / "best.pt").is_file():
        return "skip"
    if (run_dir / "weights" / "last.pt").is_file():
        return "resume"
    return "fresh"


def train_model_a(cfg: Config, data_yaml: Path) -> Path:
    """Train (or resume, or skip) Model A; returns the path to best.pt."""
    run_dir = cfg.runs_dir / cfg.run_name
    best = run_dir / "weights" / "best.pt"
    action = decide_train_action(run_dir)
    print(f"[train] action={action} run_dir={run_dir}")
    if action == "skip":
        return best

    YOLO = _load_yolo_class()
    if action == "resume":
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(cfg.model_name)
        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            seed=cfg.seed,
            erasing=cfg.erasing,
            close_mosaic=cfg.close_mosaic,
            save_period=cfg.save_period,
            project=str(cfg.runs_dir),
            name=cfg.run_name,
            exist_ok=True,
        )
    if not best.is_file():
        raise RuntimeError(
            f"training finished but {best} does not exist — check the run "
            f"directory {run_dir} for Ultralytics errors"
        )
    (run_dir / FINISHED_MARKER).touch()
    return best
