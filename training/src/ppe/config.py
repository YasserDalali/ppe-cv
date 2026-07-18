"""Single source of truth for every fill-in variable and hyperparameter.

Secrets are NEVER stored in git. load_secrets() reads ROBOFLOW_API_KEY,
KAGGLE_USERNAME and KAGGLE_KEY, in order of preference, from:
  1) <drive_root>/secrets.env (Colab)
  2) training/secrets.env (local, gitignored)
  3) environment variables
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CANONICAL_CLASSES = [
    "Person", "helmet", "vest", "gloves", "boots", "goggles",
    "no_helmet", "no_gloves", "no_goggle", "gas mask",
]

SECRET_KEYS = ["ROBOFLOW_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]

# training/secrets.env (this file is training/src/ppe/config.py)
DEFAULT_LOCAL_SECRETS = Path(__file__).resolve().parents[2] / "secrets.env"


class SecretsError(RuntimeError):
    pass


@dataclass
class Config:
    # --- locations ---
    drive_root: Path = Path("/content/drive/MyDrive/ppe-training")
    work_root: Path = Path("/content/ppe-work")
    local_secrets: Path = DEFAULT_LOCAL_SECRETS
    # --- sources (Roboflow slugs verified by scripts/check_sources.py) ---
    # NOTE (2026-07-17): the original datasetppe/ppe_detection-dnfen v3 CDN
    # export returns GCS NoSuchKey — forked into yasser-dalali/ppe_detection-
    # dnfen-tpjga v1 (308 images, plain YOLOv8 export) to work around it.
    ppe_workspace: str = "yasser-dalali"
    ppe_project: str = "ppe_detection-dnfen-tpjga"
    ppe_version: int = 1
    gasmask_workspace: str = "daniil-yarmov"
    gasmask_project: str = "gas-masks"
    gasmask_version: int = 1                    # pinned: v1, 384 images, no substitute
    ocp_workspace: str = "yasser-dalali"
    # Roboflow slug for the project whose display name is "OCP" (207 images).
    # Live versions: v2 exists (yolov8 export OK); v1 does not.
    ocp_project: str = "ocp-snzjw"
    ocp_version: int = 2
    sh17_kaggle_slug: str = "mugheesahmad/sh17-dataset-for-ppe-detection"
    # --- Model B (hosted; never trained) ---
    # Only version 1 has a trained hosted model; its published map/recall/precision
    # match generic_published. Version 3 is dataset-only (no model endpoint).
    generic_model_id: str = "ppe_detection-dnfen/1"
    generic_published: dict = field(default_factory=lambda: {"P": 58.1, "R": 64.4, "mAP50": 59.1})
    # --- split & sampling ---
    seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    ocp_dup_factor: int = 3        # train split only
    # OCP is the target deployment domain (207 site photos) — worth the x3
    # weight. Gas-masks is a generic public set standing in for one class;
    # it doesn't deserve the same boost, and every duplicate there is pure
    # epoch-time cost with no OCP-relevance payoff.
    gasmask_dup_factor: int = 1    # train split only
    # SH17 (Kaggle) dwarfs the other 3 curated sources combined (~900 images)
    # and both remap time and epoch time scale with it directly. Common
    # classes (Person, gloves, vest, no_helmet) already hit mAP50 > 0.85 with
    # a few hundred examples, so a random cap on SH17's train slice trims
    # mostly-redundant common-class images. The split decision (and this cap)
    # happens on raw filenames before remap runs, so capped-out images are
    # never remapped either — see merge.compute_raw_split. None disables the
    # cap. Train split only.
    sh17_train_cap: int | None = 1500
    # val/test used to stay uncapped ("evaluate on the full source"), but
    # Ultralytics validates every epoch, so an uncapped SH17 val split was
    # driving most of the per-epoch cost even after the train cap. Capped
    # proportionally to sh17_train_cap (val/test fracs are each ~0.15 vs.
    # train's 0.70, so ~1/5 of the train cap keeps the same relative size).
    # None disables the cap (full source, pre-cap behavior).
    sh17_val_cap: int | None = 300
    sh17_test_cap: int | None = 300
    # --- training ---
    model_name: str = "yolov8n.pt"
    imgsz: int = 640
    batch: int = 16
    epochs: int = 50
    patience: int = 20             # stop once val mAP plateaus this many epochs
    erasing: float = 0.4
    close_mosaic: int = 10
    save_period: int = 1
    run_name: str = "model_a"
    # --- evaluation ---
    eval_conf: float = 0.25
    eval_iou: float = 0.5
    camera_fps: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.drive_root / "raw"

    @property
    def prepared_dir(self) -> Path:
        return self.drive_root / "prepared"

    @property
    def runs_dir(self) -> Path:
        return self.drive_root / "runs"

    @property
    def results_dir(self) -> Path:
        return self.drive_root / "results"

    @property
    def wheels_dir(self) -> Path:
        return self.drive_root / "wheels"

    @property
    def local_dataset_dir(self) -> Path:
        return self.work_root / "fused"


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_secrets(cfg: Config) -> dict[str, str]:
    """Drive secrets.env > training/secrets.env > environment variables."""
    locations = [cfg.drive_root / "secrets.env", cfg.local_secrets]
    merged: dict[str, str] = {k: os.environ.get(k, "") for k in SECRET_KEYS}
    for loc in reversed(locations):  # lowest-priority file first, Drive last (wins)
        if loc.is_file():
            merged.update({k: v for k, v in parse_env_file(loc).items() if v})
    missing = [k for k in SECRET_KEYS if not merged.get(k)]
    if missing:
        raise SecretsError(
            f"Missing secrets {missing}. Looked in: {locations[0]} (Drive), "
            f"{locations[1]} (local), then environment variables. "
            "Create secrets.env with KEY=value lines (see GUIDE.md); never commit it."
        )
    return {k: merged[k] for k in SECRET_KEYS}
