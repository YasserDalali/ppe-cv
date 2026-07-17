# PPE Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the static, tested `training/` package + thin Colab notebook that implements Steps 1–6 of `content/engineering-requirements.md` per the approved spec `docs/superpowers/specs/2026-07-16-ppe-training-pipeline-design.md`.

**Architecture:** All logic lives in an importable `training/src/ppe/` package (config, download, remap, merge, train, predict, evaluate, report). The Colab notebook is ~10 cells: config + one call per step. Every step is idempotent against Drive artifacts. Tests run locally with synthetic fixture datasets; no real training or network by default (heavy deps lazy-imported, network calls mocked).

**Tech Stack:** Python 3.12 locally / Colab Python, ultralytics (YOLOv8n), roboflow + inference (hosted Model B), supervision (metrics), kaggle (SH17), pytest.

## Global Constraints

- Final classes, exactly these 10, in this order, no `none`: `Person, helmet, vest, gloves, boots, goggles, no_helmet, no_gloves, no_goggle, gas mask`.
- Secrets only from `<DRIVE_ROOT>/secrets.env` → `training/secrets.env` (gitignored) → env vars. Never committed, never printed, never hardcoded.
- OCP source: Roboflow `yasser-dalali/OCP` v1, yolov8 export. Pipeline **fails fast** if unavailable — no OCP-less mode.
- Gas masks: Daniil Yarmov `gas-masks` **v1** (384 images) — pinned, no substitute.
- Model B is never trained: hosted `ppe_detection-dnfen/3`; published own-val P 58.1 / R 64.4 / mAP50 59.1 inserted as-is.
- SEED=0; all splits deterministic. Train: yolov8n, imgsz 640, batch 16, epochs 50, erasing 0.4, close_mosaic 10, save_period 1.
- Remap hard-fails on any unaccounted image/box loss (the old 43%-drop bug class).
- `metrics.md` must contain zero `[ ]` after generation (tested).
- Default pytest run does no training and no network; CPU smoke train behind `@pytest.mark.slow` (excluded by default).
- Step 7 (LLM judge), alert dispatch, paper edits: out of scope.

## File Structure

```
training/
  pytest.ini                  # pythonpath=src, addopts -m "not slow", slow marker
  requirements.txt            # pinned (exact pins resolved during Task 1/12)
  GUIDE.md                    # Task 11
  colab_train.ipynb           # Task 11
  scripts/check_sources.py    # Task 2 — live class-name check (names only, no secrets printed)
  src/ppe/__init__.py
  src/ppe/config.py           # Task 1 — Config dataclass + secrets loading
  src/ppe/remap.py            # Task 3 — name-based remap + accounting + remap.json
  src/ppe/merge.py            # Task 4 — merge, split, duplication, holdouts, split_summary
  src/ppe/download.py         # Task 5 — 4 sources, cache under raw/, validation
  src/ppe/predict.py          # Task 6 — Boxes, UltralyticsPredictor, RoboflowPredictor
  src/ppe/evaluate.py         # Task 7 — shared harness (P/R matcher + supervision mAP), Model A extras
  src/ppe/train.py            # Task 8 — auto-resume training
  src/ppe/report.py           # Task 9 — metrics.md, README.md, handoff zip
  tests/conftest.py           # Task 2 — synthetic fixture builders
  tests/test_config.py        # Task 1
  tests/test_remap.py         # Task 3
  tests/test_merge.py         # Task 4
  tests/test_download.py      # Task 5
  tests/test_predict.py       # Task 6
  tests/test_evaluate.py      # Task 7
  tests/test_train.py         # Task 8
  tests/test_report.py        # Task 9
  tests/test_integration.py   # Task 10 — full synthetic pipeline, YOLO mocked
  tests/test_smoke_train.py   # Task 12 — @pytest.mark.slow real 1-epoch CPU train
```

Local dev env: `training/.venv` (gitignored) with the light deps only: `pytest pyyaml pillow numpy supervision`. ultralytics/roboflow/inference/kaggle are **lazy-imported inside functions** so the default suite never needs them.

---

### Task 1: Scaffolding + config.py (Config dataclass, secrets loading)

**Files:**
- Create: `training/pytest.ini`, `training/requirements.txt`, `training/src/ppe/__init__.py`, `training/src/ppe/config.py`, `training/tests/test_config.py`
- Modify: `.gitignore` (add `training/.venv/`)

**Interfaces:**
- Produces: `ppe.config.Config` dataclass (fields below), `CANONICAL_CLASSES: list[str]`, `load_secrets(cfg) -> dict[str,str]` (keys `ROBOFLOW_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`; raises `SecretsError` naming every location searched), `parse_env_file(path) -> dict[str,str]`. Dir properties on Config: `raw_dir`, `prepared_dir`, `runs_dir`, `results_dir`, `wheels_dir` (all under `drive_root`), plus `local_dataset_dir` under `work_root`.

- [x] **Step 1: venv + gitignore + pytest.ini + requirements.txt**

```bash
cd /home/yasser/ppe-cv/training
python3 -m venv .venv
.venv/bin/pip install -q pytest pyyaml pillow numpy supervision
echo 'training/.venv/' >> ../.gitignore
```

`training/pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = src
addopts = -m "not slow" -q
markers =
    slow: real CPU smoke training / live network, opt-in via -m slow
```

`training/requirements.txt` (Colab-side pins; local-verifiable ones pinned from the venv resolve in this step, heavy ones best-known and re-verified in the first Colab session — see Task 12):
```
ultralytics==8.4.95
roboflow~=1.2
inference~=0.53
supervision==<pin from .venv>
kaggle~=1.7
pytest==<pin from .venv>
PyYAML==<pin from .venv>
pillow==<pin from .venv>
numpy==<pin from .venv>
```

- [x] **Step 2: Write failing tests** — `training/tests/test_config.py`

```python
from pathlib import Path
import pytest
from ppe.config import CANONICAL_CLASSES, Config, SecretsError, load_secrets, parse_env_file


def test_canonical_classes_exact_order():
    assert CANONICAL_CLASSES == [
        "Person", "helmet", "vest", "gloves", "boots", "goggles",
        "no_helmet", "no_gloves", "no_goggle", "gas mask",
    ]


def test_config_dirs_derive_from_drive_root(tmp_path):
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")
    assert cfg.raw_dir == tmp_path / "drive" / "raw"
    assert cfg.prepared_dir == tmp_path / "drive" / "prepared"
    assert cfg.runs_dir == tmp_path / "drive" / "runs"
    assert cfg.results_dir == tmp_path / "drive" / "results"
    assert cfg.wheels_dir == tmp_path / "drive" / "wheels"
    assert cfg.local_dataset_dir == tmp_path / "work" / "fused"


def test_parse_env_file(tmp_path):
    p = tmp_path / "secrets.env"
    p.write_text("# comment\nROBOFLOW_API_KEY=abc\n\nKAGGLE_USERNAME = bob\nexport KAGGLE_KEY=k1\n")
    assert parse_env_file(p) == {
        "ROBOFLOW_API_KEY": "abc", "KAGGLE_USERNAME": "bob", "KAGGLE_KEY": "k1",
    }


def test_load_secrets_prefers_drive_then_local_then_env(tmp_path, monkeypatch):
    drive = tmp_path / "drive"; drive.mkdir()
    (drive / "secrets.env").write_text("ROBOFLOW_API_KEY=from-drive\n")
    local = tmp_path / "local.env"
    local.write_text("ROBOFLOW_API_KEY=from-local\nKAGGLE_USERNAME=lu\n")
    monkeypatch.setenv("KAGGLE_KEY", "from-env")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    cfg = Config(drive_root=drive, work_root=tmp_path, local_secrets=local)
    s = load_secrets(cfg)
    assert s == {"ROBOFLOW_API_KEY": "from-drive", "KAGGLE_USERNAME": "lu", "KAGGLE_KEY": "from-env"}


def test_load_secrets_missing_key_names_locations(tmp_path, monkeypatch):
    for k in ("ROBOFLOW_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = Config(drive_root=tmp_path / "nope", work_root=tmp_path, local_secrets=tmp_path / "no.env")
    with pytest.raises(SecretsError) as e:
        load_secrets(cfg)
    msg = str(e.value)
    assert "ROBOFLOW_API_KEY" in msg and "secrets.env" in msg
```

- [x] **Step 3: Run to verify failure** — `cd training && .venv/bin/python -m pytest tests/test_config.py` → import error (no `ppe.config`).

- [x] **Step 4: Implement `training/src/ppe/config.py`** (and empty `__init__.py`)

```python
"""Single source of truth for every fill-in variable and hyperparameter."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CANONICAL_CLASSES = [
    "Person", "helmet", "vest", "gloves", "boots", "goggles",
    "no_helmet", "no_gloves", "no_goggle", "gas mask",
]

SECRET_KEYS = ["ROBOFLOW_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]

# training/secrets.env (three levels up from this file: src/ppe/config.py -> training/)
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
    ppe_workspace: str = "datasetppe"          # fill-in; verified at runtime
    ppe_project: str = "ppe_detection-dnfen"
    ppe_version: int = 3
    gasmask_workspace: str = "daniil-yarmov"
    gasmask_project: str = "gas-masks"
    gasmask_version: int = 1                    # pinned: v1, 384 images, no substitute
    ocp_workspace: str = "yasser-dalali"
    ocp_project: str = "OCP"
    ocp_version: int = 1
    sh17_kaggle_slug: str = "mugheesahmad/sh17-dataset-for-ppe-detection"
    # --- Model B (hosted; never trained) ---
    generic_model_id: str = "ppe_detection-dnfen/3"
    generic_published: dict = field(default_factory=lambda: {"P": 58.1, "R": 64.4, "mAP50": 59.1})
    # --- split & sampling ---
    seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    ocp_dup_factor: int = 3        # train split only
    gasmask_dup_factor: int = 3    # train split only
    # --- training ---
    model_name: str = "yolov8n.pt"
    imgsz: int = 640
    batch: int = 16
    epochs: int = 50
    erasing: float = 0.4
    close_mosaic: int = 10
    save_period: int = 1
    run_name: str = "model_a"
    # --- evaluation ---
    eval_conf: float = 0.25
    eval_iou: float = 0.5
    camera_fps: int = 5

    @property
    def raw_dir(self) -> Path: return self.drive_root / "raw"
    @property
    def prepared_dir(self) -> Path: return self.drive_root / "prepared"
    @property
    def runs_dir(self) -> Path: return self.drive_root / "runs"
    @property
    def results_dir(self) -> Path: return self.drive_root / "results"
    @property
    def wheels_dir(self) -> Path: return self.drive_root / "wheels"
    @property
    def local_dataset_dir(self) -> Path: return self.work_root / "fused"


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_secrets(cfg: Config) -> dict[str, str]:
    """Drive secrets.env > training/secrets.env > environment variables."""
    locations = [cfg.drive_root / "secrets.env", cfg.local_secrets]
    merged: dict[str, str] = {}
    for loc in reversed(locations):          # lowest priority first
        if loc.is_file():
            merged.update({k: v for k, v in parse_env_file(loc).items() if v})
    for key in SECRET_KEYS:                  # env vars fill remaining gaps only
        merged.setdefault(key, os.environ.get(key, ""))
    # re-apply file priority: drive overrides local overrides env
    for loc in reversed(locations):
        if loc.is_file():
            merged.update({k: v for k, v in parse_env_file(loc).items() if v})
    missing = [k for k in SECRET_KEYS if not merged.get(k)]
    if missing:
        raise SecretsError(
            f"Missing secrets {missing}. Looked in: "
            f"{locations[0]} (Drive), {locations[1]} (local), then environment variables. "
            "Create secrets.env with KEY=value lines (see GUIDE.md); never commit it."
        )
    return {k: merged[k] for k in SECRET_KEYS}
```

- [x] **Step 5: Run to verify pass** — `cd training && .venv/bin/python -m pytest tests/test_config.py` → all pass.
- [x] **Step 6: Commit** — `git add -A && git commit -m "feat(training): scaffolding + config with secrets loading"`

---

### Task 2: Synthetic fixtures (conftest.py) + live source-name check script

**Files:**
- Create: `training/tests/conftest.py`, `training/scripts/check_sources.py`

**Interfaces:**
- Produces (conftest): `SH17_CLASSES` (17 real names), `write_png(path, w=32, h=32)`, `make_yolo_source(root, name, class_names, images, layout)` where `images` is `list[list[tuple[int, float, float, float, float]]]` (per image: list of (cls_id, cx, cy, w, h) normalized) and `layout` is `"flat"` (SH17-style `images/` + `labels/` + `data.yaml`) or `"roboflow"` (`train|valid|test/images|labels` + `data.yaml`). Returns source dir. Also fixture `fixture_sources(tmp_path)` building all 4 fake sources: `roboflow_ppe`, `sh17`, `gasmask`, `ocp`.
- Consumes: nothing.

- [x] **Step 1: Write `training/tests/conftest.py`**

```python
from pathlib import Path

import pytest
import yaml
from PIL import Image

# SH17's 17 real class names (Kaggle original, mugheesahmad).
SH17_CLASSES = [
    "person", "head", "face", "glasses", "face_mask_medical", "face_guard",
    "ear", "earmuffs", "hands", "gloves", "foot", "shoes", "safety_vest",
    "tools", "helmet", "medical_suit", "safety_suit",
]

PPE_FIXTURE_CLASSES = ["person", "helmet", "no-helmet", "vest", "no-vest"]
GASMASK_FIXTURE_CLASSES = ["gas_mask"]
OCP_FIXTURE_CLASSES = ["Person", "helmet", "vest", "gloves", "boots", "goggles", "gas mask"]


def write_png(path: Path, w: int = 32, h: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (127, 127, 127)).save(path)


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str, boxes) -> None:
    write_png(img_dir / f"{stem}.jpg".replace(".jpg", ".png"))
    lbl_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def make_yolo_source(root: Path, name: str, class_names: list[str], images, layout: str = "roboflow") -> Path:
    """images: list of box-lists; one image per entry, named <name>_<i>."""
    src = root / name
    if layout == "flat":
        for i, boxes in enumerate(images):
            _write_pair(src / "images", src / "labels", f"{name}_{i}", boxes)
    elif layout == "roboflow":
        n = len(images)
        cuts = {"train": range(0, max(n - 2, 1)), "valid": range(max(n - 2, 1), max(n - 1, 1)), "test": range(max(n - 1, 1), n)}
        for split, idxs in cuts.items():
            for i in idxs:
                _write_pair(src / split / "images", src / split / "labels", f"{name}_{i}", images[i])
    else:
        raise ValueError(layout)
    (src / "data.yaml").write_text(yaml.safe_dump({"nc": len(class_names), "names": class_names}))
    return src


@pytest.fixture
def fixture_sources(tmp_path):
    """All 4 fake sources with real-ish structures and class names."""
    root = tmp_path / "raw"
    box = (0, 0.5, 0.5, 0.4, 0.4)

    def boxes_for(class_names, per_image=2):
        return [[(i % len(class_names), 0.5, 0.5, 0.3, 0.3) for i in range(per_image)] for _ in range(10)]

    return {
        "roboflow_ppe": make_yolo_source(root, "roboflow_ppe", PPE_FIXTURE_CLASSES, boxes_for(PPE_FIXTURE_CLASSES)),
        "sh17": make_yolo_source(root, "sh17", SH17_CLASSES,
                                 [[(i, 0.5, 0.5, 0.2, 0.2)] for i in range(17)], layout="flat"),
        "gasmask": make_yolo_source(root, "gasmask", GASMASK_FIXTURE_CLASSES, boxes_for(GASMASK_FIXTURE_CLASSES, 1)),
        "ocp": make_yolo_source(root, "ocp", OCP_FIXTURE_CLASSES, boxes_for(OCP_FIXTURE_CLASSES)),
    }
```

- [x] **Step 2: Sanity-check fixtures import** — `cd training && .venv/bin/python -m pytest --collect-only` → collects, no errors.

- [x] **Step 3: Write `training/scripts/check_sources.py`** — live check (run manually once; prints class names ONLY, never keys):

```python
"""One-off: verify Roboflow slugs + fetch real class names to finalize NAME_MAPS.

Usage: .venv/bin/python scripts/check_sources.py   (needs requests; reads training/secrets.env)
Prints ONLY workspace/project/version and class names. Never prints keys.
"""
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.config import Config, load_secrets

cfg = Config(drive_root=Path("/nonexistent"))
key = load_secrets(cfg)["ROBOFLOW_API_KEY"]

for ws, proj, ver in [
    (cfg.ppe_workspace, cfg.ppe_project, cfg.ppe_version),
    (cfg.gasmask_workspace, cfg.gasmask_project, cfg.gasmask_version),
    (cfg.ocp_workspace, cfg.ocp_project, cfg.ocp_version),
]:
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}?api_key={key}"
    try:
        data = json.load(urlopen(url))
        classes = data.get("version", {}).get("classes") or data.get("project", {}).get("classes")
        images = data.get("version", {}).get("images")
        print(f"OK   {ws}/{proj}/{ver}: images={images} classes={sorted(classes) if isinstance(classes, dict) else classes}")
    except Exception as exc:  # noqa: BLE001 — diagnostic script
        print(f"FAIL {ws}/{proj}/{ver}: {type(exc).__name__}: {exc}")
```

- [x] **Step 4: Run the live check** and record the real class-name lists as comments in `remap.py`'s NAME_MAPS (Task 3). Fix `ppe_workspace` in config.py if the guess `datasetppe` is wrong (find the right workspace by opening the Universe page for `ppe_detection-dnfen`). Verify gas-masks v1 reports 384 images (README answer #1 evidence). Also confirm the SH17 Kaggle slug: `.venv/bin/pip install -q kaggle && KAGGLE_USERNAME=... KAGGLE_KEY=... .venv/bin/python -c "..."` — or just note it for Task 5's runtime validation.

- [x] **Step 5: Commit** — `git add -A && git commit -m "test(training): synthetic fixture builders + live source check script"`

---

### Task 3: remap.py — name-based remap, accounting, old-bug regression

**Files:**
- Create: `training/src/ppe/remap.py`, `training/tests/test_remap.py`

**Interfaces:**
- Consumes: `CANONICAL_CLASSES` from config; fixtures from conftest.
- Produces: `DROP = "DROP"`, `normalize_name(s) -> str` (lower, strip, spaces/hyphens→underscore), `RemapError`, `NAME_MAPS: dict[str, dict[str, str]]` for sources `roboflow_ppe|sh17|gasmask|ocp` (values = canonical class or DROP; keys normalized), `read_class_names(source_dir) -> list[str]` (data.yaml `names` list or dict), `RemapReport` dataclass (`source, images_before, images_after, boxes_before, boxes_after, boxes_dropped, dropped_by_class: dict, boxes_by_class: dict`, method `as_dict()`), `remap_source(source_dir, out_dir, name_map, source_name) -> RemapReport` (copies images, rewrites labels; empty label file = background image, kept), `remap_all(cfg_or_none, raw_dirs: dict[str, Path], out_root: Path) -> dict[str, RemapReport]` (writes `out_root/remap.json` = `{"classes": CANONICAL_CLASSES, "maps": {source: {orig: target}}, "reports": {source: report_dict}}`).

Key behaviors (all tested):
1. Map **by name**: label ids index into the source's own `names` list; the name is normalized and looked up in `name_map`. Unknown name → `RemapError` listing the unmapped names (this is what catches the old SH17 IDs≥10 bug — nothing is ever silently dropped).
2. Label id out of range for the source's names → `RemapError`.
3. DROP boxes removed + counted per class; their images kept with empty label files.
4. Accounting hard-fail: `images_before != images_after` or `boxes_before != boxes_after + boxes_dropped` → `RemapError`.
5. Handles both layouts: label files discovered recursively (`**/labels/*.txt`), image matched by stem in sibling `images/` dir.

SH17 table (explicit, human-readable — the whole point of the fix):

```python
SH17_NAME_MAP = {
    "person": "Person", "helmet": "helmet", "safety_vest": "vest",
    "gloves": "gloves", "shoes": "boots", "glasses": "goggles",
    # body parts / non-PPE → dropped, images kept as background
    "head": DROP, "face": DROP, "ear": DROP, "earmuffs": DROP,
    "face_mask_medical": DROP, "face_guard": DROP, "hands": DROP,
    "foot": DROP, "tools": DROP, "medical_suit": DROP, "safety_suit": DROP,
}
```

`PPE_NAME_MAP`, `GASMASK_NAME_MAP` (e.g. `{"gas_mask": "gas mask"}`), `OCP_NAME_MAP` (identity for present classes; `no_*` absent on OCP is fine — zero boxes from that source): seed from Task 2's live-check output; unknown-name hard-fail protects against drift.

Tests (`training/tests/test_remap.py`): normalized lookup; SH17 fixture remap produces expected ids + report numbers (17 images / 17 boxes in → 17 images, 6 kept boxes (person, glasses, gloves, shoes, safety_vest, helmet), 11 dropped); background image kept with empty label; **regression test**: a name_map missing the names for ids ≥ 10 (simulating the old 10-class assumption) raises `RemapError` mentioning the missing names — proving silent drops are impossible; out-of-range id raises; `remap_all` over `fixture_sources` writes valid `remap.json` with all 4 sources.

- [x] **Step 1: Write failing tests** (as listed above, concrete asserts)
- [x] **Step 2: Run** → fails (no module)
- [x] **Step 3: Implement `remap.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): name-based remap with hard-fail accounting + old-bug regression test`

---

### Task 4: merge.py — merge, deterministic split, duplication, holdouts, split_summary

**Files:**
- Create: `training/src/ppe/merge.py`, `training/tests/test_merge.py`

**Interfaces:**
- Consumes: remapped source dirs (output layout of `remap_source`: `<out>/<source>/images/*.png|jpg` + `<out>/<source>/labels/*.txt`, all label ids already canonical), `CANONICAL_CLASSES`, `Config` fields `seed/train_frac/val_frac/test_frac/ocp_dup_factor/gasmask_dup_factor`.
- Produces: `MergeError`, `SplitResult` dataclass (`dataset_dir: Path, ocp_test_dir: Path, proxy_dir: Path, summary: dict`), `merge_and_split(cfg, remapped: dict[str, Path], out_root: Path) -> SplitResult`, `zip_dir(src_dir: Path, zip_path: Path) -> Path`, `write_data_yaml(dest_dir, rel_train, rel_val, rel_test)`.

Behavior:
1. Filenames prefixed `{source}__{orig}` (provenance, no collisions).
2. Per-source deterministic split: sorted stems, `random.Random(cfg.seed).shuffle`, `n_train = int(n*train_frac)`, `n_val = int(n*val_frac)`, rest test. Copy into `train|val|test/{images,labels}`.
3. Duplication AFTER split, train only: sources `ocp` (factor `ocp_dup_factor`) and `gasmask` (`gasmask_dup_factor`); factor F = original + (F−1) copies suffixed `__dup1..__dup{F-1}`. Never val/test.
4. Eval-set dirs (images/ + labels/ + data.yaml with the 10 classes): `ocp_test/` = OCP test-split pairs; `industrial_proxy/` = SH17 + gasmask test-split pairs (no OCP — asserted).
5. Disjointness hard checks (raise `MergeError`): train∩val∩test stems disjoint per source (dup suffix stripped); every `ocp_test` stem absent from train; no `ocp__` file in proxy.
6. `split_summary.json` in `out_root`: seed, fractions, dup factors actually used, per-source per-split image counts (before and after duplication), per-split per-class box counts, eval-set sizes.
7. `data.yaml`: `names: CANONICAL_CLASSES`, `nc: 10`, relative `train/val/test` image paths.

Tests: split ratios on 20-image fixture source (14/3/3); determinism (two runs → identical file lists); duplication counts (10 ocp train images ×3 → 30, val/test untouched); ocp_test ∩ train = ∅; proxy has no ocp files; summary counts match filesystem; data.yaml names exactly CANONICAL_CLASSES; `zip_dir` round-trip.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `merge.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): deterministic merge/split with duplication, holdouts, split_summary`

---

### Task 5: download.py — 4 sources, cache under raw/, validation, fail-fast OCP

**Files:**
- Create: `training/src/ppe/download.py`, `training/tests/test_download.py`

**Interfaces:**
- Consumes: `Config`, secrets dict from `load_secrets`.
- Produces: `DownloadError`, `SourceStats` dataclass (`name, images, boxes, class_names, per_class_boxes`), `SOURCES = ["roboflow_ppe", "sh17", "gasmask", "ocp"]`, `ensure_source(cfg, secrets, name) -> Path` (skips when `raw/<name>/.complete` marker exists; writes marker after validation), `download_all(cfg, secrets) -> dict[str, Path]`, `validate_source(source_dir, name) -> SourceStats` (reads data.yaml names — for sh17, tolerate `classes.txt` fallback; counts images + boxes; label id in range check; readable `DownloadError` messages saying what/where/how to fix).
- Roboflow sources via lazy `from roboflow import Roboflow`; `rf.workspace(ws).project(p).version(v).download("yolov8", location=str(dest))`. SH17 via lazy `import kaggle` after setting `KAGGLE_USERNAME`/`KAGGLE_KEY` env from secrets; `kaggle.api.dataset_download_files(slug, path=dest, unzip=True)`.
- OCP failure (download exception, missing dir, empty) → `DownloadError` explicitly stating OCP is required, there is no OCP-less mode, and which workspace/project/version was attempted.

Tests (network mocked with `monkeypatch` on the private `_download_roboflow` / `_download_kaggle` helpers): skip-when-marker-present (downloader not called); marker written after success; validation stats on fixture source (image/box counts, per-class distribution); corrupted source (label id 99) → readable error naming the file; OCP failure message contains "OCP" + "no OCP-less mode"; gasmask images-count recorded (fixture: exact count check wiring for the 384 assertion at runtime — implemented as `stats.images` recorded into the returned SourceStats, real check is informational log not hard assert since fixture ≠ 384).

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `download.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): cached downloads for 4 sources with validation and fail-fast OCP`

---

### Task 6: predict.py — Boxes + Ultralytics/Roboflow predictor backends

**Files:**
- Create: `training/src/ppe/predict.py`, `training/tests/test_predict.py`

**Interfaces:**
- Consumes: nothing internal (numpy only; ultralytics/inference lazy).
- Produces:

```python
@dataclass
class Boxes:
    xyxy: np.ndarray            # (N,4) float, absolute pixels
    class_names: list[str]      # len N, backend's own names (mapping happens in evaluate)
    confidence: np.ndarray      # (N,)

class UltralyticsPredictor:
    def __init__(self, weights: Path, conf: float, iou: float, imgsz: int = 640): ...
    @property
    def class_names(self) -> list[str]: ...          # from model.names
    def predict(self, image_path: Path) -> Boxes: ...

class RoboflowPredictor:
    def __init__(self, model_id: str, api_key: str, conf: float, iou: float): ...
    @property
    def class_names(self) -> list[str]: ...          # discovered from model metadata / first inference
    def predict(self, image_path: Path) -> Boxes: ...
```

- `RoboflowPredictor` preferred path: lazy `from inference import get_model`, `model.infer(str(path), confidence=conf, iou_threshold=iou)`; predictions give center x/y/width/height in pixels → convert to xyxy. Fallback (constructor arg `backend="rest"` or automatic on ImportError): lazy `roboflow` hosted predict with `confidence=conf*100, overlap=iou*100`, same Boxes output. Known-risk isolation: both paths inside this one class.
- Tests: fake ultralytics result object → Boxes conversion (xyxy passthrough, names via model.names dict); fake inference response (center-format) → correct xyxy math; ImportError of `inference` falls back to REST path (mocked); conf/iou passed through to backend calls.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `predict.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): predictor backends for local .pt and hosted Roboflow model`

---

### Task 7: evaluate.py — shared harness + Model A extras

**Files:**
- Create: `training/src/ppe/evaluate.py`, `training/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Boxes` + predictor protocol from predict.py, `normalize_name` from remap.py, eval-set dirs from merge.py (`images/`, `labels/`, `data.yaml`), `Config` (`eval_conf`, `eval_iou`, `camera_fps`, `generic_published`).
- Produces:

```python
SHARED_CANDIDATES = ["person", "helmet", "vest", "gloves", "no_helmet", "no_gloves", "no_goggle"]

def shared_class_list(names_a: list[str], names_b: list[str]) -> list[str]
    # normalized, case-insensitive intersection of both models' names with SHARED_CANDIDATES,
    # returned in SHARED_CANDIDATES order ("Person" matches "person")

@dataclass
class EvalResult:
    precision: float; recall: float; map50: float
    n_images: int; shared_classes: list[str]

def load_eval_set(eval_set_dir: Path) -> list[tuple[Path, Boxes]]     # GT, denormalized via PIL image size
def match_counts(gt: Boxes, pred: Boxes, iou_thr: float) -> tuple[int, int, int]  # tp, fp, fn; greedy conf-desc, class-aware
def evaluate_predictor(predictor, eval_set_dir, shared: list[str], conf: float, iou_thr: float) -> EvalResult
    # filters GT and predictions to shared classes (normalized names), micro P/R from match_counts,
    # mAP50 via supervision MeanAveragePrecision over per-image Detections
def compare_all(cfg, predictor_ours, predictor_generic, eval_sets: dict[str, Path]) -> list[dict]
    # 6 rows: [{"model": "Generic", "eval_set": "Own val (construction)", "P": 58.1, ...}, Generic×proxy,
    #  Generic×ocp, Ours×fused_val, Ours×proxy, Ours×ocp] — eval_sets keys: "fused_val", "industrial_proxy", "ocp_test"
def cameras_at_fps(total_ms_per_frame: float, fps: int) -> int         # floor((1000/total_ms)/fps)
def model_a_val(cfg, best_pt: Path, data_yaml: Path) -> dict
    # lazy ultralytics: YOLO(best_pt).val(data=..., split="val") →
    # {"per_class": [{"class", "P", "R", "mAP50", "images"}...+ ALL row], "speed": {"preprocess","inference","postprocess","total"},
    #  "cameras": cameras_at_fps(...), "camera_formula": "floor((1000/total_ms)/fps)", "confusion_matrix_png": path}
    # per-class image counts computed from the val split's label files
```

- IoU + greedy matching implemented here (pure numpy, hand-verifiable); mAP50 delegated to `supervision` (`from supervision.metrics import MeanAveragePrecision` — adjust to the pinned version's API; pin whatever the venv resolved in Task 1).
- Tests vs hand-computed values: `match_counts` on crafted boxes (1 TP at IoU .6, 1 FP wrong class, 1 FN → P=0.5, R=0.5); perfect predictions → P=R=mAP50=1.0; empty predictions → P=0 (defined 0-safe), R=0; `shared_class_list(["Person","helmet","boots"], ["person","HELMET","vest"]) == ["person","helmet"]`; `cameras_at_fps(20.0, 5) == 10`; `compare_all` with two stub predictors returns exactly 6 rows, row 1 = published numbers as-is; `evaluate_predictor` restricts both GT and preds to shared classes (a boots-only GT image contributes no FN).

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `evaluate.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): shared eval harness with hand-verified P/R and supervision mAP`

---

### Task 8: train.py — Model A training with auto-resume

**Files:**
- Create: `training/src/ppe/train.py`, `training/tests/test_train.py`

**Interfaces:**
- Consumes: `Config` training fields, fused `data.yaml` path.
- Produces: `decide_train_action(run_dir: Path) -> str` (`"skip"` if `run_dir/.finished` and `weights/best.pt` exist; `"resume"` if `weights/last.pt` exists without `.finished`; else `"fresh"`), `train_model_a(cfg, data_yaml: Path) -> Path` (returns best.pt path; run dir = `cfg.runs_dir / cfg.run_name`).
- Fresh: lazy `from ultralytics import YOLO; YOLO(cfg.model_name).train(data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch, seed=cfg.seed, erasing=cfg.erasing, close_mosaic=cfg.close_mosaic, save_period=cfg.save_period, project=str(cfg.runs_dir), name=cfg.run_name, exist_ok=True)`. Resume: `YOLO(str(last_pt)).train(resume=True)`. After either completes, write `.finished` marker. Skip: return existing best.pt immediately.
- Checkpoints land on Drive because `project=cfg.runs_dir` points at Drive; dataset stays on local disk (`data_yaml` under `work_root`) — the notebook wiring (Task 11) restores the prepared zip to local disk first. Max loss on disconnect: current epoch (`save_period=1`).
- Tests: `decide_train_action` all three states via touched files; `train_model_a` with mocked YOLO class (monkeypatch `ppe.train._load_yolo`) asserts exact kwargs passthrough, `.finished` written on success, resume path constructs `YOLO(last_pt)` + `train(resume=True)`, skip path never instantiates YOLO.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `train.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): auto-resuming Model A training`

---

### Task 9: report.py — metrics.md autofill, README, handoff zip

**Files:**
- Create: `training/src/ppe/report.py`, `training/tests/test_report.py`

**Interfaces:**
- Consumes: comparison rows from `compare_all`, `model_a_val` dict, `split_summary.json` dict, `RemapReport` dicts, `Config`.
- Produces:

```python
def build_metrics_md(comparison: list[dict], per_class: list[dict], runtime: dict,
                     training_block: dict, shared_classes: list[str]) -> str
    # exact tables from engineering-requirements Step 6, every value filled;
    # notes the shared-class list and the P/R conf threshold; camera formula stated
def check_no_placeholders(text: str) -> None      # raises ReportError if "[ ]" or "__" survives
def build_readme(cfg, answers: dict) -> str
    # answers: gasmask_v1_confirmed(+image count), headline_split ("val"), epochs_gpu,
    # low_image_classes (computed: classes with < 200 train boxes, from split_summary),
    # alert_system ("designed only"); plus fixed notes: Model B hosted → no generic.pt; step 7 skipped
def build_handoff(cfg, artifacts: dict) -> Path
    # artifacts: best_pt, remap_json, data_yaml, metrics_md_text, readme_text, results_csv,
    # split_summary, dashboard_png, confusion_png → results/handoff-<YYYY-MM-DD>/ laid out per spec,
    # then zip_dir(...) → results/handoff-<date>.zip; returns handoff dir
def low_image_classes(split_summary: dict, threshold: int = 200) -> list[str]
```

- Handoff layout (spec): `best.pt, remap.json, data.yaml, metrics.md, results.csv, split_summary.json, figures/dashboard.png, figures/confusion_matrix.png, README.md`. `dashboard.png` = Ultralytics `results.png` copied/renamed; `confusion_matrix.png` = `confusion_matrix_normalized.png`.
- Tests: `build_metrics_md` output has zero `[ ]` and no bare `__` (checked via `check_no_placeholders`, which is itself tested to catch both); 6 comparison rows present incl. published Generic row; per-class table has all 10 classes + ALL; `build_readme` contains the 5 answers + "designed only" + "no generic.pt" + step-7 note; `build_handoff` produces exactly the spec layout and the zip exists on the (tmp) results dir; `low_image_classes` picks the right classes from a crafted summary.

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run** → fail
- [x] **Step 3: Implement `report.py`**
- [x] **Step 4: Run** → pass
- [x] **Step 5: Commit** — `feat(training): report generation with zero-placeholder guarantee + handoff zip`

---

### Task 10: Integration test — full synthetic pipeline, YOLO mocked

**Files:**
- Create: `training/tests/test_integration.py`

**Interfaces:** Consumes everything. No new production code (fix wiring bugs it finds, in place).

- Flow, mirroring exactly the notebook cell sequence: `fixture_sources` pre-placed under `raw/` with `.complete` markers (downloaders monkeypatched to fail loudly if called) → `download_all` (skips) → `remap_all` → `merge_and_split` → `train_model_a` with mocked YOLO that fabricates `weights/best.pt`, `results.csv`, `results.png`, `confusion_matrix_normalized.png` in the run dir → `evaluate` both models using stub predictors (`Ours` = stub reading GT labels back = perfect; `Generic` = stub emitting nothing) → `compare_all` → `build_metrics_md` → `build_handoff`.
- Asserts: every artifact exists exactly where the notebook expects (`prepared/fused.zip`, `runs/model_a/weights/best.pt`, `results/handoff-<date>/…` full layout); metrics.md passes `check_no_placeholders`; Ours rows show P=R=mAP50=1.0 (perfect stub) and Generic rows 0.0; `remap.json` + `split_summary.json` land inside the handoff.

- [x] **Step 1: Write the integration test** (expected to mostly pass; treat failures as wiring bugs and fix them)
- [x] **Step 2: Run full suite** — `cd training && .venv/bin/python -m pytest` → all green
- [x] **Step 3: Commit** — `test(training): end-to-end synthetic pipeline integration test`

---

### Task 11: colab_train.ipynb + GUIDE.md

**Files:**
- Create: `training/colab_train.ipynb` (write the JSON directly), `training/GUIDE.md`

**Notebook cells (~10, one call per step):**
1. *(md)* Title + "just Run all; re-running after a disconnect is safe and resumes".
2. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`.
3. Wheelhouse install: if `wheels/` empty → `pip download -r requirements.txt -d wheels/` once; then `pip install --no-index --find-links wheels/ -r requirements.txt` (falls back to plain `pip install -r` if offline install fails).
4. **Config cell** (the only cell a user edits): `sys.path.insert(0, f"{DRIVE_ROOT}/training/src")`, build `Config(drive_root=..., work_root=Path("/content/ppe-work"))`, `secrets = load_secrets(cfg)`.
5. Step 1: `paths = download_all(cfg, secrets)`.
6. Steps 2–3: if `prepared/fused.zip` missing → `remap_all` + `merge_and_split` + `zip_dir` to Drive; then always unzip to `cfg.local_dataset_dir` (local disk for speed).
7. Step 4: `best = train_model_a(cfg, local_data_yaml)`.
8. Step 5: build `RoboflowPredictor(cfg.generic_model_id, secrets["ROBOFLOW_API_KEY"], cfg.eval_conf, cfg.eval_iou)` + `UltralyticsPredictor(best, ...)`.
9. Step 6: `rows = compare_all(...)`; `extras = model_a_val(...)`.
10. Report: `build_metrics_md` + `build_readme` + `build_handoff`; print handoff path.

**GUIDE.md** (plain English, complete beginner): what each step does; creating the two free API keys (Roboflow account → settings → API key; Kaggle → account → Create New Token) and writing `secrets.env` (3 lines, exact format) into `MyDrive/ppe-training/`; uploading `training/` to Drive; opening the notebook in Colab + selecting T4 GPU runtime; what to edit in the config cell (usually nothing); OCP downloads automatically from Roboflow `yasser-dalali/OCP` v1 once the key is set; disconnect recovery = "Run all" again (why: idempotent steps + every-epoch checkpoints); where every output lands (Drive map from the spec); how to hand off `results/handoff-<date>.zip`.

- [x] **Step 1: Write GUIDE.md**
- [x] **Step 2: Write colab_train.ipynb** (nbformat-4 JSON; validate with `python -c "import json; json.load(open('colab_train.ipynb'))"`)
- [x] **Step 3: Smoke-import check**: run the notebook's config-cell code locally against a tmp drive_root (secrets from training/secrets.env) to prove the sys.path + imports wiring is correct.
- [x] **Step 4: Commit** — `docs(training): Colab notebook + beginner GUIDE`

---

### Task 12: Opt-in CPU smoke train + final pin/verification pass

**Files:**
- Create: `training/tests/test_smoke_train.py` (`@pytest.mark.slow`)
- Modify: `training/requirements.txt` (finalize pins)

- [x] **Step 1: Write the slow test**: synthetic 10-class dataset from fixtures → real `train_model_a` with `epochs=1, imgsz=64, batch=2, model="yolov8n.pt"` override → asserts best.pt exists + `.finished` marker + resume decision flips to "skip".
- [x] **Step 2: Install heavy deps into the venv** (`.venv/bin/pip install ultralytics`) and run once: `cd training && .venv/bin/python -m pytest -m slow tests/test_smoke_train.py` → proves the real Ultralytics train call works. If the local machine can't take torch, document in GUIDE.md that the smoke test runs in Colab via `!pytest -m slow`.
- [x] **Step 3: Default suite still green + slow excluded**: `.venv/bin/python -m pytest` collects everything except the slow test.
- [x] **Step 4: Finalize requirements.txt pins** from `pip freeze` (the versions that actually ran).
- [x] **Step 5: Commit** — `test(training): opt-in CPU smoke train + pinned requirements`

---

## Self-Review Notes

- Spec coverage: Steps 1–6 → Tasks 5,3,4,8,6,7; report → 9; notebook+GUIDE → 11; testing strategy → 2,10,12; secrets rule → 1; known-risk (inference pkg fallback) → 6; live slug verification → 2. Out-of-scope items excluded per spec.
- Type consistency: `Boxes` defined once (Task 6), consumed by Task 7; `RemapReport`/`remap.json` shape defined in Task 3, consumed by 9/10; eval-set dir contract (images/labels/data.yaml) defined in Task 4, consumed by 7; `eval_sets` keys `fused_val|industrial_proxy|ocp_test` fixed in Task 7 and used in 10/11.
- The `fused_val` eval set for `compare_all` is the fused dataset's `val/` split wrapped as an eval-set dir; `merge_and_split` writes it the same way as the two holdouts (add `fused_val_dir` to `SplitResult` — noted here so Task 4 includes it: **`SplitResult` also carries `fused_val_dir: Path`**, tested for existence and correct class list).
