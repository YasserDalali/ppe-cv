# PPE Training Pipeline — Design Spec

**Date:** 2026-07-16
**Status:** Approved by user (brainstorming session)
**Implements:** Steps 1–6 of `content/engineering-requirements.md`. Step 7 (LLM judge) is explicitly out of scope.
**Deadline context:** paper submission 2026-07-23 (I2CEAI 2026); camera-ready 2026-09-23 allows number refresh.

## Goal

Static, thoroughly-tested Python code plus a thin Colab notebook that — once the OCP
dataset link is filled in — downloads 4 datasets, fixes the SH17 remap bug, merges and
splits a 10-class fused dataset, trains Model A (fused) on a Colab T4 with
disconnect-proof checkpointing, evaluates Model A and the Roboflow-hosted generic
Model B through one shared harness, and writes every number/figure the paper needs
into a handoff results directory. Plus a plain-English GUIDE.md.

## User decisions (recorded)

1. **SH17 source = Kaggle original** (mugheesahmad's SH17), real class names, remap by
   name. Requires Kaggle API key (fill-in variable).
2. **Model B is NOT trained.** The existing Roboflow-hosted generic model
   (DatasetPPE `ppe_detection-dnfen`, published own-val P 58.1 / R 64.4 / mAP50 59.1)
   is run by inference over our test splits and scored against our ground truth.
   Consequence: no `generic.pt` in the handoff (stated in README.md); comparison-table
   metrics for BOTH models come from one shared evaluation harness.
3. **Architecture = `src/` package + thin notebook.** All logic in importable modules;
   the notebook holds only the config cell and one call per step.
4. Design approved as-is, including keeping the CPU-training smoke test behind an
   opt-in pytest marker (excluded from the default suite) rather than deleting it.

## Code layout — `training/` in this repo

```
training/
  src/ppe/
    config.py       # dataclass: every fill-in variable and hyperparameter
    download.py     # Step 1 — fetch & cache all 4 sources
    remap.py        # Step 2 — SH17 name-based remap, remap.json, drop accounting
    merge.py        # Step 3 — merge, OCP weighting, gas-mask oversample, split, holdouts
    train.py        # Step 4 — Model A training with auto-resume
    predict.py      # backends: UltralyticsPredictor (.pt) / RoboflowPredictor (hosted)
    evaluate.py     # Step 6 — shared eval harness (supervision) + Ultralytics val for A
    report.py       # metrics.md autofill, figures, README.md, split summary, handoff zip
  tests/            # pytest; synthetic fixture datasets; no real training by default
  colab_train.ipynb # ~10 cells: mount Drive → config → steps in order
  requirements.txt  # pinned versions (ultralytics, roboflow, inference, supervision,
                    # kaggle, gdown, pytest, ...)
  GUIDE.md          # plain-English walkthrough
```

## Config (single source of truth, mirrored as the notebook's config cell)

```python
ROBOFLOW_API_KEY = ""        # user fills (free account)
KAGGLE_USERNAME  = ""        # user fills (SH17 original)
KAGGLE_KEY       = ""
OCP_DRIVE_LINK   = ""        # ← user fills when the OCP zip is ready
GENERIC_MODEL_ID = "ppe_detection-dnfen/3"   # Roboflow hosted Model B (version verified at runtime)
DRIVE_ROOT       = "/content/drive/MyDrive/ppe-training"
SEED             = 0         # all splits deterministic
# training: model=yolov8n, imgsz=640, batch=16, epochs=50,
#           erasing=0.4, close_mosaic=10, save_period=1
```

- Pipeline **fails fast** with a readable message if `OCP_DRIVE_LINK` is empty — OCP
  images are part of the fusion and the headline test set; there is no OCP-less mode.
- Final classes, exactly these 10, in this order, no `none`:
  `Person, helmet, vest, gloves, boots, goggles, no_helmet, no_gloves, no_goggle, gas mask`

### Required OCP zip structure (validated with readable errors)

```
ocp_dataset.zip
  images/     *.jpg / *.png
  labels/     *.txt        # YOLO format, using OUR 10-class IDs
  classes.txt              # the 10 class names, one per line (sanity check)
```

Validation: classes.txt must match our class list; every label file's class IDs ∈ 0–9;
report image/box counts and per-class distribution on load.

## Drive workspace — durability layer

```
MyDrive/ppe-training/
  wheels/     # pip wheelhouse: first run `pip download`s pinned deps; later sessions
              # install offline from here in seconds
  raw/        # downloaded dataset zips/dirs — downloaded once, ever
  prepared/   # fused+split dataset as one zip — built once, ever
  runs/       # Ultralytics project dir; last.pt/best.pt written EVERY epoch (save_period=1)
  results/    # handoff folder + handoff zip
```

**Idempotency rule:** every step checks Drive for its finished artifact and skips
itself if present. Recovery from a Colab disconnect = "Run all" again: downloads skip,
merge skips, training auto-resumes from `last.pt` (detect unfinished run → Ultralytics
`resume=True`). Training reads the dataset from local `/content` disk (restored from
the `prepared/` zip) for speed but writes checkpoints directly to Drive. Maximum loss
on disconnect: the current epoch.

## Step-by-step behavior

### Step 1 — download (`download.py`)
- Roboflow PPE_Detection (DatasetPPE `ppe_detection-dnfen`) and gas-masks v1
  (Daniil Yarmov `gas-masks/1`, 384 images — exact version pinned in code, per the
  requirements "use this one, don't pick another") via the `roboflow` package.
- SH17 original from Kaggle by dataset slug.
- OCP zip from the user's Drive link via `gdown`.
- All cached under `raw/`; skip when present.

### Step 2 — remap (`remap.py`)
- Read SH17's real class names from its own metadata; map by **name** with an explicit
  human-readable table in code (e.g. `helmet→helmet`, `safety_vest→vest`,
  body-part/tool classes → DROP). Also verify/remap Roboflow PPE class names and map
  gas-mask dataset → `gas mask`. OCP is assumed already in our IDs (validated).
- Rewrite every label file. Count images and boxes before/after; **hard-fail if any
  image or box disappears unaccounted for**. Boxes of DROPped classes are logged and
  removed; their images remain as background images (empty label file is valid YOLO).
- Outputs: `remap.json` (the mapping) + drop-accounting report.
- A regression test simulates the old bug (SH17 IDs ≥ 10 unmapped → silent drops) and
  proves the accounting catches it.

### Step 3 — merge + split (`merge.py`)
- One 10-class dataset; filenames prefixed by source (provenance + no collisions).
- Seeded deterministic 70/15/15 train/val/test split, stratified per source.
- OCP sampling weight and gas-mask oversampling implemented by **physical file
  duplication** with suffixed names (verifiable counts, no custom sampler).
  Duplication factors live in config; defaults: OCP ×3, gas-mask ×3 (train split
  only — val/test are never duplicated). Actual factors used are recorded in
  `split_summary.json`.
- Held-out sets: **OCP test** (never in train, disjointness asserted) and
  **industrial proxy** (SH17 + gas-mask leftovers, no OCP).
- Output: fused dataset + `data.yaml` + `split_summary.json` with exact counts per
  source/class/split (the requirements demand written-down split numbers).

### Step 4 — train Model A (`train.py`)
- YOLOv8n, imgsz 640, batch 16, epochs 50, HSV + flip defaults, `erasing=0.4`,
  mosaic on with `close_mosaic=10` (off last 10 epochs), `save_period=1`,
  `project=<Drive>/runs`, seeded. Auto-resume when an unfinished run exists.
- Output: `best.pt`, `results.csv`, Ultralytics plots.

### Step 5 — Model B (`predict.py`)
- No training. `RoboflowPredictor` runs `GENERIC_MODEL_ID` over test images —
  preferred path: `inference` package (model runs locally in Colab, no per-image API
  metering); fallback: hosted REST API. Model B's class names are read at runtime and
  mapped to our shared-class names.
- Published own-val numbers (P 58.1 / R 64.4 / mAP50 59.1) are inserted into the
  comparison table as-is, per the requirements.

### Step 6 — evaluate (`evaluate.py`)
- **Shared harness** for the comparison table: for each (model, eval set) pair —
  fused val, industrial proxy, OCP test × {Generic, Ours} — run the model's predictor
  with identical conf/IoU, restrict predictions AND ground truth to the **shared
  classes** (person, helmet, vest, gloves + no_* variants, intersected at runtime
  with Model B's actual class list; name matching is case-insensitive, so our
  `Person` matches their `person`; the final shared list is recorded in
  `metrics.md`), compute P / R / mAP50 with the `supervision` library. Same metric
  code for both models.
- **Model A extras** via Ultralytics `model.val()`: per-class P/R/mAP50 table
  (with per-class image counts), normalized confusion matrix PNG, and speed numbers
  (preprocess / inference / postprocess ms → total ms/frame → "~N cameras at X FPS"
  derivation, formula stated in output).

### Report (`report.py`)

```
results/handoff-<date>/
  best.pt, remap.json, data.yaml
  metrics.md          # every [ ] filled programmatically: comparison table (6 rows),
                      # per-class table, runtime block, training block
  results.csv
  split_summary.json
  figures/dashboard.png            # losses + P/R/mAP curves
  figures/confusion_matrix.png     # normalized
  README.md           # the 5 required answers auto-filled:
                      #   gas-masks v1 confirmed; val-vs-test stated; epochs+GPU;
                      #   low-image classes listed; alert system: "designed only".
                      # Also states: Model B hosted → no generic.pt; step 7 skipped.
  → zipped to Drive as handoff-<date>.zip
```

`metrics.md` generation must leave **zero** unfilled `[ ]` placeholders (tested).

## Testing strategy (no real training by default)

- **Synthetic fixtures:** miniature fake datasets (tiny random images + YOLO labels)
  mimicking each source's real structure, including a fake SH17 with its 17 real
  class names and a fake OCP zip.
- **Unit tests:** remap correctness + old-bug regression; split ratios; OCP-test ∩
  train = ∅; duplication counts; evaluator vs hand-computed P/R/mAP on crafted
  predictions; metrics.md has no `[ ]` left; OCP zip validation errors are readable.
- **Integration test:** full pipeline over synthetic sources with `YOLO.train`
  mocked — verifies wiring and that every artifact appears where the notebook
  expects it.
- **Opt-in CPU smoke train** (`@pytest.mark.slow`, excluded by default): real
  YOLOv8n train, 1 epoch, tiny imgsz, synthetic data. Run once during implementation
  to prove the train call works; kept opt-in for future Ultralytics upgrades.

## GUIDE.md

Written for a complete beginner: what each pipeline step does in plain language; how
to create the two free API keys (Roboflow, Kaggle); how to upload `training/` to
Drive and open the notebook in Colab; selecting the T4 runtime; what to fill in the
config cell (including the OCP link, later); why "just Run all again" is the entire
disconnect-recovery procedure; where every output lands; how to hand off the zip.

## Known risk

Model B's local-inference path depends on Roboflow's `inference` package supporting
that Universe model in Colab. Likely fine; if licensing/packaging blocks it, fallback
is the hosted REST API over the same test images (slower, metered). Isolated behind
the `RoboflowPredictor` interface so the swap touches one file. Verified early in
implementation, not on training day.

## Out of scope

- Step 7 (LLM judge / Ollama / Qwen) — skipped for now, noted in README.md.
- Alert dispatch, tracking, access control — "designed only" per requirements.
- Editing the paper (`paper/`) — this work only produces numbers/figures for it.
