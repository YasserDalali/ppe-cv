# GUIDE — running the PPE training pipeline (no experience needed)

This folder trains **Model A** (our fused PPE detector), evaluates it against
the hosted generic **Model B**, and produces a single zip with every number
and figure the paper needs. You run it in Google Colab; everything important
is saved to your Google Drive as it goes, so a disconnect never loses more
than the current training epoch.

## What each step does (plain language)

1. **Download** — fetches 4 datasets into `Drive/ppe-training/raw/`:
   the public Roboflow PPE set, SH17 (from Kaggle), the gas-masks set
   (v1, 384 images — fixed, don't substitute), and the OCP site photos
   (from Roboflow project `OCP`). Each source is downloaded once, ever.
2. **Remap** — rewrites every label file onto our single 10-class list
   (`Person, helmet, vest, gloves, boots, goggles, no_helmet, no_gloves,
   no_goggle, gas mask`) by **class name**. If anything can't be mapped the
   run stops loudly — no silent data loss (the old run lost 43% of images
   this way). The mapping is saved as `remap.json`.
3. **Merge + split** — one fused dataset, split 70/15/15 train/val/test
   with a fixed seed (same split every run). OCP and gas-mask train images
   are physically duplicated ×3 so the model sees them more often. Two extra
   held-out sets are built: **ocp_test** (OCP images never trained on) and
   **industrial_proxy** (SH17 + gas-mask test images, no OCP). Exact counts
   land in `split_summary.json`.
4. **Train Model A** — YOLOv8n, 640×640, batch 16, 50 epochs on the Colab T4.
   Checkpoints are written to Drive **every epoch**.
5. **Model B** — NOT trained. The existing hosted Roboflow model
   (`ppe_detection-dnfen/1`) is run over the same test images.
6. **Evaluate** — the exact same scoring code runs both models on the same
   images and classes, filling the comparison table, the per-class table,
   speed numbers, and the confusion matrix.
   Finally everything is zipped into `results/handoff-<date>.zip`.

Step 7 (the LLM judge) is intentionally **not** part of this pipeline.

## One-time setup (~15 minutes)

### 1. Create the two free API keys

- **Roboflow:** sign up at https://app.roboflow.com → click your workspace
  icon (top-left) → *Settings* → *API Keys* → copy the **Private API Key**.
- **Kaggle:** sign in at https://www.kaggle.com → click your avatar →
  *Settings* → *API* section → **Create New Token**. A `kaggle.json` file
  downloads; open it in any text editor — it contains your `username` and `key`.

### 2. Create the Drive folder and secrets file

In Google Drive, create a folder **`ppe-training`** directly inside *My Drive*.
Inside it create a plain text file named **`secrets.env`** with exactly three
lines (paste your own values):

```
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxx
KAGGLE_USERNAME=yourkagglename
KAGGLE_KEY=xxxxxxxxxxxxxxxx
```

Never put these keys in code, notebooks, or git. This file is the only place
they live.

### 3. Add the private GitHub token to Colab Secrets

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate**.
2. Resource owner: **YasserDalali**. Repository access: **Only select
   repositories** → `ppe-cv`.
3. Permissions → Repository → **Contents: Read-only**. Generate and copy.
4. In Colab, open the key icon (**Secrets**), add name `GITHUB_TOKEN`, paste
   the token, and enable **Notebook access** for this notebook.

Never put this token in Drive, chat, the notebook, or Git. If it was ever
pasted outside Colab Secrets, revoke it and create a new one.

### 4. Open the notebook

Download `training/colab_train.ipynb` from the private GitHub repository, then
in https://colab.research.google.com choose *File → Upload notebook*. Only the
small notebook is uploaded. It clones the current `main` branch into the VM
on every Run All. Do not upload `training/` to Drive; Drive stores only
secrets, datasets, checkpoints, and results.

### 5. Fix the DatasetPPE download (required — Universe CDN is broken)

Roboflow’s public project `datasetppe/ppe_detection-dnfen` still lists a
YOLOv8 export, but every signed download URL returns GCS **NoSuchKey**
(a ~250-byte XML error that the SDK then raises as `BadZipFile`). Gas-masks
and your own OCP project download fine; this is specific to DatasetPPE.

**Do one of the following before Run all:**

**A (preferred). Fork into your workspace**
1. Open https://universe.roboflow.com/datasetppe-8juj2/ppe_detection-dnfen
2. *Use this Dataset* / download into workspace **yasser-dalali**.
3. *Generate* → create a version with plain **YOLOv8** export.
4. In Colab’s config cell (or `training/src/ppe/config.py` on `main`), set
   `ppe_workspace`, `ppe_project`, `ppe_version` to that fork’s slugs
   (project slug is under *Settings*, not the display name).

**B. Manual zip drop on Drive**
1. On the Universe page, download **YOLOv8** in the browser (or from a
   fork that exports successfully).
2. Unzip into `MyDrive/ppe-training/raw/roboflow_ppe/` so `data.yaml`
   sits at that folder’s root (train/valid/test underneath).
3. Create an empty file `MyDrive/ppe-training/raw/roboflow_ppe/.complete`
   so Step 1 skips the broken CDN download.

### 6. OCP dataset version

Project `yasser-dalali/ocp-snzjw` — use **version 2** (already generated;
yolov8 export works). Version 1 does not exist. If you recreate the
project, Generate a version with plain YOLOv8 and set `ocp_version`
accordingly.

## Running it

1. Open the notebook in Colab.
2. Menu *Runtime → Change runtime type* → Hardware accelerator: **T4 GPU**.
3. Menu *Runtime → Run all*. Approve the Drive-access popup.
4. Wait. Training 50 epochs on a T4 typically takes a few hours. You can
   close the tab; Drive keeps the checkpoints.

**If Colab disconnects:** reconnect and *Run all* again. Code is cloned fresh
from GitHub, downloads and dataset-building skip themselves, and training
resumes from the last saved epoch. Worst case you lose the epoch in progress.

## Where everything lands (on Drive, under `ppe-training/`)

```
raw/        the 4 downloaded datasets (downloaded once, ever)
prepared/   fused.zip — the merged+split dataset (built once, ever)
runs/       Ultralytics training runs; last.pt/best.pt saved every epoch
results/    handoff-<date>/ and handoff-<date>.zip  ← the deliverable
```

## Handing off

Send the authors the single file `results/handoff-<date>.zip`. It contains
`best.pt`, `remap.json`, `data.yaml`, the fully-filled `metrics.md`,
`results.csv`, `split_summary.json`, the training dashboard and confusion
matrix figures, and a `README.md` answering the 5 required questions
(including: Model B is the hosted Roboflow model, so there is **no
generic.pt**, and Step 7 / LLM judge was skipped).

## For developers

Local test suite (no GPU, no network, no real training):

```bash
cd training
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest            # fast suite
.venv/bin/python -m pytest -m slow    # opt-in: real 1-epoch CPU smoke train
```

`inference` (Roboflow local Model B) is **not** in `requirements.txt` — its pins
fight Colab / ultralytics / supervision. Model B uses the hosted REST API
fallback automatically.

`scripts/check_sources.py` verifies the Roboflow slugs/class names with your
key (prints names only, never keys).
