# Preparing the dataset locally instead of in Colab

Steps 1–3 of the pipeline (download the 4 raw sources, remap labels onto the
canonical 10-class list, merge + split) can run on your own machine instead
of inside Colab. This avoids re-running the flaky Roboflow/Kaggle downloads
every time you open a fresh Colab VM, and lets you inspect `remap.json` /
`split_summary.json` before committing GPU time to training.

The notebook already trusts whatever it finds on Drive: if
`prepared/fused.zip` is there, it skips rebuilding and just restores it to
local disk (this was already how disconnect-recovery worked — see
`ensure_prepared()` in `training/src/ppe/merge.py`). Steps 1–3 in
`colab_train.ipynb` were updated (2026-07-18) so **Step 1's download also
gets skipped** when `fused.zip` is already cached, instead of redundantly
re-downloading all 4 raw sources on every Colab run.

## 1. One-time: fill in real API keys

`training/secrets.env` exists but currently has empty placeholders:

```
ROBOFLOW_API_KEY=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

Fill in real values (see `GUIDE.md` → "Create the two free API keys" for
where to get them). This file is gitignored — never commit it.

## 2. Run the local prepare script

```bash
cd /home/yasser/ppe-cv/training
.venv/bin/python scripts/prepare_local.py
```

Optionally pass `--out DIR` to choose where the local staging folder goes
(defaults to `~/ppe-training-local`). This runs `download_all` →
`remap_all` → `merge_and_split` (the same code Colab runs), producing:

```
<out>/raw/                the 4 downloaded datasets (with .complete markers)
<out>/prepared/fused.zip  the merged+split dataset
<out>/_work/fused/        a local extracted copy — not needed on Drive
```

Expect this to take a while: it downloads ~4 datasets and zips the fused
result. Progress bars print throughout (this is the same code that had
`tqdm` added across the pipeline for exactly this kind of long-running,
easy-to-mistake-for-a-hang local run).

If it fails partway through (e.g. the DatasetPPE Roboflow CDN bug — see
`GUIDE.md` § 5), fix the underlying issue and re-run: already-downloaded
sources are skipped via their `.complete` marker, and remap/merge only
start once all 4 raw sources are present.

## 3. Upload the result to Google Drive

Upload these two folders into your Drive's `ppe-training/` folder, merging
with (not replacing) whatever is already there — in particular, don't
overwrite `MyDrive/ppe-training/secrets.env`:

```
<out>/raw/       -> MyDrive/ppe-training/raw/
<out>/prepared/  -> MyDrive/ppe-training/prepared/
```

Do this however you'd normally move files to Drive (drag-and-drop in the
Drive web UI, the Drive desktop sync folder, or `rclone` if you have a
Drive remote already configured, e.g.
`rclone copy <out>/prepared "gdrive:ppe-training/prepared" --progress`).
The script's own final printout repeats the exact paths to upload.

You do **not** need to upload `<out>/_work/` — that's just the local
extracted copy the script keeps around; the notebook rebuilds its own from
`prepared/fused.zip`.

## 4. Run the notebook as usual

Open `colab_train.ipynb` in Colab and *Runtime → Run all*. You should see:

- Step 1 cell prints `prepared/fused.zip already on Drive — skipping raw
  download` and does **not** call Roboflow/Kaggle.
- Steps 2+3 cell prints `fused.zip: cached on Drive — skipping rebuild` and
  just restores the dataset to local disk.

`secrets.env` on Drive still needs a non-empty `ROBOFLOW_API_KEY` (Step 5
uses it to call the hosted Model B). The Kaggle keys aren't used in this
skip path but `load_secrets()` still requires all three keys to be
non-empty, so leave placeholder values in if you don't want the real ones
on Drive.

## Notes

- `scripts/prepare_local.py` and `ensure_prepared()` are the same code, just
  pointed at a local `drive_root` instead of Colab's Drive mount — see
  `training/tests/test_integration.py` for the same pattern used in tests.
- If you ever need to force a rebuild (e.g. after changing `remap.py`'s
  name maps or `config.py`'s split fractions), delete
  `MyDrive/ppe-training/prepared/fused.zip` from Drive — Steps 1–3 in the
  notebook will fall back to the normal from-scratch build.
