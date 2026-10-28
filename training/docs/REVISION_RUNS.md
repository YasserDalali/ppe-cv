# Revision runs (ICCEAI 2026, paper 8) — training roadmap

Goal: one set of training runs produces every number the reviewers asked
for. Every model uses the same recipe; each differs from the next in one
thing only.

| Model | Trained on | Answers |
|---|---|---|
| **B** construction baseline | Roboflow construction set only (215 images) | our own benchmark, full settings known |
| **P** public only (OCP off) | construction + SH17 (1,500) + gas-mask | more public data |
| **S** size-matched public | P + 497 extra SH17 images (same size as A) | data volume alone |
| **A** ours (OCP on) | P + plant frames from 4 of 5 folds (×3) | plant pictures alone (S → A) |

Plant data: the 207 original frames only, split into 5 folds of whole
recordings (40–43 frames each). A is trained once per fold and tested on the
recordings it never saw. B, P and S never see plant frames, so they are
tested on all 207.

## Notebooks (`training/notebooks/`)

| Notebook | Runtime | Runs | ≈ T4 time |
|---|---|---|---|
| `job0_prepare` | CPU | builds `prepared/pool.zip`, re-scores the old model | 30 min |
| `job1_B_A012` | T4 GPU | B ×3 seeds, A folds 0–2, speed test | 3.6 h |
| `job2_S` | T4 GPU | S ×3 seeds | 3.4 h |
| `job3_A34` | T4 GPU | A folds 3–4 | 2.2 h |
| `job4_P` | T4 GPU | P ×3 seeds | 2.8 h |
| `collect` | CPU | all tables + figures | 10 min |

Order: **job0 first** (everything needs the pool). Then jobs 1–4 at the same
time, one Colab tab each. Then `collect`.

If GPU time runs short: jobs 1, 2 and 3 (B, S, A) answer the main remark of
all four reviewers. Job 4 (P) adds the OCP-off step.

## Before job0 — one manual step

Upload `~/Downloads/OCP.yolov8.zip` (207 frames, no augmentation) to
`MyDrive/ppe-training/raw_uploads/`. The code refuses any other export.

## Where results land (Drive, `ppe-training/`)

- `runs/<run_id>/weights/best.pt` — each trained model
- `results/revision/<run_id>/scores.json` — every score of that run
- `results/revision/final/` — `summary.md` plus the LaTeX tables and
  `pr_curves.pdf` for the paper

## If something stops

Reconnect and **Run all** again. Finished runs are skipped (their
`scores.json` exists); an unfinished run resumes from its last epoch.

## After training: ByteTrack demo (local, no GPU)

Download one fold model and run it on a video from that fold's test set
(`ocp_folds` table in `summary.md`):

```bash
cd training
.venv/bin/python scripts/track_demo.py --weights best_A_f0.pt \
  --video ~/Desktop/PPE-OCP/done/VID_20260716_164618.mp4 \
  --splits <pool>/splits.json --fold 0 --out track_demo/
```

## Regenerating the notebooks

`colab_train.ipynb` is the template. After editing it:
`.venv/bin/python scripts/make_job_notebooks.py`.
