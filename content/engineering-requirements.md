# Junior Dev Guide — Fix the PPE Paper Numbers

**Your job:** train the models, get the numbers + figures, zip them up, send them back.  
**Deadline:** need this before **July 23, 2026**.  
**You do NOT write the paper.** Authors paste your numbers into LaTeX.

---

## TL;DR — do these 8 things in order

1. Download 4 datasets
2. Remap SH17 labels (IMPORTANT — old run broke here)
3. Merge into one 10-class dataset + split train/val/test
4. Train **Model A** (fused / “ours”)
5. Train or load **Model B** (Roboflow-only / “generic”)
6. Eval both on the same test images → fill the comparison table
7. Run the LLM judge on 3–5 examples
8. Zip everything in “What to send back”

---

## What we’re building (1 minute)

```
CCTV frame → YOLOv8 detects PPE → Qwen judge says confirm/reject → alert
```

**The one experiment that matters:**


| Model           | Trained on         | Test on                       |
| --------------- | ------------------ | ----------------------------- |
| **B = Generic** | Roboflow PPE only  | OCP images + industrial proxy |
| **A = Ours**    | All datasets fused | Same images as B              |


We want to show **A beats B** on plant images.

---

## Step 1 — Get the data


| #   | Dataset | What for |
| --- | ------- | -------- |
| 1   | [Roboflow PPE-Detection](https://universe.roboflow.com) | Generic model + part of fusion |
| 2   | **SH17** | Industrial PPE classes |
| 3   | [**Roboflow gas masks v1**](https://universe.roboflow.com/daniil-yarmov/gas-masks/dataset/1) (Daniil Yarmov, **384 images**) | Add `gas mask` class — **use this one, don’t pick another** |
| 4   | **OCP site images** (ask Amine/Yasser for access) | Real plant data — keep some images **out of training** for the test |


**Final classes (exactly these 10 — no `none`):**

```
Person, helmet, vest, gloves, boots, goggles,
no_helmet, no_gloves, no_goggle, gas mask
```

---

## Step 2 — Fix SH17 remap (don’t skip)

Old merge **threw away ~43% of images** because SH17 uses class IDs 0–16 and we only have 10 classes.

**Do this:**

1. Look up SH17’s real class names (not just numbers)
2. Make a map: `SH17 class → our class` (or drop body-part classes)
3. Rewrite every label file with the new IDs
4. **Then** merge

Save your map as `remap.json`.  
If you skip this, the paper numbers are wrong.

---

## Step 3 — Merge + split

1. **Merge all 4 sources into one YOLO dataset**
2. Give **OCP images higher sampling weight** (sample them more often)
3. **Oversample** gas-mask images
4. Split roughly **70 / 15 / 15** train / val / test (stratified if you can)
5. Hold out some **OCP images** that never appear in train → this is the **OCP test set**
6. Also keep a **proxy test set** = leftover SH17 + gas-mask images (no OCP) for publishable results

Write down the exact split % and image counts.

---

## Step 4 — Train Model A (ours)

```text
Model:     YOLOv8n
Size:      640×640
Batch:     16
Epochs:    50
GPU:       Colab T4 is fine
Aug:       HSV + flip + erase + mosaic (turn mosaic OFF last ~10 epochs)
Output:    best.pt
```

Use Ultralytics. Save the full training folder (`results.csv`, plots, etc.).

**Do not reuse old numbers from `content/metrics.txt`.** That run is dead.

---

## Step 5 — Model B (generic)

Train YOLOv8n **only** on Roboflow PPE-Detection  
(or download their pretrained weights if you already have them).

Same image size / confidence settings as Model A when you evaluate.

Their published own-val numbers (keep as-is for the paper):

- P 58.1 · R 64.4 · mAP50 59.1

---

## Step 6 — Evaluate (fill this table)

Use the **same** test images and settings for both models.  
Only score **shared classes**: person, helmet, vest, gloves + no_* versions.

Copy this into a file called `metrics.md` and fill every `[ ]`:

```markdown
## Comparison (THE important table)

| Model | Eval set | P % | R % | mAP50 % |
|-------|----------|-----|-----|---------|
| Generic | Own val (construction) | 58.1 | 64.4 | 59.1 |
| Generic | Industrial proxy | [ ] | [ ] | [ ] |
| Generic | OCP site | [ ] | [ ] | [ ] |
| Ours | Fused val | [ ] | [ ] | [ ] |
| Ours | Industrial proxy | [ ] | [ ] | [ ] |
| Ours | OCP site | [ ] | [ ] | [ ] |

## Per-class (Ours, on val or test — say which)

| Class | P % | R % | mAP50 % | # images |
|-------|-----|-----|---------|----------|
| Person | | | | |
| helmet | | | | |
| no_helmet | | | | |
| vest | | | | |
| gloves | | | | |
| no_gloves | | | | |
| goggles | | | | |
| no_goggle | | | | |
| boots | | | | |
| gas mask | | | | |
| ALL | | | | |

## Runtime (same GPU you used)

- preprocess: __ ms
- inference: __ ms
- postprocess: __ ms
- total: __ ms / frame
- ~__ cameras at __ FPS each

## Training

- epochs: __
- ultralytics version: __
- GPU: __
- split used for headline numbers: val OR test? __
```

Also export:

- training dashboard image (losses + P/R/mAP curves)
- normalized confusion matrix PNG

---

## Step 7 — LLM judge (easy qualitative part)

Install **Ollama**, pull **Qwen2.5-VL**.

For **3–5** detections from the model:

1. Send crop + full frame
2. Ask it to reply JSON: `confirm` / `reject` / `recheck` + one-sentence reason

**Must include:**

- at least 1 fake alarm the judge **rejects** (glare / occlusion)
- at least 1 real violation it **confirms**

Prompt to use (copy-paste):

```
You are a safety-compliance verifier for industrial CCTV. A PPE detector flagged the attached image region as "<class>" with confidence <conf>; the full frame is provided for context. Decide whether this is a genuine violation. Reason about: occlusion by equipment or other workers, reflections and glare, distance and image resolution, lighting and dust conditions, and whether pose or clothing could mimic a violation. Reply in JSON: {"verdict": "confirm"|"reject"|"recheck", "confidence": 0-1, "reason": "<one sentence>"}.
```

Save results as `judge_cases.csv`:

```text
case_id, flagged_class, verdict, reason, crop_file
```

**Do NOT invent a “false positives reduced by X%” number.** Just the cases.

---

## What to send back

One zip folder: `handoff-YOURNAME-DATE.zip`

```
handoff/
  best.pt                 # Model A
  generic.pt              # Model B
  remap.json              # SH17 → our classes
  data.yaml
  metrics.md              # filled tables above
  results.csv             # ultralytics training log
  figures/
    dashboard.png
    confusion_matrix.png
    case-1.jpg ... case-N.jpg
  judge_cases.csv
  README.md               # see below
```

### `README.md` — answer these 5 questions

1. Confirm you used gas masks v1: https://universe.roboflow.com/daniil-yarmov/gas-masks/dataset/1 (384 images)
2. Headline numbers from **val** or **test**?
3. Epochs + GPU used?
4. Any class with almost no images? (list them)
5. Alert system: did you build it, or is it just “designed on paper”? → say **designed only** unless you really built it

---

## Done when

- [ ] Remap done (no silent image drops)
- [ ] `gas mask` in the class list, no `none`
- [ ] Comparison table fully filled (no blanks)
- [ ] Per-class table filled
- [ ] New dashboard + confusion matrix
- [ ] Runtime numbers written
- [ ] ≥3 judge cases with reasons
- [ ] Zip sent to authors

---

## Don’t do these

- Don’t reuse old `content/metrics.txt` numbers
- Don’t publish raw OCP faces without asking (blur or skip photos)
- Don’t claim the judge reduces false alarms by a %
- Don’t add fancy features (tracking, access control, etc.) — out of scope

---

## Stuck?

Ask for: OCP dataset access or Colab GPU help.  
Gas-mask dataset is fixed (link in Step 1) — don’t substitute another.  
Paper draft lives in `paper/sections/` — you don’t need to edit it.