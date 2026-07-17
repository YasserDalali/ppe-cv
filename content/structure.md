Here's a detailed structure for your paper, modeled directly on the reference papers' conventions (numbered sections, related-work comparison table, methodology broken into acquisition→preprocessing→model→judge/decision stages, results with tables/figures, discussion of trade-offs, conclusion/future work). Intro statistics and citations are pulled from *pitch* where they fit.

**Source map (read Markdown, not PDFs):** see [`ROUTING.md`](ROUTING.md). Short labels used below:

| Label | File | Role |
| ----- | ---- | ---- |
| *tongue* | [`md/tongue-diagnosis-mobilenet.md`](md/tongue-diagnosis-mobilenet.md) | Abstract + section rhythm |
| *traffic* | [`md/traffic-flow-management.md`](md/traffic-flow-management.md) | Related-work table, Materials & Methods, IoT/deploy framing |
| *drone* | [`md/drone-aerial-accident-detection.md`](md/drone-aerial-accident-detection.md) | YOLO pipeline, trade-offs, limitations table |
| *pitch* | [`md/ppe-systeme-pitch-fr.md`](md/ppe-systeme-pitch-fr.md) | Our EPI stats, vigilance argument, 4-stage system |
| *refs* | [`references.txt`](references.txt) | Citation seeds |

Regenerate PDF→Markdown anytime: `./scripts/pdfs_to_markdown.sh`

**Target venue:** I2CEAI 2026 — 2nd International Conference on Intelligent Computing, Engineering and Artificial Intelligence for Imaging Systems (co-organized by the Association Marocaine de l'IA et la Transition Digitale, FPK/USMS Khouribga, and EMSI Marrakech). Accepted + presented papers appear in **Scopus-indexed proceedings**. CFP: <https://morocco4ai.org/index.php/call-for-papers/>

**Key dates:** regular-paper submission **July 23, 2026** (⚠️ ~1 week from 2026-07-15) · notification Aug 23 · camera-ready upload Sept 23 · author registration Oct 23. Numbers can still be refreshed in the camera-ready, but the submitted draft must stand on its own for review.

**Format:** IEEE conference paper — IEEEtran two-column, **max 8 pages** incl. references, numbered [n] citations, English only. Rough page budget: Intro ≈1, Related Work ≈1, Methods ≈2, Results ≈2, Discussion ≈1, Conclusion + refs ≈1.

---

## Title

**"Beyond Human Vigilance: A CCTV-Based Deep Learning System for Real-Time PPE Compliance Detection in Industrial Sites"**

*(Alternative: "From Passive Surveillance to Proactive Safety: An LLM-Verified YOLOv8 Framework for Industrial PPE Monitoring")*

**Keywords:** PPE compliance · YOLOv8 · Vigilance decrement · LLM-based alert verification · Industrial safety · Computer vision · CCTV retrofitting

---

## Abstract

Follow the *tongue* / *traffic* one-paragraph pattern: problem → gap → proposed system → dataset → model → key quantitative result → contribution statement.

Draft skeleton:

> Occupational accidents remain a critical global challenge, with millions of fatalities and non-fatal injuries recorded annually and an estimated economic cost approaching 4% of global GDP. Conventional CCTV-based monitoring depends on continuous human attention, which is known to degrade sharply within 20–35 minutes of sustained viewing. This paper presents an automated machine-vision system that repurposes existing industrial CCTV infrastructure to detect personal protective equipment (PPE) non-compliance and unauthorized entry into hazardous zones in real time. The system fine-tunes a YOLOv8 detector on a composite dataset combining the public PPE-Detection corpus (Roboflow Universe), the SH17 dataset, and a self-annotated, site-specific dataset from OCP Group operations — with gas-mask augmentation and higher sampling weight assigned to the OCP-specific data to reflect deployment conditions. A secondary reasoning stage — a Qwen vision-language model served through Ollama — re-evaluates flagged detections (accounting for occlusion, distance, glare, and lighting) before an alert is issued, reducing false-positive fatigue relative to raw detector output. Experimental results show [X]% precision, [X]% recall, and [X]% mAP@50 after [X] epochs of training at 640×640 resolution; qualitative case studies illustrate how the verification stage suppresses false alerts under occlusion, glare, and long-range conditions. The proposed pipeline requires no new camera hardware and provides continuous, fatigue-free coverage, offering a scalable blueprint for proactive industrial safety monitoring.

---

## 1. Introduction

Mirror the *traffic* / *drone* structure: global problem stats → gap in current practice → proposed contributions as a bulleted list → paper roadmap.

**Paragraph 1 — Scale of the problem** (cite directly from *pitch* sourced stats):

- 2.78 million deaths/year from work-related accidents and diseases worldwide
- 374 million non-fatal occupational injuries recorded annually worldwide
- $3 trillion global annual cost of occupational accidents and diseases (~3.94% of world GDP)
- $1B+ weekly cost of workplace accidents in the U.S.
- *Source: International Labour Organization, Safety and Health at Work (2023); National Safety Council estimates.*

**Paragraph 2 — Why current monitoring fails (the vigilance bottleneck):**

- Sustained attention collapses within 20–35 minutes of continuous monitoring (Parasuraman, 1984; Sawin & Scerbo, 1995)
- After 20 minutes of active viewing, operators miss up to 95% of on-screen activity (Velastin et al., 2006)
- Trained CCTV operators in real ore-processing plants detected only ~50% of targeted risk behaviors, with a high false-alarm rate (Donald et al., *Applied Ergonomics*, 2015)
- In 90-minute sessions, attention lapses affect 23% of operators in the first 30 minutes, rising to 60% in the next 30 (Sandia National Laboratories, 2014)
- Frame this as "one instant decides the year" — the 525,600-minutes-per-year argument from *pitch* (useful as a motivating calculation, optionally as Table 1 or a callout box, akin to *drone*'s illustrative figures).

**Paragraph 3 — Gap statement:** Existing works address single-purpose PPE detection but rarely combine (a) site-specific fine-tuning for domain shift (dust, lighting, uniforms), (b) coverage of chemical-processing classes such as gas masks, and (c) a false-positive mitigation layer.

**Paragraph 4 — Contributions (bulleted, same style as *drone*):**

- **Composite domain-adapted dataset**: fusion of public PPE-Detection (Roboflow), SH17 for augmented accuracy, and a self-annotated OCP dataset with weighted sampling favoring site-specific data, plus gas-mask class augmentation.
- **LLM-based judge stage**: a vision-language reasoning pass (Qwen-VL served via Ollama) over flagged detections to suppress false positives from occlusion, reflections, distance, and lighting artifacts.
- **Retrofit deployment model**: no new camera hardware; alerts pushed to existing infrastructure.
- **Empirical validation**: training curves, precision/recall, confusion analysis, and qualitative LLM-judge case studies.

**Paragraph 5 — Roadmap** (Section II reviews related work, Section III details the system and dataset, Section IV presents results, Section V discusses limitations, Section VI concludes).

---

## 2. Related Work

Follow the *tongue* / *traffic* convention: prose + a **comparison table**.

Subsections or a single table covering:

- Generic PPE detection benchmarks (helmet/vest datasets, Roboflow Universe PPE projects)
- SH17 dataset and its role in industrial human-object PPE labeling
- Domain adaptation / fine-tuning literature for site-specific CCTV conditions (dust, low light — relevant to phosphate/mining environments)
- LLM-as-judge / vision-language reasoning literature for reducing false positives in detection pipelines (this is your differentiator — cite recent VLM-verification or "detect-then-reason" papers)
- Vigilance/human-factors literature already used in *pitch* (Parasuraman 1984; Sawin & Scerbo 1995; Velastin et al. 2006; Donald et al. 2015; Sandia 2014) — positioned here as the human-factors justification for automation, similar to how *traffic* cited HSM/RIPCORD-iSEREST as its policy framework table.  
[note: also pull from *refs* / `references.txt`]

**Table 1** (mirror *traffic* Table 1 format):


| Study/Source           | Method                  | Focus                               | Key Finding                            |
| ---------------------- | ----------------------- | ----------------------------------- | -------------------------------------- |
| Roboflow PPE-Detection | YOLO-based              | Helmet/vest detection               | Baseline public benchmark              |
| SH17 Dataset           | Object detection        | 17-class industrial PPE/human parts | Broad generic coverage                 |
| Donald et al. (2015)   | Human observation study | CCTV operator detection rate        | ~50% detection, high false alarms      |
| Velastin et al. (2006) | Vigilance study         | Attention decay                     | Up to 95% missed activity after 20 min |
| [others]               | ...                     | ...                                 | ...                                    |


---

## 3. Materials and Methods

Structure this like *drone*'s Methodology (System Overview → Architectural detail → Dataset) merged with *traffic*'s implementation/logic subsections.

### 3.1 System Overview

Four-stage pipeline (matches *pitch* "COMMENT FONCTIONNE" diagram exactly):

1. **Base model** — pretrained object-detection backbone (YOLOv8, Ultralytics)
2. **OCP data fine-tuning** — domain adaptation on self-annotated site imagery
3. **LLM judge** — Qwen-VL (served via Ollama) reasoning pass over flagged detections
4. **Alert dispatch** — confirmed violations pushed to existing camera/network infrastructure

Include a pipeline figure (equivalent to *drone* Fig. 1 or *traffic* Fig. 1 block diagram).

### 3.2 Dataset Construction

- **Sources combined**: Roboflow Universe "PPE-Detection" (public baseline), SH17 (generic industrial human/PPE parts), OCP self-annotated dataset (site cameras, real dust/lighting conditions)
- **Class augmentation**: added/reinforced gas-mask class, since public datasets under-represent respiratory PPE
- **Weighting strategy**: OCP-specific samples given higher sampling weight during training to bias the model toward deployment-realistic conditions (state the exact ratio/weighting scheme you used, e.g. oversampling factor or loss weighting)
- **Annotation protocol**: describe tools used (Roboflow annotate — match *drone*'s explicit tool naming) and the actual class list: Person, helmet, vest, gloves, boots, goggles, no_helmet, no_gloves, no_goggle (+ gas mask in the rerun). The `none` class is a merge artifact — the `sh17-hmkpl` data.yaml used bare numeric class names (0–16), which the fusion collapsed into `none`; clean it out in the rerun and do not feature it in the paper.
- **Split**: training/validation/test percentages (mirror *drone*'s 70/15/15 stratified format)
- **Fusion QA (for the rerun)**: remap SH17's 17 class IDs into the fused class map before merging — the previous fusion silently dropped 2,645/6,182 train images (~43%) as "corrupt" because SH17 label IDs ≥ 10 exceeded the 10-class map. Use SH17's published 17-class list to name the numeric classes in `sh17-hmkpl`'s data.yaml first (its bare `0`–`16` names are also what produced the junk `none` class).

#### 3.2.1 OCP Site Dataset (self-annotated, Roboflow project)

The site-specific component described above is a self-annotated Roboflow project built from OCP CCTV imagery:

- **Source images**: 207 images, all 207 annotated (100% annotation coverage), 7 classes *(class names TBD — confirm exact label list against the Roboflow project before drafting; likely a subset of the fused class map, e.g. Person + a handful of PPE/no-PPE states)*
- **Source split**: 145 training / 41 validation / 21 test images (≈70% / 20% / 10%)
- **Generated (post-augmentation) split**: with 2 outputs per training example, the training set expands to 290 images — 82% train / 12% valid / 6% test (290 / 41 / 21)
- **Standalone OCP-only training run**: this generated version was also trained on its own (not fused) to establish baseline detection performance on OCP site imagery in isolation — report as an OCP-only baseline in §4.3, compared against the OCP-weighted fused model. Metrics are `[X]` placeholders until reported.
- **Note on scale**: 207 source images is small relative to the fused public sources (Roboflow PPE-Detection + SH17); frame this explicitly in §5.3 Limitations as the reason OCP samples are up-weighted in the fused run rather than relied on alone, and as motivation for future-work dataset expansion (§6).

### 3.3 Preprocessing

If applicable — image resizing to 640×640, augmentation (flip, brightness/contrast jitter to simulate dust/glare), normalization. Keep this short unless you did something comparable to *drone*'s hybrid preprocessing pipeline.

**OCP dataset preprocessing/augmentation (as configured in Roboflow):**

- **Auto-orient**: applied (strips EXIF rotation)
- **Resize**: stretch to 512×512 — ⚠️ note this differs from the 640×640 resolution used for the main fused-dataset training run (§3.4); decide before drafting whether the OCP subset is resized again to 640×640 at fusion time, or whether 512×512 is the resolution for a separate OCP-only ablation (relevant to §4.3)
- **Augmentation** (applied to expand the 207-image source set):
  - Crop: 8% minimum – 19% maximum zoom
  - Rotation: between −13° and +13°
  - Brightness: between −24% and +24%

### 3.4 Model Training

- Architecture: YOLOv8 (Ultralytics) — state the exact variant (previous run for reference: YOLOv8n, 3.0M params, 8.1 GFLOPs, 6.3 MB weights — the small footprint supports the retrofit narrative)
- Environment: Google Colab, Python
- Training regime: [X] epochs, 640×640 input resolution
- Output artifact: `best.pt`
- Loss components tracked: box, class, DFL (regenerate the training dashboard figure from the rerun — reuse it as Fig. X)
- Hyperparameters (fill in from the rerun log; previous run for reference: batch 16, lr0 0.01, auto optimizer, momentum 0.937, weight decay 5e-4, 3 warmup epochs, mosaic closed for last 10 epochs, HSV jitter + horizontal flip + random erasing)
- **All numbers remain `[X]` placeholders until the rerun we run ourselves.**

### 3.5 LLM Judge Stage

This is your differentiator vs. *tongue* / *traffic* / *drone* — give it real weight:

- **Model & serving**: Qwen vision-language model (Qwen2.5-VL family) served through Ollama — cloud-hosted for the pilot, self-hostable on-prem so frames never leave the site network (privacy + retrofit argument).
- **Input**: cropped detection + surrounding frame for context, plus metadata (flagged class, detector confidence, camera ID).
- **Reasoning objective**: assess occlusion, reflections, distance, lighting, and detection plausibility before confirming a violation.
- **Output**: structured JSON — `{"verdict": "confirm" | "reject" | "recheck", "confidence": 0–1, "reason": "<one sentence>"}`. Only `confirm` reaches a human; `recheck` re-queues the detection for evaluation on subsequent frames.
- **Draft judge prompt** (tune during drafting; include the final version in the paper or an appendix):

  > You are a safety-compliance verifier for industrial CCTV. A PPE detector flagged the attached image region as "<class>" with confidence <conf>; the full frame is provided for context. Decide whether this is a genuine violation. Reason about: occlusion by equipment or other workers, reflections and glare, distance and image resolution, lighting and dust conditions, and whether pose or clothing could mimic a violation. Reply in JSON: {"verdict": "confirm"|"reject"|"recheck", "confidence": 0-1, "reason": "<one sentence>"}.

- **Rationale citation**: reference Donald et al. (2015)'s finding on high false-alarm rates in manual CCTV monitoring as the motivation for this filtering stage (already used this framing in *pitch*)
- **Evaluation scope**: qualitative only in this paper (see §4.4) — state this explicitly and list quantitative judge benchmarking as future work.

### 3.6 Alert and Deployment Architecture

Real-time alert dispatch to safety personnel with timestamped clips, running on existing camera/network hardware — no new capital investment (mirrors *traffic*'s "signal to IoT controller" deployment framing, and *pitch*'s "AUCUNE LIMITE DE 20 MIN" slide).

---

## 4. Results and Analysis

Match *tongue*'s numbered subsections and *drone*'s table conventions.

### 4.1 Training Performance

- Regenerate the dashboard from the rerun: Box/Class/DFL loss curves, Precision & Recall vs. epoch, mAP@50, mAP@50-95 — cite exact final-epoch values. All values stay `[X]` until the rerun; decide whether headline tables report the val split or the held-out test split (a held-out test set is the stronger claim).

### 4.2 Detection Performance by Class

**Table 2** — Precision/Recall/mAP@50 broken down per PPE class (helmet, vest, gloves, boots, gas mask, no-PPE variants), following *drone*'s Table II format:


| Class               | Precision (%) | Recall (%) | mAP@50 (%) |
| ------------------- | ------------- | ---------- | ---------- |
| Person              |               |            |            |
| helmet              |               |            |            |
| no_helmet           |               |            |            |
| vest                |               |            |            |
| gloves              |               |            |            |
| no_gloves           |               |            |            |
| goggles             |               |            |            |
| no_goggle           |               |            |            |
| boots               |               |            |            |
| gas mask *(rerun)*  |               |            |            |

Notes for drafting: flag low-support classes explicitly (in the previous run boots and no_goggle had <35 val instances each), and address the helmet vs. no_helmet support imbalance in §4.5 — reviewers will notice if it isn't owned openly.


### 4.3 Effect of Dataset Weighting (OCP vs. generic data)

Ablation-style comparison, structured like *drone*'s "generic vs. adapted model" section (already drafted in *pitch* as "VOIR CE QUE LES MODÈLES GÉNÉRIQUES RATENT"): show side-by-side detection quality of the generic-only model vs. the OCP-weighted model on the same site images (reuse your annotated example image with red/blue boxes as Fig. X).

Three-way comparison to report here: (a) generic-only model, (b) OCP-only model — trained solely on the 207-image OCP set (§3.2.1) to measure standalone baseline performance on site imagery, (c) OCP-weighted fused model. (b) isolates how much the site-specific data alone can do before fusion/weighting; metrics stay `[X]` placeholders until reported.

### 4.4 LLM Judge Qualitative Case Studies

Qualitative evaluation only — no false-positive-reduction percentage is claimed anywhere in the paper, including the abstract. Present 3–5 real cases as a figure or table: detection crop → judge reasoning (verbatim `reason` field) → verdict. Include at minimum (a) one false positive correctly rejected (e.g., glare or occlusion artifact), (b) one true violation confirmed, and ideally (c) one borderline `recheck` case. State the trade-off honestly — the judge adds latency and can over-suppress genuine violations (same honesty *drone* shows with RT-DETR-L vs YOLOv8) — and note quantitative benchmarking as future work.

### 4.5 Confusion Matrix / Error Analysis

If you have per-class confusion data (like *tongue*'s Table 3), include it — most useful for distinguishing e.g. gas-mask vs. no-gas-mask, or helmet vs. hood confusions in OCP imagery.

### 4.6 Runtime and Deployment Feasibility

Inference time per frame, feasibility of running across multiple camera feeds concurrently (mirror *traffic*'s concurrent-feeds framing) — even an estimate here strengthens the "retrofit, no new hardware" claim.

---

## 5. Discussion

Match *drone*'s Discussion structure (subsections on trade-offs and limitations); *ct-semantic-segmentation* also has an explicit Limitations section if you want a shorter template.

### 5.1 Why Domain Adaptation Matters

Discuss the generic-vs-OCP-weighted comparison result in prose; connect back to the "dust, angles, lighting" argument from *pitch*.

### 5.2 Trade-offs of the LLM Judge Stage

Latency added vs. false-positive reduction gained; when the judge might over-suppress genuine violations. Discuss the serving choice: Ollama-hosted Qwen can run on-prem (frames never leave the site network — a privacy win) or in the cloud (no on-site GPU needed, but adds egress and latency); state which deployment the pilot used.

### 5.3 Limitations

- Small object/distance limitations (echo *drone*'s small-object caveat)
- Weather/dust degradation (echo *drone*'s weather caveat, relevant to phosphate operations)
- Gas-mask class scarcity even after augmentation
- Judge stage evaluated qualitatively only — no quantitative false-positive-reduction figure; defer benchmarking to future work
- Reliance on existing camera placement/angle coverage — blind spots not addressed by better detection

**Table 3** (optional, mirroring *drone*'s "Summary of Impact and Limitations") summarizing parameters, gains, and limitations in one place.

---

## 6. Conclusion and Future Work

- Restate contribution: retrofit-friendly, domain-adapted, LLM-verified PPE 
- Quantify headline result (final mAP, false-positive reduction)
- Future work: expanding OCP dataset across more sites/seasons, temporal/tracking models (person re-identification across frames, similar to *tongue*'s future-work mentions), integration with access-control/IoT systems (similar to *traffic*'s IoT signal integration), edge deployment for latency reduction, multi-camera fusion for full-body PPE checks.

**Acknowledgements** (if applicable, mirror *traffic*'s OCP/EMSI acknowledgement format).

---

## References

Combine (see also *refs* / `references.txt` and the reference lists at the end of each file in `md/`):

- ILO, *Safety and Health at Work* (2023); National Safety Council estimates — for intro statistics (*pitch*)
- Parasuraman, R. (1984); Sawin, D.A. & Scerbo, M.W. (1995) — vigilance decrement
- Velastin, S.A. et al. (2006) — CCTV attention/missed-activity study
- Donald, I. et al., *Applied Ergonomics* (2015) — CCTV operator detection rate in ore-processing plants
- Sandia National Laboratories (2014) — attention loss over 90-minute sessions
- Ultralytics YOLOv8 citation (Jocher et al., 2023) — same citation style as *drone*
- SH17 dataset citation
- Roboflow PPE-Detection dataset citation (Roboflow Universe, same style as *traffic*'s dataset citations)
- Qwen / Qwen2.5-VL technical report — judge model citation; Ollama as serving tool (footnote or software citation per IEEE style)
- Any LLM/VLM-as-judge or reasoning-verification papers you draw on for Section 3.5/5.2
- ⚠️ Verify the Velastin et al. (2006) attribution for the "95% missed after 20 min" claim — the `references.txt` entry is an abandoned-object-detection paper, likely not the vigilance source; find the correct citation or soften the claim
- Align the ILO fatality figure with the cited year (2.78 M is the older figure; the ILO 2023 release says "nearly 3 million")
- Still missing from `references.txt`: Parasuraman (1984), Donald et al. (2015), Sandia (2014), Roboflow PPE-Detection dataset

---

## Open items before drafting

- **⚠️ Timeline**: regular-paper submission is **July 23, 2026** (~1 week away). Decide what lands in the submitted version (the fixed-fusion rerun at minimum) vs. what gets refreshed in the camera-ready by Sept 23 (e.g., the OCP fine-tuning run and the §4.3 side-by-side). The prose can be drafted now — placeholders only block the tables/figures.
- **All training metrics stay `[X]` placeholders** until we rerun training ourselves. The rerun must include the gas-mask class and fix the SH17 label remap — the previous fusion silently dropped 2,645/6,182 train images (~43%) whose SH17 class IDs were ≥ 10 (see Fusion QA in §3.2).
- §4.3 needs the generic model evaluated on the *same* OCP images for the side-by-side figure.
- Collect 3–5 judge case examples for §4.4 — use **placeholder images until OCP imagery clearance** comes through (face blurring for any published frames).
- Decide val split vs. held-out test split for headline tables.
- Confirm whether the alert-dispatch stage (§3.6) is implemented or presented as proposed architecture — say which, explicitly.

