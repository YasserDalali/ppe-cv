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

---

## Title

**"Beyond Human Vigilance: A CCTV-Based Deep Learning System for Real-Time PPE Compliance and Hazard-Zone Intrusion Detection in Industrial Sites"**

*(Alternative: "From Passive Surveillance to Proactive Safety: An LLM-Verified YOLOv8 Framework for PPE and Restricted-Zone Monitoring")*

**Keywords:** PPE compliance · Hazard zone detection · YOLOv8 · Vigilance decrement · LLM-based alert verification · Industrial safety · Computer vision · CCTV retrofitting

---

## Abstract

Follow the *tongue* / *traffic* one-paragraph pattern: problem → gap → proposed system → dataset → model → key quantitative result → contribution statement.

Draft skeleton:

> Occupational accidents remain a critical global challenge, with millions of fatalities and non-fatal injuries recorded annually and an estimated economic cost approaching 4% of global GDP. Conventional CCTV-based monitoring depends on continuous human attention, which is known to degrade sharply within 20–35 minutes of sustained viewing. This paper presents an automated machine-vision system that repurposes existing industrial CCTV infrastructure to detect personal protective equipment (PPE) non-compliance and unauthorized entry into hazardous zones in real time. The system fine-tunes a YOLOv8 detector on a composite dataset combining the public PPE-Detection corpus (Roboflow Universe), the SH17 dataset, and a self-annotated, site-specific dataset from OCP Group operations — with gas-mask augmentation and higher sampling weight assigned to the OCP-specific data to reflect deployment conditions. A secondary LLM-based reasoning stage re-evaluates flagged detections (accounting for occlusion, distance, glare, and lighting) before an alert is issued, reducing false-positive fatigue relative to raw detector output. Experimental results show [X]% precision, [X]% recall, and [X]% mAP@50 after 50 epochs of training at 640×640 resolution, with the LLM verification stage reducing false alerts by [X]%. The proposed pipeline requires no new camera hardware and provides continuous, fatigue-free coverage, offering a scalable blueprint for proactive industrial safety monitoring.

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

**Paragraph 3 — Gap statement:** Existing works (PPE detection, address single-purpose detection but rarely combine (a) site-specific fine-tuning for domain shift (dust, lighting, uniforms), (b) PPE datasets ignore chemical processing classes such as gas masks , and (c) a false-positive mitigation layer.

**Paragraph 4 — Contributions (bulleted, same style as *drone*):**

- **Composite domain-adapted dataset**: fusion of public PPE-Detection (Roboflow), SH17 for augmented accuracy, and a self-annotated OCP dataset with weighted sampling favoring site-specific data, plus gas-mask class augmentation.
- **Dual-task detection**: unified pipeline for PPE compliance and hazard-zone intrusion detection.
- **LLM-based judge stage**: a reasoning pass over flagged detections to suppress false positives from occlusion, reflections, distance, and lighting artifacts.
- **Retrofit deployment model**: no new camera hardware; alerts pushed to existing infrastructure.
- **Empirical validation**: training curves, precision/recall, and confusion analysis over 50 epochs.

**Paragraph 5 — Roadmap** (Section II reviews related work, Section III details the system and dataset, Section IV presents results, Section V discusses limitations, Section VI concludes).

---

## 2. Related Work

Follow the *tongue* / *traffic* convention: prose + a **comparison table**.

Subsections or a single table covering:

- Generic PPE detection benchmarks (helmet/vest datasets, Roboflow Universe PPE projects)
- SH17 dataset and its role in industrial human-object PPE labeling
- Domain adaptation / fine-tuning literature for site-specific CCTV conditions (dust, low light — relevant to phosphate/mining environments)
- Restricted-zone / intrusion detection via object detection + geofencing or polygon-zone logic
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
3. **LLM judge** — reasoning pass over flagged detections
4. **Alert dispatch** — confirmed violations pushed to existing camera/network infrastructure

Include a pipeline figure (equivalent to *drone* Fig. 1 or *traffic* Fig. 1 block diagram).

### 3.2 Dataset Construction

- **Sources combined**: Roboflow Universe "PPE-Detection" (public baseline), SH17 (generic industrial human/PPE parts), OCP self-annotated dataset (site cameras, real dust/lighting conditions)
- **Class augmentation**: added/reinforced gas-mask class, since public datasets under-represent respiratory PPE
- **Weighting strategy**: OCP-specific samples given higher sampling weight during training to bias the model toward deployment-realistic conditions (state the exact ratio/weighting scheme you used, e.g. oversampling factor or loss weighting)
- **Annotation protocol**: describe tools used (Roboflow annotate — match *drone*'s explicit tool naming) and class list (helmet, vest, gloves, boots, gas mask, no-helmet, no-vest, etc.)
- **Split**: training/validation/test percentages (mirror *drone*'s 70/15/15 stratified format)
- **Hazard-zone data**: describe how restricted-zone imagery/polygons were annotated (bounding polygons per camera, person-in-zone logic) if this is trained/rule-based rather than purely learned

### 3.3 Preprocessing

If applicable — image resizing to 640×640, augmentation (flip, brightness/contrast jitter to simulate dust/glare), normalization. Keep this short unless you did something comparable to *drone*'s hybrid preprocessing pipeline.

### 3.4 Model Training

- Architecture: YOLOv8 (Ultralytics)
- Environment: Google Colab, Python
- Training regime: 50 epochs, 640×640 input resolution
- Output artifact: `best.pt`
- Loss components tracked: box, class, DFL (as in your existing training dashboard figure — reuse it as Fig. X)
- Hyperparameters (batch size, optimizer, learning rate schedule — fill in from your actual run)

### 3.5 Hazard-Zone Intrusion Logic

Describe the geometric/rule layer: how a detected person's bounding box/position is checked against a defined restricted polygon per camera, dwell-time thresholds if any, and how this combines with PPE detection (e.g., "no PPE + inside hazard zone" = escalated alert severity).

### 3.6 LLM Judge Stage

This is your differentiator vs. *tongue* / *traffic* / *drone* — give it real weight:

- Input: cropped detection + context (bounding box confidence, surrounding frame)
- Reasoning objective: assess occlusion, reflections, distance, lighting, and detection plausibility before confirming a violation
- Output: confirm / reject / request re-check
- Rationale citation: reference Donald et al. (2015)'s finding on high false-alarm rates in manual CCTV monitoring as the motivation for this filtering stage (already used this framing in *pitch*)

### 3.7 Alert and Deployment Architecture

Real-time alert dispatch to safety personnel with timestamped clips, running on existing camera/network hardware — no new capital investment (mirrors *traffic*'s "signal to IoT controller" deployment framing, and *pitch*'s "AUCUNE LIMITE DE 20 MIN" slide).

---

## 4. Results and Analysis

Match *tongue*'s numbered subsections and *drone*'s table conventions.

### 4.1 Training Performance

- Reuse/expand your existing dashboard: Box/Class/DFL loss curves, Precision & Recall vs. epoch, mAP@50, mAP@50-95 (you already have this figure — cite exact final-epoch values)

### 4.2 Detection Performance by Class

**Table 2** — Precision/Recall/mAP@50 broken down per PPE class (helmet, vest, gloves, boots, gas mask, no-PPE variants) and for hazard-zone intrusion detection, following *drone*'s Table II format:


| Class                 | Precision (%) | Recall (%) | mAP@50 (%) |
| --------------------- | ------------- | ---------- | ---------- |
| Helmet                |               |            |            |
| Vest                  |               |            |            |
| Gas mask              |               |            |            |
| No-PPE (violation)    |               |            |            |
| Hazard-zone intrusion |               |            |            |


### 4.3 Effect of Dataset Weighting (OCP vs. generic data)

Ablation-style comparison, structured like *drone*'s "generic vs. adapted model" section (already drafted in *pitch* as "VOIR CE QUE LES MODÈLES GÉNÉRIQUES RATENT"): show side-by-side detection quality of the generic-only model vs. the OCP-weighted model on the same site images (reuse your annotated example image with red/blue boxes as Fig. X).

### 4.4 LLM Judge Impact on False Positives

Quantify: raw detector alert volume vs. post-judge confirmed-alert volume; false-positive reduction rate; any missed true positives introduced by the judge (state the trade-off honestly, as *drone* does with RT-DETR-L vs YOLOv8).

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

Latency added vs. false-positive reduction gained; when the judge might over-suppress genuine violations.

### 5.3 Limitations

- Small object/distance limitations (echo *drone*'s small-object caveat)
- Weather/dust degradation (echo *drone*'s weather caveat, relevant to phosphate operations)
- Gas-mask class scarcity even after augmentation
- Reliance on existing camera placement/angle coverage — blind spots not addressed by better detection

**Table 3** (optional, mirroring *drone*'s "Summary of Impact and Limitations") summarizing parameters, gains, and limitations in one place.

---

## 6. Conclusion and Future Work

- Restate contribution: retrofit-friendly, domain-adapted, LLM-verified PPE + hazard-zone monitoring system
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
- Any LLM/VLM-as-judge or reasoning-verification papers you draw on for Section 3.6/5.2

---

A few things:

- Place holder metrics only

