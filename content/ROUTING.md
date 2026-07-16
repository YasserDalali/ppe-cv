# Content routing — what to read, and why

Prefer the Markdown conversions under [`md/`](md/) over the source PDFs. Regenerate with:

```bash
./scripts/pdfs_to_markdown.sh
```

This corpus is **structure inspiration** for the PPE / hazard-zone paper outlined in [`structure.md`](structure.md). Content domain differs; layout conventions (numbered sections, related-work tables, pipeline stages, per-class metrics, limitations table) are what we borrow.

---

## Primary structure templates (read these first)

| Role in our paper | Markdown (prefer) | Source PDF | What it is | Structural cues to copy |
| ----------------- | ----------------- | ---------- | ---------- | ----------------------- |
| **Tongue paper** | [`md/tongue-diagnosis-mobilenet.md`](md/tongue-diagnosis-mobilenet.md) | `661251.pdf` | MobileNetV3 + TCM tongue → renal disorder detection (EMSI/UCA) | Abstract one-paragraph pattern; Intro → Related Work → Methodology (preprocess→features→models) → Results (metrics table) → Conclusion |
| **Traffic paper** | [`md/traffic-flow-management.md`](md/traffic-flow-management.md) | `661256.pdf` | AI traffic flow: blockage, emergency priority, potholes, IoT signals (EMSI/UCA) | Related-work **comparison table**; Materials & Methods with multi-module CV + IoT deployment framing; quantitative mAP/precision/recall; scalable “blueprint” conclusion |
| **Drone paper** | [`md/drone-aerial-accident-detection.md`](md/drone-aerial-accident-detection.md) | `Drone (16).pdf` | UAV + hybrid preprocessing + YOLOv8 vs RT-DETR for aerial accident detection | Intro problem stats → contributions bullets → methodology (preprocess→detect) → generic vs adapted trade-offs → impact/limitations summary table → future work |
| **PPE pitch (our project)** | [`md/ppe-systeme-pitch-fr.md`](md/ppe-systeme-pitch-fr.md) | `PPE_Systeme_FR.pdf` | French OCP tech note / pitch: EPI compliance via CCTV vision | **Source of stats, vigilance argument, 4-stage pipeline, domain-adaptation messaging** — content for our paper, not just layout |

---

## Same-lab peers (secondary; shared EMSI formatting)

Useful when you need another example of the same venue/lab skeleton (intro stats → related work → methods → results → conclusion). Lower priority for PPE-specific structure.

| Markdown | Source PDF | Topic | Optional borrow |
| -------- | ---------- | ----- | --------------- |
| [`md/eco-track-trash-classification.md`](md/eco-track-trash-classification.md) | `661252.pdf` | ECO-TRACK: waste CNN + face ID + route optimization | IoT/edge deployment narrative; Materials & Methods → Experiments |
| [`md/tuber-ensemble-xray.md`](md/tuber-ensemble-xray.md) | `661260.pdf` | TUBER-ENSEMBLE: ResNet/DenseNet/Inception soft-voting on TB X-rays | Ensemble / comparative model table patterns |
| [`md/ia-med-disease-prediction.md`](md/ia-med-disease-prediction.md) | `661266.pdf` | IA-Med: ML disease prediction + chatbot | Integrated “system + UX” framing; Results & Discussion |
| [`md/ct-semantic-segmentation.md`](md/ct-semantic-segmentation.md) | `662647.pdf` | Multi-organ CT segmentation (UNet++ / DeepLabv3 / custom) | Explicit **Limitations** subsection; per-class metric tables (Dice/IoU) |

---

## Supporting files (not papers)

| File | Role |
| ---- | ---- |
| [`structure.md`](structure.md) | Full outline of *our* paper; cites the templates above by role |
| [`references.txt`](references.txt) | Short bibliography seeds (ILO, vigilance studies, SH17, YOLOv8, VLM review, …) |

---

## Suggested reading order for drafting

1. `ppe-systeme-pitch-fr.md` — claims, numbers, pipeline story  
2. `traffic-flow-management.md` — related-work table + multi-module methods  
3. `drone-aerial-accident-detection.md` — YOLO results / trade-offs / limitations table  
4. `tongue-diagnosis-mobilenet.md` — abstract + results subsection rhythm  
5. Peek at CT / TB / IA-Med only if you need another table or limitations phrasing  
6. Keep `references.txt` open while filling Related Work / intro citations

