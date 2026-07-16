# Chapter Plan — I2CEAI 2026 PPE Compliance Paper

Produced by academic-paper `plan` mode (Socratic session, 2026-07-15). Companion to
[`structure.md`](structure.md) — that file holds the section skeleton; this file holds the
argument layer, evidence map, and drafting schedule. Where they conflict, this file wins on
*argument*, structure.md wins on *section layout*.

---

## INSIGHT Collection

- **[INSIGHT: thesis_statement]** Existing CCTV infrastructure + a domain-adapted YOLOv8 + an
  LLM judge is a practical, low-cost path to continuous, fatigue-free PPE compliance
  monitoring — the integrated retrofit-and-verify pipeline is the argument, not any single
  component.
- **[INSIGHT: gap_statement]** PPE datasets and detection papers are construction-centric
  (helmet/vest); chemical-processing PPE (gas masks, goggles) and phosphate-site conditions
  (dust, lighting, site uniforms) are unstudied.
- **[INSIGHT: contribution_claim]** (user's selection, verbatim from structure.md): "Composite
  domain-adapted dataset: fusion of public PPE-Detection (Roboflow), SH17 for augmented
  accuracy, and a self-annotated OCP dataset with weighted sampling favoring site-specific
  data, plus gas-mask class augmentation." — This is the bullet the paper stands on; the other
  three bullets support it.
- **[INSIGHT: key_finding_expectation]** The fused-dataset YOLOv8n reaches usable
  precision/recall at millisecond-level inference — deployable accuracy on existing hardware
  is the headline of Section 4.
- **[INSIGHT: reader_takeaway]** Detection ≠ safety system: a detector alone moves the
  vigilance problem downstream (alert fatigue); verification is what turns detections into a
  usable safety system.
- **[INSIGHT: methods_defense]** Why an LLM judge rather than confidence thresholding or
  temporal smoothing: (1) thresholds trade recall for precision blindly, while the judge
  reasons about *why* a detection looks wrong (occlusion, glare, distance) and can keep
  low-confidence true violations; (2) the judge's stated reason accompanies each alert —
  explainability for safety staff, operator trust, auditability. Both go in §3.5 and §5.2.
- **[INSIGHT: weakest_point]** The judge is the differentiator but is evaluated qualitatively
  only — a reviewer can say the paper's novelty is unmeasured. The plan schedules the
  strongest defense here (see Defense Schedule).
- **[OPEN: results_posture]** How hard to lean into weak per-class results is deferred until
  the rerun's numbers exist. Default fallback if numbers stay weak: own them openly in §4.5 +
  Limitations (support imbalance, multi-source difficulty, nano-model trade-off).

### Argument chain (spine of the paper)

Vigilance decrement makes manual CCTV monitoring structurally broken (§1) → existing PPE
detection work can't fill the gap because it is construction-centric and ignores
chemical-industry classes and conditions (§2) → we build a composite domain-adapted dataset
(the load-bearing contribution) and fine-tune YOLOv8 on it (§3.2–3.4) → results show
deployable accuracy at real-time speed on commodity hardware (§4.1–4.3, 4.6) → but detection
alone would flood staff with false alerts, recreating the vigilance problem downstream (§1
stats + Donald 2015) → the LLM judge verifies before alerting (§3.5, §4.4 qualitative) → the
whole pipeline retrofits onto existing cameras (§3.6) → takeaway: detection ≠ safety system;
verification closes the loop (§5).

Note the deliberate division of labor: the **dataset** carries the scientific contribution;
the **judge** carries the memorable message. Do not let the writing invert this — the judge is
argued from design rationale + literature + case studies, never from quantitative claims.

---

## Defense Schedule (argument stress test outcomes)

**Primary: "judge is unquantified" (user-selected weakest point)** — five-layer defense:
1. **Scope the claim explicitly** in §3.5 and §4.4: the judge is presented as an architectural
   contribution with qualitative evidence; say "we do not claim a measured false-positive
   reduction" in so many words.
2. **Motivate from literature, not from our data**: Donald et al. (2015) high false-alarm
   finding + alert-fatigue framing make the judge *necessary by design*, independent of our
   measurements.
3. **Make the case studies do real work**: 3–5 cases with verbatim judge reasoning, including
   ≥1 correctly rejected false positive and ≥1 confirmed violation — concrete enough that a
   reviewer sees the mechanism functioning.
4. **Limitations entry** (§5.3): quantitative judge benchmarking named as the top item of
   future work, with a sketch of the protocol (labeled alert set, confirm/reject
   confusion matrix) to show we know how it would be measured.
5. **Never let the abstract/intro promise a number** the paper doesn't have (already enforced
   in structure.md).

**Secondary weak points** (from the same stress test, lower priority but wordable now):
- *Gap vs. evidence mismatch*: the gap claims a chemical-industry blind spot; at submission
  the evidence may be gas-mask-class-only (rerun) with OCP data still pending. Mitigation:
  word the gap as "under-represented classes AND conditions", let the gas-mask class +
  augmentation carry the class half now, and scope the OCP conditions half as the deployment
  stage (camera-ready refresh if the OCP run lands by Sept 23).
- *OCP promise, public-data results*: keep every OCP-specific results claim in §4.3
  conditional; the pipeline (§3) may describe the OCP stage as designed and in progress —
  never as evaluated — until its numbers exist.
- *Modest detector numbers*: defense pre-drafted in §4.5/§5.3 (multi-source fusion difficulty,
  nano model chosen for retrofit feasibility, support imbalance) — activated or relaxed once
  the rerun decides [OPEN: results_posture].

---

## Chapter-by-chapter plan

Page budget: 8 pages IEEE two-column ≈ 5,500–6,000 words total; floats (pipeline figure,
dashboard, 2 example figures, 3 tables) ≈ 1.5–2 pages, leaving ≈ 4,300–4,800 words of text.

| # | Section | Words | Opening move | Core job | Assets needed | Blockers |
|---|---------|-------|--------------|----------|---------------|----------|
| — | Abstract | ~200 | Problem stat (ILO) | Compress the argument chain; no unpromised numbers | — | rerun (final numbers) |
| 1 | Introduction | ~750 | 2.78M deaths/yr + vigilance collapse | Urgency: "one instant decides the year"; gap para leads with chemical-industry blind spot; contributions bulleted, **dataset first** | pitch stats | none — draft now |
| 2 | Related Work | ~700 + Table 1 | Construction-centric PPE detection survey | Show every prior thread stops short: generic benchmarks (no chemical classes), vigilance literature (motivates automation), VLM-verification (nascent) → the combined gap | references.txt + verified citations | citation verification only |
| 3 | Materials & Methods | ~1,300 + Fig 1 | Four-stage pipeline figure | §3.2 dataset = contribution section, give it the most space (fusion, remap fix, gas-mask augmentation, weighting design); §3.5 judge with prompt + defense layer 1–2 | pipeline figure; judge prompt (drafted) | none — draft now with `[X]` |
| 4 | Results | ~1,000 + dashboard, Tables 2–3, case-study figure | Headline: accuracy + 2.3ms-class inference | Lead §4.1/4.2 with deployable-accuracy finding; §4.3 conditional OCP; §4.4 case studies (defense layer 3); §4.6 real-time feasibility | rerun log, regenerated dashboard, judge examples (placeholder imagery until clearance) | **rerun**; judge examples |
| 5 | Discussion | ~600 | The takeaway, stated plainly | "Detection ≠ safety system" is the organizing idea; §5.2 judge trade-offs (defense layers 1+4); §5.3 limitations incl. judge-unquantified | — | rerun (for §5.1 framing) |
| 6 | Conclusion + Future Work | ~250 | Restate contribution + headline number | Future work: judge benchmarking protocol first, then OCP expansion, tracking, edge deployment | — | rerun (headline number) |
| — | References | ~1 page | — | 20–25 refs, IEEE numbered; every citation verified (no fabrication — see structure.md ⚠️ items) | references.txt + additions | verification pass |

**Back-matter checklist** (venue-dependent, confirm against I2CEAI author kit): AI-usage
disclosure statement, data availability statement, ethics/consent note for site imagery,
acknowledgements (OCP + EMSI), funding line if any.

---

## Drafting schedule (submission 2026-07-23)

| Day | Date | Writing track | Parallel user track |
|-----|------|---------------|---------------------|
| 1 | Jul 15–16 | §1 Introduction + §2 Related Work (zero number-dependencies) + citation verification | Start rerun prep: fix SH17 remap, add gas-mask class |
| 2 | Jul 17 | §3 Methods complete with `[X]` placeholders; pipeline figure | Launch training rerun |
| 3 | Jul 18 | §5 Discussion + §6 Conclusion skeletons; judge case-study format | Rerun finishes (~1 h compute); export log + dashboard |
| 4 | Jul 19 | §4 Results filled from rerun; regenerate dashboard; decide [OPEN: results_posture] | Collect judge examples (Qwen via Ollama) |
| 5 | Jul 20 | §4.4 case studies; abstract; full-draft integration pass | Image clearance status check |
| 6 | Jul 21 | Internal review (`/ars-reviewer`) + revision | — |
| 7 | Jul 22 | IEEE formatting (IEEEtran LaTeX), citation check (`/ars-citation-check`), back-matter | — |
| 8 | Jul 23 | Buffer + submit | — |

OCP fine-tuning run and §4.3 side-by-side: targeted at the Sept 23 camera-ready, not the
submission (per scope decisions of 2026-07-15).

---

## Repositioning addendum (plan-mode session, 2026-07-16)

User redirected the paper's center of gravity; the following INSIGHTs **supersede** their
2026-07-15 counterparts above. Applied to the draft the same day.

- **[INSIGHT: thesis_statement — updated]** Generic benchmark training does not transfer to a
  real, operating chemical plant; a model fused and adapted for industrial/chemical conditions
  (CCTV angles, gas masks, site uniforms) does. The generic-vs-adapted comparison is the
  paper's central experiment; the retrofit+verify pipeline is what makes the result deployable.
- **[INSIGHT: contribution_claim — updated, user's words]** "The core novelty of this work is
  applying CV to a real, operating chemical plant, which almost no published PPE-detection
  work actually does… generic benchmark model vs. our industrial-adapted model is the paper's
  central experiment, and it's what justifies calling this a domain-specific contribution
  rather than another YOLO variant."
- **[INSIGHT: reader_takeaway — updated]** Adaptation core, judge secondary (user decision):
  primary takeaway is the transfer failure of generic benchmarks; "detection ≠ safety system"
  survives as the systems-level second lesson in §5.2.
- **Central experiment design (user decisions):** eval on BOTH held-out OCP site imagery
  (headline) and an industrial proxy set (held-out SH17 + respirator images, publishable
  without clearance). Generic baseline already exists: the Roboflow-hosted PPE_Detection model
  — 308 images, 9 construction classes, P 58.1 / R 64.4 / mAP50 59.1 on its own val split.
  Comparison restricted to the shared class subset, identical resolution/thresholds.
- **Title (user decision):** "Beyond Human Vigilance: Domain-Adapted PPE Compliance Detection
  on CCTV at an Operating Chemical Plant".
- **New defense items:** (1) *data-volume confound* — baseline trained on far less data;
  owned in Limitations, size-matched control = future work item #1; (2) *in-distribution
  proxy critique* — answered by making OCP the headline set; (3) comparison scaffold (Table
  II + Fig. 3) ships with provisional bar heights, every value red-[X] until the evaluation
  rerun fills the cross-domain cells.
- **Page-budget alert:** the added comparison subsection + table + chart + confusion-matrix
  and case-crop placeholders push the draft to 9 pages vs. the 8-page I2CEAI limit — a
  trimming pass is required before submission.
