# Webcam head-as-vest false positive — Diagnostics — Design

Date: 2026-07-24
Scope: `yolo_streamlit_app/`

## Problem

On webcam input, the model persistently detects the user's head as a `vest`. This needs
to be debugged efficiently: is it a code/config bug (label-index mismatch, ONNX export
defect), a confidence-tuning issue, or a genuine model/data limitation (the training set
has zero close-up-face images — SH17's `head`/`face` boxes are dropped as background
during remap, per `training/src/ppe/remap.py`, so the model has never been taught "head
≠ PPE" at webcam-selfie framing/distance)?

The goal of this design is **diagnosis only** — producing clear evidence to confirm or
rule out each hypothesis. No fix (retraining, threshold changes, suppression logic) is
in scope here; that's a follow-up once the root cause is known.

## Non-goals

- No retraining or dataset changes.
- No changes to `utils/zones.py` mismatch logic itself (used as-is, read-only).
- No auto-suppression of false positives anywhere in the app.
- No changes to the Live Detection tab's detection/rendering behavior.
- Not a general-purpose benchmarking tool (the existing Benchmark tab already covers
  resolution/FPS tradeoffs) — this is specifically about isolating the head/vest
  confusion.

## Hypotheses under test (cheapest to rule out first)

1. **Label-index bug**: the model's class-name-to-index mapping is misaligned somewhere
   between training (`data.yaml` / `CANONICAL_CLASSES`) and what's loaded at inference
   time (`.pt` or the ONNX export).
2. **ONNX export defect**: `scripts/export_onnx.py` (`dynamic=True, simplify=True`)
   introduced a discrepancy between `best.pt` and `best.onnx` output.
3. **Confidence borderline noise**: the false vest box only appears at low confidence
   and isn't a real high-confidence confusion.
4. **Domain shift (leading hypothesis)**: the model has never seen close-up, face-filling
   frames during training (only distant industrial-scene photos), so it has no learned
   distinction between "head" and "vest" at that framing/distance.

## Architecture

New read-only module `yolo_streamlit_app/utils/diagnostics.py`, plus a new **"Diagnostics"**
tab added to the existing `st.tabs(...)` call in `app.py`. Reuses `utils.inference.Detection`
and `utils.zones.classify_zones` rather than duplicating any detection/zone logic.

```python
EXPECTED_CLASSES = [  # mirrors paper/results/data.yaml and training/src/ppe/config.py::CANONICAL_CLASSES
    "Person", "helmet", "vest", "gloves", "boots", "goggles",
    "no_helmet", "no_gloves", "no_goggle", "gas mask",
]

def check_class_names(model_names: dict[int, str]) -> dict:
    """Compares model_names (in index order) against EXPECTED_CLASSES.
    Returns {"ok": bool, "expected": [...], "actual": [...], "mismatches": [(idx, expected, actual), ...]}."""

def load_reference_pt_model():
    """Lazy-loads paper/results/best.pt via ultralytics YOLO for comparison purposes.
    Path resolved as Path(__file__).resolve().parents[2] / "paper" / "results" / "best.pt"
    (repo-root-relative, independent of cwd). Cached in st.session_state.diag_pt_model.
    Returns None with a warning message if the file doesn't exist, instead of raising."""

def compare_models(pt_model, active_model, frame, conf, iou, use_tracker) -> dict:
    """Runs both models on the same frame. Returns per-model detection lists plus a
    simple disagreement report: detections (by class + rough bbox overlap) present in
    one model's output but not the other's, called out separately for `vest`/`helmet`
    classes near the top of frame."""

def confidence_sweep(model, frame, thresholds, iou, use_tracker) -> list[dict]:
    """Runs inference at each threshold in `thresholds` (default
    [0.1, 0.25, 0.5, 0.75, 0.9]). Returns one row per threshold: which classes
    survived, each with its confidence and bbox."""
```

All functions take plain data (frame arrays, model wrappers, thresholds) and return
plain dicts/lists — no Streamlit calls inside `diagnostics.py`. `app.py` owns all
rendering, consistent with how `utils/benchmark.py` and `utils/tracking.py` are already
structured.

## Plumbing: capturing a frame to diagnose

The live-detection `while` loop in `app.py` already processes a `resized` frame each
iteration. Add one line there to stash it:

```python
st.session_state.last_frame = resized.copy()
```

Buttons in other tabs aren't interactive while that loop is running (Streamlit executes
the script top-to-bottom per rerun, and the `while` loop blocks until "Start / Stop
stream" is unchecked) — so the natural workflow is: stream → stop → switch to
Diagnostics tab → the last frame before stopping is what gets diagnosed. No new capture
mechanism is needed beyond this one assignment.

## UI — "Diagnostics" tab

Added to the existing `st.tabs(...)` call in `app.py` (exact position among tabs is
whatever's convenient at implementation time — this doesn't depend on tab order).

1. **Class-name check** (auto-runs whenever `st.session_state.model` is loaded/reloaded,
   no button needed): a pass/fail badge plus a small table of `(index, expected, actual)`
   for any mismatches. Reads `st.session_state.model.names` (Ultralytics models expose
   this on both `.pt` and `.onnx`-loaded instances).

2. **"Run full diagnostic on last frame" button** (disabled with a hint if
   `st.session_state.last_frame` is unset): runs, in order —
   - `compare_models` (`.pt` vs. whatever's active) — table of disagreements.
   - `confidence_sweep` on the active model — table of survivors per threshold.
   - `zones.classify_zones` on the active model's detections at the current sidebar
     conf/iou settings — the existing mismatch flag, surfaced here explicitly rather
     than just as an inline `(!)` on the video.

3. **Sample A/B comparison** (the framing/domain-shift test): two buttons, "Save last
   frame as sample A" / "...as sample B", each storing
   `{frame, detections, zone_report}` in `st.session_state.diag_samples["A"|"B"]`.
   Workflow: stream close-up → stop → save as A; back up so your torso is in frame →
   stream → stop → save as B. Once both exist, render their full diagnostic reports
   side by side (`st.columns(2)`) so it's immediately visible whether the vest-on-head
   detection is present in the close-up sample but absent (or resolves correctly) in the
   full-body sample.

Nothing here writes to disk or persists across app restarts — samples A/B live only in
`st.session_state` for the duration of the debugging session.

## Testing approach

No existing test suite covers `app.py`/`utils/` in this project. Verification is manual:

- Run `streamlit run app.py`, load the model, confirm the class-name check shows "OK"
  (sanity check that hypothesis 1 is a non-issue before going further).
- Reproduce the bug on webcam, stop the stream, run the full diagnostic, and read off
  whether `.pt` and `.onnx` agree (hypothesis 2), whether the vest box survives at
  conf=0.9 (hypothesis 3), and what the zone-mismatch report says.
- Run the sample A/B workflow (close-up vs. backed-up) and compare the two reports —
  this is the direct test of hypothesis 4.
- Report back which hypothesis the evidence points to; no further action taken in this
  design's scope.
