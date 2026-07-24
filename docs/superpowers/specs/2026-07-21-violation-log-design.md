# Violation Log — Design

Date: 2026-07-21
Scope: `yolo_streamlit_app/`

## Problem

The Streamlit app already detects PPE zone mismatches live (a PPE item detected in a
body zone it has no business being in — e.g. a "helmet" near someone's feet) and flags
them on-screen with a red `(!)` box. Nothing persists these events. We want every such
violation saved locally (screenshot + metadata) and viewable in a new page within the
app, as a stopgap until a real database is plugged in.

## Non-goals

- No database integration yet (explicitly deferred by the user).
- No alerting/notification (email, webhook, etc.).
- No auth/access control on the log viewer.
- No log rotation/retention policy beyond a display cap in the UI.

## Architecture

New module `yolo_streamlit_app/utils/violation_log.py` containing a `ViolationLogger`
class, following the existing `TemporalFilter` pattern (`utils/tracking.py`): a stateful
object instantiated once and stored in `st.session_state`, updated once per processed
frame from the live-detection loop in `app.py`.

- `st.session_state.violation_logger = ViolationLogger()` — created alongside the
  existing `st.session_state.temporal_filter`.
- In the live loop in `app.py`, immediately after `zones.classify_zones(confirmed)`
  produces `items` (guarded by the existing `enable_zones and flag_mismatches`
  checkboxes — logging reuses that existing toggle, no new sidebar control), call
  `violation_logger.process(item, persons, resized, source_type, source_value)` for
  every item where `item["mismatch"]` is `True`.
- All filesystem I/O (screenshot encoding/saving, JSONL append/read/clear) is
  encapsulated inside `ViolationLogger`. `app.py` and the new UI tab never touch the
  filesystem directly — they call methods on the logger.
- Storage root: `yolo_streamlit_app/logs/`, resolved via
  `Path(__file__).resolve().parent.parent / "logs"` from within
  `utils/violation_log.py`, so it's independent of the process's working directory.
  Added to `.gitignore` (`yolo_streamlit_app/logs/`).

## Incident lifecycle (dedup)

A single ongoing violation (e.g. one person's misplaced helmet, flagged every frame for
several seconds) must produce one log entry, not one per frame. `ViolationLogger` tracks
this with an in-memory dict:

```python
open_incidents: dict[key, last_seen_timestamp]
```

- **Key**: `(track_id, cls_name)` when the detection carries a `track_id` (tracker
  enabled — the default). This uniquely identifies "this specific PPE item on this
  specific tracked person."
- **Fallback key** (`track_id is None`, tracker disabled): no stable identity is
  available, so match against currently-open incidents of the same `cls_name` by IoU on
  the item's bbox (threshold 0.3), mirroring the existing no-tracker fallback already
  used by `TemporalFilter.update()` in `utils/tracking.py`. If no open incident of that
  class matches by IoU, treat as new.
- **New incident** (key/IoU-match not found in `open_incidents`): draw the screenshot,
  save it, append a JSONL record, add the key to `open_incidents` with
  `last_seen = now`.
- **Ongoing incident** (match found): update `last_seen = now` only. No new file or
  record is written.
- **Expiry**: at the top of every `process()` call, drop any `open_incidents` entry
  whose `last_seen` is more than `timeout` seconds old (default **2.0s** — deliberately
  above `TemporalFilter`'s own 1.5s timeout so a single dropped/late frame doesn't cause
  a spurious re-open). Once expired, the same violation reappearing later is logged as a
  brand-new incident.

## Screenshot content

Per the "full frame, but only that violation highlighted" requirement: the screenshot is
a fresh copy of the *raw resized frame* (`resized.copy()`, before any other boxes/zone
lines are drawn on it) with only the single violating detection's box drawn — red
rectangle + label `"{cls_name} | {zone_label} (!)"` — using the same drawing style
already used in the live-view loop. No other detections, person boxes, or zone divider
lines appear on it.

## Data schema & file layout

```
yolo_streamlit_app/logs/
  violations.jsonl
  screenshots/
    2026-07-21T14-32-05.123_helmet_track7.jpg
    ...
```

Filename pattern: `{timestamp with ':' -> '-'}_{cls_name}_{track id or 'na'}.jpg`
(filesystem-safe, sortable, self-describing).

One JSON object per line, appended (never rewritten in place):

```json
{
  "timestamp": "2026-07-21T14:32:05.123",
  "track_id": 7,
  "cls_name": "helmet",
  "conf": 0.81,
  "bbox": [x1, y1, x2, y2],
  "zone_idx": 2,
  "zone_label": "legs/bottom",
  "expected_zone_idx": 0,
  "person_bbox": [x1, y1, x2, y2],
  "source_type": "file",
  "source_value": "OCP.mp4",
  "image_path": "screenshots/2026-07-21T14-32-05.123_helmet_track7.jpg"
}
```

`image_path` is relative to `logs/`, so the log directory can be moved/archived as a
unit. `track_id` may be `null` (no-tracker mode).

## UI — "Violation Log" tab

Fourth tab added to the existing `st.tabs(...)` call in `app.py`:
`["Live Detection", "Benchmark", "About / Deployment", "Violation Log"]`.

- On each render, reads `violations.jsonl` via `violation_logger.load_records()`, most
  recent first, capped to the last 200 entries (perf guard — this is a flat-file reader
  with no indexing).
- A `st.selectbox` class filter, defaulting to "All classes", populated from the classes
  actually present in the loaded records.
- Reverse-chronological card list: each card renders `st.image(screenshot_path)` plus a
  caption line (`timestamp · cls_name · track #N · zone_label · conf`).
- A "Clear log" control gated behind a confirmation checkbox (it's a destructive,
  irreversible action — deletes `violations.jsonl` and every file under
  `screenshots/`), consistent with treating destructive actions carefully.
- A caption showing the total entry count (pre-filter, pre-cap).

## Testing approach

No existing test suite in this project. Verification is manual:

- Run `streamlit run app.py` against the bundled `OCP.mp4` demo clip with "Flag PPE
  detected in the wrong zone" enabled, confirm entries appear in the Violation Log tab
  with correct screenshots and metadata.
- Watch `open_incidents` behavior over a longer run to confirm a persisting violation
  produces exactly one entry, and that it produces a second entry only after clearing
  (person/PPE leaves frame or the mismatch resolves) and reoccurring.
- Toggle "Enable ByteTrack object tracking" off to exercise the no-tracker IoU-fallback
  dedup path.
