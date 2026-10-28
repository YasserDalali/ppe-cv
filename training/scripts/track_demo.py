"""ByteTrack demo on one held-out OCP video (reviewer R1-5, R2-11).

Runs a fold model with ByteTrack over a video whose recording is in that
fold's TEST set (refuses otherwise, so the demo never shows a video the
model trained on). Compares two alert rules on the same detections:

  raw      an alert whenever a violation box (no_helmet / no_gloves) appears
           in a frame where the previous frame had none
  tracked  the app's rule (yolo_streamlit_app/utils/tracking.py): a track ID
           alerts once, after it persisted MIN_FRAMES consecutive frames

There are no frame-level violation labels for the plant videos, so this
shows alert stability, not accuracy.

Usage (local, CPU is fine):
  .venv/bin/python scripts/track_demo.py --weights A_f0/best.pt \
      --video ~/Desktop/PPE-OCP/done/VID_20260716_163549.mp4 \
      --splits pool/splits.json --fold 0 --out track_demo/
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.pool import recording_id  # noqa: E402

VIOLATIONS = {"no_helmet", "no_gloves"}
MIN_FRAMES = 5  # same as TemporalFilter(min_frames=5) in the app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True, help="pool splits.json")
    ap.add_argument("--fold", type=int, required=True, help="fold the weights were trained for")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    folds = json.loads(args.splits.read_text())["ocp"]["folds"]
    rec = recording_id(args.video.name)
    if rec not in folds[str(args.fold)]:
        raise SystemExit(f"{rec} is not in fold {args.fold}'s test recordings "
                         f"{folds[str(args.fold)]} — the model trained on it")

    import cv2
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = None
    rows, strip = [], []
    prev_raw = False
    raw_events = 0
    streak: dict = {}
    confirmed: set = set()
    track_ids = set()
    results = model.track(source=str(args.video), tracker="bytetrack.yaml", stream=True,
                          persist=True, conf=args.conf, imgsz=args.imgsz, verbose=False)
    for i, r in enumerate(results):
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None else []
        ids = r.boxes.id.int().tolist() if (r.boxes is not None and r.boxes.id is not None) else [None] * len(names)
        track_ids.update(t for t in ids if t is not None)
        viol_ids = {t for n, t in zip(names, ids) if n in VIOLATIONS and t is not None}
        raw = any(n in VIOLATIONS for n in names)
        raw_events += int(raw and not prev_raw)
        prev_raw = raw
        streak = {t: streak.get(t, 0) + 1 for t in viol_ids}
        new = {t for t, k in streak.items() if k >= MIN_FRAMES and t not in confirmed}
        confirmed |= new
        rows.append({"frame": i, "time_s": round(i / fps, 2), "boxes": len(names),
                     "violation_boxes": sum(n in VIOLATIONS for n in names),
                     "raw_alert": int(raw), "new_confirmed_alerts": len(new)})
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.out / f"{rec}_bytetrack.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        strip.append(frame)
    if writer is not None:
        writer.release()
    if not rows:
        raise SystemExit(f"no frames decoded from {args.video}")

    with (args.out / f"{rec}_frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    picks = [strip[int(k * (len(strip) - 1) / 3)] for k in range(4)]
    small = [cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))) for f in picks]
    cv2.imwrite(str(args.out / f"{rec}_strip.jpg"), cv2.hconcat(small))
    minutes = len(rows) / fps / 60
    summary = {
        "video": args.video.name, "recording": rec, "fold": args.fold, "weights": str(args.weights),
        "frames": len(rows), "fps": round(fps, 2), "minutes": round(minutes, 2),
        "track_ids": len(track_ids), "min_frames": MIN_FRAMES, "conf": args.conf,
        "raw_alert_events": raw_events, "tracked_confirmed_alerts": len(confirmed),
        "raw_alerts_per_min": round(raw_events / minutes, 2) if minutes else None,
        "tracked_alerts_per_min": round(len(confirmed) / minutes, 2) if minutes else None,
        "note": "alert stability only — plant videos have no frame-level violation labels",
    }
    (args.out / f"{rec}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
