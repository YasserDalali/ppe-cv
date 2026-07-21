"""
Temporal filtering: only "confirm" a detection once it has persisted across
N consecutive frames. Uses ByteTrack track IDs when available (accurate,
handles brief occlusion); falls back to IoU box-matching frame-to-frame if
tracking is disabled.
"""
import time
from collections import defaultdict


class TemporalFilter:
    def __init__(self, min_frames=5, iou_threshold=0.3, timeout=1.5):
        self.min_frames = min_frames
        self.iou_threshold = iou_threshold
        self.timeout = timeout
        self.id_counts = defaultdict(int)
        self.id_last_seen = {}
        self.fallback_tracks = []  # used only when no track_id is present

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        """
        detections: list of Detection objects (see utils/inference.py)
        Returns: list of confirmed Detection objects only.
        """
        now = time.time()
        confirmed = []
        has_ids = any(d.track_id is not None for d in detections)

        if has_ids:
            for d in detections:
                if d.track_id is None:
                    continue
                self.id_counts[d.track_id] += 1
                self.id_last_seen[d.track_id] = now
                if self.id_counts[d.track_id] >= self.min_frames:
                    confirmed.append(d)

            stale = [tid for tid, last in self.id_last_seen.items() if now - last > self.timeout]
            for tid in stale:
                self.id_counts.pop(tid, None)
                self.id_last_seen.pop(tid, None)
        else:
            for d in detections:
                matched = False
                for track in self.fallback_tracks:
                    if track["cls"] == d.cls_name and self._iou(d.bbox, track["bbox"]) > self.iou_threshold:
                        track["bbox"] = d.bbox
                        track["count"] += 1
                        track["last_seen"] = now
                        matched = True
                        if track["count"] >= self.min_frames:
                            confirmed.append(d)
                        break
                if not matched:
                    self.fallback_tracks.append(
                        {"bbox": d.bbox, "cls": d.cls_name, "count": 1, "last_seen": now}
                    )
            self.fallback_tracks = [t for t in self.fallback_tracks if now - t["last_seen"] < self.timeout]

        return confirmed

    def reset(self):
        self.id_counts.clear()
        self.id_last_seen.clear()
        self.fallback_tracks.clear()
