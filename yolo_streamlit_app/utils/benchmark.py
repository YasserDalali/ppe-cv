"""
Benchmark utility: measures inference latency/FPS across resolution presets
using a handful of frames from the active source, so the settings page can
show a concrete resource-vs-accuracy tradeoff before you commit to settings.

Detection count is used as a rough *stability* proxy (does the model keep
finding the same objects as resolution drops) -- it is NOT a substitute for
real mAP/precision-recall evaluation on a labeled test set. If you have
labeled validation data, use Ultralytics' `model.val()` for true accuracy
numbers instead.
"""
import time
import pandas as pd
from utils.streaming import resize_for_inference, RESOLUTION_PRESETS


def run_benchmark(model, sample_frames, conf=0.5, iou=0.45, use_tracker=False):
    rows = []
    for label, size in RESOLUTION_PRESETS.items():
        latencies = []
        det_counts = []
        for frame in sample_frames:
            resized = resize_for_inference(frame, size)
            t0 = time.time()
            result = model.predict(resized, conf=conf, iou=iou, use_tracker=use_tracker)
            detections = result[0]
            latencies.append((time.time() - t0) * 1000)
            det_counts.append(len(detections))

        avg_latency = sum(latencies) / len(latencies)
        avg_dets = sum(det_counts) / len(det_counts)
        rows.append({
            "resolution": label,
            "px": size,
            "avg_latency_ms": round(avg_latency, 1),
            "est_fps": round(1000 / avg_latency, 1) if avg_latency > 0 else 0,
            "avg_detections": round(avg_dets, 2),
        })

    return pd.DataFrame(rows)
