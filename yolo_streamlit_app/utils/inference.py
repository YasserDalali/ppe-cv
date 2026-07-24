"""
Unified inference wrapper supporting local (Ultralytics, in-process) and
remote (HTTP call to a hosted FastAPI server) inference, behind one interface
so the rest of the app doesn't care which mode is active.
"""
import time
import requests
import numpy as np
import cv2


class Detection:
    __slots__ = ["bbox", "cls_name", "conf", "track_id"]

    def __init__(self, bbox, cls_name, conf, track_id=None):
        self.bbox = bbox          # [x1, y1, x2, y2]
        self.cls_name = cls_name
        self.conf = conf
        self.track_id = track_id


class LocalModel:
    """Runs the YOLO model in-process, on whatever CPU/GPU this machine has."""

    def __init__(self, weights_path="models/best.onnx"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.names = self.model.names

    def predict(self, frame, conf=0.5, iou=0.45, use_tracker=True, tracker_cfg="bytetrack.yaml"):
        t0 = time.time()
        if use_tracker:
            results = self.model.track(
                frame, conf=conf, iou=iou, tracker=tracker_cfg,
                persist=True, verbose=False
            )
        else:
            results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)
        latency_ms = (time.time() - t0) * 1000

        detections = []
        r = results[0]
        for box in r.boxes:
            cls_name = self.names[int(box.cls)]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if (use_tracker and box.id is not None) else None
            detections.append(Detection(bbox, cls_name, confidence, track_id))

        return detections, latency_ms


class RemoteModel:
    """Calls a hosted FastAPI inference server (see server/app.py) instead of
    loading the model in this process. Useful so the Streamlit app itself can
    run on a thin/free host while the actual model runs elsewhere."""

    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url.rstrip("/")

    def predict(self, frame, conf=0.5, iou=0.45, use_tracker=True, tracker_cfg="bytetrack.yaml"):
        t0 = time.time()
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return [], 0.0, "Failed to encode frame"

        files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
        params = {"conf": conf, "iou": iou, "use_tracker": use_tracker}

        try:
            resp = requests.post(f"{self.endpoint_url}/predict", files=files, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return [], 0.0, str(e)

        latency_ms = (time.time() - t0) * 1000
        detections = [
            Detection(d["bbox"], d["class"], d["confidence"], d.get("track_id"))
            for d in data.get("detections", [])
        ]
        return detections, latency_ms, None


def load_model(mode, weights_path=None, remote_url=None):
    if mode == "local":
        return LocalModel(weights_path or "models/best.onnx")
    elif mode == "remote":
        if not remote_url:
            raise ValueError("remote_url is required for remote mode")
        return RemoteModel(remote_url)
    raise ValueError(f"Unknown mode: {mode}")
