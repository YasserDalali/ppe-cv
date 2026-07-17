"""Step 5 — predictor backends behind one interface.

UltralyticsPredictor runs a local .pt; RoboflowPredictor runs the hosted
generic Model B, preferring the `inference` package (model runs locally in
Colab, no per-image API metering) and falling back to the hosted REST API.
The known-risk swap between those two paths lives entirely in this file.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


class PredictError(RuntimeError):
    pass


@dataclass
class Boxes:
    xyxy: np.ndarray                       # (N,4) float, absolute pixels
    class_names: list = field(default_factory=list)   # backend's own names, len N
    confidence: np.ndarray = None          # (N,)

    @staticmethod
    def empty() -> "Boxes":
        return Boxes(np.zeros((0, 4)), [], np.zeros((0,)))


def _load_yolo_model(weights: Path):
    from ultralytics import YOLO  # lazy: runtime-only dependency

    return YOLO(str(weights))


def _load_inference_model(model_id: str, api_key: str):
    from inference import get_model  # lazy; ImportError triggers REST fallback

    return get_model(model_id=model_id, api_key=api_key)


def _rest_infer(image_path: Path, model_id: str, api_key: str, conf: float, iou: float) -> dict:
    """Hosted REST fallback: POST base64 image to detect.roboflow.com."""
    data = base64.b64encode(Path(image_path).read_bytes())
    url = (f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
           f"&confidence={int(conf * 100)}&overlap={int(iou * 100)}")
    req = Request(url, data=data,
                  headers={"Content-Type": "application/x-www-form-urlencoded"})
    return json.load(urlopen(req))


def _fetch_class_names(api_key: str, workspace: str, project: str, version: int) -> list[str]:
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}?api_key={api_key}"
    data = json.load(urlopen(url))
    classes = data.get("version", {}).get("classes") or data.get("project", {}).get("classes")
    if isinstance(classes, dict):
        return sorted(classes)
    if isinstance(classes, list):
        return list(classes)
    raise PredictError(f"could not read class names for {workspace}/{project}/{version}")


class UltralyticsPredictor:
    def __init__(self, weights: Path, conf: float, iou: float, imgsz: int = 640):
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self._model = _load_yolo_model(weights)

    @property
    def class_names(self) -> list[str]:
        names = self._model.names
        return [str(names[k]) for k in sorted(names)]

    def predict(self, image_path: Path) -> Boxes:
        results = self._model.predict(source=str(image_path), conf=self.conf,
                                      iou=self.iou, imgsz=self.imgsz, verbose=False)
        r = results[0]
        if r.boxes is None:
            return Boxes.empty()
        xyxy = r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
        if len(xyxy) == 0:
            return Boxes.empty()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy().astype(float)
        names = self._model.names
        return Boxes(xyxy, [str(names[int(c)]) for c in cls_ids], conf)


class RoboflowPredictor:
    """Hosted Model B. backend="auto" prefers the local `inference` package."""

    def __init__(self, model_id: str, api_key: str, conf: float, iou: float,
                 class_names: list[str] | None = None,
                 metadata_coords: tuple[str, str, int] | None = None,
                 backend: str = "auto"):
        self.model_id, self._api_key = model_id, api_key
        self.conf, self.iou = conf, iou
        self._explicit_names = class_names
        self._metadata_coords = metadata_coords
        self._fetched_names: list[str] | None = None
        self._model = None
        self.backend = None
        if backend in ("auto", "inference"):
            try:
                self._model = _load_inference_model(model_id, api_key)
                self.backend = "inference"
            except ImportError:
                if backend == "inference":
                    raise
        if self.backend is None:
            self.backend = "rest"

    @property
    def class_names(self) -> list[str]:
        if self._explicit_names is not None:
            return list(self._explicit_names)
        model_names = getattr(self._model, "class_names", None)
        if model_names:
            return list(model_names)
        if self._metadata_coords is not None:
            if self._fetched_names is None:
                self._fetched_names = _fetch_class_names(self._api_key, *self._metadata_coords)
            return list(self._fetched_names)
        raise PredictError(
            f"cannot determine class names for hosted model {self.model_id}: "
            "backend exposes none — pass class_names=... or "
            "metadata_coords=(workspace, project, version)"
        )

    def predict(self, image_path: Path) -> Boxes:
        if self.backend == "inference":
            responses = self._model.infer(str(image_path), confidence=self.conf,
                                          iou_threshold=self.iou)
            preds = responses[0].predictions
            rows = [(p.x, p.y, p.width, p.height,
                     getattr(p, "class_name", None) or getattr(p, "class_", ""),
                     p.confidence) for p in preds]
        else:
            data = _rest_infer(image_path, self.model_id, self._api_key, self.conf, self.iou)
            rows = [(p["x"], p["y"], p["width"], p["height"], p["class"], p["confidence"])
                    for p in data.get("predictions", [])]
        if not rows:
            return Boxes.empty()
        xyxy = np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                         for x, y, w, h, _, _ in rows], dtype=float)
        return Boxes(xyxy, [str(c) for _, _, _, _, c, _ in rows],
                     np.array([cf for *_, cf in rows], dtype=float))
