import numpy as np
import pytest

from ppe.predict import PredictError, RoboflowPredictor, UltralyticsPredictor


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeUltraBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(cls)
        self.conf = FakeTensor(conf)


class FakeUltraResult:
    def __init__(self, xyxy, cls, conf):
        self.boxes = FakeUltraBoxes(xyxy, cls, conf)


class FakeYOLO:
    def __init__(self):
        self.names = {0: "person", 1: "helmet"}
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [FakeUltraResult([[0, 0, 10, 10], [5, 5, 20, 20]], [0, 1], [0.9, 0.8])]


def test_ultralytics_predictor_converts_and_passes_settings(monkeypatch, tmp_path):
    fake = FakeYOLO()
    monkeypatch.setattr("ppe.predict._load_yolo_model", lambda w: fake)
    p = UltralyticsPredictor(tmp_path / "best.pt", conf=0.25, iou=0.5, imgsz=640)
    assert p.class_names == ["person", "helmet"]
    boxes = p.predict(tmp_path / "img.png")
    assert boxes.xyxy.shape == (2, 4)
    assert boxes.class_names == ["person", "helmet"]
    assert boxes.confidence.tolist() == [0.9, 0.8]
    kw = fake.calls[0]
    assert kw["conf"] == 0.25 and kw["iou"] == 0.5 and kw["imgsz"] == 640
    assert kw["verbose"] is False


class FakeRfPrediction:
    def __init__(self, x, y, w, h, cls, conf):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.class_name, self.confidence = cls, conf


class FakeRfResponse:
    def __init__(self, preds):
        self.predictions = preds


class FakeInferenceModel:
    class_names = ["person", "helmet", "vest"]

    def __init__(self):
        self.calls = []

    def infer(self, source, **kwargs):
        self.calls.append(kwargs)
        return [FakeRfResponse([FakeRfPrediction(50, 50, 20, 10, "person", 0.77)])]


def test_roboflow_inference_backend_center_to_xyxy(monkeypatch, tmp_path):
    fake = FakeInferenceModel()
    monkeypatch.setattr("ppe.predict._load_inference_model", lambda mid, key: fake)
    p = RoboflowPredictor("ppe_detection-dnfen/3", "key", conf=0.25, iou=0.5)
    assert p.backend == "inference"
    assert p.class_names == ["person", "helmet", "vest"]
    boxes = p.predict(tmp_path / "img.png")
    assert boxes.xyxy.tolist() == [[40.0, 45.0, 60.0, 55.0]]
    assert boxes.class_names == ["person"] and boxes.confidence.tolist() == [0.77]
    kw = fake.calls[0]
    assert kw["confidence"] == 0.25 and kw["iou_threshold"] == 0.5


def test_roboflow_falls_back_to_rest(monkeypatch, tmp_path):
    def no_inference(mid, key):
        raise ImportError("inference not installed")

    calls = []

    def fake_rest(image_path, model_id, api_key, conf, iou):
        calls.append((model_id, conf, iou))
        return {"predictions": [
            {"x": 50, "y": 50, "width": 20, "height": 10, "class": "helmet", "confidence": 0.6},
        ]}

    monkeypatch.setattr("ppe.predict._load_inference_model", no_inference)
    monkeypatch.setattr("ppe.predict._rest_infer", fake_rest)
    monkeypatch.setattr("ppe.predict._fetch_class_names",
                        lambda key, ws, proj, ver: ["person", "helmet"])
    p = RoboflowPredictor("ppe_detection-dnfen/3", "key", conf=0.4, iou=0.5,
                          metadata_coords=("datasetppe", "ppe_detection-dnfen", 3))
    assert p.backend == "rest"
    boxes = p.predict(tmp_path / "img.png")
    assert boxes.xyxy.tolist() == [[40.0, 45.0, 60.0, 55.0]]
    assert boxes.class_names == ["helmet"]
    assert calls == [("ppe_detection-dnfen/3", 0.4, 0.5)]
    assert p.class_names == ["person", "helmet"]


def test_roboflow_explicit_class_names_win(monkeypatch):
    monkeypatch.setattr("ppe.predict._load_inference_model",
                        lambda mid, key: FakeInferenceModel())
    p = RoboflowPredictor("m/1", "key", conf=0.25, iou=0.5, class_names=["a", "b"])
    assert p.class_names == ["a", "b"]


def test_roboflow_no_names_available_is_readable(monkeypatch):
    def no_inference(mid, key):
        raise ImportError("nope")

    monkeypatch.setattr("ppe.predict._load_inference_model", no_inference)
    p = RoboflowPredictor("m/1", "key", conf=0.25, iou=0.5)
    with pytest.raises(PredictError) as e:
        _ = p.class_names
    assert "class names" in str(e.value)


def test_empty_predictions_give_empty_boxes(monkeypatch, tmp_path):
    class EmptyModel(FakeInferenceModel):
        def infer(self, source, **kwargs):
            return [FakeRfResponse([])]

    monkeypatch.setattr("ppe.predict._load_inference_model", lambda mid, key: EmptyModel())
    p = RoboflowPredictor("m/1", "key", conf=0.25, iou=0.5)
    boxes = p.predict(tmp_path / "img.png")
    assert boxes.xyxy.shape == (0, 4) and boxes.class_names == []
