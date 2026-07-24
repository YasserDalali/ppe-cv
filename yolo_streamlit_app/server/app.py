"""
Standalone FastAPI inference server. Deploy this separately (Hugging Face
Spaces with the Docker SDK, Render, a VPS, or a GPU box) and point the
Streamlit app's "Remote" mode at its URL (e.g. https://your-space.hf.space).
"""
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO

app = FastAPI(title="YOLO Inference Server")
model = YOLO("models/best.onnx")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.5,
    iou: float = 0.45,
    use_tracker: bool = True,
):
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if use_tracker:
        results = model.track(
            frame, conf=conf, iou=iou, tracker="bytetrack.yaml", persist=True, verbose=False
        )
    else:
        results = model.predict(frame, conf=conf, iou=iou, verbose=False)

    r = results[0]
    detections = []
    for box in r.boxes:
        track_id = int(box.id[0]) if (use_tracker and box.id is not None) else None
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist(),
            "track_id": track_id,
        })

    return {"detections": detections}
