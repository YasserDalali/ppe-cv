"""
Video source handling: webcam index, RTSP/IP camera URL, or an uploaded file.
Also holds the resolution presets used for the resource/accuracy tradeoff.
"""
import cv2

RESOLUTION_PRESETS = {
    "320 (fastest, lowest accuracy)": 320,
    "480": 480,
    "640 (balanced, default)": 640,
    "960": 960,
    "1280 (best accuracy, slowest)": 1280,
}


def open_source(source_type, source_value):
    """
    source_type: "webcam" | "rtsp" | "http" | "file"
    source_value: webcam index (int), RTSP/HTTP stream URL (str), or file path (str)
    """
    if source_type == "webcam":
        cap = cv2.VideoCapture(int(source_value))
    else:
        cap = cv2.VideoCapture(source_value)  # RTSP URL, HTTP/MJPEG URL, and file paths all work here

    if not cap.isOpened():
        raise ConnectionError(f"Could not open source: {source_value}")
    return cap


def resize_for_inference(frame, target_size):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))
