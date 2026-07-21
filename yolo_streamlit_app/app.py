import time
import cv2
import psutil
import streamlit as st

from utils.inference import load_model
from utils.tracking import TemporalFilter
from utils.streaming import open_source, resize_for_inference, RESOLUTION_PRESETS
from utils.benchmark import run_benchmark
from utils import zones

st.set_page_config(page_title="YOLO Detection Dashboard", layout="wide")

# ---------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "model_mode" not in st.session_state:
    st.session_state.model_mode = None
if "temporal_filter" not in st.session_state:
    st.session_state.temporal_filter = TemporalFilter()

if st.session_state.model is None:
    try:
        st.session_state.model = load_model("local", weights_path="models/best.onnx")
        st.session_state.model_mode = "local"
    except Exception:
        pass  # sidebar "Load / reload model" button remains available as a fallback

st.title("YOLO Detection Dashboard")

tab_live, tab_benchmark, tab_about = st.tabs(["Live Detection", "Benchmark", "About / Deployment"])

# ===========================================================
# SIDEBAR -- all configuration lives here
# ===========================================================
with st.sidebar:
    st.header("Inference")
    mode_label = st.radio("Run model", ["Remote (hosted demo)", "Local (this machine)"], index=1)
    mode_key = "remote" if mode_label.startswith("Remote") else "local"

    weights_path = "models/best.onnx"
    remote_url = ""
    if mode_key == "local":
        weights_path = st.text_input("Local weights path", value="models/best.onnx")
    else:
        remote_url = st.text_input(
            "Remote inference server URL",
            value="https://your-space-or-server.example.com",
            help="Points at the FastAPI server in server/app.py once deployed. See the About tab.",
        )

    if st.button("Load / reload model", use_container_width=True):
        try:
            with st.spinner("Loading model..."):
                st.session_state.model = load_model(mode_key, weights_path=weights_path, remote_url=remote_url)
                st.session_state.model_mode = mode_key
            st.success(f"Model loaded ({mode_key}).")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.divider()
    st.header("Source")
    source_type_label = st.selectbox(
        "Source type", ["IP Camera (RTSP)", "IP Camera (HTTP/MJPEG)", "Webcam", "Video file"]
    )
    source_type = {
        "IP Camera (RTSP)": "rtsp",
        "IP Camera (HTTP/MJPEG)": "http",
        "Webcam": "webcam",
        "Video file": "file",
    }[source_type_label]

    source_value = None
    if source_type == "rtsp":
        source_value = st.text_input("RTSP URL", value="rtsp://user:pass@camera_ip:554/stream1")
    elif source_type == "http":
        source_value = st.text_input("HTTP/MJPEG stream URL", value="http://camera_ip:8080/video")
    elif source_type == "webcam":
        source_value = st.number_input("Webcam index", min_value=0, max_value=10, value=0)
    else:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            temp_path = f"/tmp/{uploaded.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())
            source_value = temp_path

    st.divider()
    st.header("Detection tuning")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)
    iou = st.slider("NMS / IoU threshold", 0.05, 0.95, 0.45, 0.05)
    resolution_label = st.select_slider(
        "Inference resolution (resource vs accuracy)",
        options=list(RESOLUTION_PRESETS.keys()),
        value="640 (balanced, default)",
    )
    target_fps = st.slider("Target processing FPS (throttle)", 1, 30, 10)

    st.divider()
    st.header("Tracking & filtering")
    use_tracker = st.checkbox("Enable ByteTrack object tracking", value=True)
    min_frames = st.slider("Temporal filter: min consecutive frames to confirm", 1, 30, 5)
    st.session_state.temporal_filter.min_frames = min_frames

    st.divider()
    st.header("PPE zone classification")
    enable_zones = st.checkbox("Split person box into 3 zones (top/mid/bottom)", value=True)
    flag_mismatches = st.checkbox("Flag PPE detected in the wrong zone", value=True)

    st.divider()
    show_resources = st.checkbox("Show live resource usage", value=True)

# ===========================================================
# LIVE DETECTION TAB
# ===========================================================
with tab_live:
    col_video, col_stats = st.columns([3, 1])
    frame_slot = col_video.empty()
    stats_slot = col_stats.empty()

    run = st.checkbox("Start / Stop stream", value=False, key="run_checkbox")

    if run:
        if st.session_state.model is None:
            st.warning("Load a model from the sidebar first.")
        elif source_value is None:
            st.warning("Provide a valid source in the sidebar first.")
        else:
            try:
                cap = open_source(source_type, source_value)
            except ConnectionError as e:
                st.error(str(e))
                cap = None

            if cap is not None:
                target_size = RESOLUTION_PRESETS[resolution_label]
                frame_interval = 1.0 / target_fps
                last_time = 0.0

                # NOTE: this while loop relies on Streamlit's standard behavior of
                # interrupting a running script when a new widget interaction comes in.
                # Unchecking "Start / Stop stream" triggers a rerun that stops this loop.
                while st.session_state.run_checkbox:
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Stream ended or disconnected.")
                        break

                    now = time.time()
                    if now - last_time < frame_interval:
                        continue
                    last_time = now

                    resized = resize_for_inference(frame, target_size)
                    model = st.session_state.model
                    result = model.predict(resized, conf=conf, iou=iou, use_tracker=use_tracker)

                    if len(result) == 3:
                        detections, latency_ms, err = result
                        if err:
                            st.error(f"Remote inference error: {err}")
                            break
                    else:
                        detections, latency_ms = result

                    confirmed = st.session_state.temporal_filter.update(detections)
                    annotated = resized.copy()
                    mismatch_count = 0

                    if enable_zones:
                        persons, items = zones.classify_zones(confirmed)

                        # person boxes + dashed zone divider lines
                        for p in persons:
                            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            for by in p["boundaries"][1:-1]:
                                y = int(by)
                                for dx in range(x1, x2, 12):  # dashed line
                                    cv2.line(annotated, (dx, y), (min(dx + 6, x2), y), (255, 200, 0), 1)

                        # PPE item boxes, colored by zone match
                        for it in items:
                            d = it["detection"]
                            x1, y1, x2, y2 = [int(v) for v in d.bbox]
                            is_bad = flag_mismatches and it["mismatch"]
                            color = (0, 0, 255) if is_bad else (0, 255, 0)
                            label = f"{d.cls_name} {d.conf:.2f}"
                            if it["zone_label"]:
                                label += f" | {it['zone_label']}"
                            if is_bad:
                                label += " (!)"
                                mismatch_count += 1
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        for d in confirmed:
                            x1, y1, x2, y2 = [int(v) for v in d.bbox]
                            label = f"{d.cls_name} {d.conf:.2f}"
                            if d.track_id is not None:
                                label += f" #{d.track_id}"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    frame_slot.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                    with stats_slot.container():
                        st.metric("Latency", f"{latency_ms:.0f} ms")
                        st.metric("Est. FPS", f"{1000/latency_ms:.1f}" if latency_ms > 0 else "-")
                        st.metric("Raw detections", len(detections))
                        st.metric("Confirmed (post-filter)", len(confirmed))
                        if enable_zones and flag_mismatches:
                            st.metric("Zone mismatches flagged", mismatch_count)
                        if show_resources:
                            st.metric("CPU", f"{psutil.cpu_percent()}%")
                            st.metric("RAM", f"{psutil.virtual_memory().percent}%")

                cap.release()
    else:
        st.info("Toggle 'Start / Stop stream' above to begin.")

# ===========================================================
# BENCHMARK TAB
# ===========================================================
with tab_benchmark:
    st.subheader("Accuracy vs. resource consumption")
    st.caption(
        "Grabs a handful of frames from your current source and runs inference at "
        "each resolution preset, so you can see the latency/FPS tradeoff before "
        "committing to settings. Detection count is a rough stability proxy, not a "
        "substitute for true mAP evaluation on labeled data."
    )

    n_sample_frames = st.slider("Sample frames to test with", 3, 20, 5)

    if st.button("Run benchmark"):
        if st.session_state.model is None:
            st.warning("Load a model from the sidebar first.")
        elif source_value is None:
            st.warning("Provide a source in the sidebar first.")
        else:
            try:
                cap = open_source(source_type, source_value)
                frames = []
                for _ in range(n_sample_frames):
                    ok, f = cap.read()
                    if ok:
                        frames.append(f)
                cap.release()

                if not frames:
                    st.error("Couldn't grab any frames from the source.")
                else:
                    with st.spinner("Benchmarking..."):
                        df = run_benchmark(
                            st.session_state.model, frames, conf=conf, iou=iou, use_tracker=use_tracker
                        )

                    st.dataframe(df, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Estimated FPS by resolution (higher = lighter on resources)")
                        st.bar_chart(df.set_index("resolution")["est_fps"])
                    with col2:
                        st.caption("Avg detections found by resolution (rough accuracy proxy)")
                        st.bar_chart(df.set_index("resolution")["avg_detections"])
            except ConnectionError as e:
                st.error(str(e))

# ===========================================================
# ABOUT / DEPLOYMENT TAB
# ===========================================================
with tab_about:
    st.markdown(
        """
### How this app is set up

- **Local mode**: runs `best.onnx` directly in this Streamlit process via Ultralytics.
- **Remote mode**: sends frames to a hosted FastAPI server (`server/app.py`), so this
  Streamlit app can run on a thin/free host while inference happens elsewhere.

See `README.md` in this project for full deployment instructions (Hugging Face Spaces,
Render, VPS options) and guidance on tuning confidence/IoU/temporal filtering to
reduce false positives.
        """
    )
