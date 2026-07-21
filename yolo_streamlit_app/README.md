# YOLO Detection Dashboard

A Streamlit app for `best.onnx` (ONNX-exported from the trained `best.pt`): live detection from IP cameras/RTSP/webcam/video files,
a settings panel for confidence/IoU/resolution/FPS, ByteTrack object tracking, temporal
filtering to cut false positives, and a benchmark tab to visualize the accuracy-vs-resource
tradeoff. Supports running the model **locally** or against a **remote** hosted inference
server, toggleable from the sidebar.

## Project structure

```
yolo_streamlit_app/
├── app.py                 # Main Streamlit app
├── requirements.txt        # Deps for the Streamlit app
├── models/
│   └── best.onnx           # <-- ONNX-exported weights for LOCAL mode
├── scripts/
│   └── export_onnx.py      # Exports paper/results/best.pt -> ONNX, copies into both model slots
├── utils/
│   ├── inference.py        # Local + remote model wrappers
│   ├── tracking.py         # Temporal filter (frame-persistence confirmation)
│   ├── streaming.py        # Source (webcam/RTSP/file) handling + resizing
│   └── benchmark.py        # Latency/FPS-vs-resolution benchmarking
└── server/                 # Standalone FastAPI inference server (for REMOTE mode)
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    └── models/
        └── best.onnx        # <-- separate copy of ONNX weights for the server deployment
```

## Running locally

```bash
cd yolo_streamlit_app
pip install -r requirements.txt
# (re)generate the ONNX weights from the trained paper/results/best.pt:
python scripts/export_onnx.py
streamlit run app.py
```

In the sidebar, choose **Local (this machine)** as the inference mode, set the weights
path (defaults to `models/best.onnx`), click **Load / reload model**, pick your source, and
toggle **Start / Stop stream**.

## Deploying the demo for free (so it doesn't run on your CPU)

There are two independent pieces you can deploy separately:

### 1. The FastAPI inference server (`server/`) — Hugging Face Spaces (Docker SDK)

This is the piece that actually runs the model. Free tier gives you a CPU Space (works
fine for a small YOLO model, e.g. `n`/`s` variants — this one already ships ONNX-exported)
or a paid GPU Space if you need more throughput.

1. Create a new Space on huggingface.co → SDK: **Docker**.
2. Push the contents of `server/` (including a copy of `best.onnx` in `server/models/`) to
   the Space's repo.
3. It will build the Dockerfile and expose the app on port 7860 automatically.
4. Your inference URL will be `https://<your-username>-<space-name>.hf.space`.

Note: Spaces sleep after inactivity on the free tier and wake on the next request
(~10-30s cold start) — fine for demos, not for 24/7 monitoring.

### 2. The Streamlit app (`app.py`) — Hugging Face Spaces (Streamlit SDK) or Streamlit Cloud

Since this app talks to the remote server over HTTP in "Remote" mode, the Streamlit app
itself barely does any compute — it just displays frames and draws boxes — so it runs
comfortably on any free CPU tier:

- **Streamlit Community Cloud**: connect your GitHub repo, free, no Docker needed.
- **Hugging Face Spaces (Streamlit SDK)**: same idea, push `app.py` + `requirements.txt` +
  `utils/`.

In the sidebar, set mode to **Remote (hosted demo)** and paste the server's URL
(e.g. `https://<your-username>-<space-name>.hf.space`).

### Switching between local and remote

The sidebar toggle is exactly this: pick **Local** to run `best.onnx` in-process (needs the
weights in `models/best.onnx` on whatever machine is running the Streamlit app), or
**Remote** to call the hosted FastAPI server instead. No code changes needed — it's a
runtime setting.

### The honest limitation of free tiers

Free hosting works well for demos and occasional single-stream testing. Continuous 24/7
inference on a live CCTV feed is a sustained-compute workload, which is exactly what free
tiers throttle/sleep to prevent. For actual round-the-clock camera monitoring, budget for
a small VPS (~$5-15/mo, e.g. Hetzner/OVH, CPU-only is fine for a small ONNX-exported
model) or a GPU box if you need multiple concurrent streams.

## Reducing false positives

The app exposes the main levers directly:

- **Confidence threshold**: the cheapest lever. Push it up if you're seeing noisy low-
  confidence boxes.
- **IoU/NMS threshold**: reduces duplicate/overlapping boxes on the same object.
- **Resolution**: lower resolution = faster but occasionally noisier on small objects;
  use the Benchmark tab to see the actual tradeoff on your footage before deciding.
- **Temporal filtering** (`min consecutive frames to confirm`): a detection only gets
  drawn/reported once it has persisted for N frames — single-frame flicker false
  positives get filtered out automatically. Raise this if false positives still slip
  through; lower it if you're missing genuinely fast-moving objects.
- **ByteTrack**: keeps a persistent ID per object across frames (handles brief occlusion),
  which is what the temporal filter uses to count "N consecutive frames" per object
  rather than just per raw box.

If false positives cluster around specific backgrounds/lighting even after tuning these,
the next step is retraining with more negative examples from your actual footage —
tuning inference settings can only go so far if the model itself hasn't seen that pattern.

## Benchmark tab

Grabs a few frames from your current source and runs inference at each resolution
preset (320/480/640/960/1280), showing latency/FPS and a rough detection-count stability
proxy per resolution. This is meant to help you pick a resolution/FPS setting quickly on
your actual footage — it is not a substitute for proper mAP evaluation on labeled data
(`model.val()` in Ultralytics, if you have a labeled validation set).
