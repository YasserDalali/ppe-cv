"""
One-off setup script: exports the trained weights (paper/results/best.pt) to
ONNX and drops a copy into each of the app's two model slots (local + server).

Run from anywhere:
    python scripts/export_onnx.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_WEIGHTS = REPO_ROOT / "paper" / "results" / "best.pt"
APP_ROOT = Path(__file__).resolve().parents[1]
DEST_PATHS = [
    APP_ROOT / "models" / "best.onnx",
    APP_ROOT / "server" / "models" / "best.onnx",
]


def main():
    model = YOLO(str(SOURCE_WEIGHTS))
    exported_path = Path(model.export(format="onnx", dynamic=True, simplify=True, half=False))

    for dest in DEST_PATHS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(exported_path, dest)
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
