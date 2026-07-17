from pathlib import Path

import pytest
import yaml
from PIL import Image

# SH17's 17 real class names (Kaggle original, mugheesahmad).
SH17_CLASSES = [
    "person", "head", "face", "glasses", "face_mask_medical", "face_guard",
    "ear", "earmuffs", "hands", "gloves", "foot", "shoes", "safety_vest",
    "tools", "helmet", "medical_suit", "safety_suit",
]

PPE_FIXTURE_CLASSES = ["person", "helmet", "no-helmet", "vest", "no-vest"]
GASMASK_FIXTURE_CLASSES = ["0", "gas mask"]  # mirrors live v1: junk class '0' + real one
OCP_FIXTURE_CLASSES = ["Person", "helmet", "vest", "gloves", "boots", "goggles", "gas mask"]


def write_png(path: Path, w: int = 32, h: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (127, 127, 127)).save(path)


def _write_pair(img_dir: Path, lbl_dir: Path, stem: str, boxes) -> None:
    write_png(img_dir / f"{stem}.png")
    lbl_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes]
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def make_yolo_source(root: Path, name: str, class_names: list[str], images,
                     layout: str = "roboflow") -> Path:
    """Build a fake dataset. images: list of box-lists (one image per entry).

    layout "flat": SH17-style images/ + labels/ + data.yaml.
    layout "roboflow": train|valid|test/{images,labels} + data.yaml (last two
    images go to valid and test).
    """
    src = root / name
    if layout == "flat":
        for i, boxes in enumerate(images):
            _write_pair(src / "images", src / "labels", f"{name}_{i}", boxes)
    elif layout == "roboflow":
        n = len(images)
        cuts = {
            "train": range(0, max(n - 2, 1)),
            "valid": range(max(n - 2, 1), max(n - 1, 1)),
            "test": range(max(n - 1, 1), n),
        }
        for split, idxs in cuts.items():
            for i in idxs:
                _write_pair(src / split / "images", src / split / "labels",
                            f"{name}_{i}", images[i])
    else:
        raise ValueError(f"unknown layout {layout!r}")
    (src / "data.yaml").write_text(yaml.safe_dump({"nc": len(class_names), "names": class_names}))
    return src


def boxes_for(class_names: list[str], n_images: int = 10, per_image: int = 2):
    return [
        [(i % len(class_names), 0.5, 0.5, 0.3, 0.3) for i in range(per_image)]
        for _ in range(n_images)
    ]


@pytest.fixture
def fixture_sources(tmp_path):
    """All 4 fake sources with real-ish structures and class names."""
    root = tmp_path / "raw"
    return {
        "roboflow_ppe": make_yolo_source(
            root, "roboflow_ppe", PPE_FIXTURE_CLASSES, boxes_for(PPE_FIXTURE_CLASSES)),
        "sh17": make_yolo_source(
            root, "sh17", SH17_CLASSES,
            [[(i, 0.5, 0.5, 0.2, 0.2)] for i in range(17)], layout="flat"),
        "gasmask": make_yolo_source(
            root, "gasmask", GASMASK_FIXTURE_CLASSES,
            [[(1, 0.5, 0.5, 0.3, 0.3)] for _ in range(10)]),
        "ocp": make_yolo_source(
            root, "ocp", OCP_FIXTURE_CLASSES, boxes_for(OCP_FIXTURE_CLASSES)),
    }
