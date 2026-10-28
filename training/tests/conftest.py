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


# --- 2026-10 revision fixtures -------------------------------------------

OCP_RECORDINGS = {  # recording -> frames (mirrors the real VID_/IMG_ naming)
    "VID_20260716_163549": 4, "VID_20260716_163810": 3, "VID_20260716_163945": 3,
    "VID_20260716_164039": 2, "VID_20260716_164109": 2, "IMG_9845": 2,
}


def make_ocp_originals(root: Path) -> Path:
    """Roboflow-layout OCP export, frames named like the real ones."""
    src = root / "ocp"
    i = 0
    for rec, n in OCP_RECORDINGS.items():
        for k in range(n):
            split = ("train", "valid", "test")[i % 3]
            stem = f"{rec}_f{k:05d}_jpg.rf.h{i:04d}"
            _write_pair(src / split / "images", src / split / "labels", stem,
                        [(0, 0.5, 0.5, 0.3, 0.6), (1, 0.5, 0.2, 0.1, 0.1)])
            # distinct pixels so byte hashes differ
            Image.new("RGB", (32, 32), (i, 50, 100)).save(src / split / "images" / f"{stem}.png")
            i += 1
    (src / "data.yaml").write_text(yaml.safe_dump(
        {"nc": len(OCP_FIXTURE_CLASSES), "names": OCP_FIXTURE_CLASSES}))
    (src / "README.roboflow.txt").write_text(
        f"The dataset includes {i} images.\nNo pre-processing or augmentation was applied.\n")
    return src


def _distinct_png(path: Path, i: int) -> None:
    Image.new("RGB", (32, 32), (i % 256, (i // 256) % 256, 7)).save(path)


@pytest.fixture
def revision_raw(tmp_path):
    """Public sources + OCP originals, big enough for caps, folds and the
    size-matched SH17 extra pool (see revision_cfg)."""
    root = tmp_path / "raw"
    ppe = make_yolo_source(root, "roboflow_ppe", PPE_FIXTURE_CLASSES,
                           boxes_for(PPE_FIXTURE_CLASSES, n_images=20))
    sh17 = make_yolo_source(root, "sh17", SH17_CLASSES,
                            [[(i % 17, 0.5, 0.5, 0.2, 0.2), (14, 0.4, 0.3, 0.1, 0.1)]
                             for i in range(80)], layout="flat")
    gas = make_yolo_source(root, "gasmask", GASMASK_FIXTURE_CLASSES,
                           [[(1, 0.5, 0.5, 0.3, 0.3)] for _ in range(10)])
    for n, src in enumerate((ppe, sh17, gas)):
        for j, img in enumerate(sorted(src.rglob("*.png"))):
            _distinct_png(img, 1000 * (n + 1) + j)
    return {"roboflow_ppe": ppe, "sh17": sh17, "gasmask": gas,
            "ocp": make_ocp_originals(root)}


def revision_cfg(tmp_path, **kw):
    from ppe.config import Config

    base = dict(drive_root=tmp_path / "drive", work_root=tmp_path / "work",
                n_folds=5, sh17_train_cap=10, sh17_val_cap=5, sh17_test_cap=5,
                sh17_extra_pool=40, ocp_expected_images=sum(OCP_RECORDINGS.values()),
                n_boot=30)
    base.update(kw)
    return Config(**base)
