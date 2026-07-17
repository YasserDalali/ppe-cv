"""Step 2 — name-based label remap onto the fused 10-class map.

Every source's labels are rewritten by CLASS NAME (never by raw id), with
hard-fail accounting: no image or box may disappear unaccounted for. This is
the fix for the old bug where SH17 ids >= 10 were silently dropped (~43% of
train images lost).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ppe.config import CANONICAL_CLASSES

DROP = "DROP"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class RemapError(RuntimeError):
    pass


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


# --- Explicit, human-readable per-source maps (keys are normalized names). ---
# Real class lists verified live against each source on 2026-07-17
# (scripts/check_sources.py); unknown names hard-fail at run time.

# SH17 (Kaggle original, 17 classes)
SH17_NAME_MAP = {
    "person": "Person",
    "helmet": "helmet",
    "safety_vest": "vest",
    "gloves": "gloves",
    "shoes": "boots",
    "glasses": "goggles",
    # body parts / non-PPE -> dropped; their images stay as background
    "head": DROP, "face": DROP, "ear": DROP, "earmuffs": DROP,
    "face_mask_medical": DROP, "face_guard": DROP, "hands": DROP,
    "foot": DROP, "tools": DROP, "medical_suit": DROP, "safety_suit": DROP,
}

# datasetppe/ppe_detection-dnfen v3: gloves, helmet, no-gloves, no-helmet,
# no-shoes, no-vest, person, shoes, vest
PPE_NAME_MAP = {
    "person": "Person",
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "shoes": "boots",
    "no_helmet": "no_helmet",
    "no_gloves": "no_gloves",
    # no canonical slot for these negatives -> dropped
    "no_shoes": DROP,
    "no_vest": DROP,
}

# daniil-yarmov/gas-masks v1: classes ['0', 'gas mask'] — '0' is a junk class
GASMASK_NAME_MAP = {
    "gas_mask": "gas mask",
    "0": DROP,
}

# yasser-dalali/ocp-snzjw v1: Person, boots, gas mask, gloves, goggles, helmet,
# vest (no negative classes — those ids simply get zero boxes from OCP)
OCP_NAME_MAP = {
    "person": "Person",
    "helmet": "helmet",
    "vest": "vest",
    "gloves": "gloves",
    "boots": "boots",
    "goggles": "goggles",
    "gas_mask": "gas mask",
    "no_helmet": "no_helmet",
    "no_gloves": "no_gloves",
    "no_goggle": "no_goggle",
}

NAME_MAPS = {
    "roboflow_ppe": PPE_NAME_MAP,
    "sh17": SH17_NAME_MAP,
    "gasmask": GASMASK_NAME_MAP,
    "ocp": OCP_NAME_MAP,
}


@dataclass
class RemapReport:
    source: str
    images_before: int = 0
    images_after: int = 0
    boxes_before: int = 0
    boxes_after: int = 0
    boxes_dropped: int = 0
    dropped_by_class: dict = field(default_factory=dict)
    boxes_by_class: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def read_class_names(source_dir: Path) -> list[str]:
    """Class names from the source's own metadata (data.yaml names)."""
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        p = Path(source_dir) / candidate
        if p.is_file():
            names = yaml.safe_load(p.read_text()).get("names")
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=int)]
            if isinstance(names, list):
                return [str(n) for n in names]
            raise RemapError(f"{p}: 'names' is neither list nor dict")
    raise RemapError(f"No data.yaml with class names found under {source_dir}")


def _find_images(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(source_dir).rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.parent.name == "images"
    )


def remap_source(source_dir: Path, out_dir: Path, name_map: dict[str, str],
                 source_name: str) -> RemapReport:
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    names = read_class_names(source_dir)

    unmapped = [n for n in names if normalize_name(n) not in name_map]
    if unmapped:
        raise RemapError(
            f"{source_name}: class names with NO entry in the remap table: {unmapped}. "
            "Refusing to continue — this is exactly how the old silent-drop bug "
            "started. Add each name to the map (target class or DROP)."
        )

    images = _find_images(source_dir)
    if not images:
        raise RemapError(f"{source_name}: no images found under {source_dir}")

    rep = RemapReport(source=source_name, images_before=len(images))
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    for img in images:
        if (out_dir / "images" / img.name).exists():
            raise RemapError(
                f"{source_name}: duplicate image filename {img.name!r} across splits"
            )
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        out_lines: list[str] = []
        if label.is_file():
            for raw in label.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                cls_id = int(parts[0])
                rep.boxes_before += 1
                if not 0 <= cls_id < len(names):
                    raise RemapError(
                        f"{source_name}: label id {cls_id} out of range "
                        f"(source has {len(names)} classes) in {label.name}"
                    )
                orig = names[cls_id]
                target = name_map[normalize_name(orig)]
                if target == DROP:
                    rep.boxes_dropped += 1
                    rep.dropped_by_class[orig] = rep.dropped_by_class.get(orig, 0) + 1
                    continue
                new_id = CANONICAL_CLASSES.index(target)
                out_lines.append(" ".join([str(new_id)] + parts[1:]))
                rep.boxes_after += 1
                rep.boxes_by_class[target] = rep.boxes_by_class.get(target, 0) + 1
        shutil.copy2(img, out_dir / "images" / img.name)
        (out_dir / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )
        rep.images_after += 1

    if rep.images_after != rep.images_before:
        raise RemapError(
            f"{source_name}: image count changed {rep.images_before} -> "
            f"{rep.images_after}; unaccounted loss"
        )
    if rep.boxes_before != rep.boxes_after + rep.boxes_dropped:
        raise RemapError(
            f"{source_name}: box accounting broken: {rep.boxes_before} != "
            f"{rep.boxes_after} kept + {rep.boxes_dropped} dropped"
        )
    return rep


def remap_all(cfg, raw_dirs: dict[str, Path], out_root: Path) -> dict[str, RemapReport]:
    """Remap every source; write out_root/remap.json (map + drop accounting)."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, RemapReport] = {}
    maps_used: dict[str, dict[str, str]] = {}
    for source, src_dir in raw_dirs.items():
        if source not in NAME_MAPS:
            raise RemapError(f"No NAME_MAP defined for source {source!r}")
        name_map = NAME_MAPS[source]
        reports[source] = remap_source(src_dir, out_root / source, name_map, source)
        maps_used[source] = {
            normalize_name(n): name_map[normalize_name(n)]
            for n in read_class_names(Path(src_dir))
        }
    (out_root / "remap.json").write_text(json.dumps({
        "classes": CANONICAL_CLASSES,
        "maps": maps_used,
        "reports": {s: r.as_dict() for s, r in reports.items()},
    }, indent=2))
    return reports
