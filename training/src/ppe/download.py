"""Step 1 — fetch & cache all 4 dataset sources under <drive>/raw/.

Idempotent: each source directory carries a ".complete" marker written only
after download + validation succeed; present marker means skip. Heavy
packages (roboflow, kaggle) are lazy-imported so tests never need them.

Downloads stage on local disk (`cfg.work_root`) first, then copy to Drive.
Writing zip bytes straight onto Google Drive FUSE from Colab often yields
BadZipFile ("File is not a zip file") even when the Roboflow export is fine.
"""
from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ppe.config import Config
from ppe.remap import RemapError, _find_images, read_class_names

SOURCES = ["roboflow_ppe", "sh17", "gasmask", "ocp"]

MARKER = ".complete"


class DownloadError(RuntimeError):
    pass


@dataclass
class SourceStats:
    name: str
    images: int = 0
    boxes: int = 0
    class_names: list = field(default_factory=list)
    per_class_boxes: dict = field(default_factory=dict)


def _scratch_path(cfg: Config, name: str) -> Path:
    """Local staging path that must NOT exist yet.

    Roboflow's ``Version.download(location=...)`` short-circuits when the
    path already exists (default ``overwrite=False``), returning an empty
    Dataset. ``tempfile.mkdtemp`` would trigger that trap.
    """
    parent = cfg.work_root / "_dl_scratch"
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{name}-{uuid.uuid4().hex}"


def _dataset_root(scratch: Path) -> Path:
    """Roboflow usually unpacks into ``location/``; occasionally one nested dir."""
    if (scratch / "data.yaml").is_file() or (scratch / "data.yml").is_file():
        return scratch
    yaml_hits = sorted(
        p for p in scratch.rglob("data.y*ml")
        if p.is_file() and "names" in (yaml.safe_load(p.read_text()) or {})
    )
    if yaml_hits:
        return yaml_hits[0].parent
    # images/labels layout without yaml yet (Kaggle SH17)
    if any(scratch.rglob("images")) or any(scratch.rglob("classes.txt")):
        return scratch
    raise DownloadError(
        f"download staging at {scratch} has no data.yaml / images — contents: "
        f"{sorted(p.name for p in scratch.iterdir()) if scratch.is_dir() else scratch}"
    )


def _promote_scratch(scratch: Path, dest: Path) -> None:
    """Copy a finished local download onto dest (usually Drive), replacing any partial."""
    for z in scratch.rglob("*.zip"):
        z.unlink(missing_ok=True)
    root = _dataset_root(scratch)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, dest)


def _download_roboflow(cfg: Config, api_key: str, workspace: str, project: str,
                       version: int, dest: Path) -> None:
    from roboflow import Roboflow  # lazy: runtime-only dependency

    dest = Path(dest)
    last_exc: Exception | None = None
    # One retry: Roboflow/CDN occasionally serves a truncated body.
    for attempt in range(2):
        scratch = _scratch_path(cfg, dest.name)
        try:
            rf = Roboflow(api_key=api_key)
            # overwrite=True is required if scratch somehow pre-exists; the path
            # is chosen not to exist, but be explicit about Roboflow's skip logic.
            rf.workspace(workspace).project(project).version(version).download(
                "yolov8", location=str(scratch), overwrite=True)
            _promote_scratch(scratch, dest)
            return
        except (zipfile.BadZipFile, OSError, DownloadError) as exc:
            last_exc = exc
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            if attempt == 0:
                print(f"[download] {dest.name}: {type(exc).__name__}: {exc}; retrying once…")
                continue
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    assert last_exc is not None
    raise last_exc


def _download_kaggle(cfg: Config, secrets: dict, dest: Path) -> None:
    os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]
    import kaggle  # lazy: authenticates on import using the env vars above

    dest = Path(dest)
    scratch = _scratch_path(cfg, dest.name)
    scratch.mkdir(parents=True, exist_ok=True)  # kaggle writes into an existing dir
    try:
        kaggle.api.dataset_download_files(
            cfg.sh17_kaggle_slug, path=str(scratch), unzip=True)
        _promote_scratch(scratch, dest)
    except Exception:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _ensure_data_yaml(source_dir: Path, name: str) -> None:
    """SH17 from Kaggle may ship classes.txt (or a differently named yaml)
    instead of data.yaml — synthesize one so every downstream step is uniform."""
    try:
        read_class_names(source_dir)
        return
    except RemapError:
        pass
    for yml in sorted(source_dir.rglob("*.y*ml")):
        names = yaml.safe_load(yml.read_text()).get("names")
        if names:
            if isinstance(names, dict):
                names = [str(names[k]) for k in sorted(names, key=int)]
            (source_dir / "data.yaml").write_text(
                yaml.safe_dump({"nc": len(names), "names": [str(n) for n in names]}))
            return
    classes_txt = next(iter(sorted(source_dir.rglob("classes.txt"))), None)
    if classes_txt:
        names = [line.strip() for line in classes_txt.read_text().splitlines() if line.strip()]
        (source_dir / "data.yaml").write_text(
            yaml.safe_dump({"nc": len(names), "names": names}))
        return
    raise DownloadError(
        f"{name}: no data.yaml / *.yaml with names / classes.txt found under "
        f"{source_dir} — cannot determine class names"
    )


def validate_source(source_dir: Path, name: str) -> SourceStats:
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise DownloadError(f"{name}: expected dataset at {source_dir}, found nothing")
    _ensure_data_yaml(source_dir, name)
    class_names = read_class_names(source_dir)
    images = _find_images(source_dir)
    if not images:
        raise DownloadError(
            f"{name}: no images under {source_dir} — download looks broken; "
            f"delete the folder and re-run to retry"
        )
    stats = SourceStats(name=name, class_names=class_names, images=len(images))
    for img in images:
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not label.is_file():
            continue  # unlabeled image = background, valid YOLO
        for line in label.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if not 0 <= cls_id < len(class_names):
                raise DownloadError(
                    f"{name}: label id {cls_id} out of range (source has "
                    f"{len(class_names)} classes) in {label.name}"
                )
            cls = class_names[cls_id]
            stats.boxes += 1
            stats.per_class_boxes[cls] = stats.per_class_boxes.get(cls, 0) + 1
    return stats


def _roboflow_coords(cfg: Config, name: str) -> tuple[str, str, int]:
    return {
        "roboflow_ppe": (cfg.ppe_workspace, cfg.ppe_project, cfg.ppe_version),
        "gasmask": (cfg.gasmask_workspace, cfg.gasmask_project, cfg.gasmask_version),
        "ocp": (cfg.ocp_workspace, cfg.ocp_project, cfg.ocp_version),
    }[name]


def ensure_source(cfg: Config, secrets: dict, name: str) -> Path:
    if name not in SOURCES:
        raise DownloadError(f"unknown source {name!r}; expected one of {SOURCES}")
    dest = cfg.raw_dir / name
    if (dest / MARKER).is_file():
        return dest
    try:
        if name == "sh17":
            _download_kaggle(cfg, secrets, dest)
        else:
            ws, proj, ver = _roboflow_coords(cfg, name)
            _download_roboflow(cfg, secrets["ROBOFLOW_API_KEY"], ws, proj, ver, dest)
        stats = validate_source(dest, name)
    except DownloadError:
        if dest.exists() and not (dest / MARKER).is_file():
            shutil.rmtree(dest, ignore_errors=True)
        raise
    except Exception as exc:
        if dest.exists() and not (dest / MARKER).is_file():
            shutil.rmtree(dest, ignore_errors=True)
        if name == "ocp":
            raise DownloadError(
                f"OCP dataset download failed ({type(exc).__name__}: {exc}). "
                f"Attempted Roboflow {cfg.ocp_workspace}/{cfg.ocp_project}/{cfg.ocp_version}. "
                "OCP images are part of the fusion and the headline test set — "
                "there is no OCP-less mode. If the version does not exist yet, "
                "generate it in the Roboflow UI (project OCP -> Generate -> "
                "Create version 1, yolov8 export) and re-run."
            ) from exc
        hint = ""
        if isinstance(exc, zipfile.BadZipFile) or "BadZipFile" in type(exc).__name__:
            hint = (
                " Hint: zip was corrupt (common when extracting on Google Drive). "
                f"Delete Drive folder raw/{name}/ if it remains, re-upload this "
                "training/ code, and re-run — downloads now stage on local disk first."
            )
        raise DownloadError(
            f"{name}: download failed ({type(exc).__name__}: {exc}).{hint}"
        ) from exc
    if name == "gasmask" and stats.images != 384:
        print(f"[download] note: gas-masks v1 expected 384 images, found {stats.images}")
    print(f"[download] {name}: images={stats.images} boxes={stats.boxes} "
          f"per-class={stats.per_class_boxes}")
    (dest / MARKER).touch()
    return dest


def download_all(cfg: Config, secrets: dict) -> dict[str, Path]:
    return {name: ensure_source(cfg, secrets, name) for name in SOURCES}
