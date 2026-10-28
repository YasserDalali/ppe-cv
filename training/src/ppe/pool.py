"""2026-10 revision — build the data pool once, assemble one dataset per run.

The pool (``prepared/pool.zip`` on Drive) holds every remapped source plus
the split decisions (``splits.json``), per-class counts (``counts.json``)
and a content hash per image. ``assemble_run`` then builds one run's dataset
on local disk: train (OCP x3 copies made after the split, train only), the
shared public val split, and the eval sets.

Leakage rules (reviewers R2-3, R4-M1, R4-2):
- OCP frames are split by recording (``VID_*`` / ``IMG_*``), never by frame.
- Roboflow sources are split by original picture (``<orig>_jpg.rf.<hash>``),
  so augmented copies of one picture never land on both sides.
- A run stops if a recording, or a byte-identical OCP image, is on both
  sides. Byte-identical public images (duplicates inside a public dataset)
  are dropped from the eval side and counted in run_meta.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from tqdm import tqdm

from ppe.config import (CANONICAL_CLASSES, EXPERIMENTS, PUBLIC_SOURCES, Config,
                        parse_run_id)
from ppe.merge import (_copy_pair, _safe_stem, _split_indices,
                       _unzip_with_progress, _write_eval_set, zip_dir)

POOL_ZIP = "pool.zip"
EVAL_SETS = ["ocp", "public_test", "construction_test", "industrial_proxy"]


class PoolError(RuntimeError):
    pass


_RECORDING_RE = re.compile(r"(VID_\d+_\d+|IMG_\d+)")


def recording_id(image_name: str) -> str:
    """'ocp__VID_20260716_163549_f00211_jpg.rf.x.jpg' -> 'VID_20260716_163549';
    'IMG_9845_MOV_f00012...' -> 'IMG_9845'. Anything else is its own group."""
    m = _RECORDING_RE.search(image_name)
    return m.group(1) if m else Path(image_name).stem


def group_key(image_name: str) -> str:
    """Roboflow names copies of one picture '<orig>_jpg.rf.<hash>.jpg'."""
    return image_name.split(".rf.")[0] if ".rf." in image_name else Path(image_name).stem


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- splitting

def split_public(cfg: Config, source: str, names: list[str]) -> dict[str, list[str]]:
    """Seeded 70/15/15 split over picture groups (see group_key). For SH17 the
    train slice is capped at sh17_train_cap; the next sh17_extra_pool train
    images go to 'extra' (used only by the size-matched model S)."""
    groups: dict[str, list[str]] = {}
    for n in sorted(names):
        groups.setdefault(group_key(n), []).append(n)
    keys = sorted(groups)
    wide = cfg
    if source == "sh17" and cfg.sh17_train_cap is not None:
        wide = replace(cfg, sh17_train_cap=cfg.sh17_train_cap + cfg.sh17_extra_pool)
    idx = _split_indices(wide, len(keys), source)
    out = {split: [n for i in idx[split] for n in groups[keys[i]]]
           for split in ("train", "val", "test")}
    out["extra"] = []
    if source == "sh17" and cfg.sh17_train_cap is not None:
        train_keys = [keys[i] for i in idx["train"]]
        out["train"] = [n for k in train_keys[:cfg.sh17_train_cap] for n in groups[k]]
        out["extra"] = [n for k in train_keys[cfg.sh17_train_cap:] for n in groups[k]]
    return out


def make_ocp_folds(names: list[str], n_folds: int) -> dict[int, list[str]]:
    """Recordings -> folds, largest first onto the lightest fold. Deterministic
    (no randomness: sorted by frame count, then name)."""
    counts = Counter(recording_id(n) for n in names)
    if len(counts) < n_folds:
        raise PoolError(f"only {len(counts)} OCP recordings for {n_folds} folds")
    folds: dict[int, list[str]] = {f: [] for f in range(n_folds)}
    load = [0] * n_folds
    for rec, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        f = min(range(n_folds), key=lambda i: (load[i], i))
        folds[f].append(rec)
        load[f] += c
    return {f: sorted(recs) for f, recs in folds.items()}


def ocp_train_frames(doc: dict, fold: int) -> list[str]:
    test = set(doc["ocp"]["folds"][str(fold)])
    return [n for n in doc["ocp"]["frames"] if recording_id(n) not in test]


def size_match_extra(cfg: Config, doc: dict) -> int:
    """Extra SH17 images that give S the same train size as A (fold mean)."""
    per_fold = [len(ocp_train_frames(doc, int(f))) for f in doc["ocp"]["folds"]]
    return round(statistics.fmean(per_fold) * cfg.ocp_dup_factor)


# ---------------------------------------------------------------- pool build

def ensure_ocp_originals(cfg: Config) -> Path:
    """Unzip the uploaded un-augmented OCP export to local disk and check it
    really is the 207 originals (not the leaky augmented v2)."""
    from ppe.download import validate_source

    src = cfg.ocp_upload_zip
    if not src.is_file():
        raise PoolError(
            f"missing {src}. Upload the un-augmented OCP export "
            f"({cfg.ocp_upload_name}, {cfg.ocp_expected_images} images) there first.")
    dest = cfg.work_root / "ocp_originals"
    if dest.exists():
        shutil.rmtree(dest)
    with zipfile.ZipFile(src) as zf:
        readme = next((n for n in zf.namelist() if n.endswith("README.roboflow.txt")), None)
        text = zf.read(readme).decode(errors="replace") if readme else ""
        if "No pre-processing or augmentation" not in text:
            raise PoolError(
                f"{src.name}: README does not say 'No pre-processing or augmentation'. "
                "Export the OCP version without augmentation — augmented copies "
                "leak across splits.")
        zf.extractall(dest)
    stats = validate_source(dest, "ocp")
    if stats.images != cfg.ocp_expected_images:
        raise PoolError(f"{src.name}: expected {cfg.ocp_expected_images} images, "
                        f"found {stats.images}")
    return dest


def _label_classes(label: Path) -> list[str]:
    if not label.is_file():
        return []
    return [CANONICAL_CLASSES[int(line.split()[0])]
            for line in label.read_text().splitlines() if line.strip()]


def _count(sources_dir: Path, source: str, names: list[str]) -> dict:
    boxes: Counter = Counter()
    images: Counter = Counter()
    for n in names:
        classes = _label_classes(sources_dir / source / "labels" / f"{Path(n).stem}.txt")
        boxes.update(classes)
        images.update(set(classes))
    return {"images": len(names), "boxes": dict(sorted(boxes.items())),
            "images_with_class": dict(sorted(images.items()))}


def pool_counts(pool_dir: Path, doc: dict) -> dict:
    """Images and boxes per source x split x class, per OCP recording and per
    OCP fold (R2-2, R2-7, R3-1, R4-1)."""
    src = Path(pool_dir) / "sources"
    per_source = {s: {split: _count(src, s, names) for split, names in sp.items()}
                  for s, sp in doc["public"].items()}
    frames = doc["ocp"]["frames"]
    by_rec: dict[str, list[str]] = {}
    for n in frames:
        by_rec.setdefault(recording_id(n), []).append(n)
    per_source["ocp"] = {"all": _count(src, "ocp", frames)}
    recordings = {rec: _count(src, "ocp", ns) for rec, ns in sorted(by_rec.items())}
    folds = {}
    for f, recs in doc["ocp"]["folds"].items():
        ns = [n for r in recs for n in by_rec[r]]
        folds[f] = {"recordings": recs, **_count(src, "ocp", ns)}
    return {"classes": CANONICAL_CLASSES, "per_source": per_source,
            "ocp_recordings": recordings, "ocp_folds": folds}


def build_pool_dir(cfg: Config, raw: dict[str, Path], out_dir: Path) -> Path:
    """Remap + split + hash + count into out_dir (sources/, splits.json,
    counts.json). raw: source name -> raw dir, PUBLIC_SOURCES + 'ocp'."""
    from ppe.remap import _find_images, remap_all

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    public = {s: split_public(cfg, s, [p.name for p in _find_images(raw[s])])
              for s in PUBLIC_SOURCES}
    ocp_names = sorted(p.name for p in _find_images(raw["ocp"]))
    folds = make_ocp_folds(ocp_names, cfg.n_folds)
    keep = {s: {n for names in sp.values() for n in names} for s, sp in public.items()}
    remap_all(cfg, {s: raw[s] for s in [*PUBLIC_SOURCES, "ocp"]}, out_dir / "sources", keep=keep)

    hashes = {}
    for source in [*PUBLIC_SOURCES, "ocp"]:
        for img in tqdm(sorted((out_dir / "sources" / source / "images").iterdir()),
                        desc=f"[pool] hashing {source}", unit="img"):
            hashes[f"{source}/{img.name}"] = file_sha1(img)
    doc = {
        "seed": cfg.seed,
        "fractions": [cfg.train_frac, cfg.val_frac, cfg.test_frac],
        "sh17_caps": [cfg.sh17_train_cap, cfg.sh17_val_cap, cfg.sh17_test_cap],
        "public": public,
        "ocp": {"frames": ocp_names, "folds": {str(f): r for f, r in folds.items()}},
        "hashes": hashes,
    }
    (out_dir / "splits.json").write_text(json.dumps(doc, indent=1))
    (out_dir / "counts.json").write_text(json.dumps(pool_counts(out_dir, doc), indent=2))
    return out_dir


def ensure_pool(cfg: Config, secrets: dict | None = None) -> Path:
    """Build prepared/pool.zip once (needs secrets only then), and restore it
    to local disk. Returns the local pool dir."""
    zip_path = cfg.prepared_dir / POOL_ZIP
    if zip_path.is_file() and not zipfile.is_zipfile(zip_path):
        print(f"[pool] {zip_path.name}: partial write from an interrupted run — rebuilding")
        zip_path.unlink()
    if not zip_path.is_file():
        if secrets is None:
            raise PoolError(f"no {zip_path} yet — run the job0 notebook first")
        from ppe.download import ensure_source

        print("[pool] building the pool (public sources from Drive raw/, OCP originals)…")
        raw = {s: ensure_source(cfg, secrets, s) for s in PUBLIC_SOURCES}
        raw["ocp"] = ensure_ocp_originals(cfg)
        build = build_pool_dir(cfg, raw, cfg.work_root / "pool_build")
        zip_dir(build, zip_path)
        shutil.rmtree(build)
    local = cfg.local_pool_dir
    if not (local / "splits.json").is_file():
        if local.exists():
            shutil.rmtree(local)
        _unzip_with_progress(zip_path, local)
    return local


# ---------------------------------------------------------------- per run

@dataclass
class RunData:
    run_id: str
    dataset_dir: Path
    data_yaml: Path
    eval_sets: dict
    meta: dict


def _entries(pool_dir: Path, pairs: list[tuple[str, str]]) -> list[tuple[Path, Path, str]]:
    src = Path(pool_dir) / "sources"
    return [(src / s / "images" / n, src / s / "labels" / f"{Path(n).stem}.txt",
             _safe_stem(s, Path(n).stem)) for s, n in pairs]


def resize_like_ultralytics(src: Path, dst: Path, imgsz: int) -> Path:
    """Write ``src`` exactly as Ultralytics' BaseDataset.load_image() hands it
    to the model (longest side -> imgsz, cv2.INTER_LINEAR, same rounding), as
    lossless PNG. Loading it back gives r == 1, so training sees bit-identical
    pixels without decoding the full-size original every epoch — the .npy disk
    cache stores originals at full size (~36 MB per SH17 photo)."""
    import cv2
    from ultralytics.utils.patches import imread

    im = imread(str(src))
    if im is None:
        raise PoolError(f"cannot read image {src}")
    h0, w0 = im.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:
        w, h = min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial.png")
    if not cv2.imwrite(str(tmp), im):
        raise PoolError(f"cannot write {tmp}")
    tmp.replace(dst)
    return dst


def _resized(pool_dir: Path, entries: list, imgsz: int) -> list:
    """Same (img, label, stem) entries with img swapped for its resized PNG,
    made once per pool image under pool/resized<imgsz>/ and reused by later runs."""
    from concurrent.futures import ThreadPoolExecutor

    def target(img: Path) -> Path:
        return Path(pool_dir) / f"resized{imgsz}" / img.parent.parent.name / f"{img.stem}.png"

    todo = {img: target(img) for img, _, _ in entries if not target(img).is_file()}
    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:  # cv2 releases the GIL
            list(tqdm(ex.map(lambda kv: resize_like_ultralytics(kv[0], kv[1], imgsz), todo.items()),
                      total=len(todo), desc=f"[assemble] resizing to {imgsz}", unit="img"))
    return [(target(img), label, stem) for img, label, stem in entries]


def check_and_clean(train: list[tuple[str, str]], others: dict[str, list[tuple[str, str]]],
                    hashes: dict[str, str]) -> tuple[dict, dict]:
    """Raise on any OCP recording or OCP image shared by train and another
    set; drop byte-identical public duplicates from the other sets."""
    train_recs = {recording_id(n) for s, n in train if s == "ocp"}
    train_hashes = {hashes[f"{s}/{n}"] for s, n in train}
    cleaned, dropped = {}, {}
    for name, pairs in others.items():
        shared = train_recs & {recording_id(n) for s, n in pairs if s == "ocp"}
        if shared:
            raise PoolError(f"leak: OCP recordings {sorted(shared)} in train and {name}")
        dup = [(s, n) for s, n in pairs if hashes[f"{s}/{n}"] in train_hashes]
        if any(s == "ocp" for s, _ in dup):
            raise PoolError(f"leak: OCP image identical to a train image in {name}: "
                            f"{[n for s, n in dup if s == 'ocp'][:3]}")
        dup_set = set(dup)
        cleaned[name] = [p for p in pairs if p not in dup_set]
        dropped[name] = len(dup)
    return cleaned, dropped


def run_plan(cfg: Config, run_id: str, doc: dict) -> dict:
    """Which (source, image) pairs go where for one run — no file I/O."""
    exp, seed, fold = parse_run_id(run_id)
    spec = EXPERIMENTS[exp]
    train = [(s, n) for s in spec["sources"] for n in doc["public"][s]["train"]]
    if spec["sh17_extra"]:
        n_extra = size_match_extra(cfg, doc)
        extra = doc["public"]["sh17"]["extra"]
        if n_extra > len(extra):
            raise PoolError(f"{run_id}: needs {n_extra} extra SH17 images, pool has "
                            f"{len(extra)} — raise sh17_extra_pool and rebuild the pool")
        train += [("sh17", n) for n in extra[:n_extra]]
    ocp_train = [("ocp", n) for n in ocp_train_frames(doc, fold)] if spec["ocp"] else []
    test_recs = set(doc["ocp"]["folds"][str(fold)]) if fold is not None else None
    ocp_eval = [("ocp", n) for n in doc["ocp"]["frames"]
                if test_recs is None or recording_id(n) in test_recs]
    pub = doc["public"]
    return {
        "experiment": exp, "seed": seed, "fold": fold,
        "train": train, "ocp_train": ocp_train,
        "n_train": len(train) + len(ocp_train) * cfg.ocp_dup_factor,
        "val": [(s, n) for s in PUBLIC_SOURCES for n in pub[s]["val"]],
        "eval": {
            "ocp": ocp_eval,
            "public_test": [(s, n) for s in PUBLIC_SOURCES for n in pub[s]["test"]],
            "construction_test": [("roboflow_ppe", n) for n in pub["roboflow_ppe"]["test"]],
            "industrial_proxy": [(s, n) for s in ("sh17", "gasmask") for n in pub[s]["test"]],
        },
        "test_recordings": sorted(test_recs) if test_recs is not None else None,
    }


def assemble_run(cfg: Config, run_id: str, pool_dir: Path | None = None) -> RunData:
    pool_dir = Path(pool_dir or cfg.local_pool_dir)
    doc = json.loads((pool_dir / "splits.json").read_text())
    plan = run_plan(cfg, run_id, doc)
    exp = plan["experiment"]
    if exp == "S":
        a_sizes = [run_plan(cfg, f"A_f{f}", doc)["n_train"] for f in EXPERIMENTS["A"]["folds"]]
        target = statistics.fmean(a_sizes)
        if abs(plan["n_train"] - target) > 0.01 * target:
            raise PoolError(f"{run_id}: train size {plan['n_train']} is not within 1% "
                            f"of A's mean {target:.0f}")

    train_all = plan["train"] + plan["ocp_train"]
    cleaned, dropped = check_and_clean(train_all, {"val": plan["val"], **plan["eval"]},
                                       doc["hashes"])

    out = cfg.work_root / "runs" / run_id
    if out.exists():
        shutil.rmtree(out)
    dataset = out / "dataset"
    train_dir = dataset / "train"
    train_e = _entries(pool_dir, plan["train"])
    ocp_e = _entries(pool_dir, plan["ocp_train"])
    val_e = _entries(pool_dir, cleaned["val"])
    if cfg.train_resize:  # eval sets keep the originals
        train_e, ocp_e, val_e = (_resized(pool_dir, e, cfg.imgsz) for e in (train_e, ocp_e, val_e))
    for img, label, stem in tqdm(train_e, desc=f"[assemble] {run_id} train", unit="img"):
        _copy_pair(img, label, train_dir, stem)
    for img, label, stem in ocp_e:
        _copy_pair(img, label, train_dir, stem)
        for k in range(1, cfg.ocp_dup_factor):  # x3 weight: after the split, train only
            _copy_pair(img, label, train_dir, f"{stem}__dup{k}")
    for img, label, stem in val_e:
        _copy_pair(img, label, dataset / "val", stem)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset), "train": "train/images", "val": "val/images",
        "nc": len(CANONICAL_CLASSES), "names": CANONICAL_CLASSES}, sort_keys=False))

    eval_sets = {}
    for name in EVAL_SETS:
        d = out / "eval" / name
        _write_eval_set(_entries(pool_dir, cleaned[name]), d)
        eval_sets[name] = d

    n_train_files = len(list((train_dir / "images").iterdir()))
    if n_train_files != plan["n_train"]:
        raise PoolError(f"{run_id}: wrote {n_train_files} train images, planned {plan['n_train']}")
    train_classes: Counter = Counter()
    for lf in (train_dir / "labels").iterdir():
        train_classes.update(_label_classes(lf))
    spec = EXPERIMENTS[exp]
    meta = {
        "run_id": run_id, "experiment": exp, "label": spec["label"],
        "seed": plan["seed"], "fold": plan["fold"],
        "train_sources": spec["sources"] + (["ocp"] if spec["ocp"] else []),
        "n_train_images": n_train_files,
        "n_train_unique": len(train_all),
        "n_sh17_extra": len(set(n for s, n in plan["train"] if s == "sh17")
                            & set(doc["public"]["sh17"]["extra"])),
        "n_ocp_train_frames": len(plan["ocp_train"]),
        "ocp_dup_factor": cfg.ocp_dup_factor,
        "n_val_images": len(cleaned["val"]),
        "train_class_boxes": dict(sorted(train_classes.items())),
        "train_classes": [c for c in CANONICAL_CLASSES if train_classes.get(c)],
        "ocp_test_recordings": plan["test_recordings"],
        "eval_sets": {name: {"images": len(cleaned[name]),
                             "recordings": len({recording_id(n) for _, n in cleaned[name]})}
                      for name in EVAL_SETS},
        "dropped_identical_to_train": dropped,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[assemble] {run_id}: train={n_train_files} val={len(cleaned['val'])} "
          + " ".join(f"{k}={v['images']}" for k, v in meta["eval_sets"].items()))
    return RunData(run_id, dataset, data_yaml, eval_sets, meta)


def ocp_all_eval_set(cfg: Config, pool_dir: Path, dest: Path) -> Path:
    """All 207 OCP frames as one eval set (ground truth for pooled scoring)."""
    doc = json.loads((Path(pool_dir) / "splits.json").read_text())
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    _write_eval_set(_entries(pool_dir, [("ocp", n) for n in doc["ocp"]["frames"]]), dest)
    return dest
