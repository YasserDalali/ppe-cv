"""Step 3 — merge remapped sources, deterministic split, duplication, holdouts.

- Filenames prefixed "{source}__{orig}" for provenance and collision safety.
- Seeded 70/15/15 split, stratified per source (each source split separately).
- OCP / gas-mask sampling weight implemented as physical file duplication with
  "__dupN" suffixes, train split ONLY (verifiable counts, no custom sampler).
- Holdout eval sets: ocp_test (disjointness from train asserted), industrial
  proxy (SH17 + gas-mask test images, no OCP), fused_val (the val split).
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from ppe.config import CANONICAL_CLASSES, Config


class MergeError(RuntimeError):
    pass


# Some Roboflow exports (e.g. gas-masks) concatenate every image tag into the
# filename, producing 200+ char stems. Prefixed with "{source}__" that can
# exceed the OS filename limit (255 bytes). Truncate + append a short hash of
# the full original stem so names stay safe, unique, and deterministic.
_MAX_STEM_LEN = 180


def _safe_stem(source: str, orig_stem: str) -> str:
    stem = f"{source}__{orig_stem}"
    if len(stem) <= _MAX_STEM_LEN:
        return stem
    prefix = f"{source}__"
    digest = hashlib.sha1(orig_stem.encode()).hexdigest()[:10]
    keep = _MAX_STEM_LEN - len(prefix) - len(digest) - 1
    return f"{prefix}{orig_stem[:keep]}_{digest}"


@dataclass
class SplitResult:
    dataset_dir: Path
    fused_val_dir: Path
    ocp_test_dir: Path
    proxy_dir: Path
    summary: dict


def write_data_yaml(dataset_dir: Path) -> Path:
    """(Re)write data.yaml with an absolute path — call again after unzipping
    the prepared dataset to a new location."""
    dataset_dir = Path(dataset_dir)
    out = dataset_dir / "data.yaml"
    out.write_text(yaml.safe_dump({
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }, sort_keys=False))
    return out


def zip_dir(src_dir: Path, zip_path: Path) -> Path:
    """Write to a temp path and rename into place, so a disconnect mid-write
    (this can take a long time onto Drive) never leaves a partial zip_path
    behind that a later run would mistake for a finished build.

    Writes file-by-file (rather than shutil.make_archive) so there's a real
    progress bar — this used to run silently for 30-45+ minutes onto Drive
    FUSE with zero feedback, which is indistinguishable from a hang."""
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = zip_path.with_name(zip_path.name + ".partial")
    tmp_path.unlink(missing_ok=True)
    files = [p for p in src_dir.rglob("*") if p.is_file()]
    print(f"[zip] {zip_path.name}: writing {len(files)} files "
          "(can be slow over Drive FUSE)…")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in tqdm(files, desc=f"[zip] {zip_path.name}", unit="file"):
            zf.write(p, str(p.relative_to(src_dir)))
    tmp_path.replace(zip_path)
    print(f"[zip] {zip_path.name}: done ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def _pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    source_dir = Path(source_dir)
    pairs = []
    for img in sorted((source_dir / "images").iterdir()):
        label = source_dir / "labels" / f"{img.stem}.txt"
        if not label.is_file():
            raise MergeError(f"missing label for {img} — remap step should have written it")
        pairs.append((img, label))
    return pairs


def _copy_pair(img: Path, label: Path, dest: Path, new_stem: str) -> None:
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dest / "images" / f"{new_stem}{img.suffix}")
    shutil.copy2(label, dest / "labels" / f"{new_stem}.txt")


def _write_eval_set(pairs: list[tuple[Path, Path, str]], dest: Path) -> None:
    for img, label, stem in pairs:
        _copy_pair(img, label, dest, stem)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data.yaml").write_text(yaml.safe_dump(
        {"nc": len(CANONICAL_CLASSES), "names": CANONICAL_CLASSES}, sort_keys=False))


def _count_boxes_by_class(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    labels_dir = split_dir / "labels"
    if not labels_dir.is_dir():
        return counts
    for lf in labels_dir.iterdir():
        for line in lf.read_text().splitlines():
            line = line.strip()
            if line:
                cls = CANONICAL_CLASSES[int(line.split()[0])]
                counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


def ensure_prepared(cfg: Config, secrets: dict) -> Path:
    """Steps 2+3 with idempotency: build the fused dataset once, zip it to
    Drive, and restore it to fast local disk. Returns the local fused dataset
    dir (data.yaml inside). Safe to re-run after any disconnect."""
    from ppe.download import download_all  # local import: avoids module cycle
    from ppe.remap import remap_all

    zip_path = cfg.prepared_dir / "fused.zip"
    if zip_path.is_file() and not zipfile.is_zipfile(zip_path):
        # A prior run was interrupted (e.g. Colab disconnect) mid-write —
        # zip_dir() is atomic now, but an old partial file may still be here.
        print(f"[prepare] {zip_path.name}: found but not a valid zip (partial "
              "write from an interrupted run) — discarding and rebuilding")
        zip_path.unlink()
    if zip_path.is_file():
        print(f"[prepare] {zip_path.name}: cached on Drive — skipping rebuild")
    else:
        print("[prepare] no cached fused.zip — building the fused dataset from scratch")
        print("[prepare] downloading/verifying raw sources…")
        raw = download_all(cfg, secrets)
        build = cfg.work_root / "prepared_build"
        if build.exists():
            shutil.rmtree(build)
        print("[prepare] splitting before remap (so capped-out SH17 images "
              "are never remapped)…")
        raw_split = {source: compute_raw_split(cfg, source, raw[source]) for source in raw}
        keep = {source: {name for names in split.values() for name in names}
                for source, split in raw_split.items()}
        print("[prepare] remapping labels by class name…")
        remap_all(cfg, raw, build / "remapped", keep=keep)
        print("[prepare] merging + splitting…")
        merge_and_split(cfg, {n: build / "remapped" / n for n in raw}, build,
                        raw_split=raw_split)
        shutil.copy2(build / "remapped" / "remap.json", build / "remap.json")
        shutil.rmtree(build / "remapped")  # intermediates stay out of the zip
        zip_dir(build, zip_path)
        shutil.rmtree(build)

    local = cfg.local_dataset_dir
    if not (local / "data.yaml").is_file():
        print(f"[prepare] restoring fused dataset from {zip_path.name} to local disk…")
        cfg.work_root.mkdir(parents=True, exist_ok=True)
        _unzip_with_progress(zip_path, cfg.work_root)
    else:
        print("[prepare] local fused dataset already present — skipping restore")
    write_data_yaml(local)  # refresh absolute paths for this machine
    return local


def _unzip_with_progress(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=f"[restore] {zip_path.name}", unit="file"):
            zf.extract(m, dest)


def _split_indices(cfg: Config, n: int, source: str) -> dict[str, list[int]]:
    """Deterministic 70/15/15 index split (name-sorted input assumed), with
    the SH17 train cap applied. Shared by the raw pre-remap split (so capped
    images are never remapped) and merge_and_split's own fallback path."""
    rng = random.Random(cfg.seed)
    order = list(range(n))
    rng.shuffle(order)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train_idx = order[:n_train]
    if source == "sh17" and cfg.sh17_train_cap is not None and len(train_idx) > cfg.sh17_train_cap:
        train_idx = train_idx[:cfg.sh17_train_cap]
    return {
        "train": train_idx,
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def compute_raw_split(cfg: Config, source: str, raw_dir: Path) -> dict[str, list[str]]:
    """Decide train/val/test membership (by filename) straight from the raw
    downloaded source, before remap runs. This lets the caller remap only the
    images that will actually be used — SH17's train slice gets capped at
    ``cfg.sh17_train_cap``, and remapping the rest just to discard them was
    wasted work (SH17 dwarfs the other 3 sources combined)."""
    from ppe.remap import _find_images  # local import: avoids module cycle

    images = _find_images(raw_dir)
    if not images:
        raise MergeError(f"{source}: no images found under {raw_dir}")
    idx = _split_indices(cfg, len(images), source)
    return {split: [images[i].name for i in idxs] for split, idxs in idx.items()}


def merge_and_split(cfg: Config, remapped: dict[str, Path], out_root: Path,
                    raw_split: dict[str, dict[str, list[str]]] | None = None) -> SplitResult:
    """``raw_split``, when given, is the pre-remap split decision from
    ``compute_raw_split`` (one entry per source that was pre-split) — reused
    here instead of re-deriving it, since ``remapped`` may already contain
    only the kept subset. Sources absent from ``raw_split`` fall back to
    splitting whatever pairs are present (used directly by tests)."""
    out_root = Path(out_root)
    dataset_dir = out_root / "fused"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dup_factors = {"ocp": cfg.ocp_dup_factor, "gasmask": cfg.gasmask_dup_factor}
    dup_factors = {s: f for s, f in dup_factors.items() if s in remapped}

    assignments: dict[str, dict[str, list[tuple[Path, Path, str]]]] = {}
    per_source: dict[str, dict[str, int]] = {}
    for source in sorted(remapped):
        pairs = _pairs(remapped[source])
        if raw_split and source in raw_split:
            by_name = {img.name: (img, label) for img, label in pairs}
            splits = {split: [by_name[name] for name in names]
                      for split, names in raw_split[source].items()}
        else:
            order = sorted(range(len(pairs)), key=lambda i: pairs[i][0].name)
            pairs = [pairs[i] for i in order]
            idx = _split_indices(cfg, len(pairs), source)
            splits = {split: [pairs[i] for i in idxs] for split, idxs in idx.items()}
        assignments[source] = {
            split: [(img, label, _safe_stem(source, img.stem)) for img, label in entries]
            for split, entries in splits.items()
        }
        per_source[source] = {split: len(idxs) for split, idxs in splits.items()}

    # copy into fused train/val/test, duplicating train pairs for weighted sources
    total_pairs = sum(len(e) for by_split in assignments.values() for e in by_split.values())
    total_dups = sum(
        len(by_split["train"]) * (dup_factors.get(source, 1) - 1)
        for source, by_split in assignments.items()
    )
    print(f"[merge] copying {total_pairs} image/label pairs into train/val/test "
          f"({total_dups} extra train duplicates for {sorted(dup_factors)})…")
    with tqdm(total=total_pairs + total_dups, desc="[merge] copying", unit="file") as pbar:
        for source, by_split in assignments.items():
            factor = dup_factors.get(source, 1)
            for split, entries in by_split.items():
                for img, label, stem in entries:
                    _copy_pair(img, label, dataset_dir / split, stem)
                    pbar.update(1)
                    if split == "train" and factor > 1:
                        for k in range(1, factor):
                            _copy_pair(img, label, dataset_dir / split, f"{stem}__dup{k}")
                            pbar.update(1)
            per_source[source]["train_after_dup"] = per_source[source]["train"] * factor

    write_data_yaml(dataset_dir)

    # holdout / eval sets
    ocp_entries = assignments.get("ocp", {}).get("test", [])
    proxy_entries = [e for s in ("sh17", "gasmask") for e in assignments.get(s, {}).get("test", [])]
    val_entries = [e for s in assignments for e in assignments[s]["val"]]
    ocp_test_dir = out_root / "ocp_test"
    proxy_dir = out_root / "industrial_proxy"
    fused_val_dir = out_root / "fused_val"
    print(f"[merge] writing eval sets: ocp_test={len(ocp_entries)} "
          f"industrial_proxy={len(proxy_entries)} fused_val={len(val_entries)}")
    _write_eval_set(ocp_entries, ocp_test_dir)
    _write_eval_set(proxy_entries, proxy_dir)
    _write_eval_set(val_entries, fused_val_dir)

    # hard disjointness checks
    train_base = {p.stem.split("__dup")[0] for p in (dataset_dir / "train" / "images").iterdir()}
    val_stems = {p.stem for p in (dataset_dir / "val" / "images").iterdir()}
    test_stems = {p.stem for p in (dataset_dir / "test" / "images").iterdir()}
    if train_base & val_stems or train_base & test_stems or val_stems & test_stems:
        raise MergeError("train/val/test splits overlap — split logic is broken")
    ocp_test_stems = {stem for _, _, stem in ocp_entries}
    if ocp_test_stems & train_base:
        raise MergeError("OCP test images leaked into train — refusing to continue")
    if any(stem.startswith("ocp__") for _, _, stem in proxy_entries):
        raise MergeError("industrial proxy must not contain OCP images")

    summary = {
        "seed": cfg.seed,
        "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac},
        "dup_factors": dup_factors,
        "per_source": per_source,
        "per_class_boxes": {
            split: _count_boxes_by_class(dataset_dir / split)
            for split in ("train", "val", "test")
        },
        "eval_sets": {
            "fused_val": len(val_entries),
            "ocp_test": len(ocp_entries),
            "industrial_proxy": len(proxy_entries),
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[merge] done — train={len(train_base)} val={len(val_stems)} test={len(test_stems)}")
    return SplitResult(dataset_dir, fused_val_dir, ocp_test_dir, proxy_dir, summary)
