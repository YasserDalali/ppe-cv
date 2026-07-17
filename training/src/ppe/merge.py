"""Step 3 — merge remapped sources, deterministic split, duplication, holdouts.

- Filenames prefixed "{source}__{orig}" for provenance and collision safety.
- Seeded 70/15/15 split, stratified per source (each source split separately).
- OCP / gas-mask sampling weight implemented as physical file duplication with
  "__dupN" suffixes, train split ONLY (verifiable counts, no custom sampler).
- Holdout eval sets: ocp_test (disjointness from train asserted), industrial
  proxy (SH17 + gas-mask test images, no OCP), fused_val (the val split).
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from ppe.config import CANONICAL_CLASSES, Config


class MergeError(RuntimeError):
    pass


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
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    made = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=src_dir)
    return Path(made)


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
    if not zip_path.is_file():
        raw = download_all(cfg, secrets)
        build = cfg.work_root / "prepared_build"
        if build.exists():
            shutil.rmtree(build)
        remap_all(cfg, raw, build / "remapped")
        merge_and_split(cfg, {n: build / "remapped" / n for n in raw}, build)
        shutil.copy2(build / "remapped" / "remap.json", build / "remap.json")
        shutil.rmtree(build / "remapped")  # intermediates stay out of the zip
        zip_dir(build, zip_path)
        shutil.rmtree(build)

    local = cfg.local_dataset_dir
    if not (local / "data.yaml").is_file():
        cfg.work_root.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(zip_path, cfg.work_root)
    write_data_yaml(local)  # refresh absolute paths for this machine
    return local


def merge_and_split(cfg: Config, remapped: dict[str, Path], out_root: Path) -> SplitResult:
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
        rng = random.Random(cfg.seed)
        order = sorted(range(len(pairs)), key=lambda i: pairs[i][0].name)
        rng.shuffle(order)
        n = len(pairs)
        n_train = int(n * cfg.train_frac)
        n_val = int(n * cfg.val_frac)
        splits = {
            "train": order[:n_train],
            "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        assignments[source] = {
            split: [(pairs[i][0], pairs[i][1], f"{source}__{pairs[i][0].stem}") for i in idxs]
            for split, idxs in splits.items()
        }
        per_source[source] = {split: len(idxs) for split, idxs in splits.items()}

    # copy into fused train/val/test, duplicating train pairs for weighted sources
    for source, by_split in assignments.items():
        factor = dup_factors.get(source, 1)
        for split, entries in by_split.items():
            for img, label, stem in entries:
                _copy_pair(img, label, dataset_dir / split, stem)
                if split == "train" and factor > 1:
                    for k in range(1, factor):
                        _copy_pair(img, label, dataset_dir / split, f"{stem}__dup{k}")
        per_source[source]["train_after_dup"] = per_source[source]["train"] * factor

    write_data_yaml(dataset_dir)

    # holdout / eval sets
    ocp_entries = assignments.get("ocp", {}).get("test", [])
    proxy_entries = [e for s in ("sh17", "gasmask") for e in assignments.get(s, {}).get("test", [])]
    val_entries = [e for s in assignments for e in assignments[s]["val"]]
    ocp_test_dir = out_root / "ocp_test"
    proxy_dir = out_root / "industrial_proxy"
    fused_val_dir = out_root / "fused_val"
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
    return SplitResult(dataset_dir, fused_val_dir, ocp_test_dir, proxy_dir, summary)
