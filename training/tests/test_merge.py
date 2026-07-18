import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from conftest import write_png
from ppe.config import CANONICAL_CLASSES, Config
from ppe.merge import ensure_prepared, merge_and_split, zip_dir
from ppe.remap import remap_all

SECRETS = {"ROBOFLOW_API_KEY": "k", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "kk"}


def make_remapped_source(root: Path, name: str, n_images: int, cls_id: int = 0) -> Path:
    """A source dir in remap-output layout: <root>/<name>/{images,labels}."""
    src = root / name
    for i in range(n_images):
        write_png(src / "images" / f"{name}_{i}.png")
        (src / "labels").mkdir(parents=True, exist_ok=True)
        (src / "labels" / f"{name}_{i}.txt").write_text(f"{cls_id} 0.5 0.5 0.2 0.2\n")
    return src


def cfg_for(tmp_path) -> Config:
    return Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")


def stems(d: Path) -> set[str]:
    return {p.stem for p in (d / "images").iterdir()}


def base_stems(d: Path) -> set[str]:
    """Stems with any __dupN suffix stripped."""
    return {s.split("__dup")[0] for s in stems(d)}


def test_split_ratios_and_prefixes(tmp_path):
    src = make_remapped_source(tmp_path / "remapped", "sh17", 20)
    res = merge_and_split(cfg_for(tmp_path), {"sh17": src}, tmp_path / "out")
    assert len(stems(res.dataset_dir / "train")) == 14
    assert len(stems(res.dataset_dir / "val")) == 3
    assert len(stems(res.dataset_dir / "test")) == 3
    assert all(s.startswith("sh17__") for s in stems(res.dataset_dir / "train"))


def test_split_deterministic(tmp_path):
    src = make_remapped_source(tmp_path / "remapped", "sh17", 20)
    r1 = merge_and_split(cfg_for(tmp_path), {"sh17": src}, tmp_path / "out1")
    r2 = merge_and_split(cfg_for(tmp_path), {"sh17": src}, tmp_path / "out2")
    for split in ("train", "val", "test"):
        assert stems(r1.dataset_dir / split) == stems(r2.dataset_dir / split)


def test_sh17_train_cap_limits_train_only(tmp_path):
    src = make_remapped_source(tmp_path / "remapped", "sh17", 20)
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work", sh17_train_cap=5)
    res = merge_and_split(cfg, {"sh17": src}, tmp_path / "out")
    # uncapped would be 14/3/3 (see test_split_ratios_and_prefixes); cap trims only train
    assert len(stems(res.dataset_dir / "train")) == 5
    assert len(stems(res.dataset_dir / "val")) == 3
    assert len(stems(res.dataset_dir / "test")) == 3


def test_sh17_train_cap_does_not_affect_other_sources(tmp_path):
    src = make_remapped_source(tmp_path / "remapped", "ocp", 20)
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work", sh17_train_cap=5)
    res = merge_and_split(cfg, {"ocp": src}, tmp_path / "out")
    ocp_train = [s for s in stems(res.dataset_dir / "train") if "__dup" not in s]
    assert len(ocp_train) == 14  # int(20*0.7), cap only applies to sh17


def test_sh17_train_cap_none_disables_cap(tmp_path):
    src = make_remapped_source(tmp_path / "remapped", "sh17", 20)
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work", sh17_train_cap=None)
    res = merge_and_split(cfg, {"sh17": src}, tmp_path / "out")
    assert len(stems(res.dataset_dir / "train")) == 14


def test_duplication_train_only(tmp_path):
    root = tmp_path / "remapped"
    remapped = {
        "ocp": make_remapped_source(root, "ocp", 10),
        "gasmask": make_remapped_source(root, "gasmask", 10, cls_id=9),
    }
    res = merge_and_split(cfg_for(tmp_path), remapped, tmp_path / "out")
    train = stems(res.dataset_dir / "train")
    ocp_train = [s for s in train if s.startswith("ocp__")]
    gm_train = [s for s in train if s.startswith("gasmask__")]
    # int(10*0.7)=7 originals, x3 => 21 each
    assert len(ocp_train) == 21 and len(gm_train) == 21
    assert len([s for s in ocp_train if "__dup" in s]) == 14
    # val/test never duplicated
    for split in ("val", "test"):
        assert not any("__dup" in s for s in stems(res.dataset_dir / split))
    # every dup has a matching label file
    for p in (res.dataset_dir / "train" / "images").iterdir():
        assert (res.dataset_dir / "train" / "labels" / f"{p.stem}.txt").is_file()


def test_long_roboflow_filenames_are_truncated_safely(tmp_path):
    # Roboflow exports (esp. gas-masks) can concatenate every image tag into
    # a ~244-char stem — short enough to write to disk unprefixed, but past
    # the OS filename limit (255 bytes) once "{source}__" (9 chars) is added.
    # All 10 stems share the same first 240 chars (only the last 4 differ),
    # so truncation alone can't disambiguate them — the hash suffix must.
    def long_stem(i: int) -> str:
        return ("tag-" * 61)[:240] + f"{i:04d}"

    src = tmp_path / "remapped" / "gasmask"
    for i in range(10):
        stem = long_stem(i)
        assert len(stem) == 244
        write_png(src / "images" / f"{stem}.jpg")
        (src / "labels").mkdir(parents=True, exist_ok=True)
        (src / "labels" / f"{stem}.txt").write_text("9 0.5 0.5 0.2 0.2\n")

    res = merge_and_split(cfg_for(tmp_path), {"gasmask": src}, tmp_path / "out")
    all_stems = stems(res.dataset_dir / "train") | stems(res.dataset_dir / "val") | stems(res.dataset_dir / "test")
    assert len(all_stems) == 10 + 14  # 10 originals + (7 train * (dup_factor-1)=2) dups
    for s in base_stems(res.dataset_dir / "train") | base_stems(res.dataset_dir / "val") | base_stems(res.dataset_dir / "test"):
        assert len(s) <= 200
        assert s.startswith("gasmask__")


def test_holdouts_disjoint_and_ocp_free_proxy(tmp_path, fixture_sources):
    remapped_root = tmp_path / "remapped"
    remap_all(None, fixture_sources, remapped_root)
    remapped = {n: remapped_root / n for n in fixture_sources}
    res = merge_and_split(cfg_for(tmp_path), remapped, tmp_path / "out")
    # OCP test set: never in train
    assert stems(res.ocp_test_dir), "ocp_test must not be empty"
    assert stems(res.ocp_test_dir) & base_stems(res.dataset_dir / "train") == set()
    assert all(s.startswith("ocp__") for s in stems(res.ocp_test_dir))
    # proxy: SH17 + gasmask only, no OCP
    assert stems(res.proxy_dir), "proxy must not be empty"
    assert not any(s.startswith("ocp__") for s in stems(res.proxy_dir))
    # fused_val mirrors the val split
    assert stems(res.fused_val_dir) == stems(res.dataset_dir / "val")
    # each eval set dir is self-contained
    for d in (res.ocp_test_dir, res.proxy_dir, res.fused_val_dir):
        assert (d / "labels").is_dir() and (d / "data.yaml").is_file()


def test_data_yaml_and_summary(tmp_path, fixture_sources):
    remapped_root = tmp_path / "remapped"
    remap_all(None, fixture_sources, remapped_root)
    remapped = {n: remapped_root / n for n in fixture_sources}
    res = merge_and_split(cfg_for(tmp_path), remapped, tmp_path / "out")
    data = yaml.safe_load((res.dataset_dir / "data.yaml").read_text())
    assert data["names"] == CANONICAL_CLASSES and data["nc"] == 10
    assert data["path"] == str(res.dataset_dir)
    summary = json.loads((tmp_path / "out" / "split_summary.json").read_text())
    assert summary == res.summary
    assert summary["seed"] == 0
    assert summary["dup_factors"] == {"ocp": 3, "gasmask": 3}
    for source in fixture_sources:
        per = summary["per_source"][source]
        for split in ("train", "val", "test"):
            expected = len([s for s in stems(res.dataset_dir / split)
                            if s.startswith(f"{source}__") and "__dup" not in s])
            assert per[split] == expected, (source, split)
        actual_after_dup = len([s for s in stems(res.dataset_dir / "train")
                                if s.startswith(f"{source}__")])
        assert per["train_after_dup"] == actual_after_dup
    assert summary["eval_sets"]["ocp_test"] == len(stems(res.ocp_test_dir))
    assert "train" in summary["per_class_boxes"]


def test_ensure_prepared_rebuilds_after_partial_zip(tmp_path, fixture_sources, monkeypatch):
    cfg = cfg_for(tmp_path)
    for name, src in fixture_sources.items():
        dest = cfg.raw_dir / name
        shutil.copytree(src, dest)
        (dest / ".complete").touch()
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda *a, **k: pytest.fail("network download attempted"))
    monkeypatch.setattr("ppe.download._download_kaggle",
                        lambda *a, **k: pytest.fail("network download attempted"))

    local = ensure_prepared(cfg, SECRETS)
    assert (local / "data.yaml").is_file()

    # Simulate a disconnect mid-zip-write (pre-fix, zip_dir wrote straight to
    # zip_path with no atomic rename, so a crash left exactly this on Drive).
    zip_path = cfg.prepared_dir / "fused.zip"
    zip_path.write_bytes(b"not a real zip, truncated by a disconnect")
    shutil.rmtree(cfg.work_root)

    local = ensure_prepared(cfg, SECRETS)  # must rebuild, not raise BadZipFile
    assert (local / "data.yaml").is_file()
    assert zipfile.is_zipfile(zip_path)


def test_zip_dir_roundtrip(tmp_path):
    d = tmp_path / "d"
    write_png(d / "images" / "a.png")
    (d / "note.txt").write_text("hi")
    z = zip_dir(d, tmp_path / "d.zip")
    assert z == tmp_path / "d.zip" and z.is_file()
    with zipfile.ZipFile(z) as zf:
        assert sorted(n for n in zf.namelist() if not n.endswith("/")) == ["images/a.png", "note.txt"]
