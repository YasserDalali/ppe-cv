import json
import statistics
import zipfile
from pathlib import Path

import pytest

from conftest import OCP_RECORDINGS, revision_cfg
from ppe.config import EXPERIMENTS
from ppe.pool import (PoolError, assemble_run, build_pool_dir, check_and_clean,
                      ensure_ocp_originals, ensure_pool, group_key, make_ocp_folds,
                      recording_id, run_plan, split_public)

REAL_RECORDING_FRAMES = [33, 27, 23, 21, 18, 13, 13, 12, 12, 9, 9, 8, 5, 4]  # 207 originals


def test_recording_id_and_group_key():
    assert recording_id("ocp__VID_20260716_163549_f00211_jpg.rf.8bba.jpg") == "VID_20260716_163549"
    assert recording_id("IMG_9845_MOV_f00012_jpg.rf.x.jpg") == "IMG_9845"
    assert recording_id("sh17_42.jpg") == "sh17_42"
    assert group_key("pic_7_jpg.rf.abc.jpg") == group_key("pic_7_jpg.rf.def.jpg") == "pic_7_jpg"
    assert group_key("plain.png") == "plain"


def test_split_public_keeps_roboflow_copies_together(tmp_path):
    names = [f"pic{i}_jpg.rf.{a}.jpg" for i in range(30) for a in "abc"]
    sp = split_public(revision_cfg(tmp_path), "roboflow_ppe", names)
    side = {}
    for split in ("train", "val", "test"):
        for n in sp[split]:
            assert side.setdefault(group_key(n), split) == split, n
    assert sorted(sp["train"] + sp["val"] + sp["test"]) == sorted(names)
    assert sp["extra"] == []


def test_split_public_sh17_extra_pool_is_disjoint(tmp_path):
    cfg = revision_cfg(tmp_path, sh17_train_cap=10, sh17_extra_pool=20)
    sp = split_public(cfg, "sh17", [f"img{i:03d}.jpg" for i in range(100)])
    assert len(sp["train"]) == 10 and len(sp["extra"]) == 20
    assert not set(sp["extra"]) & (set(sp["train"]) | set(sp["val"]) | set(sp["test"]))
    # the capped train slice is unchanged by reserving the extra pool
    plain = split_public(revision_cfg(tmp_path, sh17_train_cap=10, sh17_extra_pool=0),
                         "sh17", [f"img{i:03d}.jpg" for i in range(100)])
    assert plain["train"] == sp["train"]


def test_make_ocp_folds_on_the_real_distribution():
    names = [f"ocp__VID_20260716_1{r:05d}_f{k:05d}.jpg"
             for r, n in enumerate(REAL_RECORDING_FRAMES) for k in range(n)]
    folds = make_ocp_folds(names, 5)
    assert folds == make_ocp_folds(list(reversed(names)), 5)  # deterministic
    recs = [r for f in folds.values() for r in f]
    assert sorted(recs) == sorted({recording_id(n) for n in names})  # each tested once
    loads = [sum(1 for n in names if recording_id(n) in set(f)) for f in folds.values()]
    mean = statistics.fmean(loads)
    assert all(abs(x - mean) <= 0.15 * mean for x in loads), loads


def test_make_ocp_folds_needs_enough_recordings():
    with pytest.raises(PoolError):
        make_ocp_folds(["VID_1_1_f0.jpg", "VID_2_2_f0.jpg"], 5)


@pytest.fixture
def pool(tmp_path, revision_raw):
    cfg = revision_cfg(tmp_path)
    return cfg, build_pool_dir(cfg, revision_raw, tmp_path / "pool")


def test_build_pool_writes_splits_counts_and_hashes(pool):
    cfg, pool_dir = pool
    doc = json.loads((pool_dir / "splits.json").read_text())
    counts = json.loads((pool_dir / "counts.json").read_text())
    assert len(doc["ocp"]["frames"]) == sum(OCP_RECORDINGS.values())
    assert sorted(r for f in doc["ocp"]["folds"].values() for r in f) == sorted(OCP_RECORDINGS)
    assert len(doc["hashes"]) == sum(1 for _ in (pool_dir / "sources").rglob("images/*"))
    assert counts["per_source"]["sh17"]["extra"]["images"] == 40
    assert set(counts["ocp_recordings"]) == set(OCP_RECORDINGS)
    assert sum(f["images"] for f in counts["ocp_folds"].values()) == len(doc["ocp"]["frames"])


def _stems(d: Path) -> set:
    return {p.stem for p in (d / "images").iterdir()}


def test_assemble_A_fold_has_no_recording_on_both_sides(pool):
    cfg, pool_dir = pool
    run = assemble_run(cfg, "A_f0", pool_dir)
    train = _stems(run.dataset_dir / "train")
    ocp_train_recs = {recording_id(s) for s in train if s.startswith("ocp__")}
    ocp_test_recs = {recording_id(s) for s in _stems(run.eval_sets["ocp"])}
    assert ocp_test_recs and ocp_train_recs and not ocp_train_recs & ocp_test_recs
    assert ocp_test_recs == set(run.meta["ocp_test_recordings"])
    # x3 copies exist, only in train, only for OCP
    dups = [s for s in train if "__dup" in s]
    assert dups and all(s.startswith("ocp__") for s in dups)
    assert len(dups) == 2 * run.meta["n_ocp_train_frames"]
    for d in [run.dataset_dir / "val", *run.eval_sets.values()]:
        assert not any("__dup" in s for s in _stems(d))
    assert run.meta["n_train_images"] == len(train)


def test_assemble_B_P_S_never_train_on_ocp_and_score_all_frames(pool):
    cfg, pool_dir = pool
    n_frames = sum(OCP_RECORDINGS.values())
    for run_id in ("B_s0", "P_s1", "S_s2"):
        run = assemble_run(cfg, run_id, pool_dir)
        assert not any(s.startswith("ocp__") for s in _stems(run.dataset_dir / "train"))
        assert len(_stems(run.eval_sets["ocp"])) == n_frames
    b = assemble_run(cfg, "B_s0", pool_dir)
    assert all(s.startswith("roboflow_ppe__") for s in _stems(b.dataset_dir / "train"))


def test_S_matches_A_train_size_and_P_does_not(pool):
    cfg, pool_dir = pool
    doc = json.loads((pool_dir / "splits.json").read_text())
    a_mean = statistics.fmean(run_plan(cfg, f"A_f{f}", doc)["n_train"]
                              for f in EXPERIMENTS["A"]["folds"])
    s = assemble_run(cfg, "S_s0", pool_dir)
    assert abs(s.meta["n_train_images"] - a_mean) <= 0.01 * a_mean
    assert s.meta["n_sh17_extra"] > 0
    p = run_plan(cfg, "P_s0", doc)
    assert p["n_train"] < s.meta["n_train_images"]


def test_same_val_set_for_every_model(pool):
    cfg, pool_dir = pool
    vals = {rid: _stems(assemble_run(cfg, rid, pool_dir).dataset_dir / "val")
            for rid in ("B_s0", "P_s0", "S_s0", "A_f1")}
    assert len({frozenset(v) for v in vals.values()}) == 1
    assert not any(s.startswith("ocp__") for s in vals["A_f1"])


def test_check_and_clean_stops_on_ocp_leaks_and_drops_public_duplicates():
    hashes = {"ocp/VID_1_1_f0.jpg": "h1", "ocp/VID_1_1_f9.jpg": "h2",
              "ocp/VID_2_2_f0.jpg": "h1", "sh17/a.jpg": "p1", "sh17/b.jpg": "p1",
              "sh17/c.jpg": "p2"}
    with pytest.raises(PoolError, match="VID_1_1"):  # same recording on both sides
        check_and_clean([("ocp", "VID_1_1_f0.jpg")], {"test": [("ocp", "VID_1_1_f9.jpg")]}, hashes)
    with pytest.raises(PoolError, match="identical"):  # same bytes, other recording
        check_and_clean([("ocp", "VID_1_1_f0.jpg")], {"test": [("ocp", "VID_2_2_f0.jpg")]}, hashes)
    cleaned, dropped = check_and_clean([("sh17", "a.jpg")],
                                       {"test": [("sh17", "b.jpg"), ("sh17", "c.jpg")]}, hashes)
    assert cleaned["test"] == [("sh17", "c.jpg")] and dropped == {"test": 1}


def _upload_ocp(cfg, ocp_dir: Path, readme: str | None = None) -> None:
    cfg.ocp_upload_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cfg.ocp_upload_zip, "w") as zf:
        for p in ocp_dir.rglob("*"):
            if p.is_file() and not (readme is not None and p.name == "README.roboflow.txt"):
                zf.write(p, str(p.relative_to(ocp_dir)))
        if readme is not None:
            zf.writestr("README.roboflow.txt", readme)


def test_ensure_ocp_originals_rejects_augmented_or_wrong_exports(tmp_path, revision_raw):
    cfg = revision_cfg(tmp_path)
    with pytest.raises(PoolError, match="missing"):
        ensure_ocp_originals(cfg)
    _upload_ocp(cfg, revision_raw["ocp"], readme="Augmentations applied: 2 outputs per example")
    with pytest.raises(PoolError, match="augmentation"):
        ensure_ocp_originals(cfg)
    _upload_ocp(cfg, revision_raw["ocp"])
    with pytest.raises(PoolError, match="expected 999"):
        ensure_ocp_originals(revision_cfg(tmp_path, ocp_expected_images=999))
    assert (ensure_ocp_originals(cfg) / "data.yaml").is_file()


def test_ensure_pool_builds_once_then_restores_without_secrets(tmp_path, revision_raw, monkeypatch):
    cfg = revision_cfg(tmp_path)
    with pytest.raises(PoolError, match="job0"):
        ensure_pool(cfg, None)
    _upload_ocp(cfg, revision_raw["ocp"])
    monkeypatch.setattr("ppe.download.ensure_source", lambda c, s, name: revision_raw[name])
    pool_dir = ensure_pool(cfg, {"ROBOFLOW_API_KEY": "k"})
    assert (cfg.prepared_dir / "pool.zip").is_file() and (pool_dir / "splits.json").is_file()
    monkeypatch.setattr("ppe.download.ensure_source",
                        lambda *a: pytest.fail("pool must not be rebuilt"))
    import shutil
    shutil.rmtree(pool_dir)
    assert (ensure_pool(cfg, None) / "counts.json").is_file()
