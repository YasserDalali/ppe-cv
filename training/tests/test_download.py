import pytest
import yaml

from conftest import GASMASK_FIXTURE_CLASSES, SH17_CLASSES, make_yolo_source, write_png
from ppe.config import Config
from ppe.download import (
    SOURCES,
    DownloadError,
    download_all,
    ensure_source,
    validate_source,
)

SECRETS = {"ROBOFLOW_API_KEY": "k", "KAGGLE_USERNAME": "u", "KAGGLE_KEY": "kk"}


def cfg_for(tmp_path) -> Config:
    return Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")


def build_gasmask(dest):
    make_yolo_source(dest.parent, dest.name, GASMASK_FIXTURE_CLASSES,
                     [[(1, 0.5, 0.5, 0.3, 0.3)] for _ in range(10)])


def test_skip_when_marker_present(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    build_gasmask(cfg.raw_dir / "gasmask")
    (cfg.raw_dir / "gasmask" / ".complete").touch()
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda *a, **k: pytest.fail("downloader called despite marker"))
    assert ensure_source(cfg, SECRETS, "gasmask") == cfg.raw_dir / "gasmask"


def test_download_validate_and_mark(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    calls = []
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda cfg, key, ws, proj, ver, dest: (calls.append((ws, proj, ver)),
                                                               build_gasmask(dest))[0])
    path = ensure_source(cfg, SECRETS, "gasmask")
    assert (path / ".complete").is_file()
    assert calls == [("daniil-yarmov", "gas-masks", 1)]
    # second call skips
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda *a, **k: pytest.fail("re-downloaded despite marker"))
    ensure_source(cfg, SECRETS, "gasmask")


def test_validate_source_stats(tmp_path):
    build_gasmask(tmp_path / "gasmask")
    stats = validate_source(tmp_path / "gasmask", "gasmask")
    assert stats.images == 10 and stats.boxes == 10
    assert stats.class_names == GASMASK_FIXTURE_CLASSES
    assert stats.per_class_boxes == {"gas mask": 10}


def test_validate_source_bad_label_id(tmp_path):
    src = make_yolo_source(tmp_path, "bad", ["a"], [[(99, 0.5, 0.5, 0.1, 0.1)]], layout="flat")
    with pytest.raises(DownloadError) as e:
        validate_source(src, "bad")
    assert "bad_0.txt" in str(e.value) and "99" in str(e.value)


def test_ocp_failure_is_fail_fast_and_readable(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr("ppe.download._download_roboflow", boom)
    with pytest.raises(DownloadError) as e:
        ensure_source(cfg, SECRETS, "ocp")
    msg = str(e.value)
    assert "OCP" in msg and "no OCP-less mode" in msg
    assert "yasser-dalali/ocp-snzjw/1" in msg
    assert "generate" in msg.lower()  # hint: generate a version in Roboflow UI


def test_sh17_classes_txt_fallback_synthesizes_data_yaml(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)

    def fake_kaggle(cfg_, secrets, dest):
        for i in range(3):
            write_png(dest / "images" / f"sh17_{i}.png")
            (dest / "labels").mkdir(parents=True, exist_ok=True)
            (dest / "labels" / f"sh17_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (dest / "classes.txt").write_text("\n".join(SH17_CLASSES) + "\n")

    monkeypatch.setattr("ppe.download._download_kaggle", fake_kaggle)
    path = ensure_source(cfg, SECRETS, "sh17")
    data = yaml.safe_load((path / "data.yaml").read_text())
    assert data["names"] == SH17_CLASSES


def test_download_all_returns_all_sources(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    monkeypatch.setattr("ppe.download._download_roboflow",
                        lambda cfg_, key, ws, proj, ver, dest: build_gasmask(dest))

    def fake_kaggle(cfg_, secrets, dest):
        make_yolo_source(dest.parent, dest.name, SH17_CLASSES,
                         [[(0, 0.5, 0.5, 0.2, 0.2)]], layout="flat")

    monkeypatch.setattr("ppe.download._download_kaggle", fake_kaggle)
    paths = download_all(cfg, SECRETS)
    assert set(paths) == set(SOURCES) == {"roboflow_ppe", "sh17", "gasmask", "ocp"}
    for p in paths.values():
        assert (p / ".complete").is_file()


def test_bad_zip_cleans_partial_and_hints(tmp_path, monkeypatch):
    import zipfile

    cfg = cfg_for(tmp_path)
    dest = cfg.raw_dir / "roboflow_ppe"
    dest.mkdir(parents=True)
    (dest / "junk.txt").write_text("partial")

    def boom(*a, **k):
        raise zipfile.BadZipFile("File is not a zip file")

    monkeypatch.setattr("ppe.download._download_roboflow", boom)
    with pytest.raises(DownloadError) as e:
        ensure_source(cfg, SECRETS, "roboflow_ppe")
    msg = str(e.value)
    assert "BadZipFile" in msg and "stage on local disk" in msg
    assert not dest.exists()  # partial Drive folder removed


def test_promote_scratch_unwraps_nested_roboflow_layout(tmp_path):
    from ppe.download import _promote_scratch

    scratch = tmp_path / "scratch"
    nested = scratch / "ppe_detection-dnfen-3"
    build_gasmask(nested)
    dest = tmp_path / "raw" / "roboflow_ppe"
    _promote_scratch(scratch, dest)
    assert (dest / "data.yaml").is_file()
    assert not (dest / "ppe_detection-dnfen-3").exists()


def test_scratch_path_does_not_precreate_directory(tmp_path):
    from ppe.download import _scratch_path

    cfg = cfg_for(tmp_path)
    p = _scratch_path(cfg, "roboflow_ppe")
    assert not p.exists()
    assert p.parent.is_dir()
