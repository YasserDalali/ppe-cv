import json

import pytest
import yaml

from conftest import SH17_CLASSES, make_yolo_source
from ppe.config import CANONICAL_CLASSES
from ppe.remap import (
    DROP,
    NAME_MAPS,
    RemapError,
    SH17_NAME_MAP,
    normalize_name,
    read_class_names,
    remap_all,
    remap_source,
)


def test_normalize_name():
    assert normalize_name(" No-Helmet ") == "no_helmet"
    assert normalize_name("gas mask") == "gas_mask"
    assert normalize_name("Person") == "person"


def test_sh17_map_covers_all_17_names():
    assert {normalize_name(n) for n in SH17_CLASSES} == set(SH17_NAME_MAP)
    kept = {k: v for k, v in SH17_NAME_MAP.items() if v != DROP}
    assert kept == {
        "person": "Person", "helmet": "helmet", "safety_vest": "vest",
        "gloves": "gloves", "shoes": "boots", "glasses": "goggles",
    }


def test_sh17_remap_accounting_and_ids(fixture_sources, tmp_path):
    out = tmp_path / "remapped" / "sh17"
    rep = remap_source(fixture_sources["sh17"], out, SH17_NAME_MAP, "sh17")
    assert rep.images_before == 17 and rep.images_after == 17
    assert rep.boxes_before == 17
    assert rep.boxes_after == 6
    assert rep.boxes_dropped == 11
    assert rep.boxes_before == rep.boxes_after + rep.boxes_dropped
    # helmet was SH17 id 14 -> canonical id 1
    assert (out / "labels" / "sh17_14.txt").read_text().split()[0] == "1"
    # person id 0 -> Person id 0; glasses id 3 -> goggles id 5
    assert (out / "labels" / "sh17_0.txt").read_text().split()[0] == "0"
    assert (out / "labels" / "sh17_3.txt").read_text().split()[0] == "5"
    # dropped class (head, id 1): image kept, label file empty = background
    assert (out / "images" / "sh17_1.png").exists()
    assert (out / "labels" / "sh17_1.txt").read_text().strip() == ""
    assert rep.dropped_by_class["head"] == 1


def test_old_bug_regression_unmapped_names_hard_fail(fixture_sources, tmp_path):
    """Old bug: SH17 ids >= 10 silently dropped. A map that only covers the
    first 10 names must hard-fail, not silently drop."""
    partial_map = {normalize_name(n): DROP for n in SH17_CLASSES[:10]}
    with pytest.raises(RemapError) as e:
        remap_source(fixture_sources["sh17"], tmp_path / "out", partial_map, "sh17")
    msg = str(e.value)
    assert "shoes" in msg and "helmet" in msg  # names with ids >= 10 are listed


def test_out_of_range_label_id_hard_fail(tmp_path):
    src = make_yolo_source(tmp_path, "bad", ["a", "b"], [[(5, 0.5, 0.5, 0.1, 0.1)]], layout="flat")
    with pytest.raises(RemapError) as e:
        remap_source(src, tmp_path / "out", {"a": DROP, "b": DROP}, "bad")
    assert "bad_0.txt" in str(e.value)


def test_read_class_names_list_and_dict(tmp_path):
    d1 = tmp_path / "s1"
    d1.mkdir()
    (d1 / "data.yaml").write_text(yaml.safe_dump({"names": ["x", "y"]}))
    assert read_class_names(d1) == ["x", "y"]
    d2 = tmp_path / "s2"
    d2.mkdir()
    (d2 / "data.yaml").write_text(yaml.safe_dump({"names": {0: "x", 1: "y"}}))
    assert read_class_names(d2) == ["x", "y"]


def test_remap_all_four_sources_writes_remap_json(fixture_sources, tmp_path):
    out = tmp_path / "remapped"
    reports = remap_all(None, fixture_sources, out)
    assert set(reports) == {"roboflow_ppe", "sh17", "gasmask", "ocp"}
    data = json.loads((out / "remap.json").read_text())
    assert data["classes"] == CANONICAL_CLASSES
    assert set(data["maps"]) == set(reports)
    # gasmask junk class "0" dropped, real one kept
    assert data["maps"]["gasmask"]["gas_mask"] == "gas mask"
    assert data["maps"]["gasmask"]["0"] == DROP
    for name, rep in data["reports"].items():
        assert rep["boxes_before"] == rep["boxes_after"] + rep["boxes_dropped"], name
        assert rep["images_before"] == rep["images_after"], name
    # every source produced images+labels dirs
    for name in reports:
        assert (out / name / "images").is_dir() and (out / name / "labels").is_dir()


def test_name_maps_cover_live_class_lists():
    """Maps must cover the real class names fetched live on 2026-07-17."""
    live = {
        "roboflow_ppe": ["gloves", "helmet", "no-gloves", "no-helmet", "no-shoes",
                         "no-vest", "person", "shoes", "vest"],
        "gasmask": ["0", "gas mask"],
        "ocp": ["Person", "boots", "gas mask", "gloves", "goggles", "helmet", "vest"],
        "sh17": SH17_CLASSES,
    }
    for source, names in live.items():
        missing = [n for n in names if normalize_name(n) not in NAME_MAPS[source]]
        assert not missing, f"{source}: unmapped {missing}"
