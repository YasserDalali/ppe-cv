import pytest

from ppe.config import Config
from ppe.train import decide_train_action, train_model_a


def cfg_for(tmp_path) -> Config:
    return Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")


def run_dir_with(tmp_path, files):
    d = tmp_path / "drive" / "runs" / "model_a"
    for f in files:
        p = d / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return d


def test_decide_fresh_when_no_run(tmp_path):
    assert decide_train_action(tmp_path / "nope") == "fresh"


def test_decide_resume_when_unfinished(tmp_path):
    d = run_dir_with(tmp_path, ["weights/last.pt"])
    assert decide_train_action(d) == "resume"


def test_decide_skip_when_finished(tmp_path):
    d = run_dir_with(tmp_path, ["weights/last.pt", "weights/best.pt", ".finished"])
    assert decide_train_action(d) == "skip"


class FakeYOLO:
    instances = []

    def __init__(self, source):
        self.source = source
        self.train_calls = []
        FakeYOLO.instances.append(self)

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        if not kwargs.get("resume"):
            run = kwargs["project"] + "/" + kwargs["name"]
        else:
            run = str(self.source).rsplit("/weights/", 1)[0]
        from pathlib import Path
        w = Path(run) / "weights"
        w.mkdir(parents=True, exist_ok=True)
        (w / "best.pt").touch()
        (w / "last.pt").touch()


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakeYOLO.instances = []
    monkeypatch.setattr("ppe.train._load_yolo_class", lambda: FakeYOLO)


def test_fresh_train_passes_exact_hyperparams(tmp_path):
    cfg = cfg_for(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()
    best = train_model_a(cfg, data_yaml)
    assert best == cfg.runs_dir / "model_a" / "weights" / "best.pt"
    assert best.is_file()
    assert (cfg.runs_dir / "model_a" / ".finished").is_file()
    (model,) = FakeYOLO.instances
    assert model.source == "yolov8n.pt"
    (kw,) = model.train_calls
    assert kw == {
        "data": str(data_yaml), "epochs": 50, "imgsz": 640, "batch": 16,
        "seed": 0, "erasing": 0.4, "close_mosaic": 10, "save_period": 1,
        "project": str(cfg.runs_dir), "name": "model_a", "exist_ok": True,
    }


def test_resume_uses_last_pt(tmp_path):
    cfg = cfg_for(tmp_path)
    last = cfg.runs_dir / "model_a" / "weights" / "last.pt"
    last.parent.mkdir(parents=True, exist_ok=True)
    last.touch()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.touch()
    train_model_a(cfg, data_yaml)
    (model,) = FakeYOLO.instances
    assert model.source == str(last)
    assert model.train_calls == [{"resume": True}]
    assert (cfg.runs_dir / "model_a" / ".finished").is_file()


def test_skip_returns_best_without_touching_yolo(tmp_path):
    cfg = cfg_for(tmp_path)
    d = cfg.runs_dir / "model_a"
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.pt").touch()
    (d / "weights" / "last.pt").touch()
    (d / ".finished").touch()
    best = train_model_a(cfg, tmp_path / "data.yaml")
    assert best == d / "weights" / "best.pt"
    assert FakeYOLO.instances == []
