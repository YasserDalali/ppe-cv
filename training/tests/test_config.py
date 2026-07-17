import pytest

from ppe.config import CANONICAL_CLASSES, Config, SecretsError, load_secrets, parse_env_file


def test_canonical_classes_exact_order():
    assert CANONICAL_CLASSES == [
        "Person", "helmet", "vest", "gloves", "boots", "goggles",
        "no_helmet", "no_gloves", "no_goggle", "gas mask",
    ]


def test_config_dirs_derive_from_drive_root(tmp_path):
    cfg = Config(drive_root=tmp_path / "drive", work_root=tmp_path / "work")
    assert cfg.raw_dir == tmp_path / "drive" / "raw"
    assert cfg.prepared_dir == tmp_path / "drive" / "prepared"
    assert cfg.runs_dir == tmp_path / "drive" / "runs"
    assert cfg.results_dir == tmp_path / "drive" / "results"
    assert cfg.wheels_dir == tmp_path / "drive" / "wheels"
    assert cfg.local_dataset_dir == tmp_path / "work" / "fused"


def test_parse_env_file(tmp_path):
    p = tmp_path / "secrets.env"
    p.write_text("# comment\nROBOFLOW_API_KEY=abc\n\nKAGGLE_USERNAME = bob\nexport KAGGLE_KEY=k1\n")
    assert parse_env_file(p) == {
        "ROBOFLOW_API_KEY": "abc", "KAGGLE_USERNAME": "bob", "KAGGLE_KEY": "k1",
    }


def test_load_secrets_prefers_drive_then_local_then_env(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "secrets.env").write_text("ROBOFLOW_API_KEY=from-drive\n")
    local = tmp_path / "local.env"
    local.write_text("ROBOFLOW_API_KEY=from-local\nKAGGLE_USERNAME=lu\n")
    monkeypatch.setenv("KAGGLE_KEY", "from-env")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    cfg = Config(drive_root=drive, work_root=tmp_path, local_secrets=local)
    s = load_secrets(cfg)
    assert s == {"ROBOFLOW_API_KEY": "from-drive", "KAGGLE_USERNAME": "lu", "KAGGLE_KEY": "from-env"}


def test_load_secrets_missing_key_names_locations(tmp_path, monkeypatch):
    for k in ("ROBOFLOW_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = Config(drive_root=tmp_path / "nope", work_root=tmp_path, local_secrets=tmp_path / "no.env")
    with pytest.raises(SecretsError) as e:
        load_secrets(cfg)
    msg = str(e.value)
    assert "ROBOFLOW_API_KEY" in msg and "secrets.env" in msg
