import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_clones_code_from_github_and_keeps_artifacts_on_drive():
    source = _source(ROOT / "colab_train.ipynb")

    assert "https://github.com/YasserDalali/ppe-cv.git" in source
    assert "REPO_BRANCH = 'revision'" in source
    assert "'--branch', REPO_BRANCH" in source
    assert "CODE = Path('/content/ppe-cv/training')" in source
    assert "DRIVE_ROOT = Path('/content/drive/MyDrive/ppe-training')" in source
    assert "userdata.get('GITHUB_TOKEN')" in source
    assert "GIT_ASKPASS" in source
    assert "x-access-token" in source
    assert "https://{github_token}@" not in source
    assert "DRIVE_ROOT / 'training'" not in source
    assert "run_job(cfg, JOB, secrets)" in source


def test_generated_job_notebooks_preset_their_job(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_job_notebooks", ROOT / "scripts" / "make_job_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    template = json.loads((ROOT / "colab_train.ipynb").read_text())
    from ppe.config import JOBS

    for job in JOBS:
        nb = mod.make(job, template)
        source = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert f"JOB = '{job}'" in source
        assert "JOB = 'job1'  # job0" not in source or job == "job1"
        assert ("accelerator" in nb["metadata"]) == (job not in mod.CPU_JOBS)
