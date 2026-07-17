import json
from pathlib import Path


def test_notebook_clones_code_from_github_and_keeps_artifacts_on_drive():
    notebook = json.loads(
        (Path(__file__).parents[1] / "colab_train.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "https://github.com/YasserDalali/ppe-cv.git" in source
    assert "REPO_BRANCH = 'main'" in source
    assert "'--branch', REPO_BRANCH" in source
    assert "CODE = Path('/content/ppe-cv/training')" in source
    assert "DRIVE_ROOT = Path('/content/drive/MyDrive/ppe-training')" in source
    assert "userdata.get('GITHUB_TOKEN')" in source
    assert "GIT_CONFIG_VALUE_0" in source
    assert "Authorization: Bearer" in source
    assert "https://{github_token}@" not in source
    assert "DRIVE_ROOT / 'training'" not in source
