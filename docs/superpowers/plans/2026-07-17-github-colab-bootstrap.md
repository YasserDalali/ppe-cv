# GitHub Colab Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Colab load pipeline code from GitHub while Drive stores only secrets and durable artifacts.

**Architecture:** A notebook bootstrap cell reads `GITHUB_TOKEN` from Colab
Secrets, recreates `/content/ppe-cv` from the private repository's
`main` branch, installs `/content/ppe-cv/training/requirements.txt`,
and imports from `/content/ppe-cv/training/src`. Existing `Config.drive_root`
continues to direct datasets, checkpoints, and results to Drive.

**Tech Stack:** Google Colab, Git, Python, Jupyter notebook JSON, pytest

## Global Constraints

- Repository: `https://github.com/YasserDalali/ppe-cv.git`
- Branch: `main`
- Code root: `/content/ppe-cv/training`
- Artifact root: `/content/drive/MyDrive/ppe-training`
- GitHub authentication: fine-grained `GITHUB_TOKEN` in Colab Secrets
- Secrets remain outside Git at `<artifact root>/secrets.env`.

---

### Task 1: Specify notebook bootstrap behavior

**Files:**
- Create: `training/tests/test_colab_notebook.py`

**Interfaces:**
- Consumes: `training/colab_train.ipynb`
- Produces: assertions defining GitHub/local-code and Drive/artifact boundaries

- [ ] **Step 1: Write the failing test**

```python
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
    assert "CODE = Path('/content/ppe-cv/training')" in source
    assert "DRIVE_ROOT = Path('/content/drive/MyDrive/ppe-training')" in source
    assert "userdata.get('GITHUB_TOKEN')" in source
    assert "GIT_CONFIG_VALUE_0" in source
    assert "DRIVE_ROOT / 'training'" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd training && PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab_notebook.py -q`

Expected: FAIL because the current notebook reads code from Drive.

### Task 2: Replace Drive code synchronization with Git clone

**Files:**
- Modify: `training/colab_train.ipynb`
- Modify: `training/GUIDE.md`

**Interfaces:**
- Consumes: public GitHub branch and mounted Drive
- Produces: local `CODE`, `REQS`, import path, and durable `DRIVE_ROOT`

- [ ] **Step 1: Replace the setup cell**

The cell must remove `/content/ppe-cv`, run:

```bash
git clone --depth 1 --branch main \
  https://github.com/YasserDalali/ppe-cv.git /content/ppe-cv
```

Git authentication is supplied through `GIT_CONFIG_VALUE_0` in the clone
process environment. It then installs from
`/content/ppe-cv/training/requirements.txt`.

- [ ] **Step 2: Point package imports at the clone**

```python
sys.path.insert(0, str(CODE / "src"))
```

- [ ] **Step 3: Update the guide**

Document opening the notebook in Colab, mounting Drive, and using Run All.
Remove all code-upload and wheelhouse instructions.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
cd training
PYTHONPATH=src .venv/bin/python -m pytest tests/test_colab_notebook.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: all tests pass.

### Task 3: Publish the runnable branch

**Files:**
- Commit all approved training changes and the design/plan documents.

**Interfaces:**
- Consumes: verified local development tree
- Produces: `origin/main`, cloneable by Colab

- [ ] **Step 1: Review status and diff**

Run: `git status --short && git diff --check && git diff`

- [ ] **Step 2: Commit**

```bash
git add training docs/superpowers/specs/2026-07-17-github-colab-bootstrap-design.md docs/superpowers/plans/2026-07-17-github-colab-bootstrap.md
git commit -m "fix(training): bootstrap Colab code from GitHub"
```

- [ ] **Step 3: Push**

Run: `git push origin main`

Expected: branch is available to the private-repository token used by Colab.
