"""Generate one Colab notebook per revision job from colab_train.ipynb.

Each copy differs only in its JOB cell, title and runtime hint, so the
copies never drift from the runner. Re-run after editing colab_train.ipynb.

Usage: python scripts/make_job_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppe.config import JOBS  # noqa: E402

FILES = {"job0": "job0_prepare", "job1": "job1_B_A012", "job2": "job2_S",
         "job3": "job3_A34", "job4": "job4_P", "collect": "collect"}
CPU_JOBS = {"job0", "collect"}
JOB_LINE = "JOB = 'job1'  # job0 | job1 | job2 | job3 | job4 | collect\n"


def make(job: str, template: dict) -> dict:
    nb = json.loads(json.dumps(template))
    job_cells = [c for c in nb["cells"] if "".join(c["source"]) == JOB_LINE]
    if len(job_cells) != 1:
        raise SystemExit(f"template must have exactly one cell equal to {JOB_LINE!r}")
    job_cells[0]["source"] = [f"JOB = '{job}'  # preset by scripts/make_job_notebooks.py\n"]
    runtime = "CPU (no GPU needed)" if job in CPU_JOBS else "T4 GPU"
    runs = ", ".join(JOBS[job]) or "no training"
    nb["cells"].insert(0, {
        "cell_type": "markdown", "id": "job-header", "metadata": {},
        "source": [f"# {FILES[job]} — runtime: **{runtime}**\n", "\n", f"Runs: {runs}\n"],
    })
    if job in CPU_JOBS:
        nb["metadata"].pop("accelerator", None)
    else:
        nb["metadata"]["accelerator"] = "GPU"
    return nb


def main() -> None:
    template = json.loads((ROOT / "colab_train.ipynb").read_text())
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for job in JOBS:
        path = out_dir / f"{FILES[job]}.ipynb"
        path.write_text(json.dumps(make(job, template), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path.relative_to(ROOT)}  ({', '.join(JOBS[job]) or 'no training'})")


if __name__ == "__main__":
    main()
