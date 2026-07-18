"""Run download + remap + merge locally instead of in Colab.

Builds the exact same `raw/` + `prepared/fused.zip` layout the notebook
builds under Drive's `ppe-training/`, but on local disk. Upload the two
printed folders into Google Drive's `MyDrive/ppe-training/` afterward and
the notebook's Step 1 / Steps 2+3 cells will detect the cached artifacts
and skip straight to training — see training/docs/LOCAL_PREPARE.md.

Usage: .venv/bin/python scripts/prepare_local.py [--out DIR]
       (reads training/secrets.env — fill in real ROBOFLOW_API_KEY /
       KAGGLE_USERNAME / KAGGLE_KEY first)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.config import Config, load_secrets  # noqa: E402
from ppe.merge import ensure_prepared  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--out", type=Path, default=Path.home() / "ppe-training-local",
    help="local staging dir mirroring Drive's ppe-training/ (default: ~/ppe-training-local)",
)
args = parser.parse_args()

cfg = Config(drive_root=args.out, work_root=args.out / "_work")
secrets = load_secrets(cfg)
local_dataset = ensure_prepared(cfg, secrets)

print(f"\nDone. Upload these two folders into Google Drive's MyDrive/ppe-training/ "
      f"(merge, don't replace, if that folder already has a secrets.env):")
print(f"  {cfg.raw_dir}       -> MyDrive/ppe-training/raw/")
print(f"  {cfg.prepared_dir}  -> MyDrive/ppe-training/prepared/")
print(f"\n({local_dataset} is just the local working copy — not needed on Drive)")
