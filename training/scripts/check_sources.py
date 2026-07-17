"""One-off: verify Roboflow slugs + fetch real class names to finalize NAME_MAPS.

Usage: .venv/bin/python scripts/check_sources.py   (reads training/secrets.env)
Prints ONLY workspace/project/version and class names. Never prints keys.
"""
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppe.config import Config, load_secrets  # noqa: E402

cfg = Config(drive_root=Path("/nonexistent"))
key = load_secrets(cfg)["ROBOFLOW_API_KEY"]

for ws, proj, ver in [
    (cfg.ppe_workspace, cfg.ppe_project, cfg.ppe_version),
    (cfg.gasmask_workspace, cfg.gasmask_project, cfg.gasmask_version),
    (cfg.ocp_workspace, cfg.ocp_project, cfg.ocp_version),
]:
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}?api_key={key}"
    try:
        data = json.load(urlopen(url))
        version = data.get("version", {})
        project = data.get("project", {})
        classes = version.get("classes") or project.get("classes")
        if isinstance(classes, dict):
            classes = sorted(classes)
        print(f"OK   {ws}/{proj}/{ver}: images={version.get('images')} classes={classes}")
    except Exception as exc:  # noqa: BLE001 — diagnostic script
        print(f"FAIL {ws}/{proj}/{ver}: {type(exc).__name__}: {exc}")
