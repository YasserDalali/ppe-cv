# GitHub Colab Bootstrap Design

**Date:** 2026-07-17
**Status:** Approved

## Goal

Use GitHub as the source of pipeline code while retaining Google Drive only
for secrets and durable artifacts.

## Design

The notebook mounts Drive, reads a fine-grained `GITHUB_TOKEN` from Colab
Secrets, clones the private repository's `main` branch into
`/content/ppe-cv`, and imports `ppe` from `/content/ppe-cv/training/src`.
Dependencies install from the cloned `training/requirements.txt`. Git receives
the token through an environment-scoped HTTP authorization header, never a
token-bearing URL or printed command.

`DRIVE_ROOT` remains `/content/drive/MyDrive/ppe-training`, so `secrets.env`,
raw datasets, prepared data, checkpoints, and results survive runtime
recycling. No code, wheelhouse, or uploaded `training/` directory is read
from Drive.

On each Run All, the local clone is deleted and recreated from the named
branch. This avoids stale-code problems and is cheap because the repository
contains no downloaded dependencies or model weights.

## Success criteria

- The notebook contains the repository URL and `main` branch.
- Private-repository authentication uses the Colab `GITHUB_TOKEN` secret.
- The setup cell clones code under `/content`, not Drive.
- `REQS` and `sys.path` point at the local clone.
- `DRIVE_ROOT` continues to back all durable pipeline artifacts.
- The guide contains no instruction to upload or synchronize code to Drive.
