# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A workspace for drafting an academic paper: a CCTV-based YOLOv8 + LLM-judge system for real-time PPE compliance detection at OCP Group industrial sites. There is no application code — the repo holds the paper outline, reference papers used as structure templates, and the raw training-run data.

## Where everything lives

Read `content/ROUTING.md` first — it is the authoritative reading guide (which template paper to copy which structural convention from, and the suggested reading order for drafting).

- `content/structure.md` — the full section-by-section outline of the paper being written. Metric values in it are intentionally `[X]` placeholders.
- `content/md/*.md` — pdftotext conversions of the reference papers plus the French project pitch (`ppe-systeme-pitch-fr.md`, the source of the paper's stats, vigilance argument, and 4-stage pipeline). Always read these, never the PDFs.
- `content/pdfs(do not depend on)/` — source PDFs; the folder name is the instruction.
- `content/references.txt` — bibliography seeds.
- `content/metrics.txt` — raw Google Colab training log (~700KB, too large to read whole). The real numbers to fill structure.md's placeholders come from here.
- `content/dashboard.png` — training-curves figure, intended for reuse in Results.

## Reading metrics.txt

The file starts with pip-install noise; the useful parts are the trainer config (grep for `engine/trainer`) and the final per-class validation table at the very end (read the tail). Headline facts of the run: YOLOv8n (Ultralytics 8.4.95), 50 epochs, 640×640, Tesla T4, dataset `PPE-Fusion` (includes SH17); final all-class P=0.792, R=0.621, mAP50=0.676, mAP50-95=0.446; weights saved as `best.pt`. Classes: Person, boots, gloves, goggles, helmet, no_gloves, no_goggle, no_helmet, none, vest.

## Commands

The only script is the PDF→Markdown converter (requires `pdftotext`/`pdfinfo` from poppler-utils):

```bash
./scripts/pdfs_to_markdown.sh
```

Caveat: it globs `content/*.pdf`, but the PDFs now live in `content/pdfs(do not depend on)/`, so as-is it exits with "No PDFs". The `content/md/` conversions are already up to date; only fix the path (or move PDFs back) if a regeneration is actually needed.
