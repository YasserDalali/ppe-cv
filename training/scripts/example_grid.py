"""Qualitative figure for the paper: plant frames, A (top row) vs. B (bottom row).

Each column is one frame, cropped around the worker so boxes and labels stay
readable at column width. Boxes are the predictions at confidence >= 0.25
(the operating point), A from its out-of-fold model, B from seed 0.

Inputs: original OCP frames (any folder layout; matched by file name without
the pool's "ocp__" prefix) and a small JSON {"A"|"B": {frame: {xyxy, cls,
conf}}} exported from preds_ocp.json on Colab.

    python scripts/example_grid.py --frames <dir> --preds example_preds.json \
        --out ../paper/figures/ocp_examples.jpg FRAME [FRAME ...]
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

COLORS = {"A": (0, 170, 0), "B": (210, 0, 0)}
TAGS = {"A": "A (ours, recording unseen)", "B": "B (construction only)"}
PANEL = (600, 800)  # width, height of each crop after resizing (3:4)


def crop_box(entries: dict, size: tuple[int, int], margin: float = 0.15) -> tuple:
    """Region around A's detections (plus any B box touching it), padded and
    widened or heightened to the 3:4 panel shape, kept inside the frame."""
    a = entries["A"]["xyxy"]
    x1, y1 = min(b[0] for b in a), min(b[1] for b in a)
    x2, y2 = max(b[2] for b in a), max(b[3] for b in a)
    for b in entries["B"]["xyxy"]:
        if b[0] < x2 and b[2] > x1 and b[1] < y2 and b[3] > y1:
            x1, y1, x2, y2 = min(x1, b[0]), min(y1, b[1]), max(x2, b[2]), max(y2, b[3])
    w, h = (x2 - x1) * (1 + 2 * margin), (y2 - y1) * (1 + 2 * margin)
    ratio = PANEL[0] / PANEL[1]
    w, h = max(w, h * ratio), max(h, w / ratio)
    W, H = size
    w, h = min(w, W), min(h, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - w / 2, 0), W - w)
    top = min(max(cy - h / 2, 0), H - h)
    return left, top, left + w, top + h


def panel(img: Image.Image, entry: dict, box: tuple, model: str) -> Image.Image:
    crop = img.crop(tuple(int(round(v)) for v in box)).resize(PANEL, Image.LANCZOS)
    sx = PANEL[0] / (box[2] - box[0])
    sy = PANEL[1] / (box[3] - box[1])
    d = ImageDraw.Draw(crop)
    font = ImageFont.load_default(size=34)
    color = COLORS[model]
    for (x1, y1, x2, y2), c, s in zip(entry["xyxy"], entry["cls"], entry["conf"]):
        r = ((x1 - box[0]) * sx, (y1 - box[1]) * sy, (x2 - box[0]) * sx, (y2 - box[1]) * sy)
        d.rectangle(r, outline=color, width=4)
        text = f"{c} {s:.2f}"
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
        ty = r[1] - th - 4 if r[1] - th - 4 > 0 else r[1] + 2
        tx = min(max(r[0], 0), PANEL[0] - tw - 4)
        d.rectangle((tx, ty, tx + tw + 4, ty + th + 4), fill=color)
        d.text((tx + 2, ty), text, fill="white", font=font)
    tag = ImageFont.load_default(size=32)
    tw, th = d.textbbox((0, 0), TAGS[model], font=tag)[2:]
    d.rectangle((0, PANEL[1] - th - 10, tw + 12, PANEL[1]), fill="black")
    d.text((6, PANEL[1] - th - 8), TAGS[model], fill="white", font=tag)
    return crop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("frame", nargs="+", help="frame names as in preds (with the ocp__ prefix)")
    args = ap.parse_args()

    preds = json.loads(args.preds.read_text())
    files = {p.name: p for p in args.frames.rglob("*.jpg")}
    gap = 12
    grid = Image.new("RGB", (len(args.frame) * (PANEL[0] + gap) - gap, 2 * PANEL[1] + gap), "white")
    for col, name in enumerate(args.frame):
        img = ImageOps.exif_transpose(Image.open(files[name.removeprefix("ocp__")])).convert("RGB")
        entries = {m: preds[m][name] for m in ("A", "B")}
        box = crop_box(entries, img.size)
        for row, model in enumerate(("A", "B")):
            grid.paste(panel(img, entries[model], box, model),
                       (col * (PANEL[0] + gap), row * (PANEL[1] + gap)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out, quality=90)
    print(f"wrote {args.out} ({grid.width}x{grid.height})")


if __name__ == "__main__":
    main()
