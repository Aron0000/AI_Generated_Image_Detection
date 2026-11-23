"""Compute real statistics from dataset/ and generate an example figure.

All numbers printed here are derived directly from the files on disk, so the
proposal can cite them without fabrication. Uses only Pillow (PIL); no matplotlib.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "Dataset"
CLASSES = ["Nature", "ADM", "BigGAN", "glide", "Midjourney",
           "stable_diffusion_v_1_5", "VQDM", "wukong"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG"}


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in EXTS)


def main() -> None:
    stats: dict = {}

    # --- per-class counts, extensions, sampled dimensions ---
    for c in CLASSES:
        files = image_files(ROOT / c)
        ext_counter = Counter(p.suffix for p in files)
        dims: set[tuple[int, int]] = set()
        for p in files[:200]:
            with Image.open(p) as im:
                dims.add(im.size)
        stats[c] = {
            "count": len(files),
            "extensions": dict(ext_counter),
            "sampled_unique_dims": len(dims),
            "dims_sample": sorted(dims)[:6],
        }

    # --- parse nature_metadata.csv (class field may hold a literal "\n") ---
    import csv

    meta_path = ROOT / "nature_metadata.csv"
    meta = {"rows": 0, "quality_counter": Counter(), "synsets": set(),
            "w": [], "h": []}
    if meta_path.exists():
        with meta_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 5:
                    continue
                cls = row[1]
                syn = re.sub(r"\D", "", cls.split("\n")[-1])
                try:
                    w = int(row[2]); h = int(row[3]); q = int(row[4])
                except ValueError:
                    continue
                meta["synsets"].add(syn)
                meta["quality_counter"][q] += 1
                meta["w"].append(w)
                meta["h"].append(h)
                meta["rows"] += 1
    meta["distinct_synsets"] = len(meta["synsets"])
    meta["quality_values"] = dict(meta["quality_counter"])
    meta["w_min"], meta["w_max"] = min(meta["w"]), max(meta["w"])
    meta["h_min"], meta["h_max"] = min(meta["h"]), max(meta["h"])

    # --- build example figure with Pillow: 2 x 4 grid, one sample per class ---
    CELL = 320
    COLS, ROWS = 4, 2
    canvas = Image.new("RGB", (COLS * CELL, ROWS * CELL), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for idx, c in enumerate(CLASSES):
        files = image_files(ROOT / c)
        img = Image.open(files[0]).convert("RGB")
        img.thumbnail((CELL, CELL))
        r, col = divmod(idx, COLS)
        x = col * CELL + (CELL - img.width) // 2
        y = r * CELL + (CELL - img.height) // 2 + 24
        canvas.paste(img, (x, y))
        draw.text((col * CELL + 6, r * CELL + 4), c, fill="black", font=font)

    out_png = Path(__file__).resolve().parent.parent / "Results" / "dataset_examples.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, dpi=(200, 200))
    print("Saved figure:", out_png.resolve())

    print("\n=== DATASET STATS (real, from disk) ===")
    meta_out = {k: (sorted(v) if isinstance(v, set) else v) for k, v in meta.items()}
    print(json.dumps({"classes": stats, "nature_metadata": meta_out},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
