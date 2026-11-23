"""Build a same-class (real vs seven generators) example figure.

The generator filenames carry a numeric prefix that appears to be the ImageNet
class index. We select files for one target class from every generator, and for
Nature we use nature_metadata.csv (synset -> filename) to pick the same class.
The result is visually verified before being trusted.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "Dataset"
GENERATORS = ["ADM", "BigGAN", "glide", "Midjourney",
              "stable_diffusion_v_1_5", "VQDM", "wukong"]

# ImageNet index 1 == n01443537 (goldfish); index 0 == n01440764 (tench).
TARGET_INDEX = 1
TARGET_SYNSET = "01443537"
LABELS = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE",
          "Midjourney": "Midjourney", "stable_diffusion_v_1_5": "SD v1.5",
          "VQDM": "VQDM", "wukong": "Wukong"}


def class_index(name: str):
    m = re.match(r"^(\d+)_", name)                       # e.g. 001_sdv5_...
    if m:
        return int(m.group(1))
    m = re.match(r"^[A-Za-z]+_\d+_\d+_\d+_(\d+)_", name)  # GLIDE_1000_200_00_001_...
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the same-class real-vs-generated example figure.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "Report" / "latex" / "figures" / "dataset_examples.png")
    args = parser.parse_args()

    picks: dict[str, Path] = {}
    for g in GENERATORS:
        files = sorted(p for p in (ROOT / g).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg"})
        cand = [p for p in files if class_index(p.name) == TARGET_INDEX]
        picks[g] = cand[0] if cand else None
        print(f"{g:24s} idx={TARGET_INDEX} candidates={len(cand):4d} -> "
              f"{picks[g].name if picks[g] else 'NONE'}")

    nature = []
    with (ROOT / "nature_metadata.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and re.sub(r"\D", "", row[1]) == TARGET_SYNSET:
                nature.append(row[0])
    print(f"Nature synset={TARGET_SYNSET} matches={len(nature)} e.g. {nature[:3]}")

    order = ["Nature", *GENERATORS]
    images: dict[str, Path] = {}
    if nature:
        images["Nature"] = ROOT / "Nature" / nature[0]
    images.update({k: v for k, v in picks.items() if v})

    CELL = 300
    COLS, ROWS = 4, 2
    canvas = Image.new("RGB", (COLS * CELL, ROWS * CELL), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for i, name in enumerate(order):
        if name not in images:
            continue
        img = Image.open(images[name]).convert("RGB")
        img.thumbnail((CELL - 16, CELL - 56))
        r, c = divmod(i, COLS)
        canvas.paste(img, (c * CELL + (CELL - img.width) // 2,
                           r * CELL + 40 + (CELL - 56 - img.height) // 2))
        label = "Nature (real)" if name == "Nature" else LABELS.get(name, name)
        draw.text((c * CELL + 6, r * CELL + 5), label, fill="black", font=font)

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(200, 200))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
