"""Draw the end-to-end system architecture in an SCI-journal style.

The figure is authored at the report text width (6.5 in) so the 5-8 pt labels
stay legible when the LaTeX report includes it at \\linewidth.  The visual
language follows the supplied reference: dashed group panels with a tinted
title band, embedded example images, and arrows that show data flow.  Every
number is taken from Results/final_v3/summary.json or the report tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"
OUT = ROOT / "Report" / "figures_final" / "fig_system_architecture_v1.png"

# ---------------------------------------------------------------- palette
NAVY = "#0F2A44"
BLUE = "#2563EB"
AMBER = "#D97706"
TEAL = "#0F766E"
GRAY = "#5B6670"
MUTED = "#94A3B8"
PANEL = "#F8FAFC"
HEAD = "#E8EEF5"
WHITE = "#FFFFFF"
HAIR = "#DDE3EA"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.0,
})

W, H = 6.5, 4.32
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


# ---------------------------------------------------------------- helpers
def panel(x, y, w, h, title, accent=BLUE, title_size=8.2):
    """Dashed group panel with a tinted title band (reference-figure style)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=0.7, edgecolor=MUTED, facecolor=WHITE, linestyle="--", zorder=1))
    ax.add_patch(Rectangle((x, y), w, 0.24, linewidth=0, facecolor=HEAD, zorder=2))
    ax.add_patch(Rectangle((x, y), 0.04, 0.24, linewidth=0, facecolor=accent, zorder=3))
    ax.text(x + 0.085, y + 0.12, title, fontsize=title_size, color=NAVY, weight="bold",
            va="center", ha="left", zorder=3)


def box(x, y, w, h, text, edge=BLUE, face=PANEL, size=6.0, bold=False, dashed=False,
        color=NAVY, lw=0.7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw, edgecolor=edge, facecolor=face,
        linestyle="--" if dashed else "-", zorder=4))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=color,
            ha="center", va="center", weight="bold" if bold else "normal",
            linespacing=1.2, zorder=5)


def box_split(x, y, w, h, title, subtitle, edge=TEAL, face=PANEL, size=6.2,
              sub_size=4.6, bold=True, sub_color=GRAY, lw=0.7):
    """A box whose main label and small annotation both stay inside the frame."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=4))
    ax.text(x + w / 2, y + h / 2 - 0.06, title, fontsize=size, color=NAVY,
            ha="center", va="center", weight="bold" if bold else "normal", zorder=5)
    ax.text(x + w / 2, y + h / 2 + 0.075, subtitle, fontsize=sub_size, color=sub_color,
            ha="center", va="center", zorder=5)


def arrow(x1, y1, x2, y2, color=BLUE, lw=1.2, style="-|>", head=9, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=head,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=6))


def tile(path, x, y, size, label):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize((120, 120), Image.LANCZOS)
    ax.imshow(im, extent=(x, x + size, y + size, y), aspect="auto", zorder=4)
    ax.add_patch(Rectangle((x, y), size, size, fill=False, linewidth=0.5,
                           edgecolor=NAVY, zorder=5))
    ax.text(x + size / 2, y + size + 0.03, label, fontsize=4.4, color=GRAY,
            ha="center", va="top", zorder=5)


def first_image(folder):
    files = sorted(p for p in (DATA / folder).iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp"})
    return files[0]


# ================================================================ data panel
panel(0.10, 0.08, 2.25, 2.00, "DATA  \u00b7  Unbiased Tiny GenImage")
classes = [("Nature", "real"), ("ADM", "ADM"), ("BigGAN", "BigGAN"), ("glide", "GLIDE"),
           ("Midjourney", "MidJourney"), ("stable_diffusion_v_1_5", "SD 1.5"),
           ("VQDM", "VQDM"), ("wukong", "Wukong")]
size, gap = 0.31, 0.05
x0, y0 = 0.16, 0.38
for i, (folder, name) in enumerate(classes):
    col, row = i % 4, i // 4
    tile(first_image(folder), x0 + col * (size + gap), y0 + row * 0.48, size, name)

# data facts column (separated from the thumbnails by a hairline)
ax.plot([1.60, 1.60], [0.38, 1.20], color=HAIR, linewidth=0.7, zorder=3)
ax.text(1.67, 0.40, "23,328", fontsize=11.5, color=NAVY, weight="bold", va="top", zorder=5)
ax.text(1.67, 0.60, "total images", fontsize=5.2, color=GRAY, va="top", zorder=5)
ax.text(1.67, 0.78, "5,828 real", fontsize=6.0, color=NAVY, weight="bold", va="top", zorder=5)
ax.text(1.67, 1.00, "17,500", fontsize=7.0, color=NAVY, weight="bold", va="top", zorder=5)
ax.text(1.67, 1.14, "generated", fontsize=4.8, color=GRAY, va="top", zorder=5)

# 70/15/15 split bar
bx, by, bw, bh = 0.16, 1.39, 2.05, 0.09
ax.add_patch(Rectangle((bx, by), bw * 0.70, bh, facecolor=BLUE, linewidth=0, zorder=4))
ax.add_patch(Rectangle((bx + bw * 0.70, by), bw * 0.15, bh, facecolor="#7DA2F5",
                       linewidth=0, zorder=4))
ax.add_patch(Rectangle((bx + bw * 0.85, by), bw * 0.15, bh, facecolor=AMBER,
                       linewidth=0, zorder=4))
ax.text(bx + bw * 0.35, by + bh / 2, "train 70%", fontsize=4.9, color=WHITE,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(bx + bw * 0.775, by + bh / 2, "val", fontsize=4.7, color=NAVY,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(bx + bw * 0.925, by + bh / 2, "test", fontsize=4.7, color=NAVY,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(0.16, 1.52, "SHA-1 hash split  \u00b7  seed 5438  \u00b7  no file copies", fontsize=4.9,
        color=GRAY, va="top", zorder=5)

box(0.16, 1.63, 2.05, 0.12, "train:  SD 1.5  \u00b7  BigGAN  \u00b7  GLIDE",
    edge=BLUE, face="#EEF3FF", size=4.9, color=NAVY)
box(0.16, 1.77, 2.05, 0.12, "test only:  ADM  \u00b7  MidJourney  \u00b7  VQDM  \u00b7  Wukong",
    edge=AMBER, face="#FFF7ED", size=4.6, color=AMBER, dashed=True)

# ================================================================ preprocess
panel(2.50, 0.08, 0.88, 2.00, "PREPROCESS", accent=TEAL, title_size=7.8)
box(2.57, 0.42, 0.74, 0.30, "resize\n224 \u00d7 224", size=5.3)
box(2.57, 0.80, 0.74, 0.62, "six variants\nclean \u00b7 JPEG 90/50\nresize 0.5 \u00b7 blur 1\ncombo",
    size=4.8)
box(2.57, 1.50, 0.74, 0.30, "no train\u2013test\nleakage", size=4.8, edge=AMBER, dashed=True,
    color=AMBER)
ax.text(2.94, 1.86, "same files, same seed", fontsize=4.6, color=GRAY,
        ha="center", va="top", zorder=5)

# ================================================================ encoders
panel(3.53, 0.08, 1.52, 2.00, "FROZEN ENCODERS", accent=TEAL)
box_split(3.61, 0.40, 1.36, 0.44, "CLIP ViT-L/14  \u2192  768-d",
          "frozen  \u00b7  304.0M image tower")
box_split(3.61, 0.94, 1.36, 0.44, "DINOv2 ViT-L/14  \u2192  1,024-d",
          "frozen  \u00b7  304.4M parameters")
box(3.61, 1.48, 1.36, 0.20, "ablation:  + ResNet-18 (512-d)", edge=MUTED, face=WHITE,
    size=4.7, dashed=True, color=GRAY)
box(3.61, 1.76, 1.36, 0.24, "L2-normalise each  \u2192  concat 1,792-d", edge=BLUE,
    face="#EEF3FF", size=5.1, bold=True)

# ================================================================ probe
panel(5.20, 0.08, 1.20, 2.00, "LINEAR PROBE", accent=BLUE)
box(5.27, 0.42, 1.06, 0.36, "StandardScaler\nfit on train split only", size=5.0)
box(5.27, 0.84, 1.06, 0.38, "Logistic regression\n1,793 weights", size=5.7, bold=True)
box(5.27, 1.28, 1.06, 0.32, "sigmoid  \u2192  P(fake)\nthreshold 0.5", size=5.0,
    edge=AMBER, face="#FFF7ED")
ax.text(5.80, 1.66, "C and threshold from\nvalidation macro-F1", fontsize=4.7,
        color=GRAY, ha="center", va="top", linespacing=1.25, zorder=5)

# ================================================================ arrows top flow
arrow(2.36, 1.05, 2.49, 1.05)
arrow(3.39, 1.05, 3.52, 1.05)
arrow(5.06, 1.05, 5.19, 1.05)

# ================================================================ evaluation
panel(0.10, 2.26, 3.15, 1.62, "EVALUATION PROTOCOL", accent=AMBER)
rows = [
    ("P1  \u00b7  unseen-4", "95.9 vs 94.1", "+1.81 pp  \u00b7  CI +0.87 to +2.81", BLUE),
    ("LOGO  \u00b7  7 folds", "97.07 vs 95.57", "+1.50 pp  \u00b7  CI +0.86 to +2.18  \u00b7  7/7", BLUE),
    ("Robustness  \u00b7  7-gen", "JPEG-50 95.6 vs 92.0",
     "combo 92.0 vs 89.3  \u00b7  six variants", AMBER),
    ("AUC  \u00b7  P1", "99.54 vs 99.13", "macro-F1 95.9 vs 94.0", BLUE),
]
for i, (name, value, note, accent) in enumerate(rows):
    y = 2.54 + i * 0.30
    ax.add_patch(Rectangle((0.20, y), 0.035, 0.25, facecolor=accent, linewidth=0, zorder=3))
    ax.text(0.27, y + 0.012, name, fontsize=6.3, color=NAVY, weight="bold", va="top", zorder=5)
    ax.text(0.27, y + 0.145, note, fontsize=4.8, color=GRAY, va="top", zorder=5)
    ax.text(3.10, y + 0.11, value, fontsize=7.0, color=accent, weight="bold",
            ha="right", va="center", zorder=5)
ax.text(0.20, 3.74, "ours vs UnivFD  \u00b7  paired bootstrap 10,000  \u00b7  McNemar p = 4.2e-4 (P1)",
        fontsize=4.8, color=GRAY, va="top", zorder=5)

# ================================================================ deployment
panel(3.42, 2.26, 2.98, 1.62, "DEPLOYMENT AND REPRODUCIBILITY", accent=TEAL)
box(3.50, 2.56, 0.86, 0.40, "browser page\nupload / drag", size=4.9)
box(4.44, 2.56, 0.94, 0.40, "demo_server.py\nsame probe", size=4.9, edge=TEAL)
box(5.46, 2.56, 0.86, 0.40, "P(fake) + latency\n247\u2013356 ms", size=4.9,
    edge=AMBER, face="#FFF7ED")
arrow(4.37, 2.76, 4.43, 2.76)
arrow(5.39, 2.76, 5.45, 2.76)
box(3.50, 3.08, 1.24, 0.32, "features cached (.npz)", size=4.9, edge=MUTED, dashed=True,
    color=GRAY)
box(4.82, 3.08, 1.48, 0.32, "JSON \u2192 figures / report / slides", size=4.9, edge=MUTED,
    dashed=True, color=GRAY)
arrow(4.75, 3.24, 4.81, 3.24, color=MUTED)
ax.text(3.50, 3.52, "python Code/run_pipeline.py", fontsize=5.4, color=NAVY,
        weight="bold", va="top", zorder=5)
ax.text(3.50, 3.67, "audit \u2192 features \u2192 probe \u2192 robustness / LOGO \u2192 report \u2192 demo \u2192 slides",
        fontsize=4.6, color=GRAY, va="top", zorder=5)

# ================================================================ cross arrows
arrow(1.20, 2.09, 1.20, 2.25, lw=1.0)
arrow(5.80, 2.09, 5.80, 2.25, lw=1.0)
ax.text(1.25, 2.11, "test files", fontsize=4.5, color=GRAY, style="italic", va="top", zorder=5)
ax.text(5.85, 2.11, "serve", fontsize=4.5, color=GRAY, style="italic", va="top", zorder=5)

# ================================================================ footer note
ax.text(0.10, 4.03, "All encoders stay frozen; only the 1,793-weight linear probe is fitted. "
        "Unseen generators are never used for training, selection or thresholding.",
        fontsize=5.3, color=NAVY, va="top", style="italic", zorder=5)
ax.text(0.10, 4.20, "Schematic shows data flow, not causal architecture. Values are measured "
        "P1, LOGO and robustness results (7-generator mean, as in the robustness table).",
        fontsize=4.7, color=GRAY, va="top", zorder=5)

fig.savefig(OUT, dpi=300, facecolor=WHITE)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
