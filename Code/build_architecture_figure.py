"""Draw the end-to-end system architecture in an SCI-journal style.

The figure is authored at the report text width (6.4 in) so the 5-8 pt labels
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

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.0,
})

W, H = 6.4, 4.10
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


# ---------------------------------------------------------------- helpers
def panel(x, y, w, h, title, accent=BLUE):
    """Dashed group panel with a tinted title band (reference-figure style)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=0.7, edgecolor=MUTED, facecolor=WHITE, linestyle="--", zorder=1))
    ax.add_patch(Rectangle((x, y), w, 0.23, linewidth=0, facecolor=HEAD, zorder=2))
    ax.add_patch(Rectangle((x, y), 0.04, 0.23, linewidth=0, facecolor=accent, zorder=3))
    ax.text(x + 0.085, y + 0.115, title, fontsize=8.0, color=NAVY, weight="bold",
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


def arrow(x1, y1, x2, y2, color=BLUE, lw=1.2, style="-|>", lw_head=9, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=lw_head,
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
    ax.text(x + size / 2, y + size + 0.025, label, fontsize=4.6, color=GRAY,
            ha="center", va="top", zorder=5)


def first_image(folder):
    files = sorted(p for p in (DATA / folder).iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp"})
    return files[0]


# ================================================================ data panel
panel(0.10, 0.08, 2.14, 1.78, "DATA  \u00b7  Unbiased Tiny GenImage")
classes = [("Nature", "real"), ("ADM", "ADM"), ("BigGAN", "BigGAN"), ("glide", "GLIDE"),
           ("Midjourney", "MidJourney"), ("stable_diffusion_v_1_5", "SD 1.5"),
           ("VQDM", "VQDM"), ("wukong", "Wukong")]
size, gap = 0.29, 0.05
x0, y0 = 0.17, 0.37
for i, (folder, name) in enumerate(classes):
    col, row = i % 4, i // 4
    tile(first_image(folder), x0 + col * (size + gap), y0 + row * 0.46, size, name)

ax.text(0.17, 1.235, "5,828 real  +  7 \u00d7 2,500  =  23,328 images", fontsize=6.2,
        color=NAVY, va="top", weight="bold", zorder=5)

# 70/15/15 split bar
bx, by, bw, bh = 0.17, 1.35, 1.99, 0.085
ax.add_patch(Rectangle((bx, by), bw * 0.70, bh, facecolor=BLUE, linewidth=0, zorder=4))
ax.add_patch(Rectangle((bx + bw * 0.70, by), bw * 0.15, bh, facecolor="#7DA2F5",
                       linewidth=0, zorder=4))
ax.add_patch(Rectangle((bx + bw * 0.85, by), bw * 0.15, bh, facecolor=AMBER,
                       linewidth=0, zorder=4))
ax.text(bx + bw * 0.35, by + bh / 2, "train 70%", fontsize=4.8, color=WHITE,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(bx + bw * 0.775, by + bh / 2, "val", fontsize=4.6, color=NAVY,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(bx + bw * 0.925, by + bh / 2, "test", fontsize=4.6, color=NAVY,
        ha="center", va="center", weight="bold", zorder=5)
ax.text(0.17, 1.455, "SHA-1 hash split  \u00b7  seed 5438  \u00b7  no file copies", fontsize=4.9,
        color=GRAY, va="top", zorder=5)

box(0.17, 1.555, 1.99, 0.115, "train:  SD 1.5  \u00b7  BigGAN  \u00b7  GLIDE",
    edge=BLUE, face="#EEF3FF", size=4.9, bold=False, color=NAVY)
box(0.17, 1.69, 1.99, 0.115, "test only:  ADM  \u00b7  MidJourney  \u00b7  VQDM  \u00b7  Wukong",
    edge=AMBER, face="#FFF7ED", size=4.5, bold=False, color=AMBER, dashed=True)

# ================================================================ preprocess
panel(2.42, 0.08, 0.84, 1.78, "PRE-PROCESS", accent=TEAL)
box(2.49, 0.40, 0.70, 0.26, "resize\n224 \u00d7 224", size=5.2)
box(2.49, 0.72, 0.70, 0.52, "six variants\nclean \u00b7 JPEG 90/50\nresize 0.5 \u00b7 blur 1\ncombo",
    size=4.8)
box(2.49, 1.30, 0.70, 0.28, "no train\u2013test\nleakage", size=4.8, edge=AMBER, dashed=True,
    color=AMBER)
ax.text(2.84, 1.64, "same files and same seed\nfor every method", fontsize=4.6,
        color=GRAY, ha="center", va="top", linespacing=1.25, zorder=5)

# ================================================================ encoders
panel(3.44, 0.08, 1.50, 1.78, "FROZEN ENCODERS", accent=TEAL)
box(3.51, 0.38, 1.36, 0.30, "CLIP ViT-L/14  \u2192  768-d", edge=TEAL, size=6.0, bold=True)
ax.text(4.19, 0.69, "frozen  \u00b7  304.0M image tower (427.6M total)", fontsize=4.6,
        color=GRAY, ha="center", va="top", zorder=5)
box(3.51, 0.80, 1.36, 0.30, "DINOv2 ViT-L/14  \u2192  1,024-d", edge=TEAL, size=6.0, bold=True)
ax.text(4.19, 1.11, "frozen  \u00b7  304.4M parameters", fontsize=4.6,
        color=GRAY, ha="center", va="top", zorder=5)
box(3.51, 1.22, 1.36, 0.20, "ablation:  + ResNet-18 (512-d)", edge=MUTED, face=WHITE,
    size=4.6, dashed=True, color=GRAY)
box(3.51, 1.50, 1.36, 0.26, "L2-normalise each  \u2192  concat 1,792-d", edge=BLUE,
    face="#EEF3FF", size=5.4, bold=True)

# ================================================================ probe
panel(5.12, 0.08, 1.20, 1.78, "LINEAR PROBE", accent=BLUE)
box(5.19, 0.40, 1.06, 0.34, "StandardScaler\nfit on train split only", size=4.9)
box(5.19, 0.78, 1.06, 0.34, "Logistic regression\n1,793 weights", size=5.6, bold=True)
box(5.19, 1.16, 1.06, 0.30, "sigmoid  \u2192  P(fake)\nthreshold 0.5", size=5.0,
    edge=AMBER, face="#FFF7ED")
ax.text(5.72, 1.53, "C and threshold from\nvalidation macro-F1", fontsize=4.6,
        color=GRAY, ha="center", va="top", linespacing=1.25, zorder=5)

# ================================================================ arrows top flow
arrow(2.25, 0.97, 2.41, 0.97)
arrow(3.27, 0.97, 3.43, 0.97)
arrow(4.95, 0.97, 5.11, 0.97)

# ================================================================ evaluation
panel(0.10, 2.06, 3.08, 1.58, "EVALUATION PROTOCOL", accent=AMBER)
rows = [
    ("P1  \u00b7  unseen-4", "95.9 vs 94.1", "+1.81 pp  \u00b7  CI +0.87 to +2.81", BLUE),
    ("LOGO  \u00b7  7 folds", "97.07 vs 95.57", "+1.50 pp  \u00b7  CI +0.86 to +2.18  \u00b7  7/7", BLUE),
    ("Robustness  \u00b7  7-gen mean", "JPEG-50 95.6 vs 92.0",
     "combo 92.0 vs 89.3  \u00b7  six variants", AMBER),
]
for i, (name, value, note, accent) in enumerate(rows):
    y = 2.40 + i * 0.38
    ax.add_patch(Rectangle((0.20, y), 0.035, 0.30, facecolor=accent, linewidth=0, zorder=3))
    ax.text(0.27, y + 0.015, name, fontsize=6.3, color=NAVY, weight="bold", va="top", zorder=5)
    ax.text(0.27, y + 0.155, note, fontsize=4.9, color=GRAY, va="top", zorder=5)
    ax.text(3.03, y + 0.125, value, fontsize=7.0, color=accent, weight="bold",
            ha="right", va="center", zorder=5)
ax.text(0.20, 3.47, "ours vs UnivFD  \u00b7  paired bootstrap 10,000  \u00b7  McNemar p = 4.2e-4 (P1)",
        fontsize=4.9, color=GRAY, va="top", zorder=5)

# ================================================================ deployment
panel(3.34, 2.06, 2.98, 1.58, "DEPLOYMENT AND REPRODUCIBILITY", accent=TEAL)
box(3.42, 2.40, 0.84, 0.36, "browser page\nupload / drag", size=4.8)
box(4.34, 2.40, 0.92, 0.36, "demo_server.py\nsame probe", size=4.8, edge=TEAL)
box(5.34, 2.40, 0.90, 0.36, "P(fake) + latency\n247\u2013356 ms", size=4.8,
    edge=AMBER, face="#FFF7ED")
arrow(4.27, 2.58, 4.33, 2.58)
arrow(5.27, 2.58, 5.33, 2.58)
box(3.42, 2.86, 1.22, 0.28, "features cached (.npz)", size=4.8, edge=MUTED, dashed=True,
    color=GRAY)
box(4.74, 2.86, 1.50, 0.28, "JSON \u2192 figures / report / slides", size=4.8, edge=MUTED,
    dashed=True, color=GRAY)
arrow(4.65, 3.00, 4.73, 3.00, color=MUTED)
ax.text(3.42, 3.22, "python Code/run_pipeline.py", fontsize=5.4, color=NAVY,
        weight="bold", va="top", zorder=5)
ax.text(3.42, 3.38, "audit \u2192 features \u2192 probe \u2192 robustness / LOGO \u2192 report \u2192 demo \u2192 slides",
        fontsize=4.6, color=GRAY, va="top", zorder=5)

# ================================================================ cross arrows
arrow(1.17, 1.87, 1.17, 2.05, lw=1.0)
arrow(5.72, 1.87, 5.72, 2.05, lw=1.0)
ax.text(1.22, 1.90, "test files", fontsize=4.4, color=GRAY, style="italic", va="top", zorder=5)
ax.text(5.77, 1.90, "serve", fontsize=4.4, color=GRAY, style="italic", va="top", zorder=5)

# ================================================================ footer note
ax.text(0.10, 3.78, "All encoders stay frozen; only the 1,793-weight linear probe is fitted. "
        "Unseen generators are never used for training, selection or thresholding.",
        fontsize=5.2, color=NAVY, va="top", style="italic", zorder=5)
ax.text(0.10, 3.96, "Schematic shows data flow, not causal architecture. Values are measured "
        "P1, LOGO and robustness results (7-generator mean, as in the robustness table).",
        fontsize=4.6, color=GRAY, va="top", zorder=5)

fig.savefig(OUT, dpi=300, facecolor=WHITE)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
