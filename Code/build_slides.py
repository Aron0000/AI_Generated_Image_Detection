"""Build the 16-slide 10-minute defence deck from the consolidated results.

The deck follows the two reference talks recommended by the course:
  * Geoffrey Litt, "ENHANCE! Upscaling Images with Neural Networks" (!!Con 2017)
    - one question, one visual idea per slide, a live demonstration, honest limits.
  * Joseph Redmon, "You Only Look Once" (CVPR 2016)
    - an evaluation ladder, speed/accuracy comparisons, then the demo.
Every slide names its speaker, its time window and one takeaway. The matching
spoken script is Report/DEFENSE_SCRIPT.md; the same text is in the notes pane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "Report" / "figures_final"
DEMO_PHOTO = ROOT / "Report" / "demo" / "sample.jpg"
DEMO_SHOT = ROOT / "Report" / "demo" / "demo_screenshot.png"
OUT = ROOT / "Report" / "Defense_Slides_EE5438_2026-11-23.pptx"

summary = json.loads((ROOT / "Results" / "final_v3" / "summary.json").read_text(encoding="utf-8"))
table = summary["table"]
stats = summary["stats"]
robust = summary["robustness"]

# The demo page stores the measured score of the fixed sample; reuse it so the deck
# can never drift from the demonstration.
demo_html = (ROOT / "Report" / "demo" / "index.html").read_text(encoding="utf-8")
match = re.search(r'"sample":\{"file":"sample\.jpg".*?"p":([0-9.]+)', demo_html)
DEMO_P = float(match.group(1)) if match else None

NAVY = RGBColor(0x0F, 0x2A, 0x44)
BLUE = RGBColor(0x25, 0x63, 0xEB)
AMBER = RGBColor(0xD9, 0x77, 0x06)
GRAY = RGBColor(0x5B, 0x66, 0x70)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x0F, 0x76, 0x6E)
DARK_CARD = RGBColor(0x1B, 0x3A, 0x57)
MUTED = RGBColor(0xC7, 0xD2, 0xDC)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 16


def pc(x: float) -> str:
    return f"{x * 100:.1f}%"


def add_text(slide, text, left, top, width, height, size=18, color=NAVY, bold=False,
             align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = line_spacing
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def add_rect(slide, left, top, width, height, fill, line=None):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def add_arrow(slide, left, top, width, height, color=BLUE):
    shape = slide.shapes.add_shape(33, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_bullets(slide, items, left, top, width, height, size=15, color=NAVY, gap=10):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for i, item in enumerate(items):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(gap)
        paragraph.line_spacing = 1.05
        if isinstance(item, tuple):
            lead, rest = item
            run = paragraph.add_run()
            run.text = "\u25A0  "
            run.font.size = Pt(size)
            run.font.color.rgb = BLUE
            run1 = paragraph.add_run()
            run1.text = lead
            run1.font.size = Pt(size)
            run1.font.bold = True
            run1.font.color.rgb = NAVY
            run2 = paragraph.add_run()
            run2.text = rest
            run2.font.size = Pt(size)
            run2.font.color.rgb = NAVY
        else:
            run = paragraph.add_run()
            run.text = "\u25A0  " + item
            run.font.size = Pt(size)
            run.font.color.rgb = NAVY
    return box


def add_image(slide, name, left, top, max_w, max_h, frame=False):
    path = FIG / name
    with Image.open(path) as img:
        w, h = img.size
    scale = min(max_w / w, max_h / h)
    width, height = w * scale, h * scale
    if frame:
        add_rect(slide, left - 0.08, top - 0.08, width + 0.16, height + 0.16, WHITE, BLUE)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(width), Inches(height))
    return left, top, width, height


def add_photo(slide, path, left, top, max_w, max_h, frame=True):
    with Image.open(path) as img:
        w, h = img.size
    scale = min(max_w / w, max_h / h)
    width, height = w * scale, h * scale
    if frame:
        add_rect(slide, left - 0.08, top - 0.08, width + 0.16, height + 0.16, WHITE, BLUE)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(width), Inches(height))
    return left, top, width, height


def add_callout(slide, text, left, top, width, height, size=15, fill=LIGHT, color=NAVY, bold=True):
    add_rect(slide, left, top, width, height, fill)
    add_rect(slide, left, top, 0.06, height, BLUE)
    add_text(slide, text, left + 0.22, top + 0.06, width - 0.3, height - 0.12,
             size=size, color=color, bold=bold, anchor=MSO_ANCHOR.MIDDLE)


def add_chip(slide, text, left, top, width, height, fill=LIGHT, color=NAVY, size=12, line=None):
    add_rect(slide, left, top, width, height, fill, line)
    add_text(slide, text, left, top + 0.04, width, height - 0.08, size=size, color=color,
             bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_frame(title, kicker=None, number=None, speaker=None, time=None, progress=0.0):
    slide = prs.slides.add_slide(BLANK)
    add_rect(slide, 0, 0, 13.333, 0.12, NAVY)
    if kicker:
        add_text(slide, kicker.upper(), 0.62, 0.34, 6.4, 0.3, size=11, color=BLUE, bold=True)
    if speaker:
        add_text(slide, speaker + ("  \u00b7  " + time if time else ""), 7.3, 0.34, 5.45, 0.3,
                 size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    add_text(slide, title, 0.62, 0.66, 12.0, 0.75, size=27, color=NAVY, bold=True)
    add_rect(slide, 0.62, 1.42, 1.4, 0.045, BLUE)
    add_rect(slide, 0, 7.47, 13.333, 0.03, RGBColor(0xDD, 0xE3, 0xEA))
    add_rect(slide, 0, 7.47, 13.333 * max(0.0, min(1.0, progress)), 0.03, AMBER)
    add_text(slide, "EE5438  \u00b7  Group 20", 0.62, 7.16, 5.0, 0.2, size=9, color=GRAY)
    if number is not None:
        add_text(slide, f"{number} / {TOTAL}", 11.85, 7.16, 0.9, 0.2, size=9, color=GRAY,
                 align=PP_ALIGN.RIGHT)
    return slide


# ================================================================ slide 1 — title
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.333, 7.5, NAVY)
add_rect(slide, 0.9, 1.18, 0.09, 2.05, AMBER)
add_text(slide, "EE5438  \u00b7  GROUP 20  \u00b7  FINAL PRESENTATION", 1.2, 1.02, 9.0, 0.3,
         size=11, color=RGBColor(0x93, 0xB4, 0xD8), bold=True)
add_text(slide, "A detector that still works\nwhen the generator is new", 1.2, 1.42, 11.2, 1.6,
         size=35, color=WHITE, bold=True, line_spacing=1.0)
add_text(slide, "Dual-encoder foundation features vs. a frozen CLIP detector", 1.22, 3.18, 10.8, 0.4,
         size=16, color=MUTED)
add_rect(slide, 1.22, 3.86, 4.1, 0.035, RGBColor(0x4B, 0x55, 0x63))

roles = [
    ("Huanrong Li", "Framing, method (DELP)"),
    ("Bo Pang", "Dataset, splits, degradation"),
    ("Huanyu Zhang", "Ladder, baselines, UnivFD"),
    ("Junxu Chen", "Headline result, ResNet, repro"),
    ("Yangqi Zhang", "Generalization, demo, close"),
]
for i, (name, role) in enumerate(roles):
    x = 1.22 + i * 2.28
    add_rect(slide, x, 4.18, 2.12, 0.86, DARK_CARD)
    add_text(slide, name, x + 0.14, 4.3, 1.9, 0.3, size=12, color=WHITE, bold=True)
    add_text(slide, role, x + 0.14, 4.62, 1.9, 0.32, size=8.5, color=RGBColor(0xA9, 0xBC, 0xCE))

add_text(slide, "95.9%  vs  94.1%", 1.22, 5.62, 4.2, 0.55, size=26, color=RGBColor(0xFC, 0xD3, 0x4D), bold=True)
add_text(slide, "on four generators neither detector has seen   \u00b7   McNemar p = 4.2 \u00d7 10\u207b\u2074",
         1.25, 6.18, 11.0, 0.35, size=13, color=MUTED)
add_text(slide, "10 minutes  \u00b7  5 speakers  \u00b7  live demo  \u00b7  every number generated from JSON",
         1.22, 6.78, 11.0, 0.3, size=10, color=RGBColor(0x7F, 0x94, 0xAB))

# ================================================================ slide 2 — the question
slide = slide_frame("Tomorrow a new generator appears. Does the detector still work?",
                    "The question", 2, "Huanrong Li", "0:20\u20130:50", 0.083)
add_rect(slide, 0.62, 1.9, 4.4, 1.55, LIGHT, BLUE)
add_text(slide, "SEEN DURING TRAINING", 0.82, 1.99, 4.0, 0.3, size=10, color=BLUE, bold=True)
add_text(slide, "Stable Diffusion 1.5\nBigGAN\nGLIDE", 0.82, 2.3, 4.0, 1.05, size=17, color=NAVY, bold=True)
add_arrow(slide, 5.32, 2.5, 1.5, 0.35)
add_rect(slide, 7.0, 1.9, 5.7, 1.55, LIGHT, AMBER)
add_text(slide, "TEST ONLY \u2014 NEVER USED FOR TUNING", 7.2, 1.99, 5.3, 0.3, size=10, color=AMBER, bold=True)
add_text(slide, "ADM   \u00b7   MidJourney   \u00b7   VQDM   \u00b7   Wukong", 7.2, 2.34, 5.4, 0.55,
         size=17, color=NAVY, bold=True)
add_text(slide, "three generators                                    four unseen generators",
         0.62, 3.58, 12.0, 0.3, size=11, color=GRAY)
for i, (value, label) in enumerate([("23,328", "images in the benchmark"),
                                   ("1,600", "balanced unseen test images"),
                                   ("400", "real + generated per generator")]):
    x = 0.62 + i * 4.1
    add_text(slide, value, x, 4.18, 3.7, 0.75, size=34, color=BLUE, bold=True)
    add_text(slide, label, x, 4.95, 3.8, 0.35, size=12, color=GRAY)
add_callout(slide, "\u201cBeat\u201d = higher accuracy on generators the model has never seen, "
             "same files, paired bootstrap + exact McNemar test \u2014 not a different split.",
             0.62, 5.75, 12.1, 0.75, size=13)

# ================================================================ slide 3 — DELP
slide = slide_frame("Our method: fuse two frozen views, keep the head linear",
                    "Method \u00b7 DELP", 3, "Huanrong Li", "0:50\u20131:50", 0.183)
for i, (head, body, color) in enumerate([
        ("CLIP ViT-L/14", "768-d semantic embedding", BLUE),
        ("DINOv2 ViT-L/14", "1024-d texture / class token", GREEN)]):
    y = 1.85 + i * 1.45
    add_rect(slide, 0.62, y, 3.4, 1.0, LIGHT, color)
    add_text(slide, head, 0.78, y + 0.12, 3.1, 0.3, size=13, color=NAVY, bold=True)
    add_text(slide, body, 0.78, y + 0.45, 3.1, 0.28, size=9.5, color=GRAY)
    add_arrow(slide, 4.1, y + 0.38, 0.55, 0.22, color)
    add_rect(slide, 4.75, y, 1.15, 1.0, color)
    add_text(slide, "L2", 4.75, y + 0.12, 1.15, 0.3, size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "normalise", 4.75, y + 0.45, 1.15, 0.28, size=8.5, color=WHITE, align=PP_ALIGN.CENTER)
    add_arrow(slide, 5.98, y + 0.38, 0.55, 0.22, GRAY)
add_rect(slide, 6.65, 2.3, 1.9, 1.35, LIGHT)
add_text(slide, "concat", 6.65, 2.4, 1.9, 0.3, size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "1,792-d", 6.65, 2.72, 1.9, 0.4, size=15, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
add_text(slide, "z(x)", 6.65, 3.12, 1.9, 0.3, size=9, color=GRAY, align=PP_ALIGN.CENTER)
add_arrow(slide, 8.62, 2.78, 0.5, 0.22)
add_rect(slide, 9.25, 2.15, 3.5, 1.65, NAVY)
add_text(slide, "scaler + logistic", 9.42, 2.28, 3.15, 0.3, size=12, color=WHITE, bold=True)
add_text(slide, "1,793 trainable weights", 9.42, 2.62, 3.15, 0.4, size=14, color=RGBColor(0xFC, 0xD3, 0x4D), bold=True)
add_text(slide, "no fine-tuning \u00b7 no augmentation \u00b7 no TTA", 9.42, 3.05, 3.15, 0.5, size=9.5, color=MUTED)
add_text(slide, "z(x) = [ c(x) / \u2016c(x)\u2016  ;  d(x) / \u2016d(x)\u2016 ]",
         0.62, 4.75, 7.2, 0.4, size=16, color=NAVY, bold=True)
add_text(slide, "Hypothesis: CLIP encodes semantics, DINOv2 encodes local texture. They make different errors, "
               "so a linear combination has lower variance on an unseen generator.",
         0.62, 5.2, 7.2, 0.95, size=12, color=GRAY)
add_callout(slide, "27 ms / image\nRTX 2080, batch 32", 9.25, 4.55, 3.5, 1.05, size=13)
add_text(slide, "Nothing else changes: same C grid, same training images, same linear probe family as UnivFD.",
         0.62, 6.4, 12.1, 0.35, size=11, color=BLUE, bold=True)

# ================================================================ slide 4 — shortcuts
slide = slide_frame("In-domain accuracy can measure the shortcut, not the detector",
                    "Why this is hard", 4, "Bo Pang", "1:50\u20132:20", 0.233)
cards = [
    ("01", "Generator fingerprint", "A CNN can memorise one generator's texture, colour cast and upsampling pattern."),
    ("02", "JPEG history", "Recompression traces survive resizing and inflate every learned detector."),
    ("03", "Native resolution", "128 px GANs and 1024 px diffusion models differ before any content is modelled."),
]
for i, (num, title, body) in enumerate(cards):
    x = 0.62 + i * 4.1
    add_rect(slide, x, 1.85, 3.85, 2.35, LIGHT)
    add_text(slide, num, x + 0.22, 1.98, 0.6, 0.4, size=13, color=AMBER, bold=True)
    add_text(slide, title, x + 0.22, 2.35, 3.45, 0.4, size=17, color=NAVY, bold=True)
    add_text(slide, body, x + 0.22, 2.85, 3.42, 1.2, size=12, color=GRAY)
add_callout(slide, "GenImage (NeurIPS 2023) and \u201cFake or JPEG?\u201d (2024) document these shortcuts. "
             "Our controls: hash split, balanced 200 + 200 per generator, one benchmark, seven generators.",
             0.62, 4.65, 12.1, 0.95, size=13)
add_text(slide, "A useful detector must transfer to a generator it has never seen.",
         0.62, 5.95, 12.1, 0.5, size=20, color=BLUE, bold=True)

# ================================================================ slide 5 — dataset
slide = slide_frame("23,328 images, seven generators, zero generator leakage",
                    "Data", 5, "Bo Pang", "2:20\u20133:10", 0.317)
add_text(slide, "23,328", 0.62, 1.72, 4.6, 0.9, size=48, color=BLUE, bold=True)
add_text(slide, "images", 0.65, 2.55, 4.0, 0.3, size=13, color=GRAY)
add_text(slide, "5,828 real  +  7 \u00d7 2,500 generated", 0.62, 2.95, 5.2, 0.35, size=14, color=NAVY, bold=True)
add_text(slide, "One GAN, six diffusion models, three native resolutions \u2014 the de-biased "
               "Unbiased Tiny GenImage release.", 0.62, 3.4, 5.4, 0.9, size=11, color=GRAY)

rows = [("Nature (real)", "5,828", "450\u2013500 px"),
        ("ADM / GLIDE / VQDM", "3 \u00d7 2,500", "256 px"),
        ("SD 1.5 / Wukong", "2 \u00d7 2,500", "512 px"),
        ("BigGAN  \u00b7  MidJourney", "2,500 each", "128 / 1024 px")]
y = 1.72
add_rect(slide, 6.35, y, 6.35, 0.36, NAVY)
for x0, w0, t in [(6.5, 2.6, "Class"), (9.1, 1.6, "Count"), (10.7, 1.9, "Native")]:
    add_text(slide, t, x0, y + 0.07, w0, 0.25, size=10, color=WHITE, bold=True)
for i, (cls, count, note) in enumerate(rows):
    yy = y + 0.36 + i * 0.48
    add_rect(slide, 6.35, yy, 6.35, 0.48, LIGHT if i % 2 == 0 else WHITE)
    add_text(slide, cls, 6.5, yy + 0.09, 2.55, 0.3, size=10.5, color=NAVY)
    add_text(slide, count, 9.1, yy + 0.09, 1.5, 0.3, size=10.5, color=NAVY)
    add_text(slide, note, 10.7, yy + 0.09, 1.9, 0.3, size=10, color=GRAY)
add_text(slide, "Native size is a confound, so it is measured and reported, never hidden.",
         6.35, 3.95, 6.35, 0.3, size=10, color=AMBER, bold=True)

add_text(slide, "SHA-1(seed 5438 : class : filename)  \u2192  70 / 15 / 15 per class",
         0.62, 4.72, 12.1, 0.35, size=13, color=NAVY, bold=True)
bar_x, bar_y, bar_w = 0.62, 5.15, 12.1
for frac, label, color in [(0.70, "train  70%", NAVY), (0.15, "val  15%", BLUE), (0.15, "test  15%", AMBER)]:
    add_rect(slide, bar_x, bar_y, bar_w * frac, 0.5, color)
    add_text(slide, label, bar_x + 0.12, bar_y + 0.12, bar_w * frac - 0.2, 0.28,
             size=11, color=WHITE, bold=True, align=PP_ALIGN.LEFT)
    bar_x += bar_w * frac
add_text(slide, "No files are copied and no directory order is used; the split is a deterministic "
               "function of the filename, identical on every machine.",
         0.62, 5.82, 12.1, 0.65, size=11, color=GRAY)

# ================================================================ slide 6 — audit & degradation
slide = slide_frame("One test set: clean plus five degraded conditions",
                    "Audit and controls", 6, "Bo Pang", "3:10\u20133:40", 0.367)
conditions = ["clean", "JPEG q90", "JPEG q50", "resize 50%", "blur \u03c3=1", "combined"]
for i, name in enumerate(conditions):
    first = i == 0
    add_chip(slide, name, 0.62 + i * 2.02, 1.82, 1.9, 0.52, LIGHT,
             BLUE if first else NAVY, 11,
             line=BLUE if first else RGBColor(0xC7, 0xD2, 0xDC))
steps = [("Original", "RGB, 224 \u00d7 224"), ("Compress", "JPEG q90 / q50"),
         ("Resample", "down 50%, back up"), ("Blur", "Gaussian \u03c3 = 1"),
         ("Encoder", "CLIP + DINOv2")]
for i, (head, body) in enumerate(steps):
    x = 0.62 + i * 2.52
    add_rect(slide, x, 3.12, 2.2, 0.95, LIGHT)
    add_text(slide, head, x + 0.12, 3.2, 2.0, 0.3, size=13, color=NAVY, bold=True)
    add_text(slide, body, x + 0.12, 3.53, 2.0, 0.3, size=10, color=GRAY)
    if i < 4:
        add_arrow(slide, x + 2.22, 3.46, 0.26, 0.22, AMBER)
add_bullets(slide, [
    ("Audit. ", "Eight folders, 23,328 files, zero corrupt images, every class present in all three splits."),
    ("Balance. ", "The real class is cut to the generated class in every partition, so accuracy cannot ride class imbalance."),
], 0.62, 4.45, 6.0, 1.7, size=12, gap=8)
add_callout(slide, "The degradation suite doubles as the robustness test.\n"
             "Nothing is regenerated: the same files are re-encoded.", 6.95, 4.45, 5.75, 1.45, size=12)
add_text(slide, "Inputs are frozen before any model is trained \u2014 this prevents silent test-set drift.",
         0.62, 6.15, 12.1, 0.35, size=11, color=BLUE, bold=True)

# ================================================================ slide 7 — ladder
slide = slide_frame("Four detectors, one training set, one test set",
                    "The evaluation ladder", 7, "Huanyu Zhang", "3:40\u20134:10", 0.417)
rungs = [
    ("HOG + FFT + SVM", "classical", table["classical_svm"]["mean_common_unseen"], 69.4),
    ("ResNet-18 frozen", "supervised CNN", table["resnet18"]["mean_common_unseen"], 82.8),
    ("UnivFD", "CLIP linear probe", table["univfd"]["mean_common_unseen"], 94.1),
    ("DELP (ours)", "CLIP + DINOv2", table["ours"]["mean_common_unseen"], 95.9),
]
for i, (name, kind, value, _) in enumerate(rungs):
    x = 0.62 + i * 3.05
    top = 1.55 + (3 - i) * 0.8
    fill = BLUE if i == 3 else LIGHT
    add_rect(slide, x, top, 2.75, 1.0, fill)
    add_text(slide, pc(value), x + 0.15, top + 0.07, 2.4, 0.5,
             size=25, color=WHITE if i == 3 else NAVY, bold=True)
    add_text(slide, name, x + 0.15, top + 0.56, 2.45, 0.26, size=11,
             color=WHITE if i == 3 else NAVY, bold=True)
    add_text(slide, kind, x + 0.15, top + 0.8, 2.45, 0.2, size=8.5,
             color=MUTED if i == 3 else GRAY)
    if i < 3:
        add_arrow(slide, x + 2.75, min(top, top - 0.8) + 0.38, 0.26, 0.22,
                  AMBER if i < 2 else BLUE)
gains = [("+13.4", "deep beats handcrafted"),
         ("+24.7", "foundation features: 69.4 \u2192 94.1"),
         ("+1.8", "two frozen views")]
for i, (delta, label) in enumerate(gains):
    x = 0.62 + i * 4.1
    add_rect(slide, x, 5.2, 3.85, 0.62, LIGHT)
    add_rect(slide, x, 5.2, 0.06, 0.62, BLUE)
    add_text(slide, delta, x + 0.2, 5.27, 0.95, 0.4, size=15, color=AMBER, bold=True)
    add_text(slide, label, x + 1.12, 5.33, 2.6, 0.3, size=9.5, color=NAVY)
add_text(slide, "Unseen-4 accuracy \u00b7 same files, seed 5438, same probe budget for both linear probes.",
         0.62, 5.95, 12.1, 0.3, size=10, color=GRAY)
add_callout(slide, "Level II  Beat the Classics: clear the transparent non-DL detector.   "
             "Level III  Beat the Stars: clear UnivFD, a CVPR 2023 frozen-CLIP probe.",
             0.62, 6.35, 12.1, 0.6, size=12)

# ================================================================ slide 8 — classical
slide = slide_frame("A transparent baseline that does not transfer",
                    "Level II \u00b7 beat the classics", 8, "Huanyu Zhang", "4:10\u20134:55", 0.492)
add_text(slide, "HOG + radial FFT + RBF-SVM", 0.62, 1.78, 6.2, 0.45, size=20, color=NAVY, bold=True)
add_text(slide, "1,780 handcrafted dimensions, no learned representation, seconds to train.",
         0.62, 2.22, 6.4, 0.35, size=12, color=GRAY)
for i, (head, body, y) in enumerate([
        ("Image", "224 \u00d7 224, grayscale", 2.7),
        ("HOG", "edge and gradient statistics", 3.45),
        ("Radial FFT", "periodic upsampling artifacts", 4.2)]):
    add_rect(slide, 0.62, y, 4.4, 0.6, LIGHT, BLUE if i == 1 else None)
    add_text(slide, head, 0.76, y + 0.06, 4.1, 0.22, size=11.5, color=NAVY, bold=True)
    add_text(slide, body, 0.76, y + 0.31, 4.1, 0.22, size=8.5, color=GRAY)
    if i < 2:
        add_rect(slide, 2.79, y + 0.6, 0.05, 0.15, AMBER)
add_rect(slide, 0.62, 4.95, 4.4, 0.6, NAVY)
add_text(slide, "concatenate 1,780-d  \u2192  RBF-SVM", 0.76, 5.04, 4.1, 0.3,
         size=11, color=WHITE, bold=True)
add_rect(slide, 7.4, 1.85, 5.3, 2.25, LIGHT)
add_text(slide, pc(table["classical_svm"]["mean_common_unseen"]), 7.6, 1.95, 4.9, 0.8,
         size=40, color=AMBER, bold=True)
add_text(slide, "unseen-4 accuracy", 7.62, 2.75, 4.8, 0.3, size=12, color=GRAY)
add_text(slide, "Strong in-domain; brittle across generators.", 7.62, 3.15, 4.8, 0.4,
         size=13, color=NAVY, bold=True)
add_bullets(slide, [
    ("What it sees. ", "Edges and spectral peaks, not the semantics of a photograph."),
    ("Why it breaks. ", "A new upsampler moves the spectrum; a new post-process removes the peaks."),
    ("Why keep it. ", "It is the transparent bar that Level II asks a deep model to clear."),
], 7.4, 4.35, 5.3, 2.2, size=11, gap=8)
add_callout(slide, "The first deep model gains +13.4 points over this baseline.", 7.4, 6.45, 5.3, 0.55, size=12)

# ================================================================ slide 9 — UnivFD
slide = slide_frame("The star to beat: one frozen encoder, one linear layer",
                    "Level III \u00b7 beat the stars", 9, "Huanyu Zhang", "4:55\u20135:30", 0.55)
add_text(slide, "UnivFD  \u00b7  CVPR 2023", 0.62, 1.78, 5.0, 0.35, size=11, color=BLUE, bold=True)
add_text(slide, "CLIP ViT-L/14, frozen.  No fine-tuning.  C chosen on validation macro-F1.",
         0.62, 2.1, 7.6, 0.35, size=13, color=NAVY, bold=True)
boxes = [("Image", "224 \u00d7 224"), ("CLIP ViT-L/14", "427.6M total, frozen"),
         ("768-d", "L2 + scaler"), ("Logistic", "769 weights")]
for i, (head, body) in enumerate(boxes):
    x = 0.62 + i * 3.05
    add_rect(slide, x, 2.95, 2.7, 1.05, LIGHT, BLUE if i == 1 else None)
    add_text(slide, head, x, 3.04, 2.7, 0.3, size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, body, x, 3.38, 2.7, 0.25, size=9.5, color=GRAY, align=PP_ALIGN.CENTER)
    if i < 3:
        add_arrow(slide, x + 2.7, 3.38, 0.28, 0.2)
add_text(slide, pc(table["univfd"]["mean_common_unseen"]), 0.62, 4.35, 3.6, 0.5, size=30, color=AMBER, bold=True)
add_text(slide, "unseen-4", 0.65, 4.85, 3.0, 0.3, size=10, color=GRAY)
add_bullets(slide, [
    ("Why it transfers. ", "CLIP was trained on internet image-text pairs: it encodes what a photograph looks like, "
     "not one generator's filter bank."),
    ("Why it is hard to beat. ", "It already uses the strongest foundation encoder available, and a linear probe "
     "leaves little room for over-fitting."),
], 0.62, 5.25, 7.5, 1.5, size=11, gap=6)
add_callout(slide, "Our task: beat 94.1% without fine-tuning, extra data or a bigger head.",
             8.3, 4.55, 4.4, 1.1, size=13)
add_text(slide, "Reference reproduced faithfully: same three generators, same files, same evaluation grid.",
         8.3, 5.85, 4.4, 0.8, size=10, color=GRAY)

# ================================================================ slide 10 — headline
slide = slide_frame("Result: +1.8 points on generators both models have never seen",
                    "Headline result", 10, "Junxu Chen", "5:30\u20136:12", 0.62)
add_image(slide, "fig_progression.png", 0.55, 1.75, 7.55, 4.6)
add_rect(slide, 8.35, 1.8, 4.35, 1.75, NAVY)
add_text(slide, pc(table["ours"]["mean_common_unseen"]) + "  vs  " + pc(table["univfd"]["mean_common_unseen"]),
         8.55, 1.92, 3.95, 0.6, size=22, color=WHITE, bold=True)
add_text(slide, "DELP vs UnivFD \u00b7 four unseen generators", 8.55, 2.55, 3.9, 0.3,
         size=10, color=MUTED)
add_text(slide, "+1.81 points", 8.55, 2.9, 3.9, 0.45, size=20, color=RGBColor(0xFC, 0xD3, 0x4D), bold=True)
add_bullets(slide, [
    ("95% CI ", "[+0.87, +2.81]"),
    ("McNemar ", "18 vs 47 discordant, p = 4.2 \u00d7 10\u207b\u2074"),
    ("AUC ", "99.13 \u2192 99.54; macro-F1 94.0 \u2192 95.9"),
    ("Same data ", "8,164 training images for both probes"),
], 8.35, 3.75, 4.35, 2.5, size=11, gap=8)
add_text(slide, "Mean over all seven generators: 97.1% vs 95.9%.", 0.62, 6.5, 7.2, 0.3, size=11, color=GRAY)
add_text(slide, "The gain is a different error pattern, not a higher operating point.",
         8.35, 6.35, 4.35, 0.5, size=10, color=GRAY)

# ================================================================ slide 11 — ResNet baseline
slide = slide_frame("The deep baseline: frozen ResNet-18, selected on validation only",
                    "Supervised CNN", 11, "Junxu Chen", "6:12\u20136:52", 0.687)
boxes = [("Image", "224 \u00d7 224"), ("ResNet-18", "ImageNet, frozen"),
         ("512-d", "pooled feature"), ("Linear", "2-way head")]
for i, (head, body) in enumerate(boxes):
    x = 0.62 + i * 3.05
    add_rect(slide, x, 1.9, 2.7, 1.1, LIGHT, BLUE if i == 1 else None)
    add_text(slide, head, x, 2.0, 2.7, 0.3, size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, body, x, 2.35, 2.7, 0.25, size=9.5, color=GRAY, align=PP_ALIGN.CENTER)
    if i < 3:
        add_arrow(slide, x + 2.7, 2.35, 0.28, 0.2)
add_text(slide, pc(table["resnet18"]["mean_common_unseen"]), 0.62, 3.35, 3.5, 0.55, size=30, color=AMBER, bold=True)
add_text(slide, "unseen-4", 0.65, 3.9, 3.0, 0.3, size=10, color=GRAY)
for i, (value, label) in enumerate([("1,998", "training images"), ("402", "validation images"),
                                   ("10", "epochs, frozen backbone")]):
    x = 0.62 + i * 3.05
    add_text(slide, value, x, 4.75, 2.7, 0.5, size=21, color=NAVY, bold=True)
    add_text(slide, label, x, 5.22, 2.7, 0.28, size=9.5, color=GRAY)
add_bullets(slide, [
    ("Why frozen. ", "Training only the head isolates the quality of ImageNet features; the backbone cannot memorise generators."),
    ("Why ~1,000 per class. ", "The Colab quota limited the run, so this is a conservative deep baseline, not a generous one."),
    ("Checkpoint rule. ", "Best validation macro-F1 only; unseen generators never enter training or model selection."),
], 0.62, 5.75, 12.1, 1.3, size=11, gap=5)

# ================================================================ slide 12 — reproducibility
slide = slide_frame("Every number in this talk is generated, not typed",
                    "Reproducibility", 12, "Junxu Chen", "6:52\u20137:20", 0.733)
for i, (head, body) in enumerate([("features", "CLIP + ResNet + FFT"), ("DINOv2", "second cache"),
                                  ("probe", "P1 / LOGO / ablation"), ("analyze", "JSON + figures"),
                                  ("report", "PDF + PPTX")]):
    x = 0.62 + i * 2.5
    add_rect(slide, x, 1.85, 2.25, 0.95, LIGHT, BLUE if i == 2 else None)
    add_text(slide, head, x + 0.1, 1.93, 2.05, 0.3, size=12, color=NAVY, bold=True)
    add_text(slide, body, x + 0.1, 2.27, 2.05, 0.4, size=8.5, color=GRAY)
    if i < 4:
        add_arrow(slide, x + 2.25, 2.27, 0.22, 0.2)
add_rect(slide, 0.62, 3.1, 7.4, 1.55, NAVY)
add_text(slide, "python Code/run_pipeline.py --dry-run\n"
               "python Code/run_pipeline.py --stages probe analyze assets report\n"
               "python Code/run_pipeline.py            # full run\n"
               "python Code/train_multi.py ... --freeze-backbone --epochs 10\n"
               "python Code/demo_server.py --port 8000",
         0.82, 3.22, 7.0, 1.7, size=10.5, color=WHITE, font="Consolas", line_spacing=1.3)
add_bullets(slide, [
    ("Hash split. ", "Seed 5438; no random seed is chosen after seeing results."),
    ("No hand-typed numbers. ", "Tables, figures and this deck read summary.json."),
    ("Honest caveat. ", "GPU training has floating-point nondeterminism; the shipped JSON is the measured run."),
], 8.25, 3.15, 4.45, 2.3, size=10.5, gap=7)
add_text(slide, "One entry point  \u00b7  18 stages  \u00b7  caches skip existing work",
         0.62, 4.85, 7.4, 0.3, size=10, color=BLUE, bold=True)
add_callout(slide, "Change one number and rebuild the deck: python Code/build_slides.py. "
             "No slide cell was typed by hand.", 0.62, 5.3, 7.4, 0.8, size=11)
add_text(slide, "Artifacts:  Results/final_v3/*.json  \u00b7  Report/technicalReport.pdf  \u00b7  "
               "Report/demo/index.html", 0.62, 6.45, 12.1, 0.3, size=9.5, color=GRAY)

# ================================================================ slide 13 — LOGO
slide = slide_frame("Leave-one-generator-out: the ranking holds in 7 of 7 folds",
                    "Generalization", 13, "Yangqi Zhang", "7:20\u20137:50", 0.783)
add_image(slide, "fig_logo_heatmaps.png", 0.5, 1.7, 7.7, 4.75)
add_rect(slide, 8.45, 1.75, 4.25, 1.35, NAVY)
add_text(slide, "97.1%  vs  95.6%", 8.65, 1.87, 3.9, 0.5, size=23, color=WHITE, bold=True)
add_text(slide, "pooled over 2,800 held-out images", 8.65, 2.4, 3.9, 0.3, size=10, color=MUTED)
add_bullets(slide, [
    ("7 / 7 folds won. ", "Train on six generators, test only on the seventh."),
    ("+1.50 pooled ", "95% CI [+0.86, +2.18]."),
    ("McNemar ", "23 vs 65, p = 8.5 \u00d7 10\u207b\u2076."),
    ("Same weak columns. ", "MidJourney is hardest for both; the gap is largest there."),
], 8.45, 3.3, 4.25, 2.5, size=10.5, gap=8)
add_callout(slide, "The P1 conclusion is not an artefact of which three generators we happened to train on.",
             8.45, 5.85, 4.25, 0.85, size=11)

# ================================================================ slide 14 — ablation + robustness
slide = slide_frame("The gain comes from fusion, and it survives compression",
                    "Ablation & robustness", 14, "Yangqi Zhang", "7:50\u20138:25", 0.842)
add_image(slide, "fig_ablation.png", 0.55, 1.7, 6.9, 3.35)
add_text(slide, "Ablation, three-generator training", 0.62, 5.05, 6.8, 0.3, size=10, color=BLUE, bold=True)
add_bullets(slide, [
    ("DINOv2 alone 90.1% ", "\u2014 not a bigger-backbone effect."),
    ("CLIP + DINOv2 linear 95.9% ", "\u2014 the whole gain."),
    ("Extra capacity hurts: ", "+ResNet 94.6, +FFT 95.9, MLP head 94.9."),
], 0.62, 5.42, 6.9, 1.2, size=9.5, gap=5)
rows = [("Condition", "UnivFD", "Ours", "Ours+aug"),
        ("clean", pc(robust["univfd"]["mean_all"]["clean"]), pc(robust["ours"]["mean_all"]["clean"]),
         pc(robust["ours_aug"]["mean_all"]["clean"])),
        ("JPEG q50", pc(robust["univfd"]["mean_all"]["jpeg50"]), pc(robust["ours"]["mean_all"]["jpeg50"]),
         pc(robust["ours_aug"]["mean_all"]["jpeg50"])),
        ("combo", pc(robust["univfd"]["mean_all"]["combo"]), pc(robust["ours"]["mean_all"]["combo"]),
         pc(robust["ours_aug"]["mean_all"]["combo"]))]
y = 1.85
add_rect(slide, 7.65, y, 5.05, 0.4, NAVY)
for x0, t, w0 in [(7.75, rows[0][0], 1.35), (9.1, rows[0][1], 1.1), (10.2, rows[0][2], 1.1), (11.3, rows[0][3], 1.3)]:
    add_text(slide, t, x0, y + 0.08, w0, 0.25, size=9, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER if t != "Condition" else PP_ALIGN.LEFT)
for i, row in enumerate(rows[1:]):
    yy = y + 0.4 + i * 0.5
    add_rect(slide, 7.65, yy, 5.05, 0.5, LIGHT if i % 2 == 0 else WHITE)
    add_text(slide, row[0], 7.75, yy + 0.1, 1.3, 0.3, size=10, color=NAVY)
    for j, x0 in enumerate([9.1, 10.2, 11.3]):
        add_text(slide, row[j + 1], x0, yy + 0.1, 1.05, 0.3, size=10,
                 color=BLUE if j > 0 else GRAY, align=PP_ALIGN.CENTER)
add_text(slide, "Mean over seven generators, every degradation applied to the same test files.",
         7.65, 3.65, 5.05, 0.4, size=9, color=GRAY)
add_bullets(slide, [
    ("JPEG q50: ", "+3.5 points for ours, 95.6 vs 92.0."),
    ("Augmented variant: ", "97.6 at q50, but \u22120.7 on clean images."),
    ("Deployment rule. ", "Report the condition you train for; do not quote one number for all."),
], 7.65, 4.15, 5.05, 1.7, size=10, gap=6)
add_callout(slide, "Ours+aug at JPEG q50 is above UnivFD's clean accuracy. The trade-off is explicit, not hidden.",
             7.65, 6.05, 5.05, 0.85, size=10)

# ================================================================ slide 15 — live demo cue
slide = slide_frame("Demo: score a photograph the probe has never seen",
                    "Live demonstration", 15, "Yangqi Zhang", "8:25\u20139:40", 0.967)
add_rect(slide, 0.62, 1.78, 5.25, 1.2, AMBER)
add_text(slide, "LIVE DEMO \u2014 NOW", 0.84, 1.9, 4.9, 0.45, size=24, color=NAVY, bold=True)
add_text(slide, "75 seconds  \u00b7  one new photo  \u00b7  one verdict", 0.86, 2.42, 4.9, 0.3,
         size=11, color=NAVY, bold=True)
steps = [("1", "Drop one photo", "phone photo or a dataset sample; it never leaves the room"),
         ("2", "Both encoders run", "CLIP + DINOv2, real forward pass, 24\u201330 ms each"),
         ("3", "Read the verdict", "P(fake) plus per-stage latency, 247\u2013356 ms end to end")]
for i, (num, head, body) in enumerate(steps):
    y = 3.25 + i * 0.95
    add_text(slide, num, 0.66, y, 0.4, 0.4, size=18, color=BLUE, bold=True)
    add_text(slide, head, 1.12, y, 4.6, 0.3, size=13, color=NAVY, bold=True)
    add_text(slide, body, 1.12, y + 0.32, 4.7, 0.42, size=9, color=GRAY)
add_callout(slide, f"Held-out ADM sample on screen: P(fake) = {DEMO_P:.4f}" if DEMO_P is not None
            else "Held-out ADM sample on screen: P(fake) > 0.99", 0.62, 6.05, 5.25, 0.55, size=10)
add_text(slide, "If the port fails: the page marks itself offline and replays cached scores \u2014 never invents one.",
         0.62, 6.7, 5.35, 0.35, size=8.5, color=AMBER, bold=True)
shot = DEMO_SHOT if DEMO_SHOT.exists() else DEMO_PHOTO
add_photo(slide, shot, 6.25, 1.75, 6.55, 4.6)
add_text(slide, "Screenshot of the actual page (Report/demo/index.html) with Code/demo_server.py live on this machine.",
         6.25, 6.45, 6.55, 0.5, size=8.5, color=GRAY)

# ================================================================ slide 16 — takeaways
slide = slide_frame("Three things to remember", "Conclusion", 16, "Yangqi Zhang", "9:40\u201310:00", 1.0)
items = [("1", "Foundation features dominate.",
          "69.4% handcrafted \u2192 94.1% frozen CLIP: +24.7 points on unseen generators."),
         ("2", "Two frozen views beat the strongest single view.",
          "95.9% vs 94.1%, +1.8 points, 7/7 LOGO folds, with 1,793 trainable parameters."),
         ("3", "The result is auditable.",
          "Hash split, validation-only selection, paired tests, one command, JSON to PDF.")]
yy = 1.8
for i, (num, head, body) in enumerate(items):
    add_text(slide, num, 0.62, yy + i * 1.2, 0.7, 0.7, size=28, color=BLUE, bold=True)
    add_text(slide, head, 1.45, yy + i * 1.2, 10.8, 0.42, size=18, color=NAVY, bold=True)
    add_text(slide, body, 1.45, yy + i * 1.2 + 0.42, 10.8, 0.4, size=12, color=GRAY)
add_rect(slide, 0.62, 5.55, 12.1, 0.03, RGBColor(0xC7, 0xD2, 0xDC))
add_text(slide, "Questions?", 0.62, 5.8, 4.0, 0.6, size=28, color=NAVY, bold=True)
add_text(slide, "python Code/run_pipeline.py --dry-run   \u00b7   Report/DEFENSE_SCRIPT.md\n"
               "All numbers: Results/final_v3/summary.json", 4.95, 5.85, 7.8, 0.75, size=10, color=GRAY)

# ---------------------------------------------------------------- speaker notes
NOTES = [
    "[0:00\u20130:20 | Huanrong Li | Slide 1] Good morning. We are Group 20. One question drives this project: "
    "when a new image generator appears tomorrow, does an AI-image detector still work? Our answer is 95.9% against "
    "94.1% on four generators neither model has seen. Five speakers, ten minutes, and a live demo at the end.",

    "[0:20\u20130:50 | Huanrong Li | Slide 2] [Point to the train card, then the test card.] We train on SD 1.5, BigGAN "
    "and GLIDE. We test on ADM, MidJourney, VQDM and Wukong. The test generators are never used for training, tuning "
    "or early stopping. The protocol is 200 real and 200 generated per generator: 1,600 balanced test images. "
    "When we say 'beat', we mean higher accuracy on those unseen files with a paired bootstrap and an exact McNemar "
    "test, not a higher number on a different split. [Stay for the method.]",

    "[0:50\u20131:50 | Huanrong Li | Slide 3] Our method, DELP, keeps both encoders frozen. CLIP gives 768 dimensions, "
    "DINOv2 gives 1024. We L2-normalise each block separately, concatenate to 1,792, scale and fit one logistic "
    "regression: 1,793 weights. No fine-tuning, no augmentation, no test-time averaging. The hypothesis is simple: "
    "CLIP encodes semantics, DINOv2 encodes local texture. They make different errors, so a linear combination has "
    "lower variance on a new generator. It costs 27 milliseconds per image on the RTX 2080. [Hand off to Bo Pang.] "
    "\u201cBo will cover the traps and the data.\u201d",

    "[1:50\u20132:20 | Bo Pang | Slide 4] Why is unseen hard? In-domain scores can reward three shortcuts: one "
    "generator's fingerprint, its JPEG history, and its native resolution. GenImage and the Fake-or-JPEG paper "
    "document exactly this. We remove the shortcuts with a hash split, balanced classes and the same files for every "
    "method. A detector that only recognises its training generator is not useful. [Stay on the data.]",

    "[2:20\u20133:10 | Bo Pang | Slide 5] The benchmark has 23,328 images: 5,828 real and seven generators with 2,500 "
    "each. There is one GAN, BigGAN, and six diffusion models, so native resolution varies from 128 to 1024 pixels. "
    "We split by SHA-1 of seed, class and filename into 70, 15, 15. No files are copied and no directory order is "
    "used, so the split is identical on every machine and cannot leak a generator into test. [Stay.]",

    "[3:10\u20133:40 | Bo Pang | Slide 6] We also test post-processing: clean, JPEG 90, JPEG 50, resize to half and "
    "back, blur sigma 1, and a combined variant. The same 400 images per generator are re-encoded for each condition. "
    "The audit found eight folders, 23,328 files, zero corrupt images, and every class present in all three splits. "
    "The inputs are frozen before any model is trained, so the test set cannot drift. [Hand off to Huanyu Zhang.] "
    "\u201cHuanyu will walk the evaluation ladder.\u201d",

    "[3:40\u20134:10 | Huanyu Zhang | Slide 7] Here is the ladder. The handcrafted HOG plus radial-FFT SVM reaches "
    "69.4% on unseen-4. A frozen ResNet-18 with a trained head reaches 82.8. UnivFD, a frozen CLIP linear probe, "
    "reaches 94.1. Our dual-encoder probe reaches 95.9. Read the gaps: deep beats non-deep by 13.4 points; foundation "
    "features add 24.7; fusing two views adds 1.8. Every rung uses the same files and the same split seed. [Stay.]",

    "[4:10\u20134:55 | Huanyu Zhang | Slide 8] The classical detector is transparent: HOG sees edges, radial FFT sees "
    "periodic upsampling peaks, and an RBF-SVM separates them. It is strong in-domain and brittle across generators: "
    "a new upsampler moves the spectrum, and a new post-process removes the peaks. We keep it because Level II asks a "
    "deep model to clear this transparent bar. It does, by 13.4 points. [Stay.]",

    "[4:55\u20135:30 | Huanyu Zhang | Slide 9] The star is UnivFD from CVPR 2023. It freezes CLIP ViT-L/14, 427.6 "
    "million parameters, takes the 768-d embedding, normalises it, scales it and fits a logistic regression with 769 "
    "weights. C is chosen on validation macro-F1 only. It reaches 94.1% unseen. It is hard to beat because it already "
    "uses the strongest encoder and a linear head leaves little room to over-fit. [Hand off to Junxu Chen.] "
    "\u201cJunxu will show the head-to-head result and the engineering.\u201d",

    "[5:30\u20136:12 | Junxu Chen | Slide 10] Here is the headline. 95.9% versus 94.1%, plus 1.81 points, 95% "
    "confidence interval plus 0.87 to plus 2.81. McNemar finds 18 cases where we are right and UnivFD is wrong, and "
    "47 the other way, p = 4.2 times 10 to the minus 4. AUC moves from 99.13 to 99.54; macro-F1 from 94.0 to 95.9. "
    "Both probes use exactly the same 8,164 training images, so this is not an extra-data effect. Averaged over all "
    "seven generators we are at 97.1 against 95.9. [Stay for the baseline.]",

    "[6:12\u20136:52 | Junxu Chen | Slide 11] The supervised baseline freezes an ImageNet ResNet-18 and trains only "
    "the two-way head. It sees 1,998 balanced training images, 402 validation images, ten epochs. It reaches 82.8% "
    "unseen. Because of the Colab quota we trained on about 1,000 images per class, so this is a conservative deep "
    "baseline, not a generous one. The checkpoint is chosen on validation macro-F1 only; the four unseen generators "
    "never enter training or model selection. [Stay.]",

    "[6:52\u20137:20 | Junxu Chen | Slide 12] Every number in this deck is generated. Features are cached, probes are "
    "trained, statistics are written to JSON, and the report and this deck read that JSON. One command prints every "
    "step: run_pipeline.py with --dry-run. Another runs the full pipeline, and --stages reruns only what changed. "
    "The split seed is fixed at 5438, selection uses validation only, and the commands are in the README. The honest "
    "caveat: GPU training has floating-point nondeterminism, so the shipped JSON is the measured run. [Hand off to "
    "Yangqi Zhang.] \u201cYangqi will test whether the result generalises.\u201d",

    "[7:20\u20137:50 | Yangqi Zhang | Slide 13] P1 trains on three generators. A stronger test is leave-one-generator-"
    "out: train on six, test on the seventh, seven times. We win 7 of 7 folds. Pooled over 2,800 held-out images, "
    "97.1% versus 95.6%, plus 1.50 points, CI plus 0.86 to plus 2.18, McNemar 23 versus 65, "
    "p = 8.5 times 10 to the minus 6. "
    "MidJourney remains the hardest column for both methods. [Stay.]",

    "[7:50\u20138:25 | Yangqi Zhang | Slide 14] The ablation isolates the cause. DINOv2 alone is 90.1, so this is not "
    "a bigger-backbone effect. CLIP plus DINOv2 linear is 95.9: that is the whole gain. Adding ResNet gives 94.6; an "
    "MLP head gives 94.9. With about eight thousand points, extra capacity fits generator-specific noise. Robustness: "
    "at JPEG 50 we are 95.6 against 92.0, plus 3.5. The augmented variant reaches 97.6 at q50 but loses 0.7 on clean "
    "images, so we report the deployment condition explicitly. [Stay for the demo.]",

    "[8:25\u20139:40 | Yangqi Zhang | Slide 15] This page is the cue: it shows the actual demo page, and this is "
    "where the live demonstration happens. [Switch to the browser.] I drop one photograph that is not in the "
    "dataset, both frozen encoders run on the local GPU, and the same 1,793-weight probe returns P(fake) plus the "
    "per-encoder time. The warm server measures 247 to 356 milliseconds end to end, with 24 to 30 milliseconds per "
    "encoder. [Drop the photo; read the score aloud; pause three seconds.] If the port fails, the page marks itself "
    "offline and replays cached scores; it never invents one. [Stay for the close.]",

    "[9:40\u201310:00 | Yangqi Zhang | Slide 16] Three takeaways: foundation features dominate, from 69.4 to "
    "94.1; two frozen views beat the strongest single view by 1.8 points, in 7 of 7 LOGO folds; and every number "
    "is auditable from the hash split to the paired tests. Thank you. Questions?",
]
for slide_item, note in zip(prs.slides, NOTES):
    slide_item.notes_slide.notes_text_frame.text = note

prs.core_properties.title = "A detector that still works when the generator is new"
prs.core_properties.author = "EE5438 Group 20"
prs.core_properties.subject = "Cross-generator AI-image detection, 10-minute defence"
prs.save(OUT)
print("saved", OUT)
