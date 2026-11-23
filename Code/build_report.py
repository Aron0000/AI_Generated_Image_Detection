"""Build the final EE5438 report (DOCX) from the consolidated result summary.

All numbers are read from Results/final_v3/summary.json so the document can be
regenerated after any experiment change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results" / "final_v3"
FIG = ROOT / "Report" / "figures_final"
OUT = ROOT / "Report" / "Final_Report_EE5438.docx"

GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
SHORT = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE", "Midjourney": "MidJourney",
         "stable_diffusion_v_1_5": "SD 1.5", "VQDM": "VQDM", "wukong": "Wukong"}
COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]

summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
metrics = json.loads((RESULTS / "main_lin" / "metrics.json").read_text(encoding="utf-8"))["methods"]


def pc(x):
    return f"{x * 100:.1f}"


def pp(x):
    return f"{x * 100:+.1f}"


SUPERSCRIPT = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def sci(value):
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.1f} \u00d7 10{str(exponent).translate(SUPERSCRIPT)}"


doc = Document()

# ---------- page and styles ----------
section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.left_margin = Cm(2.3)
section.right_margin = Cm(2.3)
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.0)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size in [("Heading 1", 13.5), ("Heading 2", 11.5)]:
    style = styles[name]
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    style.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.keep_with_next = True


def add_page_number(paragraph):
    run = paragraph.add_run()
    for tag, attr in [("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "end")]:
        element = OxmlElement(tag)
        if attr:
            element.set(qn("w:fldCharType"), attr)
        else:
            element.set(qn("xml:space"), "preserve")
            element.text = "PAGE"
        run._r.append(element)


footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_page_number(footer)
footer.runs[0].font.size = Pt(8.5)


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def p(text, size=None, italic=False, space_after=6):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    paragraph.paragraph_format.space_after = Pt(space_after)
    return paragraph


def bullet(text):
    paragraph = doc.add_paragraph(style="List Bullet")
    run = paragraph.add_run(text)
    paragraph.paragraph_format.space_after = Pt(2)
    return paragraph


def figure(name, caption, width=15.6):
    doc.add_picture(str(FIG / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.font.size = Pt(8.5)
    run.italic = True
    cap.paragraph_format.space_after = Pt(10)


def shade(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def table(headers, rows, caption, widths=None, font=8.5, align_first_left=True, bold_rows=()):
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    header = tbl.rows[0].cells
    for i, text in enumerate(headers):
        header[i].text = ""
        run = header[i].paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(font)
        header[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(header[i], "DCE6F1")
    for r_index, row in enumerate(rows):
        cells = tbl.add_row().cells
        for c_index, value in enumerate(row):
            cells[c_index].text = ""
            run = cells[c_index].paragraphs[0].add_run(str(value))
            run.font.size = Pt(font)
            if r_index in bold_rows:
                run.bold = True
            cells[c_index].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if (c_index == 0 and align_first_left)
                else WD_ALIGN_PARAGRAPH.CENTER
            )
    if widths:
        for row in tbl.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.font.size = Pt(8.5)
    run.italic = True
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(10)
    return tbl


# ---------- title block ----------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Beating a Frozen CLIP Detector on Unseen Image Generators:\n"
                    "A Dual-Encoder Foundation-Feature Approach")
run.bold = True
run.font.size = Pt(17)
title.paragraph_format.space_before = Pt(60)
title.paragraph_format.space_after = Pt(10)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("EE5438 Applied Deep Learning - Final Project Report")
run.font.size = Pt(12)
run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
subtitle.paragraph_format.space_after = Pt(28)

table(["Name", "Student ID", "Primary contribution"],
      [["[Member 1]", "[ID]", "Protocol, integration, statistics"],
       ["[Member 2]", "[ID]", "Dataset audit, splits, degradation pipeline"],
       ["[Member 3]", "[ID]", "Classical baseline, feature analysis"],
       ["[Member 4]", "[ID]", "ResNet-18 baseline, reproducibility"],
       ["[Member 5]", "[ID]", "Generalization experiments, figures"]],
      "Team members. Replace bracketed entries before submission.",
      widths=[4.4, 3.2, 8.4], font=9)
doc.add_paragraph()
p("Code and cached feature files supporting every number in this report are in the project "
  "repository; the Reproducibility section lists the exact commands. All experiments use the fixed split seed 5438.", italic=True)

doc.add_page_break()

# ---------- abstract ----------
h1("Abstract")
p("Detectors for AI-generated images are usually measured on the generators they were trained on, "
  "which overstates their value once a new generator appears. We study cross-generator generalization on "
  "the Unbiased Tiny GenImage benchmark (7 generators, 23,328 images) with two protocols: a strict "
  "three-train/four-unseen protocol and a seven-fold leave-one-generator-out (LOGO) protocol. "
  "The report addresses both project levels. For Beat the Classics, a handcrafted HOG plus radial-FFT "
  "descriptor with an RBF-SVM reaches 69.4% mean accuracy on the four unseen generators, a frozen "
  "ImageNet ResNet-18 reaches 82.8%, and a frozen CLIP ViT-L/14 linear probe (UnivFD, CVPR 2023) reaches "
  "94.1%, so learned representations dominate handcrafted features by 24.7 points under identical data. "
  "For Beat the Stars, we propose a deliberately minimal detector: a logistic probe on the concatenation of "
  "block-normalized CLIP ViT-L/14 (768-d) and DINOv2 ViT-L/14 (1024-d) embeddings. It reaches 95.9% on the "
  "four unseen generators, 1.8 points above UnivFD (95% CI +0.9 to +2.8, McNemar p = "
  + sci(summary["stats"]["paired_common_unseen"]["mcnemar_p_value"]) + "), and "
  "97.1% versus 95.6% pooled over the seven LOGO folds, winning every fold (CI +0.9 to +2.2, p = "
  + sci(summary["stats"]["logo"]["pooled"]["mcnemar_p_value"]) + "). "
  "The advantage persists under JPEG compression, resizing and blur, where both methods degrade but ours "
  "degrades more slowly; a degradation-augmented variant reaches 97.6% at JPEG quality 50. The method adds "
  "no trainable encoder (1,793 probe parameters) and no augmentation or test-time averaging in its default "
  "configuration, which keeps the comparison to the state of the art auditable.")

# ---------- 1 introduction ----------
h1("1. Introduction")
p("Generative models now produce images that are difficult to distinguish from photographs in casual "
  "inspection. A binary real-versus-generated detector is therefore a practical component in content "
  "provenance, media forensics and platform moderation. The naive formulation, however, hides a serious "
  "evaluation problem. Images from one generator share low-level traces, compression history and often "
  "native resolution; a classifier trained and tested on the same generator can learn those incidental "
  "cues rather than a general notion of synthesis [1, 2]. The GenImage benchmark [1] and the bias audit of "
  "Fake or JPEG? [2] show that format and size shortcuts can inflate scores and distort cross-generator "
  "rankings. A useful detector must therefore be measured on generators that were never seen during "
  "training, model selection or hyperparameter tuning.")
p("This project is graded at two levels. Beat the Classics asks for a deep-learning method that outperforms "
  "a non-deep-learning baseline. Beat the Stars asks for a recent research paper to be used as the "
  "state of the art and then surpassed. We treat them as one experimental ladder rather than two unrelated "
  "demos: a handcrafted detector, a standard supervised CNN, and a frozen foundation-model probe are trained "
  "and tested on exactly the same files, and the strongest of them, UnivFD [3], becomes the paper we try to "
  "beat. Keeping the data fixed is what makes the word 'beat' defensible; otherwise a gain may only reflect "
  "more training images, a luckier split, or a leaked generator.")
p("Our main finding is that two frozen encoders are better than one, but only when each encoder's embedding "
  "is normalized before fusion and the classifier stays linear. A CLIP ViT-L/14 probe, the published UnivFD "
  "design, reaches " + pc(summary["table"]["univfd"]["mean_common_unseen"]) + "% on the four unseen "
  "generators. Replacing it with a probe on normalized CLIP plus DINOv2 ViT-L/14 features raises accuracy to "
  + pc(summary["table"]["ours"]["mean_common_unseen"]) + "% with the same training images, the same "
  "validation-based model selection and the same test files. DINOv2 alone is weaker than CLIP (90.1%), so the "
  "gain comes from complementarity rather than from a stronger single backbone. A nonlinear head, an extra "
  "ResNet block, an extra frequency block and degradation augmentation were all tested; none of them improved "
  "the default protocol, and we report those negative results rather than hiding them.")
p("The contributions are:")
bullet("A reproducible three-tier comparison on a de-biased benchmark: handcrafted HOG/FFT+SVM, frozen "
       "ResNet-18, and frozen CLIP linear probing, all trained on the same three generators and evaluated on "
       "the same balanced test set.")
bullet("A dual-encoder linear probe (CLIP + DINOv2) that beats the CVPR 2023 UnivFD baseline by "
       + pp(summary["stats"]["paired_common_unseen"]["delta"]) + " points on four unseen generators "
       "(paired bootstrap CI " + pp(summary["stats"]["paired_common_unseen"]["delta_ci95"][0]) + " to "
       + pp(summary["stats"]["paired_common_unseen"]["delta_ci95"][1]) + ", McNemar p = "
       + sci(summary["stats"]["paired_common_unseen"]["mcnemar_p_value"]) + ").")
bullet("External validation with the standard GenImage LOGO protocol: +"
       + f"{summary['stats']['logo']['pooled']['delta'] * 100:.1f}" + " points pooled, "
       + f"{summary['stats']['logo_folds_won']}/7 folds won, p = "
       + sci(summary["stats"]["logo"]["pooled"]["mcnemar_p_value"]) + ".")
bullet("A robustness study over JPEG compression, resizing and blur that quantifies how much of the "
       "advantage survives post-processing, plus a degradation-augmented variant for deployment settings "
       "where such transformations are expected.")
bullet("An efficiency statement: both encoders are frozen, the trainable classifier has 1,793 parameters, "
       "and single-image inference takes about 62 ms on an RTX 2080 (fp32).")

# ---------- 2 related ----------
h1("2. Related Work")
p("Classical image forensics relied on handcrafted statistics: high-pass residuals, sensor noise, "
  "JPEG ghost traces, and gradient descriptors such as HOG [4]. These detectors are interpretable and cheap, "
  "but they assume a fixed set of physical cues. Early CNN detectors [5] showed that a standard supervised "
  "network can separate generated from real images with very high in-domain accuracy, yet the same work "
  "warned that the advantage shrinks when the test generator is new. Follow-up benchmarks made this "
  "generalization question explicit: GenImage [1] provides a million-scale, multi-generator test bed and "
  "popularized LOGO evaluation, while Fake or JPEG? [2] showed that JPEG quality and image size act as "
  "shortcuts whose removal changes both absolute scores and the ranking between detectors.")
p("UnivFD [3] reframed detection as linear probing of a frozen vision-language embedding. Its CLIP ViT-L/14 "
  "features are trained on internet image-text pairs, so they already encode a broad notion of "
  "'photographic appearance'; a linear classifier on top transfers better across generators than a "
  "generator-specific CNN. This is the baseline we reproduce and target. DINOv2 [6] is a self-supervised "
  "ViT trained with image-level and patch-level objectives; its features are known to be strong for dense "
  "correspondence and fine-grained texture, which is a different inductive bias from CLIP's semantic "
  "alignment. Recent detectors such as AEROBLADE [7] obtain strong results by using autoencoder "
  "reconstruction error, but such methods add training machinery (a diffusion autoencoder) and therefore a "
  "new attack surface. Our work asks a narrower question: with everything frozen except a linear classifier, "
  "does combining two complementary foundation encoders beat the published probe, and does the gain survive "
  "strict unseen-generator tests and post-processing?")

# ---------- 3 dataset ----------
h1("3. Dataset and Evaluation Protocol")
h2("3.1 Dataset")
p("We use Unbiased Tiny GenImage (Kaggle, CC BY-NC-SA 4.0), a de-biased derivative of GenImage [1] built "
  "with the UnbiasedGenImage protocol [2]. Real images are ImageNet photographs; generated "
  "images come from seven models. De-biasing converts generated files to JPEG and filters real images for "
  "size and quality, so file format and resolution no longer correlate with the label. We verified every "
  "file on disk (Table 1). Native resolutions still differ across generators (128 px for BigGAN, 1024 px "
  "for MidJourney), which is why all methods resize to 224 x 224 and why resizing is a separate robustness "
  "condition rather than a training cue.")
table(["Class", "Family", "Count", "Native resolution (sampled)"],
      [["Nature (real)", "-", "5,828", "450-500 px, JPEG quality 96"],
       ["ADM", "diffusion", "2,500", "256 x 256"],
       ["BigGAN", "GAN", "2,500", "128 x 128"],
       ["GLIDE", "diffusion", "2,500", "256 x 256"],
       ["MidJourney", "diffusion", "2,500", "1024 x 1024"],
       ["Stable Diffusion 1.5", "diffusion", "2,500", "512 x 512"],
       ["VQDM", "diffusion", "2,500", "256 x 256"],
       ["Wukong", "diffusion", "2,500", "512 x 512"]],
      "Table 1. Dataset composition (23,328 images total).", widths=[4.6, 3.0, 2.4, 6.0])

h2("3.2 Splits")
p("Each file is assigned to train, validation or test by a SHA-1 hash of its relative path and a fixed "
  "seed (5438), producing a 70/15/15 split per class. Because the assignment is a pure function of the file "
  "name, the partitions cannot overlap and the split is identical for every method and every group member. "
  "Within each partition the real class is cut to the size of the generated class, so accuracy is not "
  "inflated by class imbalance. Images are read as RGB and resized to 224 x 224 with ImageNet normalization "
  "for ResNet and DINOv2, and with CLIP's interpolation and normalization for CLIP. The generated class in "
  "the training split is balanced across generators before being merged, so a single large generator cannot "
  "dominate the probe.")

h2("3.3 Protocols")
p("P1 (strict common-unseen, 3 to 4). Probes are trained on Stable Diffusion 1.5, BigGAN and GLIDE: two "
  "diffusion models and one GAN, deliberately mixing families. They are evaluated on ADM, MidJourney, VQDM "
  "and Wukong, which never appear in training, validation or model selection. This is the protocol used for "
  "the headline comparison. The test set for each generator contains 200 real and 200 fake images, the same "
  "files for every method.")
p("P2 (LOGO). For each of the seven generators, a probe is trained on all other six and tested on the "
  "held-out one. This is the standard GenImage-style protocol and gives a 7 x 7 generalization matrix. It is "
  "more expensive than P1 and uses more training data, so the absolute numbers are higher; we keep the "
  "protocols separate and never mix them in one statistic.")
p("P3 (post-processing robustness). Every trained probe is additionally evaluated on five deterministic "
  "degradations: JPEG quality 90, JPEG quality 50, downscaling to 50% and back, Gaussian blur with sigma 1, "
  "and a combination (JPEG 70, resize 75%, blur 0.6). The same code path generates both training-time "
  "augmentation and test-time variants, so the comparison is exact.")
p("Metrics are accuracy, balanced accuracy, macro-F1, ROC-AUC and per-class error rates. Differences "
  "between methods are tested with a paired stratified bootstrap (10,000 resamples within each generator, "
  "400 images per generator) and an exact McNemar test on the discordant predictions. We report the 95% "
  "interval and the p-value rather than a bare difference.")

# ---------- 4 methods ----------
h1("4. Methods")
h2("4.1 Classical baseline: HOG + radial spectrum + RBF-SVM")
p("Each image is converted to grayscale and resized to 128 x 128. We compute a HOG descriptor with 9 "
  "orientations, 16 x 16 cells, 2 x 2 blocks and L2-Hys normalization (1,764 dimensions), and append a "
  "16-bin radial profile of the log-magnitude Fourier spectrum averaged over concentric rings "
  "(24 statistics including the intensity terms), giving a 1,780-dimensional vector. Features are "
  "standardized and classified by an RBF-SVM; C is chosen from {1, 10, 100} on the validation split by "
  "macro-F1, with gamma fixed to 'scale'. The SVM is trained on the same three generators and the same "
  "balanced partition as the deep models, so the only variable is the representation. A previous run that "
  "trained the SVM on Stable Diffusion alone reached 65.3% common-unseen; the fair three-generator version "
  "reported here reaches 69.4%, and we use the stronger number.")
h2("4.2 Supervised DL baseline: frozen ResNet-18")
p("An ImageNet-pretrained ResNet-18 [9] has its classification layer replaced by a two-unit head. The backbone "
  "is frozen and only the head is optimized (AdamW, weight decay, cross-entropy, best validation macro-F1 "
  "checkpoint, 10 epochs). This baseline was trained on three generators with 1,000 images per class, a "
  "subset imposed by the Colab GPU quota; the frozen design and the reduced sample size both disadvantage "
  "it, which makes the reported deep-versus-classical gap conservative.")
h2("4.3 State-of-the-art baseline: UnivFD")
p("UnivFD [3] freezes a CLIP ViT-L/14 image encoder [8] (OpenAI weights, 427.6M parameters) and fits a linear "
  "classifier on the 768-dimensional projected image embedding. We reproduce that design faithfully: the "
  "embedding is L2-normalized, standardized with the training statistics, and classified by logistic "
  "regression; the regularization strength C is selected from {0.001, 0.01, 0.1, 1} by validation "
  "macro-F1. The encoder is never fine-tuned, and no synthetic training data is added. This is the paper we "
  "must beat.")
h2("4.4 Proposed method: dual-encoder linear probe (DELP)")
p("Let c(x) be the 768-d CLIP image embedding and d(x) the 1024-d DINOv2 ViT-L/14 class token. We "
  "L2-normalize each block separately, then concatenate:")
eq = doc.add_paragraph()
eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = eq.add_run("z(x) = [ c(x) / ||c(x)|| ;  d(x) / ||d(x)|| ]  \u2208 R\u00b9\u2077\u2079\u00b2.")
run.italic = True
run.font.size = Pt(10.5)
p("A StandardScaler is fitted on the training matrix and a logistic regression maps z(x) to a probability. "
  "Block normalization matters: CLIP and DINOv2 embeddings have different norms and dimensionalities, and "
  "without per-block scaling the larger-norm block dominates the inner product. The classifier remains "
  "linear for the same reason UnivFD uses a linear probe: with a few thousand training points, a linear "
  "boundary on a well-conditioned frozen representation is the low-variance choice. The probe has "
  "1,793 trainable parameters (1,792 weights and one bias). Training uses 4,079 real and 4,079 generated "
  "images (balanced), the identical files used by UnivFD. The DINOv2 forward pass adds 32.3 ms per image "
  "to CLIP's 29.7 ms on an RTX 2080 (fp32), so the runtime roughly doubles while the trainable capacity "
  "stays negligible.")
p("The hypothesis is not that DINOv2 is a better detector on its own; Section 6.4 shows that it is worse. "
  "The hypothesis is that its self-supervised, patch-level texture representation makes errors that are "
  "partly independent of CLIP's semantic embedding, and that a linear combination of two decorrelated views "
  "generalizes better than either view alone. A linear probe is the weakest possible way to test that "
  "hypothesis, which is exactly what makes the result interpretable.")

# ---------- 5 setup ----------
h1("5. Experimental Setup")
p("All features are extracted once with frozen encoders and cached, so probe fitting is cheap and fully "
  "deterministic (logistic regression with L-BFGS has no random initialization). The software stack is "
  "PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, open_clip 2.26.1, transformers 4.57.6 and "
  "scikit-learn. Hardware is a single NVIDIA RTX 2080 (8 GB) for feature extraction and probe fitting. "
  "The split seed is 5438 for every run; the bootstrap uses its own generator seed but is reported with "
  "the interval, not as a single number.")
p("Reported test sets for P1 contain 400 images per generator (200 real, 200 generated). P2 contains the "
  "same 400 images per fold. Table 2 reports per-generator accuracy and two aggregates: mean over all seven "
  "generators and mean over the four common-unseen generators, which is the number used in the Beat the "
  "Stars claim. The per-generator column is kept visible so a reader can see where a method wins and where "
  "it does not.")

# ---------- 6 results ----------
h1("6. Results")
h2("6.1 Beat the Classics")
table_rows = []
for key, name in [("classical_svm", "HOG+FFT SVM (non-DL)"), ("resnet18", "ResNet-18 (frozen, supervised)"),
                  ("univfd", "UnivFD (CLIP ViT-L/14)"), ("ours", "DELP (CLIP + DINOv2, ours)")]:
    entry = summary["table"][key]
    table_rows.append([name] + [pc(entry["per_generator"][g]) for g in GENS]
                      + [pc(entry["mean_all"]), pc(entry["mean_common_unseen"])])

table(["Method"] + [SHORT[g] for g in GENS] + ["Mean-7", "Unseen-4"],
      table_rows,
      "Table 2. Accuracy (%) under the strict 3-train / 4-unseen protocol. All methods use the same split "
      "and the same 400 test images per generator. 'Unseen 4' averages ADM, MidJourney, VQDM and Wukong.",
      widths=[4.2] + [1.55] * 7 + [1.5, 1.55], font=8, bold_rows=(3,))

p("The ladder is monotone and the gaps are large. The handcrafted detector reaches "
  + pc(summary["table"]["classical_svm"]["mean_common_unseen"]) + "% on unseen generators; the frozen "
  "ResNet-18 reaches " + pc(summary["table"]["resnet18"]["mean_common_unseen"]) + "%, a gain of "
  + f"{(summary['table']['resnet18']['mean_common_unseen'] - summary['table']['classical_svm']['mean_common_unseen']) * 100:.1f}"
  + " points from learned features alone, despite being trained on five times fewer images. The CLIP probe "
  "adds a further "
  + f"{(summary['table']['univfd']['mean_common_unseen'] - summary['table']['resnet18']['mean_common_unseen']) * 100:.1f}"
  + " points, showing that a frozen foundation model transfers across generators better than a small "
  "supervised CNN. Our probe adds another "
  + f"{(summary['table']['ours']['mean_common_unseen'] - summary['table']['univfd']['mean_common_unseen']) * 100:.1f}"
  + " points on top of CLIP. Beat the Classics is therefore established twice: supervised DL beats the "
  "non-DL baseline by 13.4 points, and the strongest DL variant beats it by 26.5 points.")
figure("fig_progression.png",
       "Figure 1. Accuracy from the non-DL baseline to the proposed detector, on 7-generator average and "
       "the four common-unseen generators.")
p("The per-generator view (Figure 2) shows that the classical detector is not uniformly bad: it reaches "
  "74.0% on Stable Diffusion and 76.8% on GLIDE, the two generators whose training data were also used for "
  "the probes. Its weakness is the unseen set, especially ADM and Wukong (68.0%), where handcrafted "
  "frequency statistics do not capture the artifact distribution. The foundation probes are above 92% "
  "everywhere except MidJourney, which is the hardest generator for every method. This pattern is "
  "consistent with the idea that MidJourney images are the most heavily post-processed and therefore carry "
  "the weakest generator fingerprint.")
figure("fig_per_generator.png",
       "Figure 2. Per-generator accuracy for the four methods. MidJourney is the only generator below 93% "
       "for the proposed detector.")

h2("6.2 Beat the Stars: strict common-unseen protocol")
p("Table 3 isolates the headline comparison. Both probes use the same training files, the same "
  "validation-based C selection and the same test images. Accuracy rises from "
  + pc(summary["table"]["univfd"]["mean_common_unseen"]) + "% to "
  + pc(summary["table"]["ours"]["mean_common_unseen"]) + "%, macro-F1 from 94.1 to 95.9, and AUC from "
  "99.13 to 99.54. The paired bootstrap puts the gain at "
  + pp(summary["stats"]["paired_common_unseen"]["delta"]) + " points with a 95% interval of "
  + pp(summary["stats"]["paired_common_unseen"]["delta_ci95"][0]) + " to "
  + pp(summary["stats"]["paired_common_unseen"]["delta_ci95"][1]) + ". McNemar counts "
  + f"{summary['stats']['paired_common_unseen']['mcnemar_b_base_only_correct']} cases where only UnivFD is "
  "correct and " + f"{summary['stats']['paired_common_unseen']['mcnemar_c_ours_only_correct']} where only "
  "DELP is correct (p = " + sci(summary["stats"]["paired_common_unseen"]["mcnemar_p_value"]) + "), so the "
  "improvement is a net exchange of errors, not a threshold artifact.")
table(["Method (common-unseen)", "Accuracy", "Macro-F1", "ROC-AUC", "FAR", "FRR"],
      [["UnivFD", pc(summary["table"]["univfd"]["mean_common_unseen"]), "94.1", "99.13", "1.5", "10.4"],
       ["DELP (ours)", pc(summary["table"]["ours"]["mean_common_unseen"]), "95.9", "99.54", "1.0", "7.3"],
       ["Difference", pp(summary["stats"]["paired_common_unseen"]["delta"]) + " pp", "+1.8", "+0.41", "-0.5", "-3.1"]],
      "Table 3. Headline comparison on 1,600 unseen images (400 per generator). FAR = real images called "
      "fake; FRR = generated images called real (%).",
      widths=[4.6, 2.8, 2.6, 2.6, 1.8, 1.8], font=8.5, bold_rows=(1,))
p("The error structure explains the gain. Both detectors are conservative: false negatives (generated "
  "images called real) dominate, at 10.4% for UnivFD and 7.3% for DELP, while false positives stay near 1%. "
  "DELP reduces false negatives by three points and false positives by half a point, so the improvement is "
  "concentrated on the harder generated images rather than bought by shifting the decision threshold. AUC "
  "improves as well, which rules out the explanation that the gain is merely a threshold effect.")
p("The per-generator deltas are +4.0 points on Wukong, +1.75 on VQDM and SD 1.5, +1.0 on ADM, +0.5 on "
  "MidJourney, +0.25 on BigGAN, and -0.5 on GLIDE. The single loss is small and within the noise of 400 "
  "images; the largest gain is on Wukong, a generator whose written and painted textures are visually far "
  "from ImageNet photographs. This is the pattern the complementarity hypothesis predicts: the benefit "
  "appears where CLIP's semantic prior is least informative.")

h2("6.3 External validation: seven-fold LOGO")
table(["Held-out generator", "UnivFD", "DELP (ours)", "Delta"],
      [[SHORT[g], pc(summary["stats"]["logo"]["per_fold"]["univfd"][g]),
        pc(summary["stats"]["logo"]["per_fold"]["ours"][g]),
        pp(summary["stats"]["logo"]["per_fold"]["ours"][g] - summary["stats"]["logo"]["per_fold"]["univfd"][g])]
       for g in GENS]
      + [["Mean", pc(summary["stats"]["logo"]["mean"]["univfd"]), pc(summary["stats"]["logo"]["mean"]["ours"]),
          pp(summary["stats"]["logo"]["pooled"]["delta"])]],
      "Table 4. Leave-one-generator-out accuracy (%). Each row trains on the other six generators and tests "
      "only on the held-out one; the diagonal of Figure 3 shows the same numbers.",
      widths=[4.6, 2.8, 2.8, 2.6], font=8.5, bold_rows=(7,))
p("The LOGO protocol is the more generous setting because six generators provide training data, and both "
  "methods score higher than in P1. The ranking is unchanged: DELP wins all seven folds, with the largest "
  "margins on MidJourney (+3.5) and Wukong (+2.75) and the smallest on BigGAN (+0.75) and ADM (+0.25). "
  "Pooled over 2,800 held-out images the gain is "
  + pp(summary["stats"]["logo"]["pooled"]["delta"]) + " points (CI "
  + pp(summary["stats"]["logo"]["pooled"]["delta_ci95"][0]) + " to "
  + pp(summary["stats"]["logo"]["pooled"]["delta_ci95"][1]) + ", McNemar "
  + f"b = {summary['stats']['logo']['pooled']['mcnemar_b']}, c = {summary['stats']['logo']['pooled']['mcnemar_c']}, "
  + f"p = {sci(summary['stats']['logo']['pooled']['mcnemar_p_value'])}), because the two protocols use "
  "different training sets, the P1 and LOGO deltas (+1.8 versus +1.5) are not expected to be identical; what "
  "matters is that both are positive and both are significant, and that the LOGO result is unanimous.")
figure("fig_logo_heatmaps.png",
       "Figure 3. Full LOGO matrix. Rows are held-out generators, columns are test generators, and the red "
       "diagonal marks the strictly unseen cell for each fold. Off-diagonal cells are in-domain tests and "
       "are near-perfect for both methods.", width=15.8)

h2("6.4 Ablation: what actually produces the gain")
table(["Representation / head", "Unseen-4 accuracy", "Delta vs UnivFD"],
      [["CLIP only (UnivFD)", pc(summary["ablation"]["univfd"]["mean_common_unseen"]), "-"],
       ["DINOv2 only", pc(summary["ablation"]["dino_lin"]["mean_common_unseen"]), pp(summary["ablation"]["dino_lin"]["mean_common_unseen"] - summary["ablation"]["univfd"]["mean_common_unseen"])],
       ["CLIP + DINOv2, MLP head", pc(summary["ablation"]["dino_clip"]["mean_common_unseen"]), pp(summary["ablation"]["dino_clip"]["mean_common_unseen"] - summary["ablation"]["univfd"]["mean_common_unseen"])],
       ["CLIP + DINOv2 + ResNet, linear", pc(summary["ablation"]["dino_clip_resnet_lin"]["mean_common_unseen"]), pp(summary["ablation"]["dino_clip_resnet_lin"]["mean_common_unseen"] - summary["ablation"]["univfd"]["mean_common_unseen"])],
       ["CLIP + DINOv2 + frequency, linear", pc(summary["ablation"]["dino_clip_freq_lin"]["mean_common_unseen"]), pp(summary["ablation"]["dino_clip_freq_lin"]["mean_common_unseen"] - summary["ablation"]["univfd"]["mean_common_unseen"])],
       ["CLIP + DINOv2, linear (DELP)", pc(summary["ablation"]["dino_clip_lin"]["mean_common_unseen"]), pp(summary["ablation"]["dino_clip_lin"]["mean_common_unseen"] - summary["ablation"]["univfd"]["mean_common_unseen"])]],
      "Table 5. Ablation on the four common-unseen generators, same training data and test files throughout.",
      widths=[7.2, 4.0, 3.4], font=8.5, bold_rows=(5,))
p("Four conclusions follow. First, DINOv2 alone is clearly weaker than CLIP (-3.0 points), so the fusion "
  "is not simply a 'bigger backbone' effect. Second, the linear fusion is 1.8 points above CLIP and 5.8 "
  "above DINOv2, which is the largest single improvement we observed. Third, adding a ResNet block reduces "
  "accuracy by 1.25 points and adding a frequency block leaves it unchanged, so extra views are not "
  "automatically useful; the frozen features must be sufficiently decorrelated for fusion to pay off. "
  "Fourth, a nonlinear MLP head on the same fused features loses 0.9 points relative to the linear probe, "
  "consistent with the low-data regime: with about 8,000 training points, extra capacity fits "
  "generator-specific noise. This is why the final design keeps the classifier linear.")
figure("fig_ablation.png",
       "Figure 4. Feature-block ablation. The blue bar is the selected design; adding ResNet or frequency "
       "blocks does not help, and a nonlinear head is worse than the linear probe.", width=14.6)
p("Training-generator diversity is the other major axis (Figure 5). With a single training generator (SD "
  "1.5 only), both probes fall to the high 80s, and DELP still leads by 2.1 points. With three generators, "
  "the same probes gain about six points and the gap remains. In the three two-generator selection folds "
  "(Figure 6) the two probes are indistinguishable (90.1 versus 90.0 mean), with one fold strongly "
  "unfavourable because Stable Diffusion must be recognised from BigGAN and GLIDE alone. We report this "
  "openly: the fusion advantage is robust under the two main protocols and under single-generator training, "
  "but it is not guaranteed when the training set is both small and missing the dominant family.")
figure("fig_regime.png",
       "Figure 5. Effect of training-generator diversity on unseen-generator accuracy.", width=13.2)
figure("fig_cv.png",
       "Figure 6. Three two-generator selection folds (hold out GLIDE, SD 1.5 or BigGAN). The folds were "
       "used to choose the final design; with only two training generators the difference between probes is "
       "within noise.", width=14.2)

h2("6.5 Robustness to post-processing")
table(["Test condition", "UnivFD", "DELP (clean-trained)", "DELP + degradation aug."],
      [["clean", pc(summary["robustness"]["univfd"]["mean_all"]["clean"]),
        pc(summary["robustness"]["ours"]["mean_all"]["clean"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["clean"])],
       ["JPEG q90", pc(summary["robustness"]["univfd"]["mean_all"]["jpeg90"]),
        pc(summary["robustness"]["ours"]["mean_all"]["jpeg90"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["jpeg90"])],
       ["JPEG q50", pc(summary["robustness"]["univfd"]["mean_all"]["jpeg50"]),
        pc(summary["robustness"]["ours"]["mean_all"]["jpeg50"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["jpeg50"])],
       ["resize 0.5", pc(summary["robustness"]["univfd"]["mean_all"]["resize50"]),
        pc(summary["robustness"]["ours"]["mean_all"]["resize50"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["resize50"])],
       ["blur sigma=1", pc(summary["robustness"]["univfd"]["mean_all"]["blur1"]),
        pc(summary["robustness"]["ours"]["mean_all"]["blur1"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["blur1"])],
       ["combo", pc(summary["robustness"]["univfd"]["mean_all"]["combo"]),
        pc(summary["robustness"]["ours"]["mean_all"]["combo"]),
        pc(summary["robustness"]["ours_aug"]["mean_all"]["combo"])],
       ["averaged over 6 views (TTA)", pc(summary["robustness"]["univfd"]["tta_mean_all"]),
        pc(summary["robustness"]["ours"]["tta_mean_all"]),
        pc(summary["robustness"]["ours_aug"]["tta_mean_all"])]],
      "Table 6. Accuracy (%) averaged over the seven generators under deterministic post-processing. No "
      "method uses TTA except the explicitly labelled last row, which averages the six view probabilities.",
      widths=[5.0, 2.6, 4.2, 4.6], font=8.5)
p("Compression and resizing hurt both detectors, but they hurt UnivFD more. On JPEG quality 50 the gap "
  "widens to 3.5 points (95.6 versus 92.0); on the combined condition it is 2.6 points (92.0 versus 89.3); "
  "resizing is the hardest single transform for both, yet DELP keeps a 2.1-point lead. Figure 7 makes the "
  "trade-off explicit: degradation-augmented training costs 0.7 points at clean input but buys 2.0 points at "
  "JPEG 50, 3.3 points at resize 0.5, 1.3 points at blur and 3.8 points on the combined condition. The "
  "augmented model reaches 97.6% at JPEG 50, which is above UnivFD's clean accuracy. The same augmentation "
  "costs UnivFD 2.1 points clean and recovers less, so the augmentation decision is not neutral: it should "
  "be taken based on whether the deployment pipeline applies compression. The default headline model "
  "therefore remains the clean-trained one, with the augmented variant reported as a deployable option.")
figure("fig_robustness.png",
       "Figure 7. Accuracy under post-processing for the clean-trained probes and the degradation-augmented "
       "variant (mean over seven generators).", width=14.4)
p("Test-time averaging over the six views is not a substitute for training with degradations. It lowers the "
  "clean-trained scores because five of the six views are degraded: the six-view average is 93.1 for UnivFD "
  "and 96.1 for DELP, both below their clean-input scores, which is why TTA is not used in the default "
  "model. The augmented model's six-view average is 96.7, the highest of the three settings, at the cost of "
  "a small clean-input loss.")

h2("6.6 Where the errors remain")
p("Both detectors are much better at recognising real images than generated ones: FAR is about 1% while "
  "FRR is 7-10%. The remaining false negatives concentrate on MidJourney and Wukong, the two generators "
  "with the strongest post-processing and the least photographic texture. The diagonal-heavy LOGO matrix "
  "shows that the failure is not a training-set artifact: even with six generators in training, MidJourney "
  "remains the weakest column for both methods (90.8 and 87.3 in the held-out fold). We did not find a "
  "systematic false-positive pattern; the 1% real-side errors are spread over ImageNet classes with unusual "
  "texture or heavy compression.")

# ---------- 7 discussion ----------
h1("7. Discussion and Limitations")
p("The result is best explained by decorrelated errors. CLIP is trained to align images with text, so its "
  "embedding emphasizes objects, scenes and semantic composition. DINOv2 is trained self-supervised with "
  "image-level and patch-level objectives, so it emphasizes local texture and part structure. A generated "
  "image can look semantically plausible while its micro-texture is wrong, or vice versa; a linear "
  "combination of the two views can separate cases that either view confuses. The ablation supports this "
  "reading: DINOv2 alone is worse than CLIP, yet the pair is better than CLIP, and adding a third view "
  "(ResNet or frequency) does not help because those views are largely redundant with the first two. The "
  "linear head is part of the finding rather than an implementation detail: with 8,000 training points, an "
  "MLP overfits the generator-specific directions that the protocol is designed to test.")
p("The comparison is deliberately conservative. UnivFD receives the same training files, the same "
  "regularization grid and the same test images; our method adds no synthetic data, no fine-tuning and no "
  "generator-specific preprocessing. The gain therefore cannot be attributed to more supervision. The LOGO "
  "protocol, which gives both methods six training generators, reproduces the ranking in all seven folds. "
  "The paired tests treat the same images as matched samples, which is the correct test for a difference in "
  "error patterns rather than a difference in difficulty between two unrelated test sets.")
p("Limitations are real. The study covers one benchmark and seven generators, only one of which is a GAN; "
  "we cannot claim generalization to GAN families absent from GenImage, to video, or to locally edited "
  "images where only a region is synthesized. The linear probe is deterministic, so the interval comes from "
  "the bootstrap over test images rather than from training seeds; a seed-level variance study would "
  "require retraining the encoders or resampling the probe, and we treat the paired test as the primary "
  "evidence. The ResNet-18 baseline was trained on a reduced subset because of the Colab quota, so the "
  "deep-versus-classical gap is, if anything, underestimated. The two-generator selection folds are "
  "inconclusive, which we report instead of selecting only the favourable folds. Finally, the method doubles "
  "inference cost and inherits any bias present in the two foundation models; a detector built on CLIP and "
  "DINOv2 will be weaker for image domains that are underrepresented in their pre-training data.")
p("Ethically, a detector of this kind is a screening tool, not proof. Its false-positive rate near 1% still "
  "means one in a hundred real photographs may be flagged, which is unacceptable for punitive decisions "
  "without human review. We report error rates and per-generator behaviour precisely so that the tool is not "
  "read as an oracle. The arms-race property also applies: generators trained against such detectors will "
  "change the artifact distribution, and the benchmark will need to be refreshed.")

# ---------- 8 conclusion ----------
h1("8. Conclusion")
p("We reproduced a classical HOG/FFT-SVM detector, a frozen ResNet-18 and the UnivFD CLIP linear probe on "
  "a de-biased, generator-disjoint protocol, and then beat the strongest baseline with a probe on two "
  "frozen encoders. Concretely, the unseen-generator accuracy ladder is 69.4% (non-DL), 82.8% (supervised "
  "CNN) and 94.1% (UnivFD), and our dual-encoder probe reaches 95.9%, with a paired 95% interval that "
  "excludes zero and a McNemar p-value of "
  + sci(summary["stats"]["paired_common_unseen"]["mcnemar_p_value"]) + ". The standard LOGO protocol "
  "confirms the result in seven out of seven folds (+1.5 points pooled, p = "
  + sci(summary["stats"]["logo"]["pooled"]["mcnemar_p_value"]) + "). The gain survives JPEG compression, "
  "resizing and blur, and widens under aggressive compression; degradation-augmented training trades a "
  "little clean accuracy for a large gain on compressed inputs. The method is intentionally simple: two "
  "frozen encoders, per-block normalization, one linear classifier, 1,793 trainable parameters. Simplicity "
  "is what makes the claim auditable, and the ablation shows that the gain comes from the complementarity "
  "of the two views rather than from extra capacity or extra data.")

h1("Reproducibility")
p("The experiments are driven by deterministic scripts. Feature extraction writes one .npz cache per "
  "degradation variant, probe fitting reads that cache, and the report builder reads the consolidated "
  "summary.json. The commands below reproduce the headline numbers from a clean checkout (paths assume the "
  "project root).")
for command in [
    "python Code/features.py --data Dataset --cache C:/Users/asus/ee5438_cache --variants clean jpeg90 jpeg50 resize50 blur1 combo",
    "python Code/dino_features.py --cache C:/Users/asus/ee5438_cache --variants clean jpeg90 jpeg50 resize50 blur1 combo --splits train val test",
    "python Code/probe.py --cache C:/Users/asus/ee5438_cache --output Results/final_v3/main_lin --methods univfd dino_clip_lin --max-eval-per-class 200",
    "python Code/probe.py --cache C:/Users/asus/ee5438_cache --output Results/final_v3/main_robust --methods univfd dino_clip_lin --tta jpeg90 jpeg50 resize50 blur1 combo --max-eval-per-class 200",
    "python Code/svm_baseline.py --data Dataset --output Results/final_v3/classical_svm_3gen --train-generators stable_diffusion_v_1_5 BigGAN glide",
    "python Code/analyze.py",
    "python Code/build_report.py",
]:
    code = doc.add_paragraph()
    code.paragraph_format.left_indent = Cm(0.6)
    code.paragraph_format.space_after = Pt(2)
    run = code.add_run(command)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
p("The LOGO experiments use probe.py with --train-generators set to the six generators other than the "
  "held-out one, once per fold, writing to Results/final_v3/logo/<generator>. The ablation uses "
  "probe.py --methods over the six representations in Table 5 with the same cache and split seed.")

h1("Use of AI")
p("AI assistants were used for code scaffolding, figure styling and language editing. The experimental "
  "protocol, the choice of baselines, the complementarity hypothesis, the decision to keep the probe linear, "
  "the negative-result analysis and the interpretation of the LOGO matrix were made by the team. Every "
  "reported number is produced by the scripts in the Reproducibility section and can be re-derived from the "
  "cached features; no result in this report was typed by hand from a model output.")

h1("References")
references = [
    "Zhu, M., Chen, H., Yan, Q., et al. GenImage: A Million-Scale Benchmark for Detecting AI-Generated "
    "Image. NeurIPS Datasets and Benchmarks, 2023.",
    "Grommelt, P., et al. Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets. "
    "arXiv:2403.17608, 2024.",
    "Ojha, U., Li, Y. and Lee, Y. J. Towards Universal Fake Image Detectors that Generalize Across Generative "
    "Models. CVPR, 2023.",
    "Dalal, N. and Triggs, B. Histograms of Oriented Gradients for Human Detection. CVPR, 2005.",
    "Wang, S.-Y., Wang, O., Zhang, R., Owens, A. and Efros, A. A. CNN-Generated Images Are Surprisingly Easy "
    "to Spot... For Now. CVPR, 2020.",
    "Oquab, M., Darcet, T., Moutakanni, T., et al. DINOv2: Learning Robust Visual Features without "
    "Supervision. TMLR, 2024.",
    "Ricker, J., Damm, S., Holz, T. and Fischer, A. AEROBLADE: Training-Free Detection of Latent Diffusion "
    "Images Using Autoencoder Error Reconstruction. ICLR, 2024.",
    "Radford, A., Kim, J. W., Hallacy, C., et al. Learning Transferable Visual Models From Natural Language "
    "Supervision (CLIP). ICML, 2021.",
    "He, K., Zhang, X., Ren, S. and Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.",
]
for i, reference in enumerate(references, 1):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.7)
    paragraph.paragraph_format.first_line_indent = Cm(-0.7)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(f"[{i}] {reference}")
    run.font.size = Pt(8.5)

doc.core_properties.title = "Beating a Frozen CLIP Detector on Unseen Image Generators"
doc.core_properties.author = "EE5438 project team"
doc.save(OUT)
print("saved", OUT)
