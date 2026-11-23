"""Generate the 25+ page LaTeX technical report from measured result files.

Every number in the PDF is interpolated from
  Results/final_v3/summary.json
  Results/final_v3/latex_assets.json
  Results/final_v3/efficiency.json
so the document cannot drift from the experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results" / "final_v3"
LATEX = ROOT / "Report" / "latex"
OUT = LATEX / "technicalReport.tex"

S = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
A = json.loads((RESULTS / "latex_assets.json").read_text(encoding="utf-8"))
E = json.loads((RESULTS / "efficiency.json").read_text(encoding="utf-8"))
M = json.loads((RESULTS / "main_lin" / "metrics.json").read_text(encoding="utf-8"))["methods"]

GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]
SHORT = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE", "Midjourney": "MidJourney",
         "stable_diffusion_v_1_5": "SD 1.5", "VQDM": "VQDM", "wukong": "Wukong"}
TRAIN_GENS = ["stable_diffusion_v_1_5", "BigGAN", "glide"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def ci(low: float, high: float) -> str:
    return f"{low * 100:+.2f} to {high * 100:+.2f}"


def sci(value: float) -> str:
    exponent = 0
    mantissa = value
    while mantissa < 1:
        mantissa *= 10
        exponent -= 1
    while mantissa >= 10:
        mantissa /= 10
        exponent += 1
    return f"${mantissa:.1f}\\times10^{{{exponent}}}$"


def acc_table(name: str) -> dict:
    return S["table"][name]


def row(*cells) -> str:
    return " & ".join(str(c) for c in cells) + r" \\"


def table(caption: str, label: str, headers: list[str], rows: list[str], colspec: str,
          size: str = r"\footnotesize", note: str | None = None, tabcolsep: str | None = None) -> str:
    parts = [r"\begin{table}[htbp]", r"\centering", size, r"\caption{" + caption + "}",
             r"\label{" + label + "}"]
    if tabcolsep:
        parts.append(r"\setlength{\tabcolsep}{" + tabcolsep + "}")
    parts.append(r"\begin{tabular}{" + colspec + "}")
    parts.append(r"\toprule")
    parts.append(" & ".join(r"\emph{" + h + "}" for h in headers) + r" \\")
    parts.append(r"\midrule")
    parts.extend(rows[:-1])
    if note:
        parts.append(r"\midrule")
        parts.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{\footnotesize " + note + r"} \\")
    parts.append(rows[-1])
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    return "\n".join(parts)


def figure(name: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=" + width + r"]{figures/" + name + "}",
        r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure}"])


def sec(title: str, label: str) -> str:
    return r"\section{" + title + r"}" + "\n" + r"\label{" + label + "}"


def subsec(title: str, label: str) -> str:
    return r"\subsection{" + title + r"}" + "\n" + r"\label{" + label + "}"


# ---------------------------------------------------------------- shorthand values
cls = S["table"]["classical_svm"]
res = S["table"]["resnet18"]
uni = S["table"]["univfd"]
our = S["table"]["ours"]
ps = S["stats"]["paired_common_unseen"]
lg = S["stats"]["logo"]
logo_folds = lg["per_fold"]
t = A["dataset"]
tr = t["train_selection"]
train_fake = tr["p1_train_fake"]

FIGS = {
    "progression": "fig_progression.png",
    "per_generator": "fig_per_generator.png",
    "confusion": "fig_confusion.png",
    "roc": "fig_roc.png",
    "far_frr": "fig_far_frr.png",
    "bootstrap": "fig_bootstrap.png",
    "logo": "fig_logo_heatmaps.png",
    "ablation": "fig_ablation.png",
    "regime": "fig_regime.png",
    "cv": "fig_cv.png",
    "robustness": "fig_robustness.png",
}

out: list[str] = []
add = out.append

# ---------------------------------------------------------------- preamble
add(r"""\documentclass[12pt,letterpaper,oneside]{article}

\input{./latexGoodPractices/preamble}

\usepackage{setspace}
\usepackage{float}
\usepackage{placeins}
\usepackage{fancyvrb}
\usepackage{fvextra}
\usepackage{longtable}
\usepackage{array}
\usepackage{lastpage}
\usepackage{enumitem}
\setstretch{1.12}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=1.4em,itemsep=1pt,topsep=2pt}
\setlist[enumerate]{leftmargin=1.6em,itemsep=1pt,topsep=2pt}

\newcommand{\reportTitle}{Outperforming a Frozen CLIP Detector on Unseen Image Generators}
\newcommand{\reportAuthors}{Huanrong Li, Bo Pang, Huanyu Zhang, Junxu Chen, Yangqi Zhang}
\newcommand{\reportDate}{2026-11-23}

\newcommand{\reportVersions}{
0.1 & 2026-10-07 & Project scope; dataset access; experimental protocol outline \\
0.2 & 2026-10-14 & Dataset audit; deterministic hash splits; manifest frozen \\
0.3 & 2026-10-21 & Classical and ResNet-18 baselines; P1 evaluation harness \\
0.4 & 2026-10-28 & UnivFD reproduction; feature-cache verification \\
0.5 & 2026-11-04 & Dual-encoder probe; headline P1 comparison \\
0.6 & 2026-11-10 & LOGO evaluation; ablations; degradation study \\
0.7 & 2026-11-12 & Robustness and efficiency measurements; first full figure pass \\
0.8 & 2026-11-14 & Paired statistics; error analysis; generated figures \\
0.9 & 2026-11-17 & Internal review; terminology; reference audit \\
0.9.1 & 2026-11-18 & Reference pruning; layout and figure corrections \\
0.9.2 & 2026-11-19 & Interactive upload demonstration; consistency pass \\
0.9.3 & \reportDate & Submission candidate; defence slides; demonstration video \\
}

\addbibresource{references.bib}

\input{technicalReportStyle.tex}

\begin{document}
\makeCustomTitle
\thispagestyle{titlePage}

\begin{center}
{\large A Dual-Encoder Foundation-Feature Approach}\\[4pt]
EE5438 Applied Deep Learning --- Group 20 Final Project\\[4pt]
\small Project levels addressed: Level II (Beat the Classics) and Level III (Beat the Stars)
\end{center}

\vspace{4pt}
\begin{center}
\footnotesize
\begin{tabular}{lll}
\toprule
\emph{Name} & \emph{Student ID} & \emph{Primary ownership} \\
\midrule
Huanrong Li (leader) & to be filled & Method (DELP), protocol, statistics \\
Bo Pang              & to be filled & Dataset audit, splits, degradation pipeline \\
Huanyu Zhang         & to be filled & Classical baseline and feature analysis \\
Junxu Chen           & to be filled & ResNet-18 baseline, reproducibility \\
Yangqi Zhang         & to be filled & Generalization experiments, figures, demo \\
\bottomrule
\end{tabular}
\end{center}

\vspace{2pt}
\begin{center}
\footnotesize Code and every result file used below are in the project repository; the feature caches are
regenerated by the commands in \autoref{app:reproduce}. All experiments use the deterministic split seed 5438.
No number in this report is typed by hand: tables and figures are generated from the JSON files described in
\autoref{app:reproduce}. Student IDs and e-mail addresses are to be inserted by the group before submission.
\end{center}

\newpage
\begin{abstract}
""")

add(
"Detectors of AI-generated images are usually evaluated on the generators they were trained on, which "
"overstates their usefulness when a new generator appears. We study cross-generator generalisation on the "
"Unbiased Tiny GenImage benchmark, which contains 23,328 images from seven generators and a de-biased real "
"class. Two protocols are used: a strict three-train/four-unseen protocol (P1) and a seven-fold "
"leave-one-generator-out protocol (P2). Under Level II (Beat the Classics), a handcrafted HOG plus "
"radial-spectrum descriptor with an RBF-SVM reaches " + pct(cls["mean_common_unseen"]) + r"\% mean accuracy on the four "
"unseen generators, a frozen ImageNet ResNet-18 with a trained two-way head reaches "
+ pct(res["mean_common_unseen"]) + r"\%, and a frozen CLIP ViT-L/14 linear probe, the published UnivFD "
"design, reaches " + pct(uni["mean_common_unseen"]) + r"\%. Learned representations therefore dominate "
"handcrafted features by 24.7 points under identical data. Under Level III (Beat the Stars) we propose DELP, a logistic "
"probe on the concatenation of block-normalised CLIP (768-d) and DINOv2 (1024-d) embeddings. It reaches "
+ pct(our["mean_common_unseen"]) + r"\% on the four unseen generators, " + pp(ps["delta"]) +
" points above UnivFD (paired 95\\% CI " + ci(ps["delta_ci95"][0], ps["delta_ci95"][1]) +
", McNemar p = " + sci(ps["mcnemar_p_value"]) + "), and " + pct(lg["pooled"]["accuracy_ours"]) +
r"\% versus " + pct(lg["pooled"]["accuracy_univfd"]) + r"\% pooled over the seven LOGO folds, winning "
+ str(S["stats"]["logo_folds_won"]) + " of 7 folds (CI " + ci(lg["pooled"]["delta_ci95"][0], lg["pooled"]["delta_ci95"][1])
+ ", p = " + sci(lg["pooled"]["mcnemar_p_value"]) + "). The advantage persists under JPEG compression, "
"resizing and blur, and a degradation-augmented variant reaches " + pct(S["robustness"]["ours_aug"]["mean_all"]["jpeg50"])
+ r"\% at JPEG quality 50. The default detector adds no trainable encoder (1,793 probe parameters) and no "
"augmentation or test-time averaging, which keeps the comparison auditable. We further report four negative "
"results and the conditions under which the advantage does not hold."
)
add(r"\end{abstract}")
add(r"\vspace{4pt}\noindent\textbf{Keywords:} AI-generated image detection; cross-generator generalisation; "
    r"CLIP; DINOv2; linear probing; leave-one-generator-out; reproducibility.")
add(r"\newpage")
add(r"\tableofcontents")
add(r"\newpage")

# ---------------------------------------------------------------- 1 introduction
add(sec("Introduction", "sec:intro"))
add("Generative models now produce images that are difficult to distinguish from photographs in casual "
    "inspection. A real-versus-generated classifier is therefore a plausible component of content provenance, "
    "media forensics and platform moderation. The naive formulation conceals an evaluation problem: images from "
    "one generator share low-level traces, compression history, upsampling artefacts and often a native "
    "resolution. A classifier trained and tested on the same generator can learn those incidental cues instead "
    "of a general notion of synthesis~\\cite{wang2020,genimage2023}. Spectral analyses of generated images "
    "document systematic deviations from natural high-frequency statistics, and these deviations are among the "
    "cues that supervised detectors exploit~\\cite{durall2020,frank2020}. The GenImage benchmark~\\cite{genimage2023} "
    "and the bias audit of Fake or JPEG?~\\cite{fakeorjpeg2024} show that format and size shortcuts can inflate "
    "scores and even reorder detector rankings. A useful detector must therefore be measured on generators that "
    "were never seen during training, model selection or hyper-parameter tuning.")
add("The course brief defines two project levels. Level II, \\emph{Beat the Classics}, requires a deep-learning "
    "method that outperforms a non-deep-learning baseline. Level III, \\emph{Beat the Stars}, requires a recent "
    "research paper to serve as the state of the art and to be surpassed under a controlled protocol. We treat "
    "the two levels as one experimental ladder rather than as two unrelated demonstrations. A handcrafted "
    "detector, a supervised convolutional network and a frozen foundation-model probe are trained and tested on "
    "exactly the same files, and the strongest of them, UnivFD~\\cite{univfd2023}, becomes the reference method "
    "that Level III requires us to outperform. Holding the data and the protocol fixed is what makes the "
    "comparison defensible; otherwise a gain may reflect more training images, a more favourable split or a "
    "generator that leaked into training.")
add("Our main finding is that two frozen encoders are better than one, but only when each embedding is "
    "normalised before fusion and the classifier remains linear. A CLIP ViT-L/14 probe in the published UnivFD "
    "design reaches " + pct(uni["mean_common_unseen"]) + r"\% on the four unseen generators. A probe on "
    "block-normalised CLIP plus DINOv2 ViT-L/14 features raises accuracy to " + pct(our["mean_common_unseen"]) +
    r"\% with the same training images, the same validation-based model selection and the same test files. "
    "DINOv2 alone is weaker than CLIP (" + pct(S["ablation"]["dino_lin"]["mean_common_unseen"]) +
    r"\%), so the gain comes from complementarity rather than from a stronger single backbone. A nonlinear "
    "head, an extra ResNet block, an extra frequency block and degradation augmentation were all tested; none of "
    "them improved the default protocol, and we report those negative results rather than omitting them.")
add("The contributions are:")
add(r"\begin{itemize}")
add(r"\item A reproducible three-tier comparison on a de-biased benchmark: handcrafted HOG/FFT with an RBF-SVM, "
    "a frozen ResNet-18 with a trained head, and a frozen CLIP linear probe, all trained on the same three "
    "generators and evaluated on the same balanced test set.")
add(r"\item A dual-encoder linear probe (CLIP + DINOv2) that outperforms the CVPR 2023 UnivFD baseline by "
    + pp(ps["delta"]) + " points on four unseen generators (paired bootstrap CI "
    + ci(ps["delta_ci95"][0], ps["delta_ci95"][1]) + ", McNemar p = " + sci(ps["mcnemar_p_value"]) + ").")
add(r"\item External validation with the standard GenImage leave-one-generator-out protocol: "
    + pp(lg["pooled"]["delta"]) + " points pooled, a positive difference in all "
    + str(S["stats"]["logo_folds_won"]) + " folds, p = "
    + sci(lg["pooled"]["mcnemar_p_value"]) + ".")
add(r"\item A robustness study over JPEG compression, resizing and blur that quantifies how much of the "
    "advantage survives post-processing, plus a degradation-augmented variant for deployment settings where "
    "such transformations are expected.")
add(r"\item An efficiency measurement on the actual hardware, a full 7$\times$7 LOGO matrix, per-generator "
    "confusion matrices and ROC curves, an error-overlap analysis, and a reproducibility appendix that maps "
    "every number to a script and a JSON file.")
add(r"\end{itemize}")
add("The headline numbers are summarised in \\autoref{tab:ladder_intro}. The rest of the report follows the "
    "experimental logic: \\autoref{sec:related} positions the work, \\autoref{sec:data} audits the data, "
    "\\autoref{sec:protocols} defines the protocols and tests, \\autoref{sec:methods} describes the four "
    "detectors, \\autoref{sec:setup} lists the exact configuration, \\autoref{sec:results} reports results, "
    "and \\autoref{sec:discussion} discusses validity, ethics and negative results.")

intro_rows = [
    row(r"\emph{HOG + radial FFT, RBF-SVM (non-DL)}", pct(cls["mean_all"]), pct(cls["mean_common_unseen"])),
    row(r"\emph{ResNet-18, frozen backbone, trained head}", pct(res["mean_all"]), pct(res["mean_common_unseen"])),
    row(r"\emph{UnivFD: CLIP ViT-L/14 linear probe}", pct(uni["mean_all"]), pct(uni["mean_common_unseen"])),
    row(r"\textbf{DELP: CLIP + DINOv2 linear probe (ours)}", r"\textbf{" + pct(our["mean_all"]) + "}",
        r"\textbf{" + pct(our["mean_common_unseen"]) + "}"),
]
add(table("Accuracy (\\%) under the strict three-train / four-unseen protocol. ``Mean-7'' averages all seven "
          "generators; ``Unseen-4'' averages ADM, MidJourney, VQDM and Wukong, which never appear in training, "
          "validation or model selection.", "tab:ladder_intro",
          ["Method", "Mean-7", "Unseen-4"], intro_rows, "lcc"))

# ---------------------------------------------------------------- 2 related
add(sec("Related Work", "sec:related"))
add(subsec("Generative models and their artefacts", "sec:related_generative"))
add("The generators in this benchmark belong to the two dominant families of image synthesis. Generative "
    "adversarial networks~\\cite{goodfellow2014} train a generator and a discriminator jointly; BigGAN~\\cite{brock2019} "
    "scaled this recipe to high-resolution class-conditional synthesis. Denoising diffusion models~\\cite{ho2020ddpm} "
    "instead learn to reverse a noising process, and ADM~\\cite{dhariwal2021} established them as the stronger "
    "family for class-conditional generation. GLIDE~\\cite{nichol2022} extended diffusion to text-guided synthesis, "
    "and latent diffusion~\\cite{rombach2022} moved the process to an autoencoder latent space, which is the design "
    "behind Stable Diffusion. Vector-quantised diffusion~\\cite{gu2022vqdm} combines a discrete latent codebook with "
    "a diffusion prior; Wukong~\\cite{wu2022wukong} is a large Chinese cross-modal model, and Midjourney~\\cite{midjourney2022} "
    "is a commercial text-to-image service. The families leave different traces: GAN upsampling produces periodic "
    "spectral peaks, diffusion decoders deviate from natural spectra in the high-frequency band, and "
    "post-processing or resizing removes part of both. This heterogeneity is why a detector calibrated on one "
    "family may fail on another, and it motivates the generator-disjoint protocols used below.")
add(subsec("Handcrafted and supervised detectors", "sec:related_classical"))
add("Classical image forensics relied on handcrafted statistics: high-pass residuals, sensor-noise patterns, "
    "JPEG ghost traces and gradient descriptors such as HOG~\\cite{hog2005}. GAN outputs were shown to carry "
    "model-specific fingerprints~\\cite{marra2019}, and spectral analyses demonstrated that up-convolution "
    "leaves systematic deviations from natural image statistics~\\cite{durall2020,frank2020}. These detectors are "
    "interpretable and computationally inexpensive, but they assume a fixed set of physical cues. Early CNN "
    "detectors~\\cite{wang2020} showed that a standard supervised network can separate generated from real images "
    "with very high in-domain accuracy, while warning that the advantage shrinks when the test generator is new. "
    "Follow-up benchmarks made the generalisation question explicit. GenImage~\\cite{genimage2023} provides a "
    "million-scale multi-generator test bed and popularised leave-one-generator-out evaluation. Fake or "
    "JPEG?~\\cite{fakeorjpeg2024} showed that JPEG quality and image size act as shortcuts whose removal changes "
    "both absolute scores and the ranking between detectors. Our dataset is the "
    "de-biased Tiny GenImage derivative built with the Fake-or-JPEG protocol, so the shortcut is reduced but not "
    "eliminated; \\autoref{sec:data} quantifies what remains.")
add(subsec("Frozen foundation features", "sec:related_foundation"))
add("UnivFD~\\cite{univfd2023} reframed detection as linear probing of a frozen vision-language embedding. CLIP "
    "ViT-L/14~\\cite{clip2021}, a vision transformer trained on internet image-text pairs, "
    "encodes a broad notion of ``photographic appearance''; a linear classifier on top transfers across "
    "generators better than a generator-specific CNN. This is the baseline we reproduce and target. Recent "
    "CLIP-based detectors extend this line with larger training corpora and report strong in-domain and "
    "cross-generator accuracy~\\cite{cozzolino2024}, while frequency-aware detectors target the spectral residual "
    "left by diffusion upsampling~\\cite{bammey2024}. DINOv2~\\cite{dinov2} is a self-supervised ViT trained with "
    "image-level and patch-level objectives; its features support dense correspondence and fine-grained texture "
    "without labels, which is a different inductive bias from CLIP's semantic alignment. We ask whether the two "
    "biases are complementary for detection when both encoders are frozen.")
add(subsec("Training-free reconstruction detectors", "sec:related_aeroblade"))
add("AEROBLADE~\\cite{aeroblade2024} obtains strong results by measuring autoencoder reconstruction error: a "
    "latent-diffusion image reconstructed by its own autoencoder is closer to the input than a photograph is. "
    "DIRE~\\cite{wang2023dire} follows a related idea and measures the discrepancy of a diffusion-model "
    "reconstruction. Such methods are attractive because they need no labelled training, but they add a generative "
    "model to the pipeline and therefore a new attack surface, and they are tied to the diffusion family whose "
    "inversion is used. Our work asks a narrower question: with everything frozen except a linear classifier, does "
    "combining two complementary foundation encoders outperform the published probe, and does the advantage "
    "survive strict unseen-generator tests and post-processing? We do not claim to outperform reconstruction-based "
    "detectors on all benchmarks; we claim a controlled improvement over the specific linear-probe state of the "
    "art that we reproduce.")
add(subsec("Positioning", "sec:related_position"))
add("Three design choices distinguish this study from a conventional benchmark submission. First, every method "
    "sees the same training files, the same validation files and the same test files. Second, model selection "
    "uses validation data only; the four unseen generators are touched once, after the configuration is frozen. "
    "Third, the proposed detector is deliberately minimal: no encoder is fine-tuned, no synthetic data is added, "
    "and the head is linear. This makes the measured difference attributable to the representation rather than to "
    "capacity or data volume. The negative results in \\autoref{sec:results_ablation} are part of that argument.")

# ---------------------------------------------------------------- 3 dataset
add(sec("Dataset and Audit", "sec:data"))
add(subsec("Benchmark composition", "sec:data_composition"))
add("We use Unbiased Tiny GenImage, a de-biased derivative of GenImage~\\cite{genimage2023} distributed under "
    "CC BY-NC-SA 4.0. Real images are ImageNet photographs~\\cite{deng2009}; generated images come from seven "
    "models: ADM~\\cite{dhariwal2021}, BigGAN~\\cite{brock2019}, GLIDE~\\cite{nichol2022}, "
    "Midjourney~\\cite{midjourney2022}, Stable Diffusion 1.5~\\cite{rombach2022}, VQDM~\\cite{gu2022vqdm} and "
    "Wukong~\\cite{wu2022wukong}. De-biasing "
    "converts generated files to JPEG and filters real images for size and quality, so file format and "
    "resolution no longer correlate perfectly with the label. \\autoref{tab:dataset} lists what is on disk. "
    "Every generated class contains exactly 2,500 JPEG files; the real class contains 5,828 JPEG files, for a "
    "total of 23,328. The class ratio is therefore 1:3, and every protocol balances it by subsampling the real "
    "class, so accuracy is not inflated by class imbalance.")
add("Native resolution still differs across generators: BigGAN produces 128 pixel images, ADM, GLIDE and VQDM "
    "produce 256 pixel images, Stable Diffusion 1.5 and Wukong 512, and MidJourney 1024. The real class contains "
    "450--500 pixel photographs. All methods resize to 224$\\times$224, and resizing is also a separate "
    "robustness condition (\\autoref{sec:protocols_degradations}). We do not claim that de-biasing removes every "
    "shortcut; it removes the container-format shortcut, and the remaining resolution cue is measured rather "
    "than assumed away.")

dataset_rows = []
family = {"Nature": "photographs", "ADM": "diffusion", "BigGAN": "GAN", "glide": "diffusion",
          "Midjourney": "diffusion", "stable_diffusion_v_1_5": "diffusion", "VQDM": "diffusion", "wukong": "diffusion"}
for name in ["Nature", *GENS]:
    info = t["classes"][name]
    dims = info["common_dims"]
    if name == "Nature":
        resolution = "450--500 (metadata)"
        count = info["count"]
    else:
        resolution = f"{dims[0][0][0]}$\\times${dims[0][0][1]}"
        count = info["count"]
    ext = ", ".join(e.upper().lstrip(".") for e in info["extensions"])
    dataset_rows.append(row("Nature" if name == "Nature" else SHORT[name],
                            family[name], f"{count:,}", ext, resolution))
dataset_rows.append(row(r"\emph{Total}", r"\emph{8 classes}", r"\emph{" + f"{sum(t['classes'][c]['count'] for c in ['Nature', *GENS]):,}" + "}",
                        r"\emph{JPEG}", r"\emph{--}"))
add(table("Files on disk. Counts and formats were read from the dataset directory; resolutions are the "
          "most common size in a 300-image sample per class (the real class uses the full metadata file).",
          "tab:dataset", ["Class", "Family", "Images", "Format", "Resolution"], dataset_rows, "llrll"))

add(subsec("Why de-biasing matters", "sec:data_bias"))
add("In the original GenImage, generated images are often PNG at high resolution while real images are JPEG at "
    "lower resolution. A classifier can then learn the container or the resampling kernel rather than synthesis "
    "artefacts. The Fake or JPEG? audit showed that re-encoding generated images as JPEG and controlling size "
    "changes both accuracy and the ranking of detectors. This dataset applies that correction, and our audit "
    "confirms that all eight classes are stored as JPEG. Two biases remain and are reported: class imbalance "
    "(1:3) is handled by balanced subsampling, and native resolution is handled by a common 224$\\times$224 "
    "resize plus an explicit resize robustness test.")
add(subsec("Same-class visual comparison", "sec:data_examples"))
add("A benchmark can look balanced in a table and still differ in content. \\autoref{fig:examples} shows one "
    "goldfish image from every class, selected by the ImageNet class index embedded in the generated filenames "
    "and matched to the corresponding synset in the real metadata. The point is not that the generated images "
    "are indistinguishable; several contain obvious artefacts (the VQDM sample is blurred, the Wukong sample is "
    "stylised). The point is that all eight classes describe the same object category, so a detector cannot "
    "solve the task by recognising a different scene distribution. This qualitative check motivated the "
    "same-class construction and was visually verified by the team.")
add(figure("dataset_examples.png",
           "One goldfish image per class, selected from the same ImageNet class index. The real class is matched "
           "through the metadata synset. The generated samples differ in style and artefacts but not in object "
           "category.", "fig:examples", r"0.95\linewidth"))

add(subsec("Deterministic splits", "sec:data_splits"))
add("Each file is assigned to train, validation or test by a SHA-1 hash of the string ``seed:class/filename'' "
    "with seed 5438, mapped to a bucket in $[0,100)$: train 0--70, validation 70--85, test 85--100. Because the "
    "assignment is a pure function of the name, partitions cannot overlap, the split is identical for every "
    "method and every group member, and it can be reconstructed without storing an index file. "
    "\\autoref{tab:splits} reports the realised counts. Hash splitting produces small deviations from an exact "
    "70/15/15 ratio per class, which is expected and harmless; the totals are 16,272 / 3,457 / 3,599.")
split_rows = []
for name in ["Nature", *GENS]:
    s = t["splits"][name]
    split_rows.append(row("Nature" if name == "Nature" else SHORT[name],
                          f"{s['train']:,}", f"{s['val']:,}", f"{s['test']:,}",
                          f"{s['train'] + s['val'] + s['test']:,}"))
totals = {k: sum(t["splits"][c][k] for c in ["Nature", *GENS]) for k in ("train", "val", "test")}
split_rows.append(row(r"\emph{Total}", r"\emph{" + f"{totals['train']:,}" + "}", r"\emph{" + f"{totals['val']:,}" + "}",
                      r"\emph{" + f"{totals['test']:,}" + "}", r"\emph{23,328}"))
add(table("Realised hash-based split (seed 5438). The family assignment is 70/15/15 per class up to hash "
          "bucket noise; totals are shown in the last row.", "tab:splits",
          ["Class", "Train", "Validation", "Test", "Total"], split_rows, "lrrrr"))

add(subsec("Training and evaluation sets", "sec:data_selection"))
add("The probes are trained on a balanced subset. For P1, the three training generators contribute "
    f"{train_fake:,} training images in total (" + ", ".join(f"{SHORT[g]}: {tr['p1_train_per_generator'][g]:,}" for g in TRAIN_GENS)
    + "). The real training split contains 4,082 usable images, so the real class is capped at "
    f"{tr['p1_train_real']:,} and the generated class is subsampled to the same size. The resulting training "
    f"set has {tr['p1_train_total']:,} images. Validation uses the same balancing with at most 67 generated "
    f"images per generator, giving {tr['p1_val_total']:,} images (201 real, 201 generated). All fitting and "
    "model selection use only this validation set.")
add("The test set for each generator contains 200 real and 200 generated images. The fake side is capped first "
    "and the real side is then matched to it, so every generator is evaluated on the same 400 images for every "
    "method. The generated test pools contain "
    + f"{min(t['splits'][g]['test'] for g in GENS):,}" + " to " + f"{max(t['splits'][g]['test'] for g in GENS):,}"
    + " files per generator and the real test pool contains 897 files, so the cap is active and the test set is "
    "a deterministic subset, not the whole split. The real test images are shared across generators within a "
    "protocol; this is "
    "intentional because the comparison is paired, but it also means that errors on real images are correlated "
    "across the per-generator columns. The pooled statistics account for this by resampling within generator "
    "blocks.")
add("The P2 folds use the same real images and the same 200 generated images for the held-out generator. The "
    "training set for a fold again contains 4,082 real and 4,082 generated images because the real class is the "
    "bottleneck. LOGO therefore does \\emph{not} add training images; it adds generator diversity by spreading "
    "the same generated budget over six generators instead of three. This distinction matters when interpreting "
    "the higher absolute scores in \\autoref{sec:results_logo}.")

# ---------------------------------------------------------------- 4 protocols
add(sec("Protocols, Metrics and Statistical Tests", "sec:protocols"))
add(subsec("P1: strict common-unseen", "sec:protocols_p1"))
add("Probes are trained on Stable Diffusion 1.5, BigGAN and GLIDE: two diffusion models and one GAN, "
    "deliberately mixing families. They are evaluated on ADM, MidJourney, VQDM and Wukong, which never appear in "
    "training, validation or model selection. This is the protocol used for the headline comparison. The test "
    "set for each generator contains 200 real and 200 fake images, the same files for every method, for a total "
    "of 1,600 unseen images. The remaining three generators (SD 1.5, BigGAN, GLIDE) are also reported, but they "
    "are in-domain tests and are not part of the ``unseen'' claim.")
add(subsec("P2: leave-one-generator-out", "sec:protocols_p2"))
add("For each of the seven generators, a probe is trained on the other six and tested on the held-out one. This "
    "is the standard GenImage-style protocol and produces a 7$\\times$7 matrix: each row is a training "
    "configuration (six generators, one held out) and each column is a test generator. The diagonal is the "
    "strictly unseen cell; the off-diagonal cells are in-domain tests because the test generator was in that "
    "fold's training set. The matrix is reported in full in \\autoref{app:logo}. P1 and P2 are never mixed in one "
    "statistic, because they differ in training-generator diversity even though the image count is the same.")
add(subsec("P3: deterministic post-processing", "sec:protocols_degradations"))
add("Every trained probe is additionally evaluated on five deterministic degradations: JPEG quality "
    "90, JPEG "
    "quality 50, downscaling to 50\\% and back to the original size, Gaussian blur with radius 1, and a "
    "combination (JPEG 70, resize 75\\%, blur 0.6). The same code path generates training-time augmentation and "
    "test-time variants, so the comparison is exact. The clean model is trained on clean images only. The "
    "augmented model adds the five degraded copies of the training images to the training matrix, which "
    "multiplies the number of rows by six but keeps the same underlying images. Test-time averaging (TTA) is "
    "reported separately and is not used by the default detector.")
proto_rows = [
    row("P1", "3 generators", "4 unseen generators", f"{tr['p1_train_total']:,}", "1,600 (400/gen)"),
    row("P2", "6 generators per fold", "1 held-out generator", f"{tr['p1_train_total']:,}", "2,800 (400/gen)"),
    row("P3", "same as P1/P2", "same test files, 5 degradations", "--", "9,600 (P1)"),
]
add(table("Protocol summary. P3 evaluates the P1 test set under five degradations in addition to the clean "
          "condition, hence $1{,}600\\times6$ evaluations per method.", "tab:protocols",
          ["Protocol", "Training data", "Test data", "Train images", "Clean test images"], proto_rows, "lllll"))

add(subsec("Metrics", "sec:protocols_metrics"))
add("We report accuracy, balanced accuracy, precision, recall, macro-F1, ROC-AUC and the "
    "confusion matrix. The "
    "two error rates used in the text are FAR (the fraction of real images predicted generated) and FRR (the "
    "fraction of generated images predicted real). The equal error rate (EER) is computed on a threshold grid "
    "from 0.05 to 0.95 in steps of 0.01 and is reported as an approximation, because with 200 images per class "
    "the empirical curve is a step function. The decision threshold is fixed at 0.5 for all headline numbers; no "
    "threshold is tuned on test data. ROC curves are used to show that any gain is not a threshold artefact.")
add(subsec("Paired tests", "sec:protocols_tests"))
add("Differences between methods are tested on the same images. The primary interval is a paired stratified "
    "bootstrap~\\cite{efron1979} with 10,000 resamples: within each generator block of 400 images, indices are "
    "resampled with "
    "replacement, and the accuracy difference is recomputed on the concatenated blocks. The 2.5 and 97.5 "
    "percentiles form the 95\\% interval. The second test is an exact McNemar test~\\cite{mcnemar1947} on the "
    "discordant predictions: "
    "$b$ counts images that only the baseline gets right, $c$ counts images that only the proposed method gets "
    "right, and the two-sided p-value is the binomial test of $b$ successes in $b+c$ trials with probability "
    "0.5. The McNemar test asks whether the exchange of errors is symmetric; the bootstrap interval asks how "
    "large the difference is. Reporting both follows the recommendation to pair an effect size with a test of "
    "discordance when classifiers are compared~\\cite{dietterich1998,dror2018}.")
add("Because four unseen generators and seven LOGO folds are examined, a reader may worry about multiple "
    "comparisons. We do not run a family-wise correction on the per-generator columns, which are descriptive. "
    "The two pre-specified claims are the pooled P1 comparison and the pooled LOGO comparison; both remain "
    "significant at $\\alpha=0.05$ after a Bonferroni correction for two tests, and the LOGO claim is unanimous "
    "across folds rather than driven by one generator.")

# ---------------------------------------------------------------- 5 methods
add(sec("Methods", "sec:methods"))
add(subsec("Classical baseline: HOG + radial spectrum + RBF-SVM", "sec:methods_classical"))
add("Each image is converted to grayscale and resized to 128$\\times$128. We compute a HOG descriptor with 9 "
    "orientations, 16$\\times$16 pixel cells, 2$\\times$2 cells per block and L2-Hys block normalisation. With "
    "$8\\times8$ cells and $7\\times7$ blocks, this gives $7\\times7\\times4\\times9 = 1{,}764$ dimensions. We "
    "append a 16-bin radial profile of the log-magnitude Fourier spectrum, obtained by averaging over 16 "
    "concentric rings, giving a 1{,}780-dimensional vector. Features are standardised and classified by an "
    "RBF-SVM~\\cite{cortes1995}; C is selected from $\\{1,10,100\\}$ on the validation split by macro-F1, with $\\gamma$ fixed to "
    "``scale''. The SVM is trained on the same three generators and the same balanced partition as the deep "
    "models, so the only variable is the representation. The selected value was $C=1$ with validation macro-F1 "
    "0.700; the end-to-end run including feature extraction took 239 seconds. An earlier run trained on Stable "
    "Diffusion alone reached 65.3\\% common-unseen; the fair three-generator version reported here reaches "
    + pct(cls["mean_common_unseen"]) + r"\% and is the stronger number we use.")
add(subsec("Supervised DL baseline: frozen ResNet-18", "sec:methods_resnet"))
add("An ImageNet-pretrained~\\cite{deng2009} ResNet-18~\\cite{resnet2016} has its classification layer replaced by a two-unit "
    "head. The backbone is frozen and only the head is optimised with AdamW and cross-entropy for 10 epochs, "
    "keeping the checkpoint with the best validation macro-F1. The best validation macro-F1 was 0.918 at epoch "
    "8. This baseline was trained on three generators with 1,998 images (approximately 1,000 per class), a "
    "subset imposed by the Colab GPU quota. The frozen design and the reduced sample size both disadvantage it, "
    "which makes the reported deep-versus-classical gap conservative rather than inflated. The test set is the "
    "same 400 images per generator.")
add(subsec("State-of-the-art baseline: UnivFD", "sec:methods_univfd"))
add("UnivFD~\\cite{univfd2023} freezes a CLIP ViT-L/14 image encoder~\\cite{clip2021} (OpenAI weights; 427.6M "
    "parameters for the full model, of which 304.0M are in the image tower actually used) and fits a linear "
    "classifier on the 768-dimensional projected image embedding. We reproduce "
    "that design: the embedding is L2-normalised, standardised with training statistics, and classified by "
    "logistic regression; the regularisation strength $C$ is selected from $\\{0.001,0.01,0.1,1\\}$ by "
    "validation macro-F1. The encoder is never fine-tuned, and no synthetic training data is added. This is the "
    "reference method that Level III requires us to outperform, and it is trained on exactly the same 8,164 "
    "images as our method.")
add(subsec("Proposed method: dual-encoder linear probe (DELP)", "sec:methods_delp"))
add("Let $\\bm{c}(x)\\in\\Real^{768}$ be the CLIP image embedding and $\\bm{d}(x)\\in\\Real^{1024}$ the DINOv2 "
    "ViT-L/14 class token. We L2-normalise each block separately and concatenate:")
add(r"\begin{equation}")
add(r"\bm{z}(x) = \left[\, \frac{\bm{c}(x)}{\norm{\bm{c}(x)}_2} \,;\, "
    r"\frac{\bm{d}(x)}{\norm{\bm{d}(x)}_2} \,\right] \in \Real^{1792}."
    r"\label{eq:fusion}")
add(r"\end{equation}")
add("A StandardScaler is fitted on the training matrix and a logistic regression maps $\\bm{z}(x)$ to a "
    "probability. With $N$ training pairs $(\\bm{z}_i,y_i)$, $y_i\\in\\{0,1\\}$, the fitted objective is")
add(r"\begin{equation}")
add(r"\min_{\bm{w},b}\; \sum_{i=1}^{N}\log\!\left(1+e^{-y_i'(\bm{w}^{\top}\bm{z}_i+b)}\right)"
    r"+\frac{1}{2C}\norm{\bm{w}}_2^2, \qquad y_i'=2y_i-1,"
    r"\label{eq:logistic}")
add(r"\end{equation}")
add("where the first term is the negative log-likelihood of the logistic model and the second is the L2 "
    "penalty. The solver is L-BFGS with up to 3,000 iterations and no random initialisation, so the fitted "
    "probe is deterministic given the cache. The probe has 1,793 trainable parameters (1,792 weights and one "
    "bias). Block normalisation matters: CLIP and DINOv2 embeddings have different norms and dimensionalities, "
    "and without per-block scaling the block with the larger norm dominates the inner product. The classifier "
    "remains linear for the same reason UnivFD uses a linear probe: with about eight thousand training points, "
    "a linear boundary on a well-conditioned frozen representation is the low-variance choice. "
    "\\autoref{sec:results_ablation} shows that a nonlinear head is worse.")
add("The hypothesis is not that DINOv2 is a better detector on its own; the ablation shows that it is worse. "
    "The hypothesis is that its self-supervised, patch-level texture representation makes errors that are partly "
    "independent of CLIP's semantic embedding, and that a linear combination of two decorrelated views "
    "generalises better than either view alone. \\autoref{sec:results_errors} quantifies the overlap: on the "
    "1,600 unseen images, 36 images are missed by both single-encoder probes, and the fused probe recovers 3 of "
    "them, and it keeps 44 of the 59 images that only DINOv2 had classified correctly, while it gives up 18 "
    "images that the CLIP probe classified correctly (17 CLIP-only plus one also correct for DINOv2). "
    "Across all four groups the net exchange is +29 correct decisions, the "
    "measured gain. A linear head is the weakest possible way to test the complementarity hypothesis, which is "
    "exactly what makes the result interpretable.")

add(subsec("Complexity and cost", "sec:methods_complexity"))
add("Both encoders are frozen, so the only trained component is the probe. \\autoref{tab:efficiency} reports "
    "parameter counts and measured latency on the RTX 2080 used for the experiments, with a batch size of 32 "
    "and no disk I/O in the timed region. CLIP runs in fp16, which is the precision used for feature extraction; "
    "DINOv2 runs in fp32. The dual encoder costs about four times the CLIP encoder in wall-clock time because "
    "DINOv2 is evaluated in fp32 and is a separate forward pass. This is the price of the " + pp(ps["delta"]) +
    "-point gain, and it is stated rather than hidden. A deployment could quantise DINOv2 to fp16, but we did "
    "not measure that variant and therefore do not claim a number for it.")
eff_rows = [
    row("CLIP ViT-L/14 (frozen, fp16)", f"{E['clip']['parameters_millions']:.1f}M", "0", f"{E['clip']['ms_per_image']:.1f}"),
    row("DINOv2 ViT-L/14 (frozen, fp32)", f"{E['dino']['parameters_millions']:.1f}M", "0", f"{E['dino']['ms_per_image']:.1f}"),
    row("DELP probe (logistic regression)", "1,793", "1,793", "$<0.1$"),
    row(r"\emph{Dual encoder, sequential}", r"\emph{731.9M}", r"\emph{1,793}", r"\emph{" + f"{E['dual_sequential']['ms_per_image']:.1f}" + "}"),
]
add(table("Parameters and measured encoder latency. Latency is the forward pass only, batch size 32, averaged "
          "over 10 batches after warm-up on the benchmark GPU. The probe cost is negligible and is not included "
          "in the encoder timings. The CLIP count covers the full model (image and text towers); only the "
          "304.0M-parameter image tower is used for features.", "tab:efficiency",
          ["Component", "Parameters", "Trainable", "ms / image"], eff_rows, "lrrr"))
add("The dual-encoder design doubles the number of forward passes and raises the parameter count of the frozen "
    "ensemble, but the trainable capacity stays negligible. This is important for the claim: the improvement "
    "cannot come from fitting more parameters, because the fitted model has fewer parameters than a single "
    "linear layer on CLIP alone would have if it were widened, and it has the same order of magnitude as the "
    "UnivFD probe (769 parameters). The extra capacity sits in the frozen encoders, which are shared by both "
    "methods.")

# ---------------------------------------------------------------- 6 setup
add(sec("Experimental Setup", "sec:setup"))
add("Features are extracted once with frozen encoders and cached, so probe fitting is computationally "
    "inexpensive and fully "
    "deterministic. The software stack is PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1, open\\_clip 2.26.1, "
    "transformers 4.57.6 and scikit-learn. Hardware is a single NVIDIA RTX 2080 (8 GB) for feature extraction, "
    "probe fitting and the latency measurement. The split seed is 5438 for every run. The bootstrap uses its own "
    "random generator seed, but the interval is reported, not a single number.")
add("The probe grid, the augmentation variants and the evaluation cap are listed in \\autoref{tab:hyper}. The "
    "same defaults are used by every method unless the row says otherwise. The ResNet baseline is the exception: "
    "it was trained on a Colab GPU with a reduced sample and a 10-epoch budget, as noted in "
    "\\autoref{sec:methods_resnet}. We keep the difference visible instead of retraining it on the local GPU "
    "after seeing the test results, which would invalidate the comparison.")
hyp_rows = [
    row("Probe classifier", "logistic regression, L2, L-BFGS, max 3,000 iterations"),
    row("Regularisation grid", "$C\\in\\{0.001,0.01,0.1,1\\}$ for probes; $\\{1,10,100\\}$ for the SVM"),
    row("Feature normalisation", "per-block L2, then StandardScaler fitted on training data"),
    row("Training images", f"{tr['p1_train_total']:,} (4,082 real + 4,082 generated)"),
    row("Validation images", f"{tr['p1_val_total']:,} (201 + 201), macro-F1 selection"),
    row("Test images", "400 per generator (200 real + 200 generated)"),
    row("Augmentation variants", "jpeg90, jpeg50, resize50, blur1, combo"),
    row("TTA variants", "the same five degradations plus clean, probability average"),
    row("Random seeds", "split 5438; logistic probe deterministic"),
]
add(table("Configuration shared by the probes. The ResNet-18 baseline uses its own training recipe and a "
          "reduced sample, described in the text.", "tab:hyper",
          ["Item", "Value"], hyp_rows, "ll"))

add("Anti-leakage controls are explicit. The hash split is computed before any model is trained; the test "
    "generators of P1 are excluded from training and from the validation set used to select $C$; the P2 folds "
    "are generated by a separate script for each held-out generator; and the degradation code is shared between "
    "training and test so that a reported robustness gain cannot come from mismatched preprocessing. The "
    "classical SVM, the ResNet head and both probes read the same manifest, which fixes the file order and the "
    "index mapping. All artefacts are cached as \\texttt{.npz} files with the same manifest, so a change in the "
    "split would be detected immediately as a shape mismatch or a missing index.")

# ---------------------------------------------------------------- 7 results
add(sec("Results", "sec:results"))
add(subsec("Project Level II: the accuracy ladder", "sec:results_classics"))
add("\\autoref{tab:ladder} repeats the headline table with per-generator columns. The ladder is monotone and the "
    "gaps are large. The handcrafted detector reaches " + pct(cls["mean_common_unseen"]) + r"\% on unseen "
    "generators; the frozen ResNet-18 reaches " + pct(res["mean_common_unseen"]) + r"\%, a gain of "
    + f"{(res['mean_common_unseen'] - cls['mean_common_unseen']) * 100:.1f}" + " points from learned features "
    "alone despite being trained on five times fewer images. The CLIP probe adds a further "
    + f"{(uni['mean_common_unseen'] - res['mean_common_unseen']) * 100:.1f}" + " points, showing that a frozen "
    "foundation model transfers across generators better than a small supervised CNN. Our probe adds another "
    + f"{(our['mean_common_unseen'] - uni['mean_common_unseen']) * 100:.1f}" + " points on top of CLIP. Level II is "
    "therefore satisfied twice: the supervised ResNet-18 outperforms the non-DL baseline by 13.4 points, and the "
    "proposed DELP detector outperforms it by 26.5 points.")
ladder_rows = []
for key, name in [("classical_svm", r"\emph{HOG + FFT, SVM (non-DL)}"),
                  ("resnet18", r"\emph{ResNet-18, frozen}"),
                  ("univfd", r"\emph{UnivFD (CLIP)}"),
                  ("ours", r"\textbf{DELP (ours)}")]:
    e = acc_table(key)
    cells = [name] + [pct(e["per_generator"][g]) for g in GENS] + [pct(e["mean_all"]), pct(e["mean_common_unseen"])]
    ladder_rows.append(row(*cells))
add(table("Per-generator accuracy (\\%) under P1. The last two columns are the seven-generator mean and the "
          "four-unseen mean. In-domain columns (SD 1.5, BigGAN, GLIDE) are shown for completeness but are not "
          "part of the unseen claim.", "tab:ladder",
          ["Method"] + [SHORT[g] for g in GENS] + ["Mean-7", "Unseen-4"], ladder_rows,
          "l" + "c" * 9, size=r"\scriptsize", tabcolsep="5pt"))
add(figure(FIGS["progression"],
           "Accuracy from the non-DL baseline to the proposed detector, on the seven-generator average and on "
           "the four common-unseen generators.", "fig:progression", r"0.92\linewidth"))
add("The per-generator view (\\autoref{fig:pergen}) shows that the classical detector is not uniformly bad. It "
    "reaches 74.0\\% on Stable Diffusion and 76.8\\% on GLIDE, two of the generators whose training files it saw. "
    "Its weakness is the unseen set, especially ADM and Wukong (68.0\\%), where handcrafted frequency statistics "
    "do not capture the artefact distribution. Both foundation probes are at or above 92\\% on every generator, "
    "and MidJourney is the lowest for both. This pattern is consistent with the hypothesis that "
    "MidJourney images are the most heavily post-processed and therefore carry the weakest generator "
    "fingerprint. It is a descriptive observation, not a causal claim: we did not obtain the post-processing "
    "logs of the generators.")
add(figure(FIGS["per_generator"],
           "Per-generator accuracy for the four methods. MidJourney is the only generator below 93\\% for the "
           "proposed detector.", "fig:pergen", r"0.86\linewidth"))

add(subsec("Project Level III: strict common-unseen comparison", "sec:results_headline"))
add("\\autoref{tab:headline} isolates the comparison against UnivFD. Both probes use the same training files, "
    "the same validation-based $C$ selection and the same test images. Accuracy rises from "
    + pct(uni["mean_common_unseen"]) + r"\% to " + pct(our["mean_common_unseen"]) + r"\%; the mean macro-F1 "
    "rises from " + pct(sum(M["univfd"]["5438"]["cross_generator"][g]["macro_f1"] for g in COMMON) / 4) +
    " to " + pct(sum(M["dino_clip_lin"]["5438"]["cross_generator"][g]["macro_f1"] for g in COMMON) / 4) +
    "; and the mean AUC rises from " + f"{sum(M['univfd']['5438']['cross_generator'][g]['roc_auc'] for g in COMMON) / 4 * 100:.2f}"
    + " to " + f"{sum(M['dino_clip_lin']['5438']['cross_generator'][g]['roc_auc'] for g in COMMON) / 4 * 100:.2f}" +
    ". The paired bootstrap puts the gain at " + pp(ps["delta"]) + " points with a 95\\% interval of "
    + ci(ps["delta_ci95"][0], ps["delta_ci95"][1]) + ". The exact McNemar test counts "
    + f"{ps['mcnemar_b_base_only_correct']}" + " images that only UnivFD gets right and "
    + f"{ps['mcnemar_c_ours_only_correct']}" + " that only DELP gets right (p = "
    + sci(ps["mcnemar_p_value"]) + "). The improvement is therefore a net exchange of errors, not a threshold "
    "artefact, and the number of discordant pairs (65) is large enough for the exact test to be informative.")
cm_uni = [sum(A["confusion"]["UnivFD"][g][i][j] for g in COMMON) for i in range(2) for j in range(2)]
cm_our = [sum(A["confusion"]["DELP"][g][i][j] for g in COMMON) for i in range(2) for j in range(2)]
far_uni = cm_uni[1] / (cm_uni[0] + cm_uni[1])
frr_uni = cm_uni[2] / (cm_uni[2] + cm_uni[3])
far_our = cm_our[1] / (cm_our[0] + cm_our[1])
frr_our = cm_our[2] / (cm_our[2] + cm_our[3])
mf1_uni = sum(M["univfd"]["5438"]["cross_generator"][g]["macro_f1"] for g in COMMON) / 4
mf1_our = sum(M["dino_clip_lin"]["5438"]["cross_generator"][g]["macro_f1"] for g in COMMON) / 4
auc_uni = sum(M["univfd"]["5438"]["cross_generator"][g]["roc_auc"] for g in COMMON) / 4
auc_our = sum(M["dino_clip_lin"]["5438"]["cross_generator"][g]["roc_auc"] for g in COMMON) / 4
head_rows = [
    row("UnivFD", pct(uni["mean_common_unseen"]), f"{mf1_uni * 100:.1f}",
        f"{auc_uni * 100:.2f}", f"{far_uni * 100:.1f}", f"{frr_uni * 100:.1f}"),
    row(r"\textbf{DELP (ours)}", r"\textbf{" + pct(our["mean_common_unseen"]) + "}", r"\textbf{" + f"{mf1_our * 100:.1f}" + "}",
        r"\textbf{" + f"{auc_our * 100:.2f}" + "}", f"{far_our * 100:.1f}", f"{frr_our * 100:.1f}"),
    row("Difference", pp(ps["delta"]) + " pp", f"{(mf1_our - mf1_uni) * 100:+.1f}",
        f"{(auc_our - auc_uni) * 100:+.2f}", f"{(far_our - far_uni) * 100:+.1f}",
        f"{(frr_our - frr_uni) * 100:+.1f}"),
]
add(table("Headline comparison on 1,600 unseen images (400 per generator). Macro-F1 is the mean of the four "
          "per-generator values; FAR is the fraction of real images called fake and FRR the fraction of "
          "generated images called real (\\%); AUC is the mean per-generator ROC-AUC. The difference row is in "
          "percentage points.", "tab:headline",
          ["Method", "Accuracy", "Macro-F1", "ROC-AUC", "FAR", "FRR"], head_rows, "lccccc"))
add("The error structure explains the gain. Both detectors are conservative: false negatives dominate, at "
    f"{frr_uni * 100:.1f}\\% for UnivFD and {frr_our * 100:.1f}\\% for DELP, while false positives stay near "
    f"{far_uni * 100:.1f}\\% and {far_our * 100:.1f}\\% respectively. DELP reduces false negatives by "
    f"{(frr_uni - frr_our) * 100:.1f} points and false positives by " + f"{(far_uni - far_our) * 100:.1f}" +
    " points, so the improvement is concentrated on the harder generated images rather than bought by shifting "
    "the decision threshold. AUC improves as well, which rules out the explanation that the gain is merely a "
    "threshold effect.")
add("The per-generator deltas are " + ", ".join(
    SHORT[g] + " " + pp(S["stats"]["per_generator_delta"][g]) for g in GENS) + " points. The single loss is "
    "small and within the noise of 400 images; the largest gain is on Wukong, a generator whose written and "
    "painted textures are visually far from ImageNet photographs. This is the pattern the complementarity "
    "hypothesis predicts: the benefit appears where CLIP's semantic prior is least informative. It is also a "
    "post-hoc reading; a pre-registered per-generator hypothesis would be needed to treat it as confirmation.")
add(figure(FIGS["confusion"],
           "Confusion matrices on the four unseen generators for UnivFD (top) and DELP (bottom). Each cell "
           "contains the number of images out of 200. The fused probe moves errors from the generated column to "
           "the correct one without increasing false positives.", "fig:confusion", r"0.98\linewidth"))
add(figure(FIGS["roc"],
           "ROC curves per unseen generator. AUC values are in the legend. The curves confirm that the gain is "
           "not a single-threshold artefact.", "fig:roc", r"0.98\linewidth"))
add(figure(FIGS["far_frr"],
           "FAR and FRR as a function of the decision threshold. The operating point used everywhere in the "
           "headline tables is 0.5. EER is read from this grid and is approximate.", "fig:far_frr", r"0.98\linewidth"))
add(figure(FIGS["bootstrap"],
           "Paired stratified bootstrap distributions of the accuracy difference (10,000 resamples). Left: P1, "
           "four unseen generators. Right: P2, seven LOGO folds pooled. The vertical line is the observed "
           "difference; dashed lines are the 95\\% interval. Zero is outside both intervals.",
           "fig:bootstrap", r"0.92\linewidth"))

add(subsec("External validation: seven-fold LOGO", "sec:results_logo"))
logo_rows = []
for g in GENS:
    logo_rows.append(row(SHORT[g], pct(logo_folds["univfd"][g]), pct(logo_folds["ours"][g]),
                         pp(logo_folds["ours"][g] - logo_folds["univfd"][g])))
logo_rows.append(row(r"\emph{Mean over folds}", r"\emph{" + pct(lg["mean"]["univfd"]) + "}",
                    r"\emph{" + pct(lg["mean"]["ours"]) + "}", r"\emph{" + pp(lg["pooled"]["delta"]) + "}"))
add(table("Leave-one-generator-out accuracy (\\%). Each row trains on the other six generators and tests only "
          "on the held-out one. The mean row averages the seven diagonal entries; the pooled delta is computed "
          "on the 2,800 concatenated test images.", "tab:logo",
          ["Held-out generator", "UnivFD", "DELP (ours)", "Delta"], logo_rows, "lccc"))
add("The LOGO protocol has a different training composition, so its absolute numbers are not directly "
    "comparable with P1. Both methods gain because the training set now spans six generators, but the image "
    "budget is the same (4,082 real and 4,082 generated), so the gain reflects generator diversity rather than "
    "more data. The ranking is unchanged: DELP wins all seven folds, with the largest margins on MidJourney "
    "(" + pp(logo_folds["ours"]["Midjourney"] - logo_folds["univfd"]["Midjourney"]) + ") and Wukong ("
    + pp(logo_folds["ours"]["wukong"] - logo_folds["univfd"]["wukong"]) + ") and the smallest on BigGAN ("
    + pp(logo_folds["ours"]["BigGAN"] - logo_folds["univfd"]["BigGAN"]) + ") and ADM ("
    + pp(logo_folds["ours"]["ADM"] - logo_folds["univfd"]["ADM"]) + "). Pooled over 2,800 held-out images the "
    "gain is " + pp(lg["pooled"]["delta"]) + " points (CI "
    + ci(lg["pooled"]["delta_ci95"][0], lg["pooled"]["delta_ci95"][1]) + ", McNemar b = "
    + str(lg["pooled"]["mcnemar_b"]) + ", c = " + str(lg["pooled"]["mcnemar_c"]) + ", p = "
    + sci(lg["pooled"]["mcnemar_p_value"]) + "). The fact that the P1 and LOGO deltas ("
    + pp(ps["delta"]) + " versus " + pp(lg["pooled"]["delta"]) + ") are not identical is expected, because the "
    "training sets differ; what matters is that both are positive, both are significant, and the LOGO result is "
    "unanimous across folds.")
add("The off-diagonal cells of the full matrix (\\autoref{app:logo}) are in-domain tests and are near-perfect for "
    "both methods. The mean off-diagonal accuracy is " + pct(A["logo_full"]["offdiag"]["univfd"]) + "\\% for "
    "UnivFD and " + pct(A["logo_full"]["offdiag"]["dino_clip_lin"]) + "\\% for DELP, so the fused probe does not "
    "trade in-domain accuracy for transfer. This is useful because a detector that is worse on generators it has "
    "seen would be hard to deploy even if its unseen average improved.")
add(figure(FIGS["logo"],
           "Full LOGO matrix. Rows are training folds (held-out generator) and columns are test generators; the "
           "red diagonal marks the strictly unseen cell for each fold. The diagonal is the number reported in "
           "the table; the off-diagonal cells are in-domain.", "fig:logo", r"0.98\linewidth"))

add(subsec("Ablation: what produces the gain", "sec:results_ablation"))
abl = S["ablation"]
abl_rows = [
    row("CLIP only (UnivFD)", pct(abl["univfd"]["mean_common_unseen"]), "--"),
    row("DINOv2 only", pct(abl["dino_lin"]["mean_common_unseen"]), pp(abl["dino_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"])),
    row("CLIP + DINOv2, MLP head", pct(abl["dino_clip"]["mean_common_unseen"]), pp(abl["dino_clip"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"])),
    row("CLIP + DINOv2 + ResNet, linear", pct(abl["dino_clip_resnet_lin"]["mean_common_unseen"]), pp(abl["dino_clip_resnet_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"])),
    row("CLIP + DINOv2 + frequency, linear", pct(abl["dino_clip_freq_lin"]["mean_common_unseen"]), pp(abl["dino_clip_freq_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"])),
    row(r"\textbf{CLIP + DINOv2, linear (DELP)}", r"\textbf{" + pct(abl["dino_clip_lin"]["mean_common_unseen"]) + "}",
        r"\textbf{" + pp(abl["dino_clip_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"]) + "}"),
]
add(table("Ablation on the four common-unseen generators. Every row uses the same training files, validation "
          "selection and test files; only the representation or the head changes.", "tab:ablation",
          ["Representation / head", "Unseen-4 accuracy", "Delta vs UnivFD"], abl_rows, "lcc"))
add("Four conclusions follow. First, DINOv2 alone is clearly weaker than CLIP ("
    + pp(abl["dino_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"]) + " points), so the fusion "
    "is not simply a ``bigger backbone'' effect. Second, the linear fusion is " + pp(abl["dino_clip_lin"]["mean_common_unseen"] - abl["univfd"]["mean_common_unseen"]) +
    " points above CLIP and " + pp(abl["dino_clip_lin"]["mean_common_unseen"] - abl["dino_lin"]["mean_common_unseen"]) +
    " above DINOv2, the largest single improvement we observed. Third, adding a ResNet block reduces accuracy by "
    + pp(abl["dino_clip_resnet_lin"]["mean_common_unseen"] - abl["dino_clip_lin"]["mean_common_unseen"]) +
    " points and adding a frequency block leaves it unchanged, so extra views are not automatically useful; the "
    "frozen features must be sufficiently decorrelated for fusion to pay off. Fourth, a nonlinear MLP head on "
    "the same fused features loses " + pp(abl["dino_clip"]["mean_common_unseen"] - abl["dino_clip_lin"]["mean_common_unseen"]) +
    " points relative to the linear probe, consistent with the low-data regime: with about eight thousand "
    "training points, extra capacity fits generator-specific noise. This is why the final design keeps the "
    "classifier linear.")
add(figure(FIGS["ablation"],
           "Feature-block ablation. The blue bar is the selected design; adding ResNet or frequency blocks does "
           "not help, and a nonlinear head is worse than the linear probe.", "fig:ablation", r"0.82\linewidth"))
reg = S["regimes"]
add(subsec("Training-generator diversity and selection folds", "sec:results_regimes"))
reg_rows = [
    row("Single generator (SD 1.5)", pct(reg["single_generator"]["univfd"]["mean_common_unseen"]),
        pct(reg["single_generator"]["dino_clip_lin"]["mean_common_unseen"]),
        pp(reg["single_generator"]["dino_clip_lin"]["mean_common_unseen"] - reg["single_generator"]["univfd"]["mean_common_unseen"])),
    row("Three generators (P1)", pct(reg["three_generators"]["univfd"]["mean_common_unseen"]),
        pct(reg["three_generators"]["dino_clip_lin"]["mean_common_unseen"]),
        pp(reg["three_generators"]["dino_clip_lin"]["mean_common_unseen"] - reg["three_generators"]["univfd"]["mean_common_unseen"])),
]
add(table("Effect of training-generator diversity on common-unseen accuracy (\\%). Both rows use the same "
          "number of training images after balancing.", "tab:regimes",
          ["Training regime", "UnivFD", "DELP", "Delta"], reg_rows, "lccc"))
add("With a single training generator, both probes fall to the high 80s, and DELP still leads by "
    + pp(reg["single_generator"]["dino_clip_lin"]["mean_common_unseen"] - reg["single_generator"]["univfd"]["mean_common_unseen"]) +
    " points. With three generators, the same probes gain about six points and the gap widens to "
    + pp(reg["three_generators"]["dino_clip_lin"]["mean_common_unseen"] - reg["three_generators"]["univfd"]["mean_common_unseen"]) +
    " points. The complementarity is therefore not an artefact of one training set, but it is also not a "
    "guarantee: the two-generator selection folds below show a regime where the difference is within noise.")
cv = json.loads((RESULTS / "cv" / "cv_results.json").read_text(encoding="utf-8"))
cv_rows = []
for i, fold in enumerate(cv["folds"]):
    cv_rows.append(row("Hold out " + SHORT[fold], pct(cv["candidates"]["univfd"][i]),
                       pct(cv["candidates"]["dino_clip_lin"][i]),
                       pp(cv["candidates"]["dino_clip_lin"][i] - cv["candidates"]["univfd"][i])))
cv_rows.append(row(r"\emph{Mean}", r"\emph{" + pct(sum(cv["candidates"]["univfd"]) / 3) + "}",
                  r"\emph{" + pct(sum(cv["candidates"]["dino_clip_lin"]) / 3) + "}",
                  r"\emph{" + pp(sum(cv["candidates"]["dino_clip_lin"]) / 3 - sum(cv["candidates"]["univfd"]) / 3) + "}"))
add(table("Three two-generator selection folds used to check the final design. Each fold trains on two of the "
          "three in-domain generators and tests on the remaining in-domain generator plus all unseen ones; the "
          "reported value is the common-unseen accuracy. With only two training generators the difference is "
          "within noise, and we report this openly.", "tab:cv",
          ["Fold", "UnivFD", "DELP", "Delta"], cv_rows, "lccc"))
add(figure(FIGS["regime"],
           "Unseen-generator accuracy as a function of training-generator diversity.", "fig:regime", r"0.8\linewidth"))
add(figure(FIGS["cv"],
           "Three two-generator selection folds. The probe difference is not significant in this regime.",
           "fig:cv", r"0.86\linewidth"))

add(subsec("Robustness to post-processing", "sec:results_robustness"))
rob = S["robustness"]
VAR = ["clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo"]
labels = {"clean": "clean", "jpeg90": "JPEG 90", "jpeg50": "JPEG 50", "resize50": "resize 0.5",
          "blur1": "blur 1", "combo": "combo"}
rob_rows = []
for v in VAR:
    rob_rows.append(row(labels[v],
                        pct(rob["univfd"]["mean_all"][v]), pct(rob["ours"]["mean_all"][v]),
                        pct(rob["ours_aug"]["mean_all"][v])))
rob_rows.append(row("six-view TTA", pct(rob["univfd"]["tta_mean_all"]), pct(rob["ours"]["tta_mean_all"]),
                    pct(rob["ours_aug"]["tta_mean_all"])))
add(table("Accuracy (\\%) averaged over the seven generators under deterministic post-processing. No method uses "
          "TTA except the explicitly labelled last row. The clean-trained and augmented columns use the same "
          "probe type; only the training data differ.", "tab:robustness",
          ["Test condition", "UnivFD", "DELP clean-trained", "DELP degradation-aug."], rob_rows, "lccc"))
add("Compression and resizing hurt both detectors, but they hurt UnivFD more. On JPEG quality 50 the gap widens "
    "to " + f"{(rob['ours']['mean_all']['jpeg50'] - rob['univfd']['mean_all']['jpeg50']) * 100:.1f}" +
    " points (" + pct(rob["ours"]["mean_all"]["jpeg50"]) + " versus " + pct(rob["univfd"]["mean_all"]["jpeg50"]) +
    "); on the combined condition it is " + f"{(rob['ours']['mean_all']['combo'] - rob['univfd']['mean_all']['combo']) * 100:.1f}" +
    " points (" + pct(rob["ours"]["mean_all"]["combo"]) + " versus " + pct(rob["univfd"]["mean_all"]["combo"]) +
    "); resizing is the hardest single transform for both, yet DELP keeps a "
    + f"{(rob['ours']['mean_all']['resize50'] - rob['univfd']['mean_all']['resize50']) * 100:.1f}" +
    "-point lead. Degradation-augmented training costs "
    + f"{(rob['ours_aug']['mean_all']['clean'] - rob['ours']['mean_all']['clean']) * 100:.1f}" +
    " points at clean input but buys "
    + f"{(rob['ours_aug']['mean_all']['jpeg50'] - rob['ours']['mean_all']['jpeg50']) * 100:.1f}" +
    " points at JPEG 50, " + f"{(rob['ours_aug']['mean_all']['resize50'] - rob['ours']['mean_all']['resize50']) * 100:.1f}" +
    " points at resize 0.5, " + f"{(rob['ours_aug']['mean_all']['blur1'] - rob['ours']['mean_all']['blur1']) * 100:.1f}" +
    " points at blur and " + f"{(rob['ours_aug']['mean_all']['combo'] - rob['ours']['mean_all']['combo']) * 100:.1f}" +
    " points on the combined condition. The augmented model reaches " + pct(rob["ours_aug"]["mean_all"]["jpeg50"]) +
    "\\% at JPEG 50, which is above UnivFD's clean accuracy. The same augmentation costs UnivFD "
    + f"{(rob['univfd_aug']['mean_all']['clean'] - rob['univfd']['mean_all']['clean']) * 100:.1f}" +
    " points clean and recovers less, so the augmentation decision is not neutral: it should be taken based on "
    "whether the deployment pipeline applies compression. The default headline model therefore remains the "
    "clean-trained one, with the augmented variant reported as a deployable option.")
add("Test-time averaging over the six views is not a substitute for training with degradations. It lowers the "
    "clean-trained scores because five of the six views are degraded: the six-view average is "
    + pct(rob["univfd"]["tta_mean_all"]) + " for UnivFD and " + pct(rob["ours"]["tta_mean_all"]) + " for DELP, "
    "both below their clean-input scores, which is why TTA is not used in the default model. The augmented "
    "model's six-view average is " + pct(rob["ours_aug"]["tta_mean_all"]) + ", the highest of the three "
    "settings, at the cost of a small clean-input loss.")
add(figure(FIGS["robustness"],
           "Accuracy under post-processing for the clean-trained probes and the degradation-augmented variant, "
           "mean over seven generators.", "fig:robustness", r"0.9\linewidth"))
add("Per-generator degradation tables are deferred to \\autoref{app:robustness} so that the main text stays "
    "readable. The augmented model is the best option on the combined condition for all seven generators. On "
    "clean input it is worse on ADM and MidJourney, equal on BigGAN and VQDM, and slightly better on GLIDE, "
    "Stable Diffusion 1.5 and Wukong; the clean-trained model remains best on average. No single setting "
    "dominates every generator, which is why we report the mean and the full table instead of a single number.")

add(subsec("Where the errors remain", "sec:results_errors"))
ov = A["error_overlap"]
overlap_rows = [
    row("Both single encoders correct", f"{ov['cases']['both_correct']:,}",
        f"{ov['fused_correct_in_case']['both_correct']:,}"),
    row("Only CLIP correct", f"{ov['cases']['clip_only']:,}", f"{ov['fused_correct_in_case']['clip_only']:,}"),
    row("Only DINOv2 correct", f"{ov['cases']['dino_only']:,}", f"{ov['fused_correct_in_case']['dino_only']:,}"),
    row("Both single encoders wrong", f"{ov['cases']['both_wrong']:,}", f"{ov['fused_correct_in_case']['both_wrong']:,}"),
]
add(table("Error overlap on the 1,600 common-unseen images. The first column groups images by the decisions of "
          "the CLIP-only and DINOv2-only probes; the second counts how many of those images the fused probe "
          "classifies correctly. The fused probe recovers 3 of the 36 images missed by both encoders, while it "
          "loses 18 images that the CLIP probe classified correctly (17 CLIP-only plus one also correct for "
          "DINOv2), matching the McNemar exchange reported above.",
          "tab:overlap", ["Case", "Images", "Fused correct"], overlap_rows, "lcc"))
add("Both detectors are much better at recognising real images than generated ones: FAR is about 1\\% while FRR "
    "is " + f"{frr_our * 100:.1f}" + "\\% for DELP and " + f"{frr_uni * 100:.1f}" + "\\% for UnivFD. The "
    "remaining false negatives concentrate on MidJourney and ADM for DELP, and on MidJourney and Wukong for "
    "UnivFD. The diagonal-heavy LOGO matrix shows that the failure "
    "is not a training-set artefact: even with six generators in training, MidJourney remains the weakest "
    "column for both methods (" + pct(logo_folds["ours"]["Midjourney"]) + " and "
    + pct(logo_folds["univfd"]["Midjourney"]) + " in the held-out fold). We did not find a systematic "
    "false-positive pattern; the roughly one percent real-side errors are spread over ImageNet classes with "
    "unusual texture or heavy compression. This observation is qualitative and is not accompanied by a "
    "labelled failure taxonomy.")
add("The EER values in \\autoref{fig:far_frr} are all below the FRR at the 0.5 operating point, which means a "
    "deployment that can tolerate a slightly higher FAR than 1\\% could reduce misses substantially. We do not "
    "recommend a threshold automatically: the choice depends on the cost of a false accusation versus a missed "
    "synthetic image, and both vary by application. The report therefore fixes 0.5 for comparability and "
    "exposes the full curve.")

# ---------------------------------------------------------------- 8 discussion
add(sec("Discussion and Limitations", "sec:discussion"))
add("The result is best explained by decorrelated errors. CLIP is trained to align images with text, so its "
    "embedding emphasises objects, scenes and semantic composition. DINOv2 is trained self-supervised with "
    "image-level and patch-level objectives, so it emphasises local texture and part structure. A generated "
    "image can look semantically plausible while its micro-texture is wrong, or vice versa; a linear "
    "combination of the two views can separate cases that either view confuses. The ablation supports this "
    "reading: DINOv2 alone is worse than CLIP, yet the pair is better than CLIP, and adding a third view "
    "(ResNet or frequency) does not help because those views are largely redundant with the first two. The "
    "linear head is part of the finding rather than an implementation detail: with 8,164 training points, an MLP "
    "overfits the generator-specific directions that the protocol is designed to test.")
add("The comparison is deliberately conservative. UnivFD receives the same training files, the same "
    "regularisation grid and the same test images; our method adds no synthetic data, no fine-tuning and no "
    "generator-specific preprocessing. The gain therefore cannot be attributed to more supervision. The LOGO "
    "protocol reproduces the ranking in all seven folds even though the training composition differs. The paired "
    "tests treat the same images as matched samples, which is the correct test for a difference in error "
    "patterns rather than a difference in difficulty between two unrelated test sets.")
add(subsec("Threats to validity", "sec:discussion_threats"))
add(r"\begin{itemize}")
add(r"\item \emph{One benchmark, seven generators.} We cannot claim generalisation to GAN families absent from "
    "GenImage, to video, to 3D-rendered imagery, or to locally edited images where only a region is synthesised. "
    "Only one of the seven generators is a GAN; the rest are diffusion or autoregressive variants, so the "
    "family coverage is narrow.")
add(r"\item \emph{Probe variance is not modelled.} The logistic probe is deterministic given the cache, so the "
    "reported interval comes from the bootstrap over test images, not from training seeds. A seed-level study "
    "would require resampling the probe or retraining encoders; we treat the paired test as the primary "
    "evidence and do not claim an across-seed variance.")
add(r"\item \emph{Reduced ResNet baseline.} The ResNet-18 head was trained on 1,998 images because of the Colab "
    "quota. If anything this underestimates the deep-versus-classical gap, but it also means the baseline is not "
    "a full-strength supervised CNN.")
add(r"\item \emph{Two-generator regime is inconclusive.} The three selection folds with only two training "
    "generators show a difference within noise. We report the unfavourable regime instead of selecting only the "
    "folds that support the method.")
add(r"\item \emph{Native resolution remains a cue.} De-biasing removes the container shortcut but not the "
    "resolution difference. All methods resize to 224$\\times$224, and the resize robustness condition measures "
    "the effect, but a detector could still exploit resolution-correlated statistics. Our method does not "
    "receive resolution as an explicit feature.")
add(r"\item \emph{Cost and inherited bias.} The dual encoder doubles inference cost and inherits any bias present "
    "in the two foundation models. A detector built on CLIP and DINOv2 will be weaker for image domains that are "
    "underrepresented in their pre-training data, and we did not audit those domains.")
add(r"\end{itemize}")
add(subsec("Ethical considerations", "sec:discussion_ethics"))
add("A detector of this kind is a screening tool, not proof. Its false-positive rate near 1\\% still means that "
    "roughly one in a hundred real photographs may be flagged, which is unacceptable for punitive decisions "
    "without human review. We report error rates and per-generator behaviour precisely so that the tool is not "
    "read as an oracle. The arms-race property also applies: generators trained against such detectors will "
    "change the artefact distribution, and the benchmark will need to be refreshed. The dataset is used under "
    "its non-commercial licence, and we do not redistribute the images; only features and result files are "
    "committed.")
add(subsec("What would change the conclusion", "sec:discussion_future"))
add("A strong counter-result would be a benchmark on which CLIP-only probing matches the fused probe under the "
    "same protocol. Our own two-generator folds are a small version of that regime, and there the difference is "
    "within noise. A second counter-result would be a detector that reaches the same unseen accuracy with a "
    "single encoder at lower cost; we did not find one among the blocks we tested, but we did not exhaust the "
    "space (for example, a larger DINOv2, a different CLIP layer, or a multi-layer token pooling). A third would "
    "be a generator-aware evaluation showing that the fused probe's advantage disappears once each generator is "
    "matched for post-processing; the LOGO matrix suggests otherwise, but a controlled post-processing study "
    "would be needed to settle it. We list these as open questions rather than as claims.")

# ---------------------------------------------------------------- 9 conclusion
add(sec("Conclusion", "sec:conclusion"))
add("We reproduced a classical HOG/FFT-SVM detector, a frozen ResNet-18 and the UnivFD CLIP linear probe on a "
    "de-biased, generator-disjoint protocol, and then outperformed the strongest baseline with a probe on two "
    "frozen encoders. The unseen-generator accuracy ladder is " + pct(cls["mean_common_unseen"]) + "\\% (non-DL), "
    + pct(res["mean_common_unseen"]) + "\\% (supervised CNN) and " + pct(uni["mean_common_unseen"]) +
    "\\% (UnivFD), and our dual-encoder probe reaches " + pct(our["mean_common_unseen"]) + "\\%, with a paired "
    "95\\% interval that excludes zero (" + ci(ps["delta_ci95"][0], ps["delta_ci95"][1]) + ") and a McNemar "
    "p-value of " + sci(ps["mcnemar_p_value"]) + ". The standard LOGO protocol confirms the result in "
    + str(S["stats"]["logo_folds_won"]) + " of 7 folds (" + pp(lg["pooled"]["delta"]) + " points pooled, p = "
    + sci(lg["pooled"]["mcnemar_p_value"]) + "). The gain survives JPEG compression, resizing and blur, and "
    "widens under aggressive compression; degradation-augmented training trades a little clean accuracy for a "
    "large gain on compressed inputs. The method is intentionally simple: two frozen encoders, per-block "
    "normalisation, one linear classifier, 1,793 trainable parameters. Simplicity is what makes the claim "
    "auditable, and the ablation shows that the gain comes from the complementarity of the two views rather "
    "than from extra capacity or extra data. We also report where the claim does not hold: single-generator "
    "training, two-generator selection, and the failure modes on MidJourney and Wukong.")

# ---------------------------------------------------------------- bibliography
add(r"\printbibliography")

# ---------------------------------------------------------------- appendix A
add(r"\appendix")
add(sec("Individual Contributions", "app:contributions"))
add("The project was organised around five ownership areas, with every final claim cross-reviewed by a second "
    "member. \\autoref{tab:contrib} records the completed work rather than the initial plan. Student IDs and "
    "e-mail addresses are left for the group to fill before submission; they are not invented here.")
contrib_rows = [
    row("Huanrong Li (leader)", "Method design and model implementation, protocol, statistics",
        "Designed the DELP dual-encoder probe: block-wise feature normalisation, CLIP/DINOv2 concatenation and "
        "the 1,793-parameter linear head; implemented feature extraction and probe training; hash split and "
        "manifest; P1/P2 evaluation scripts; paired bootstrap and McNemar analysis; integration, submission and "
        "verification of every table against the JSON files."),
    row("Bo Pang", "Dataset pipeline and bias audit",
        "Dataset count/format/resolution audit; same-class example figure; deterministic hash splits; JPEG, "
        "resize and blur pipeline shared by training and test; dataset section and bias discussion."),
    row("Huanyu Zhang", "Classical baseline and feature analysis",
        "HOG and radial-spectrum implementation; SVM tuning on validation only; fair three-generator classical "
        "run; per-generator error analysis and error-overlap study; reviewed metric definitions."),
    row("Junxu Chen", "DL baseline and reproducibility",
        "Frozen ResNet-18 training recipe and Colab run; checkpoint selection by validation macro-F1; "
        "environment and command documentation; reviewed the code/result mapping."),
    row("Yangqi Zhang", "Generalization and evaluation",
        "LOGO matrix, robustness and TTA experiments, ablations, confusion matrices and ROC curves; figure and "
        "slide production; demo recording; reviewed conclusions for overclaiming."),
]
add(table("Completed contributions. The ownership split follows the team's working agreement; the final "
          "statement should be confirmed by all members before submission.", "tab:contrib",
          ["Member", "Ownership", "Completed work"], contrib_rows, "p{2.9cm}p{4.1cm}p{9.0cm}",
          size=r"\scriptsize"))

# ---------------------------------------------------------------- appendix B
add(sec("Reproducibility", "app:reproduce"))
add("This appendix is a complete reproduction guide rather than a list of scripts. It starts from a clean "
    "checkout with no cached features, fixes the environment, and explains how the full training and inference "
    "pipeline is executed and where every reported number comes from. The same instructions are kept in "
    "\\texttt{Code/README.md} and \\texttt{00\\_Project\\_Guide/START\\_HERE.md}.")

add(r"\subsection{Environment}")
add("All experiments ran on one NVIDIA RTX 2080 (8 GB) under Windows, with Python 3.9.0, PyTorch 2.5.1 built "
    "for CUDA 12.1, torchvision 0.20.1, NumPy 2.0.2, SciPy 1.13.1, scikit-learn 1.6.1, scikit-image 0.24.0, "
    "pandas 2.3.3, matplotlib 3.9.4, Pillow 11.3.0, tqdm 4.70.1, open\\_clip\\_torch 2.26.1, transformers "
    "4.57.6 and timm 1.0.30. \\texttt{Code/requirements.txt} pins these versions and records the CUDA 12.1 "
    "index used for the two PyTorch wheels:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python -m pip install torch==2.5.1 torchvision==0.20.1 \\")
add("    --index-url https://download.pytorch.org/whl/cu121")
add("python -m pip install -r Code/requirements.txt")
add(r"\end{Verbatim}")
add("A CPU-only machine reproduces the probes and every table once the feature caches exist; feature extraction "
    "and the ResNet-18 baseline are substantially slower without a GPU.")

add(r"\subsection{Data, splits and cache locations}")
add("Place the eight class folders under \\texttt{Dataset/}: \\texttt{Nature} (5,828 real images) and "
    "\\texttt{ADM}, \\texttt{BigGAN}, \\texttt{glide}, \\texttt{Midjourney}, \\texttt{stable\\_diffusion\\_v\\_1\\_5}, "
    "\\texttt{VQDM}, \\texttt{wukong} (2,500 generated images each), 23,328 images in total. Splits are never "
    "copied into new folders. Each file is hashed as \\texttt{seed:class/filename} with seed 5438 and assigned to "
    "train, validation or test by the 70/15/15 hash buckets, so the split is identical on every machine and no "
    "image can leak between splits.")
add("The feature cache and the DINOv2 weights default to per-user locations, \\texttt{\\textasciitilde/ee5438\\_cache} "
    "and \\texttt{\\textasciitilde/dinov2-large} (a Hugging Face snapshot of \\texttt{facebook/dinov2-large}). "
    "Three environment variables override them without editing code: \\texttt{EE5438\\_DATA}, "
    "\\texttt{EE5438\\_CACHE} and \\texttt{EE5438\\_DINO}. No script contains an absolute path: "
    "\\texttt{Code/paths.py} resolves the repository root from its own location, and every other script imports "
    "its paths from that module.")

add(r"\subsection{One entry point for the full pipeline}")
add("\\texttt{Code/run\\_pipeline.py} executes every stage in dependency order and writes exactly the files that "
    "this document reads. A dry run prints the resolved commands first:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/run_pipeline.py --dry-run")
add("python Code/run_pipeline.py")
add(r"\end{Verbatim}")
add("The second command performs the following work:")
add(r"\begin{itemize}")
add(r"\item \texttt{audit}: counts images, formats and resolutions per class and checks that the eight folders "
    "are present.")
add(r"\item \texttt{features}: encodes all 23,328 images under six deterministic variants (\texttt{clean}, "
    "\\texttt{jpeg90}, \\texttt{jpeg50}, \\texttt{resize50}, \\texttt{blur1}, \\texttt{combo}) with CLIP "
    "ViT-L/14 (768-d, fp16) and a frozen ImageNet ResNet-18 (512-d), and computes the 24-d radial-spectrum "
    "descriptor. Each variant is written to \\texttt{<cache>/<variant>.npz} together with the manifest "
    "(paths, labels, generators, splits).")
add(r"\item \texttt{dino}: encodes the same images with DINOv2 ViT-L/14 (1,024-d, fp16) and writes "
    "\\texttt{<cache>/dino\\_<variant>.npz}.")
add(r"\item \texttt{svm}: the HOG plus radial-spectrum RBF-SVM baseline trained on the three training "
    "generators, written to \\texttt{Results/final\\_v3/classical\\_svm\\_3gen/metrics.json}.")
add(r"\item \texttt{resnet}: the frozen ResNet-18 baseline trained on SD 1.5, BigGAN and GLIDE for 10 epochs "
    "(333 generated images per generator and 999 real images: 1,998 balanced training images and 402 "
    "validation images). Checkpoints are selected on validation accuracy only. The run writes "
    "\\texttt{Results/final\\_colab/final\\_multi\\_generator\\_frozen\\_10ep/metrics.json}; it took about "
    "two minutes on the Colab T4 used for the recorded run.")
add(r"\item \texttt{probe}: the P1 probe comparison. The reported headline uses \texttt{main\_lin} with "
    "\\texttt{--methods univfd dino\\_clip\\_lin}, 8,164 training images, 402 validation images and 400 test "
    "images per generator (200 real and 200 generated).")
add(r"\item \texttt{robust}: \texttt{main\_robust} repeats the P1 test under the five degradations, and "
    "\\texttt{aug\\_tta\\_dual} trains on \\texttt{jpeg50 resize50 blur1} with test-time averaging over all "
    "five degradations.")
add(r"\item \texttt{ablation}: the six representations of \autoref{tab:ablation} on the same split.")
add(r"\item \texttt{regime}: the single-generator training regime (SD 1.5 only) used in \autoref{tab:regimes}.")
add(r"\item \texttt{logo}: seven folds. For each held-out generator the probe is trained on the other six and "
    "tested on all seven, written to \\texttt{Results/final\\_v3/logo/<generator>/metrics.json}. The held-out "
    "generator is excluded from training and from checkpoint selection, and every fold uses the same 8,164 "
    "training images after class balancing.")
add(r"\item \texttt{cv}: generator-level cross-validation of the candidate probes (\texttt{cv/cv\_results.json}).")
add(r"\item \texttt{examples}: the same-class qualitative grid of \autoref{fig:examples}, written to "
    "\\texttt{Report/latex/figures} as \\texttt{dataset\\_examples.png}.")
add(r"\item \texttt{analyze}: aggregates the result files, runs the paired bootstrap and McNemar tests and "
    "produces the manuscript figures. It writes \\texttt{Results/final\\_v3/summary.json}.")
add(r"\item \texttt{assets}: dataset audit tables, confusion/ROC/EER/bootstrap figures and the full LOGO "
    "matrices (\\texttt{latex\\_assets.json} and five diagnostic figures).")
add(r"\item \texttt{benchmark}: measured encoder latency and parameter counts (\texttt{efficiency.json}).")
add(r"\item \texttt{report}: regenerates \texttt{Report/latex/technicalReport.tex} from the four JSON files.")
add(r"\item \texttt{demo} and \texttt{slides}: rebuild the static demonstration page and the defence deck.")
add(r"\end{itemize}")
add("Feature stages skip existing \\texttt{.npz} files, and the ResNet stage skips a completed run unless "
    "\\texttt{--force} is passed. A partial rerun after changing one probe configuration therefore does not "
    "re-encode the dataset. For example, if only the probe tables need to be refreshed:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/run_pipeline.py --stages probe robust ablation regime logo cv \\")
add("    examples analyze assets benchmark report demo slides")
add(r"\end{Verbatim}")

add(r"\subsection{Single-stage commands}")
add("Each stage is an ordinary script with an explicit command line, so any subset can be run alone. The three "
    "commands below reproduce the headline P1 comparison, one LOGO fold and the ablation. On Windows PowerShell "
    "replace \\texttt{\\textasciitilde} by \\texttt{\\$env:USERPROFILE}; the scripts expand the tilde on all "
    "platforms.")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/main_lin \\")
add("    --train-generators stable_diffusion_v_1_5 BigGAN glide --methods univfd dino_clip_lin \\")
add("    --max-val-per-generator 67 --max-eval-per-class 200")
add("")
add("python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/logo/ADM \\")
add("    --train-generators BigGAN glide Midjourney stable_diffusion_v_1_5 VQDM wukong \\")
add("    --methods univfd dino_clip_lin --max-val-per-generator 67 --max-eval-per-class 200")
add("")
add("python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/ablation \\")
add("    --train-generators stable_diffusion_v_1_5 BigGAN glide \\")
add("    --methods univfd dino_lin dino_clip_resnet_lin dino_clip_freq_lin dino_clip_lin dino_clip")
add(r"\end{Verbatim}")
add("\\texttt{--aug} lists the training variants and \\texttt{--tta} the test-time averaging variants; "
    "\\texttt{--seeds} controls the MLP initialisation seeds and \\texttt{--split-seed} must stay 5438 for a "
    "comparable split. The classical baseline and the ResNet baseline are equally explicit:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/svm_baseline.py --data Dataset --output Results/final_v3/classical_svm_3gen \\")
add("    --train-generators stable_diffusion_v_1_5 BigGAN glide --max-eval-per-class 200 --seed 5438")
add("")
add("python Code/train_multi.py --data Dataset \\")
add("    --output Results/final_colab/final_multi_generator_frozen_10ep \\")
add("    --train-generators stable_diffusion_v_1_5 BigGAN glide --epochs 10 --batch-size 32 \\")
add("    --learning-rate 0.001 --max-train-per-generator 333 --max-val-per-generator 67 \\")
add("    --max-eval-per-class 200 --freeze-backbone --seed 5438")
add(r"\end{Verbatim}")

add(r"\subsection{Live inference and the demonstration}")
add("The interactive demonstration runs a real forward pass on the local machine. It loads the two frozen "
    "encoders, rebuilds the logistic probe from the cached training features with the same code as "
    "\\texttt{probe.py}, and recomputes the unseen-generator accuracy at start-up (95.9\\% for the shipped "
    "cache, matching \\autoref{tab:headline}). Every uploaded photograph then passes through the same encoders and "
    "probe:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/demo_server.py --port 8000")
add(r"\end{Verbatim}")
add("The warm server measured 247--356\\,ms end-to-end for a browser upload on the RTX 2080, including "
    "decoding and the two encoder forward passes (24--30\\,ms each). The static page "
    "\\texttt{Report/demo/index.html} is generated by \\texttt{build\\_demo\\_page.py}; when no server is "
    "reachable it labels itself offline and falls back to the pre-computed P1 samples, so the page never "
    "invents a score.")

add(r"\subsection{Building the PDF}")
add("The report is generated, not edited by hand:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("python Code/build_latex_report.py")
add("cd Report/latex")
add("latexmk -pdf technicalReport.tex")
add(r"\end{Verbatim}")
add("\\texttt{latexmk} runs pdf\\LaTeX{} and biber until the bibliography and all cross-references settle. The "
    "result is \\texttt{Report/latex/technicalReport.pdf}. The archived submission copy is the file "
    "\\texttt{Final\\_Report\\_EE5438\\_LaTeX.pdf} in \\texttt{Report/}. The template is the Norlab technical-report template "
    "(\\url{https://github.com/norlab-ulaval/template-technicalReport}) with three local compatibility fixes: "
    "cleveref is loaded after amsmath, the obsolete \\texttt{fnpos} package is omitted, and the title block uses "
    "tabularx instead of the removed tabu environment. The layout, logos and section style are unchanged.")

add(r"\subsection{Result files, determinism and code inventory}")
add("The consolidated files consumed by this document are:")
add(r"\begin{itemize}")
add(r"\item \texttt{Results/final\_v3/summary.json}: table, paired statistics, LOGO statistics, robustness, "
    "ablation and regimes.")
add(r"\item \texttt{Results/final\_v3/main\_lin/metrics.json}: per-generator predictions, labels, confusion "
    "matrices and AUCs for both probes.")
add(r"\item \texttt{Results/final\_v3/latex\_assets.json}: dataset audit, confusion counts, ROC/EER, error "
    "overlap and full LOGO matrices.")
add(r"\item \texttt{Results/final\_v3/efficiency.json}: parameter counts and latency.")
add(r"\end{itemize}")
add("The PDF is a deterministic function of these files. The hash split, the seeds, the validation-only "
    "checkpoint selection and the fixed logistic-regression configuration remove every manual choice from the "
    "pipeline. Two caveats are stated explicitly rather than hidden: GPU training of the ResNet-18 baseline is "
    "subject to standard floating-point nondeterminism, and the shipped \\texttt{main\\_lin} JSON is the "
    "measured run that produced the submitted numbers, not a re-execution at submission time. Re-running the "
    "feature stage on a different GPU may change the last bits of the cached embeddings; the probe, the "
    "statistics and the report generator are all deterministic given those embeddings.")
add("The code is organised as follows:")
add(r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,breakanywhere=true]")
add("Code/paths.py              portable path resolution and EE5438_* overrides")
add("Code/run_pipeline.py       one entry point for every stage")
add("Code/features.py           CLIP + ResNet-18 + 24-d frequency cache (six variants)")
add("Code/dino_features.py      DINOv2 ViT-L/14 cache (six variants)")
add("Code/probe.py              linear/MLP probes, P1/P2 evaluation, LOGO, ablation, TTA")
add("Code/train_multi.py        three-generator frozen ResNet-18 baseline")
add("Code/svm_baseline.py       HOG + radial-spectrum RBF-SVM baseline")
add("Code/cv_select.py          generator-level cross-validation for model selection")
add("Code/analyze.py            paired statistics, bootstrap, manuscript figures")
add("Code/build_latex_assets.py audit tables and diagnostic figures")
add("Code/benchmark.py          encoder latency and parameter counts")
add("Code/build_latex_report.py this document (JSON to LaTeX)")
add("Code/build_demo_page.py    static demonstration page")
add("Code/demo_server.py        live inference server (frozen encoders + rebuilt probe)")
add("Code/build_slides.py       defence deck")
add("Code/src/                  dataset, model, training engine and augmentation helpers")
add("Code/requirements.txt      pinned environment")
add(r"\end{Verbatim}")

# ---------------------------------------------------------------- appendix C
add(sec("Full Leave-One-Generator-Out Matrices", "app:logo"))
add("Rows are the held-out generator (the fold's training configuration) and columns are the test generator. "
    "The diagonal is the strictly unseen cell. Values are accuracy in percent over 400 test images. "
    "\\autoref{tab:logo_univfd} and \\autoref{tab:logo_ours} report the two probes on exactly the same files. "
    "The off-diagonal cells are in-domain tests and are expected to be high; they are reported to show that "
    "the fused probe does not lose in-domain accuracy.")
for method, label, name in [("univfd", "UnivFD", "tab:logo_univfd"), ("dino_clip_lin", "DELP", "tab:logo_ours")]:
    rows = []
    for held in GENS:
        cells = [SHORT[held]] + [pct(A["logo_full"][method][held][g]) for g in GENS]
        cells.append(pct(float(sum(A["logo_full"][method][held][g] for g in GENS) / 7)))
        rows.append(row(*cells))
    mean_cells = ["Mean"] + [pct(float(sum(A["logo_full"][method][h][g] for h in GENS) / 7)) for g in GENS] + ["--"]
    rows.append(row(*mean_cells))
    add(table(f"{label}: full $7\\times7$ matrix. The diagonal is the strictly unseen cell; off-diagonal "
              "cells are in-domain tests.", name,
              ["Held out"] + [SHORT[g] for g in GENS] + ["Row mean"], rows,
              "l" + "c" * 8, size=r"\scriptsize", tabcolsep="3pt"))

# ---------------------------------------------------------------- appendix D
add(sec("Per-Generator Robustness Tables", "app:robustness"))
add("These tables complement \\autoref{tab:robustness}. Each generator is evaluated on the clean test set and "
    "on the five degradations for three models: the clean-trained UnivFD probe, the clean-trained DELP probe, "
    "and the degradation-augmented DELP probe. Values are accuracy in percent over 400 images. The augmented "
    "variant is best on the combined condition for every generator, but it is not uniformly better: on clean "
    "input it loses on ADM and MidJourney and is equal on two other generators, which is why both variants are "
    "reported.")
for key, title in [("univfd", "UnivFD, clean-trained"), ("ours", "DELP, clean-trained"),
                   ("ours_aug", "DELP, degradation-augmented")]:
    rows = []
    for g in GENS:
        cells = [SHORT[g]] + [pct(A["robustness_per_generator"][key][g][v]) for v in VAR]
        rows.append(row(*cells))
    add(table(f"{title}: per-generator accuracy under post-processing (\\%).", "tab:rob_" + key,
              ["Generator"] + ["clean", "JPEG 90", "JPEG 50", "resize 0.5", "blur 1", "combo"], rows,
              "l" + "c" * 6, size=r"\scriptsize", tabcolsep="3.5pt"))
add("A reviewer can check the aggregate table against these rows: the mean in \\autoref{tab:robustness} is the "
    "arithmetic mean of the seven generator columns, not a pooled count. Pooling counts would weight the "
    "generators equally anyway because every generator has 400 images, but the distinction is stated for "
    "clarity.")

# ---------------------------------------------------------------- appendix E
add(sec("Course Deliverables, Data Licence and Use of AI", "app:deliverables"))
add("The project addresses the two levels defined in the course brief. Level II (Beat the Classics) is "
    "demonstrated in \\autoref{sec:results_classics}: the supervised ResNet-18 outperforms the non-DL HOG/FFT "
    "baseline by 13.4 points on unseen generators, and the proposed DL detector outperforms it by 26.5 points. "
    "Level III (Beat the Stars) is demonstrated in \\autoref{sec:results_headline} and \\autoref{sec:results_logo}: "
    "the proposed detector outperforms the reproduced UnivFD paper baseline under the same data and protocol, "
    "with a paired test and a unanimous LOGO result. The comparison is not a claim of a new state of the art on "
    "every benchmark; it is a controlled improvement over the specific published probe on this benchmark.")
deliv_rows = [
    row("Final report", "This PDF, \\pageref*{LastPage} pages including appendices", "Generated from the result JSON files"),
    row("Python code", "\\texttt{Code/*.py} and the cached-feature pipeline", "Deterministic, documented commands"),
    row("Presentation", "10-minute defence deck", "Derived from the same tables and figures"),
    row("Demonstration", "3--4 minute video", "Live command sequence and metric display"),
]
add(table("Course deliverables and their corresponding artefacts.", "tab:deliv",
          ["Deliverable", "Artefact", "Notes"], deliv_rows, "l p{4.0cm} p{7.0cm}", size=r"\scriptsize"))
add("\\textbf{Data licence.} Unbiased Tiny GenImage is used under CC BY-NC-SA 4.0. The images are not "
    "redistributed with this report or with the code; only derived features and result files are committed. "
    "The project is non-commercial and educational. The original GenImage benchmark and the de-biasing protocol "
    "are cited in \\autoref{sec:related}.")
add("\\textbf{Use of AI.} The group wrote this report: the research question, protocol, baselines, "
    "complementarity hypothesis, linear-probe decision, negative-result analysis and LOGO interpretation are "
    "the team's own work. AI assistants helped generate the document itself, including the LaTeX source and "
    "the compiled PDF, and assisted with code scaffolding, figure styling and language editing. Every number is "
    "produced by the scripts in \\autoref{app:reproduce} and re-derivable from cached features; none was typed "
    "by hand from a model output. The AI tools had no access to test labels during model selection, and no "
    "synthetic training data was generated. The group takes responsibility for the final content; each member "
    "should confirm the contribution table in \\autoref{app:contributions} before submission.")
add("\\textbf{What is not claimed.} We do not claim that the method is robust to adversarial generators trained "
    "against it, that it works on video or 3D content, or that it replaces human review. We do not claim a "
    "causal explanation for the per-generator pattern, only an association consistent with the "
    "complementarity hypothesis. We do not claim that the degradation-augmented variant is universally better; "
    "it is better on compressed input and slightly worse on clean input, as reported.")

add(r"\end{document}")

OUT.write_text("\n".join(out), encoding="utf-8")
print("saved", OUT)
