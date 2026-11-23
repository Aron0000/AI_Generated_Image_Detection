from __future__ import annotations

"""Aggregate results, run paired significance tests and build report figures.

Figures follow one journal-style visual system: serif type to match the
manuscript, a muted five-colour palette, no in-figure titles (the LaTeX
caption is the title), thin 0.6 pt rules and 300 dpi output.  The same
palette is used by build_latex_assets.py so the whole report reads as one
document.
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results" / "final_v3"
FIG = ROOT / "Report" / "figures_final"
LATEX_FIG = ROOT / "Report" / "latex" / "figures"
GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]
TRAIN_GENS = ["stable_diffusion_v_1_5", "BigGAN", "glide"]
LABEL = {
    "stable_diffusion_v_1_5": "SD 1.5", "BigGAN": "BigGAN", "glide": "GLIDE",
    "ADM": "ADM", "Midjourney": "Midjourney", "VQDM": "VQDM", "wukong": "Wukong",
}

# Muted print palette.  Colour identifies the method, not the ranking.
INK = "#222222"
COLORS = {"ours": "#1f3a5f", "univfd": "#a9762f", "resnet": "#3f6f6a", "classical": "#8a8f94"}
ACCENT = "#9c4a3c"          # used only for annotations / the LOGO diagonal
GRID = "#d8d8d8"
SEQ = LinearSegmentedColormap.from_list(
    "paper_seq", ["#f6f5f0", "#cfd8e0", "#8ea6bb", "#49698a", "#1f3a5f", "#122033"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_entry(name):
    data = load(RESULTS / "main_lin" / "metrics.json")
    return data["methods"][name]["5438"]


def acc_from(entry, generator, variant="clean"):
    return entry["cross_generator"][generator]["degradations"][variant]["accuracy"]


def preds(entry, generator):
    return (np.asarray(entry["probabilities"][generator]) >= 0.5).astype(int), np.asarray(entry["labels"][generator])


def mean_over(entry, generators, variant="clean"):
    return float(np.mean([acc_from(entry, g, variant) for g in generators]))


def tidy(ax, grid_axis: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def stratified_bootstrap(y, p1, p2, gens_list, reps=10000, seed=0):
    """Paired bootstrap CI for accuracy difference, resampling within each generator."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(reps)
    for r in range(reps):
        idx = []
        for start in range(0, n, 400):
            sample = rng.integers(start, min(start + 400, n), 400)
            idx.append(sample)
        idx = np.concatenate(idx)
        deltas[r] = (p2[idx] == y[idx]).mean() - (p1[idx] == y[idx]).mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def mcnemar(p1, p2, y):
    b = int(((p1 == y) & (p2 != y)).sum())
    c = int(((p1 != y) & (p2 == y)).sum())
    return b, c, float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0


def build_summary():
    ours = method_entry("dino_clip_lin")
    base = method_entry("univfd")
    classical = load(RESULTS / "classical_svm_3gen" / "metrics.json")
    resnet = load(ROOT / "Results" / "final_colab" / "final_multi_generator_frozen_10ep" / "metrics.json")

    table = {}
    for name, values in [("classical_svm", {g: classical["cross_generator"][g]["accuracy"] for g in GENS}),
                         ("resnet18", {g: resnet["cross_generator"][g]["accuracy"] for g in GENS}),
                         ("univfd", {g: base["cross_generator"][g]["accuracy"] for g in GENS}),
                         ("ours", {g: ours["cross_generator"][g]["accuracy"] for g in GENS})]:
        table[name] = {
            "per_generator": values,
            "mean_all": float(np.mean(list(values.values()))),
            "mean_common_unseen": float(np.mean([values[g] for g in COMMON])),
            "worst_common_unseen": float(np.min([values[g] for g in COMMON])),
        }

    # Paired statistics on the four common-unseen generators.
    y_parts, p_base, p_ours = [], [], []
    for g in COMMON:
        pb, yb = preds(base, g)
        po, _ = preds(ours, g)
        y_parts.append(yb); p_base.append(pb); p_ours.append(po)
    y = np.concatenate(y_parts); p1 = np.concatenate(p_base); p2 = np.concatenate(p_ours)
    b, c, p_value = mcnemar(p1, p2, y)
    ci_low, ci_high = stratified_bootstrap(y, p1, p2, COMMON)

    stats = {
        "paired_common_unseen": {
            "n": int(len(y)),
            "accuracy_univfd": float((p1 == y).mean()),
            "accuracy_ours": float((p2 == y).mean()),
            "delta": float((p2 == y).mean() - (p1 == y).mean()),
            "delta_ci95": [ci_low, ci_high],
            "mcnemar_b_base_only_correct": b,
            "mcnemar_c_ours_only_correct": c,
            "mcnemar_p_value": p_value,
        },
        "per_generator_delta": {g: float(table["ours"]["per_generator"][g] - table["univfd"]["per_generator"][g])
                                for g in GENS},
    }

    # Leave-one-generator-out (standard GenImage protocol): 6 train -> 1 unseen.
    logo = {"per_fold": {"univfd": {}, "ours": {}}, "pooled": {}}
    y_logo, p_logo_base, p_logo_ours = [], [], []
    for held_out in GENS:
        path = RESULTS / "logo" / held_out / "metrics.json"
        fold = load(path)["methods"]
        base_entry = fold["univfd"]["5438"]
        ours_entry = fold["dino_clip_lin"]["5438"]
        logo["per_fold"]["univfd"][held_out] = base_entry["cross_generator"][held_out]["accuracy"]
        logo["per_fold"]["ours"][held_out] = ours_entry["cross_generator"][held_out]["accuracy"]
        pb, yb = preds(base_entry, held_out)
        po, _ = preds(ours_entry, held_out)
        y_logo.append(yb); p_logo_base.append(pb); p_logo_ours.append(po)
    y_all = np.concatenate(y_logo); p1_all = np.concatenate(p_logo_base); p2_all = np.concatenate(p_logo_ours)
    b, c, p_value = mcnemar(p1_all, p2_all, y_all)
    ci = stratified_bootstrap(y_all, p1_all, p2_all, GENS)
    logo["mean"] = {k: float(np.mean(list(v.values()))) for k, v in logo["per_fold"].items()}
    logo["pooled"] = {
        "n": int(len(y_all)),
        "accuracy_univfd": float((p1_all == y_all).mean()),
        "accuracy_ours": float((p2_all == y_all).mean()),
        "delta": float((p2_all == y_all).mean() - (p1_all == y_all).mean()),
        "delta_ci95": ci,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p_value": p_value,
    }
    stats["logo"] = logo
    stats["logo_folds_won"] = int(sum(logo["per_fold"]["ours"][g] > logo["per_fold"]["univfd"][g] for g in GENS))

    summary = {"table": table, "stats": stats, "robustness": robustness_table(),
               "ablation": ablation_table(), "regimes": regime_table()}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


VARIANTS = ["clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo"]


def robustness_table():
    """Degraded-test accuracy for clean-trained and degradation-augmented probes."""
    clean_trained = load(RESULTS / "main_robust" / "metrics_tta.json")["methods"]
    aug_trained = load(RESULTS / "aug_tta_dual" / "metrics_tta.json")["methods"]
    out = {}
    for label, source, method in [("univfd", clean_trained, "univfd"),
                                  ("ours", clean_trained, "dino_clip_lin"),
                                  ("ours_aug", aug_trained, "dino_clip_lin")]:
        entry = source[method]["5438"]
        out[label] = {
            "mean_all": {v: float(np.mean([acc_from(entry, g, v) for g in GENS])) for v in VARIANTS},
            "mean_common": {v: float(np.mean([acc_from(entry, g, v) for g in COMMON])) for v in VARIANTS},
            "tta_mean_all": float(np.mean([entry["cross_generator"][g]["accuracy"] for g in GENS])),
        }
    out["univfd_aug"] = {
        "mean_all": {v: float(np.mean([acc_from(aug_trained["univfd"]["5438"], g, v) for g in GENS])) for v in VARIANTS},
    }
    return out


def ablation_table():
    data = load(RESULTS / "ablation" / "metrics.json")["methods"]
    return {m: {"mean_common_unseen": float(data[m]["5438"]["mean_common_unseen"])} for m in data}


def regime_table():
    single = load(RESULTS / "single_gen_dual" / "metrics.json")["methods"]
    multi = load(RESULTS / "main_lin" / "metrics.json")["methods"]
    return {
        "single_generator": {m: {"mean_common_unseen": float(single[m]["5438"]["mean_common_unseen"]),
                                  "mean_all": float(single[m]["5438"]["mean_all"])} for m in single},
        "three_generators": {m: {"mean_common_unseen": float(multi[m]["5438"]["mean_common_unseen"]),
                                   "mean_all": float(multi[m]["5438"]["mean_all"])} for m in multi},
    }


def fig_progression(summary, robustness=None):
    names = ["classical_svm", "resnet18", "univfd", "ours"]
    display = ["HOG + FFT\nSVM (non-DL)", "ResNet-18\n(multi-gen)", "UnivFD\n(CLIP ViT-L)", "Ours\n(CLIP + DINOv2)"]
    means = [summary["table"][n]["mean_all"] * 100 for n in names]
    common = [summary["table"][n]["mean_common_unseen"] * 100 for n in names]
    colors = [COLORS["classical"], COLORS["resnet"], COLORS["univfd"], COLORS["ours"]]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.bar(x - width / 2, means, width, color=colors, edgecolor=INK, linewidth=0.5,
           label="mean over seven generators")
    ax.bar(x + width / 2, common, width, color=colors, alpha=0.32, edgecolor=colors,
           linewidth=0.6, hatch="////", label="unseen generators (4)")
    for i, (m, c) in enumerate(zip(means, common)):
        ax.text(i - width / 2, m + 0.7, f"{m:.1f}", ha="center", va="bottom", fontsize=7.2)
        ax.text(i + width / 2, c + 0.7, f"{c:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=7.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 102)
    ax.set_xlim(-0.6, 3.6)
    tidy(ax, "y")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.4)
    save(fig, "fig_progression.png")


def fig_per_generator(summary):
    names = ["classical_svm", "resnet18", "univfd", "ours"]
    labels = ["HOG+FFT SVM", "ResNet-18", "UnivFD", "DELP (ours)"]
    x = np.arange(len(GENS)); width = 0.19
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    for i, (name, label) in enumerate(zip(names, labels)):
        values = [summary["table"][name]["per_generator"][g] * 100 for g in GENS]
        ax.bar(x + (i - 1.5) * width, values, width * 0.96,
               color=[COLORS["classical"], COLORS["resnet"], COLORS["univfd"], COLORS["ours"]][i],
               edgecolor=INK, linewidth=0.4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[g] for g in GENS], fontsize=7.6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(45, 103)
    ax.axhline(50, color="#a8a8a8", lw=0.6, ls="--")
    ax.set_xlim(-0.6, 6.9)
    ax.text(6.82, 51.2, "chance", fontsize=6.4, color="#7a7a7a", ha="right")
    tidy(ax, "y")
    ax.legend(fontsize=6.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              frameon=False, columnspacing=1.2, handlelength=1.4)
    save(fig, "fig_per_generator.png")


def fig_ablation():
    data = load(RESULTS / "ablation" / "metrics.json")["methods"]
    order = ["univfd", "dino_lin", "dino_clip_resnet_lin", "dino_clip_freq_lin", "dino_clip_lin", "dino_clip"]
    labels = ["CLIP only (UnivFD)", "DINOv2 only", "CLIP + DINOv2 + ResNet-18",
              "CLIP + DINOv2 + spectrum", "CLIP + DINOv2, linear (DELP)", "CLIP + DINOv2, MLP"]
    values = [data[m]["5438"]["mean_common_unseen"] * 100 for m in order]
    colors = ["#d5d7d9", "#d5d7d9", "#c3c7cb", "#c3c7cb", COLORS["ours"], "#b8c3cf"]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = np.arange(len(order))
    bars = ax.barh(y, values, height=0.62, color=colors, edgecolor=INK, linewidth=0.45)
    for i, (bar, v) in enumerate(zip(bars, values)):
        weight = "bold" if i == 4 else "normal"
        ax.text(v + 0.12, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center",
                ha="left", fontsize=7.4, fontweight=weight)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlim(85, 99.3)
    ax.set_xlabel("Accuracy on the four unseen generators (%)")
    ax.invert_yaxis()
    tidy(ax, "x")
    save(fig, "fig_ablation.png")


def fig_robustness(summary):
    robust = summary["robustness"]
    variants = VARIANTS
    display = ["clean", "JPEG q90", "JPEG q50", "resize 0.5", "blur 1", "combo"]
    series = [("univfd", "UnivFD, clean-trained", COLORS["univfd"], "o"),
              ("ours", "DELP, clean-trained", COLORS["ours"], "s"),
              ("ours_aug", "DELP, degradation-augmented", ACCENT, "^")]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for key, label, color, marker in series:
        ys = [robust[key]["mean_all"][v] * 100 for v in variants]
        ax.plot(range(len(variants)), ys, marker=marker, color=color, label=label,
                lw=1.3, ms=4.0, mec=color, mfc="white", mew=1.0)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(display, fontsize=7.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(86, 100)
    tidy(ax, "y")
    ax.legend(fontsize=6.9, loc="lower left", frameon=False, handlelength=2.0, borderaxespad=0.5)
    save(fig, "fig_robustness.png")


def fig_regime(summary):
    regimes = summary["regimes"]
    settings = [("single_generator", "SD 1.5 only"), ("three_generators", "SD 1.5 + BigGAN + GLIDE")]
    methods = [("univfd", "UnivFD", COLORS["univfd"]), ("dino_clip_lin", "DELP (ours)", COLORS["ours"])]
    x = np.arange(len(settings)); width = 0.30
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, (m, label, color) in enumerate(methods):
        values = [regimes[s][m]["mean_common_unseen"] * 100 for s, _ in settings]
        bars = ax.bar(x + (i - 0.5) * width, values, width, color=color,
                      edgecolor=INK, linewidth=0.5, label=label)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in settings], fontsize=7.8)
    ax.set_ylabel("Accuracy, four unseen generators (%)")
    ax.set_ylim(84, 99)
    tidy(ax, "y")
    ax.legend(fontsize=7.2, loc="upper left", frameon=False, handlelength=1.4)
    save(fig, "fig_regime.png")


def fig_logo():
    rows = GENS
    matrices = {}
    for name in ["univfd", "dino_clip_lin"]:
        matrix = np.zeros((len(rows), len(GENS)))
        for i, held_out in enumerate(rows):
            path = RESULTS / "logo" / held_out / "metrics.json"
            entry = load(path)["methods"][name]["5438"]
            for j, test_gen in enumerate(GENS):
                matrix[i, j] = entry["cross_generator"][test_gen]["accuracy"] * 100
        matrices[name] = matrix
    fig = plt.figure(figsize=(7.4, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.035], hspace=0.55, wspace=0.30)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[1, :])
    vmin, vmax = 80.0, 100.0
    for ax, (name, title) in zip(axes, [("univfd", "(a) UnivFD (published probe)"),
                                        ("dino_clip_lin", "(b) DELP (CLIP + DINOv2)")]):
        matrix = matrices[name]
        im = ax.imshow(matrix, cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="nearest")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6.0,
                        color="white" if matrix[i, j] > 93 else INK)
        ax.set_xticks(range(len(GENS)))
        ax.set_xticklabels([LABEL[g] for g in GENS], rotation=45, ha="right", fontsize=6.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LABEL[g] for g in rows], fontsize=6.8)
        ax.set_xlabel("test generator", fontsize=7.2)
        ax.set_ylabel("held-out generator", fontsize=7.2)
        ax.set_title(title, fontsize=7.6, loc="left", pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # The diagonal is the only strictly unseen cell in each fold.
        for i in range(len(rows)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=ACCENT, lw=0.9))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.set_label("accuracy (%) on the held-out generator", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8, width=0.4, length=2.5)
    save(fig, "fig_logo_heatmaps.png")


def fig_cv():
    data = load(RESULTS / "cv" / "cv_results.json")
    candidates = list(data["candidates"].keys())
    labels = {"univfd": "CLIP (UnivFD)", "dino_clip_lin": "CLIP + DINOv2 (linear)",
              "clip_mlp": "CLIP (MLP)", "dino_clip": "CLIP + DINOv2 (MLP)"}
    folds = data["folds"]
    colors = {"univfd": COLORS["univfd"], "dino_clip_lin": COLORS["ours"],
              "clip_mlp": "#c0c4c8", "dino_clip": "#b3c2d2"}
    x = np.arange(len(folds)); width = 0.8 / len(candidates)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for i, c in enumerate(candidates):
        values = [np.asarray(data["candidates"][c])[folds.index(f)] * 100 for f in folds]
        ax.bar(x + (i - (len(candidates) - 1) / 2) * width, values, width * 0.92,
               color=colors[c], edgecolor=INK, linewidth=0.35, label=labels[c])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[f] for f in folds], fontsize=6.9)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_ylim(65, 103)
    tidy(ax, "y")
    ax.legend(fontsize=6.3, ncol=2, loc="upper left", frameon=False, columnspacing=1.0, handlelength=1.3)
    save(fig, "fig_cv.png")


def copy_to_latex() -> None:
    """The report compiles from Report/latex/figures; keep it in sync."""
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    for png in sorted(FIG.glob("*.png")):
        shutil.copy2(png, LATEX_FIG / png.name)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    fig_progression(summary)
    fig_per_generator(summary)
    fig_regime(summary)
    if (RESULTS / "ablation" / "metrics.json").exists():
        fig_ablation()
    if (RESULTS / "main_robust" / "metrics_tta.json").exists():
        fig_robustness(summary)
    if (RESULTS / "logo" / GENS[0] / "metrics.json").exists():
        fig_logo()
    if (RESULTS / "cv" / "cv_results.json").exists():
        fig_cv()
    copy_to_latex()
    print("figures written to", FIG, "and copied to", LATEX_FIG)
