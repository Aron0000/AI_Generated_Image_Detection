"""Build the audit tables and diagnostic figures used by the LaTeX report.

Everything written here is derived from files on disk or from the stored
probability vectors in Results/final_v3; nothing is typed by hand.
Outputs:
  Results/final_v3/latex_assets.json
  Report/latex/figures/fig_confusion.png
  Report/latex/figures/fig_roc.png
  Report/latex/figures/fig_bootstrap.png
  Report/latex/figures/fig_far_frr.png
  Report/latex/figures/fig_logo_full.png
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe
from paths import CACHE, DATA, ROOT

RESULTS = ROOT / "Results" / "final_v3"
FIG = ROOT / "Report" / "latex" / "figures"
GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]
LABEL = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE", "Midjourney": "Midjourney",
         "stable_diffusion_v_1_5": "SD 1.5", "VQDM": "VQDM", "wukong": "Wukong"}
SHORT = LABEL
CLASSES = ["Nature", *GENS]

INK = "#222222"
NAVY = "#1f3a5f"
GOLD = "#a9762f"
ACCENT = "#9c4a3c"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 8.0, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def method_entry(method: str, folder: Path = RESULTS / "main_lin"):
    return load(folder / "metrics.json")["methods"][method]["5438"]


def dataset_audit() -> dict:
    audit = {"classes": {}, "splits": {}, "train_selection": {}, "nature_metadata": {}}
    for cls in CLASSES:
        files = [p for p in (DATA / cls).iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        exts = Counter(p.suffix.lower() for p in files)
        dims = Counter()
        for path in sorted(files)[:300]:
            with Image.open(path) as image:
                dims[image.size] += 1
        audit["classes"][cls] = {
            "count": len(files),
            "extensions": dict(exts),
            "sampled": sum(dims.values()),
            "unique_dims_in_sample": len(dims),
            "common_dims": [[list(k), v] for k, v in dims.most_common(4)],
        }

    manifest = load(CACHE / "manifest.json")
    for cls in CLASSES:
        audit["splits"][cls] = {
            split: int(sum(1 for g, s in zip(manifest["generators"], manifest["splits"])
                           if g == cls and s == split))
            for split in ("train", "val", "test")
        }

    # Exact probe training / validation sets, using the same selection functions
    # as probe.py so the numbers cannot drift from the experiments.
    train_ids = probe.select_train(manifest, ["stable_diffusion_v_1_5", "BigGAN", "glide"], "train", None, 5438)
    val_ids = probe.select_train(manifest, ["stable_diffusion_v_1_5", "BigGAN", "glide"], "val", 67, 5438)
    labels = np.asarray(manifest["labels"])
    audit["train_selection"] = {
        "p1_train_total": len(train_ids),
        "p1_train_fake": int(labels[train_ids].sum()),
        "p1_train_real": int((labels[train_ids] == 0).sum()),
        "p1_val_total": len(val_ids),
        "p1_val_fake": int(labels[val_ids].sum()),
        "p1_val_real": int((labels[val_ids] == 0).sum()),
        "p1_train_per_generator": {
            g: int(sum(1 for i in train_ids if manifest["generators"][i] == g))
            for g in GENS
        },
        "test_per_class": 200,
    }

    meta_path = DATA / "nature_metadata.csv"
    if meta_path.exists():
        widths, heights, qualities, synsets, rows = [], [], Counter(), set(), 0
        with meta_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                syn = "".join(ch for ch in row[1].split("\n")[-1] if ch.isdigit())
                try:
                    w, h, q = int(row[2]), int(row[3]), int(row[4])
                except ValueError:
                    continue
                rows += 1
                widths.append(w); heights.append(h); qualities[q] += 1; synsets.add(syn)
        audit["nature_metadata"] = {
            "rows": rows, "distinct_synsets": len(synsets),
            "width_min": min(widths), "width_max": max(widths),
            "height_min": min(heights), "height_max": max(heights),
            "quality": dict(qualities),
        }
    return audit


def confusion_figure(entries: dict) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4))
    out = {}
    for row, (method, entry) in enumerate(entries.items()):
        out[method] = {}
        for col, gen in enumerate(COMMON):
            cm = np.asarray(entry["cross_generator"][gen]["confusion_matrix"])
            out[method][gen] = cm.tolist()
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=400, interpolation="nearest")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > 200 else INK, fontsize=8.2)
            ax.set_xticks([0, 1], ["real", "fake"])
            ax.set_yticks([0, 1], ["real", "fake"])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(SHORT[gen], fontsize=7.8, pad=4)
            if col == 0:
                ax.set_ylabel(f"{method}\ntrue", fontsize=7.4)
            if row == 1:
                ax.set_xlabel("predicted", fontsize=7.4)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    fig.savefig(FIG / "fig_confusion.png")
    plt.close(fig)
    return out


def roc_figure(entries: dict) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.3), sharey=True)
    out = {}
    for col, gen in enumerate(COMMON):
        ax = axes[col]
        out[gen] = {}
        for method, entry in entries.items():
            y = np.asarray(entry["labels"][gen])
            p = np.asarray(entry["probabilities"][gen])
            fpr, tpr, _ = roc_curve(y, p)
            auc = entry["cross_generator"][gen]["roc_auc"]
            out[gen][method] = {"auc": float(auc), "points": int(len(y))}
            ax.plot(fpr, tpr, lw=0.9, drawstyle="steps-post",
                    label=f"{method} (AUC {auc:.3f})",
                    color=NAVY if method == "DELP" else GOLD)
        ax.plot([0, 1], [0, 1], ls=":", color="#9a9a9a", lw=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
        ax.set_title(SHORT[gen], fontsize=7.8)
        ax.set_xlabel("false positive rate", fontsize=7.0)
        if col == 0:
            ax.set_ylabel("true positive rate", fontsize=7.0)
        ax.tick_params(labelsize=6.4)
        ax.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig_roc.png")
    plt.close(fig)
    return out


def gather_preds(entry, gens):
    y, p = [], []
    for gen in gens:
        y.append(np.asarray(entry["labels"][gen]))
        p.append(np.asarray(entry["probabilities"][gen]))
    return np.concatenate(y), np.concatenate(p)


def stratified_bootstrap(y, p1, p2, reps=10000, seed=0):
    """Resample within each generator (400 images per block) and keep returns."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(reps)
    for r in range(reps):
        idx = []
        for start in range(0, n, 400):
            idx.append(rng.integers(start, min(start + 400, n), 400))
        idx = np.concatenate(idx)
        deltas[r] = (p2[idx] == y[idx]).mean() - (p1[idx] == y[idx]).mean()
    return deltas


def bootstrap_figure(base, ours) -> dict:
    y, p1 = gather_preds(base, COMMON)
    _, p2 = gather_preds(ours, COMMON)
    d1 = stratified_bootstrap(y, (p1 >= 0.5).astype(int), (p2 >= 0.5).astype(int), seed=0)

    y2, q1 = gather_preds(base, GENS)
    _, q2 = gather_preds(ours, GENS)
    # For the LOGO comparison we must use the fold-specific probabilities.
    y_logo, p_logo, q_logo = [], [], []
    for gen in GENS:
        fold = load(RESULTS / "logo" / gen / "metrics.json")["methods"]
        y_logo.append(np.asarray(fold["univfd"]["5438"]["labels"][gen]))
        p_logo.append(np.asarray(fold["univfd"]["5438"]["probabilities"][gen]))
        q_logo.append(np.asarray(fold["dino_clip_lin"]["5438"]["probabilities"][gen]))
    yl = np.concatenate(y_logo); pl = np.concatenate(p_logo); ql = np.concatenate(q_logo)
    d2 = stratified_bootstrap(yl, (pl >= 0.5).astype(int), (ql >= 0.5).astype(int), seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, d, title, observed in [
        (axes[0], d1, "(a) P1: four unseen generators (n = 1,600)",
         ((p2 >= 0.5) == y).mean() - ((p1 >= 0.5) == y).mean()),
        (axes[1], d2, "(b) P2: seven LOGO folds pooled (n = 2,800)",
         ((ql >= 0.5) == yl).mean() - ((pl >= 0.5) == yl).mean()),
    ]:
        ax.hist(d * 100, bins=70, color=NAVY, alpha=0.62, edgecolor="none")
        lo, hi = np.percentile(d * 100, [2.5, 97.5])
        ax.axvline(observed * 100, color=INK, lw=1.0, label=f"observed {observed * 100:+.2f} pp")
        ax.axvline(lo, color=ACCENT, lw=0.8, ls="--", label=f"95% CI [{lo:+.2f}, {hi:+.2f}]")
        ax.axvline(hi, color=ACCENT, lw=0.8, ls="--")
        ax.axvline(0, color="#9a9a9a", lw=0.7, ls=":")
        ax.set_title(title, fontsize=7.8, loc="left")
        ax.set_xlabel("accuracy difference, DELP $-$ UnivFD (percentage points)", fontsize=7.0)
        ax.set_ylabel("bootstrap replicates", fontsize=7.0)
        ax.tick_params(labelsize=6.4)
        ax.legend(fontsize=6.0, frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.3)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(FIG / "fig_bootstrap.png")
    plt.close(fig)
    return {
        "p1": {"ci": [float(np.percentile(d1 * 100, 2.5)), float(np.percentile(d1 * 100, 97.5))],
               "mean": float(d1.mean() * 100), "reps": 10000},
        "logo": {"ci": [float(np.percentile(d2 * 100, 2.5)), float(np.percentile(d2 * 100, 97.5))],
                 "mean": float(d2.mean() * 100), "reps": 10000},
    }


def far_frr_figure(base, ours) -> dict:
    thresholds = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.3), sharex=True, sharey=True)
    out = {}
    legend_handles = None
    for k, gen in enumerate(COMMON):
        ax = axes[k // 2, k % 2]
        out[gen] = {}
        for method, entry in [("UnivFD", base), ("DELP", ours)]:
            y = np.asarray(entry["labels"][gen])
            p = np.asarray(entry["probabilities"][gen])
            far = np.array([((p >= t) & (y == 0)).sum() / (y == 0).sum() for t in thresholds])
            frr = np.array([((p < t) & (y == 1)).sum() / (y == 1).sum() for t in thresholds])
            color = NAVY if method == "DELP" else GOLD
            line_far, = ax.plot(thresholds, far, color=color, lw=1.0, ls="-",
                                label=f"{method} FAR")
            line_frr, = ax.plot(thresholds, frr, color=color, lw=1.0, ls="--",
                                label=f"{method} FRR")
            if legend_handles is None:
                legend_handles = [line_far, line_frr]
            else:
                legend_handles.extend([line_far, line_frr])
            gap = np.abs(far - frr)
            k_min = int(np.argmin(gap))
            out[gen][method] = {
                "far_at_0.5": float(((p >= 0.5) & (y == 0)).sum() / (y == 0).sum()),
                "frr_at_0.5": float(((p < 0.5) & (y == 1)).sum() / (y == 1).sum()),
                "eer_threshold": float(thresholds[k_min]),
                "eer": float((far[k_min] + frr[k_min]) / 2),
            }
        ax.set_title(f"({chr(97 + k)}) {SHORT[gen]}", fontsize=7.8, loc="left")
        ax.set_ylim(0, 0.32)
        ax.grid(color="#d8d8d8", lw=0.4)
        ax.set_axisbelow(True)
        if k >= 2:
            ax.set_xlabel("decision threshold", fontsize=7.0)
        if k % 2 == 0:
            ax.set_ylabel("error rate", fontsize=7.0)
        ax.tick_params(labelsize=6.4)
    fig.legend(legend_handles[:4], ["UnivFD FAR", "UnivFD FRR", "DELP FAR", "DELP FRR"],
               loc="lower center", ncol=4, fontsize=6.4, frameon=False,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0, w_pad=1.0)
    fig.savefig(FIG / "fig_far_frr.png")
    plt.close(fig)
    return out


def logo_full() -> dict:
    out = {"univfd": {}, "dino_clip_lin": {}, "offdiag": {}}
    for held in GENS:
        fold = load(RESULTS / "logo" / held / "metrics.json")["methods"]
        for method in ("univfd", "dino_clip_lin"):
            entry = fold[method]["5438"]
            out[method][held] = {g: float(entry["cross_generator"][g]["accuracy"]) for g in GENS}
    for method in ("univfd", "dino_clip_lin"):
        seen = [out[method][h][g] for h in GENS for g in GENS if g != h]
        out["offdiag"][method] = float(np.mean(seen))
    return out


def robustness_per_generator() -> dict:
    clean = load(RESULTS / "main_robust" / "metrics_tta.json")["methods"]
    aug = load(RESULTS / "aug_tta_dual" / "metrics_tta.json")["methods"]
    variants = ["clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo"]
    out = {}
    for label, source, method in [("univfd", clean, "univfd"), ("ours", clean, "dino_clip_lin"),
                                  ("ours_aug", aug, "dino_clip_lin")]:
        entry = source[method]["5438"]
        out[label] = {g: {v: float(entry["cross_generator"][g]["degradations"][v]["accuracy"])
                          for v in variants} for g in GENS}
    return out


def error_overlap() -> dict:
    """How the two component encoders fail, and whether fusion repairs each case."""
    ablation = load(RESULTS / "ablation" / "metrics.json")["methods"]
    counts = {k: 0 for k in ("both_correct", "clip_only", "dino_only", "both_wrong")}
    fused_fixed = {k: 0 for k in counts}
    total = 0
    for gen in COMMON:
        y = np.asarray(ablation["univfd"]["5438"]["labels"][gen])
        pc = np.asarray(ablation["univfd"]["5438"]["probabilities"][gen]) >= 0.5
        pd = np.asarray(ablation["dino_lin"]["5438"]["probabilities"][gen]) >= 0.5
        pf = np.asarray(ablation["dino_clip_lin"]["5438"]["probabilities"][gen]) >= 0.5
        total += len(y)
        for label, mask in [
            ("both_correct", (pc == y) & (pd == y)),
            ("clip_only", (pc == y) & (pd != y)),
            ("dino_only", (pc != y) & (pd == y)),
            ("both_wrong", (pc != y) & (pd != y)),
        ]:
            counts[label] += int(mask.sum())
            fused_fixed[label] += int((mask & (pf == y)).sum())
    return {"n": total, "cases": counts, "fused_correct_in_case": fused_fixed}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    base = method_entry("univfd")
    ours = method_entry("dino_clip_lin")
    assets = {
        "error_overlap": error_overlap(),
        "dataset": dataset_audit(),
        "confusion": confusion_figure({"UnivFD": base, "DELP": ours}),
        "roc": roc_figure({"UnivFD": base, "DELP": ours}),
        "bootstrap": bootstrap_figure(base, ours),
        "far_frr": far_frr_figure(base, ours),
        "logo_full": logo_full(),
        "robustness_per_generator": robustness_per_generator(),
    }
    out = RESULTS / "latex_assets.json"
    out.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    print("saved", out)
    print(json.dumps({k: v for k, v in assets.items() if k not in ("confusion", "logo_full", "robustness_per_generator")}, indent=2)[:3000])


if __name__ == "__main__":
    main()
