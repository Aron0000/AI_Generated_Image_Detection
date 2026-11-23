from __future__ import annotations

"""Classical non-deep-learning baseline: HOG + radial frequency features + RBF-SVM.

This is the "classic" detector the deep models must beat. The feature extractor
and the split logic match the rest of the project so comparisons are honest.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import degrade
from src.data import GENERATORS, image_files, limited, split_files

TEST_VARIANTS = ["clean", "jpeg90", "jpeg50", "resize50", "blur1"]


def feature(path: Path, variant: str = "clean") -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    image = degrade(image, variant)
    gray = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
    x = np.asarray(gray, dtype=np.float32) / 255.0

    h = hog(
        x, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True,
    )
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x))))
    yy, xx = np.indices(spectrum.shape)
    cy, cx = (np.array(spectrum.shape) - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, rr.max(), 17)
    radial = [
        float(spectrum[(rr >= a) & (rr < b)].mean()) if ((rr >= a) & (rr < b)).any() else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ]
    return np.concatenate([h, np.asarray(radial, dtype=np.float32)])


def build_xy(root: Path, generators: list[str], split: str, maximum: int | None,
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    real = limited(split_files(image_files(root / "Nature"), split, seed), maximum, seed + 1)
    fake = []
    for position, generator in enumerate(generators):
        fake.extend(limited(split_files(image_files(root / generator), split, seed),
                            maximum, seed + 2 + position))
    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    paths = real + fake
    y = np.asarray([0] * n + [1] * n)
    x = np.vstack([feature(p) for p in paths])
    return x, y


def summarize(y, pred, score) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "count": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-generators", nargs="+", default=["stable_diffusion_v_1_5"])
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-eval-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5438)
    args = parser.parse_args()
    args.data = args.data.expanduser()
    args.output = args.output.expanduser()

    started = time.time()
    x_train, y_train = build_xy(args.data, args.train_generators, "train", args.max_train_per_class, args.seed)
    x_val, y_val = build_xy(args.data, args.train_generators, "val", args.max_eval_per_class, args.seed)

    best = None
    for c in (1.0, 10.0, 100.0):
        pipeline = Pipeline([("scale", StandardScaler()),
                             ("svm", SVC(C=c, kernel="rbf", gamma="scale"))])
        pipeline.fit(x_train, y_train)
        f1 = f1_score(y_val, pipeline.predict(x_val), average="macro", zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, c, pipeline)

    val_f1, best_c, model = best
    results = {
        "method": "HOG_frequency_RBF_SVM",
        "train_generators": args.train_generators,
        "best_c": best_c,
        "validation_macro_f1": float(val_f1),
        "training_seconds": time.time() - started,
        "cross_generator": {},
    }
    for generator in GENERATORS:
        fake = limited(split_files(image_files(args.data / generator), "test", args.seed),
                       args.max_eval_per_class, args.seed + 2)
        real = limited(split_files(image_files(args.data / "Nature"), "test", args.seed),
                       len(fake), args.seed + 3)
        paths = real + fake
        y = np.asarray([0] * len(real) + [1] * len(fake))
        per_variant = {}
        for variant in TEST_VARIANTS:
            x = np.vstack([feature(p, variant) for p in paths])
            pred = model.predict(x)
            per_variant[variant] = summarize(y, pred, model.decision_function(x))
        results["cross_generator"][generator] = dict(per_variant["clean"])
        results["cross_generator"][generator]["degradations"] = per_variant
        print(f"{generator:24s} clean={per_variant['clean']['accuracy']:.4f}", flush=True)

    accs = [results["cross_generator"][g]["accuracy"] for g in GENERATORS]
    unseen = [results["cross_generator"][g]["accuracy"] for g in GENERATORS if g not in args.train_generators]
    results["mean_all"] = float(np.mean(accs))
    results["mean_unseen"] = float(np.mean(unseen))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.output / 'metrics.json'}  mean_all={results['mean_all']:.4f}")


if __name__ == "__main__":
    main()
