from __future__ import annotations

"""Generator-level cross-validation for model selection.

Models are trained on two of the three training generators and validated on the
third (held-out) generator. The held-out generator is never used for gradient
updates, so this score estimates cross-generator generalisation without
touching the four final unseen generators.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from paths import CACHE, RESULTS_V3
from probe import (fit_logistic, fit_mlp, feature_block, path_index,
                   predict_mlp, select_train, score)


def single_set(manifest, generator, split, maximum, seed=5438):
    index = path_index(manifest)
    fake = [p for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
            if g == generator and s == split]
    real = [p for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
            if g == "Nature" and s == split]
    from src.data import limited
    fake = limited([Path(p) for p in fake], maximum, seed + 2)
    real = limited([Path(p) for p in real], len(fake), seed + 3)
    ids = [index[str(p)] for p in real] + [index[str(p)] for p in fake]
    labels = np.asarray([0] * len(real) + [1] * len(fake))
    return ids, labels


def main():
    parser = argparse.ArgumentParser(description="Generator-level cross-validation for model selection.")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=RESULTS_V3 / "cv")
    parser.add_argument("--max-eval-per-class", type=int, default=200)
    args = parser.parse_args()

    cache = args.cache
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    table = {"clean": dict(np.load(cache / "clean.npz", allow_pickle=True))}
    dino = {"clean": dict(np.load(cache / "dino_clean.npz", allow_pickle=True))}
    folds = [
        ("glide", ["stable_diffusion_v_1_5", "BigGAN"]),
        ("stable_diffusion_v_1_5", ["BigGAN", "glide"]),
        ("BigGAN", ["stable_diffusion_v_1_5", "glide"]),
    ]
    candidates = ["univfd", "dino_clip_lin", "clip_mlp", "dino_clip"]
    results = {name: [] for name in candidates}
    knn_results = {k: [] for k in (10, 20, 40)}

    for held_out, train_gens in folds:
        train_ids = select_train(manifest, train_gens, "train", None, 5438)
        val_ids, y_val = single_set(manifest, held_out, "val", args.max_eval_per_class)
        test_ids, y_test = single_set(manifest, held_out, "test", args.max_eval_per_class)
        labels = np.asarray(manifest["labels"])
        y_train = labels[train_ids]
        for name in candidates:
            x_train = feature_block(name, table, "clean", train_ids, dino)
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train).astype(np.float32)
            x_val = scaler.transform(feature_block(name, table, "clean", val_ids, dino)).astype(np.float32)
            x_test = scaler.transform(feature_block(name, table, "clean", test_ids, dino)).astype(np.float32)
            if name in ("univfd", "dino_clip_lin"):
                model, _ = fit_logistic(x_train, y_train, x_val, y_val)
                prob = model.predict_proba(x_test)[:, 1]
            else:
                model, _ = fit_mlp(x_train, y_train, x_val, y_val, 5438)
                prob = predict_mlp(model, x_test)
            acc = score(y_test, (prob >= 0.5).astype(int), prob)["accuracy"]
            results[name].append(acc)
            print(f"fold held-out={held_out:22s} {name:12s} acc={acc:.4f}", flush=True)

        x_train = feature_block("univfd", table, "clean", train_ids, dino)
        x_train = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
        x_test = feature_block("univfd", table, "clean", test_ids, dino)
        x_test = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-8)
        for k in knn_results:
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance").fit(x_train, y_train)
            prob = knn.predict_proba(x_test)[:, 1]
            knn_results[k].append(score(y_test, knn.predict(x_test), prob)["accuracy"])

    print("\nCV summary (mean held-out-generator accuracy, 3 folds):")
    for name in candidates:
        print(f"{name:12s} {np.mean(results[name])*100:.2f} +/- {np.std(results[name])*100:.2f}")
    for k, values in knn_results.items():
        print(f"knn-{k:<8d} {np.mean(values)*100:.2f} +/- {np.std(values)*100:.2f}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_results.json").write_text(json.dumps(
        {"folds": [f[0] for f in folds], "candidates": results, "knn": knn_results},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
