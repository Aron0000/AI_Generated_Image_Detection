from __future__ import annotations

"""Train frozen-feature probes and evaluate them across generators.

Methods:
  univfd      linear probe on frozen CLIP ViT-L/14 features (UnivFD, CVPR 2023)
  clip_mlp    same feature, non-linear head (head-capacity control)
  resnet      linear probe on frozen ImageNet ResNet-18 features
  freq        probe on the hand-crafted frequency descriptor
  clip_resnet late-fusion of CLIP and ResNet features
  craf        CLIP + ResNet + frequency fusion (our method)

All methods share the same split, balancing rule, validation-based checkpoint
selection and test set. Degradation-augmented training and test-time
augmentation are opt-in flags so their effect can be ablated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import GENERATORS, limited, stable_bucket

UNSEEN_COMMON = ["ADM", "Midjourney", "VQDM", "wukong"]


def path_index(manifest: dict) -> dict:
    return {p: i for i, p in enumerate(manifest["paths"])}


def select_train(manifest: dict, generators: list[str], split: str,
                 max_per_generator: int | None, seed: int = 5438) -> list[int]:
    """Mirror MultiGeneratorImageDataset balancing so every method sees the same data."""
    fake = []
    for position, generator in enumerate(generators):
        paths = [Path(p) for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
                 if g == generator and s == split]
        fake.extend(limited(paths, max_per_generator, seed + 100 + position))
    real = [Path(p) for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
            if g == "Nature" and s == split]
    real = limited(real, len(fake), seed + 500)
    if len(real) < len(fake):
        fake = limited(fake, len(real), seed + 600)
    index = path_index(manifest)
    return [index[str(p)] for p in real + fake]


def select_test(manifest: dict, generator: str, max_per_class: int | None, seed: int = 5438) -> tuple[list[int], np.ndarray]:
    """Mirror BinaryImageDataset: fake capped first, then real matched to fake count."""
    index = path_index(manifest)
    fake = [Path(p) for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
            if g == generator and s == "test"]
    real = [Path(p) for p, g, s in zip(manifest["paths"], manifest["generators"], manifest["splits"])
            if g == "Nature" and s == "test"]
    fake = limited(fake, max_per_class, seed)
    real = limited(real, min(len(fake), max_per_class or len(fake)), seed + 2)
    fake = limited(fake, len(real), seed + 3)
    ids = [index[str(p)] for p in real + fake]
    labels = np.asarray([0] * len(real) + [1] * len(fake))
    return ids, labels


def score(y, pred, prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "count": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, width: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, 2),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(x_train, y_train, x_val, y_val, seed: int, epochs: int = 80, patience: int = 15) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()),
        batch_size=256, shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    best_state, best_f1, stale = None, -1.0, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val_t).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_f1


def fit_logistic(x_train, y_train, x_val, y_val) -> tuple[LogisticRegression, float]:
    best_model, best_f1 = None, -1.0
    for c in (0.001, 0.01, 0.1, 1.0):
        model = LogisticRegression(C=c, max_iter=3000)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_model = f1, model
    return best_model, best_f1


def predict_mlp(model, x) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).float().to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


FEATURE_BLOCKS = {
    "univfd": ("clip",),
    "clip_mlp": ("clip",),
    "resnet": ("resnet",),
    "freq": ("freq",),
    "clip_resnet": ("clip", "resnet"),
    "craf": ("clip", "resnet", "freq"),
    "dino": ("dino",),
    "dino_lin": ("dino",),
    "dino_clip": ("dino", "clip"),
    "dino_clip_lin": ("dino", "clip"),
    "dino_clip_resnet_lin": ("dino", "clip", "resnet"),
    "dino_clip_freq_lin": ("dino", "clip", "freq"),
    "dino_clip_resnet_freq_lin": ("dino", "clip", "resnet", "freq"),
    "dino_craf": ("dino", "clip", "freq"),
}


def feature_block(method: str, tables: dict, variant: str, ids: list[int],
                  dino_tables: dict | None = None) -> np.ndarray:
    table = tables[variant]
    parts = []
    for block_name in FEATURE_BLOCKS[method]:
        if block_name == "clip":
            block = np.asarray(table["clip"][ids], dtype=np.float32)
            block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-8
            parts.append(block)
        elif block_name == "resnet":
            parts.append(np.asarray(table["resnet"][ids], dtype=np.float32))
        elif block_name == "dino":
            block = np.asarray(dino_tables[variant]["dino"][ids], dtype=np.float32)
            block /= np.linalg.norm(block, axis=1, keepdims=True) + 1e-8
            parts.append(block)
        else:
            parts.append(np.asarray(table["freq"][ids], dtype=np.float32))
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-generators", nargs="+", default=["stable_diffusion_v_1_5", "BigGAN", "glide"])
    parser.add_argument("--methods", nargs="+",
                        default=["univfd", "clip_mlp", "resnet", "freq", "clip_resnet", "craf"])
    parser.add_argument("--aug", nargs="*", default=[])
    parser.add_argument("--tta", nargs="*", default=[])
    parser.add_argument("--max-train-per-generator", type=int, default=None)
    parser.add_argument("--max-val-per-generator", type=int, default=67)
    parser.add_argument("--max-eval-per-class", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5438])
    parser.add_argument("--split-seed", type=int, default=5438)
    args = parser.parse_args()
    args.cache = args.cache.expanduser()
    args.output = args.output.expanduser()

    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    variants = sorted({v for v in ["clean", *args.aug, *args.tta]})
    tables = {v: dict(np.load(args.cache / f"{v}.npz", allow_pickle=True)) for v in variants}
    dino_tables = {}
    for variant in variants:
        path = args.cache / f"dino_{variant}.npz"
        if path.exists():
            dino_tables[variant] = dict(np.load(path, allow_pickle=True))

    train_ids = select_train(manifest, args.train_generators, "train",
                             args.max_train_per_generator, args.split_seed)
    val_ids = select_train(manifest, args.train_generators, "val",
                           args.max_val_per_generator, args.split_seed)
    labels = np.asarray(manifest["labels"])
    y_train, y_val = labels[train_ids], labels[val_ids]

    test_sets = {g: select_test(manifest, g, args.max_eval_per_class, args.split_seed) for g in GENERATORS}
    test_variants = ["clean", *args.tta]

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    results = {"config": config, "methods": {}}
    for method in args.methods:
        for seed in args.seeds:
            # Assemble training matrix, with optional degraded copies as extra samples.
            x_train = feature_block(method, tables, "clean", train_ids, dino_tables)
            if args.aug:
                for variant in args.aug:
                    x_train = np.vstack([x_train, feature_block(method, tables, variant, train_ids, dino_tables)])
                y_fit = np.concatenate([y_train] * (1 + len(args.aug)))
            else:
                y_fit = y_train
            x_val = feature_block(method, tables, "clean", val_ids, dino_tables)

            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train).astype(np.float32)
            x_val = scaler.transform(x_val).astype(np.float32)

            if method in ("univfd", "resnet", "freq", "dino_lin", "dino_clip_lin",
                          "dino_clip_resnet_lin", "dino_clip_freq_lin", "dino_clip_resnet_freq_lin"):
                model, val_f1 = fit_logistic(x_train, y_fit, x_val, y_val)
                predict = lambda x: model.predict_proba(x)[:, 1]
            else:
                model, val_f1 = fit_mlp(x_train, y_fit, x_val, y_val, seed)
                predict = lambda x: predict_mlp(model, x)

            entry = {"seed": seed, "validation_macro_f1": float(val_f1),
                     "cross_generator": {}, "probabilities": {}, "labels": {}}
            for generator in GENERATORS:
                ids, y = test_sets[generator]
                probabilities, per_variant = [], {}
                for variant in test_variants:
                    x = scaler.transform(feature_block(method, tables, variant, ids, dino_tables).astype(np.float32)).astype(np.float32)
                    prob = predict(x)
                    probabilities.append(prob)
                    pred = (prob >= 0.5).astype(int)
                    per_variant[variant] = score(y, pred, prob)
                prob = np.mean(probabilities, axis=0)
                pred = (prob >= 0.5).astype(int)
                result = score(y, pred, prob)
                result["degradations"] = per_variant
                entry["cross_generator"][generator] = result
                entry["probabilities"][generator] = prob.tolist()
                entry["labels"][generator] = y.tolist()

            accs = [entry["cross_generator"][g]["accuracy"] for g in GENERATORS]
            unseen = [entry["cross_generator"][g]["accuracy"] for g in GENERATORS if g not in args.train_generators]
            entry["mean_all"] = float(np.mean(accs))
            entry["mean_unseen"] = float(np.mean(unseen))
            entry["mean_common_unseen"] = float(np.mean([entry["cross_generator"][g]["accuracy"] for g in UNSEEN_COMMON]))
            entry["worst_unseen"] = float(np.min(unseen))
            results["methods"].setdefault(method, {})[str(seed)] = entry

            print(f"{method:14s} seed={seed} val_f1={val_f1:.4f} "
                  f"common_unseen={entry['mean_common_unseen']:.4f} unseen={entry['mean_unseen']:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    name = "metrics_tta.json" if args.tta else ("metrics_aug.json" if args.aug else "metrics.json")
    (args.output / name).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("saved", args.output / name)


if __name__ == "__main__":
    main()
