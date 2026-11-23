"""Train one ResNet-18 on several generators and evaluate every generator.

This reproduces the ResNet-18 row of the report: a frozen ImageNet backbone
with a two-way head, trained for 10 epochs on 333 images per training
generator and 999 real images, with the checkpoint selected on the validation
split only.  The four unseen generators are never used for training or for
checkpoint selection.

Example (the configuration reported in the paper):
    python Code/train_multi.py \
        --output Results/final_colab/final_multi_generator_frozen_10ep \
        --train-generators stable_diffusion_v_1_5 BigGAN glide \
        --epochs 10 --batch-size 32 --learning-rate 0.001 \
        --max-train-per-generator 333 --max-val-per-generator 67 \
        --max-eval-per-class 200 --freeze-backbone
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from paths import DATA
from src.data import BinaryImageDataset, GENERATORS, MultiGeneratorImageDataset
from src.engine import run_epoch
from src.model import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-generator ResNet-18 training and cross-generator test.")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-generators", nargs="+", choices=GENERATORS, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-per-generator", type=int, default=333)
    parser.add_argument("--max-val-per-generator", type=int, default=67)
    parser.add_argument("--max-eval-per-class", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=5438)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measure(result: dict) -> dict:
    y = result["targets"]
    pred = result["predictions"]
    prob = result["probabilities"]
    return {
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "ai_precision": precision_score(y, pred, zero_division=0),
        "ai_recall": recall_score(y, pred, zero_division=0),
        "ai_f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "count": result["count"],
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def validation_key(validation: dict) -> tuple[float, float]:
    """Accuracy first, macro-F1 only as the exact-tie breaker."""
    return float(validation["accuracy"]), float(validation["macro_f1"])


def selection_metadata(history: list[dict]) -> dict:
    selected = max(history, key=lambda row: validation_key(row["validation"]))
    return {
        "checkpoint_metric": "validation_accuracy",
        "checkpoint_tie_breaker": "validation_macro_f1",
        "checkpoint_full_tie": "keep_earliest_epoch",
        "best_epoch": selected["epoch"],
        "best_validation_accuracy": selected["validation"]["accuracy"],
        "best_validation_macro_f1": selected["validation"]["macro_f1"],
        "selected_validation_loss": selected["validation"]["loss"],
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new output directory")
    seed_everything(args.seed)
    args.output.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = build_resnet18(not args.no_pretrained, args.freeze_backbone)
    model.to(device)
    if weights:
        train_transform = eval_transform = weights.transforms()
    else:
        train_transform = eval_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_set = MultiGeneratorImageDataset(
        args.data, args.train_generators, "train", train_transform,
        args.max_train_per_generator, args.seed)
    val_set = MultiGeneratorImageDataset(
        args.data, args.train_generators, "val", eval_transform,
        args.max_val_per_generator, args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"Device: {device} | train={len(train_set)} | val={len(val_set)}", flush=True)
    print("Training generators:", ", ".join(args.train_generators), flush=True)
    print(f"Train generator counts: {train_set.generator_counts}", flush=True)

    history: list[dict] = []
    best_key = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train": measure(train_result), "validation": measure(val_result)}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={row['train']['loss']:.4f}, "
              f"val_accuracy={row['validation']['accuracy']:.4f}, "
              f"val_macro_f1={row['validation']['macro_f1']:.4f}", flush=True)
        key = validation_key(row["validation"])
        if key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.output / "best_model.pt")

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    results = {
        "device": str(device),
        "method": "multi_generator_frozen_resnet18_layer4_fc",
        "train_generators": args.train_generators,
        "config": vars(args),
        **selection_metadata(history),
        "history": history,
        "cross_generator": {},
    }
    for generator in GENERATORS:
        test_set = BinaryImageDataset(args.data, [generator], "test", eval_transform,
                                      args.max_eval_per_class, args.seed)
        test_loader = DataLoader(test_set, shuffle=False, **loader_args)
        result = measure(run_epoch(model, test_loader, criterion, device))
        results["cross_generator"][generator] = result
        tag = "seen" if generator in args.train_generators else "unseen"
        print(f"{generator:24s} ({tag}): accuracy={result['accuracy']:.4f}, "
              f"macro_f1={result['macro_f1']:.4f}, roc_auc={result['roc_auc']:.4f}", flush=True)

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
