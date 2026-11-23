from __future__ import annotations

import argparse
import json
import os
import random
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

from src.data import BinaryImageDataset, GENERATORS
from src.engine import run_epoch
from src.model import build_resnet18
from src.augmentations import build_robust_train_transform
from paths import DATA, ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Train on one generator and test on unseen generators.")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--train-generator", choices=GENERATORS, default="stable_diffusion_v_1_5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "latest")
    parser.add_argument("--seed", type=int, default=5438)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(result):
    y = result["targets"]
    pred = result["predictions"]
    prob = result["probabilities"]

    return {
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "balanced_accuracy": balanced_accuracy_score(y, pred),

        # Binary metrics: class 1 = AI-generated
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),

        # Main report metric
        "macro_f1": f1_score(
            y,
            pred,
            average="macro",
            zero_division=0
        ),

        "roc_auc": roc_auc_score(y, prob),

        "count": result["count"],
        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=[0, 1]
        ).tolist(),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models_cache"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, weights = build_resnet18(not args.no_pretrained, args.freeze_backbone)
    model.to(device)
    if weights:
        eval_transform = weights.transforms()
        train_transform = build_robust_train_transform(eval_transform)
    else:
        train_transform = eval_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_set = BinaryImageDataset(args.data, [args.train_generator], "train", train_transform,
                                   args.max_train_per_class, args.seed)
    val_set = BinaryImageDataset(args.data, [args.train_generator], "val", eval_transform,
                                 args.max_eval_per_class, args.seed)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=1e-4)

    history, best_accuracy = [], -1.0
    print(f"Device: {device} | train={len(train_set)} | val={len(val_set)}")
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train": metrics(train_result), "validation": metrics(val_result)}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={row['train']['loss']:.4f}, "
              f"val_accuracy={row['validation']['accuracy']:.4f}")
        if row["validation"]["accuracy"] > best_accuracy:
            best_accuracy = row["validation"]["accuracy"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.output / "best_model.pt")

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    results = {"device": str(device), "train_generator": args.train_generator,
               "history": history, "cross_generator": {}}
    for generator in GENERATORS:
        test_set = BinaryImageDataset(args.data, [generator], "test", eval_transform,
                                      args.max_eval_per_class, args.seed)
        test_loader = DataLoader(test_set, shuffle=False, **loader_args)
        result = metrics(run_epoch(model, test_loader, criterion, device))
        results["cross_generator"][generator] = result
        tag = "in-domain" if generator == args.train_generator else "unseen"
        print(f"{generator:24s} ({tag}): accuracy={result['accuracy']:.4f}, f1={result['f1']:.4f}")

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

