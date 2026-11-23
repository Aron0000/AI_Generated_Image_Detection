from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("metrics", type=Path, nargs="?", default=Path("outputs/smoke_test/metrics.json"))
args = parser.parse_args()
with args.metrics.open(encoding="utf-8") as handle:
    results = json.load(handle)

names = list(results["cross_generator"])
accuracy = [100 * results["cross_generator"][name]["accuracy"] for name in names]
train_generator = results["train_generator"]
colors = ["#E45756" if name == train_generator else "#4C78A8" for name in names]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, accuracy, color=colors)
ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="random baseline (50%)")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Train on {train_generator}: cross-generator evaluation")
ax.tick_params(axis="x", rotation=25)
ax.legend()
for bar, value in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=9)
fig.tight_layout()
output = args.metrics.with_name("cross_generator_accuracy.png")
fig.savefig(output, dpi=180)
print(output.resolve())

