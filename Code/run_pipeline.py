"""One entry point for the full experiment pipeline.

Stages run in dependency order and write exactly the files that the report
consumes.  Use ``--dry-run`` to print the commands without executing them, and
``--stages`` to run a subset.

Typical full reproduction from a clean checkout (dataset in Dataset/):
    python Code/run_pipeline.py --dry-run
    python Code/run_pipeline.py

Fast path when the feature caches already exist (skip feature extraction):
    python Code/run_pipeline.py --stages probe robust ablation regime logo cv analyze assets benchmark demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paths import CACHE, CODE, DATA, DINO_WEIGHTS, RESULTS, ROOT

V3 = RESULTS / "final_v3"
RESNET_OUT = RESULTS / "final_colab" / "final_multi_generator_frozen_10ep"
GENERATORS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_5", "VQDM", "wukong"]
TRAIN_GENERATORS = ["stable_diffusion_v_1_5", "BigGAN", "glide"]
VARIANTS = ["clean", "jpeg90", "jpeg50", "resize50", "blur1", "combo"]

STAGES = [
    "audit", "features", "dino", "svm", "resnet", "probe", "robust", "ablation",
    "regime", "logo", "cv", "examples", "analyze", "assets", "benchmark", "demo",
]


def script(name: str) -> list[str]:
    return [sys.executable, str(CODE / name)]


def probe(output: str, methods: list[str], train: list[str] | None = None,
          aug: list[str] | None = None, tta: list[str] | None = None) -> list[str]:
    command = script("probe.py") + [
        "--cache", str(CACHE),
        "--output", str(V3 / output),
        "--train-generators", *(train or TRAIN_GENERATORS),
        "--methods", *methods,
        "--max-val-per-generator", "67",
        "--max-eval-per-class", "200",
        "--split-seed", "5438",
        "--seeds", "5438",
    ]
    if aug:
        command += ["--aug", *aug]
    if tta:
        command += ["--tta", *tta]
    return command


def build(stage: str) -> list[list[str]]:
    if stage == "audit":
        return [script("inspect_dataset.py")]
    if stage == "features":
        return [script("features.py") + [
            "--data", str(DATA), "--cache", str(CACHE),
            "--variants", *VARIANTS, "--batch-size", "64", "--workers", "4", "--seed", "5438",
        ]]
    if stage == "dino":
        return [script("dino_features.py") + [
            "--cache", str(CACHE), "--variants", *VARIANTS,
            "--batch-size", "32", "--workers", "4", "--splits", "train", "val", "test",
        ]]
    if stage == "svm":
        return [script("svm_baseline.py") + [
            "--data", str(DATA), "--output", str(V3 / "classical_svm_3gen"),
            "--train-generators", *TRAIN_GENERATORS,
            "--max-eval-per-class", "200", "--seed", "5438",
        ]]
    if stage == "resnet":
        return [script("train_multi.py") + [
            "--data", str(DATA), "--output", str(RESNET_OUT),
            "--train-generators", *TRAIN_GENERATORS,
            "--epochs", "10", "--batch-size", "32", "--learning-rate", "0.001",
            "--max-train-per-generator", "333", "--max-val-per-generator", "67",
            "--max-eval-per-class", "200", "--freeze-backbone", "--seed", "5438",
        ]]
    if stage == "probe":
        return [probe("main_lin", ["univfd", "dino_clip_lin"])]
    if stage == "robust":
        return [
            probe("main_robust", ["univfd", "dino_clip_lin"],
                  tta=["jpeg90", "jpeg50", "resize50", "blur1", "combo"]),
            probe("aug_tta_dual", ["univfd", "dino_clip_lin"],
                  aug=["jpeg50", "resize50", "blur1"],
                  tta=["jpeg90", "jpeg50", "resize50", "blur1", "combo"]),
        ]
    if stage == "ablation":
        return [probe("ablation",
                      ["univfd", "dino_lin", "dino_clip_resnet_lin",
                       "dino_clip_freq_lin", "dino_clip_lin", "dino_clip"])]
    if stage == "regime":
        return [probe("single_gen_dual", ["univfd", "dino_clip_lin"],
                      train=["stable_diffusion_v_1_5"])]
    if stage == "logo":
        commands = []
        for held_out in GENERATORS:
            train = [generator for generator in GENERATORS if generator != held_out]
            commands.append(probe(f"logo/{held_out}", ["univfd", "dino_clip_lin"], train=train))
        return commands
    if stage == "cv":
        return [script("cv_select.py") + [
            "--cache", str(CACHE), "--output", str(V3 / "cv"), "--max-eval-per-class", "200",
        ]]
    if stage == "examples":
        return [script("make_example_grid.py")]
    if stage == "analyze":
        return [script("analyze.py")]
    if stage == "assets":
        return [script("build_latex_assets.py")]
    if stage == "benchmark":
        return [script("benchmark.py")]
    if stage == "demo":
        return [script("build_demo_page.py")]
    raise ValueError(stage)


def main() -> None:
    global CACHE, DATA, DINO_WEIGHTS
    parser = argparse.ArgumentParser(description="Run the full EE5438 reproduction pipeline.")
    parser.add_argument("--stages", nargs="+", choices=STAGES + ["all"], default=["all"])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--dino-dir", type=Path, default=DINO_WEIGHTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="retrain the ResNet-18 baseline even if it exists")
    args = parser.parse_args()

    CACHE, DATA, DINO_WEIGHTS = args.cache, args.data, args.dino_dir

    selected = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    for stage in selected:
        commands = build(stage)
        if stage == "resnet" and (RESNET_OUT / "metrics.json").exists() and not args.force:
            print(f"[{stage}] skip: {RESNET_OUT / 'metrics.json'} already exists (use --force to retrain)")
            continue
        for command in commands:
            if args.dry_run:
                print(f"[{stage}] " + subprocess.list2cmdline(command))
                continue
            print(f"[{stage}] " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    if args.dry_run:
        print("\ndry run only: no command was executed")


if __name__ == "__main__":
    main()
