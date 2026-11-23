# EE5438 Group 20 — AI-Generated Image Detection

Code for the final project *Outperforming a Frozen CLIP Detector on Unseen Image
Generators*. The pipeline trains and evaluates four detectors on the Unbiased
Tiny GenImage benchmark (23,328 images, seven generators) under two protocols:
a three-train/four-unseen split (P1) and seven-fold leave-one-generator-out
(P2). Every number in the report is generated from the JSON files written here.

## Repository layout

```
Dataset/                  eight class folders (not distributed with the code)
Code/                     all Python entry points (this folder)
Code/src/                 dataset, model, training engine and augmentation helpers
Results/final_v3/         JSON results and figures consumed by the report
Results/final_colab/      recorded ResNet-18 run
Report/                   LaTeX sources, figures, PDF, slides, demo page
```

## Environment

Versions measured on the experimental machine (Python 3.9.0, one RTX 2080, CUDA 12.1):

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Code/requirements.txt
```

`Code/requirements.txt` pins every remaining package (NumPy, SciPy, scikit-learn,
scikit-image, pandas, matplotlib, Pillow, open_clip_torch, transformers, timm,
python-pptx).

## Data, splits and caches

Place the eight class folders under `Dataset/`: `Nature` (5,828 real images) and
`ADM`, `BigGAN`, `glide`, `Midjourney`, `stable_diffusion_v_1_5`, `VQDM`,
`wukong` (2,500 generated images each). Splits are computed by hashing
`seed:class/filename` with seed 5438 into the 70/15/15 buckets; no files are
moved or copied.

The feature cache and DINOv2 weights live outside the repository by default:

| Path | Default | Override |
| --- | --- | --- |
| dataset | `Dataset/` | `EE5438_DATA` |
| feature cache | `~/ee5438_cache/` | `EE5438_CACHE` |
| DINOv2 weights | `~/dinov2-large/` (Hugging Face snapshot of `facebook/dinov2-large`) | `EE5438_DINO` |

No script contains an absolute path: `Code/paths.py` resolves the repository
root from its own location.

## Full reproduction from scratch

```bash
python Code/run_pipeline.py --dry-run    # print every resolved command
python Code/run_pipeline.py              # run every stage in dependency order
```

The feature stages skip `.npz` files that already exist and the ResNet stage
skips a completed run unless `--force` is passed, so a partial rerun never
re-encodes the dataset:

```bash
python Code/run_pipeline.py --stages probe robust ablation regime logo cv examples analyze assets benchmark report demo slides
```

### Stages

| Stage | Script | Main output |
| --- | --- | --- |
| `audit` | `inspect_dataset.py` | console audit of counts, formats, dimensions |
| `features` | `features.py` | `<cache>/<variant>.npz`: CLIP 768-d, ResNet-18 512-d, 24-d frequency descriptor, manifest |
| `dino` | `dino_features.py` | `<cache>/dino_<variant>.npz`: DINOv2 1024-d |
| `svm` | `svm_baseline.py` | `Results/final_v3/classical_svm_3gen/metrics.json` |
| `resnet` | `train_multi.py` | `Results/final_colab/final_multi_generator_frozen_10ep/metrics.json` |
| `probe` | `probe.py` | `Results/final_v3/main_lin/metrics.json` |
| `robust` | `probe.py` | `main_robust/` and `aug_tta_dual/` |
| `ablation` | `probe.py` | `Results/final_v3/ablation/metrics.json` |
| `regime` | `probe.py` | `Results/final_v3/single_gen_dual/metrics.json` |
| `logo` | `probe.py` | `Results/final_v3/logo/<generator>/metrics.json` (7 folds) |
| `cv` | `cv_select.py` | `Results/final_v3/cv/cv_results.json` |
| `examples` | `make_example_grid.py` | `Report/latex/figures/dataset_examples.png` |
| `analyze` | `analyze.py` | `Results/final_v3/summary.json`, manuscript figures |
| `assets` | `build_latex_assets.py` | `Results/final_v3/latex_assets.json`, diagnostic figures |
| `benchmark` | `benchmark.py` | `Results/final_v3/efficiency.json` |
| `report` | `build_latex_report.py` | `Report/latex/technicalReport.tex` |
| `demo` | `build_demo_page.py` | `Report/demo/index.html` |
| `slides` | `build_slides.py` + `build_speaker_scripts.py` | `Report/Defense_Slides_EE5438_2026-11-23.pptx` (16 slides, speaker notes) and `Report/scripts/*.docx` (five per-speaker scripts generated from `Report/DEFENSE_SCRIPT.md`) |

## Running one experiment directly

The scripts are ordinary CLIs; `run_pipeline.py` only sequences them.

```bash
# headline P1 comparison (UnivFD vs. DELP)
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/main_lin \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --methods univfd dino_clip_lin \
    --max-val-per-generator 67 --max-eval-per-class 200

# one leave-one-generator-out fold (held-out = ADM)
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/logo/ADM \
    --train-generators BigGAN glide Midjourney stable_diffusion_v_1_5 VQDM wukong \
    --methods univfd dino_clip_lin --max-val-per-generator 67 --max-eval-per-class 200

# ablation over six representations
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/ablation \
    --train-generators stable_diffusion_v_1_5 BigGAN glide \
    --methods univfd dino_lin dino_clip_resnet_lin dino_clip_freq_lin dino_clip_lin dino_clip

# classical baseline and the ResNet-18 baseline
python Code/svm_baseline.py --data Dataset --output Results/final_v3/classical_svm_3gen \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --max-eval-per-class 200

python Code/train_multi.py --data Dataset \
    --output Results/final_colab/final_multi_generator_frozen_10ep \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --epochs 10 --batch-size 32 \
    --learning-rate 0.001 --max-train-per-generator 333 --max-val-per-generator 67 \
    --max-eval-per-class 200 --freeze-backbone
```

On Windows PowerShell use `$env:USERPROFILE` instead of `~`; the scripts expand
the tilde on every platform. `--aug` selects training variants, `--tta` selects
test-time averaging variants, and `--split-seed` must stay `5438`.

## Live demonstration

```bash
python Code/demo_server.py --port 8000
```

The server loads the two frozen encoders, rebuilds the logistic probe from the
cached features with the same code as `probe.py`, recomputes the unseen
accuracy at start-up and then classifies uploads through the same path. The
static page in `Report/demo/index.html` falls back to pre-computed samples and
labels itself offline when no server is reachable.

## Building the PDF

```bash
python Code/build_latex_report.py
cd Report/latex
latexmk -pdf technicalReport.tex
```

`latexmk` runs pdfLaTeX and biber until the bibliography settles. The result is
`Report/latex/technicalReport.pdf`; `Report/Final_Report_EE5438_LaTeX.pdf` is
the archived submission copy.

## Determinism

The hash split, split seed 5438, validation-only checkpoint selection and the
fixed logistic-regression configuration remove manual choices. The feature
caches are keyed by the manifest, and `probe.py`, `analyze.py` and
`build_latex_report.py` are deterministic given the cached embeddings. GPU
training of the ResNet-18 baseline remains subject to standard floating-point
nondeterminism; the shipped `metrics.json` is the measured run behind the
submitted numbers.

## Files not part of the pipeline

Exclude `.venv/`, `__pycache__/`, `models_cache/` and the archived duplicate
`final_metrics_version/` when submitting. The single-generator trainers
`train.py` and `train_aug.py`, the proposal-stage helpers
`generate_dataset_figures.py`, `plot_results.py` and `build_report.py` are kept
for provenance; `src/augmentations.py` supports `train_aug.py`. None of them is
required by `run_pipeline.py`.
