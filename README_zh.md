# AI_Generated_Image_Detection

[English](README.md) | [简体中文](README_zh.md)

一个面向 AI 生成图像检测的项目，重点解决**训练阶段从未见过的图像生成器**的检测问题。

核心方法 **DELP**：使用冻结的 CLIP ViT-L/14 与 DINOv2 ViT-L/14 提取特征，逐块做 L2 归一化后拼接，再接一个只有 **1,793 个参数**的逻辑回归探针。在 Unbiased Tiny GenImage 基准（23,328 张图像、7 个生成器）上，对未见生成器的平均准确率达到 **95.9%**，而冻结 CLIP 检测器为 69.4%、ResNet-18 为 82.8%、UnivFD 为 94.1%。

本仓库只包含可运行代码（`Code/`）。数据集、特征缓存、模型权重、结果和生成的图表不在仓库中，需要自行准备或由脚本重新生成。

---

## 1. 仓库结构

| 路径 | 说明 | 是否包含 |
| --- | --- | --- |
| `Code/` | 所有 Python 入口脚本与 `src/`（数据划分、模型、训练引擎） | 是 |
| `Code/requirements.txt` | 固定版本的依赖 | 是 |
| `Dataset/` | 8 个类别目录，共 23,328 张图像 | 否 —— 需自行下载（如 GenImage） |
| `Results/final_v3/` | 各实验的 `metrics.json`、汇总与图表 | 否 —— 由脚本生成 |
| `~/ee5438_cache/` | CLIP / ResNet / DINOv2 特征缓存（`.npz`） | 否 —— 由 `features.py` / `dino_features.py` 生成 |
| `~/dinov2-large/` | DINOv2 权重的本地 Hugging Face 快照 | 否 —— 需自行下载 |

脚本中没有硬编码的绝对路径。`Code/paths.py` 会根据自身位置推导项目根目录，并支持用环境变量覆盖：

| 内容 | 默认值 | 覆盖变量 |
| --- | --- | --- |
| 数据集 | `<root>/Dataset` | `EE5438_DATA` |
| 特征缓存 | `~/ee5438_cache` | `EE5438_CACHE` |
| DINOv2 权重 | `~/dinov2-large` | `EE5438_DINO` |

---

## 2. 环境

实测环境：Python 3.9、RTX 2080（8 GB）、CUDA 12.1、PyTorch 2.5.1。

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# CUDA 12.1 的 PyTorch 安装命令；CPU 环境去掉 --index-url
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r Code/requirements.txt
```

首次运行时，`open_clip` 会自动下载 CLIP ViT-L/14 权重（需要联网）。DINOv2 权重需准备本地快照：

```bash
pip install huggingface_hub
huggingface-cli download facebook/dinov2-large --local-dir ~/dinov2-large
```

---

## 3. 数据

将 8 个类别目录放在 `Dataset/` 下：

```
Dataset/
├── Nature/                    5,828 张真实图像
├── ADM/                       2,500 张
├── BigGAN/                    2,500 张
├── glide/                     2,500 张
├── Midjourney/                2,500 张
├── stable_diffusion_v_1_5/    2,500 张
├── VQDM/                      2,500 张
└── wukong/                    2,500 张
```

划分不会移动或复制文件。每张图像通过对 `seed:类名/文件名` 做哈希（seed 固定为 `5438`）分配到 70/15/15 的 train/val/test 桶，因此任何机器上的划分完全一致。可用以下命令检查数据集：

```bash
python Code/inspect_dataset.py
```

---

## 4. 完整流程

```bash
python Code/run_pipeline.py --dry-run   # 打印所有命令但不执行
python Code/run_pipeline.py             # 按依赖顺序执行所有阶段
```

该流程依次完成：特征提取 → DINOv2 特征 → SVM / ResNet-18 基线 → 线性探针 → 鲁棒性、消融与留一生成器评估 → 交叉验证 → 汇总分析 → 诊断图 → 耗时基准测试 → 网页 Demo。所有 JSON 结果写入 `Results/final_v3/`。

| 阶段 | 脚本 | 主要产物 |
| --- | --- | --- |
| `audit` | `inspect_dataset.py` | 类别数量、格式、分辨率审计 |
| `features` | `features.py` | `<cache>/<variant>.npz`：CLIP 768 维、ResNet-18 512 维、24 维频谱特征 |
| `dino` | `dino_features.py` | `<cache>/dino_<variant>.npz`：DINOv2 1024 维 |
| `svm` | `svm_baseline.py` | `Results/final_v3/classical_svm_3gen/metrics.json` |
| `resnet` | `train_multi.py` | `Results/final_colab/final_multi_generator_frozen_10ep/metrics.json` |
| `probe` | `probe.py` | `Results/final_v3/main_lin/metrics.json`（P1 主对比） |
| `robust` | `probe.py` | `main_robust/`、`aug_tta_dual/`（退化与 TTA） |
| `ablation` | `probe.py` | `Results/final_v3/ablation/metrics.json`（6 种表示） |
| `regime` | `probe.py` | `Results/final_v3/single_gen_dual/metrics.json` |
| `logo` | `probe.py` | `Results/final_v3/logo/<generator>/metrics.json`（7 折） |
| `cv` | `cv_select.py` | `Results/final_v3/cv/cv_results.json` |
| `examples` | `make_example_grid.py` | 真实图像与生成图像的对比图 |
| `analyze` | `analyze.py` | `Results/final_v3/summary.json` + 评估图 |
| `assets` | `build_latex_assets.py` | `Results/final_v3/latex_assets.json` + 诊断图 |
| `benchmark` | `benchmark.py` | `Results/final_v3/efficiency.json` |
| `demo` | `build_demo_page.py` | `Report/demo/index.html` |

特征阶段会跳过已存在的 `.npz` 文件，ResNet 阶段会跳过已完成的运行（`--force` 可强制重训）。如果只修改了探针配置，可以只重跑相关阶段：

```bash
python Code/run_pipeline.py --stages probe robust ablation regime logo cv analyze assets benchmark demo
```

---

## 5. 单独训练实验

`run_pipeline.py` 只是按顺序调用下面的脚本，每个脚本都可以单独运行。

### 5.1 主对比（UnivFD vs. DELP）

```bash
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/main_lin \
  --train-generators stable_diffusion_v_1_5 BigGAN glide \
  --methods univfd dino_clip_lin \
  --max-val-per-generator 67 --max-eval-per-class 200
```

### 5.2 留一生成器交叉验证（以留出 ADM 为例）

```bash
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/logo/ADM \
  --train-generators BigGAN glide Midjourney stable_diffusion_v_1_5 VQDM wukong \
  --methods univfd dino_clip_lin --max-val-per-generator 67 --max-eval-per-class 200
```

### 5.3 表示方式消融

```bash
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/ablation \
  --train-generators stable_diffusion_v_1_5 BigGAN glide \
  --methods univfd dino_lin dino_clip_resnet_lin dino_clip_freq_lin dino_clip_lin dino_clip
```

### 5.4 经典基线（SVM）

```bash
python Code/svm_baseline.py --data Dataset --output Results/final_v3/classical_svm_3gen \
  --train-generators stable_diffusion_v_1_5 BigGAN glide --max-eval-per-class 200
```

### 5.5 ResNet-18 基线（10 epoch，冻结主干）

```bash
python Code/train_multi.py --data Dataset \
  --output Results/final_colab/final_multi_generator_frozen_10ep \
  --train-generators stable_diffusion_v_1_5 BigGAN glide --epochs 10 --batch-size 32 \
  --learning-rate 0.001 --max-train-per-generator 333 --max-val-per-generator 67 \
  --max-eval-per-class 200 --freeze-backbone
```

`--split-seed` 必须保持 `5438`。`--aug` 指定训练退化，`--tta` 指定测试时平均。在 PowerShell 中用 `$env:USERPROFILE` 代替 `~`；脚本本身在所有平台都会展开波浪号。

---

## 6. 网页 Demo

Demo 服务会加载两个冻结编码器，用与 `probe.py` 相同的代码从缓存特征重建逻辑回归探针，并对上传的照片返回真实的 P(fake)。

前置条件：特征缓存已生成（运行 `features` 和 `dino` 阶段），且页面已生成（运行 `demo` 阶段，或执行 `python Code/build_demo_page.py`）。

```bash
python Code/demo_server.py --port 8000            # 自动打开浏览器
python Code/demo_server.py --port 8000 --no-open  # 不打开浏览器
```

然后访问 **http://127.0.0.1:8000**。启动时控制台会打印探针重建信息和未见生成器的准确率（约 95.9%）。拖拽、粘贴或上传任意照片，即可看到七阶段动画以及实测的 P(fake) 和耗时。未启动服务时，直接打开 `Report/demo/index.html` 会进入 `offline` 模式，只回放预计算样本。图片只在本地处理，不会离开本机。

---

## 7. 可复现性

- 哈希划分、固定 seed `5438`、仅用验证集选择 checkpoint、固定的逻辑回归配置，消除了人工选择。
- 在缓存特征给定的前提下，`probe.py`、`analyze.py` 和 `build_latex_assets.py` 的结果是确定的。
- ResNet-18 的 GPU 训练受标准浮点非确定性影响；仓库中保留的 `metrics.json` 是报告数字所对应的实测运行结果。
- 单生成器训练器（`train.py`、`train_aug.py`）和早期辅助脚本（`generate_dataset_figures.py`、`plot_results.py`）保留作溯源，不属于 `run_pipeline.py` 的必经流程。

详细的单阶段命令见 `Code/README.md`（提供英文与中文两种版本）。
