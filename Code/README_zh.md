# AI 生成图像检测 —— 代码

[English](README.md) | [简体中文](README_zh.md)

项目 *Outperforming a Frozen CLIP Detector on Unseen Image Generators* 的代码。该流水线在 Unbiased Tiny GenImage 基准（23,328 张图像、7 个生成器）上训练和评估四种检测器，采用两种协议：三训练/四未见划分（P1）和七折留一生成器（P2）。结果文件中的所有数字都由这里生成。

## 仓库结构

```
Dataset/                  8 个类别目录（不随代码分发）
Code/                     所有 Python 入口脚本（本目录）
Code/src/                 数据集、模型、训练引擎与增强辅助模块
Results/final_v3/         JSON 结果与图表
Results/final_colab/      记录下来的 ResNet-18 运行结果
```

## 环境

实测环境（Python 3.9.0、单张 RTX 2080、CUDA 12.1）：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Code/requirements.txt
```

`Code/requirements.txt` 固定了其余所有依赖（NumPy、SciPy、scikit-learn、scikit-image、pandas、matplotlib、Pillow、open_clip_torch、transformers、timm）。

## 数据、划分与缓存

将 8 个类别目录放在 `Dataset/` 下：`Nature`（5,828 张真实图像）以及 `ADM`、`BigGAN`、`glide`、`Midjourney`、`stable_diffusion_v_1_5`、`VQDM`、`wukong`（各 2,500 张生成图像）。划分通过对 `seed:类名/文件名` 做哈希（seed 5438）分配到 70/15/15 的桶，不移动或复制任何文件。

特征缓存和 DINOv2 权重默认放在仓库外：

| 路径 | 默认值 | 覆盖变量 |
| --- | --- | --- |
| 数据集 | `Dataset/` | `EE5438_DATA` |
| 特征缓存 | `~/ee5438_cache/` | `EE5438_CACHE` |
| DINOv2 权重 | `~/dinov2-large/`（`facebook/dinov2-large` 的 Hugging Face 快照） | `EE5438_DINO` |

脚本中没有绝对路径：`Code/paths.py` 会根据自身位置推导仓库根目录。

## 从零完整复现

```bash
python Code/run_pipeline.py --dry-run    # 打印每条实际命令
python Code/run_pipeline.py              # 按依赖顺序执行所有阶段
```

特征阶段会跳过已存在的 `.npz` 文件，ResNet 阶段会跳过已完成的运行（除非传入 `--force`），因此部分重跑不会重新编码整个数据集：

```bash
python Code/run_pipeline.py --stages probe robust ablation regime logo cv analyze assets benchmark demo
```

### 阶段

| 阶段 | 脚本 | 主要产物 |
| --- | --- | --- |
| `audit` | `inspect_dataset.py` | 数量、格式、尺寸的控制台审计 |
| `features` | `features.py` | `<cache>/<variant>.npz`：CLIP 768 维、ResNet-18 512 维、24 维频谱特征、manifest |
| `dino` | `dino_features.py` | `<cache>/dino_<variant>.npz`：DINOv2 1024 维 |
| `svm` | `svm_baseline.py` | `Results/final_v3/classical_svm_3gen/metrics.json` |
| `resnet` | `train_multi.py` | `Results/final_colab/final_multi_generator_frozen_10ep/metrics.json` |
| `probe` | `probe.py` | `Results/final_v3/main_lin/metrics.json` |
| `robust` | `probe.py` | `main_robust/` 与 `aug_tta_dual/` |
| `ablation` | `probe.py` | `Results/final_v3/ablation/metrics.json` |
| `regime` | `probe.py` | `Results/final_v3/single_gen_dual/metrics.json` |
| `logo` | `probe.py` | `Results/final_v3/logo/<generator>/metrics.json`（7 折） |
| `cv` | `cv_select.py` | `Results/final_v3/cv/cv_results.json` |
| `examples` | `make_example_grid.py` | 真实图像与生成图像的对比图 |
| `analyze` | `analyze.py` | `Results/final_v3/summary.json`、评估图 |
| `assets` | `build_latex_assets.py` | `Results/final_v3/latex_assets.json`、诊断图 |
| `benchmark` | `benchmark.py` | `Results/final_v3/efficiency.json` |
| `demo` | `build_demo_page.py` | `Report/demo/index.html` |

## 直接运行单个实验

这些脚本都是普通的 CLI，`run_pipeline.py` 只是把它们串起来。

```bash
# P1 主对比（UnivFD vs DELP）
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/main_lin \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --methods univfd dino_clip_lin \
    --max-val-per-generator 67 --max-eval-per-class 200

# 留一生成器单折（留出 ADM）
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/logo/ADM \
    --train-generators BigGAN glide Midjourney stable_diffusion_v_1_5 VQDM wukong \
    --methods univfd dino_clip_lin --max-val-per-generator 67 --max-eval-per-class 200

# 六种表示的消融
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/ablation \
    --train-generators stable_diffusion_v_1_5 BigGAN glide \
    --methods univfd dino_lin dino_clip_resnet_lin dino_clip_freq_lin dino_clip_lin dino_clip

# 经典基线与 ResNet-18 基线
python Code/svm_baseline.py --data Dataset --output Results/final_v3/classical_svm_3gen \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --max-eval-per-class 200

python Code/train_multi.py --data Dataset \
    --output Results/final_colab/final_multi_generator_frozen_10ep \
    --train-generators stable_diffusion_v_1_5 BigGAN glide --epochs 10 --batch-size 32 \
    --learning-rate 0.001 --max-train-per-generator 333 --max-val-per-generator 67 \
    --max-eval-per-class 200 --freeze-backbone
```

在 Windows PowerShell 中用 `$env:USERPROFILE` 代替 `~`；脚本在所有平台都会展开波浪号。`--aug` 选择训练退化，`--tta` 选择测试时平均，`--split-seed` 必须保持 `5438`。

## 实时 Demo

```bash
python Code/demo_server.py --port 8000
```

服务会加载两个冻结编码器，用与 `probe.py` 相同的代码从缓存特征重建逻辑回归探针，在启动时重新计算未见生成器准确率，然后通过同一条路径对上传图像分类。当无法连接服务时，`Report/demo/index.html` 中的静态页面会回退到预计算样本，并明确标注为离线状态。

## 可复现性

哈希划分、固定 seed 5438、仅用验证集选择 checkpoint、固定的逻辑回归配置，消除了人工选择。特征缓存以 manifest 为键，在缓存嵌入给定的前提下，`probe.py`、`analyze.py` 与 `build_latex_assets.py` 是确定性的。ResNet-18 的 GPU 训练仍受标准浮点非确定性影响；仓库中保留的 `metrics.json` 是所报告数字对应的实测运行结果。

## 不属于流水线的文件

提交时应排除 `.venv/`、`__pycache__/`、`models_cache/` 和归档的重复目录 `final_metrics_version/`。单生成器训练器 `train.py`、`train_aug.py` 以及早期辅助脚本 `generate_dataset_figures.py`、`plot_results.py` 保留作溯源；`src/augmentations.py` 服务于 `train_aug.py`。这些都不是 `run_pipeline.py` 的必经流程。
