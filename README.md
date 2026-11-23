# AI_Generated_Image_Detection

EE5438 Group 20 课程项目：**Beating a Frozen CLIP Detector on Unseen Image Generators** —— 研究如何检测「训练时从未见过其生成器」的 AI 生成图像。

核心方法 **DELP**：用冻结的 CLIP ViT-L/14 与 DINOv2 ViT-L/14 提取特征，逐块 L2 归一化后拼接，再接一个仅 **1,793 个参数**的逻辑回归探针。在 Unbiased Tiny GenImage 基准（23,328 张图、7 个生成器）上，未见过生成器的平均准确率达 **95.9%**，优于冻结 CLIP（69.4%）、ResNet-18（82.8%）与 UnivFD（94.1%）。

本仓库**只包含可运行代码**（`Code/`）。数据集、特征缓存、权重、实验结果、报告与答辩材料因体积或隐私原因未上传，需按本文自行准备或由脚本重新生成。

---

## 1. 仓库内容与目录约定

| 路径 | 说明 | 是否随仓库提供 |
| --- | --- | --- |
| `Code/` | 全部 Python 入口脚本及 `src/`（数据划分、模型、训练引擎） | ✅ 是 |
| `Code/requirements.txt` | 锁定版本的依赖清单 | ✅ 是 |
| `Dataset/` | 8 个类别共 23,328 张图片 | ❌ 需自行下载（如 GenImage） |
| `Results/final_v3/` | 各实验 `metrics.json`、汇总与图表 | ❌ 由训练脚本生成 |
| `Report/` | LaTeX 报告、PPT、讲稿、网页 Demo 页面 | ❌ 由构建脚本生成 |
| `~/ee5438_cache/` | CLIP / ResNet / DINOv2 特征缓存（`.npz`） | ❌ 由 `features.py` / `dino_features.py` 生成 |
| `~/dinov2-large/` | DINOv2 ViT-L/14 权重的 Hugging Face 快照 | ❌ 需自行下载 |

脚本中没有硬编码绝对路径。`Code/paths.py` 从自身位置推导仓库根目录，并支持用环境变量覆盖：

| 内容 | 默认路径 | 覆盖变量 |
| --- | --- | --- |
| 数据集 | `<项目根>/Dataset` | `EE5438_DATA` |
| 特征缓存 | `~/ee5438_cache` | `EE5438_CACHE` |
| DINOv2 权重 | `~/dinov2-large` | `EE5438_DINO` |

---

## 2. 环境安装

实测环境：Python 3.9、RTX 2080（8 GB）、CUDA 12.1、PyTorch 2.5.1。

```bash
# 1) 创建虚拟环境（Windows 用 .venv\Scripts\activate）
python -m venv .venv
source .venv/bin/activate

# 2) 按 CUDA 版本安装 PyTorch；无 GPU 时去掉 --index-url，安装 CPU 版即可
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 3) 其余依赖
python -m pip install -r Code/requirements.txt
```

首次运行 `features.py` / `demo_server.py` 时，`open_clip` 会自动下载 CLIP ViT-L/14 权重（需联网）；DINOv2 权重需准备本地快照：

```bash
pip install huggingface_hub
huggingface-cli download facebook/dinov2-large --local-dir ~/dinov2-large
```

---

## 3. 数据准备

将 8 个类别文件夹放在 `Dataset/` 下：

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

数据集划分**不移动或复制文件**：对 `seed:类名/文件名` 做 SHA-1 哈希（seed 固定为 `5438`），按 70/15/15 落入 train/val/test，因此任何机器上划分完全一致。

检查数据是否完整：

```bash
python Code/inspect_dataset.py
```

---

## 4. 一条命令跑完整项目（训练 + 评估 + 生成报告与 Demo）

```bash
# 先打印全部将要执行的命令，确认路径与参数
python Code/run_pipeline.py --dry-run

# 按依赖顺序执行所有阶段
python Code/run_pipeline.py
```

这会依次完成：特征提取 → DINOv2 特征 → SVM / ResNet-18 基线 → 线性探针 → 鲁棒性 / 消融 / LOGO 评估 → 交叉验证 → 汇总分析 → 生成 LaTeX 报告与网页 Demo。所有 JSON 结果写入 `Results/final_v3/`，报告与 Demo 页面写入 `Report/`。

完整阶段列表：

| 阶段 | 脚本 | 主要产物 |
| --- | --- | --- |
| `audit` | `inspect_dataset.py` | 类别数量、格式、分辨率审计 |
| `features` | `features.py` | `<cache>/<variant>.npz`：CLIP 768-d、ResNet-18 512-d、24-d 频谱特征 |
| `dino` | `dino_features.py` | `<cache>/dino_<variant>.npz`：DINOv2 1024-d |
| `svm` | `svm_baseline.py` | `Results/final_v3/classical_svm_3gen/metrics.json` |
| `resnet` | `train_multi.py` | `Results/final_colab/final_multi_generator_frozen_10ep/metrics.json` |
| `probe` | `probe.py` | `Results/final_v3/main_lin/metrics.json`（P1 主对比） |
| `robust` | `probe.py` | `main_robust/`、`aug_tta_dual/`（退化与 TTA 鲁棒性） |
| `ablation` | `probe.py` | `Results/final_v3/ablation/metrics.json`（6 种表示） |
| `regime` | `probe.py` | `Results/final_v3/single_gen_dual/metrics.json` |
| `logo` | `probe.py` | `Results/final_v3/logo/<生成器>/metrics.json`（7 折） |
| `cv` | `cv_select.py` | `Results/final_v3/cv/cv_results.json` |
| `examples` | `make_example_grid.py` | `Report/latex/figures/dataset_examples.png` |
| `analyze` | `analyze.py` | `Results/final_v3/summary.json` + 论文插图 |
| `assets` | `build_latex_assets.py` | `Results/final_v3/latex_assets.json` + 诊断图 |
| `benchmark` | `benchmark.py` | `Results/final_v3/efficiency.json` |
| `report` | `build_latex_report.py` | `Report/latex/technicalReport.tex` |
| `demo` | `build_demo_page.py` | `Report/demo/index.html` |
| `slides` | `build_slides.py` + `build_speaker_scripts.py` | 答辩 PPT 与分段讲稿 |

特征阶段会跳过已存在的 `.npz`，ResNet 阶段跳过已完成的 run（加 `--force` 强制重训）。只改探针配置时无需重新编码 23,328 张图：

```bash
python Code/run_pipeline.py --stages probe robust ablation regime logo cv examples analyze assets benchmark report demo slides
```

---

## 5. 训练：单独运行实验

`run_pipeline.py` 只是按顺序调用下面的 CLI，每个脚本都可单独运行。

### 5.1 P1 主对比（UnivFD vs. DELP）

```bash
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/main_lin \
  --train-generators stable_diffusion_v_1_5 BigGAN glide \
  --methods univfd dino_clip_lin \
  --max-val-per-generator 67 --max-eval-per-class 200
```

### 5.2 LOGO（留一生成器交叉验证）

以留出 ADM 为例：

```bash
python Code/probe.py --cache ~/ee5438_cache --output Results/final_v3/logo/ADM \
  --train-generators BigGAN glide Midjourney stable_diffusion_v_1_5 VQDM wukong \
  --methods univfd dino_clip_lin --max-val-per-generator 67 --max-eval-per-class 200
```

### 5.3 六种表示的消融实验

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

### 5.5 ResNet-18 多生成器基线（10 epoch，冻结主干）

```bash
python Code/train_multi.py --data Dataset \
  --output Results/final_colab/final_multi_generator_frozen_10ep \
  --train-generators stable_diffusion_v_1_5 BigGAN glide --epochs 10 --batch-size 32 \
  --learning-rate 0.001 --max-train-per-generator 333 --max-val-per-generator 67 \
  --max-eval-per-class 200 --freeze-backbone
```

> `--split-seed` 必须保持 `5438`；`--aug` 选择训练退化，`--tta` 选择测试时平均。
> PowerShell 中用 `$env:USERPROFILE` 代替 `~`；脚本本身会正确展开波浪号。

---

## 6. 运行 Demo（网页实时检测）

Demo 从 `Report/demo/index.html` 提供页面，由本地服务加载冻结编码器并从缓存重训探针，对上传图片返回真实的 P(fake)。

**前置条件**：特征缓存已生成（跑过 `run_pipeline.py` 的 `features` 和 `dino` 阶段），页面已生成（`demo` 阶段或手动执行 `python Code/build_demo_page.py`）。

```bash
# 启动本地推理服务（会自动打开浏览器）
python Code/demo_server.py --port 8000

# 无图形界面 / 不自动打开浏览器时
python Code/demo_server.py --port 8000 --no-open
```

然后访问 **http://127.0.0.1:8000** 。服务启动时会在控制台打印探针重建情况和 P1 unseen-4 准确率（约 95.9%）。

- 拖拽 / 粘贴 / 上传任意照片 → 七阶段动画后返回 P(fake) 与实测耗时。
- 未启动服务时，直接打开 `Report/demo/index.html` 会进入 **offline** 模式，只回放预计算样本。
- 图片只在本地处理，不会上传。

---

## 7. 生成 PDF 报告

```bash
python Code/build_latex_report.py
cd Report/latex
latexmk -pdf technicalReport.tex
```

`latexmk` 自动运行 pdfLaTeX 与 biber，直至参考文献和交叉引用稳定。产物为 `Report/latex/technicalReport.pdf`。

---

## 8. 复现性与已知限制

- 哈希划分 + 固定 seed `5438` + 仅用验证集选 checkpoint + 固定逻辑回归配置，消除了人工选择。
- 在缓存特征给定的前提下，`probe.py`、`analyze.py`、`build_latex_report.py` 结果确定。
- ResNet-18 的 GPU 训练受标准浮点非确定性影响；仓库中的 `metrics.json` 是提交报告所依据的实测运行结果。
- 单生成器训练器 `train.py` / `train_aug.py` 和 proposal 阶段的辅助脚本保留作溯源，不属于 `run_pipeline.py` 的必经流程。

更细的单阶段命令与产物说明见 `Code/README.md`。
