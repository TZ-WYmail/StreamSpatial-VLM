# StreamSpatial-VLM：基于2D-3D互引导的流式空间理解高效框架

> **毕业设计项目** · 南方科技大学 · 计算机科学与技术
> 学生：谭政（12212320）｜ 指导教师：郑锋副教授

---

## 📂 项目文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 实验指南（本文档） | `README.md` | 环境配置、数据准备、实验步骤 |
| 技术设计与实验规划 | [`docs/technical-design.md`](docs/technical-design.md) | 系统架构、核心模块设计、消融实验方案 |
| 任务书与开题报告 | [`docs/project-proposal.md`](docs/project-proposal.md) | 研究内容、要求指标、进程安排 |
| 评估表 | [`docs/assessment.md`](docs/assessment.md) | 导师/检查员评估维度与对应内容 |

---

## 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [实验流程总览](#实验流程总览)
- [Step 1：数据预处理](#step-1数据预处理)
- [Step 2：基线 Profiling（冗余分析）](#step-2基线-profiling冗余分析)
- [Step 3：基线评估（VG-LLM 全量推理）](#step-3基线评估vg-llm-全量推理)
- [Step 4：消融实验 A~E](#step-4消融实验-ae)
- [Step 5：完整框架评估](#step-5完整框架评估)
- [Step 6：可视化与定性分析](#step-6可视化与定性分析)
- [Step 7：流式推理 Demo](#step-7流式推理-demo)
- [实验结果汇总](#实验结果汇总)
- [常见问题](#常见问题)

---

## 项目概述

StreamSpatial-VLM 通过 **2D-3D 双向互引导** 机制，在保持空间理解精度的前提下大幅提升推理效率：

| 核心模块 | 功能 | 关键参数 |
|---------|------|---------|
| **2D→3D Gate** | 语义差异门控，避免无效 3D 计算 | 阈值 $\tau$，历史步长 $k$ |
| **3D→2D Zip** | 几何引导 token 压缩 | 保留率 $r$，权重 $(\alpha, \beta)$ |
| **增量 KV Cache** | 滑动窗口流式推理 | 窗口大小 $w$ |

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n ssvlm python=3.10 -y
conda activate ssvlm
```

### 2. 安装依赖

```bash
cd StreamSpatial-VLM
pip install -r requirements.txt

# VGGT（从源码安装）
pip install git+https://github.com/facebookresearch/vggt.git

# Depth-Anything-V2（从源码安装）
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 4. 配置 Hugging Face Token

模型下载需要 HF Token。在 `configs/baseline_vgllm.yaml` 中设置（或通过环境变量）：

```bash
export HF_TOKEN="your_hf_token_here"
huggingface-cli login --token $HF_TOKEN
```

---

## 数据准备

### 期望的目录结构

下载完模型和数据集后，目录结构应如下：

```
StreamSpatial-VLM/
├── models_cache/                          # 模型权重（自动下载）
│   └── models--Qwen--Qwen2.5-VL-7B-Instruct/
│   └── models--facebook--vggt-1b/
│
├── data/raw/                              # 原始数据集
│   ├── spar7m/                            # SPAR-7M
│   │   ├── annotations/
│   │   │   ├── val.json                   # 验证集标注
│   │   │   └── train.json                 # 训练集标注（可选）
│   │   ├── scene_*/                       # 场景数据
│   │   │   ├── frame_*.jpg                # RGB 帧
│   │   │   ├── depth_pred/                # 预处理后的深度图 (.npy)
│   │   │   └── pose_conf/                 # 预处理后的位姿置信度 (.npy)
│   │   └── ...
│   │
│   ├── scanqa/                            # ScanQA
│   │   ├── ScanQA_v1.0_val.json           # 验证集标注
│   │   ├── ScanQA_v1.0_train.json         # 训练集标注（可选）
│   │   └── scans/
│   │       └── scene_*/
│   │           ├── color/frame_*.jpg
│   │           ├── depth_pred/
│   │           └── pose_conf/
│   │
│   └── scanrefer/                         # ScanRefer
│       ├── ScanRefer_filtered_val.json     # 验证集标注
│       ├── ScanRefer_filtered_train.json   # 训练集标注（可选）
│       └── scans/
│           └── scene_*/
│               ├── color/frame_*.jpg
│               ├── depth_pred/
│               └── pose_conf/
│
├── configs/                               # 实验配置
├── models/                                # 核心模块代码
├── eval/                                  # 评估脚本
├── scripts/                               # 运行脚本
├── profiling/                             # Profiling 工具
├── demo/                                  # Demo 系统
└── results/                               # 实验结果输出
```

### 数据集获取方式

| 数据集 | 获取方式 | 预估大小 |
|--------|---------|---------|
| **SPAR-7M** | HuggingFace: `jasonzhango/SPAR-7M` 或论文项目页 | ~30GB |
| **ScanQA** | [ScanQA GitHub](https://github.com/ScanQA/ScanQA) | ~5GB |
| **ScanRefer** | [ScanRefer 官网](https://daveredrum.github.io/ScanRefer/)（需邮件申请） | ~3GB |
| **ScanNet** | [ScanNet 官网](https://github.com/ScanNet/ScanNet)（需邮件申请 TOS） | ~50GB |

### 快速下载（使用项目脚本）

```bash
cd StreamSpatial-VLM

# 设置 HF Token
export HF_TOKEN="your_hf_token_here"

# 下载模型权重（Qwen2.5-VL-7B + VGGT-1B）
bash scripts/download_resources.sh --confirm

# 下载 SPAR-7M 数据集（如已在 HuggingFace 托管）
python download_datasets.py --dataset spar7m
```

---

## 实验流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    完整实验流程                               │
│                                                             │
│  Step 1: 数据预处理（深度图 + 位姿置信度图）                  │
│     ↓                                                       │
│  Step 2: 基线 Profiling（冗余分析报告）                       │
│     ↓                                                       │
│  Step 3: 基线评估（VG-LLM 全量推理）                         │
│     ↓                                                       │
│  Step 4: 消融实验 A~E（5 组，共 24 个子实验）                 │
│     ↓                                                       │
│  Step 5: 完整框架评估（SPAR-7M + ScanQA + ScanRefer）        │
│     ↓                                                       │
│  Step 6: 可视化与定性分析                                    │
│     ↓                                                       │
│  Step 7: 流式推理 Demo                                       │
└─────────────────────────────────────────────────────────────┘
```

> **预计总耗时**：在 RTX 3060 12GB 上，全量实验约需 3~5 天（含预处理）。
> 建议使用 `nohup` 或 `tmux` 在后台运行长时间任务。

---

## Step 1：数据预处理

在运行任何评估之前，需要为每个数据集生成**深度图**和**位姿置信度图**。

### 1.1 生成深度图（Depth-Anything-V2）

```bash
cd StreamSpatial-VLM

# SPAR-7M
python data/preprocess_depth.py \
    --input_dir data/raw/spar7m \
    --output_dir data/raw/spar7m/depth_pred \
    --model_size vitl \
    --batch_size 8

# ScanQA
python data/preprocess_depth.py \
    --input_dir data/raw/scanqa/scans \
    --output_dir data/raw/scanqa/scans/depth_pred \
    --model_size vitl \
    --batch_size 8

# ScanRefer
python data/preprocess_depth.py \
    --input_dir data/raw/scanrefer/scans \
    --output_dir data/raw/scanrefer/scans/depth_pred \
    --model_size vitl \
    --batch_size 8
```

### 1.2 生成位姿置信度图（VGGT-1B）

```bash
cd StreamSpatial-VLM

# SPAR-7M
python data/preprocess_pose.py \
    --input_dir data/raw/spar7m \
    --output_dir data/raw/spar7m/pose_conf \
    --batch_size 4

# ScanQA
python data/preprocess_pose.py \
    --input_dir data/raw/scanqa/scans \
    --output_dir data/raw/scanqa/scans/pose_conf \
    --batch_size 4

# ScanRefer
python data/preprocess_pose.py \
    --input_dir data/raw/scanrefer/scans \
    --output_dir data/raw/scanrefer/scans/pose_conf \
    --batch_size 4
```

### 1.3 验证预处理结果

```bash
# 检查文件数量是否匹配
find data/raw/spar7m/depth_pred -name "*.npy" | wc -l
find data/raw/spar7m/pose_conf -name "*.npy" | wc -l
```

---

## Step 2：基线 Profiling（冗余分析）

**目的**：定量刻画 VG-LLM 的计算冗余，为优化提供数据驱动依据。

```bash
cd StreamSpatial-VLM

python profiling/profile_vgllm.py \
    --data_root data/raw/spar7m \
    --num_videos 50 \
    --output_dir results/profiling
```

**输出内容**：

| 分析维度 | 输出文件 | 说明 |
|---------|---------|------|
| 帧级冗余率 | `frame_redundancy.json` | 相邻帧 CLS token 余弦相似度分布 |
| Token 级冗余率 | `token_redundancy.json` | 低注意力权重 patch 比例 |
| 3D 网络耗时 | `timing_breakdown.json` | VGGT 占总推理时间比例 |
| 显存分配 | `memory_analysis.json` | KV-Cache 占显存比例 |
| 可视化图表 | `profiling_report.pdf` | 综合分析报告 |

**预期结论**：
- >70% 帧相似度 > 0.85（帧级冗余高）
- >60% token 注意力权重极低（token 级冗余高）
- VGGT 占总推理时间 >40%
- KV-Cache 占显存 >50%

---

## Step 3：基线评估（VG-LLM 全量推理）

**目的**：建立精度上界和速度下界，作为后续对比基准。

```bash
cd StreamSpatial-VLM

# 使用基线配置（关闭所有加速模块）
bash scripts/run_eval.sh \
    --config configs/baseline_vgllm.yaml \
    --dataset all \
    --data_root data/raw \
    --output_dir results/baseline \
    --max_samples -1
```

> `--max_samples -1` 表示使用全部验证集样本。快速验证时可设为 `--max_samples 200`。

**单独运行某个数据集**：

```bash
# 仅 SPAR-7M
bash scripts/run_eval.sh --config configs/baseline_vgllm.yaml --dataset spar7m \
    --data_root data/raw --output_dir results/baseline --max_samples -1

# 仅 ScanQA
bash scripts/run_eval.sh --config configs/baseline_vgllm.yaml --dataset scanqa \
    --data_root data/raw --output_dir results/baseline --max_samples -1

# 仅 ScanRefer
bash scripts/run_eval.sh --config configs/baseline_vgllm.yaml --dataset scanrefer \
    --data_root data/raw --output_dir results/baseline --max_samples -1
```

**后台运行（推荐）**：

```bash
nohup bash scripts/run_eval.sh \
    --config configs/baseline_vgllm.yaml \
    --dataset all \
    --data_root data/raw \
    --output_dir results/baseline \
    --max_samples -1 \
    > results/baseline/run.log 2>&1 &

echo $! > results/baseline/run.pid
```

---

## Step 4：消融实验 A~E

### 消融实验概览

| 组别 | 名称 | 子实验数 | 配置文件 | 关键变量 |
|------|------|---------|---------|---------|
| **A** | 模块贡献度 | 5 | `ablation_A.yaml` | Gate / Zip / KV 开关 |
| **B** | 门控阈值 $\tau$ | 6 | `ablation_B.yaml` | $\tau \in [0.05, 0.30]$ |
| **C** | Token 保留率 $r$ | 4 | `ablation_C.yaml` | $r \in \{0.25, 0.50, 0.75, 1.0\}$ |
| **D** | KV 窗口 $w$ | 4 | `ablation_D.yaml` | $w \in \{4, 8, 16, 32\}$ |
| **E** | 重要性权重 $\alpha/\beta$ | 5 | `ablation_E.yaml` | 5 种权重组合 |

### 4.1 快速验证（推荐先跑）

先用少量样本验证流程是否通畅：

```bash
cd StreamSpatial-VLM

# 快速模式：每组仅跑 200 个样本
bash scripts/run_ablation.sh --group all --max_samples 200
```

### 4.2 运行全部消融实验

```bash
cd StreamSpatial-VLM

# 后台运行全部 5 组消融（全量样本）
nohup bash scripts/run_ablation.sh \
    --group all \
    --data_root data/raw \
    --output_dir results/ablation \
    --max_samples -1 \
    > results/ablation/run_all.log 2>&1 &

echo $! > results/ablation/run_all.pid
```

### 4.3 运行单组消融

```bash
# 仅运行消融组 A（模块贡献度）
bash scripts/run_ablation.sh --group A --max_samples -1

# 仅运行消融组 B（门控阈值）
bash scripts/run_ablation.sh --group B --max_samples -1

# 仅运行消融组 C（Token 保留率）
bash scripts/run_ablation.sh --group C --max_samples -1

# 仅运行消融组 D（KV 窗口）
bash scripts/run_ablation.sh --group D --max_samples -1

# 仅运行消融组 E（重要性权重）
bash scripts/run_ablation.sh --group E --max_samples -1
```

### 4.4 使用 Python 脚本直接运行

```bash
cd StreamSpatial-VLM
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/run_ablation.py \
    --data_root_spar7m data/raw/spar7m \
    --data_root_scanqa data/raw/scanqa \
    --output_dir results/ablation \
    --ablation A,B,C \
    --max_samples -1 \
    --device cuda
```

### 4.5 查看消融结果

```bash
# Markdown 格式表格
cat results/ablation/ablation_tables.md

# JSON 原始结果
cat results/ablation/ablation_results.json | python -m json.tool
```

### 4.6 消融组 A 详细说明（模块贡献度）

| 配置 | Gate | Zip | KV | 说明 |
|------|------|-----|-----|------|
| A-0 | ✗ | ✗ | ✗ | VG-LLM 全量基线 |
| A-1 | ✓ | ✗ | ✗ | 仅开启 2D→3D 门控 |
| A-2 | ✗ | ✓ | ✗ | 仅开启 3D→2D 压缩 |
| A-3 | ✗ | ✗ | ✓ | 仅开启增量 KV Cache |
| A-4 | ✓ | ✓ | ✓ | **完整框架（Ours）** |

---

## Step 5：完整框架评估

**目的**：使用最优配置在所有数据集上进行完整评估，生成最终对比表格。

```bash
cd StreamSpatial-VLM

# 使用默认配置（Gate + Zip + KV 全开）
bash scripts/run_eval.sh \
    --config configs/streamspatial_default.yaml \
    --dataset all \
    --data_root data/raw \
    --output_dir results/full_eval \
    --max_samples -1
```

**后台运行**：

```bash
nohup bash scripts/run_eval.sh \
    --config configs/streamspatial_default.yaml \
    --dataset all \
    --data_root data/raw \
    --output_dir results/full_eval \
    --max_samples -1 \
    > results/full_eval/run.log 2>&1 &

echo $! > results/full_eval/run.pid
```

### 生成最终对比表格

评估完成后，将基线结果和完整框架结果汇总：

| 方法 | SPAR-7M Acc | ScanQA EM | ScanRefer Acc@0.25 | FPS | 峰值显存 |
|------|-------------|-----------|---------------------|-----|----------|
| Qwen2.5-VL（无3D） | `results/baseline/...` | - | - | - | - |
| VG-LLM（全量） | `results/baseline/...` | `results/baseline/...` | `results/baseline/...` | - | - |
| **StreamSpatial-VLM** | `results/full_eval/...` | `results/full_eval/...` | `results/full_eval/...` | - | - |

---

## Step 6：可视化与定性分析

### 6.1 门控触发热力图

展示不同场景下 Gate 的触发频率与时序分布：

```bash
cd StreamSpatial-VLM
python utils/visualizer.py \
    --mode gate_heatmap \
    --input_dir results/full_eval \
    --output_dir results/visualizations
```

### 6.2 重要性图可视化

对比深度方差图、位姿不确定性图与最终保留 token 分布：

```bash
python utils/visualizer.py \
    --mode importance_map \
    --input_dir results/full_eval \
    --output_dir results/visualizations
```

### 6.3 流式推理时序图

展示逐帧推理延迟与显存占用曲线：

```bash
python utils/visualizer.py \
    --mode timing_chart \
    --input_dir results/full_eval \
    --output_dir results/visualizations
```

### 6.4 错误案例分析

```bash
python utils/visualizer.py \
    --mode error_analysis \
    --input_dir results/full_eval \
    --output_dir results/visualizations
```

---

## Step 7：流式推理 Demo

**目的**：准备答辩现场演示的实时流式空间理解系统。

```bash
cd StreamSpatial-VLM

# 基本用法
python demo/streaming_demo.py \
    --config configs/streamspatial_default.yaml \
    --video path/to/your/video.mp4 \
    --query "场景中有哪些物体？它们的空间关系如何？" \
    --output_dir results/demo \
    --save_viz

# 使用 shell 脚本
bash scripts/run_inference.sh \
    --video path/to/your/video.mp4 \
    --query "桌子上有什么？" \
    --max_frames 32
```

**Demo 输出**：
- 每帧的门控状态（是否触发 3D）
- Token 压缩率
- 逐帧推理延迟
- 最终空间问答结果
- 可视化图像（门控热力图 + 重要性图）

---

## 实验结果汇总

所有实验结果统一存放在 `results/` 目录下：

```
results/
├── profiling/                    # Step 2: 冗余分析报告
│   ├── frame_redundancy.json
│   ├── token_redundancy.json
│   ├── timing_breakdown.json
│   └── profiling_report.pdf
│
├── baseline/                     # Step 3: 基线评估结果
│   ├── spar7m/
│   ├── scanqa/
│   └── scanrefer/
│
├── ablation/                     # Step 4: 消融实验结果
│   ├── ablation_results.json     # 所有消融的汇总 JSON
│   ├── ablation_tables.md        # Markdown 格式对比表
│   └── run_all.log               # 运行日志
│
├── full_eval/                    # Step 5: 完整框架评估
│   ├── spar7m/
│   ├── scanqa/
│   └── scanrefer/
│
├── visualizations/               # Step 6: 可视化图表
│
└── demo/                         # Step 7: Demo 输出
```

---

## 常见问题

### Q1: CUDA Out of Memory

```bash
# 方案 1：减小 batch_size
# 在配置文件中修改 inference.batch_size: 1

# 方案 2：使用更小的图像分辨率
# 在配置文件中修改 data.image_size: [336, 336]

# 方案 3：启用 gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Q2: 模型下载失败

```bash
# 检查 HF Token 是否有效
huggingface-cli login --token $HF_TOKEN

# 手动指定本地缓存路径
export HF_HOME="path/to/models_cache"

# 使用镜像源（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: 数据集路径不匹配

确保 `--data_root` 参数指向包含数据集子目录的父目录。例如：
- `--data_root data/raw`（包含 `spar7m/`、`scanqa/`、`scanrefer/`）
- 而不是 `--data_root data/raw/spar7m`

### Q4: 如何恢复中断的实验

```bash
# 检查 PID 是否还在运行
kill -0 $(cat results/ablation/run_all.pid) 2>/dev/null && echo "运行中" || echo "已停止"

# 查看日志尾部
tail -100 results/ablation/run_all.log

# 重新运行（从断点继续，需脚本支持）
bash scripts/run_ablation.sh --group A --max_samples -1
```

### Q5: 如何调整实验参数

编辑对应的 YAML 配置文件：

```bash
# 修改默认配置
vim configs/streamspatial_default.yaml

# 修改消融配置
vim configs/ablation/ablation_B.yaml
```

关键参数速查：

| 参数 | 路径 | 说明 |
|------|------|------|
| `gate.tau` | `configs/streamspatial_default.yaml` | 门控阈值，越小越频繁触发 3D |
| `zip.keep_ratio` | `configs/streamspatial_default.yaml` | Token 保留率，0.25~1.0 |
| `zip.alpha` / `zip.beta` | `configs/streamspatial_default.yaml` | 深度方差 / 位姿熵权重 |
| `kv_cache.window_size` | `configs/streamspatial_default.yaml` | KV 滑动窗口帧数 |
| `data.max_frames` | `configs/streamspatial_default.yaml` | 每个视频最大帧数 |
| `inference.max_new_tokens` | `configs/streamspatial_default.yaml` | LLM 最大生成长度 |

---

## 许可证

本项目仅用于学术研究。数据集请遵循各自官方许可协议。
