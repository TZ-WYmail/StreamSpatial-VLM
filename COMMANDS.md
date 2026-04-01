# StreamSpatial-VLM 命令执行指南

快速参考：所有 Phase 的可执行命令

---

## Phase 0: 基础设施与数据准备（前置依赖）

### Step 0.1: 5090 环境适配

```bash
# 设置环境变量
export DS_DISABLE_COMM_OP_LIBS=1

# 运行环境配置脚本
chmod +x setup.sh
./setup.sh
```

### Step 0.2: 数据集与权重获取

#### SPAR-7M 下载

```bash
# 方式1: 使用HuggingFace CLI
pip install huggingface_hub
huggingface-cli download --repo-type dataset SPAR7M/SPAR-7M --local-dir data/raw/spar7m

# 方式2: 使用Python脚本（SPAR-7M 精简版）
python download_datasets.py --dataset spar7m

# 方式3: 下载SPAR-7M-RGBD版本
python download_datasets.py --dataset spar7m-rgbd

# 解压缩各个数据集
cd data/raw/spar7m
tar -xvzf scannet.tar.gz
tar -xvzf scannetpp.tar.gz
tar -xvzf structured3d.tar.gz
tar -xvzf rxr.tar.gz
```

#### ScanQA 下载

```bash
# 从官方GitHub获取
git clone https://github.com/ATR-DBI/ScanQA.git
mv ScanQA/data/* data/raw/scanqa/
```

#### 基线权重下载

**方式1: 使用HuggingFace CLI**
```bash
huggingface-cli download VG-LLM/VG-LLM-7B --local-dir checkpoints/VG-LLM/
```

**方式2: 使用Python脚本**
```bash
python download_weights.py --model vgllm-8b --local-dir checkpoints/VG-LLM --yes
```

**查看支持的模型列表**
```bash
python download_weights.py --list
```

### Step 0.3: 离线 3D 几何先验生成

#### 使用 prepare_scannet_layout.py 准备 SPAR 布局

```bash
# 基础用法（仅复制RGB图像）
python3 data/raw/spar7m/prepare_scannet_layout.py \
    --scannet-root /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/data/ScanNet \
    --out-root /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/data/raw/spar7m/spar/scannet/images

# 完整用法（复制RGB+深度图像+相机参数）
python3 data/raw/spar7m/prepare_scannet_layout.py \
    --scannet-root /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/data/ScanNet \
    --out-root /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/data/raw/spar7m/spar/scannet/images \
    --copy \
    --rgbd \
    --scenes-list scenes.txt

# 参数说明
# --scannet-root: ScanNet数据根目录（指向scans/父目录）
# --out-root: 输出SPAR格式的目录
# --copy: 复制而不是符号链接
# --rgbd: 同时复制深度图像、相机内参、位姿信息
# --scenes-list: 指定场景列表文件（可选）
```

#### Depth-Anything-V2 推理

```bash
python scripts/generate_depth.py \
    --input data/raw/spar7m/videos \
    --output data/processed/spar7m/depth_pred \
    --device cuda:0
```

#### VGGT 位姿估计

```bash
python scripts/generate_pose.py \
    --input data/raw/spar7m/videos \
    --output data/processed/spar7m/pose_conf \
    --device cuda:0
```

---

## Phase 1: 基线复现与冗余分析（动机论证）

### Step 1.1: VG-LLM 基线 Profiling

```bash
# 运行profiling（在Python脚本中调用）
python -c "
from profiling.profile_vgllm import VGLLMProfiler

profiler = VGLLMProfiler(
    model_path='checkpoints/VG-LLM',
    device='cuda'
)

results = profiler.profile_video(
    video_path='data/raw/spar7m/videos/scene_0001',
    question='What is the spatial relationship between the chair and the table?'
)

print(f'端到端耗时: {results[\"total_time\"]:.2f}s')
print(f'VGGT耗时: {results[\"vggt_time\"]:.2f}s')
print(f'VGGT占比: {results[\"vggt_ratio\"]:.1%}')
print(f'峰值显存: {results[\"peak_memory\"]:.2f}GB')
print(f'每帧Token数: {results[\"tokens_per_frame\"]}')
"
```

---

## Phase 2: 核心模块开发（创新点实现）

本阶段主要是代码开发，无新增命令行执行。

模块位置：
- Step 2.1: `models/gate_2d3d.py` - 2D→3D 语义门控
- Step 2.2: `models/zip_3d2d.py` - 3D→2D 几何引导压缩
- Step 2.3: `models/kv_cache.py` - 增量流式缓存

---

## Phase 3: 系统集成（组装 StreamSpatial-VLM）

### Step 3.1: StreamSpatial-VLM 主模型

主模型文件：`models/stream_spatial_vlm.py`

### Step 3.2: 数据加载测试

```bash
# 测试SPAR-7M数据加载
python -c "
from data.dataset_loaders.spar7m import SPAR7MDataset

spar7m = SPAR7MDataset(
    data_root='data/raw/spar7m',
    processed_root='data/processed/spar7m',
    split='val',
    load_depth=True,
    load_pose=True
)

sample = spar7m[0]
print(f'Images: {sample[\"images\"].shape}')
print(f'Depth maps: {sample[\"depth_maps\"].shape}')
print(f'Poses: {sample[\"poses\"].shape}')
"
```

---

## Phase 4: 实验验证与消融（证明有效）

### Step 4.1: 主实验（对比基线）

#### SPAR-7M 评估

```bash
python eval/eval_spar7m.py \
    --config configs/streamspatial_default.yaml \
    --data_root data/raw/spar7m \
    --processed_root data/processed/spar7m \
    --output results/spar7m_results.json
```

#### ScanQA 评估

```bash
python eval/eval_scanqa.py \
    --config configs/streamspatial_default.yaml \
    --data_root data/raw/scanqa \
    --output results/scanqa_results.json
```

#### ScanRefer 评估

```bash
python eval/eval_scanrefer.py \
    --config configs/streamspatial_default.yaml \
    --data_root data/raw/scanrefer \
    --output results/scanrefer_results.json
```

### Step 4.2: 消融实验（证明每个模块都有用）

```bash
# 运行所有消融实验
python scripts/run_ablation.py \
    --config_dir configs/ablation/ \
    --data_root data/raw/spar7m \
    --output results/ablation_results.json

# 或使用shell脚本
bash scripts/run_ablation.sh
```

### Step 4.3: 敏感性分析（超参调优）

#### Gate 阈值 τ 曲线图

```bash
python scripts/sensitivity_analysis.py \
    --param gate_threshold \
    --range 0.90 0.99 0.01 \
    --output results/gate_sensitivity.png
```

#### Zip 保留率 r 曲线图

```bash
python scripts/sensitivity_analysis.py \
    --param zip_ratio \
    --range 0.3 1.0 0.1 \
    --output results/zip_sensitivity.png
```

---

## Phase 5: 成果产出（答辩准备）

### Step 5.1: 实时流式 Demo

```bash
# 启动Gradio演示应用
python demo/streaming_demo.py \
    --config configs/streamspatial_default.yaml \
    --port 7860
```

访问地址：`http://localhost:7860`

### Step 5.2: 项目文档

依赖项：
```bash
pip install -r requirements.txt
```

---

## 快速启动命令速查表

```bash
# 1. 环境配置
./setup.sh

# 2. 下载数据集
python download_datasets.py --dataset spar7m-rgbd --yes

# 3. 下载基线权重
python download_weights.py --model vgllm-8b --yes

# 4. 准备SPAR布局
python3 data/raw/spar7m/prepare_scannet_layout.py \
    --scannet-root /path/to/ScanNet \
    --out-root /path/to/output \
    --copy --rgbd

# 5. 生成伪深度
python scripts/generate_depth.py --input data/raw/spar7m/videos --output data/processed/spar7m/depth_pred

# 6. 生成伪位姿
python scripts/generate_pose.py --input data/raw/spar7m/videos --output data/processed/spar7m/pose_conf

# 7. 运行消融实验
python scripts/run_ablation.py --config_dir configs/ablation/ --data_root data/raw/spar7m

# 8. 评估主实验
python eval/eval_spar7m.py --config configs/streamspatial_default.yaml --data_root data/raw/spar7m

# 9. 启动Demo
python demo/streaming_demo.py --config configs/streamspatial_default.yaml --port 7860
```

---

## 常见命令模式

### 模型相关
```bash
# 列出所有支持的模型
python download_weights.py --list

# 下载特定模型
python download_weights.py --model <MODEL_NAME> --yes --verify-all

# 使用本地镜像加速（中国用户）
python download_weights.py --model vgllm-8b --no-mirror
```

### 数据相关
```bash
# 验证数据集完整性
python -c "from data.dataset_loaders import SPAR7MDataset; ds = SPAR7MDataset('data/raw/spar7m'); print(len(ds))"

# 检查特定样本
python -c "from data.dataset_loaders import SPAR7MDataset; ds = SPAR7MDataset('data/raw/spar7m'); print(ds[0].keys())"
```

### 评估相关
```bash
# 生成评估报告
bash scripts/run_eval.sh

# 查看评估结果
cat results/eval_results.json

# 生成对比表格
python utils/generate_report.py --results results/ablation_results.json --output results/table.md
```

### 推理相关
```bash
# 快速推理测试
python scripts/run_inference.sh --config configs/streamspatial_default.yaml --input test_video.mp4 --output result.json
```

---

## 环境变量设置

```bash
# RTX 5090 专用
export DS_DISABLE_COMM_OP_LIBS=1

# HuggingFace 加速（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 指定GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch优化
export CUDA_LAUNCH_BLOCKING=1  # 调试用
export TORCH_CUDNN_BENCHMARK=1  # 性能优化
```

---

**最后更新**：2026年4月  
**维护者**：谭政
