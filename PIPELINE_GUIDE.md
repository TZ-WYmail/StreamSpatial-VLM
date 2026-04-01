# StreamSpatial-VLM 完整数据处理流程指南

## 📋 概览

本文档介绍如何后台启动 SPAR-7M-RGBD 完整数据下载与预处理，预计耗时 **24-48 小时**。

### 🎯 整体流程

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: 环境检查 (5分钟)                              │
│  - Conda 环境验证                                       │
│  - 脚本文件检查                                         │
│  - 磁盘空间检查 (需要 >250GB)                            │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  Phase 2: 下载 SPAR-7M-RGBD (24-48小时)                  │
│  - 从 HuggingFace 下载 200GB+ 数据                      │
│  - 支持断点续传                                         │
│  包含:                                                 │
│    • ScanNet (11K RGB + 5.5K 深度)                     │
│    • ScanNetPP (50K RGB)                              │
│    • Structured3D (100K RGB)                          │
│    • RXR (200K+ RGB)                                  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  Phase 3: 生成深度图 (12-24小时)                         │
│  使用: Depth-Anything-V2-ViT-L (无监督)                 │
│  为每个数据集生成伪深度图 (.npy 格式)                    │
│    • ScanNet: 5,578 个深度图                           │
│    • ScanNetPP: ~50K 个深度图                          │
│    • Structured3D: ~100K 个深度图                      │
│    • RXR: ~200K+ 个深度图                              │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  Phase 4: 生成位姿置信度 (6-12小时)                      │
│  使用: VGGT-1B (无监督)                                 │
│  为每个数据集生成置信度图 (.npy 格式)                    │
│    • ScanNet: 5,578 个置信度图                         │
│    • ScanNetPP: ~50K 个置信度图                        │
│    • Structured3D: ~100K 个置信度图                    │
│    • RXR: ~200K+ 个置信度图                            │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  Phase 5: 验证 (10分钟)                                 │
│  - 检查数据完整性                                       │
│  - 统计各数据集完成度                                   │
│  - 输出最终报告                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 快速启动

### 方式 1: 使用 Python 脚本（推荐）

**完整流程（后台运行）**:
```bash
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM

# 后台启动（返回前台，进程继续）
nohup conda run -n streamspatial python scripts/launch_pipeline.py &

# 查看进程
jobs -l

# 实时监控
python scripts/launch_pipeline.py --monitor

# 检查状态
python scripts/launch_pipeline.py --mode check
```

**仅下载**:
```bash
python scripts/launch_pipeline.py --mode download
```

**仅预处理**:
```bash
python scripts/launch_pipeline.py --mode preprocess
```

### 方式 2: 使用 Bash 脚本

**完整流程**:
```bash
bash scripts/pipeline_download_and_process.sh
```

**带监控面板**:
```bash
bash scripts/pipeline_download_and_process.sh --monitor
```

**检查状态**:
```bash
bash scripts/pipeline_download_and_process.sh --check-only
```

---

## 📊 数据流与文件结构

### 输入数据结构（下载后）

```
data/raw/spar7m/
├── scannet/                           # ScanNet 部分
│   ├── images/
│   │   ├── scene_*/                   # 多个场景
│   │   │   ├── *.jpg                  # RGB 帧
│   │   │   └── ...
│   │   └── ...
│   └── qa_jsonl/                      # QA 标注
│
├── scannetpp/                         # ScanNetPP 部分
│   ├── images/
│   └── qa_jsonl/
│
├── structured3d/                      # Structured3D 部分
│   ├── images/
│   └── qa_jsonl/
│
└── rxr/                               # RXR 部分
    ├── images/
    └── qa_jsonl/
```

### 处理后的数据结构

```
data/raw/spar7m/
├── scannet/
│   ├── images/                        # 原始 RGB
│   ├── depth_pred/                    # ✨ 生成的深度图 (Phase 3)
│   │   ├── scene_0001/
│   │   │   ├── *.npy                  # 深度图（numpy 格式）
│   │   │   └── ...
│   │   └── ...
│   ├── pose_conf/                     # ✨ 生成的位姿置信度 (Phase 4)
│   │   ├── scene_0001/
│   │   │   ├── *.npy                  # 置信度图（numpy 格式）
│   │   │   └── ...
│   │   └── ...
│   └── qa_jsonl/
│
├── scannetpp/
│   ├── images/
│   ├── depth_pred/
│   ├── pose_conf/
│   └── qa_jsonl/
│
├── structured3d/
│   ├── images/
│   ├── depth_pred/
│   ├── pose_conf/
│   └── qa_jsonl/
│
└── rxr/
    ├── images/
    ├── depth_pred/
    ├── pose_conf/
    └── qa_jsonl/
```

---

## 📝 详细步骤说明

### Phase 1: 环境检查 (自动，<1分钟)

脚本会自动检查：
- ✓ Conda 环境 `streamspatial` 是否存在
- ✓ 项目目录结构是否完整
- ✓ 所需脚本文件是否都存在
- ✓ 磁盘是否有 >250GB 空间

**若检查失败**，按错误信息修复：
```bash
# 重新创建 Conda 环境
conda create -n streamspatial python=3.10 -y

# 安装依赖
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM
pip install -r requirements.txt
```

### Phase 2: 下载 SPAR-7M-RGBD (24-48小时)

**实际执行的命令**:
```bash
conda run -n streamspatial python download_datasets.py \
    --dataset spar7m-rgbd \
    --data-dir data/raw
```

**下载的数据**:
| 数据集 | 大小 | 项目 |
|--------|------|------|
| ScanNet | ~2GB | RGB(5.5K) + 深度 (已有) |
| ScanNetPP | ~30GB | RGB(50K) |
| Structured3D | ~40GB | RGB(100K) |
| RXR | ~130GB | RGB(200K+) |
| 总计 | ~200GB | - |

**下载日志**:
```bash
tail -f results/logs/download.log
```

### Phase 3: 生成深度图 (12-24小时)

**实际执行的命令**（为每个数据集）:
```bash
conda run -n streamspatial python data/preprocess_depth.py \
    --input_dir data/raw/spar7m/scannet/images \
    --output_dir data/raw/spar7m/scannet/depth_pred \
    --model_size vitl \
    --batch_size 8
```

**工作原理**:
1. 读取 RGB 图像
2. 使用 Depth-Anything-V2-ViT-L 推理
3. 输出为 .npy 格式（节省空间）

**性能指标** (RTX 5090):
| 数据集 | 图像数 | 耗时 |
|--------|--------|------|
| ScanNet | 5.5K | ~3h |
| ScanNetPP | 50K | ~6h |
| Structured3D | 100K | ~12h |
| RXR | 200K+ | ~24h |

**深度图日志**:
```bash
tail -f results/logs/depth_scannet.log
tail -f results/logs/depth_scannetpp.log
# ...
```

### Phase 4: 生成位姿置信度 (6-12小时)

**实际执行的命令**（为每个数据集）:
```bash
conda run -n streamspatial python data/preprocess_pose.py \
    --input_dir data/raw/spar7m/scannet/images \
    --output_dir data/raw/spar7m/scannet/pose_conf \
    --batch_size 4 \
    --img_size 518
```

**工作原理**:
1. 读取 RGB 图像
2. 使用 VGGT-1B 推理（从深度梯度推断置信度）
3. 输出为 .npy 格式

**性能指标** (RTX 5090):
| 数据集 | 图像数 | 耗时 |
|--------|--------|------|
| ScanNet | 5.5K | ~1.5h |
| ScanNetPP | 50K | ~3h |
| Structured3D | 100K | ~6h |
| RXR | 200K+ | ~12h |

### Phase 5: 验证 (自动，<1分钟)

脚本会自动输出：
```
==================== 数据集状态 ====================

📦 scannet
   ├─ 图像: 11156 个
   ├─ 深度图: ✅
   ├─ 位姿: ✅
   └─ 总大小: 3.5 GB

📦 scannetpp
   ├─ 图像: 50000 个
   ├─ 深度图: ✅
   ├─ 位姿: ✅
   └─ 总大小: 150.2 GB

...

==================== 磁盘使用 ====================
已用: 220.5 GB / 总计: 1000.0 GB (使用率: 22%)
可用: 779.5 GB
```

---

## 📊 监控与日志

### 实时监控

```bash
# Python 监控面板（自动刷新）
python scripts/launch_pipeline.py --monitor
```

输出示例:
```
==================================================
StreamSpatial-VLM 数据处理监控
时间: 2026-04-01 12:34:56
==================================================

📥 下载: 运行中
🔍 深度处理: 运行中
📍 位姿处理: 空闲

最近日志:
[INFO] [2026-04-01 12:34:56] [DEPTH] 为 scannet 生成深度图...
[SUCCESS] [2026-04-01 12:34:10] [DEPTH] scannet 深度图生成完成 (100/100)
...

按 Ctrl+C 退出监控
```

### 日志文件

所有日志保存在 `results/logs/`:

| 日志文件 | 内容 |
|---------|------|
| `pipeline.log` | 主流程日志 |
| `download.log` | 下载详情 |
| `depth_scannet.log` | ScanNet 深度生成 |
| `depth_scannetpp.log` | ScanNetPP 深度生成 |
| `pose_scannet.log` | ScanNet 位姿生成 |
| `pose_scannetpp.log` | ScanNetPP 位姿生成 |
| `pipeline_state.json` | 处理状态（JSON） |

### 查看日志

```bash
# 查看主日志
cat results/logs/pipeline.log | tail -100

# 实时监控下载
tail -f results/logs/download.log

# 监控深度生成
tail -f results/logs/depth_scannet.log

# 对比时间戳
grep "SUCCESS" results/logs/pipeline.log
```

---

## ⚙️ 高级用法

### 断点续传

数据下载可能中断，使用断点续传恢复：

```bash
# 自动检测并恢复
python scripts/launch_pipeline.py --mode download

# 或手动指定
export RESUME_DOWNLOAD=1
python scripts/launch_pipeline.py --mode download
```

### 指定数据集

仅处理特定数据集：

```bash
# 仅处理 ScanNet
python scripts/launch_pipeline.py --datasets scannet

# 仅处理 ScanNetPP 和 RXR
python scripts/launch_pipeline.py --datasets scannetpp

python scripts/launch_pipeline.py --datasets rxr
```

### 自定义批处理大小

调整 GPU 内存使用：

```bash
# 增大批处理（需要 24GB+ VRAM）
# 编辑 scripts/launch_pipeline.py 中的:
# --batch_size 16

# 减小批处理（节省内存）
# --batch_size 4
```

### 仅预处理已下载的数据

```bash
# 跳过下载，仅预处理
python scripts/launch_pipeline.py --mode preprocess
```

---

## 🔧 故障排除

### 问题 1: "磁盘空间不足"

```bash
# 检查磁盘
df -h

# 清空垃圾（如必要）
rm -rf ~/Downloads/*
```

### 问题 2: 下载中断

```bash
# 自动续传
python scripts/launch_pipeline.py --mode download

# 或手动清理并重试
rm -rf data/raw/spar7m  # 删除不完整数据
python scripts/launch_pipeline.py
```

### 问题 3: GPU 显存溢出

```bash
# 降低批处理大小
# 编辑 scripts/launch_pipeline.py
# --batch_size 4 -> --batch_size 2

# 或使用较小的模型
# --model_size vitl -> --model_size vitb
```

### 问题 4: 进程卡住

```bash
# 查看进程
ps aux | grep python

# 杀死特定进程
kill -9 <PID>

# 重新启动
python scripts/launch_pipeline.py
```

---

## 📈 性能预期

### 时间估算（RTX 5090）

| 阶段 | 操作 | 耗时 |
|------|------|------|
| Phase 1 | 环境检查 | 1分钟 |
| Phase 2 | 下载 200GB | 24-48 小时 |
| Phase 3 | 深度生成 | 12-24 小时 |
| Phase 4 | 位姿生成 | 6-12 小时 |
| Phase 5 | 验证 | 10 分钟 |
| **总计** | - | **42-85 小时** |

### 并行化可能性

由于各种依赖关系，目前的流程是**序列执行**：
- Phase 2 (下载) → Phase 3 (深度) → Phase 4 (位姿)

可以优化的地方：
- 下载时可以同时处理已下载的 ScanNet 数据
- 多数据集深度生成可以并行

### 磁盘空间需求

```
ScanNet:     2 GB (RGB + 深度已有)
ScanNetPP:   35 GB (RGB) + 15 GB (深度) + 15 GB (位姿) = 65 GB
Structured3D: 45 GB (RGB) + 20 GB (深度) + 20 GB (位姿) = 85 GB
RXR:          130 GB (RGB) + 50 GB (深度) + 50 GB (位姿) = 230 GB
─────────────────────────────────────────────────
总计:         ~382 GB
```

---

## ✅ 完成检查清单

完整处理后，检查以下项：

- [ ] `results/logs/pipeline.log` 显示 "PHASE 5: 验证完成"
- [ ] `data/raw/spar7m/scannet/depth_pred/` 包含 5,578 个 .npy 文件
- [ ] `data/raw/spar7m/scannet/pose_conf/` 包含 5,578 个 .npy 文件
- [ ] `data/raw/spar7m/scannetpp/depth_pred/` 包含 ~50K 个 .npy 文件
- [ ] `data/raw/spar7m/scannetpp/pose_conf/` 包含 ~50K 个 .npy 文件
- [ ] 类似地检查 Structured3D 和 RXR

**验证命令**:
```bash
# 快速检查完成度
python scripts/launch_pipeline.py --mode check

# 详细统计
find data/raw/spar7m -name "*.npy" | wc -l
```

---

## 📞 获取帮助

若遇到问题：

1. **检查日志**:
   ```bash
   tail -n 50 results/logs/pipeline.log
   ```

2. **查看特定阶段日志**:
   ```bash
   cat results/logs/download.log | grep ERROR
   cat results/logs/depth_scannet.log | grep ERROR
   ```

3. **重新启动**:
   ```bash
   # 清理并重启
   python scripts/launch_pipeline.py --mode check
   python scripts/launch_pipeline.py
   ```

---

**准备好了？开始启动流程！** 🚀

```bash
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM
nohup conda run -n streamspatial python scripts/launch_pipeline.py &
python scripts/launch_pipeline.py --monitor
```
