# 🚀 StreamSpatial-VLM 完整数据处理启动指南

> 从现在开始，到完整的 SPAR-7M-RGBD 数据集准备就绪！

---

## 📍 当前位置

✅ **已完成**:
- 基础环境配置
- VG-LLM 冗余分析 (18.4% 帧冗余)
- 三核心模块验证 (Gate/Zip/KV-Cache)
- ScanNet 子集数据准备 (11K RGB + 5.5K depth)

⏳ **正准备启动**:
- 完整 SPAR-7M-RGBD 数据下载 (200+GB)
- 全自动预处理管线 (深度图 + 位姿)

---

## 🎯 **一行命令启动**

```bash
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM && bash start_pipeline.sh
```

**就这样。剩下的一切都自动处理。**

---

## 📊 42-84 小时怎么分配

```
NOW (T=0h)
  │
  ├─────────────────────────────────── 24-48 小时 ───────────────────────────
  │  🔄 下载 SPAR-7M-RGBD (~200GB)
  │     • ScanNet        11K RGB + 5.5K depth (已有)
  │     • ScanNetPP      50K RGB images       (新)
  │     • Structured3D   100K RGB images      (新)  
  │     • RXR            200K+ RGB images     (新)
  │  
  ├─────────────────────── 8-10 小时 ──────────────── 自动开始
  │  🔷 深度图生成 (Depth-Anything-V2-vitl)
  │     • 对所有 RGB 图像生成伪深度
  │  
  ├─────────── 4-5 小时 ──────────── 并行进行
  │  🟣 位姿置信度生成 (VGGT-1B)
  │     • 计算每帧的三维空间一致性
  │  
  ├─ 1-2 小时 ┤
  │  📦 元数据整理
  │     • 生成 Parquet 统一格式
  │  
  └─ DONE ✅ (T=42-84h)
     完整的 SPAR-7M-RGBD 准备就绪！
```

---

## 🔍 三种使用方式

### 方式 1: **懒人 (推荐)**

```bash
bash /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/start_pipeline.sh
# 一句话启动，48 小时后数据准备完毕
```

### 方式 2: **想看进度**

```bash
# 终端 1: 启动处理
bash start_pipeline.sh

# 终端 2: 实时监控进度
tail -f results/logs/full_pipeline.log

# 或启动监控面板
python scripts/launch_pipeline.py --monitor
```

### 方式 3: **完全手动控制**

```bash
# 查看所有可用选项
python scripts/launch_pipeline.py --help

# 手动启动各阶段
python scripts/launch_pipeline.py --action start
python scripts/launch_pipeline.py --action status
python scripts/launch_pipeline.py --action monitor
python scripts/launch_pipeline.py --action pause
python scripts/launch_pipeline.py --action resume
```

---

## 📝 快速参考卡

| 用途 | 命令 |
|------|------|
| **启动处理** | `bash start_pipeline.sh` |
| **查看日志** | `tail -f results/logs/full_pipeline.log` |
| **check 状态** | `python scripts/launch_pipeline.py --mode check` |
| **启动监控** | `python scripts/launch_pipeline.py --monitor` |
| **停止** | `kill $(cat results/logs/pipeline.pid)` |
| **查看数据大小** | `du -sh data/raw/spar7m/spar/*/` |
| **查看生成的深度图** | `find data/raw/spar7m -name 'depth_pred' -type d` |

---

## 🎓 中间环节和故障恢复

### 如果中途断网了

**自动处理** ✓ 该脚本配置了断点续传

```bash
# 恢复时只需再运行一遍启动命令
bash start_pipeline.sh --download
```

### 如果进程卡死了

```bash
# 查看是否还在运行
ps aux | grep "python.*launch_pipeline"

# 强制停止
kill -9 $(cat results/logs/pipeline.pid)

# 从中断点恢复
bash start_pipeline.sh
```

### 如果想暂停然后继续

```bash
# 暂停
kill -STOP $(cat results/logs/pipeline.pid)

# 继续
kill -CONT $(cat results/logs/pipeline.pid)
```

---

## ✅ 验证下载完成

下载完成后，应该看到这样的文件结构：

```bash
# 预期的数据组织
data/raw/spar7m/spar/
├── scannet/
│   └── images/            # 11K RGB + 5.5K depth ✅
├── scannetpp/
│   ├── images/            # 50K RGB 
│   ├── depth_pred/        # 50K 深度图
│   └── pose_conf/         # 50K 位姿置信度
├── structured3d/
│   ├── images/            # 100K RGB
│   ├── depth_pred/        # 100K 深度图
│   └── pose_conf/         # 100K 位姿置信度
└── rxr/
    ├── images/            # 200K+ RGB
    ├── depth_pred/        # 200K+ 深度图
    └── pose_conf/         # 200K+ 位姿置信度
```

验证命令：

```bash
# 检查总大小（应该是 200+GB）
du -sh /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/data/raw/spar7m

# 检查所有子集都有数据
ls data/raw/spar7m/spar/*/images | wc -l  # 应该是 4
ls data/raw/spar7m/spar/*/depth_pred 2>/dev/null | wc -l  # 应该是 3 (scannet 已有)
```

---

## 🎯 完成后的下一步

### 1. 运行完整消融实验

```bash
python scripts/run_ablation.py \
  --data_root_spar7m data/raw/spar7m/spar \
  --output_dir results/ablation_complete \
  --group A-E \
  --max_samples -1  # 使用所有数据
```

### 2. 生成评估报告

```bash
python eval/eval_spar7m.py \
  --data_dir data/processed/spar7m \
  --output results/evaluation_report.json
```

### 3. 生成论文图表

```bash
python scripts/generate_paper_figures.py \
  --results_dir results/ablation_complete \
  --output_dir results/figures
```

---

## 💾 硬件和存储需求

| 项目 | 需求 | 当前 |
|------|------|------|
| **GPU** | RTX 3090+ | ✅ RTX 5090 |
| **VRAM** | 24GB+ | ✅ 能用 |
| **RAM** | 32GB+ | ✅ 能用 |
| **磁盘** | **250GB+** | ⚠️ **需验证** |
| **网络** | 稳定 1Mbps+ | ⚠️ 依赖ISP |
| **时间** | 42-84h | ✅ 后台自动 |

**检查磁盘空间**:

```bash
df -h /home/tanzheng/Desktop/myproject/StreamSpatial-VLM
# 需要至少 250GB 可用空间
```

---

## 📞 故障排查

### 问题 1: "Permission denied"

```bash
chmod +x /home/tanzheng/Desktop/myproject/StreamSpatial-VLM/start_pipeline.sh
bash start_pipeline.sh  # 重试
```

### 问题 2: "No space left on device"

```bash
df -h  # 检查可用空间
# 需要清理至少 250GB
```

### 问题 3: "连接超时"

```bash
# 这是网络问题，脚本会自动重试
# 检查网络连接
ping google.com

# 继续运行
bash start_pipeline.sh --download
```

### 问题 4: "进程卡死"

```bash
# 检查是否真的在运行
tail -f results/logs/full_pipeline.log

# 如果 1 小时没有输出就是卡死了
kill -9 $(cat results/logs/pipeline.pid)
bash start_pipeline.sh
```

---

## 🎬 现在立即开始

```bash
# 只需这一个命令！
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM && bash start_pipeline.sh
```

**然后去做其他事情，48 小时后回来查看！** 

---

## 📖 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 超简洁快速指南
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - 详细技术文档  
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - 项目完成度报告
- [scripts/launch_pipeline.py](scripts/launch_pipeline.py) - Python 控制器源代码

---

## 🚀 最后的话

这是一个**完全自动化**的管线。启动后：

✅ 不需要人工干预  
✅ 不需要定期检查  
✅ 自动处理断网、重试、恢复  
✅ 生成完整的日志供事后分析  

**启动它，然后忘记它。** 🎉

接下来的 42-84 小时，系统会自动为你准备完整的 SPAR-7M-RGBD 数据集！
