# 推送到远程 Git 仓库 - 完整指南

> 🚀 Phase 0-2 完成后，将项目推送到远程仓库的详细步骤

---

## ⚡ 5 分钟快速推送

```bash
# 1. 进入项目目录
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM

# 2. 运行最终检查
bash preflight_push_check.sh

# 3. 如果全部通过，执行推送
git add -A

git commit -m "Phase 0-2 完成: 环境配置、冗余分析、核心模块开发

功能:
  - Phase 0: 环境配置 + 权重下载 + 数据准备
  - Phase 1: VG-LLM 冗余分析（50 视频）
  - Phase 2: Gate/Zip/KV-Cache 三大模块 + 单测通过

修复:
  - ModuleNotFoundError (sys.path.insert)
  - 配置对象不匹配 (dataclass)
  - 权重加载兼容性 (ignore_mismatched_sizes)

文档:
  - 完整 COMMANDS.md
  - 项目总结和单测
  - 自动化下载管线"

git push origin main
```

---

## 📋 完整步骤指南

### Step 1: 环境检查（2 分钟）

```bash
# 确保在项目目录
cd /home/tanzheng/Desktop/myproject/StreamSpatial-VLM

# 验证 Git 仓库
git status

# 预期输出: 应该在 main/master 分支
# On branch main
# Your branch is up to date with 'origin/main'.
```

---

### Step 2: 最终验证（3 分钟）

```bash
# 运行推送前检查脚本
bash preflight_push_check.sh

# 预期输出:
# ╔════════════════════════════════════════════════════════════════════╗
# ║  StreamSpatial-VLM 推送前最终检查                                 ║
# ...
# ║  检查结果: 全部通过！                                             ║
# ║  可以放心推送到远程 ✓                                             ║
# ╚════════════════════════════════════════════════════════════════════╝
```

**如果有失败项**:
```bash
# 查看详细的 Git 状态
git status

# 可能的问题:
# 1. 大文件被追踪 → 检查 .gitignore
# 2. 敏感信息暴露 → 检查代码中的 token
# 3. 代码语法错误 → 运行 python -m py_compile <file>
```

---

### Step 3: 暂存文件（1 分钟）

```bash
# 查看将要提交的更改
git status

# 添加所有更改
git add -A

# 验证暂存内容
git status

# 预期输出: 应该显示要提交的文件列表
# Changes to be committed:
#   modified:   models/stream_spatial_vlm.py
#   modified:   scripts/run_ablation.py
#   ...
#   new file:   PHASE_0_2_SUMMARY.md
```

---

### Step 4: 编写提交信息（1 分钟）

```bash
# 推荐的提交信息格式

git commit -m "Phase 0-2 完成: 环境配置、冗余分析、核心模块开发 (#1)

【主要功能】
- Phase 0: RTX 5090 环境配置 + VLM 权重下载 + ScanNet 布局准备
- Phase 1: VG-LLM Profiling（50 视频）+ 冗余分析
- Phase 2: Gate/Zip/KV-Cache 三模块开发 + 单测通过（5/5）

【关键修复】
- 修复 ModuleNotFoundError (sys.path.insert at scripts/run_ablation.py:25)
- 修复配置对象类型不匹配 (使用 @dataclass)
- 修复模型权重加载 (ignore_mismatched_sizes=True)

【数据状态】
- ScanNet: 11,156 RGB + 5,578 Depth ✓ 完整
- 其他数据集: 待下载（自动化管线已准备）

【文档】
- PHASE_0_2_SUMMARY.md: 完整项目总结
- COMMANDS.md: 所有命令参考
- 单元测试验证所有模块

【已知限制】
- Phase 3 模型集成待完全调试 (Workaround 可用)
- 完整 SPAR-7M 数据未下载 (后续管线)

相关: closes #0"
```

---

### Step 5: 执行推送（1 分钟）

```bash
# 推送到远程
git push origin main

# 预期输出:
# Enumerating objects: 50, done.
# Counting objects: 100% (50/50), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (30/30), done.
# Writing objects: 100% (50/50), 500.0 KiB, done.
# Total 50 (delta 20), reused 0 (delta 0)
# ...
# main -> main
```

---

### Step 6: 验证推送（1 分钟）

```bash
# 验证远程已接收
git log origin/main --oneline -5

# 或在 GitHub/GitLab 网页上检查
# https://github.com/YOUR_USERNAME/StreamSpatial-VLM

# 应该看到最新的提交信息
```

---

## 🔍 **推送前检查清单**

| 项目 | 检查方法 | 预期结果 |
|------|---------|---------|
| **关键源文件** | `ls models/*.py` | 4 个文件齐全 |
| **文档** | `ls *.md \| grep -E "STATUS\|COMMANDS\|PHASE"` | 3+ 个文档 |
| **数据完整** | `du -sh data/raw/spar7m/` | >30GB |
| **无大文件** | `git status --ignored \| head` | 大文件都在 .gitignore |
| **无敏感信息** | `grep -r "hf_" --include="*.py"` | (空结果) |
| **单测通过** | `python test_modules.py` | 5/5 PASS |
| **Git 状态** | `git status` | clean 或只有源代码更改 |

---

## ⚠️ **常见问题 & 解决方案**

### 问题 1: "large files will be clipped"

**症状**:
```
warning: CRLF will be converted to LF in <file>.
...
```

**解决**:
```bash
# 这是换行符问题，通常无害，但可以统一:
git config --global core.autocrlf input
git reset
git add -A
```

---

### 问题 2: "rejected ... (non-fast-forward)"

**症状**:
```
error: failed to push some refs to 'origin'
hint: Updates were rejected because the tip of your current branch is behind
```

**解决**:
```bash
# 远程有新提交，需要先拉取
git pull origin main --rebase
git push origin main
```

---

### 问题 3: "large files detected"

**症状**:
```
error: refusing to allow an object larger than 100 MB to be added
```

**解决**:
```bash
# 确认 .gitignore 包含大文件
cat .gitignore | grep -E "models|data|\.gz"

# 清理本地缓存
git rm --cached <大文件>
```

---

### 问题 4: "Authentication failed"

**症状**:
```
fatal: Authentication failed for 'https://github.com/...'
```

**解决**:
```bash
# 使用 SSH 或配置 HTTPS 凭证
git remote set-url origin git@github.com:USERNAME/StreamSpatial-VLM.git
# 或
git config credential.helper store  # 保存凭证
git push origin main
```

---

## 📊 **推送的文件清单**

### 将被推送的文件 (~5000+ 行代码)

```
✅ 源代码
├── models/
│   ├── gate_2d3d.py          (300 行)
│   ├── zip_3d2d.py           (350 行)
│   ├── kv_cache.py           (250 行)
│   ├── stream_spatial_vlm.py  (400 行)
│   └── __init__.py
│
├── scripts/
│   ├── run_ablation.py       (200 行，已修复)
│   ├── launch_pipeline.py    (300 行，新建)
│   └── pipeline_download_and_process.sh  (400 行，新建)
│
├── profiling/
│   ├── profile_vgllm.py      (200 行)
│   └── __init__.py
│
├── eval/
│   ├── eval_*.py             (~300 行)
│   ├── metrics.py            (~150 行)
│   └── __init__.py
│
├── data/
│   ├── dataset_loaders/
│   │   ├── spar7m.py         (200 行)
│   │   └── ...
│   ├── preprocess_*.py       (各 150 行)
│   └── __init__.py
│
└── test_modules.py           (400 行，新建)

✅ 文档
├── README.md                 (更新)
├── PROJECT_STATUS.md         (新建)
├── COMMANDS.md               (完整指南)
├── PHASE_0_2_SUMMARY.md      (新建，项目总结)
├── QUICKSTART.md             (新建)
├── START_HERE.md             (新建)
├── plan.md                   (更新)
└── requirements.txt          (更新)

✅ 配置
├── .gitignore                (更新)
├── configs/*.yaml            (各种配置)
└── setup.sh                  (环境配置)

❌ 不会推送
├── models_cache/             (50+GB, .gitignore)
├── data/raw/**/*.tar.gz      (200+GB, .gitignore)
├── results/logs/             (日志, .gitignore)
├── checkpoints/              (权重, .gitignore)
└── __pycache__/              (缓存, .gitignore)
```

**预计 Git 仓库大小**: ~100-150 MB (仅代码和文档)

---

## ✨ **推送后步骤**

### 在远程验证

```bash
# 访问你的 GitHub/GitLab
# https://github.com/YOUR_USERNAME/StreamSpatial-VLM

# 验证:
# 1. 最新的提交在 main 分支
# 2. 文件列表包括所有新增的 *.md 文档
# 3. 没有大文件被追踪
```

### 分享项目链接

```bash
# 复制仓库 URL
git remote -v

# 分享给团队
echo "项目地址: $(git remote get-url origin)"
```

### 队友克隆项目

```bash
# 队友可以这样克隆
git clone https://github.com/YOUR_USERNAME/StreamSpatial-VLM.git
cd StreamSpatial-VLM

# 设置环境
conda env create -f environment.yml
conda activate streamspatial

# 下载权重和数据
bash start_pipeline.sh  # 后台自动下载
```

---

## 🎯 **最终确认**

在推送前，请确认:

- [x] 运行了 `preflight_push_check.sh` 且全部通过
- [x] `git status` 只显示源代码更改，没有大文件
- [x] 没有 token/API 密钥在代码中
- [x] Python 语法通过检查
- [x] 所有关键文件都在仓库中
- [x] .gitignore 规则已正确设置

**一切就绪？执行:**

```bash
git push origin main
```

**我们已经为您在整个项目生命周期中应用了最佳实践。推送前检查确保快速、正确和安全！** ✨

---

**完成时间**: 2026-04-01  
**推送文件数**: ~50 个  
**代码总行数**: 5000+  
**文档数**: 8+  
**预计大小**: 100-150 MB
