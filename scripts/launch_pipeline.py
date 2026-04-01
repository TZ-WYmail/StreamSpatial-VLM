#!/usr/bin/env python
"""
StreamSpatial-VLM 完整数据处理管道
====================================
后台启动 SPAR-7M-RGBD 下载 + 自动预处理（深度图 + 位姿置信度）

预计耗时: 24-48h（下载 200GB + 处理）

使用方法:
    python scripts/launch_pipeline.py [OPTIONS]

选项:
    --mode {full|download|preprocess|check}  执行模式（默认: full）
    --datasets {all|scannet|scannetpp|...}   指定数据集
    --batch-size INT                          批处理大小
    --monitor                                实时监控
    --no-backup                              禁用备份日志
    --help                                   显示帮助

示例:
    # 完整流程
    python scripts/launch_pipeline.py

    # 仅下载
    python scripts/launch_pipeline.py --mode download

    # 仅预处理 + 监控
    python scripts/launch_pipeline.py --mode preprocess --monitor

    # 检查状态
    python scripts/launch_pipeline.py --mode check
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from threading import Thread
import shutil

# 项目配置
PROJECT_ROOT = Path("/home/tanzheng/Desktop/myproject/StreamSpatial-VLM")
CONDA_ENV = "streamspatial"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "spar7m"
STATE_FILE = LOG_DIR / "pipeline_state.json"

# ANSI 颜色
class Colors:
    GRAY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def setup_logging():
    """初始化日志"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR

def log(level, msg, category="MAIN"):
    """统一日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARN": Colors.YELLOW,
        "ERROR": Colors.RED,
        "DEBUG": Colors.GRAY,
    }
    
    color = color_map.get(level, Colors.BLUE)
    icon_map = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "WARN": "⚠",
        "ERROR": "✗",
        "DEBUG": "•",
    }
    icon = icon_map.get(level, "!")
    
    log_msg = f"{color}{icon}{Colors.END} [{timestamp}] [{category}] {msg}"
    print(log_msg)
    
    # 写入日志文件
    log_file = LOG_DIR / "pipeline.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{level}] [{timestamp}] [{category}] {msg}\n")

def run_command(cmd, log_file=None):
    """运行命令并捕获输出"""
    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT
                )
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"STDERR: {result.stderr}\n")
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_environment():
    """检查环境"""
    log("INFO", "检查环境...", "SETUP")
    
    # 检查 conda 环境
    success, _, _ = run_command(f"conda env list | grep {CONDA_ENV}")
    if not success:
        log("ERROR", f"Conda 环境不存在: {CONDA_ENV}", "SETUP")
        return False
    log("SUCCESS", "Conda 环境检查通过", "SETUP")
    
    # 检查项目目录
    if not PROJECT_ROOT.exists():
        log("ERROR", f"项目目录不存在: {PROJECT_ROOT}", "SETUP")
        return False
    log("SUCCESS", "项目目录检查通过", "SETUP")
    
    # 检查脚本
    scripts = [
        "download_datasets.py",
        "data/preprocess_depth.py",
        "data/preprocess_pose.py",
    ]
    
    for script in scripts:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            log("ERROR", f"脚本不存在: {script}", "SETUP")
            return False
    log("SUCCESS", "脚本检查通过", "SETUP")
    
    return True

def check_disk_space():
    """检查磁盘空间"""
    log("INFO", "检查磁盘空间...", "SETUP")
    
    stat = shutil.disk_usage(PROJECT_ROOT)
    available_gb = stat.free / (1024 ** 3)
    required_gb = 250  # 200GB 数据 + 50GB 预留
    
    log("INFO", f"可用空间: {available_gb:.1f} GB", "SETUP")
    log("INFO", f"所需空间: {required_gb} GB", "SETUP")
    
    if available_gb < required_gb:
        log("ERROR", "磁盘空间不足！", "SETUP")
        return False
    
    log("SUCCESS", "磁盘空间检查通过", "SETUP")
    return True

def save_state(state):
    """保存管道状态"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def load_state():
    """加载管道状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "started_at": None,
        "phase": "initialized",
        "status": "pending",
        "downloads": {},
        "preprocessing": {},
    }

def download_spar7m_rgbd():
    """下载 SPAR-7M-RGBD"""
    log("INFO", "======== 阶段 1: 下载 SPAR-7M-RGBD ========", "DOWNLOAD")
    log("INFO", "预计耗时: 24-48 小时", "DOWNLOAD")
    
    log_file = LOG_DIR / "download.log"
    
    cmd = (
        f"conda run -n {CONDA_ENV} python download_datasets.py "
        f"--dataset spar7m-rgbd "
        f"--data-dir {DATA_ROOT.parent}"
    )
    
    log("INFO", f"执行命令: {cmd}", "DOWNLOAD")
    log("INFO", f"日志文件: {log_file}", "DOWNLOAD")
    
    success, stdout, stderr = run_command(cmd, str(log_file))
    
    if success:
        log("SUCCESS", "SPAR-7M-RGBD 下载完成", "DOWNLOAD")
        
        # 统计下载大小
        total_size = 0
        for item in DATA_ROOT.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        
        total_gb = total_size / (1024 ** 3)
        log("INFO", f"下载总大小: {total_gb:.1f} GB", "DOWNLOAD")
        
        state = load_state()
        state["downloads"]["spar7m_rgbd"] = {
            "status": "completed",
            "size_gb": total_gb,
            "timestamp": datetime.now().isoformat(),
        }
        save_state(state)
        
        return True
    else:
        log("ERROR", "下载失败，请检查日志", "DOWNLOAD")
        log("DEBUG", stderr, "DOWNLOAD")
        return False

def generate_depth_maps(dataset):
    """生成深度图"""
    log("INFO", f"为 {dataset} 生成深度图...", "DEPTH")
    
    dataset_dir = DATA_ROOT / dataset
    if not dataset_dir.exists():
        log("WARN", f"数据集目录不存在: {dataset_dir}", "DEPTH")
        return False
    
    input_dir = dataset_dir / "images"
    output_dir = dataset_dir / "depth_pred"
    
    # 检查是否已有深度图
    existing_count = len(list(output_dir.glob("**/*.npy"))) if output_dir.exists() else 0
    image_count = len(list(input_dir.glob("**/*.jpg"))) + len(list(input_dir.glob("**/*.png"))) if input_dir.exists() else 0
    
    if image_count == 0:
        log("WARN", f"{dataset} 无图像文件", "DEPTH")
        return True
    
    if existing_count == image_count:
        log("WARN", f"{dataset} 已有 {existing_count} 个深度图，跳过", "DEPTH")
        return True
    
    log_file = LOG_DIR / f"depth_{dataset}.log"
    
    cmd = (
        f"conda run -n {CONDA_ENV} python data/preprocess_depth.py "
        f"--input_dir {input_dir} "
        f"--output_dir {output_dir} "
        f"--model_size vitl "
        f"--batch_size 8"
    )
    
    log("INFO", f"{dataset}: {image_count} 张图像 → 生成深度图...", "DEPTH")
    
    success, _, stderr = run_command(cmd, str(log_file))
    
    if success:
        final_count = len(list(output_dir.glob("**/*.npy")))
        log("SUCCESS", f"{dataset} 深度图生成完成 ({final_count}/{image_count})", "DEPTH")
        
        state = load_state()
        if "preprocessing" not in state:
            state["preprocessing"] = {}
        if dataset not in state["preprocessing"]:
            state["preprocessing"][dataset] = {}
        
        state["preprocessing"][dataset]["depth"] = {
            "status": "completed",
            "count": final_count,
            "timestamp": datetime.now().isoformat(),
        }
        save_state(state)
        
        return True
    else:
        log("ERROR", f"{dataset} 深度图生成失败", "DEPTH")
        log("DEBUG", stderr, "DEPTH")
        return False

def generate_pose_confidence(dataset):
    """生成位姿置信度"""
    log("INFO", f"为 {dataset} 生成位姿置信度图...", "POSE")
    
    dataset_dir = DATA_ROOT / dataset
    if not dataset_dir.exists():
        log("WARN", f"数据集目录不存在: {dataset_dir}", "POSE")
        return False
    
    input_dir = dataset_dir / "images"
    output_dir = dataset_dir / "pose_conf"
    
    # 检查是否已有位姿文件
    existing_count = len(list(output_dir.glob("**/*.npy"))) if output_dir.exists() else 0
    image_count = len(list(input_dir.glob("**/*.jpg"))) + len(list(input_dir.glob("**/*.png"))) if input_dir.exists() else 0
    
    if image_count == 0:
        log("WARN", f"{dataset} 无图像文件", "POSE")
        return True
    
    if existing_count == image_count:
        log("WARN", f"{dataset} 已有 {existing_count} 个位姿文件，跳过", "POSE")
        return True
    
    log_file = LOG_DIR / f"pose_{dataset}.log"
    
    cmd = (
        f"conda run -n {CONDA_ENV} python data/preprocess_pose.py "
        f"--input_dir {input_dir} "
        f"--output_dir {output_dir} "
        f"--batch_size 4 "
        f"--img_size 518"
    )
    
    log("INFO", f"{dataset}: {image_count} 张图像 → 生成位姿置信度...", "POSE")
    
    success, _, stderr = run_command(cmd, str(log_file))
    
    if success:
        final_count = len(list(output_dir.glob("**/*.npy")))
        log("SUCCESS", f"{dataset} 位姿置信度生成完成 ({final_count}/{image_count})", "POSE")
        
        state = load_state()
        if "preprocessing" not in state:
            state["preprocessing"] = {}
        if dataset not in state["preprocessing"]:
            state["preprocessing"][dataset] = {}
        
        state["preprocessing"][dataset]["pose"] = {
            "status": "completed",
            "count": final_count,
            "timestamp": datetime.now().isoformat(),
        }
        save_state(state)
        
        return True
    else:
        log("ERROR", f"{dataset} 位姿置信度生成失败", "POSE")
        log("DEBUG", stderr, "POSE")
        return False

def preprocess_all_datasets(datasets_filter=None):
    """预处理所有数据集"""
    log("INFO", "======== 阶段 2 & 3: 预处理所有数据集 ========", "PREPROCESS")
    
    all_datasets = ["scannet", "scannetpp", "structured3d", "rxr"]
    
    if datasets_filter and datasets_filter != "all":
        all_datasets = [d for d in all_datasets if d == datasets_filter]
    
    for dataset in all_datasets:
        dataset_dir = DATA_ROOT / dataset
        
        if not dataset_dir.exists() or not (dataset_dir / "images").exists():
            log("WARN", f"跳过 {dataset}（目录/图像不存在）", "PREPROCESS")
            continue
        
        image_count = len(list((dataset_dir / "images").glob("**/*.jpg"))) + \
                     len(list((dataset_dir / "images").glob("**/*.png")))
        
        if image_count == 0:
            log("WARN", f"跳过 {dataset}（无图像）", "PREPROCESS")
            continue
        
        log("INFO", f"处理 {dataset} ({image_count} 张图像)...", "PREPROCESS")
        
        # 生成深度图
        generate_depth_maps(dataset)
        
        # 生成位姿置信度
        generate_pose_confidence(dataset)
        
        time.sleep(1)

def check_status():
    """检查处理进度"""
    log("INFO", "检查数据集状态...", "STATUS")
    
    print(f"\n{Colors.BOLD}==================== 数据集状态 ===================={Colors.END}\n")
    
    all_datasets = ["scannet", "scannetpp", "structured3d", "rxr"]
    
    for dataset in all_datasets:
        dataset_dir = DATA_ROOT / dataset
        
        if not dataset_dir.exists():
            print(f"❌ {dataset:<15} | 目录不存在")
            continue
        
        images_dir = dataset_dir / "images"
        depth_dir = dataset_dir / "depth_pred"
        pose_dir = dataset_dir / "pose_conf"
        
        if not images_dir.exists():
            print(f"❌ {dataset:<15} | 无图像目录")
            continue
        
        img_count = len(list(images_dir.glob("**/*.jpg"))) + \
                   len(list(images_dir.glob("**/*.png")))
        depth_count = len(list(depth_dir.glob("**/*.npy"))) if depth_dir.exists() else 0
        pose_count = len(list(pose_dir.glob("**/*.npy"))) if pose_dir.exists() else 0
        
        # 计算大小
        total_size = 0
        for item in dataset_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        
        size_gb = total_size / (1024 ** 3)
        
        depth_status = "✅" if depth_count == img_count else f"⏳ ({depth_count}/{img_count})" if depth_count > 0 else "❌"
        pose_status = "✅" if pose_count == img_count else f"⏳ ({pose_count}/{img_count})" if pose_count > 0 else "❌"
        
        print(f"📦 {dataset:<15}")
        print(f"   ├─ 图像: {img_count:>5} 个")
        print(f"   ├─ 深度图: {depth_status}")
        print(f"   ├─ 位姿: {pose_status}")
        print(f"   └─ 总大小: {size_gb:>6.1f} GB\n")
    
    # 磁盘使用
    print(f"{Colors.BOLD}==================== 磁盘使用 ===================={Colors.END}\n")
    
    stat = shutil.disk_usage(DATA_ROOT.parent)
    used_gb = (stat.total - stat.free) / (1024 ** 3)
    total_gb = stat.total / (1024 ** 3)
    free_gb = stat.free / (1024 ** 3)
    usage_pct = (used_gb / total_gb) * 100
    
    print(f"已用: {used_gb:.1f} GB / 总计: {total_gb:.1f} GB (使用率: {usage_pct:.1f}%)")
    print(f"可用: {free_gb:.1f} GB\n")

def monitor():
    """实时监控"""
    log("INFO", "启动监控模式...", "MONITOR")
    log("INFO", "按 Ctrl+C 退出监控", "MONITOR")
    
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            
            print(f"\n{Colors.BOLD}{'='*50}")
            print(f"StreamSpatial-VLM 数据处理监控")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}{Colors.END}\n")
            
            # 显示进程状态
            dl_proc = subprocess.run(
                "pgrep -f 'download_datasets.py' | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            if int(dl_proc.stdout.strip()) > 0:
                print(f"{Colors.GREEN}📥 下载{Colors.END}: 运行中")
            else:
                print(f"{Colors.GRAY}📥 下载{Colors.END}: 空闲")
            
            depth_proc = subprocess.run(
                "pgrep -f 'preprocess_depth' | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            if int(depth_proc.stdout.strip()) > 0:
                print(f"{Colors.GREEN}🔍 深度处理{Colors.END}: 运行中")
            else:
                print(f"{Colors.GRAY}🔍 深度处理{Colors.END}: 空闲")
            
            pose_proc = subprocess.run(
                "pgrep -f 'preprocess_pose' | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            if int(pose_proc.stdout.strip()) > 0:
                print(f"{Colors.GREEN}📍 位姿处理{Colors.END}: 运行中")
            else:
                print(f"{Colors.GRAY}📍 位姿处理{Colors.END}: 空闲")
            
            print(f"\n{Colors.YELLOW}最近日志:{Colors.END}")
            if (LOG_DIR / "pipeline.log").exists():
                result = subprocess.run(
                    "tail -n 5 " + str(LOG_DIR / "pipeline.log"),
                    shell=True,
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            
            print(f"按 Ctrl+C 退出监控\n")
            time.sleep(5)
    except KeyboardInterrupt:
        log("INFO", "监控已中止", "MONITOR")

def main():
    parser = argparse.ArgumentParser(description="StreamSpatial-VLM 数据处理管道")
    parser.add_argument(
        "--mode", type=str, choices=["full", "download", "preprocess", "check"],
        default="full", help="执行模式"
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="指定数据集 (all|scannet|scannetpp|structured3d|rxr)"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="启动后监控进度"
    )
    
    args = parser.parse_args()
    
    # 初始化
    setup_logging()
    
    log("INFO", "="*50, "MAIN")
    log("INFO", "StreamSpatial-VLM 数据处理管道", "MAIN")
    log("INFO", f"模式: {args.mode}", "MAIN")
    log("INFO", "="*50, "MAIN")
    
    # 检查环境
    if not check_environment() or not check_disk_space():
        log("ERROR", "环境检查失败", "MAIN")
        sys.exit(1)
    
    # 执行模式
    if args.mode == "check":
        check_status()
    elif args.mode == "download":
        if not download_spar7m_rgbd():
            sys.exit(1)
    elif args.mode == "preprocess":
        preprocess_all_datasets(args.datasets)
    elif args.mode == "full":
        if not download_spar7m_rgbd():
            sys.exit(1)
        time.sleep(2)
        preprocess_all_datasets(args.datasets)
    
    # 最终状态检查
    check_status()
    
    log("SUCCESS", "处理流程完成", "MAIN")
    log("INFO", f"日志: {LOG_DIR / 'pipeline.log'}", "MAIN")
    
    # 监控模式
    if args.monitor:
        monitor()

if __name__ == "__main__":
    main()
