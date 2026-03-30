#!/usr/bin/env python3
"""
StreamSpatial-VLM 数据集下载脚本
支持 SPAR-7M、ScanQA、ScanRefer 等数据集
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, login, HfApi
import urllib.request
import zipfile
import json

# 配置
DEFAULT_DATA_DIR = Path("data/raw")
HF_TOKEN = "REDACTED"

# 数据集配置
DATASETS = {
    "spar7m": {
        "repo_id": "jasonzhango/SPAR-7M",
        "repo_type": "dataset",
        "description": "SPAR-7M 精简版（图片+QA）",
        "size": "~30GB"
    },
    "spar7m-rgbd": {
        "repo_id": "jasonzhango/SPAR-7M-RGBD",
        "repo_type": "dataset", 
        "description": "SPAR-7M 完整版（含深度图和相机参数）",
        "size": "~100GB"
    }
}

class DatasetDownloader:
    def __init__(self, data_dir=DEFAULT_DATA_DIR, token=HF_TOKEN):
        self.data_dir = Path(data_dir)
        self.token = token
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 登录 Hugging Face
        self._login()

    def _login(self):
        """登录 Hugging Face"""
        try:
            login(token=self.token)
            user_info = HfApi().whoami()
            print(f"✅ Hugging Face 登录成功: {user_info['name']}")
        except Exception as e:
            print(f"❌ 登录失败: {e}")
            sys.exit(1)

    def download_huggingface_dataset(self, dataset_name, local_dir=None, resume=True):
        """
        从 Hugging Face 下载数据集

        Args:
            dataset_name: 数据集名称 (spar7m 或 spar7m-rgbd)
            local_dir: 本地保存路径
            resume: 是否断点续传
        """
        if dataset_name not in DATASETS:
            print(f"❌ 未知数据集: {dataset_name}")
            print(f"可用选项: {list(DATASETS.keys())}")
            return False

        config = DATASETS[dataset_name]
        if local_dir is None:
            local_dir = self.data_dir / dataset_name
        else:
            local_dir = Path(local_dir)

        print(f"\n{'='*60}")
        print(f"📦 下载数据集: {dataset_name}")
        print(f"📝 描述: {config['description']}")
        print(f"📊 预估大小: {config['size']}")
        print(f"🎯 保存路径: {local_dir.absolute()}")
        print(f"{'='*60}\n")

        try:
            local_path = snapshot_download(
                repo_id=config["repo_id"],
                repo_type=config["repo_type"],
                local_dir=str(local_dir),
                resume_download=resume,
                token=self.token,
                local_dir_use_symlinks=False  # 禁用符号链接，直接复制文件
            )
            print(f"✅ 下载完成: {local_path}")
            return True
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False

    def download_file(self, url, output_path, desc="Downloading"):
        """下载单个文件并显示进度"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{desc}: {percent}%")
            sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, output_path, progress_hook)
            print(f"\n✅ 下载完成: {output_path}")
            return True
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            return False

    def extract_zip(self, zip_path, extract_to=None):
        """解压 zip 文件"""
        zip_path = Path(zip_path)
        if extract_to is None:
            extract_to = zip_path.parent
        else:
            extract_to = Path(extract_to)

        extract_to.mkdir(parents=True, exist_ok=True)

        print(f"📂 解压 {zip_path} 到 {extract_to}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ 解压完成")
            return True
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False

    def setup_scanqa(self):
        """设置 ScanQA 数据集（提供下载指引）"""
        scanqa_dir = self.data_dir / "scanqa"
        scanqa_dir.mkdir(parents=True, exist_ok=True)

        instructions = """
# ScanQA 数据集下载指引

由于 ScanQA 需要 Google Drive 下载，请按以下步骤操作：

1. 访问: https://github.com/ATR-DBI/ScanQA
2. 下载 ScanQA_v1.0.zip (约 2GB)
3. 解压到: {scanqa_dir}

或者使用 gdown（如果已安装）：
    pip install gdown
    gdown --id <file_id_from_github> -O {scanqa_dir}/ScanQA_v1.0.zip

注意: 只需要 train/val 集即可，test 集需要额外申请
        """.format(scanqa_dir=scanqa_dir.absolute())

        readme_path = scanqa_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(instructions)

        print(f"\n📝 ScanQA 下载指引已保存到: {readme_path}")
        print(instructions)

    def setup_scanrefer(self):
        """设置 ScanRefer 数据集（提供下载指引）"""
        scanrefer_dir = self.data_dir / "scanrefer"
        scanrefer_dir.mkdir(parents=True, exist_ok=True)

        instructions = """
# ScanRefer 数据集下载指引

ScanRefer 需要邮件申请，步骤如下：

1. 访问: https://daveredrum.github.io/ScanRefer/
2. 填写 Terms of Use 表格
3. 等待邮件回复（通常 1-3 天）
4. 下载数据并解压到: {scanrefer_dir}

替代方案（无需申请）：
下载 ScanEnts3D 预处理版本：
    wget https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip
    unzip scanrefer.zip -d /home/tanzheng/Desktop/myproject/graduation_project/StreamSpatial-VLM/data/raw/scanrefer
        """

        readme_path = scanrefer_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(instructions)

        print(f"\n📝 ScanRefer 下载指引已保存到: {readme_path}")
        print(instructions)

    def check_disk_space(self, path=".", required_gb=50):
        """检查磁盘空间"""
        import shutil
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        print(f"\n💾 磁盘空间检查:")
        print(f"   可用空间: {free_gb:.2f} GB")
        print(f"   需要空间: {required_gb} GB")
        if free_gb < required_gb:
            print(f"   ⚠️ 警告: 磁盘空间不足！")
            return False
        print(f"   ✅ 空间充足")
        return True

    def verify_download(self, dataset_name):
        """验证下载的数据集"""
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            print(f"❌ 数据集目录不存在: {dataset_dir}")
            return False

        # 统计文件
        files = list(dataset_dir.rglob("*"))
        dirs = [f for f in files if f.is_dir()]
        files = [f for f in files if f.is_file()]

        print(f"\n📊 {dataset_name} 数据集验证:")
        print(f"   目录数: {len(dirs)}")
        print(f"   文件数: {len(files)}")

        # 计算总大小
        total_size = sum(f.stat().st_size for f in files)
        total_size_gb = total_size / (1024**3)
        print(f"   总大小: {total_size_gb:.2f} GB")

        return True

def main():
    parser = argparse.ArgumentParser(description="StreamSpatial-VLM 数据集下载工具")
    parser.add_argument("--dataset", type=str, choices=["spar7m", "spar7m-rgbd", "all"],
                       default="spar7m", help="要下载的数据集")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help="数据保存目录")
    parser.add_argument("--check-space", action="store_true",
                       help="检查磁盘空间")
    parser.add_argument("--setup-only", action="store_true",
                       help="仅设置目录结构，不下载")

    args = parser.parse_args()

    print("🚀 StreamSpatial-VLM 数据集下载工具")
    print("=" * 60)

    # 初始化下载器
    downloader = DatasetDownloader(data_dir=args.data_dir)

    # 检查磁盘空间
    if args.check_space:
        downloader.check_disk_space(args.data_dir, required_gb=50)
        return

    # 设置 ScanQA 和 ScanRefer 指引
    if args.setup_only or args.dataset == "all":
        downloader.setup_scanqa()
        downloader.setup_scanrefer()

    # 下载 Hugging Face 数据集
    if args.dataset == "all":
        datasets_to_download = ["spar7m", "spar7m-rgbd"]
    else:
        datasets_to_download = [args.dataset]

    success_count = 0
    for dataset in datasets_to_download:
        if downloader.download_huggingface_dataset(dataset):
            downloader.verify_download(dataset)
            success_count += 1

    print(f"\n{'='*60}")
    print(f"✅ 下载完成: {success_count}/{len(datasets_to_download)} 个数据集")
    print(f"📁 数据保存位置: {Path(args.data_dir).absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()