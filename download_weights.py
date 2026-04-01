#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StreamSpatial-VLM 权重下载脚本
支持 VG-LLM、Spatial-MLLM 等模型权重的下载与校验
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi

# ======================== 配置区 ========================
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # 从环境变量读取，或使用 huggingface-cli login

# 是否使用国内镜像
USE_MIRROR = True
MIRROR_URL = "https://hf-mirror.com"

# 模型配置（已验证 HuggingFace 仓库存在）
MODELS = {
    "vgllm-8b": {
        "repo_id": "zd11024/vgllm-3d-vggt-8b",
        "repo_type": "model",
        "description": "VG-LLM 基线权重 (Qwen2.5-VL-8B + VGGT, 3D场景理解)",
        "expected_size_gb": 18.5,
        "expected_files": 4,       # model-00001 ~ 00004.safetensors
        "safetensors_pattern": "model-*.safetensors",
    },
    "vgllm-qa-8b": {
        "repo_id": "zd11024/vgllm-qa-vggt-8b",
        "repo_type": "model",
        "description": "VG-LLM QA 权重 (Qwen2.5-VL-8B + VGGT, 问答优化版)",
        "expected_size_gb": 18.0,
        "expected_files": 4,
        "safetensors_pattern": "model-*.safetensors",
    },
    "spatial-mllm": {
        "repo_id": "Diankun/Spatial-MLLM-v1.1-Instruct-135K",
        "repo_type": "model",
        "description": "Spatial-MLLM v1.1 权重 (Qwen2.5-VL-3B, 空间推理)",
        "expected_size_gb": 10.6,
        "expected_files": 3,       # model-00001 ~ 00003.safetensors
        "safetensors_pattern": "model-*.safetensors",
    },
}
# ========================================================


class WeightDownloader:
    def __init__(self, checkpoint_dir=DEFAULT_CHECKPOINT_DIR, token=HF_TOKEN,
                 use_mirror=USE_MIRROR, do_login=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.token = token
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 配置镜像
        if use_mirror:
            os.environ["HF_ENDPOINT"] = MIRROR_URL
            print(f"🌐 已启用国内镜像加速: {MIRROR_URL}")
        else:
            print("🌐 使用 HuggingFace 官方源")

        # 登录（可选）
        self._do_login = bool(do_login)
        if self._do_login:
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

    def check_disk_space(self, required_gb=30):
        """检查磁盘空间"""
        stat = shutil.disk_usage(self.checkpoint_dir)
        free_gb = stat.free / (1024 ** 3)
        print(f"\n💾 磁盘空间检查:")
        print(f"   路径:     {self.checkpoint_dir.absolute()}")
        print(f"   可用空间: {free_gb:.2f} GB")
        print(f"   需要空间: {required_gb:.2f} GB")
        if free_gb < required_gb:
            print(f"   ⚠️  警告: 磁盘空间可能不足！")
            return False
        print(f"   ✅ 空间充足")
        return True

    def download_model(self, model_name, local_dir=None, ignore_patterns=None):
        """
        从 Hugging Face 下载模型权重

        Args:
            model_name: 模型名称 (对应 MODELS 字典的 key)
            local_dir: 本地保存路径 (默认: checkpoints/<model_name>)
            ignore_patterns: 要跳过的文件 glob 模式列表
        """
        if model_name not in MODELS:
            print(f"❌ 未知模型: {model_name}")
            print(f"   可用选项: {list(MODELS.keys())}")
            return False

        config = MODELS[model_name]
        if local_dir is None:
            local_dir = self.checkpoint_dir / model_name
        else:
            local_dir = Path(local_dir)

        print(f"\n{'=' * 60}")
        print(f"📦 下载模型: {model_name}")
        print(f"📝 描述: {config['description']}")
        print(f"📊 预估大小: ~{config['expected_size_gb']} GB")
        print(f"🎯 保存路径: {local_dir.absolute()}")
        print(f"🔗 仓库地址: https://huggingface.co/{config['repo_id']}")
        print(f"{'=' * 60}\n")

        # 默认跳过非必要文件（节省空间和时间）
        if ignore_patterns is None:
            ignore_patterns = [
                "*.gguf",               # GGUF 格式（留给 llama.cpp 用）
                "*.ot",                 # Optimizer states
                "logs/", "*.log",       # 训练日志
                "*.msgpack",            # 其他序列化文件
            ]

        # 检查磁盘空间
        if not self.check_disk_space(required_gb=config['expected_size_gb'] + 5):
            print("⚠️  磁盘空间不足，建议清理后重试。")
            if getattr(self, "_auto_confirm", False):
                print("自动确认: 继续下载（--yes）")
            else:
                print("是否继续？(y/n)")
                if input().strip().lower() != 'y':
                    print("❌ 已取消下载。")
                    return False

        start_time = time.time()

        try:
            local_path = snapshot_download(
                repo_id=config["repo_id"],
                repo_type=config["repo_type"],
                local_dir=str(local_dir),
                resume_download=True,
                token=self.token,
                ignore_patterns=ignore_patterns,
            )
        except KeyboardInterrupt:
            print("\n⚠️  下载被中断。再次运行本脚本即可从断点继续。")
            return False
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("💡 提示: 如果是网络超时，直接重新运行此脚本即可断点续传。")
            return False

        elapsed_min = (time.time() - start_time) / 60

        print(f"\n{'=' * 60}")
        print(f"✅ 下载完成!")
        print(f"⏱️  耗时: {elapsed_min:.1f} 分钟")
        print(f"📂 路径: {local_path}")
        print(f"{'=' * 60}")

        # 自动校验
        self.verify_download(model_name, local_dir)

        return True

    def verify_download(self, model_name, local_dir=None):
        """
        校验下载的权重文件完整性

        Args:
            model_name: 模型名称
            local_dir: 本地路径 (默认自动推断)
        """
        if model_name not in MODELS:
            print(f"❌ 未知模型: {model_name}")
            return False

        config = MODELS[model_name]
        if local_dir is None:
            local_dir = self.checkpoint_dir / model_name
        else:
            local_dir = Path(local_dir)

        if not local_dir.exists():
            print(f"❌ 目录不存在: {local_dir}")
            return False

        print(f"\n🔍 校验模型: {model_name}")
        print(f"   路径: {local_dir.absolute()}")

        # 检查关键文件
        critical_files = [
            "config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",   # 分片索引
        ]

        all_ok = True
        for cf in critical_files:
            cf_path = local_dir / cf
            if cf_path.exists():
                size_mb = cf_path.stat().st_size / (1024 ** 2)
                print(f"   ✅ {cf} ({size_mb:.2f} MB)")
            else:
                # 单文件模型可能没有 index.json
                if cf == "model.safetensors.index.json":
                    single_model = local_dir / "model.safetensors"
                    if single_model.exists():
                        print(f"   ✅ model.safetensors (单文件模型)")
                    else:
                        print(f"   ❌ 缺少: {cf} 或 model.safetensors")
                        all_ok = False
                else:
                    print(f"   ❌ 缺少: {cf}")
                    all_ok = False

        # 检查 safetensors 分片数量
        safetensors = sorted(local_dir.glob(config["safetensors_pattern"]))
        expected = config["expected_files"]
        if len(safetensors) == expected:
            total_size_gb = sum(f.stat().st_size for f in safetensors) / (1024 ** 3)
            print(f"   ✅ Safetensors 分片: {len(safetensors)}/{expected} "
                  f"(共 {total_size_gb:.2f} GB)")
        elif len(safetensors) > 0:
            print(f"   ⚠️  Safetensors 分片: {len(safetensors)}/{expected} "
                  f"(数量不匹配，可能下载未完成)")
            all_ok = False
        else:
            print(f"   ❌ 未找到任何 safetensors 文件!")
            all_ok = False

        # 总大小统计
        all_files = [f for f in local_dir.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in all_files) / (1024 ** 3)
        print(f"   📊 总文件数: {len(all_files)}")
        print(f"   📊 总大小:   {total_size:.2f} GB")

        if all_ok:
            print(f"   🎉 校验通过!")
        else:
            print(f"   ⚠️  校验发现问题，建议重新运行下载。")

        return all_ok

    def verify_all(self):
        """校验所有已下载的模型"""
        print("\n" + "=" * 60)
        print("🔍 全局校验: 扫描所有已下载模型")
        print("=" * 60)

        results = {}
        for name in MODELS:
            model_dir = self.checkpoint_dir / name
            if model_dir.exists():
                results[name] = self.verify_download(name)
            else:
                print(f"\n⏭️  {name}: 未下载，跳过")
                results[name] = None

        # 汇总
        print(f"\n{'=' * 60}")
        print("📋 校验汇总:")
        for name, ok in results.items():
            if ok is True:
                print(f"   ✅ {name}")
            elif ok is False:
                print(f"   ❌ {name} (有问题)")
            else:
                print(f"   ⏭️  {name} (未下载)")
        print(f"{'=' * 60}")

    def list_available(self):
        """列出所有可下载的模型"""
        print(f"\n{'=' * 60}")
        print("📦 可用模型列表:")
        print(f"{'=' * 60}")
        for name, config in MODELS.items():
            local_dir = self.checkpoint_dir / name
            status = "✅ 已下载" if local_dir.exists() else "⬜ 未下载"
            print(f"   [{status}] {name:20s} | ~{config['expected_size_gb']:5.1f} GB | {config['description']}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="StreamSpatial-VLM 权重下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载 VG-LLM 基线权重
  python download_weights.py --model vgllm-8b

  # 下载 Spatial-MLLM 到自定义路径
  python download_weights.py --model spatial-mllm --local-dir /data/models/spatial-mllm

  # 下载全部模型
  python download_weights.py --model all

  # 仅校验已下载的模型
  python download_weights.py --verify-all

  # 查看可用模型列表
  python download_weights.py --list

  # 不使用镜像 (海外服务器)
  python download_weights.py --model vgllm-8b --no-mirror
        """
    )
    parser.add_argument("--model", type=str, default=None,
                        help="要下载的模型名称 (vgllm-8b / vgllm-qa-8b / spatial-mllm / all)")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="自定义保存路径 (默认: checkpoints/<model_name>)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                        help="全局 checkpoint 根路径 (默认: checkpoints)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token（可选，优先级高于环境变量 HF_TOKEN）")
    parser.add_argument("--no-mirror", action="store_true",
                        help="不使用国内镜像（使用 HuggingFace 官方源）")
    parser.add_argument("--no-login", action="store_true",
                        help="跳过 Hugging Face 登录（使用本地已有凭证）")
    parser.add_argument("--verify-all", action="store_true",
                        help="校验所有已下载模型")
    parser.add_argument("--list", action="store_true",
                        help="列出可用模型及下载状态")
    parser.add_argument("--ignore-pattern", action="append", default=[],
                        help="传递给 snapshot_download 的 ignore_patterns，可多次使用")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="自动在磁盘空间不足时确认继续下载")

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or HF_TOKEN
    use_mirror = not args.no_mirror

    downloader = WeightDownloader(checkpoint_dir=args.checkpoint_dir, token=token,
                                  use_mirror=use_mirror, do_login=not args.no_login)
    # 自动确认提示（用于 --yes）
    downloader._auto_confirm = bool(args.yes)

    if args.list:
        downloader.list_available()
        return

    if args.verify_all:
        downloader.verify_all()
        return

    if args.model is None:
        parser.print_help()
        return

    # 解析要下载的模型列表
    if args.model == "all":
        models_to_download = list(MODELS.keys())
    else:
        if args.model not in MODELS:
            print(f"❌ 未知模型: {args.model}")
            print("可用模型:", ", ".join(MODELS.keys()))
            sys.exit(1)
        models_to_download = [args.model]

    for name in models_to_download:
        # local_dir 处理：若传入 --local-dir 并且是 all，则为 <local_dir>/<model>
        local_dir = None
        if args.local_dir:
            ld = Path(args.local_dir)
            local_dir = (ld / name) if args.model == "all" else ld

        ignore_patterns = args.ignore_pattern if args.ignore_pattern else None

        try:
            ok = downloader.download_model(name, local_dir=local_dir, ignore_patterns=ignore_patterns)
            if not ok:
                print(f"❌ 下载或校验失败: {name}")
        except KeyboardInterrupt:
            print("\n⚠️  用户中断")
            break

if __name__ == "__main__":
    main()
