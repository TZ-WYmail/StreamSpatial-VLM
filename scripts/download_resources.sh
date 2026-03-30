#!/usr/bin/env bash
set -euo pipefail

# 下载数据集与模型权重的辅助脚本（非交互式）
# 用法示例：
#   HF_TOKEN=xxxxx ./scripts/download_resources.sh --confirm
# 环境变量（优先使用）：
#   HF_TOKEN           Hugging Face token（若需从 HF 下载模型）
#   SCANNET_URL        可选：ScanNet 子集下载 URL（zip）
#   SCANREFER_URL      可选：ScanRefer 标注下载 URL（zip）
#   SCANQA_URL         可选：ScanQA 标注下载 URL（zip）
#   SPAR7M_URL         可选：SPAR-7M 数据下载 URL（zip）
#   MODEL_QWEN         可选：要从 HF 下载的 VLM repo id，默认 Qwen/Qwen2.5-VL-7B-Instruct
#   MODEL_VGGT         可选：VGGT repo id，默认 facebook/vggt-1b

CONFIRM=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm) CONFIRM=1; shift ;;
    --help) sed -n '1,200p' $0; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ROOT_DIR=$(dirname "$0")/..
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
echo "Workspace root: $ROOT_DIR"

DATA_DIR=$ROOT_DIR/data/raw
mkdir -p "$DATA_DIR"

HF_TOKEN=${HF_TOKEN:-}
MODEL_QWEN=${MODEL_QWEN:-Qwen/Qwen2.5-VL-7B-Instruct}
MODEL_VGGT=${MODEL_VGGT:-facebook/vggt-1b}

download_zip(){
  local url=$1; local outdir=$2; local name=$3
  if [[ -z "$url" ]]; then
    echo "  - 没有提供 $name 的 URL，跳过。"
    return
  fi
  mkdir -p "$outdir"
  tmpzip="/tmp/$(basename $url)"
  echo "  下载 $name 到 $tmpzip ..."
  curl -L "$url" -o "$tmpzip"
  echo "  解压到 $outdir ..."
  unzip -o "$tmpzip" -d "$outdir"
  rm -f "$tmpzip"
  echo "  完成 $name"
}

echo "准备下载数据集（仅当对应环境变量存在时）..."
download_zip "${SCANNET_URL:-}" "$DATA_DIR/scannet" "ScanNet (subset)"
download_zip "${SCANREFER_URL:-}" "$DATA_DIR/scanrefer" "ScanRefer"
download_zip "${SCANQA_URL:-}" "$DATA_DIR/scanqa" "ScanQA"
download_zip "${SPAR7M_URL:-}" "$DATA_DIR/spar7m" "SPAR-7M"

if [[ $CONFIRM -eq 0 ]]; then
  echo "\n注意：脚本未被确认（未传入 --confirm），不会自动下载大型模型权重。"
  echo "若你想让我自动下载模型权重（可能为数 GB），请再次运行："
  echo "  HF_TOKEN=xxx ./scripts/download_resources.sh --confirm"
  echo "或手动使用 huggingface-cli / snapshot_download 下载并将文件放到合适位置。"
  exit 0
fi

if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN 未设置，无法从 Hugging Face 下载受限模型。请设置 HF_TOKEN 环境变量。"
  exit 1
fi

echo "开始从 Hugging Face 下载模型（这可能需要较大磁盘与带宽）"
python - <<PY
from huggingface_hub import snapshot_download
import os
os.environ['HF_TOKEN']=os.environ.get('HF_TOKEN')
root = os.path.abspath('$ROOT_DIR')
models = {
  'vlm': '$MODEL_QWEN',
  'vggt': '$MODEL_VGGT'
}
for key, repo in models.items():
  print(f'Downloading {key} -> {repo} ...')
  out = snapshot_download(repo_id=repo, cache_dir=os.path.join(root, 'models_cache'))
  print(f'  saved to {out}')
PY

echo "所有请求的下载已发起（请查看上方输出以确认成功）。"
