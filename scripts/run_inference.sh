#!/usr/bin/env bash
# ============================================================
# 单视频流式推理脚本
# 用法：bash scripts/run_inference.sh --video path/to/video.mp4 --query "问题"
# ============================================================
set -euo pipefail

# ---------- 默认参数 ----------
CONFIG="configs/streamspatial_default.yaml"
VIDEO=""
QUERY="请描述场景中物体的空间关系"
OUTPUT_DIR="results/inference"
MAX_FRAMES=32
SAVE_VIZ=true

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2";     shift 2 ;;
    --video)      VIDEO="$2";      shift 2 ;;
    --query)      QUERY="$2";      shift 2 ;;
    --output)     OUTPUT_DIR="$2"; shift 2 ;;
    --max_frames) MAX_FRAMES="$2"; shift 2 ;;
    --no_viz)     SAVE_VIZ=false;  shift 1 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "错误：请通过 --video 指定输入视频路径"
  exit 1
fi

echo "========================================"
echo "  StreamSpatial-VLM 流式推理"
echo "  视频:     $VIDEO"
echo "  问题:     $QUERY"
echo "  配置:     $CONFIG"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

python demo/streaming_demo.py \
  --config "$CONFIG" \
  --video "$VIDEO" \
  --query "$QUERY" \
  --output_dir "$OUTPUT_DIR" \
  --max_frames "$MAX_FRAMES" \
  $( [[ "$SAVE_VIZ" == "true" ]] && echo "--save_viz" || echo "" )

echo ""
echo "推理完成！结果保存在: $OUTPUT_DIR"
