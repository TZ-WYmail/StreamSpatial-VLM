#!/usr/bin/env bash
# ============================================================
# 消融实验批量运行脚本
# 用法：bash scripts/run_ablation.sh [--group A|B|C|D|E|all]
# ============================================================
set -euo pipefail

GROUP="all"
DATA_ROOT="data/raw"
OUTPUT_DIR="results/ablation"
MAX_SAMPLES=200   # 消融实验快速模式

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)      GROUP="$2";      shift 2 ;;
    --data_root)  DATA_ROOT="$2";  shift 2 ;;
    --output)     OUTPUT_DIR="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  StreamSpatial-VLM 消融实验"
echo "  消融组: $GROUP"
echo "  输出:   $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

if [[ "$GROUP" == "all" ]]; then
  for g in A B C D E; do
    echo ""
    echo ">>> 运行消融组 $g ..."
    python scripts/run_ablation.py \
      --group "$g" \
      --data_root "$DATA_ROOT" \
      --output_dir "$OUTPUT_DIR" \
      --max_samples "$MAX_SAMPLES"
    echo "<<< 消融组 $g 完成"
  done
else
  python scripts/run_ablation.py \
    --group "$GROUP" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES"
fi

echo ""
echo "========================================"
echo "  消融实验全部完成！"
echo "  Markdown 表格: $OUTPUT_DIR/ablation_tables.md"
echo "========================================"
