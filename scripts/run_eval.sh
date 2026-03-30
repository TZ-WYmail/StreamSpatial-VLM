#!/usr/bin/env bash
# ============================================================
# 完整评估流水线
# 用法：bash scripts/run_eval.sh [--config CONFIG] [--dataset all|spar7m|scanqa|scanrefer]
# ============================================================
set -euo pipefail

# ---------- 默认参数 ----------
CONFIG="configs/streamspatial_default.yaml"
DATASET="all"
DATA_ROOT="data/raw"
OUTPUT_DIR="results"
MAX_SAMPLES=500   # 快速验证时限制样本数，正式评估设为 -1

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2";    shift 2 ;;
    --dataset)   DATASET="$2";  shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --output)    OUTPUT_DIR="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  StreamSpatial-VLM 评估流水线"
echo "  配置文件: $CONFIG"
echo "  数据集:   $DATASET"
echo "  输出目录: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# ---------- 评估函数 ----------
run_spar7m() {
  echo ""
  echo ">>> [1/3] 评估 SPAR-7M ..."
  python eval/eval_spar7m.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT/spar7m" \
    --output_dir "$OUTPUT_DIR/spar7m" \
    --max_samples "$MAX_SAMPLES"
  echo "<<< SPAR-7M 评估完成"
}

run_scanqa() {
  echo ""
  echo ">>> [2/3] 评估 ScanQA ..."
  python eval/eval_scanqa.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT/scanqa" \
    --output_dir "$OUTPUT_DIR/scanqa" \
    --max_samples "$MAX_SAMPLES"
  echo "<<< ScanQA 评估完成"
}

run_scanrefer() {
  echo ""
  echo ">>> [3/3] 评估 ScanRefer ..."
  python eval/eval_scanrefer.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT/scanrefer" \
    --output_dir "$OUTPUT_DIR/scanrefer" \
    --max_samples "$MAX_SAMPLES"
  echo "<<< ScanRefer 评估完成"
}

# ---------- 执行 ----------
case "$DATASET" in
  all)
    run_spar7m
    run_scanqa
    run_scanrefer
    ;;
  spar7m)   run_spar7m ;;
  scanqa)   run_scanqa ;;
  scanrefer) run_scanrefer ;;
  *) echo "未知数据集: $DATASET"; exit 1 ;;
esac

echo ""
echo "========================================"
echo "  所有评估完成！结果保存在: $OUTPUT_DIR"
echo "========================================"
