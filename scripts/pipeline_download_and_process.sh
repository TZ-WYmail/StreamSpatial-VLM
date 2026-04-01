#!/bin/bash
# ============================================================================
# StreamSpatial-VLM 完整数据处理流程
# ============================================================================
# 功能: 后台启动 SPAR-7M-RGBD 下载 + 自动预处理（深度图 + 位姿置信度）
# 预计耗时: 24-48h（下载 200GB + 处理）
# 
# 使用方法:
#   bash scripts/pipeline_download_and_process.sh [OPTIONS]
#
# 选项:
#   --check-only        仅检查状态，不启动
#   --skip-download     仅运行预处理，跳过下载
#   --monitor           启动后监控进度
#   --resume            断点续传模式
# ============================================================================

set -e

PROJECT_ROOT="/home/tanzheng/Desktop/myproject/StreamSpatial-VLM"
CONDA_ENV="streamspatial"
LOG_DIR="$PROJECT_ROOT/results/logs"
DATA_ROOT="$PROJECT_ROOT/data/raw/spar7m"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 日志函数
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_DIR/pipeline.log"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_DIR/pipeline.log"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_DIR/pipeline.log"
}

log_error() {
    echo -e "${RED}[✗]${NC} $(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_DIR/pipeline.log"
}

# ============================================================================
# 环境检查
# ============================================================================
check_environment() {
    log_info "检查环境..."
    
    # 检查 conda 环境
    if ! conda env list | grep -q "$CONDA_ENV"; then
        log_error "Conda 环境 '$CONDA_ENV' 不存在"
        exit 1
    fi
    log_success "Conda 环境检查通过"
    
    # 检查项目目录
    if [ ! -d "$PROJECT_ROOT" ]; then
        log_error "项目目录不存在: $PROJECT_ROOT"
        exit 1
    fi
    log_success "项目目录检查通过"
    
    # 检查脚本文件
    if [ ! -f "$PROJECT_ROOT/download_datasets.py" ]; then
        log_error "下载脚本不存在: $PROJECT_ROOT/download_datasets.py"
        exit 1
    fi
    log_success "下载脚本检查通过"
    
    if [ ! -f "$PROJECT_ROOT/data/preprocess_depth.py" ]; then
        log_error "深度预处理脚本不存在"
        exit 1
    fi
    log_success "深度预处理脚本检查通过"
    
    if [ ! -f "$PROJECT_ROOT/data/preprocess_pose.py" ]; then
        log_error "位姿预处理脚本不存在"
        exit 1
    fi
    log_success "位姿预处理脚本检查通过"
}

# ============================================================================
# 磁盘空间检查
# ============================================================================
check_disk_space() {
    log_info "检查磁盘空间..."
    
    AVAILABLE_GB=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | awk '{print $1 / 1024 / 1024}')
    REQUIRED_GB=250  # 200GB 数据 + 50GB 预留
    
    log_info "可用空间: ${AVAILABLE_GB:.1f} GB"
    log_info "所需空间: ${REQUIRED_GB} GB"
    
    if (( $(echo "$AVAILABLE_GB < $REQUIRED_GB" | bc -l) )); then
        log_error "磁盘空间不足！"
        exit 1
    fi
    
    log_success "磁盘空间检查通过"
}

# ============================================================================
# 下载 SPAR-7M-RGBD
# ============================================================================
download_spar7m_rgbd() {
    log_info "开始下载 SPAR-7M-RGBD (预计 24-48 小时)..."
    
    cd "$PROJECT_ROOT"
    
    conda run -n "$CONDA_ENV" python download_datasets.py \
        --dataset spar7m-rgbd \
        --data-dir "$DATA_ROOT" \
        2>&1 | tee -a "$LOG_DIR/download.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "SPAR-7M-RGBD 下载完成"
        return 0
    else
        log_error "下载失败，请检查 $LOG_DIR/download.log"
        return 1
    fi
}

# ============================================================================
# 生成深度图 (Depth-Anything-V2)
# ============================================================================
generate_depth_maps() {
    local dataset=$1
    log_info "为 $dataset 生成伪深度图..."
    
    cd "$PROJECT_ROOT"
    
    # 检查是否已有深度图
    DEPTH_COUNT=$(find "$DATA_ROOT/$dataset/depth_pred" -name "*.npy" 2>/dev/null | wc -l)
    if [ "$DEPTH_COUNT" -gt 0 ]; then
        log_warn "$dataset 已有 $DEPTH_COUNT 个深度图，跳过"
        return 0
    fi
    
    log_info "为 $dataset 生成深度图 (使用 Depth-Anything-V2-ViT-L)..."
    
    conda run -n "$CONDA_ENV" python data/preprocess_depth.py \
        --input_dir "$DATA_ROOT/$dataset/images" \
        --output_dir "$DATA_ROOT/$dataset/depth_pred" \
        --model_size vitl \
        --batch_size 8 \
        --overwrite \
        2>&1 | tee -a "$LOG_DIR/depth_${dataset}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        FINAL_COUNT=$(find "$DATA_ROOT/$dataset/depth_pred" -name "*.npy" 2>/dev/null | wc -l)
        log_success "$dataset 深度图生成完成 (共 $FINAL_COUNT 个)"
        return 0
    else
        log_error "$dataset 深度图生成失败，请检查日志"
        return 1
    fi
}

# ============================================================================
# 生成位姿置信度图 (VGGT)
# ============================================================================
generate_pose_confidence() {
    local dataset=$1
    log_info "为 $dataset 生成位姿置信度图..."
    
    cd "$PROJECT_ROOT"
    
    # 检查是否已有位姿文件
    POSE_COUNT=$(find "$DATA_ROOT/$dataset/pose_conf" -name "*.npy" 2>/dev/null | wc -l)
    if [ "$POSE_COUNT" -gt 0 ]; then
        log_warn "$dataset 已有 $POSE_COUNT 个位姿文件，跳过"
        return 0
    fi
    
    log_info "为 $dataset 生成位姿置信度图 (使用 VGGT-1B)..."
    
    conda run -n "$CONDA_ENV" python data/preprocess_pose.py \
        --input_dir "$DATA_ROOT/$dataset/images" \
        --output_dir "$DATA_ROOT/$dataset/pose_conf" \
        --batch_size 4 \
        --img_size 518 \
        --overwrite \
        2>&1 | tee -a "$LOG_DIR/pose_${dataset}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        FINAL_COUNT=$(find "$DATA_ROOT/$dataset/pose_conf" -name "*.npy" 2>/dev/null | wc -l)
        log_success "$dataset 位姿置信度图生成完成 (共 $FINAL_COUNT 个)"
        return 0
    else
        log_error "$dataset 位姿置信度图生成失败，请检查日志"
        return 1
    fi
}

# ============================================================================
# 完整预处理
# ============================================================================
preprocess_all_datasets() {
    log_info "开始预处理所有数据集..."
    
    DATASETS=("scannet" "scannetpp" "structured3d" "rxr")
    FAILED_DATASETS=()
    
    for dataset in "${DATASETS[@]}"; do
        DATASET_DIR="$DATA_ROOT/$dataset"
        
        if [ ! -d "$DATASET_DIR" ]; then
            log_warn "数据集目录不存在: $DATASET_DIR"
            continue
        fi
        
        IMAGE_COUNT=$(find "$DATASET_DIR/images" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        
        if [ "$IMAGE_COUNT" -eq 0 ]; then
            log_warn "$dataset 无图像文件，跳过"
            continue
        fi
        
        log_info "处理 $dataset ($IMAGE_COUNT 张图像)..."
        
        # 生成深度图
        if ! generate_depth_maps "$dataset"; then
            FAILED_DATASETS+=("$dataset/depth")
        fi
        
        # 生成位姿置信度
        if ! generate_pose_confidence "$dataset"; then
            FAILED_DATASETS+=("$dataset/pose")
        fi
    done
    
    # 输出失败摘要
    if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
        log_warn "以下步骤失败:"
        for failed in "${FAILED_DATASETS[@]}"; do
            log_warn "  - $failed"
        done
    fi
}

# ============================================================================
# 检查处理状态
# ============================================================================
check_status() {
    log_info "检查处理状态..."
    
    cd "$PROJECT_ROOT"
    
    echo ""
    echo "==================== 数据集状态 ===================="
    
    DATASETS=("scannet" "scannetpp" "structured3d" "rxr")
    
    for dataset in "${DATASETS[@]}"; do
        DATASET_DIR="$DATA_ROOT/$dataset"
        
        if [ ! -d "$DATASET_DIR" ]; then
            echo "❌ $dataset: 目录不存在"
            continue
        fi
        
        IMG_COUNT=$(find "$DATASET_DIR/images" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        DEPTH_COUNT=$(find "$DATASET_DIR/depth_pred" -name "*.npy" 2>/dev/null | wc -l)
        POSE_COUNT=$(find "$DATASET_DIR/pose_conf" -name "*.npy" 2>/dev/null | wc -l)
        
        if [ "$IMG_COUNT" -eq 0 ]; then
            echo "❌ $dataset: 无图像"
            continue
        fi
        
        DEPTH_STATUS="❌"
        POSE_STATUS="❌"
        
        if [ "$DEPTH_COUNT" -eq "$IMG_COUNT" ]; then
            DEPTH_STATUS="✅"
        elif [ "$DEPTH_COUNT" -gt 0 ]; then
            DEPTH_STATUS="⏳ ($DEPTH_COUNT/$IMG_COUNT)"
        fi
        
        if [ "$POSE_COUNT" -eq "$IMG_COUNT" ]; then
            POSE_STATUS="✅"
        elif [ "$POSE_COUNT" -gt 0 ]; then
            POSE_STATUS="⏳ ($POSE_COUNT/$IMG_COUNT)"
        fi
        
        echo "📦 $dataset"
        echo "   ├─ 图像: $IMG_COUNT 个"
        echo "   ├─ 深度图: $DEPTH_STATUS"
        echo "   └─ 位姿置信度: $POSE_STATUS"
    done
    
    echo ""
    echo "==================== 磁盘使用 ===================="
    du -sh "$DATA_ROOT" 2>/dev/null || echo "无法计算"
    
    echo ""
}

# ============================================================================
# 监控后台进程
# ============================================================================
monitor_progress() {
    log_info "监控模式启动..."
    log_info "按 Ctrl+C 退出监控（后台任务继续）"
    log_info ""
    
    sleep 2
    
    while true; do
        clear
        
        echo "=========================================="
        echo "StreamSpatial-VLM 数据处理监控面板"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""
        
        # 显示后台进程
        if pgrep -f "download_datasets.py" > /dev/null; then
            PID=$(pgrep -f "download_datasets.py")
            echo "📥 下载进程: PID=$PID (运行中)"
        else
            echo "📥 下载进程: (未运行)"
        fi
        
        if pgrep -f "preprocess_depth.py" > /dev/null; then
            PID=$(pgrep -f "preprocess_depth.py")
            echo "🔍 深度处理: PID=$PID (运行中)"
        else
            echo "🔍 深度处理: (未运行)"
        fi
        
        if pgrep -f "preprocess_pose.py" > /dev/null; then
            PID=$(pgrep -f "preprocess_pose.py")
            echo "📍 位姿处理: PID=$PID (运行中)"
        else
            echo "📍 位姿处理: (未运行)"
        fi
        
        echo ""
        echo "========== 实时日志 =========="
        if [ -f "$LOG_DIR/pipeline.log" ]; then
            echo "最近 10 条日志:"
            tail -n 10 "$LOG_DIR/pipeline.log"
        fi
        
        echo ""
        echo "========== 磁盘空间 =========="
        DF_OUTPUT=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {printf "可用: %s / 总计: %s (使用率: %s)", $4, $2, $5}')
        echo "$DF_OUTPUT"
        
        echo ""
        echo "按 Ctrl+C 退出监控面板"
        echo ""
        
        sleep 5
    done
}

# ============================================================================
# 显示帮助
# ============================================================================
show_help() {
    cat << EOF
StreamSpatial-VLM 完整数据处理流程

用法: bash scripts/pipeline_download_and_process.sh [OPTIONS]

选项:
    --check-only        仅检查状态，不启动
    --skip-download     仅运行预处理，跳过下载
    --monitor           启动后监控进度 (实时仪表板)
    --resume            断点续传模式
    --help              显示此帮助信息

示例:

1. 完整流程（后台下载 + 处理）：
   bash scripts/pipeline_download_and_process.sh

2. 仅检查状态：
   bash scripts/pipeline_download_and_process.sh --check-only

3. 启动并监控：
   bash scripts/pipeline_download_and_process.sh --monitor

4. 仅预处理已下载的数据：
   bash scripts/pipeline_download_and_process.sh --skip-download

流程说明:
  Phase 1: 环境检查 (5分钟)
    ✓ Conda 环境验证
    ✓ 脚本文件检查
    ✓ 磁盘空间检查

  Phase 2: 下载 SPAR-7M-RGBD (24-48小时)
    ✓ 后台下载 200GB+ 数据
    ✓ 断点续传支持

  Phase 3: 预处理 - 深度图生成 (12-24小时)
    ✓ ScanNet: Depth-Anything-V2-ViT-L
    ✓ ScanNetPP: Depth-Anything-V2-ViT-L
    ✓ Structured3D: Depth-Anything-V2-ViT-L
    ✓ RXR: Depth-Anything-V2-ViT-L

  Phase 4: 预处理 - 位姿置信度生成 (6-12小时)
    ✓ ScanNet: VGGT-1B
    ✓ ScanNetPP: VGGT-1B
    ✓ Structured3D: VGGT-1B
    ✓ RXR: VGGT-1B

  Phase 5: 验证 (10分钟)
    ✓ 统计各数据集完成度

日志位置:
    总日志: $LOG_DIR/pipeline.log
    下载日志: $LOG_DIR/download.log
    深度日志: $LOG_DIR/depth_*.log
    位姿日志: $LOG_DIR/pose_*.log

查看日志:
    tail -f $LOG_DIR/pipeline.log

EOF
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    local mode="full"
    local do_monitor=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                mode="check"
                shift
                ;;
            --skip-download)
                mode="preprocess"
                shift
                ;;
            --monitor)
                do_monitor=true
                shift
                ;;
            --resume)
                export RESUME_DOWNLOAD=1
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # ========== 执行流程 ==========
    
    log_info "========== StreamSpatial-VLM 数据处理流程 =========="
    log_info "模式: $mode"
    log_info "PID: $$"
    
    # 环境检查
    check_environment
    check_disk_space
    
    # 仅检查状态
    if [ "$mode" = "check" ]; then
        check_status
        exit 0
    fi
    
    # 完整流程或下载模式
    if [ "$mode" = "full" ]; then
        log_info "启动完整数据处理流程..."
        log_info ""
        
        # 下载数据
        if ! download_spar7m_rgbd; then
            log_error "下载失败！"
            exit 1
        fi
        log_info ""
    fi
    
    # 预处理
    if [ "$mode" = "full" ] || [ "$mode" = "preprocess" ]; then
        log_info "启动数据预处理..."
        log_info ""
        
        preprocess_all_datasets
        log_info ""
    fi
    
    # 验证
    check_status
    
    log_success "=========== 处理流程完成 ==========="
    log_info "日志文件: $LOG_DIR/pipeline.log"
    
    # 监控模式
    if [ "$do_monitor" = true ]; then
        monitor_progress
    fi
}

# ============================================================================
main "$@"
