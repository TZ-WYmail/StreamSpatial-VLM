#!/bin/bash
# ============================================================================
# StreamSpatial-VLM 一键启动脚本
# ============================================================================
# 用法: bash start_pipeline.sh [OPTIONS]
# 选项: --check, --download, --monitor, --help
# ============================================================================

PROJECT_ROOT="/home/tanzheng/Desktop/myproject/StreamSpatial-VLM"
CONDA_ENV="streamspatial"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 帮助信息
if [[ "$1" == "--help" ]]; then
    cat << 'EOF'
StreamSpatial-VLM 数据处理流程一键启动脚本

用法: bash start_pipeline.sh [OPTIONS]

选项:
  (无选项)        完整流程（下载 + 预处理）
  --check         仅检查状态
  --download      仅下载数据
  --monitor       启动监控面板
  --help          显示此帮助

快速命令:

1. 完整流程 - 后台运行（推荐）:
   bash start_pipeline.sh &
   python scripts/launch_pipeline.py --monitor

2. 仅检查状态:
   bash start_pipeline.sh --check

3. 启动监控:
   python scripts/launch_pipeline.py --monitor

4. 查看日志:
   tail -f results/logs/pipeline.log

预计耗时:
  - 下载: 24-48 小时
  - 预处理: 18-36 小时
  - 总计: 42-84 小时

数据量:
  - 总数据: 200GB+
  - 深度图: 100GB
  - 位姿: 100GB

磁盘需求: >250GB 可用空间

EOF
    exit 0
fi

# 进入项目目录
cd "$PROJECT_ROOT" || exit 1

# 选择执行模式
MODE=${1:-full}

# ============================================================================
case "$MODE" in
    --check)
        echo -e "${BLUE}📊 检查数据集状态...${NC}"
        conda run -n "$CONDA_ENV" python scripts/launch_pipeline.py --mode check
        ;;
    --download)
        echo -e "${BLUE}📥 启动数据下载...${NC}"
        echo -e "${YELLOW}预计耗时: 24-48 小时${NC}"
        echo ""
        nohup conda run -n "$CONDA_ENV" python scripts/launch_pipeline.py --mode download > results/logs/download_bg.log 2>&1 &
        sleep 2
        echo -e "${GREEN}✓ 加载中...${NC}"
        tail -f results/logs/download_bg.log &
        TAIL_PID=$!
        sleep 5
        kill $TAIL_PID 2>/dev/null
        echo ""
        echo -e "${GREEN}✓ 下载已在后台启动${NC}"
        echo -e "${YELLOW}查看进度: tail -f results/logs/download_bg.log${NC}"
        ;;
    --monitor)
        echo -e "${BLUE}📊 启动监控面板...${NC}"
        conda run -n "$CONDA_ENV" python scripts/launch_pipeline.py --monitor
        ;;
    *)
        echo -e "${BLUE}🚀 启动完整数据处理流程${NC}"
        echo -e "${YELLOW}预计耗时: 42-84 小时${NC}"
        echo ""
        
        # 检查环境
        echo -e "${BLUE}[1/3] 检查环境...${NC}"
        conda run -n "$CONDA_ENV" python scripts/launch_pipeline.py --mode check >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ 环境检查通过${NC}"
        else
            echo -e "${YELLOW}⚠ 环境检查失败${NC}"
            exit 1
        fi
        
        # 启动后台处理
        echo -e "${BLUE}[2/3] 启动后台处理...${NC}"
        nohup conda run -n "$CONDA_ENV" python scripts/launch_pipeline.py --mode full > results/logs/full_pipeline.log 2>&1 &
        PID=$!
        echo -e "${GREEN}✓ 进程已启动 (PID: $PID)${NC}"
        
        # 保存 PID
        echo "$PID" > "$PROJECT_ROOT/results/logs/pipeline.pid"
        
        sleep 2
        
        # 显示日志摘要
        echo ""
        echo -e "${BLUE}[3/3] 显示初始日志...${NC}"
        sleep 1
        echo ""
        tail -n 10 results/logs/full_pipeline.log
        
        echo ""
        echo -e "${GREEN}════════════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ 数据处理流程已启动${NC}"
        echo -e "${GREEN}════════════════════════════════════════════${NC}"
        echo ""
        echo "📊 查看进度:"
        echo "   tail -f results/logs/pipeline.log"
        echo ""
        echo "📈 启动监控面板:"
        echo "   python scripts/launch_pipeline.py --monitor"
        echo ""
        echo "📋 检查状态:"
        echo "   python scripts/launch_pipeline.py --mode check"
        echo ""
        echo "⏹ 停止进程:"
        echo "   kill $PID"
        echo ""
        ;;
esac
