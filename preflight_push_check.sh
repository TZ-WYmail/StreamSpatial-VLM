#!/bin/bash
# ============================================================================
# StreamSpatial-VLM 推送前最终验证脚本
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  StreamSpatial-VLM 推送前最终检查                                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

CHECKS_PASSED=0
CHECKS_FAILED=0

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查函数
check() {
    local name=$1
    local cmd=$2
    local expected=$3
    
    echo -n "检查: $name ... "
    
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}✅${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "${RED}❌${NC}"
        ((CHECKS_FAILED++))
    fi
}

# ============ 检查集 1: 关键源文件 ============
echo -e "${BLUE}[1] 关键源文件${NC}"
check "models/gate_2d3d.py" "test -f models/gate_2d3d.py"
check "models/zip_3d2d.py" "test -f models/zip_3d2d.py"
check "models/kv_cache.py" "test -f models/kv_cache.py"
check "models/stream_spatial_vlm.py" "test -f models/stream_spatial_vlm.py"
check "scripts/run_ablation.py" "test -f scripts/run_ablation.py"
check "test_modules.py" "test -f test_modules.py"
echo

# ============ 检查集 2: 文档 ============
echo -e "${BLUE}[2] 文档完整性${NC}"
check "README.md" "test -f README.md"
check "PROJECT_STATUS.md" "test -f PROJECT_STATUS.md"
check "COMMANDS.md" "test -f COMMANDS.md"
check "PUSH_GUIDE.md" "test -f PUSH_GUIDE.md"
check "plan.md" "test -f plan.md"
echo

# ============ 检查集 3: 数据状态 ============
echo -e "${BLUE}[3] 数据状态${NC}"
check "ScanNet 目录存在" "test -d data/raw/spar7m/spar/scannet/images"
check "ScanNet RGB 图像" "[ \$(find data/raw/spar7m/spar/scannet/images -name '*.jpg' 2>/dev/null | wc -l) -ge 5000 ]"
check "ScanNet 深度图" "[ \$(find data/raw/spar7m/spar/scannet/images -name '*.png' 2>/dev/null | wc -l) -ge 5000 ]"
echo

# ============ 检查集 4: .gitignore ============
echo -e "${BLUE}[4] .gitignore 配置${NC}"
check "models_cache 被排除" "grep -q 'models_cache' .gitignore"
check "data 被排除" "grep -q '^data/' .gitignore"
check "results 被排除" "grep -q '^results/' .gitignore"
check ".gitignore 文件存在" "test -f .gitignore"
echo

# ============ 检查集 5: Python 环境 ============
echo -e "${BLUE}[5] Python 环境${NC}"
check "Python 可用" "python --version &>/dev/null"
check "PyTorch 可用" "python -c 'import torch' &>/dev/null"
check "torch.cuda 可用" "python -c 'import torch; assert torch.cuda.is_available()' &>/dev/null"
echo

# ============ 检查集 6: 敏感信息 ============
echo -e "${BLUE}[6] 敏感信息检查${NC}"

# 检查明显的 token 赋值
if grep -r "hf_[a-zA-Z0-9]\{40,\}" --include="*.py" --include="*.sh" . 2>/dev/null | grep -v "# " | grep -q .; then
    echo -ne "未发现 HuggingFace token 赋值 ... "
    echo -e "${RED}❌ 发现可疑 token！${NC}"
    ((CHECKS_FAILED++))
else
    echo -ne "未发现 HuggingFace token 赋值 ... "
    echo -e "${GREEN}✅${NC}"
    ((CHECKS_PASSED++))
fi

# 简化密钥检查 - 只检查硬编码的值
if grep -r "HUGGING_FACE_TOKEN\s*=" --include="*.py" . 2>/dev/null | grep -v "# " | grep -q .; then
    echo -ne "未发现硬编码的 API 密钥 ... "
    echo -e "${RED}❌ 发现可疑密钥！${NC}"
    ((CHECKS_FAILED++))
else
    echo -ne "未发现硬编码的 API 密钥 ... "
    echo -e "${GREEN}✅${NC}"
    ((CHECKS_PASSED++))
fi

echo

# ============ 检查集 7: 代码质量 ============
echo -e "${BLUE}[7] 代码质量检查${NC}"
check "Python 语法 (*.py 文件)" "python -m py_compile models/*.py scripts/*.py test_modules.py 2>/dev/null"
check "关键修复已应用 (sys.path)" "grep -q 'sys.path.insert' scripts/run_ablation.py"
check "权重加载修复已应用" "grep -q 'ignore_mismatched_sizes=True' models/stream_spatial_vlm.py"
check "dataclass 配置已使用" "grep -q '@dataclass' models/*.py"
echo

# ============ 检查集 8: Git 状态 ============
echo -e "${BLUE}[8] Git 状态${NC}"
check "Git 仓库初始化" "test -d .git"
check "Master/Main 分支存在" "git rev-parse --verify main &>/dev/null || git rev-parse --verify master &>/dev/null"
check ".gitignore 存在" "test -f .gitignore"
echo

# ============ 检查集 9: 大文件排除 ============
echo -e "${BLUE}[9] 大文件排除验证${NC}"

# 检查 models_cache 是否被追踪
if git ls-files models_cache 2>/dev/null | head -1 | grep -q models_cache; then
    echo -ne "models_cache 未追踪 ... "
    echo -e "${RED}❌${NC}"
    ((CHECKS_FAILED++))
else
    echo -ne "models_cache 未追踪 ... "
    echo -e "${GREEN}✅${NC}"
    ((CHECKS_PASSED++))
fi

# 检查 data/raw 是否被追踪
if git ls-files data/raw 2>/dev/null | head -1 | grep -q /; then
    echo -ne "data/raw 未追踪 ... "
    echo -e "${RED}❌${NC}"
    ((CHECKS_FAILED++))
else
    echo -ne "data/raw 未追踪 ... "
    echo -e "${GREEN}✅${NC}"
    ((CHECKS_PASSED++))
fi

echo

# ============ 最终统计 ============
echo "╔════════════════════════════════════════════════════════════════════╗"
echo -n "║  检查结果: "
if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}全部通过！${NC}"
    echo "║"
    echo "║  可以放心推送到远程 ✓"
else
    echo -e "${RED}有 $CHECKS_FAILED 项失败${NC}"
    echo "║"
    echo "║  请先修复失败项再推送"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"

# 返回码
[ $CHECKS_FAILED -eq 0 ] && exit 0 || exit 1
