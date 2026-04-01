#!/bin/bash
# ============================================================================
# StreamSpatial-VLM RTX 5090 环境配置脚本
# 
# 功能：一键配置RTX 5090 (Blackwell架构 sm_120) 所需的环境
# 作者：谭政 (12212320)
# 日期：2025年
# ============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "StreamSpatial-VLM RTX 5090 环境配置"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Step 1: 检查CUDA和GPU
# ============================================================================
print_info "Step 1: 检查GPU和CUDA环境..."

# 检查nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    print_error "未找到nvidia-smi，请确保已安装NVIDIA驱动"
    exit 1
fi

# 显示GPU信息
print_info "检测到的GPU："
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# 检查是否为RTX 5090 (sm_120)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
print_info "GPU计算能力: $GPU_ARCH"

if [[ "$GPU_ARCH" == "12.0" ]]; then
    print_info "检测到RTX 5090 (Blackwell架构 sm_120)"
fi

# ============================================================================
# Step 2: 配置环境变量
# ============================================================================
print_info "Step 2: 配置环境变量..."

# DeepSpeed 兼容性修复
export DS_DISABLE_COMM_OP_LIBS=1
print_info "已设置 DS_DISABLE_COMM_OP_LIBS=1 (屏蔽DeepSpeed编译报错)"

# CUDA相关
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1

# PyTorch相关
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 将环境变量写入 ~/.bashrc (可选)
read -p "是否将环境变量写入 ~/.bashrc? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "" >> ~/.bashrc
    echo "# StreamSpatial-VLM RTX 5090 环境变量" >> ~/.bashrc
    echo "export DS_DISABLE_COMM_OP_LIBS=1" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=0,1,2,3" >> ~/.bashrc
    echo "export TORCH_CUDA_ARCH_LIST=\"12.0\"" >> ~/.bashrc
    print_info "环境变量已写入 ~/.bashrc"
fi

# ============================================================================
# Step 3: 创建/激活Conda环境
# ============================================================================
print_info "Step 3: 配置Conda环境..."

ENV_NAME="streamspatial"

# 检查conda是否存在
if ! command -v conda &> /dev/null; then
    print_error "未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 检查环境是否存在
if conda env list | grep -q "^${ENV_NAME} "; then
    print_info "Conda环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
        print_info "已删除旧环境"
    else
        print_info "跳过环境创建，使用现有环境"
        conda activate ${ENV_NAME}
        # 跳转到Step 5
        print_info "环境激活完成，跳过安装步骤"
        echo "source ~/.bashrc 或重新打开终端以使环境变量生效"
        exit 0
    fi
fi

# 创建新环境
print_info "创建新的Conda环境: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# ============================================================================
# Step 4: 安装PyTorch和依赖
# ============================================================================
print_info "Step 4: 安装PyTorch 2.7.0+cu128..."

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装PyTorch (CUDA 12.8版本，支持sm_120)
print_info "安装PyTorch 2.7.0+cu128..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证PyTorch安装
print_info "验证PyTorch安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU计算能力: {torch.cuda.get_device_capability(0)}')
"

# ============================================================================
# Step 5: 安装项目依赖
# ============================================================================
print_info "Step 5: 安装项目依赖..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}

# 安装requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "安装requirements.txt..."
    pip install -r requirements.txt
else
    print_warn "未找到requirements.txt，安装基础依赖..."
    pip install transformers accelerate datasets huggingface_hub
    pip install gradio opencv-python pillow numpy scipy
    pip install matplotlib seaborn tqdm
fi

# ============================================================================
# Step 6: 替换flash_attention_2为eager
# ============================================================================
print_info "Step 6: 替换flash_attention_2为eager..."

# 查找所有包含flash_attention_2的Python文件
find . -name "*.py" -type f | while read file; do
    if grep -q "flash_attention_2" "$file"; then
        print_info "修改文件: $file"
        sed -i 's/flash_attention_2/eager/g' "$file"
    fi
done

print_info "flash_attention_2 替换完成"

# ============================================================================
# Step 7: 创建必要的目录结构
# ============================================================================
print_info "Step 7: 创建目录结构..."

mkdir -p data/raw/spar7m
mkdir -p data/raw/scanqa
mkdir -p data/raw/scanrefer
mkdir -p data/processed/spar7m/depth_pred
mkdir -p data/processed/spar7m/pose_conf
mkdir -p data/processed/scanqa/depth_pred
mkdir -p data/processed/scanqa/pose_conf
mkdir -p checkpoints/VG-LLM
mkdir -p logs
mkdir -p outputs

print_info "目录结构创建完成"

# ============================================================================
# Step 8: 验证安装
# ============================================================================
print_info "Step 8: 验证安装..."

python -c "
import torch
import transformers

print('=' * 50)
print('环境验证结果')
print('=' * 50)
print(f'Python版本: {__import__(\"sys\").version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'cuDNN版本: {torch.backends.cudnn.version()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        cap = torch.cuda.get_device_capability(i)
        print(f'  计算能力: {cap[0]}.{cap[1]}')
        print(f'  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')

print('=' * 50)
print('环境配置完成!')
print('=' * 50)
"

# ============================================================================
# 完成
# ============================================================================
echo ""
print_info "=========================================="
print_info "RTX 5090 环境配置完成!"
print_info "=========================================="
echo ""
print_info "后续步骤:"
print_info "1. 激活环境: conda activate ${ENV_NAME}"
print_info "2. 下载数据集: python download_datasets.py"
print_info "3. 下载预训练权重到 checkpoints/VG-LLM/"
print_info "4. 运行测试: python -m pytest tests/"
echo ""
print_info "注意事项:"
print_warn "- 每次使用前确保激活环境: conda activate ${ENV_NAME}"
print_warn "- 如遇到DeepSpeed问题，确认已设置: export DS_DISABLE_COMM_OP_LIBS=1"
print_warn "- flash_attention_2已替换为eager模式"
echo ""
