# StreamSpatial-VLM 毕设实施蓝图与填充手册

**学生**：谭政 (12212320)  
**目标**：构建基于 2D-3D 互引导的流式空间理解高效框架  
**硬性指标**：3× 加速、精度掉点 ≤2%、峰值显存 ≤ 基线 50%、推理零额外参数、零 3D 标注依赖

---

## Phase 0: 基础设施与数据准备（前置依赖）

> **目标**：在 4×RTX 5090 上搭建稳定环境，下载基准数据与预训练权重。

### Step 0.1: 5090 环境适配

**状态**：✅ 已完成

**已完成的修复**：
- [x] PyTorch 升级至 2.7.0+cu128（解决 sm_120 兼容性）
- [x] 设置环境变量 `export DS_DISABLE_COMM_OP_LIBS=1`（屏蔽 DeepSpeed 编译报错）
- [x] 替换所有 `flash_attention_2` 为 `eager`

**实现脚本**：
- **文件路径**：`setup.sh`
- **使用方法**：
  ```bash
  chmod +x setup.sh
  ./setup.sh
  ```

**脚本功能**：
1. 自动检测GPU架构（sm_120）
2. 配置必要的环境变量
3. 创建Conda环境 `streamspatial`
4. 安装PyTorch 2.7.0+cu128
5. 替换flash_attention_2为eager
6. 创建数据目录结构

### Step 0.2: 数据集与权重获取

#### SPAR-7M 下载脚本

**文件路径**：`scripts/download_resources.sh`

**下载命令**：
```bash
# 方式1: 使用HuggingFace CLI
pip install huggingface_hub
huggingface-cli download --repo-type dataset SPAR7M/SPAR-7M --local-dir data/raw/spar7m

# 方式2: 使用Python脚本（ SPAR-7M 精简版（图片+QA））
python download_datasets.py --dataset spar7m 


# 解压缩：
cd data/raw/spar7m

# 分别解压每个数据集的标注
tar -xvzf scannet.tar.gz
tar -xvzf scannetpp.tar.gz
tar -xvzf structured3d.tar.gz
tar -xvzf rxr.tar.gz

```

**存放路径**：`data/raw/spar7m/`

**数据集结构**：
```
data/raw/spar7m/
├── videos/
│   ├── scene_0001/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── ...
├── annotations/
│   ├── train.json
│   └── val.json
└── metadata.json
```

#### ScanQA 下载脚本

**下载命令**：
```bash
# 从官方GitHub获取
git clone https://github.com/ATR-DBI/ScanQA.git
# 将数据移动到指定目录
mv ScanQA/data/* data/raw/scanqa/
```

**存放路径**：`data/raw/scanqa/`

**数据集结构**：
```
data/raw/scanqa/
├── scans/
│   ├── scene0001_00/
│   └── ...
├── questions/
│   ├── ScanQA_v1.0_train.json
│   ├── ScanQA_v1.0_val.json
│   └── ScanQA_v1.0_test.json
└── metadata.json
```

#### 基线权重下载

**VG-LLM 权重**：
```bash
# 从HuggingFace下载
huggingface-cli download VG-LLM/VG-LLM-7B --local-dir checkpoints/VG-LLM/

# 或使用Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="VG-LLM/VG-LLM-7B", local_dir="checkpoints/VG-LLM")
```

**存放路径**：`checkpoints/VG-LLM/`

### Step 0.3: 离线 3D 几何先验生成（伪深度 & 伪位姿）

> **说明**：这是整个项目的燃料，后续的 3D→2D Zip 完全依赖这里生成的数据。

#### Depth-Anything-V2 推理脚本

**文件路径**：`scripts/generate_depth.py`

**实现代码**：
```python
"""
基于 Depth-Anything-V2 的批量深度估计脚本

输入：data/raw/spar7m/videos/
输出：data/processed/spar7m/depth_pred/*.npy
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# Depth-Anything-V2 导入
from depth_anything_v2.dpt import DepthAnythingV2


def load_depth_model(device='cuda'):
    """加载Depth-Anything-V2模型"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 2304, 3072, 3072]}
    }
    
    model = DepthAnythingV2(**model_configs['vitl'])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    model = model.to(device).eval()
    
    return model


def process_video_folder(
    input_dir: str,
    output_dir: str,
    model,
    device='cuda',
    image_size=(518, 518)
):
    """
    处理视频文件夹中的所有帧
    
    Args:
        input_dir: 输入视频帧目录
        output_dir: 输出深度图目录
        model: Depth-Anything-V2模型
        device: 计算设备
        image_size: 输入图像尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        # 加载图像
        img_path = os.path.join(input_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # 推理深度
        with torch.no_grad():
            depth = model.infer_image(image, image_size)  # [H, W]
        
        # 保存为numpy数组
        output_name = os.path.splitext(img_file)[0] + '.npy'
        output_path = os.path.join(output_dir, output_name)
        np.save(output_path, depth.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='Depth-Anything-V2批量推理')
    parser.add_argument('--input', type=str, default='data/raw/spar7m/videos',
                        help='输入视频目录')
    parser.add_argument('--output', type=str, default='data/processed/spar7m/depth_pred',
                        help='输出深度图目录')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    args = parser.parse_args()
    
    # 加载模型
    print("Loading Depth-Anything-V2 model...")
    model = load_depth_model(args.device)
    
    # 处理所有场景
    scenes = sorted(os.listdir(args.input))
    for scene in scenes:
        scene_input = os.path.join(args.input, scene)
        scene_output = os.path.join(args.output, scene)
        
        if os.path.isdir(scene_input):
            process_video_folder(scene_input, scene_output, model, args.device)
    
    print("Depth estimation completed!")


if __name__ == '__main__':
    main()
```

**运行命令**：
```bash
python scripts/generate_depth.py \
    --input data/raw/spar7m/videos \
    --output data/processed/spar7m/depth_pred \
    --device cuda:0
```

#### VGGT 位姿/置信度推理脚本

**文件路径**：`scripts/generate_pose.py`

**实现代码**：
```python
"""
基于 VGGT 的位姿和置信度估计脚本

输入：data/raw/spar7m/videos/
输出：data/processed/spar7m/pose_conf/*.npy
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def load_vggt_model(device='cuda'):
    """加载VGGT模型"""
    # VGGT模型加载逻辑
    from vggt.model import VGGT
    
    model = VGGT.from_pretrained('checkpoints/VGGT/')
    model = model.to(device).eval()
    
    return model


def estimate_pose_and_confidence(
    model,
    images: list,
    device='cuda'
) -> dict:
    """
    估计相机位姿和深度置信度
    
    Args:
        model: VGGT模型
        images: 图像列表 [PIL.Image]
        device: 计算设备
        
    Returns:
        dict: {
            'pose': [4, 4] 相机位姿矩阵,
            'confidence': float 深度置信度
        }
    """
    with torch.no_grad():
        # VGGT推理
        outputs = model(images)
        
        pose = outputs['camera_poses'].cpu().numpy()  # [N, 4, 4]
        confidence = outputs['depth_confidence'].cpu().numpy()  # [N]
        
    return {
        'pose': pose,
        'confidence': confidence
    }


def process_video_folder(
    input_dir: str,
    output_dir: str,
    model,
    device='cuda',
    window_size=8
):
    """
    处理视频文件夹
    
    Args:
        input_dir: 输入视频帧目录
        output_dir: 输出位姿目录
        model: VGGT模型
        device: 计算设备
        window_size: 滑动窗口大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    # 加载所有图像
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    
    # 批量推理
    results = estimate_pose_and_confidence(model, images, device)
    
    # 保存每帧结果
    for i, img_file in enumerate(image_files):
        output_name = os.path.splitext(img_file)[0] + '.npy'
        output_path = os.path.join(output_dir, output_name)
        
        frame_result = {
            'pose': results['pose'][i],
            'confidence': results['confidence'][i]
        }
        np.save(output_path, frame_result)


def main():
    parser = argparse.ArgumentParser(description='VGGT位姿估计')
    parser.add_argument('--input', type=str, default='data/raw/spar7m/videos',
                        help='输入视频目录')
    parser.add_argument('--output', type=str, default='data/processed/spar7m/pose_conf',
                        help='输出位姿目录')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    args = parser.parse_args()
    
    # 加载模型
    print("Loading VGGT model...")
    model = load_vggt_model(args.device)
    
    # 处理所有场景
    scenes = sorted(os.listdir(args.input))
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_input = os.path.join(args.input, scene)
        scene_output = os.path.join(args.output, scene)
        
        if os.path.isdir(scene_input):
            process_video_folder(scene_input, scene_output, model, args.device)
    
    print("Pose estimation completed!")


if __name__ == '__main__':
    main()
```

**运行命令**：
```bash
python scripts/generate_pose.py \
    --input data/raw/spar7m/videos \
    --output data/processed/spar7m/pose_conf \
    --device cuda:0
```

---

## Phase 1: 基线复现与冗余分析（动机论证）

> **目标**：定量证明现有方法（VG-LLM）又慢又费显存，且 Spatial-MLLM 没有真正解决问题。

### Step 1.1: VG-LLM 基线跑通与 Profiling

#### VG-LLM 推理封装代码

**文件路径**：`profiling/profile_vgllm.py`

**已实现功能**：
- 封装了 `VGLLMProfiler` 类
- 支持输入视频并返回答案、耗时、显存
- 使用 PyTorch Forward Hook 统计冗余

**使用方法**：
```python
from profiling.profile_vgllm import VGLLMProfiler

# 初始化profiler
profiler = VGLLMProfiler(
    model_path="checkpoints/VG-LLM",
    device="cuda"
)

# 运行profiling
results = profiler.profile_video(
    video_path="data/raw/spar7m/videos/scene_0001",
    question="What is the spatial relationship between the chair and the table?"
)

# 输出结果
print(f"端到端耗时: {results['total_time']:.2f}s")
print(f"VGGT耗时: {results['vggt_time']:.2f}s")
print(f"VGGT占比: {results['vggt_ratio']:.1%}")
print(f"峰值显存: {results['peak_memory']:.2f}GB")
print(f"VGGT调用次数: {results['vggt_calls']}")
print(f"每帧Token数: {results['tokens_per_frame']}")
print(f"LLM总Token数: {results['total_tokens']}")
```

#### Hook 脚本统计冗余

**实现代码**（已在 `profiling/profile_vgllm.py` 中）：
```python
def _register_hooks(self):
    """注册Forward Hook统计冗余"""
    
    def hook_vggt(module, input, output):
        self.stats['vggt_calls'] += 1
        self.stats['vggt_times'].append(time.time())
    
    def hook_vit(module, input, output):
        # 统计2D Patch Token数量
        if isinstance(output, tuple):
            tokens = output[0]
            self.stats['patch_tokens_per_frame'] = tokens.shape[1]
    
    def hook_llm_input(module, input, output):
        # 统计LLM接收的总Token数
        if isinstance(input, tuple) and len(input) > 0:
            self.stats['total_llm_tokens'] = input[0].shape[1]
    
    # 注册hooks
    self.model.vggt.register_forward_hook(hook_vggt)
    self.model.vit.register_forward_hook(hook_vit)
    self.model.llm.register_forward_hook(hook_llm_input)
```

#### Profiling 结果记录

**预期结果**（待实际运行后填写）：

| 指标 | VG-LLM基线 | 预期值 |
|------|-----------|--------|
| 单视频端到端耗时 | [待填写] | ~10s |
| VGGT占比耗时 | [待填写] | ~60% |
| 峰值显存 | [待填写] | ~40GB |
| VGGT调用次数 | [待填写] | 16次（每帧1次） |
| 每帧Token数 | [待填写] | 576 (24×24) |
| LLM总Token数 | [待填写] | ~9216 (576×16) |

**论文Motivation图表数据**：
- 图1: VGGT冗余调用示意图
- 图2: Token数量随帧数线性增长曲线
- 表1: 计算开销分解

### Step 1.2: Spatial-MLLM 数据获取

**文件路径**：`docs/baseline_comparison.md`

#### Spatial-MLLM 对比数据表

| 方法 | SPAR-7M Acc | ScanQA EM | 加速比 | 显存 | 核心策略 |
|------|-------------|-----------|--------|------|----------|
| VG-LLM | [待填写] | [待填写] | 1.0× | [待填写] | 无优化 |
| Spatial-MLLM | ~85% | ~70% | 2.0× | - | 帧数减半 |
| **Ours (Full)** | **目标≥83%** | **目标≥68%** | **目标≥3.0×** | **目标≤20GB** | Gate+Zip+Cache |

**论文话术准备**：
> "Spatial-MLLM 将帧数减少了 50%，但单帧 Token 仍为 O(HW)，计算瓶颈未触及。我们的方法通过几何引导压缩，将单帧 Token 数量降低到 O(k)（k为关键区域数量），从根本上解决了计算瓶颈问题。"

---

## Phase 2: 核心模块开发（创新点实现）

> **目标**：实现三大无参模块。**绝对禁止在此阶段引入 `nn.Parameter` 或依赖 GT 标注。**

### Step 2.1: 模块 A - 2D→3D Gate（语义门控）

**文件路径**：`models/gate_2d3d.py`

**实现状态**：✅ 已完成

**核心代码**：
```python
class SemanticGate2D3D:
    """
    2D→3D 语义门控模块
    
    功能：基于语义相似度判断是否需要触发3D网络
    无参数：不包含任何可学习参数
    """
    
    def __init__(self, threshold: float = 0.95):
        """
        初始化语义门控
        
        Args:
            threshold: 余弦相似度阈值，默认0.95
        """
        self.threshold = threshold
        self.history_cls = None
    
    def should_trigger_3d(
        self, 
        current_cls: torch.Tensor,
        history_cls: Optional[torch.Tensor] = None
    ) -> bool:
        """
        判断是否需要触发3D网络
        
        Args:
            current_cls: 当前帧的CLS Token [B, D]
            history_cls: 历史帧的CLS Token [B, D]
            
        Returns:
            bool: True表示需要触发3D网络，False表示跳过
        """
        if history_cls is None:
            return True  # 首帧必须触发
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            current_cls, 
            history_cls, 
            dim=-1
        )
        
        # 相似度低于阈值时触发3D网络
        return similarity.mean().item() < self.threshold
    
    def update_history(self, current_cls: torch.Tensor):
        """更新历史CLS Token"""
        self.history_cls = current_cls.detach().clone()
```

**超参数定义**：
- 初始阈值 τ = 0.95
- 可调范围：0.90 ~ 0.99

### Step 2.2: 模块 B - 3D→2D Zip（几何引导压缩）

**文件路径**：`models/zip_3d2d.py`

**实现状态**：✅ 已完成

**核心代码**：
```python
class GeometryGuidedZip:
    """
    3D→2D 几何引导压缩模块
    
    功能：基于深度方差进行Token压缩
    无参数：不包含任何可学习参数
    """
    
    def __init__(self, keep_ratio: float = 0.6):
        """
        初始化压缩模块
        
        Args:
            keep_ratio: Token保留比例，默认0.6
        """
        self.keep_ratio = keep_ratio
    
    def zip_tokens(
        self,
        depth_map: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_size: int = 14
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于深度方差压缩Token
        
        Args:
            depth_map: 深度图 [H, W]
            patch_tokens: Patch Token [N, D]
            patch_size: Patch大小
            
        Returns:
            zipped_tokens: 压缩后的Token [K, D]
            keep_indices: 保留的索引 [K]
        """
        H, W = depth_map.shape
        
        # 计算Patch数量
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # 将深度图切块
        patches = depth_map[:num_patches_h * patch_size, :num_patches_w * patch_size]
        patches = patches.reshape(num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(num_patches, patch_size * patch_size)
        
        # 计算每块深度方差
        variances = patches.var(dim=-1)  # [N]
        
        # 按方差排序，保留方差最大的Token
        k = int(num_patches * self.keep_ratio)
        _, keep_indices = torch.topk(variances, k)
        keep_indices = keep_indices.sort()[0]  # 保持空间顺序
        
        # 裁剪Token
        zipped_tokens = patch_tokens[keep_indices]
        
        return zipped_tokens, keep_indices
```

**超参数定义**：
- 初始保留率 r = 0.6
- 可调范围：0.3 ~ 1.0

### Step 2.3: 模块 C - 增量流式缓存

**文件路径**：`models/kv_cache.py`

**实现状态**：✅ 已完成

**核心代码**：
```python
class IncrementalKVCache:
    """
    增量流式KV缓存
    
    功能：维护滑动窗口的历史Token
    无参数：不包含任何可学习参数
    """
    
    def __init__(self, window_size: int = 4):
        """
        初始化缓存
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.cache_2d = deque(maxlen=window_size)
        self.cache_3d = deque(maxlen=window_size)
    
    def update(
        self,
        frame_2d_tokens: torch.Tensor,
        frame_3d_tokens: Optional[torch.Tensor] = None
    ):
        """
        更新缓存
        
        Args:
            frame_2d_tokens: 当前帧2D Token
            frame_3d_tokens: 当前帧3D Token（可选）
        """
        self.cache_2d.append(frame_2d_tokens.detach().clone())
        if frame_3d_tokens is not None:
            self.cache_3d.append(frame_3d_tokens.detach().clone())
    
    def get_history(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """获取历史Token"""
        history_2d = torch.cat(list(self.cache_2d), dim=0)
        history_3d = None
        if len(self.cache_3d) > 0:
            history_3d = torch.cat(list(self.cache_3d), dim=0)
        return history_2d, history_3d
    
    def get_memory_saving(self, base_tokens_per_frame: int) -> float:
        """
        计算显存节省比例
        
        Args:
            base_tokens_per_frame: 基线每帧Token数
            
        Returns:
            float: 节省比例
        """
        if len(self.cache_2d) == 0:
            return 0.0
        
        current_tokens = self.cache_2d[-1].shape[0]
        saved_tokens = base_tokens_per_frame - current_tokens
        return saved_tokens / base_tokens_per_frame
```

**超参数定义**：
- 滑动窗口大小 window_size = 4

---

## Phase 3: 系统集成（组装 StreamSpatial-VLM）

> **目标**：把 Phase 2 的模块像插件一样插进 VG-LLM 的前向传播中。

### Step 3.1: 改造 VG-LLM 的 Forward 流程

**文件路径**：`models/stream_spatial_vlm.py`

**实现状态**：✅ 已完成

**主流程代码**：
```python
class StreamSpatialVLM(nn.Module):
    """
    StreamSpatial-VLM 主模型
    
    集成三大无参模块：
    1. SemanticGate2D3D: 语义门控
    2. GeometryGuidedZip: 几何压缩
    3. IncrementalKVCache: 流式缓存
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 加载基线模型
        self.vit = load_vit(config['vit_path'])
        self.vggt = load_vggt(config['vggt_path'])
        self.llm = load_llm(config['llm_path'])
        
        # 初始化三大模块（无参数）
        self.gate = SemanticGate2D3D(threshold=config.get('gate_threshold', 0.95))
        self.zip_module = GeometryGuidedZip(keep_ratio=config.get('zip_ratio', 0.6))
        self.cache = IncrementalKVCache(window_size=config.get('window_size', 4))
        
        # 统计信息
        self.stats = {
            'vggt_calls': 0,
            'vggt_skips': 0,
            'tokens_saved': 0
        }
    
    def forward(
        self,
        video_frames: torch.Tensor,
        depth_maps: Optional[torch.Tensor] = None,
        question: str = ""
    ) -> Dict:
        """
        流式前向传播
        
        Args:
            video_frames: [T, C, H, W] 视频帧
            depth_maps: [T, H, W] 深度图（可选）
            question: 问题文本
            
        Returns:
            dict: 包含答案和统计信息
        """
        T = video_frames.shape[0]
        all_2d_tokens = []
        all_3d_tokens = []
        
        for t in range(T):
            frame = video_frames[t:t+1]
            
            # Step 1: 2D特征提取
            feat_2d = self.vit(frame)  # [1, N, D]
            current_cls = feat_2d[:, 0]  # CLS Token
            
            # Step 2: 语义门控判断
            if self.gate.should_trigger_3d(current_cls, self.gate.history_cls):
                # 触发3D网络
                feat_3d = self.vggt(frame)
                self.stats['vggt_calls'] += 1
            else:
                # 复用历史3D特征
                feat_3d = self.cache.get_latest_3d()
                self.stats['vggt_skips'] += 1
            
            # 更新门控历史
            self.gate.update_history(current_cls)
            
            # Step 3: 几何引导压缩
            if depth_maps is not None:
                depth = depth_maps[t]
                feat_2d_zipped, _ = self.zip_module.zip_tokens(depth, feat_2d[0])
                feat_2d_zipped = feat_2d_zipped.unsqueeze(0)
            else:
                feat_2d_zipped = feat_2d
            
            # Step 4: 更新缓存
            self.cache.update(feat_2d_zipped, feat_3d)
            
            all_2d_tokens.append(feat_2d_zipped)
            all_3d_tokens.append(feat_3d)
        
        # Step 5: 获取历史Token并送入LLM
        history_2d, history_3d = self.cache.get_history()
        all_tokens = torch.cat([history_2d, history_3d], dim=0)
        
        # Step 6: LLM生成答案
        answer = self.llm.generate(all_tokens, question)
        
        return {
            'answer': answer,
            'stats': self.stats.copy()
        }
```

### Step 3.2: 离线数据加载适配

**文件路径**：`data/dataset_loaders/`

**已实现的数据加载器**：
- `spar7m.py`: SPAR-7M数据集加载器
- `scanqa.py`: ScanQA数据集加载器
- `scanrefer.py`: ScanRefer数据集加载器

**使用方法**：
```python
from data.dataset_loaders import SPAR7MDataset, ScanQADataset

# 加载SPAR-7M数据集
spar7m = SPAR7MDataset(
    data_root="data/raw/spar7m",
    processed_root="data/processed/spar7m",
    split="val",
    load_depth=True,
    load_pose=True
)

# 获取样本
sample = spar7m[0]
print(f"Images: {sample['images'].shape}")  # [16, 3, 336, 336]
print(f"Depth maps: {sample['depth_maps'].shape}")  # [16, 336, 336]
print(f"Poses: {sample['poses'].shape}")  # [16, 4, 4]
```

---

## Phase 4: 实验验证与消融（证明有效）

> **目标**：产出论文核心表格，证明满足任务书的硬性指标。

### Step 4.1: 主实验（对比基线）

#### SPAR-7M 评估脚本

**文件路径**：`eval/eval_spar7m.py`

**使用方法**：
```bash
python eval/eval_spar7m.py \
    --config configs/streamspatial_default.yaml \
    --data_root data/raw/spar7m \
    --processed_root data/processed/spar7m \
    --output results/spar7m_results.json
```

#### ScanQA 评估脚本

**文件路径**：`eval/eval_scanqa.py`

**使用方法**：
```bash
python eval/eval_scanqa.py \
    --config configs/streamspatial_default.yaml \
    --data_root data/raw/scanqa \
    --output results/scanqa_results.json
```

#### 结果记录表 1（主表）

| 方法 | SPAR-7M Acc | ScanQA EM | 端到端耗时 | 峰值显存 (GB) |
|:---|:---|:---|:---|:---|
| VG-LLM (Base) | [待填写] | [待填写] | [待填写] | [待填写] |
| Spatial-MLLM | ~85% | ~70% | 2.0× | - |
| **Ours (Full)** | **[待填写]** | **[待填写]** | **[待填写]** | **[待填写]** |

**硬性指标验证**：
- [ ] 3× 加速：端到端耗时 ≤ 基线/3
- [ ] 精度掉点 ≤2%：Acc ≥ 基线 - 2%
- [ ] 峰值显存 ≤ 基线 50%：显存 ≤ 基线 × 0.5
- [ ] 推理零额外参数：已验证（三大模块均无 nn.Parameter）
- [ ] 零 3D 标注依赖：已验证（仅使用伪深度和伪位姿）

### Step 4.2: 消融实验（证明每个模块都有用）

**文件路径**：`scripts/run_ablation.py`

**运行命令**：
```bash
python scripts/run_ablation.py \
    --config_dir configs/ablation/ \
    --data_root data/raw/spar7m \
    --output results/ablation_results.json
```

#### 结果记录表 2（消融表）

| 配置 | Gate | Zip | Cache | SPAR-7M Acc | 耗时 | 显存 |
|:---|:---:|:---:|:---:|:---|:---|:---|
| (a) Base | ✗ | ✗ | ✗ | [待填写] | [待填写] | [待填写] |
| (b) Base + Gate | ✓ | ✗ | ✗ | [待填写] | [待填写] | [待填写] |
| (c) Base + Zip | ✗ | ✓ | ✗ | [待填写] | [待填写] | [待填写] |
| (d) Base + Gate + Zip | ✓ | ✓ | ✗ | [待填写] | [待填写] | [待填写] |
| (e) Full (Gate+Zip+Cache) | ✓ | ✓ | ✓ | [待填写] | [待填写] | [待填写] |

### Step 4.3: 敏感性分析（超参调优）

#### Gate 阈值 τ 曲线图

**脚本**：`scripts/sensitivity_analysis.py`

```bash
python scripts/sensitivity_analysis.py \
    --param gate_threshold \
    --range 0.90 0.99 0.01 \
    --output results/gate_sensitivity.png
```

**预期结果**：
- X轴：τ (0.90~0.99)
- Y轴：Acc 和 VGGT触发率
- 最优值：τ ≈ 0.95

#### Zip 保留率 r 曲线图

```bash
python scripts/sensitivity_analysis.py \
    --param zip_ratio \
    --range 0.3 1.0 0.1 \
    --output results/zip_sensitivity.png
```

**预期结果**：
- X轴：r (0.3~1.0)
- Y轴：Acc 和 Token数量
- 最优值：r ≈ 0.6

---

## Phase 5: 成果产出（答辩准备）

> **目标**：满足任务书要求的"代码开源、论文、Demo"。

### Step 5.1: 实时流式 Demo

**目录**：`demo/`

**文件路径**：`demo/streaming_demo.py`

**实现状态**：✅ 已完成

**功能**：
- 左侧：Webcam/本地视频逐帧输入
- 右上：状态监控面板（当前帧是否触发 3D 网络？压缩率多少？缓存队列长度？）
- 右下：对话框（实时回答空间问题）

**运行方法**：
```bash
python demo/streaming_demo.py \
    --config configs/streamspatial_default.yaml \
    --port 7860
```

### Step 5.2: 仓库文档完善

#### README.md

**文件路径**：`README.md`

**已包含内容**：
- 项目简介
- Installation（安装指南）
- Quick Start（快速开始）
- Evaluation（评估方法）
- Demo 运行方法
- Citation（引用格式）

#### requirements.txt

**文件路径**：`requirements.txt`

**已锁定版本**：
```
torch==2.7.0
transformers>=4.40.0
accelerate>=0.30.0
gradio>=4.0.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
huggingface_hub>=0.20.0
```

### Step 5.3: 论文与答辩 PPT

#### 论文各章节素材映射

| 章节 | 对应内容 | 文件路径 |
|:---|:---|:---|
| 第一章 引言 | Phase 1 的 Profiling 数据 | `profiling/results/` |
| 第三章 方法 | Phase 2 的三个模块架构图 | `docs/technical-design.md` |
| 第四章 实验 | Phase 4 的三张表格/曲线图 | `results/` |
| 第五章 总结 | Phase 5.1 的 Demo 截图 | `demo/screenshots/` |

---

## 附录：项目文件结构

```
StreamSpatial-VLM/
├── configs/
│   ├── streamspatial_default.yaml    # 主配置文件
│   ├── baseline_vgllm.yaml           # 基线配置
│   └── ablation/                     # 消融实验配置
│       ├── ablation_A.yaml           # 基线
│       ├── ablation_B.yaml           # +Gate
│       ├── ablation_C.yaml           # +Zip
│       ├── ablation_D.yaml           # +Gate+Zip
│       └── ablation_E.yaml           # Full
├── models/
│   ├── gate_2d3d.py                  # 语义门控模块
│   ├── zip_3d2d.py                   # 几何压缩模块
│   ├── kv_cache.py                   # 流式缓存模块
│   └── stream_spatial_vlm.py         # 主模型
├── data/
│   ├── dataset_loaders/
│   │   ├── spar7m.py                 # SPAR-7M加载器
│   │   ├── scanqa.py                 # ScanQA加载器
│   │   └── scanrefer.py              # ScanRefer加载器
│   ├── raw/                          # 原始数据
│   └── processed/                    # 处理后数据
├── eval/
│   ├── eval_spar7m.py                # SPAR-7M评估
│   ├── eval_scanqa.py                # ScanQA评估
│   ├── eval_scanrefer.py             # ScanRefer评估
│   └── metrics.py                    # 评估指标
├── profiling/
│   └── profile_vgllm.py              # VG-LLM性能分析
├── scripts/
│   ├── run_inference.sh              # 推理脚本
│   ├── run_eval.sh                   # 评估脚本
│   ├── run_ablation.sh               # 消融实验脚本
│   ├── run_ablation.py               # 消融实验Python脚本
│   ├── generate_depth.py             # 深度估计脚本
│   ├── generate_pose.py              # 位姿估计脚本
│   └── download_resources.sh         # 资源下载脚本
├── demo/
│   └── streaming_demo.py             # Gradio演示
├── utils/
│   ├── config_loader.py              # 配置加载器
│   ├── speed_profiler.py             # 速度分析工具
│   └── visualizer.py                 # 可视化工具
├── docs/
│   ├── technical-design.md           # 技术设计文档
│   └── project-proposal.md           # 项目提案
├── setup.sh                     # RTX 5090环境配置
├── requirements.txt                  # 依赖列表
├── download_datasets.py              # 数据集下载脚本
├── PLAN.md                           # 本文档
└── README.md                         # 项目说明
```

---

## 💡 如何配合 AI Agent 使用这份文档

1. **不要一次性扔给 Agent**：每次只打开一个 `Step`，把 `[待补充]` 的内容和下方的"Agent 提示词参考"发给它。

2. **保持上下文隔离**：在让 Agent 写 Phase 2 的代码时，不要把 Phase 1 的环境配置贴给它，容易导致它幻觉。

3. **先要伪代码，再要实现**：对于 Phase 3.1 这种复杂的集成，先让 Agent 输出"伪代码逻辑"（就像我写的那样），确认数据流对了，再让它写真实的 PyTorch 代码。

---

**文档版本**：v1.0  
**最后更新**：2025年  
**维护者**：谭政 (12212320)
