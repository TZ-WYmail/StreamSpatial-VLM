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