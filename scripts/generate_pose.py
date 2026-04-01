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