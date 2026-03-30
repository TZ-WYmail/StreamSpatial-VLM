"""
ScanRefer 零样本评估脚本
=========================
评估指标：Acc@0.25 IoU, Acc@0.5 IoU

用法：
    python eval/eval_scanrefer.py \
        --config configs/streamspatial_default.yaml \
        --data_root /data/scanrefer \
        --output_dir results/scanrefer
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loaders.scanrefer import ScanReferDataset
from eval.metrics import compute_acc_at_iou
from utils.config_loader import load_config
from utils.speed_profiler import SpeedProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScanRefer 零样本评估")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/scanrefer")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """
    从模型输出文本中解析 3D 边界框坐标。
    期望格式：[x, y, z, dx, dy, dz] 或 "center: (x, y, z), size: (dx, dy, dz)"
    """
    # 尝试匹配 6 个数字
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numbers) >= 6:
        try:
            return [float(n) for n in numbers[:6]]
        except ValueError:
            pass
    return None


def evaluate_scanrefer(
    model,
    data_root: str,
    split: str = "val",
    max_samples: int = -1,
    device: str = "cuda",
) -> Dict[str, Any]:
    """ScanRefer 评估主函数"""
    dataset = ScanReferDataset(data_root=data_root, split=split)
    if max_samples > 0:
        dataset.annotations = dataset.annotations[:max_samples]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=ScanReferDataset.collate_fn,
    )

    pred_bboxes = []
    gt_bboxes = []
    profiler = SpeedProfiler(device=device)

    for batch in tqdm(loader, desc="ScanRefer 评估"):
        descriptions = batch["descriptions"]
        gt_bbox_list = batch["gt_bboxes"]
        frames_list = batch["frames"]
        depth_list = batch["depth_maps"]
        pose_list = batch["pose_confs"]

        for i in range(len(descriptions)):
            frames = frames_list[i].to(device)
            depth_maps = depth_list[i]
            pose_confs = pose_list[i]

            # 构造定位 prompt
            query = (
                f"Please locate the object described as: '{descriptions[i]}'. "
                "Output the 3D bounding box as [cx, cy, cz, dx, dy, dz]."
            )

            model.reset()
            profiler.start()

            T = frames.shape[0]
            for t in range(T):
                depth_t = depth_maps[t].to(device) if depth_maps is not None else None
                pose_t = pose_confs[t].to(device) if pose_confs is not None else None
                model.process_frame(frames[t], depth_t, pose_t, frame_idx=t)

            pred_text = model.answer(query)
            profiler.stop()

            pred_bbox = parse_bbox_from_text(pred_text)
            gt_bbox = gt_bbox_list[i]
            if gt_bbox is not None and isinstance(gt_bbox, torch.Tensor):
                gt_bbox = gt_bbox.tolist()

            pred_bboxes.append(pred_bbox)
            gt_bboxes.append(gt_bbox)

    # 计算 Acc@IoU
    acc_results = compute_acc_at_iou(pred_bboxes, gt_bboxes, [0.25, 0.5])

    results = {
        "dataset": "ScanRefer",
        "split": split,
        "num_samples": len(pred_bboxes),
        "acc@0.25": acc_results.get("acc@0.25", 0.0),
        "acc@0.5": acc_results.get("acc@0.5", 0.0),
        "fps": profiler.fps,
        "peak_memory_gb": profiler.peak_memory_gb,
    }

    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from models.stream_spatial_vlm import StreamSpatialVLM, StreamSpatialConfig
    model_cfg = StreamSpatialConfig(**cfg.get("model", {}))
    model_cfg.device = args.device
    model = StreamSpatialVLM(model_cfg)
    model.load_models()

    results = evaluate_scanrefer(
        model=model,
        data_root=args.data_root,
        split=args.split,
        max_samples=args.max_samples,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nScanRefer 评估结果:")
    for k, v in results.items():
        print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
