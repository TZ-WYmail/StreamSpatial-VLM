"""
SPAR-7M 评估脚本
=================
在 SPAR-7M val split 上评估 StreamSpatial-VLM 的空间问答准确率，
同时记录推理速度（FPS）和峰值显存。

用法：
    python eval/eval_spar7m.py \
        --config configs/streamspatial_default.yaml \
        --data_root /data/spar7m \
        --output_dir results/spar7m \
        --batch_size 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loaders.spar7m import SPAR7MDataset
from eval.metrics import compute_accuracy, compute_exact_match
from utils.config_loader import load_config
from utils.speed_profiler import SpeedProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPAR-7M 评估")
    parser.add_argument("--config", type=str, required=True,
                        help="模型配置文件路径（YAML）")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/spar7m")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--modality", type=str, default="video",
                        choices=["single", "multi", "video", "all"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大评估样本数，-1 表示全量")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate_spar7m(
    model,
    data_root: str,
    split: str = "val",
    modality: str = "video",
    max_samples: int = -1,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    SPAR-7M 评估主函数。

    Returns:
        results: 包含 accuracy、em、fps、peak_memory_gb 等指标
    """
    dataset = SPAR7MDataset(
        data_root=data_root,
        split=split,
        modality=modality,
    )

    if max_samples > 0:
        dataset.annotations = dataset.annotations[:max_samples]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=SPAR7MDataset.collate_fn,
    )

    predictions = []
    references = []
    profiler = SpeedProfiler(device=device)

    model.eval() if hasattr(model, "eval") else None

    for batch in tqdm(loader, desc="SPAR-7M 评估"):
        questions = batch["questions"]
        answers = batch["answers"]
        frames_list = batch["frames"]
        depth_list = batch["depth_maps"]
        pose_list = batch["pose_confs"]

        for i in range(len(questions)):
            frames = frames_list[i].to(device)      # (T, 3, H, W)
            depth_maps = depth_list[i]
            pose_confs = pose_list[i]
            question = questions[i]
            gt_answer = answers[i]

            # 重置模型状态
            model.reset()
            profiler.start()

            # 逐帧处理
            T = frames.shape[0]
            for t in range(T):
                depth_t = depth_maps[t].to(device) if depth_maps is not None else None
                pose_t = pose_confs[t].to(device) if pose_confs is not None else None
                model.process_frame(frames[t], depth_t, pose_t, frame_idx=t)

            # 生成答案
            pred = model.answer(question)
            profiler.stop()

            predictions.append(pred)
            references.append(gt_answer if isinstance(gt_answer, list) else [gt_answer])

    # 计算指标
    accuracy = compute_accuracy(predictions, references)
    em = compute_exact_match(predictions, references)
    fps = profiler.fps
    peak_mem = profiler.peak_memory_gb

    results = {
        "dataset": "SPAR-7M",
        "split": split,
        "modality": modality,
        "num_samples": len(predictions),
        "accuracy": accuracy,
        "exact_match": em,
        "fps": fps,
        "peak_memory_gb": peak_mem,
        "gate_trigger_rate": model.gate.trigger_rate if hasattr(model, "gate") else None,
        "zip_compression_ratio": model.zipper.mean_compression_ratio if hasattr(model, "zipper") else None,
    }

    return results


def main():
    args = parse_args()

    # 加载配置和模型
    cfg = load_config(args.config)
    from models.stream_spatial_vlm import StreamSpatialVLM, StreamSpatialConfig
    model_cfg = StreamSpatialConfig(**cfg.get("model", {}))
    model_cfg.device = args.device
    model = StreamSpatialVLM(model_cfg)
    model.load_models()

    # 评估
    results = evaluate_spar7m(
        model=model,
        data_root=args.data_root,
        split=args.split,
        modality=args.modality,
        max_samples=args.max_samples,
        device=args.device,
    )

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("SPAR-7M 评估结果")
    print("=" * 50)
    for k, v in results.items():
        if v is not None:
            print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    print(f"\n结果已保存至: {result_path}")


if __name__ == "__main__":
    main()
