"""
ScanQA 零样本评估脚本
======================
评估指标：EM、BLEU-4、Accuracy（Token F1）

用法：
    python eval/eval_scanqa.py \
        --config configs/streamspatial_default.yaml \
        --data_root /data/scanqa \
        --output_dir results/scanqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loaders.scanqa import ScanQADataset
from eval.metrics import compute_exact_match, compute_bleu4, compute_accuracy
from utils.config_loader import load_config
from utils.speed_profiler import SpeedProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScanQA 零样本评估")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/scanqa")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate_scanqa(
    model,
    data_root: str,
    split: str = "val",
    max_samples: int = -1,
    device: str = "cuda",
) -> Dict[str, Any]:
    """ScanQA 评估主函数"""
    dataset = ScanQADataset(data_root=data_root, split=split)
    if max_samples > 0:
        dataset.annotations = dataset.annotations[:max_samples]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=ScanQADataset.collate_fn,
    )

    predictions = []
    references = []
    profiler = SpeedProfiler(device=device)

    for batch in tqdm(loader, desc="ScanQA 评估"):
        questions = batch["questions"]
        answers_list = batch["answers"]
        frames_list = batch["frames"]
        depth_list = batch["depth_maps"]
        pose_list = batch["pose_confs"]

        for i in range(len(questions)):
            frames = frames_list[i].to(device)
            depth_maps = depth_list[i]
            pose_confs = pose_list[i]

            model.reset()
            profiler.start()

            T = frames.shape[0]
            for t in range(T):
                depth_t = depth_maps[t].to(device) if depth_maps is not None else None
                pose_t = pose_confs[t].to(device) if pose_confs is not None else None
                model.process_frame(frames[t], depth_t, pose_t, frame_idx=t)

            pred = model.answer(questions[i])
            profiler.stop()

            predictions.append(pred)
            references.append(answers_list[i])

    results = {
        "dataset": "ScanQA",
        "split": split,
        "num_samples": len(predictions),
        "exact_match": compute_exact_match(predictions, references),
        "bleu4": compute_bleu4(predictions, references),
        "accuracy_f1": compute_accuracy(predictions, references),
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

    results = evaluate_scanqa(
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

    print("\nScanQA 评估结果:")
    for k, v in results.items():
        print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
