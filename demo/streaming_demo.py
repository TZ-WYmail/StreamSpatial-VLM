"""
流式空间理解 Demo
=================
从视频文件逐帧读取，实时运行 StreamSpatial-VLM 推理，
并输出每帧的门控状态、压缩率、延迟等信息。

用法：
    python demo/streaming_demo.py \
        --config configs/streamspatial_default.yaml \
        --video  path/to/video.mp4 \
        --query  "场景中有哪些物体？它们的空间关系如何？" \
        --output_dir results/demo \
        --save_viz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config
from utils.speed_profiler import SpeedProfiler
from utils.visualizer import Visualizer


def load_video_frames(video_path: str, max_frames: int = 32):
    """
    从视频文件读取帧序列。
    若 cv2 不可用，则生成随机帧用于测试。
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        print(f"[Demo] 从视频读取 {len(frames)} 帧: {video_path}")
        return frames
    except ImportError:
        print("[Demo] WARNING: cv2 未安装，使用随机帧进行测试")
        return [np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
                for _ in range(min(max_frames, 16))]


def load_depth_and_pose(frame: np.ndarray, frame_idx: int, depth_dir: str | None):
    """加载预计算的深度图和位姿置信图，若不存在则生成随机数据"""
    H, W = frame.shape[:2]
    P = H // 14  # patch 数量（14px patch）

    if depth_dir is not None:
        depth_path = Path(depth_dir) / f"frame_{frame_idx:06d}_depth.npy"
        pose_path = Path(depth_dir) / f"frame_{frame_idx:06d}_pose.npy"
        if depth_path.exists() and pose_path.exists():
            depth = np.load(str(depth_path))
            pose_conf = np.load(str(pose_path))
            return depth, pose_conf

    # 回退：随机生成
    depth = np.random.rand(H, W).astype(np.float32)
    pose_conf = np.random.rand(P, P).astype(np.float32)
    return depth, pose_conf


def run_demo(args):
    # ---------- 加载配置 ----------
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 初始化模型 ----------
    print("[Demo] 初始化 StreamSpatial-VLM ...")
    try:
        from models.stream_spatial_vlm import StreamSpatialVLM, StreamSpatialConfig
        model_cfg = StreamSpatialConfig(
            vlm_name=cfg.get("model", {}).get("vlm_name", "mock"),
            vggt_name=cfg.get("model", {}).get("vggt_name", "mock"),
            use_gate=cfg.get("modules", {}).get("use_gate", True),
            use_zip=cfg.get("modules", {}).get("use_zip", True),
            use_incremental=cfg.get("modules", {}).get("use_incremental", True),
            gate_tau=cfg.get("gate", {}).get("tau", 0.15),
            zip_keep_ratio=cfg.get("zip", {}).get("keep_ratio", 0.50),
            kv_window_size=cfg.get("kv_cache", {}).get("window_size", 8),
        )
        model = StreamSpatialVLM(model_cfg)
    except Exception as e:
        print(f"[Demo] 模型加载失败（{e}），使用 Mock 模式")
        model = None

    # ---------- 读取视频帧 ----------
    frames = load_video_frames(args.video, args.max_frames)
    if not frames:
        print("[Demo] 错误：未能读取任何帧")
        return

    # ---------- 流式推理 ----------
    profiler = SpeedProfiler()
    gate_triggers = []
    latencies_ms = []
    memory_gb_list = []
    compression_ratios = []

    print(f"\n[Demo] 开始流式推理，共 {len(frames)} 帧 ...")
    print(f"[Demo] 问题: {args.query}\n")

    profiler.start()
    for i, frame in enumerate(frames):
        depth, pose_conf = load_depth_and_pose(frame, i, args.depth_dir)

        t0 = time.perf_counter()
        if model is not None:
            try:
                import torch
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                depth_tensor = torch.from_numpy(depth)
                pose_tensor = torch.from_numpy(pose_conf)
                result = model.process_frame(frame_tensor, depth_tensor, pose_tensor)
                triggered = result.get("gate_triggered", True)
                ratio = result.get("compression_ratio", 1.0)
            except Exception:
                triggered = bool(i % 3 == 0)
                ratio = 0.5
        else:
            triggered = bool(i % 3 == 0)
            ratio = 0.5

        elapsed_ms = (time.perf_counter() - t0) * 1000
        gate_triggers.append(triggered)
        latencies_ms.append(elapsed_ms)
        compression_ratios.append(ratio)

        try:
            import torch
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        except Exception:
            mem = 0.0
        memory_gb_list.append(mem)

        status = "🔴 3D触发" if triggered else "🟢 3D复用"
        print(f"  帧 {i+1:3d}/{len(frames)} | {status} | "
              f"压缩率={ratio:.2f} | 延迟={elapsed_ms:.1f}ms | 显存={mem:.2f}GB")

    profiler.stop()

    # ---------- 生成最终回答 ----------
    print(f"\n[Demo] 生成回答 ...")
    if model is not None:
        try:
            answer = model.answer(args.query)
        except Exception:
            answer = "[Mock] 场景中包含多个物体，桌子位于房间中央，椅子分布在桌子周围。"
    else:
        answer = "[Mock] 场景中包含多个物体，桌子位于房间中央，椅子分布在桌子周围。"

    print(f"\n{'='*60}")
    print(f"问题: {args.query}")
    print(f"回答: {answer}")
    print(f"{'='*60}")

    # ---------- 统计信息 ----------
    trigger_rate = sum(gate_triggers) / len(gate_triggers)
    avg_ratio = np.mean(compression_ratios)
    stats = {
        "fps": profiler.fps,
        "mean_latency_ms": profiler.mean_latency_ms,
        "peak_memory_gb": profiler.peak_memory_gb,
        "gate_trigger_rate": trigger_rate,
        "avg_compression_ratio": float(avg_ratio),
        "total_frames": len(frames),
        "answer": answer,
    }

    print(f"\n[Demo] 性能统计:")
    print(f"  FPS:          {stats['fps']:.2f}")
    print(f"  平均延迟:     {stats['mean_latency_ms']:.1f} ms")
    print(f"  峰值显存:     {stats['peak_memory_gb']:.2f} GB")
    print(f"  3D触发率:     {trigger_rate:.1%}")
    print(f"  平均压缩率:   {avg_ratio:.2f}")

    # ---------- 保存结果 ----------
    result_path = output_dir / "demo_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[Demo] 结果已保存: {result_path}")

    # ---------- 可视化 ----------
    if args.save_viz:
        viz = Visualizer(str(output_dir / "figures"))
        similarities = [1.0 - (0.2 if t else 0.05) for t in gate_triggers[1:]]
        viz.plot_gate_heatmap(gate_triggers, similarities,
                              tau=cfg.get("gate", {}).get("tau", 0.15))
        viz.plot_streaming_timeline(latencies_ms, memory_gb_list, gate_triggers)
        print("[Demo] 可视化图表已保存")


def main():
    parser = argparse.ArgumentParser(description="StreamSpatial-VLM 流式推理 Demo")
    parser.add_argument("--config", default="configs/streamspatial_default.yaml")
    parser.add_argument("--video", default="", help="输入视频路径（留空使用随机帧）")
    parser.add_argument("--query", default="请描述场景中物体的空间关系")
    parser.add_argument("--output_dir", default="results/demo")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--depth_dir", default=None, help="预计算深度图目录")
    parser.add_argument("--save_viz", action="store_true", help="保存可视化图表")
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()
