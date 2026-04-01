"""
VG-LLM 计算冗余分析脚本（前期 Profiling）
==========================================
定量刻画 VG-LLM 的帧级和 token 级计算冗余，
为 StreamSpatial-VLM 的优化提供数据驱动依据。

分析维度：
1. 帧级冗余率：相邻帧 CLS token 余弦相似度分布
2. Token 级冗余率：注意力权重低于阈值的 patch 比例
3. 3D 网络耗时占比：PyTorch Profiler 逐层计时
4. 显存分配分析：KV-Cache 占比

用法：
    python profiling/profile_vgllm.py \
        --data_root /data/spar7m \
        --num_videos 50 \
        --output_dir profiling/reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gate_2d3d import SemanticGate2D3D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VG-LLM 冗余分析")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=50,
                        help="分析的视频数量")
    parser.add_argument("--output_dir", type=str, default="profiling/reports")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_threshold", type=float, default=0.01,
                        help="注意力权重冗余阈值")
    return parser.parse_args()


# ------------------------------------------------------------------
# 1. 帧级冗余分析
# ------------------------------------------------------------------

def analyze_frame_redundancy(
    cls_token_sequences: List[torch.Tensor],
    tau_list: List[float] = None,
) -> Dict[str, Any]:
    """
    分析帧级冗余率。

    Args:
        cls_token_sequences: 每个视频的 CLS token 序列列表，
                             每个元素 shape (T, d)
        tau_list: 门控阈值列表

    Returns:
        分析报告字典
    """
    if tau_list is None:
        tau_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    all_similarities = []
    trigger_rates_by_tau = {tau: [] for tau in tau_list}

    for cls_tokens in tqdm(cls_token_sequences, desc="帧级冗余分析"):
        T = cls_tokens.shape[0]
        if T < 2:
            continue

        # 计算相邻帧余弦相似度
        sims = []
        for t in range(1, T):
            sim = F.cosine_similarity(
                cls_tokens[t].unsqueeze(0),
                cls_tokens[t - 1].unsqueeze(0),
            ).item()
            sims.append(sim)
        all_similarities.extend(sims)

        # 不同阈值下的触发率
        analysis = SemanticGate2D3D.analyze_redundancy(cls_tokens, tau_list)
        for tau in tau_list:
            trigger_rates_by_tau[tau].append(analysis[tau])

    sim_array = np.array(all_similarities)
    report = {
        "frame_similarity": {
            "mean": float(sim_array.mean()),
            "std": float(sim_array.std()),
            "median": float(np.median(sim_array)),
            "pct_above_0.85": float((sim_array > 0.85).mean()),
            "pct_above_0.90": float((sim_array > 0.90).mean()),
            "pct_above_0.95": float((sim_array > 0.95).mean()),
        },
        "gate_trigger_rates": {
            str(tau): float(np.mean(rates))
            for tau, rates in trigger_rates_by_tau.items()
        },
        "redundancy_conclusion": (
            f"平均 {(sim_array > 0.85).mean() * 100:.1f}% 的相邻帧相似度 > 0.85，"
            f"使用 τ=0.15 时 3D 触发率约为 "
            f"{np.mean(trigger_rates_by_tau[0.15]) * 100:.1f}%"
        ),
    }
    return report


# ------------------------------------------------------------------
# 2. Token 级冗余分析
# ------------------------------------------------------------------

def analyze_token_redundancy(
    attention_weights_list: List[torch.Tensor],
    threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    分析 token 级冗余率（基于注意力权重）。

    Args:
        attention_weights_list: 注意力权重列表，每个 shape (H, N, N)
                                H=注意力头数，N=token数
        threshold: 低于此值视为冗余

    Returns:
        分析报告字典
    """
    redundancy_rates = []

    for attn in tqdm(attention_weights_list, desc="Token 级冗余分析"):
        # 对所有头取平均，关注 CLS token 对 patch 的注意力
        # attn: (H, N, N)，取第 0 行（CLS → patch）
        mean_attn = attn.mean(dim=0)  # (N, N)
        cls_to_patch = mean_attn[0, 1:]  # (N-1,)，CLS 对各 patch 的注意力

        # 归一化
        cls_to_patch = cls_to_patch / (cls_to_patch.sum() + 1e-8)
        redundant = (cls_to_patch < threshold).float().mean().item()
        redundancy_rates.append(redundant)

    rates = np.array(redundancy_rates)
    report = {
        "token_redundancy": {
            "mean_redundancy_rate": float(rates.mean()),
            "std": float(rates.std()),
            "pct_above_50pct_redundant": float((rates > 0.5).mean()),
            "pct_above_60pct_redundant": float((rates > 0.6).mean()),
        },
        "redundancy_conclusion": (
            f"平均 {rates.mean() * 100:.1f}% 的 patch token 注意力权重 < {threshold}，"
            f"其中 {(rates > 0.6).mean() * 100:.1f}% 的帧冗余率超过 60%"
        ),
    }
    return report


# ------------------------------------------------------------------
# 3. 推理耗时分析
# ------------------------------------------------------------------

def analyze_inference_timing(
    model,
    sample_frames: torch.Tensor,
    num_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    使用 PyTorch Profiler 分析各模块推理耗时。

    Args:
        model:         StreamSpatialVLM 实例
        sample_frames: 测试帧，shape (T, 3, H, W)
        num_runs:      重复运行次数（取均值）

    Returns:
        耗时分析报告
    """
    timings = {"vit": [], "vggt": [], "zip": [], "llm": [], "total": []}

    for _ in range(num_runs):
        model.reset()
        t_total_start = time.perf_counter()

        T = sample_frames.shape[0]
        for t in range(T):
            frame = sample_frames[t].to(device)
            model.process_frame(frame, frame_idx=t)

        model.answer("What is the distance to the nearest object?")
        t_total = time.perf_counter() - t_total_start
        timings["total"].append(t_total)

        # 从模型内部统计获取分项耗时
        stats = model.get_timing_stats()
        for k in ["vit", "vggt", "zip", "llm"]:
            key = f"{k}_ms_mean"
            if key in stats:
                timings[k].append(stats[key])

    T = sample_frames.shape[0]
    total_mean = np.mean(timings["total"])

    report = {
        "num_frames": T,
        "num_runs": num_runs,
        "total_latency_ms": float(total_mean * 1000),
        "fps": float(T / total_mean),
        "module_breakdown_ms": {
            k: float(np.mean(v)) if v else 0.0
            for k, v in timings.items()
            if k != "total"
        },
    }

    # 计算各模块占比
    total_module = sum(report["module_breakdown_ms"].values())
    if total_module > 0:
        report["module_percentage"] = {
            k: f"{v / total_module * 100:.1f}%"
            for k, v in report["module_breakdown_ms"].items()
        }

    return report


# ------------------------------------------------------------------
# 4. 显存分析
# ------------------------------------------------------------------

def analyze_memory_usage(
    model,
    sample_frames: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Any]:
    """分析推理过程中的显存占用"""
    if device == "cpu":
        return {"note": "CPU 模式，无 GPU 显存统计"}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    baseline_mem = torch.cuda.memory_allocated(device) / 1024 ** 3

    model.reset()
    T = sample_frames.shape[0]
    mem_per_frame = []

    for t in range(T):
        frame = sample_frames[t].to(device)
        model.process_frame(frame, frame_idx=t)
        mem_per_frame.append(torch.cuda.memory_allocated(device) / 1024 ** 3)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    report = {
        "baseline_gb": float(baseline_mem),
        "peak_gb": float(peak_mem),
        "incremental_gb": float(peak_mem - baseline_mem),
        "mem_per_frame_gb": [float(m) for m in mem_per_frame],
        "kv_cache_stats": model.cache.get_stats(),
    }
    return report


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VG-LLM 计算冗余分析报告")
    print("=" * 60)

    # 生成 Mock 数据（实际使用时替换为真实数据加载）
    print("\n[1/4] 生成测试数据...")
    num_videos = args.num_videos
    T, d = 16, 1152  # 16 帧，ViT 特征维度

    # Mock CLS token 序列（模拟真实视频的缓慢变化）
    cls_sequences = []
    for _ in range(num_videos):
        base = torch.randn(d)
        tokens = [base]
        for t in range(1, T):
            # 模拟缓慢变化：大部分帧相似，偶尔突变
            if torch.rand(1).item() < 0.2:  # 20% 概率场景切换
                tokens.append(torch.randn(d))
            else:
                noise = torch.randn(d) * 0.05
                tokens.append(F.normalize(tokens[-1] + noise, dim=0))
        cls_sequences.append(torch.stack(tokens))

    # Mock 注意力权重（模拟稀疏注意力）
    N = 257  # 256 patches + 1 CLS
    H_heads = 16
    attn_list = []
    for _ in range(num_videos * T):
        # 模拟稀疏注意力：大部分 patch 权重极低
        attn = torch.zeros(H_heads, N, N)
        # 只有少数 patch 有高注意力
        hot_patches = torch.randint(1, N, (H_heads, 20))
        for h in range(H_heads):
            attn[h, 0, hot_patches[h]] = torch.rand(20)
        attn = F.softmax(attn, dim=-1)
        attn_list.append(attn)

    # 执行分析
    print("\n[2/4] 帧级冗余分析...")
    frame_report = analyze_frame_redundancy(cls_sequences)

    print("\n[3/4] Token 级冗余分析...")
    token_report = analyze_token_redundancy(attn_list, threshold=args.attn_threshold)

    print("\n[4/4] KV Cache 显存理论分析...")
    from models.kv_cache import IncrementalKVCache
    mem_analysis = IncrementalKVCache.analyze_window_memory(
        tokens_per_frame=128,   # 50% 压缩后
        token_dim=1152,
        feat_3d_dim=1024,
        window_sizes=[4, 8, 16, 32],
    )

    # 汇总报告
    full_report = {
        "frame_redundancy": frame_report,
        "token_redundancy": token_report,
        "kv_cache_memory": mem_analysis,
        "summary": {
            "frame_redundancy_rate": frame_report["frame_similarity"]["pct_above_0.85"],
            "token_redundancy_rate": token_report["token_redundancy"]["mean_redundancy_rate"],
            "recommended_tau": 0.15,
            "recommended_keep_ratio": 0.50,
            "recommended_window_size": 8,
        },
    }

    # 保存报告
    report_path = output_dir / "redundancy_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("分析结论：")
    print(f"  帧级冗余：{frame_report['redundancy_conclusion']}")
    print(f"  Token冗余：{token_report['redundancy_conclusion']}")
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
