"""
可视化工具
==========
用于生成论文所需的可视化图表：
1. 门控触发热力图
2. 重要性图（深度方差 + 位姿不确定性 + 保留 token 分布）
3. 流式推理时序图（延迟 + 显存曲线）
4. 消融实验对比柱状图
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch


class Visualizer:
    """可视化工具类"""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 门控触发热力图
    # ------------------------------------------------------------------

    def plot_gate_heatmap(
        self,
        trigger_sequence: List[bool],
        similarities: List[float],
        tau: float,
        save_name: str = "gate_heatmap.png",
    ):
        """
        绘制门控触发时序图。

        Args:
            trigger_sequence: 每帧是否触发 3D 推理的布尔列表
            similarities:     相邻帧余弦相似度列表
            tau:              门控阈值
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

            T = len(trigger_sequence)
            frames = list(range(T))

            # 上图：相似度曲线
            ax1.plot(frames[1:], similarities, color="#2196F3", linewidth=1.5,
                     label="余弦相似度")
            ax1.axhline(y=1 - tau, color="#F44336", linestyle="--",
                        linewidth=1.5, label=f"触发阈值 (1-τ={1-tau:.2f})")
            ax1.set_ylabel("相邻帧余弦相似度")
            ax1.set_ylim(0, 1.05)
            ax1.legend(loc="lower right")
            ax1.grid(alpha=0.3)

            # 下图：触发热力图
            trigger_arr = np.array(trigger_sequence, dtype=float).reshape(1, -1)
            ax2.imshow(trigger_arr, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=1, extent=[0, T, 0, 1])
            ax2.set_ylabel("3D 触发")
            ax2.set_xlabel("帧编号")
            ax2.set_yticks([])

            trigger_rate = sum(trigger_sequence) / len(trigger_sequence)
            fig.suptitle(
                f"2D→3D 门控触发分析  |  τ={tau}  |  触发率={trigger_rate:.1%}",
                fontsize=13,
            )
            plt.tight_layout()
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Viz] 门控热力图已保存: {save_path}")
        except ImportError:
            print("[Viz] WARNING: matplotlib 未安装，跳过可视化")

    # ------------------------------------------------------------------
    # 2. 重要性图可视化
    # ------------------------------------------------------------------

    def plot_importance_map(
        self,
        frame: np.ndarray,
        depth_var: np.ndarray,
        pose_entropy: np.ndarray,
        importance: np.ndarray,
        kept_mask: np.ndarray,
        save_name: str = "importance_map.png",
    ):
        """
        并排展示：原图 | 深度方差 | 位姿熵 | 重要性图 | 保留 token 分布

        Args:
            frame:        原始图像，(H, W, 3) uint8
            depth_var:    深度方差图，(H_p, W_p)
            pose_entropy: 位姿熵图，(H_p, W_p)
            importance:   重要性分数图，(H_p, W_p)
            kept_mask:    保留 token 掩码，(H_p, W_p) bool
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            titles = ["原始帧", "深度方差 σ²(D)", "位姿熵 H(P)", "重要性分数 S", "保留 Token"]
            data = [frame, depth_var, pose_entropy, importance, kept_mask.astype(float)]
            cmaps = [None, "plasma", "viridis", "hot", "RdYlGn"]

            for ax, title, d, cmap in zip(axes, titles, data, cmaps):
                if cmap is None:
                    ax.imshow(d)
                else:
                    ax.imshow(d, cmap=cmap)
                ax.set_title(title, fontsize=11)
                ax.axis("off")

            keep_rate = kept_mask.mean()
            fig.suptitle(f"3D→2D 几何引导压缩  |  保留率={keep_rate:.1%}", fontsize=13)
            plt.tight_layout()
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Viz] 重要性图已保存: {save_path}")
        except ImportError:
            print("[Viz] WARNING: matplotlib 未安装，跳过可视化")

    # ------------------------------------------------------------------
    # 3. 流式推理时序图
    # ------------------------------------------------------------------

    def plot_streaming_timeline(
        self,
        frame_latencies_ms: List[float],
        memory_gb: List[float],
        gate_triggers: List[bool],
        save_name: str = "streaming_timeline.png",
    ):
        """绘制逐帧推理延迟与显存占用曲线"""
        try:
            import matplotlib.pyplot as plt

            T = len(frame_latencies_ms)
            frames = list(range(T))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

            # 延迟曲线（区分触发/非触发帧）
            for t in frames:
                color = "#F44336" if gate_triggers[t] else "#4CAF50"
                ax1.bar(t, frame_latencies_ms[t], color=color, alpha=0.8, width=0.8)
            ax1.set_ylabel("帧推理延迟 (ms)")
            ax1.axhline(y=np.mean(frame_latencies_ms), color="black",
                        linestyle="--", linewidth=1, label=f"均值={np.mean(frame_latencies_ms):.1f}ms")
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#F44336", label="3D 触发帧"),
                Patch(facecolor="#4CAF50", label="3D 复用帧"),
            ]
            ax1.legend(handles=legend_elements, loc="upper right")
            ax1.grid(axis="y", alpha=0.3)

            # 显存曲线
            ax2.fill_between(frames, memory_gb, alpha=0.4, color="#2196F3")
            ax2.plot(frames, memory_gb, color="#2196F3", linewidth=1.5)
            ax2.set_ylabel("显存占用 (GB)")
            ax2.set_xlabel("帧编号")
            ax2.grid(alpha=0.3)

            fps = T / (sum(frame_latencies_ms) / 1000)
            fig.suptitle(f"流式推理性能分析  |  FPS={fps:.1f}  |  峰值显存={max(memory_gb):.2f}GB",
                         fontsize=13)
            plt.tight_layout()
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Viz] 时序图已保存: {save_path}")
        except ImportError:
            print("[Viz] WARNING: matplotlib 未安装，跳过可视化")

    # ------------------------------------------------------------------
    # 4. 消融实验对比柱状图
    # ------------------------------------------------------------------

    def plot_ablation_bar(
        self,
        ablation_results: List[Dict[str, Any]],
        metric: str = "spar7m_accuracy",
        title: str = "消融实验对比",
        save_name: str = "ablation_bar.png",
    ):
        """绘制消融实验对比柱状图"""
        try:
            import matplotlib.pyplot as plt

            configs = [r["config"] for r in ablation_results]
            values = [r.get(metric, 0.0) for r in ablation_results]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#90CAF9"] * (len(configs) - 1) + ["#1565C0"]  # 最后一个（ours）深色
            bars = ax.bar(configs, values, color=colors, edgecolor="white", linewidth=0.5)

            # 标注数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)

            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(title, fontsize=13)
            ax.set_ylim(min(values) * 0.95, max(values) * 1.05)
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[Viz] 消融柱状图已保存: {save_path}")
        except ImportError:
            print("[Viz] WARNING: matplotlib 未安装，跳过可视化")
