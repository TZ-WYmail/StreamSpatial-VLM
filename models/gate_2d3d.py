"""
2D→3D 语义门控模块 (Semantic Gate)
====================================
基于相邻帧 CLS token 的余弦距离，决定是否触发 VGGT-1B 3D 几何推理。
零额外可训练参数，纯规则判断。

数学定义：
    δ_t = 1 - cos(z_t, z_{t-k})
    g_t = 1[δ_t > τ]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateConfig:
    """门控模块超参配置"""
    tau: float = 0.15          # 余弦距离阈值，默认 0.15
    history_step: int = 1      # 历史帧步长 k，默认 1
    warmup_frames: int = 1     # 前 warmup_frames 帧强制触发 3D


class SemanticGate2D3D:
    """
    2D→3D 语义门控。

    用法示例：
        gate = SemanticGate2D3D(GateConfig(tau=0.15))
        for frame_feat in stream:          # frame_feat: (d,) CLS token
            should_run_3d = gate(frame_feat)
    """

    def __init__(self, config: GateConfig = GateConfig()):
        self.config = config
        self._history: list[torch.Tensor] = []   # 历史 CLS token 队列
        self._frame_idx: int = 0
        self._trigger_count: int = 0
        self._total_count: int = 0

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def __call__(self, cls_token: torch.Tensor) -> bool:
        """
        判断当前帧是否需要触发 3D 推理。

        Args:
            cls_token: 当前帧 2D ViT CLS token，shape (d,) 或 (1, d)

        Returns:
            True  → 触发 VGGT-1B 推理
            False → 复用上一帧 3D 缓存
        """
        cls_token = cls_token.detach().float().flatten()  # (d,)
        self._total_count += 1

        # warmup 阶段强制触发
        if self._frame_idx < self.config.warmup_frames:
            triggered = True
        elif len(self._history) < self.config.history_step:
            triggered = True
        else:
            ref = self._history[-self.config.history_step]
            delta = self._cosine_distance(cls_token, ref)
            triggered = delta > self.config.tau

        # 更新历史队列
        self._history.append(cls_token)
        if len(self._history) > max(self.config.history_step + 1, 8):
            self._history.pop(0)

        self._frame_idx += 1
        if triggered:
            self._trigger_count += 1

        return triggered

    # ------------------------------------------------------------------
    # 统计接口
    # ------------------------------------------------------------------

    @property
    def trigger_rate(self) -> float:
        """3D 网络触发率（越低越省计算）"""
        if self._total_count == 0:
            return 0.0
        return self._trigger_count / self._total_count

    def reset(self):
        """重置状态（新视频序列开始时调用）"""
        self._history.clear()
        self._frame_idx = 0
        self._trigger_count = 0
        self._total_count = 0

    def get_stats(self) -> dict:
        return {
            "total_frames": self._total_count,
            "trigger_count": self._trigger_count,
            "trigger_rate": self.trigger_rate,
            "tau": self.config.tau,
            "history_step": self.config.history_step,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """余弦距离 δ = 1 - cos(a, b)，范围 [0, 2]"""
        cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        return 1.0 - cos_sim

    # ------------------------------------------------------------------
    # 批量离线分析（用于 Profiling 实验）
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_redundancy(
        cls_tokens: torch.Tensor,
        tau_list: Optional[list[float]] = None,
    ) -> dict:
        """
        离线分析一段视频序列的帧级冗余率。

        Args:
            cls_tokens: shape (T, d)，T 帧的 CLS token 序列
            tau_list:   要分析的阈值列表

        Returns:
            dict，key 为 tau，value 为对应触发率
        """
        if tau_list is None:
            tau_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        T = cls_tokens.shape[0]
        results = {}

        for tau in tau_list:
            gate = SemanticGate2D3D(GateConfig(tau=tau))
            for t in range(T):
                gate(cls_tokens[t])
            results[tau] = gate.trigger_rate

        # 同时计算相邻帧余弦相似度分布
        sims = []
        for t in range(1, T):
            sim = F.cosine_similarity(
                cls_tokens[t].unsqueeze(0),
                cls_tokens[t - 1].unsqueeze(0),
            ).item()
            sims.append(sim)

        sim_tensor = torch.tensor(sims)
        results["similarity_stats"] = {
            "mean": sim_tensor.mean().item(),
            "std": sim_tensor.std().item(),
            "min": sim_tensor.min().item(),
            "max": sim_tensor.max().item(),
            "pct_above_0.85": (sim_tensor > 0.85).float().mean().item(),
        }

        return results
