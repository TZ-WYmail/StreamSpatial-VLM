"""
3D→2D 几何引导压缩模块 (Geometry-guided Zip)
=============================================
利用 VGGT 输出的深度图方差与位姿不确定性熵，生成无监督重要性图，
对 ViT patch token 进行 Top-K 无参裁剪。

重要性分数：
    S_i = α · σ²(D_i) + β · H(P_i)

裁剪策略：
    T_keep = TopK({S_i}, k = floor(r · N))

零额外可训练参数，无需 3D 真值标注。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ZipConfig:
    """压缩模块超参配置"""
    keep_ratio: float = 0.50    # token 保留率 r，默认 50%
    alpha: float = 0.60         # 深度方差权重
    beta: float = 0.40          # 位姿不确定性权重
    patch_size: int = 14        # ViT patch 大小（像素），用于 patch 坐标映射
    min_keep: int = 16          # 最少保留 token 数（防止过度裁剪）


class GeometryGuidedZip:
    """
    3D→2D 几何引导 Token 压缩。

    用法示例：
        zipper = GeometryGuidedZip(ZipConfig(keep_ratio=0.5))
        kept_tokens, kept_indices = zipper(patch_tokens, depth_map, pose_conf)
    """

    def __init__(self, config: ZipConfig = ZipConfig()):
        self.config = config
        self._compression_stats: list[float] = []

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def __call__(
        self,
        patch_tokens: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        pose_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 patch token 进行几何引导压缩。

        Args:
            patch_tokens: ViT 输出的 patch token，shape (N, d)
                          N = H_patch × W_patch
            depth_map:    深度图，shape (H, W)，与原图同分辨率
                          若为 None，则退化为均匀随机保留
            pose_conf:    VGGT 位姿置信度图，shape (H, W)，值域 [0, 1]
                          若为 None，则仅使用深度方差

        Returns:
            kept_tokens:  压缩后的 token，shape (K, d)
            kept_indices: 保留的 patch 索引，shape (K,)，用于位置编码对齐
        """
        N, d = patch_tokens.shape
        K = max(self.config.min_keep, int(N * self.config.keep_ratio))
        K = min(K, N)  # 不超过总数

        # 计算重要性分数
        scores = self._compute_importance(N, depth_map, pose_conf)  # (N,)

        # Top-K 选择
        _, kept_indices = torch.topk(scores, K, dim=0, largest=True, sorted=False)
        kept_indices, _ = kept_indices.sort()  # 保持空间顺序

        kept_tokens = patch_tokens[kept_indices]  # (K, d)

        # 记录压缩率统计
        self._compression_stats.append(K / N)

        return kept_tokens, kept_indices

    # ------------------------------------------------------------------
    # 重要性图计算
    # ------------------------------------------------------------------

    def _compute_importance(
        self,
        N: int,
        depth_map: Optional[torch.Tensor],
        pose_conf: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算每个 patch 的重要性分数 S_i = α·σ²(D_i) + β·H(P_i)

        Returns:
            scores: shape (N,)
        """
        device = depth_map.device if depth_map is not None else torch.device("cpu")
        scores = torch.zeros(N, device=device)

        # --- 深度方差项 ---
        if depth_map is not None:
            depth_var = self._patch_depth_variance(depth_map, N)  # (N,)
            depth_var = self._normalize(depth_var)
            scores = scores + self.config.alpha * depth_var

        # --- 位姿不确定性熵项 ---
        if pose_conf is not None:
            pose_entropy = self._patch_pose_entropy(pose_conf, N)  # (N,)
            pose_entropy = self._normalize(pose_entropy)
            scores = scores + self.config.beta * pose_entropy

        # 若两者均为 None，退化为均匀分数（随机保留）
        if depth_map is None and pose_conf is None:
            scores = torch.rand(N, device=device)

        return scores

    def _patch_depth_variance(
        self, depth_map: torch.Tensor, N: int
    ) -> torch.Tensor:
        """
        将深度图划分为 N 个 patch，计算每个 patch 内的深度方差。

        Args:
            depth_map: (H, W)
            N:         patch 数量（需为完全平方数或已知 H_p, W_p）

        Returns:
            var_per_patch: (N,)
        """
        H, W = depth_map.shape
        H_p = W_p = int(N ** 0.5)
        # 若 N 不是完全平方数，按行列数推断
        if H_p * W_p != N:
            H_p = H // self.config.patch_size
            W_p = W // self.config.patch_size

        # 重塑为 (H_p, patch_h, W_p, patch_w)
        ph = H // H_p
        pw = W // W_p
        depth_crop = depth_map[: H_p * ph, : W_p * pw]
        patches = depth_crop.reshape(H_p, ph, W_p, pw)
        # 计算每个 patch 内方差
        var = patches.var(dim=(1, 3))  # (H_p, W_p)
        return var.flatten()  # (N,)

    def _patch_pose_entropy(
        self, pose_conf: torch.Tensor, N: int
    ) -> torch.Tensor:
        """
        将位姿置信度图划分为 N 个 patch，计算每个 patch 的不确定性熵。
        不确定性 = 1 - confidence，熵 H = -p·log(p) - (1-p)·log(1-p)

        Args:
            pose_conf: (H, W)，值域 [0, 1]
            N:         patch 数量

        Returns:
            entropy_per_patch: (N,)
        """
        H, W = pose_conf.shape
        H_p = W_p = int(N ** 0.5)
        if H_p * W_p != N:
            H_p = H // self.config.patch_size
            W_p = W // self.config.patch_size

        ph = H // H_p
        pw = W // W_p
        conf_crop = pose_conf[: H_p * ph, : W_p * pw]
        patches = conf_crop.reshape(H_p, ph, W_p, pw)
        mean_conf = patches.mean(dim=(1, 3))  # (H_p, W_p)，每 patch 平均置信度

        # 二值熵
        p = mean_conf.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return entropy.flatten()  # (N,)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Min-Max 归一化到 [0, 1]"""
        x_min, x_max = x.min(), x.max()
        if (x_max - x_min).abs() < 1e-8:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    @property
    def mean_compression_ratio(self) -> float:
        """平均压缩率（保留 token 比例）"""
        if not self._compression_stats:
            return 1.0
        return sum(self._compression_stats) / len(self._compression_stats)

    def reset_stats(self):
        self._compression_stats.clear()

    # ------------------------------------------------------------------
    # 消融实验：批量分析不同保留率的影响
    # ------------------------------------------------------------------

    @staticmethod
    def ablation_keep_ratio(
        patch_tokens: torch.Tensor,
        depth_map: torch.Tensor,
        pose_conf: Optional[torch.Tensor] = None,
        ratios: Optional[list[float]] = None,
    ) -> dict:
        """
        对同一帧，分析不同保留率下的 token 分布差异。

        Returns:
            dict: ratio → {"kept_indices": Tensor, "kept_count": int}
        """
        if ratios is None:
            ratios = [0.25, 0.50, 0.75, 1.00]

        results = {}
        for r in ratios:
            zipper = GeometryGuidedZip(ZipConfig(keep_ratio=r))
            _, indices = zipper(patch_tokens, depth_map, pose_conf)
            results[r] = {
                "kept_indices": indices,
                "kept_count": indices.shape[0],
                "total_count": patch_tokens.shape[0],
            }
        return results
