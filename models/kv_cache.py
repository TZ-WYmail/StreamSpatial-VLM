"""
增量 KV 缓存模块 (Incremental KV Cache)
========================================
维护滑动窗口内的 3D 隐状态队列与 2D 压缩 token 缓存，
实现逐帧流式推理，推理复杂度从 O(T) 降至 O(w)。

驱逐策略：超出窗口的旧帧按重要性分数（Zip 输出）驱逐。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class KVCacheConfig:
    """KV 缓存超参配置"""
    window_size: int = 8          # 滑动窗口大小 w
    feat_3d_dim: int = 1024       # VGGT 3D 特征维度
    feat_2d_dim: int = 1152       # ViT patch token 维度（Qwen2.5-VL ViT）
    max_tokens_per_frame: int = 256  # 每帧最多保留 token 数（压缩后）


@dataclass
class FrameEntry:
    """单帧缓存条目"""
    frame_idx: int
    tokens_2d: torch.Tensor          # 压缩后的 2D patch tokens，(K, d)
    token_indices: torch.Tensor      # 原始 patch 索引，(K,)，用于位置编码
    feat_3d: Optional[torch.Tensor]  # VGGT 3D 特征，(C,) 或 None（复用上帧）
    importance_scores: Optional[torch.Tensor]  # token 重要性分数，(K,)
    is_3d_fresh: bool = True         # 是否为新鲜 3D 特征（非复用）


class IncrementalKVCache:
    """
    增量 KV 缓存，支持逐帧流式推理。

    用法示例：
        cache = IncrementalKVCache(KVCacheConfig(window_size=8))
        for frame_t in video_stream:
            cache.push(frame_entry)
            if query_arrived:
                tokens, pos_ids = cache.get_context()
                answer = llm_decode(tokens, pos_ids)
    """

    def __init__(self, config: KVCacheConfig = KVCacheConfig()):
        self.config = config
        self._buffer: deque[FrameEntry] = deque(maxlen=config.window_size)
        self._frame_count: int = 0
        self._eviction_count: int = 0

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def push(self, entry: FrameEntry) -> Optional[FrameEntry]:
        """
        将新帧压入缓存。若缓存已满，驱逐最旧帧。

        Returns:
            被驱逐的帧条目（若有），否则 None
        """
        evicted = None
        if len(self._buffer) == self.config.window_size:
            evicted = self._buffer[0]  # 即将被 deque 自动驱逐
            self._eviction_count += 1

        self._buffer.append(entry)
        self._frame_count += 1
        return evicted

    def get_last_3d(self) -> Optional[torch.Tensor]:
        """获取最近一帧的 3D 特征（用于 Gate=False 时复用）"""
        for entry in reversed(self._buffer):
            if entry.feat_3d is not None:
                return entry.feat_3d
        return None

    def get_last_cls(self) -> Optional[torch.Tensor]:
        """获取最近一帧的 2D CLS token（用于 Gate 差异计算）"""
        if not self._buffer:
            return None
        # CLS token 约定存储在 tokens_2d 的第 0 位（若有）
        last = self._buffer[-1]
        if last.tokens_2d is not None and last.tokens_2d.shape[0] > 0:
            return last.tokens_2d[0]
        return None

    def get_context(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取当前窗口内所有帧的 token 序列，用于 LLM 解码。

        Returns:
            all_tokens:   (total_K, d)，拼接所有帧的压缩 token
            all_pos_ids:  (total_K,)，原始 patch 索引（位置编码）
            frame_ids:    (total_K,)，每个 token 所属帧编号
        """
        all_tokens = []
        all_pos_ids = []
        frame_ids = []

        for entry in self._buffer:
            all_tokens.append(entry.tokens_2d)
            all_pos_ids.append(entry.token_indices)
            frame_ids.append(
                torch.full(
                    (entry.tokens_2d.shape[0],),
                    entry.frame_idx,
                    dtype=torch.long,
                    device=entry.tokens_2d.device,
                )
            )

        if not all_tokens:
            raise RuntimeError("KV Cache 为空，无法获取上下文")

        return (
            torch.cat(all_tokens, dim=0),
            torch.cat(all_pos_ids, dim=0),
            torch.cat(frame_ids, dim=0),
        )

    def get_3d_context(self) -> Optional[torch.Tensor]:
        """
        获取窗口内所有新鲜 3D 特征，用于 LLM 的几何条件输入。

        Returns:
            3D 特征序列，shape (num_fresh, C)，或 None
        """
        feats = [
            entry.feat_3d
            for entry in self._buffer
            if entry.feat_3d is not None and entry.is_3d_fresh
        ]
        if not feats:
            return None
        return torch.stack(feats, dim=0)  # (num_fresh, C)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def current_size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.config.window_size

    @property
    def total_tokens(self) -> int:
        """当前缓存中的总 token 数"""
        return sum(e.tokens_2d.shape[0] for e in self._buffer)

    def memory_bytes(self) -> int:
        """估算当前缓存占用显存（字节）"""
        total = 0
        for entry in self._buffer:
            total += entry.tokens_2d.numel() * 4  # float32
            if entry.feat_3d is not None:
                total += entry.feat_3d.numel() * 4
        return total

    def get_stats(self) -> dict:
        return {
            "window_size": self.config.window_size,
            "current_size": self.current_size,
            "total_frames_processed": self._frame_count,
            "eviction_count": self._eviction_count,
            "total_tokens_in_cache": self.total_tokens,
            "memory_mb": self.memory_bytes() / 1024 / 1024,
        }

    def reset(self):
        """重置缓存（新视频序列开始时调用）"""
        self._buffer.clear()
        self._frame_count = 0
        self._eviction_count = 0

    # ------------------------------------------------------------------
    # 消融实验：不同窗口大小的显存分析
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_window_memory(
        tokens_per_frame: int,
        token_dim: int,
        feat_3d_dim: int,
        window_sizes: Optional[list[int]] = None,
    ) -> dict:
        """
        理论估算不同窗口大小下的显存占用。

        Args:
            tokens_per_frame: 每帧压缩后 token 数
            token_dim:        token 特征维度
            feat_3d_dim:      3D 特征维度
            window_sizes:     要分析的窗口大小列表

        Returns:
            dict: window_size → memory_mb
        """
        if window_sizes is None:
            window_sizes = [4, 8, 16, 32]

        results = {}
        for w in window_sizes:
            # 2D token 显存
            mem_2d = w * tokens_per_frame * token_dim * 4  # float32
            # 3D 特征显存（假设每帧都有）
            mem_3d = w * feat_3d_dim * 4
            total_mb = (mem_2d + mem_3d) / 1024 / 1024
            results[w] = {
                "memory_mb": total_mb,
                "tokens_2d": w * tokens_per_frame,
                "feats_3d": w,
            }
        return results
