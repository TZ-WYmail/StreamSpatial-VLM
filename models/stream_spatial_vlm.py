"""
StreamSpatial-VLM 主框架集成
==============================
将 2D→3D Gate、3D→2D Zip、增量 KV Cache 三大模块与
Qwen2.5-VL + VGGT-1B 集成为完整的流式空间理解系统。

推理流程（逐帧）：
    1. ViT 编码当前帧 → 2D patch tokens + CLS token
    2. Gate 判断是否触发 VGGT-1B → 3D 特征（或复用缓存）
    3. Zip 利用深度方差 + 位姿不确定性裁剪 patch tokens
    4. 压缩后 tokens 写入增量 KV Cache
    5. 收到 query 时，从 Cache 取上下文 → LLM 解码输出答案
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .gate_2d3d import SemanticGate2D3D, GateConfig
from .zip_3d2d import GeometryGuidedZip, ZipConfig
from .kv_cache import IncrementalKVCache, KVCacheConfig, FrameEntry


@dataclass
class StreamSpatialConfig:
    """完整框架配置"""
    # 子模块配置
    gate: GateConfig = field(default_factory=GateConfig)
    zip: ZipConfig = field(default_factory=ZipConfig)
    cache: KVCacheConfig = field(default_factory=KVCacheConfig)

    # 模型路径
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vggt_model_name: str = "facebook/vggt-1b"

    # 推理配置
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 256

    # 消融开关（用于消融实验）
    use_gate: bool = True       # 是否启用 2D→3D Gate
    use_zip: bool = True        # 是否启用 3D→2D Zip
    use_incremental: bool = True  # 是否启用增量 KV Cache


class StreamSpatialVLM(nn.Module):
    """
    StreamSpatial-VLM 完整推理框架。

    支持两种模式：
    - 流式模式（streaming=True）：逐帧输入，收到 query 时输出答案
    - 批量模式（streaming=False）：一次性输入所有帧，用于基线对比

    用法示例（流式）：
        model = StreamSpatialVLM(config)
        model.reset()
        for frame, depth, pose_conf in video_stream:
            model.process_frame(frame, depth, pose_conf)
        answer = model.answer(query)
    """

    def __init__(self, config: StreamSpatialConfig = StreamSpatialConfig()):
        super().__init__()
        self.config = config

        # 初始化三大核心模块
        self.gate = SemanticGate2D3D(config.gate)
        self.zipper = GeometryGuidedZip(config.zip)
        self.cache = IncrementalKVCache(config.cache)

        # 延迟加载大模型（避免初始化时 OOM）
        self._vlm = None
        self._vggt = None
        self._vit_encoder = None
        self._processor = None

        # 性能统计
        self._timing: Dict[str, list] = {
            "vit": [], "gate": [], "vggt": [], "zip": [], "llm": []
        }

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_models(self):
        """加载 Qwen2.5-VL 和 VGGT-1B（首次调用时执行）"""
        if self._vlm is not None:
            return

        print(f"[StreamSpatialVLM] 加载 VLM: {self.config.vlm_model_name}")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.vlm_model_name,
                torch_dtype=self.config.dtype,
                device_map=self.config.device,
            )
            self._processor = AutoProcessor.from_pretrained(self.config.vlm_model_name)
            # 提取 ViT 视觉编码器
            self._vit_encoder = self._vlm.visual
            print("[StreamSpatialVLM] VLM 加载完成")
        except ImportError:
            print("[StreamSpatialVLM] WARNING: transformers 未安装，使用 Mock 模式")
            self._vlm = _MockVLM()
            self._vit_encoder = _MockViT()
            self._processor = None

        print(f"[StreamSpatialVLM] 加载 VGGT: {self.config.vggt_model_name}")
        try:
            # VGGT 官方接口（需安装 vggt 包）
            from vggt.models.vggt import VGGT
            self._vggt = VGGT.from_pretrained(self.config.vggt_model_name)
            self._vggt = self._vggt.to(self.config.device).to(self.config.dtype)
            print("[StreamSpatialVLM] VGGT 加载完成")
        except (ImportError, Exception) as e:
            print(f"[StreamSpatialVLM] WARNING: VGGT 加载失败 ({e})，使用 Mock 模式")
            self._vggt = _MockVGGT()

    # ------------------------------------------------------------------
    # 流式推理接口
    # ------------------------------------------------------------------

    def reset(self):
        """重置所有状态（新视频序列开始时调用）"""
        self.gate.reset()
        self.cache.reset()
        self.zipper.reset_stats()
        for v in self._timing.values():
            v.clear()

    def process_frame(
        self,
        frame: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        pose_conf: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        处理单帧，更新内部缓存。

        Args:
            frame:      原始图像帧，shape (3, H, W) 或 (1, 3, H, W)
            depth_map:  深度图，shape (H, W)，可为 None
            pose_conf:  VGGT 位姿置信度图，shape (H, W)，可为 None
            frame_idx:  帧编号（可选，默认自动递增）

        Returns:
            frame_info: 包含 gate_triggered、compression_ratio 等统计信息
        """
        if self._vlm is None:
            self.load_models()

        if frame_idx is None:
            frame_idx = self.cache._frame_count

        frame = frame.to(self.config.device)

        # Step 1: 2D ViT 编码
        t0 = time.perf_counter()
        patch_tokens, cls_token = self._encode_2d(frame)  # (N, d), (d,)
        self._timing["vit"].append(time.perf_counter() - t0)

        # Step 2: 2D→3D Gate 判断
        t0 = time.perf_counter()
        if self.config.use_gate:
            gate_triggered = self.gate(cls_token)
        else:
            gate_triggered = True  # 消融：始终触发
        self._timing["gate"].append(time.perf_counter() - t0)

        # Step 3: 3D 几何特征提取（或复用缓存）
        t0 = time.perf_counter()
        if gate_triggered:
            feat_3d, depth_map_out, pose_conf_out = self._encode_3d(
                frame, depth_map, pose_conf
            )
            is_3d_fresh = True
        else:
            feat_3d = self.cache.get_last_3d()
            depth_map_out = depth_map
            pose_conf_out = pose_conf
            is_3d_fresh = False
        self._timing["vggt"].append(time.perf_counter() - t0)

        # Step 4: 3D→2D 几何引导压缩
        t0 = time.perf_counter()
        if self.config.use_zip:
            kept_tokens, kept_indices = self.zipper(
                patch_tokens, depth_map_out, pose_conf_out
            )
        else:
            kept_tokens = patch_tokens
            kept_indices = torch.arange(patch_tokens.shape[0], device=patch_tokens.device)
        self._timing["zip"].append(time.perf_counter() - t0)

        # Step 5: 写入增量 KV Cache
        entry = FrameEntry(
            frame_idx=frame_idx,
            tokens_2d=kept_tokens,
            token_indices=kept_indices,
            feat_3d=feat_3d,
            importance_scores=None,
            is_3d_fresh=is_3d_fresh,
        )
        if self.config.use_incremental:
            self.cache.push(entry)
        else:
            # 消融：不使用增量缓存，每次重建（模拟全量推理）
            self.cache.reset()
            self.cache.push(entry)

        return {
            "frame_idx": frame_idx,
            "gate_triggered": gate_triggered,
            "tokens_before": patch_tokens.shape[0],
            "tokens_after": kept_tokens.shape[0],
            "compression_ratio": kept_tokens.shape[0] / patch_tokens.shape[0],
            "is_3d_fresh": is_3d_fresh,
        }

    def answer(self, query: str) -> str:
        """
        基于当前缓存上下文回答问题。

        Args:
            query: 自然语言问题

        Returns:
            模型生成的答案字符串
        """
        if self._vlm is None:
            self.load_models()

        t0 = time.perf_counter()
        try:
            all_tokens, pos_ids, frame_ids = self.cache.get_context()
            feat_3d_ctx = self.cache.get_3d_context()
            answer = self._llm_decode(all_tokens, pos_ids, feat_3d_ctx, query)
        except RuntimeError as e:
            answer = f"[ERROR] {e}"
        self._timing["llm"].append(time.perf_counter() - t0)

        return answer

    # ------------------------------------------------------------------
    # 内部编码方法
    # ------------------------------------------------------------------

    def _encode_2d(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        2D ViT 编码，返回 patch tokens 和 CLS token。

        Returns:
            patch_tokens: (N, d)
            cls_token:    (d,)
        """
        with torch.no_grad():
            # Qwen2.5-VL ViT 输出格式：(1, N+1, d)，第 0 位为 CLS
            vit_out = self._vit_encoder(frame.unsqueeze(0) if frame.dim() == 3 else frame)
            if isinstance(vit_out, torch.Tensor):
                tokens = vit_out.squeeze(0)  # (N+1, d) 或 (N, d)
            else:
                tokens = vit_out.last_hidden_state.squeeze(0)

            if tokens.shape[0] > 1:
                cls_token = tokens[0]       # (d,)
                patch_tokens = tokens[1:]   # (N, d)
            else:
                cls_token = tokens.mean(0)
                patch_tokens = tokens

        return patch_tokens, cls_token

    def _encode_3d(
        self,
        frame: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        pose_conf: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VGGT-1B 3D 几何编码。

        Returns:
            feat_3d:      (C,) 3D 场景特征
            depth_map:    (H, W) 深度图（VGGT 输出或传入值）
            pose_conf:    (H, W) 位姿置信度图
        """
        with torch.no_grad():
            vggt_out = self._vggt(frame.unsqueeze(0) if frame.dim() == 3 else frame)

        if isinstance(vggt_out, dict):
            feat_3d = vggt_out.get("scene_feat", vggt_out.get("features"))
            if depth_map is None:
                depth_map = vggt_out.get("depth")
            if pose_conf is None:
                pose_conf = vggt_out.get("conf")
        else:
            feat_3d = vggt_out

        if feat_3d is not None:
            feat_3d = feat_3d.squeeze(0).float()

        return feat_3d, depth_map, pose_conf

    def _llm_decode(
        self,
        tokens: torch.Tensor,
        pos_ids: torch.Tensor,
        feat_3d: Optional[torch.Tensor],
        query: str,
    ) -> str:
        """LLM 解码生成答案"""
        # Mock 实现（实际需接入 Qwen2.5-VL 的 generate 接口）
        if isinstance(self._vlm, _MockVLM):
            return self._vlm.generate(query)

        # 实际 Qwen2.5-VL 推理（简化版）
        inputs = self._processor(
            text=query,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            output_ids = self._vlm.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        answer = self._processor.decode(output_ids[0], skip_special_tokens=True)
        return answer

    # ------------------------------------------------------------------
    # 性能统计
    # ------------------------------------------------------------------

    def get_timing_stats(self) -> Dict[str, float]:
        """返回各模块平均耗时（毫秒）"""
        stats = {}
        for k, v in self._timing.items():
            if v:
                stats[f"{k}_ms_mean"] = sum(v) / len(v) * 1000
                stats[f"{k}_ms_total"] = sum(v) * 1000
        return stats

    def get_full_stats(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.get_stats(),
            "cache": self.cache.get_stats(),
            "zip_mean_ratio": self.zipper.mean_compression_ratio,
            "timing": self.get_timing_stats(),
        }


# ------------------------------------------------------------------
# Mock 模型（用于无 GPU 环境的单元测试）
# ------------------------------------------------------------------

class _MockViT(nn.Module):
    """Mock ViT 编码器，输出随机特征"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        N = 256  # 16×16 patches
        d = 1152
        return torch.randn(B, N + 1, d, device=x.device)


class _MockVGGT(nn.Module):
    """Mock VGGT，输出随机 3D 特征"""
    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]
        return {
            "scene_feat": torch.randn(B, 1024, device=x.device),
            "depth": torch.rand(H, W, device=x.device) * 10.0,
            "conf": torch.rand(H, W, device=x.device),
        }


class _MockVLM:
    """Mock VLM，返回固定答案"""
    def generate(self, query: str) -> str:
        return f"[Mock Answer] The answer to '{query[:30]}...' is approximately 2.5 meters."
