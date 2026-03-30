"""
推理速度与显存分析工具
======================
用于评估脚本中统一记录 FPS 和峰值显存。
"""

from __future__ import annotations

import time
from typing import List

import torch


class SpeedProfiler:
    """
    推理速度与显存分析器。

    用法：
        profiler = SpeedProfiler(device="cuda")
        for sample in dataset:
            profiler.start()
            model(sample)
            profiler.stop()
        print(profiler.fps, profiler.peak_memory_gb)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._start_times: List[float] = []
        self._end_times: List[float] = []
        self._peak_memory: float = 0.0

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    def start(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._start_times.append(time.perf_counter())

    def stop(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            self._peak_memory = max(self._peak_memory, peak)
        self._end_times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """平均每秒处理样本数"""
        if not self._start_times or not self._end_times:
            return 0.0
        total_time = sum(
            e - s for s, e in zip(self._start_times, self._end_times)
        )
        return len(self._start_times) / max(total_time, 1e-8)

    @property
    def mean_latency_ms(self) -> float:
        """平均单样本延迟（毫秒）"""
        if not self._start_times:
            return 0.0
        latencies = [
            (e - s) * 1000
            for s, e in zip(self._start_times, self._end_times)
        ]
        return sum(latencies) / len(latencies)

    @property
    def peak_memory_gb(self) -> float:
        """峰值显存（GB）"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        return self._peak_memory

    def reset(self):
        self._start_times.clear()
        self._end_times.clear()
        self._peak_memory = 0.0
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def summary(self) -> dict:
        return {
            "num_samples": len(self._start_times),
            "fps": self.fps,
            "mean_latency_ms": self.mean_latency_ms,
            "peak_memory_gb": self.peak_memory_gb,
        }
