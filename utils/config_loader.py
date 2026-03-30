"""配置文件加载工具（支持 YAML + 命令行覆盖）"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        config_path: YAML 文件路径

    Returns:
        配置字典
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[Config] WARNING: 配置文件不存在 {path}，使用默认配置")
        return {}

    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    except ImportError:
        print("[Config] WARNING: PyYAML 未安装，使用空配置")
        return {}


def merge_config(base: Dict, override: Dict) -> Dict:
    """递归合并两个配置字典（override 优先）"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result
