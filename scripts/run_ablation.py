"""
消融实验统一入口脚本
====================
自动运行 tech_desgin.md 中规划的 5 组消融实验（A~E），
汇总结果并生成对比表格。

用法：
    python scripts/run_ablation.py \
        --data_root /data/spar7m \
        --output_dir results/ablation \
        --ablation all          # 运行全部消融
        # 或 --ablation A,B,C  # 运行指定消融
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.eval_spar7m import evaluate_spar7m
from eval.eval_scanqa import evaluate_scanqa
from models.stream_spatial_vlm import StreamSpatialVLM, StreamSpatialConfig
from models.gate_2d3d import GateConfig
from models.zip_3d2d import ZipConfig
from models.kv_cache import KVCacheConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--data_root_spar7m", type=str, required=True)
    parser.add_argument("--data_root_scanqa", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    parser.add_argument("--ablation", type=str, default="all",
                        help="运行哪些消融：all 或 A,B,C,D,E 的子集")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="每次评估的最大样本数（加速消融）")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_model(cfg: StreamSpatialConfig) -> StreamSpatialVLM:
    model = StreamSpatialVLM(cfg)
    model.load_models()
    return model


def run_eval(model, args, datasets: List[str] = None) -> Dict[str, Any]:
    """对指定数据集运行评估，返回合并结果"""
    if datasets is None:
        datasets = ["spar7m"]
    results = {}

    if "spar7m" in datasets:
        r = evaluate_spar7m(
            model, args.data_root_spar7m,
            max_samples=args.max_samples, device=args.device,
        )
        results["spar7m_accuracy"] = r["accuracy"]
        results["spar7m_em"] = r["exact_match"]
        results["fps"] = r["fps"]
        results["peak_memory_gb"] = r["peak_memory_gb"]
        results["gate_trigger_rate"] = r.get("gate_trigger_rate")
        results["zip_compression_ratio"] = r.get("zip_compression_ratio")

    if "scanqa" in datasets and args.data_root_scanqa:
        r = evaluate_scanqa(
            model, args.data_root_scanqa,
            max_samples=args.max_samples, device=args.device,
        )
        results["scanqa_em"] = r["exact_match"]
        results["scanqa_bleu4"] = r["bleu4"]

    return results


# ------------------------------------------------------------------
# 消融 A：各核心模块贡献分析
# ------------------------------------------------------------------

def ablation_A(args) -> List[Dict]:
    """
    A0: VG-LLM 全量（无任何优化）
    A1: + 增量缓存
    A2: + Gate + 增量缓存
    A3: + Zip + 增量缓存
    A4: 完整框架（Gate + Zip + 增量缓存）
    """
    print("\n" + "=" * 50)
    print("消融 A：各核心模块贡献分析")
    print("=" * 50)

    configs = [
        ("A0: VG-LLM 全量",      False, False, False),
        ("A1: + 增量缓存",        False, False, True),
        ("A2: + Gate",            True,  False, True),
        ("A3: + Zip",             False, True,  True),
        ("A4: 完整框架 (ours)",   True,  True,  True),
    ]

    results = []
    for name, use_gate, use_zip, use_incr in configs:
        print(f"\n  运行: {name}")
        cfg = StreamSpatialConfig(
            use_gate=use_gate,
            use_zip=use_zip,
            use_incremental=use_incr,
        )
        model = build_model(cfg)
        r = run_eval(model, args)
        r["config"] = name
        r["use_gate"] = use_gate
        r["use_zip"] = use_zip
        r["use_incremental"] = use_incr
        results.append(r)
        print(f"    SPAR-7M Acc: {r.get('spar7m_accuracy', 'N/A'):.4f}  "
              f"FPS: {r.get('fps', 'N/A'):.2f}")

    return results


# ------------------------------------------------------------------
# 消融 B：门控阈值 τ 敏感性
# ------------------------------------------------------------------

def ablation_B(args) -> List[Dict]:
    """τ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}"""
    print("\n" + "=" * 50)
    print("消融 B：门控阈值 τ 敏感性分析")
    print("=" * 50)

    tau_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = []

    for tau in tau_list:
        print(f"\n  τ = {tau}")
        cfg = StreamSpatialConfig(gate=GateConfig(tau=tau))
        model = build_model(cfg)
        r = run_eval(model, args)
        r["tau"] = tau
        r["config"] = f"τ={tau}"
        results.append(r)
        print(f"    Acc: {r.get('spar7m_accuracy', 'N/A'):.4f}  "
              f"Trigger Rate: {r.get('gate_trigger_rate', 'N/A'):.3f}  "
              f"FPS: {r.get('fps', 'N/A'):.2f}")

    return results


# ------------------------------------------------------------------
# 消融 C：Token 保留率 r 敏感性
# ------------------------------------------------------------------

def ablation_C(args) -> List[Dict]:
    """r ∈ {0.25, 0.50, 0.75, 1.00}"""
    print("\n" + "=" * 50)
    print("消融 C：Token 保留率 r 敏感性分析")
    print("=" * 50)

    ratio_list = [0.25, 0.50, 0.75, 1.00]
    results = []

    for r_val in ratio_list:
        print(f"\n  r = {r_val}")
        cfg = StreamSpatialConfig(zip=ZipConfig(keep_ratio=r_val))
        model = build_model(cfg)
        r = run_eval(model, args, datasets=["spar7m", "scanqa"])
        r["keep_ratio"] = r_val
        r["config"] = f"r={r_val}"
        results.append(r)
        print(f"    SPAR-7M Acc: {r.get('spar7m_accuracy', 'N/A'):.4f}  "
              f"ScanQA EM: {r.get('scanqa_em', 'N/A'):.4f}  "
              f"FPS: {r.get('fps', 'N/A'):.2f}")

    return results


# ------------------------------------------------------------------
# 消融 D：KV 缓存窗口大小 w
# ------------------------------------------------------------------

def ablation_D(args) -> List[Dict]:
    """w ∈ {4, 8, 16, 32}"""
    print("\n" + "=" * 50)
    print("消融 D：KV 缓存窗口大小 w 分析")
    print("=" * 50)

    window_list = [4, 8, 16, 32]
    results = []

    for w in window_list:
        print(f"\n  w = {w}")
        cfg = StreamSpatialConfig(cache=KVCacheConfig(window_size=w))
        model = build_model(cfg)
        r = run_eval(model, args)
        r["window_size"] = w
        r["config"] = f"w={w}"
        results.append(r)
        print(f"    Acc: {r.get('spar7m_accuracy', 'N/A'):.4f}  "
              f"Mem: {r.get('peak_memory_gb', 'N/A'):.2f} GB  "
              f"FPS: {r.get('fps', 'N/A'):.2f}")

    return results


# ------------------------------------------------------------------
# 消融 E：重要性图权重系数 (α, β)
# ------------------------------------------------------------------

def ablation_E(args) -> List[Dict]:
    """(α, β) ∈ {(1,0), (0.6,0.4), (0.4,0.6), (0,1)}"""
    print("\n" + "=" * 50)
    print("消融 E：重要性图权重系数 (α, β) 分析")
    print("=" * 50)

    ab_list = [
        (1.0, 0.0, "仅深度方差"),
        (0.6, 0.4, "默认"),
        (0.4, 0.6, "偏位姿"),
        (0.0, 1.0, "仅位姿不确定性"),
    ]
    results = []

    for alpha, beta, desc in ab_list:
        print(f"\n  α={alpha}, β={beta} ({desc})")
        cfg = StreamSpatialConfig(zip=ZipConfig(alpha=alpha, beta=beta))
        model = build_model(cfg)
        r = run_eval(model, args)
        r["alpha"] = alpha
        r["beta"] = beta
        r["config"] = f"α={alpha},β={beta} ({desc})"
        results.append(r)
        print(f"    Acc: {r.get('spar7m_accuracy', 'N/A'):.4f}")

    return results


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_set = set(args.ablation.upper().split(",")) if args.ablation != "all" \
        else {"A", "B", "C", "D", "E"}

    all_results = {}

    if "A" in ablation_set:
        all_results["ablation_A"] = ablation_A(args)
    if "B" in ablation_set:
        all_results["ablation_B"] = ablation_B(args)
    if "C" in ablation_set:
        all_results["ablation_C"] = ablation_C(args)
    if "D" in ablation_set:
        all_results["ablation_D"] = ablation_D(args)
    if "E" in ablation_set:
        all_results["ablation_E"] = ablation_E(args)

    # 保存完整结果
    result_path = output_dir / "ablation_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 生成 Markdown 表格
    _generate_markdown_tables(all_results, output_dir)

    print(f"\n消融实验完成！结果保存至: {output_dir}")


def _generate_markdown_tables(results: dict, output_dir: Path):
    """将消融结果生成 Markdown 格式表格"""
    lines = ["# 消融实验结果汇总\n"]

    table_configs = {
        "ablation_A": {
            "title": "## 消融 A：各核心模块贡献",
            "cols": ["config", "spar7m_accuracy", "fps", "peak_memory_gb"],
            "headers": ["配置", "SPAR-7M Acc", "FPS", "峰值显存(GB)"],
        },
        "ablation_B": {
            "title": "## 消融 B：门控阈值 τ 敏感性",
            "cols": ["tau", "gate_trigger_rate", "spar7m_accuracy", "fps"],
            "headers": ["τ", "3D触发率", "SPAR-7M Acc", "FPS"],
        },
        "ablation_C": {
            "title": "## 消融 C：Token 保留率 r 敏感性",
            "cols": ["keep_ratio", "spar7m_accuracy", "scanqa_em", "fps"],
            "headers": ["r", "SPAR-7M Acc", "ScanQA EM", "FPS"],
        },
        "ablation_D": {
            "title": "## 消融 D：KV 缓存窗口大小 w",
            "cols": ["window_size", "spar7m_accuracy", "peak_memory_gb", "fps"],
            "headers": ["w", "SPAR-7M Acc", "峰值显存(GB)", "FPS"],
        },
        "ablation_E": {
            "title": "## 消融 E：重要性图权重系数",
            "cols": ["config", "spar7m_accuracy"],
            "headers": ["(α, β)", "SPAR-7M Acc"],
        },
    }

    for key, cfg in table_configs.items():
        if key not in results:
            continue
        lines.append(cfg["title"])
        lines.append("")
        headers = cfg["headers"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in results[key]:
            vals = []
            for col in cfg["cols"]:
                v = row.get(col, "-")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    md_path = output_dir / "ablation_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown 表格已保存至: {md_path}")


if __name__ == "__main__":
    main()
