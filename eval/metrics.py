"""
评估指标工具库
==============
包含：
- EM（精确匹配）
- BLEU-4
- Accuracy（多选/开放式）
- Acc@IoU（3D 定位）
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Union


# ------------------------------------------------------------------
# 文本规范化
# ------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """小写、去标点、去冠词、去多余空格"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ------------------------------------------------------------------
# 精确匹配（EM）
# ------------------------------------------------------------------

def compute_exact_match(
    predictions: List[str],
    references: List[Union[str, List[str]]],
) -> float:
    """
    计算精确匹配率。

    Args:
        predictions: 模型预测答案列表
        references:  参考答案列表（每个样本可有多个参考答案）

    Returns:
        EM 分数（0~1）
    """
    assert len(predictions) == len(references)
    correct = 0
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        pred_norm = normalize_answer(pred)
        if any(normalize_answer(r) == pred_norm for r in refs):
            correct += 1
    return correct / len(predictions) if predictions else 0.0


# ------------------------------------------------------------------
# BLEU-4
# ------------------------------------------------------------------

def compute_bleu4(
    predictions: List[str],
    references: List[Union[str, List[str]]],
) -> float:
    """
    计算 BLEU-4 分数（简化实现，不依赖 nltk）。
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tokenized = []
        hyps_tokenized = []
        for pred, refs in zip(predictions, references):
            if isinstance(refs, str):
                refs = [refs]
            refs_tokenized.append([normalize_answer(r).split() for r in refs])
            hyps_tokenized.append(normalize_answer(pred).split())
        sf = SmoothingFunction().method1
        return corpus_bleu(refs_tokenized, hyps_tokenized, smoothing_function=sf)
    except ImportError:
        # 简化版 BLEU-4（不含 brevity penalty）
        return _simple_bleu4(predictions, references)


def _simple_bleu4(
    predictions: List[str],
    references: List[Union[str, List[str]]],
) -> float:
    """简化 BLEU-4，仅用于无 nltk 环境"""
    from math import exp, log

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    total_score = 0.0
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        pred_tokens = normalize_answer(pred).split()
        ref_tokens_list = [normalize_answer(r).split() for r in refs]

        scores = []
        for n in range(1, 5):
            pred_ng = ngrams(pred_tokens, n)
            max_ref_ng = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ng = ngrams(ref_tokens, n)
                for k in ref_ng:
                    max_ref_ng[k] = max(max_ref_ng[k], ref_ng[k])
            clipped = sum(min(cnt, max_ref_ng[ng]) for ng, cnt in pred_ng.items())
            total_pred = max(sum(pred_ng.values()), 1)
            scores.append(clipped / total_pred)

        if all(s > 0 for s in scores):
            bleu = exp(sum(log(s) for s in scores) / 4)
        else:
            bleu = 0.0
        total_score += bleu

    return total_score / len(predictions) if predictions else 0.0


# ------------------------------------------------------------------
# Accuracy（开放式问答）
# ------------------------------------------------------------------

def compute_accuracy(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    soft_match: bool = True,
) -> float:
    """
    计算准确率。

    Args:
        soft_match: True 时使用 token-level F1 软匹配（类似 SQuAD）
    """
    if not soft_match:
        return compute_exact_match(predictions, references)

    scores = []
    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        best_f1 = max(_token_f1(pred, r) for r in refs)
        scores.append(best_f1)
    return sum(scores) / len(scores) if scores else 0.0


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 分数"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ------------------------------------------------------------------
# 3D 定位 Acc@IoU
# ------------------------------------------------------------------

def compute_acc_at_iou(
    pred_bboxes: list,
    gt_bboxes: list,
    iou_thresholds: List[float] = None,
) -> dict:
    """
    计算 3D 边界框定位准确率 Acc@IoU。

    Args:
        pred_bboxes: 预测边界框列表，每个为 [x, y, z, dx, dy, dz]
        gt_bboxes:   真实边界框列表
        iou_thresholds: IoU 阈值列表，默认 [0.25, 0.5]

    Returns:
        dict: {"acc@0.25": float, "acc@0.5": float}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5]

    results = {}
    for thresh in iou_thresholds:
        correct = sum(
            1 for pred, gt in zip(pred_bboxes, gt_bboxes)
            if pred is not None and gt is not None and _iou_3d(pred, gt) >= thresh
        )
        results[f"acc@{thresh}"] = correct / len(pred_bboxes) if pred_bboxes else 0.0

    return results


def _iou_3d(box1: list, box2: list) -> float:
    """
    计算两个轴对齐 3D 边界框的 IoU。
    box 格式：[cx, cy, cz, dx, dy, dz]（中心 + 尺寸）
    """
    import numpy as np

    b1 = np.array(box1, dtype=np.float32)
    b2 = np.array(box2, dtype=np.float32)

    # 转换为 min/max 格式
    b1_min = b1[:3] - b1[3:] / 2
    b1_max = b1[:3] + b1[3:] / 2
    b2_min = b2[:3] - b2[3:] / 2
    b2_max = b2[:3] + b2[3:] / 2

    inter_min = np.maximum(b1_min, b2_min)
    inter_max = np.minimum(b1_max, b2_max)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = inter_dims.prod()

    vol1 = b1[3:].prod()
    vol2 = b2[3:].prod()
    union_vol = vol1 + vol2 - inter_vol

    return float(inter_vol / (union_vol + 1e-8))
