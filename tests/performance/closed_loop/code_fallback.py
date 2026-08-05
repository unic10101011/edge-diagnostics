# -*- coding: utf-8 -*-
"""代码替代执行器：确定性规则，输出与模型路线完全一致的扁平 4 字段 EdgeResult。

规则质量明确不是本验证工具的目标（文档规定第一阶段代码规则只要能跑通整条链），
但必须：相同输入 → 相同输出；明确版本；不编造缺失测量值。
"""
from __future__ import annotations

from typing import Optional, Tuple

from .model import EdgeResult


def classify_bearing(payload: dict) -> Tuple[str, str]:
    """确定性分类：(edge_result, edge_risk_level)。阈值与 generate_test_inputs 三类输入对应。"""
    features = payload.get("features") or {}
    flags = (payload.get("perception_quality") or {}).get("flags") or []

    if "DEVICE_NOT_RUNNING" in flags:
        # 设备未运行不是轴承故障，视为 normal；仍降低 confidence（由调用方处理）
        return "normal", "low"

    vib = features.get("vibration") or {}
    kurt = vib.get("kurtosis")
    peak = vib.get("absolute_peak")
    imbal = (features.get("current_relationship") or {}).get("current_imbalance_ratio")

    def _num(x: object) -> Optional[float]:
        return float(x) if isinstance(x, (int, float)) and x == x else None

    k, p, i = _num(kurt), _num(peak), _num(imbal)

    # 无任何可判断特征时不编造测量值，但也不至于报错——返回 normal/low
    if k is None and p is None and i is None:
        return "normal", "low"

    fault = (k is not None and k >= 8.0) or (p is not None and p >= 6.0) or (i is not None and i >= 0.25)
    warning = (k is not None and k >= 4.5) or (p is not None and p >= 3.0) or (i is not None and i >= 0.08)

    if fault:
        return "fault", "high"
    if warning:
        return "warning", "medium"
    return "normal", "low"


class CodeFallbackRunner:
    def __init__(self, rule_version: str = "edge_rule_v1.0"):
        self.rule_version = rule_version

    def run(self, payload: dict) -> EdgeResult:
        if not isinstance(payload, dict) or "features" not in payload:
            raise ValueError("code_fallback: payload missing 'features'")
        edge, risk = classify_bearing(payload)
        # 规则分数：确定性的类别基准分数（占位算法，非模型置信度）
        score = {"normal": 0.6, "warning": 0.7, "fault": 0.85}[edge]
        # DEVICE_NOT_RUNNING 时降低分数表达质量 warning 的影响
        flags = (payload.get("perception_quality") or {}).get("flags") or []
        if "DEVICE_NOT_RUNNING" in flags:
            score = round(score * 0.9, 3)
        return EdgeResult(edge_result=edge, confidence=score, edge_risk_level=risk,
                          model_version=self.rule_version)
