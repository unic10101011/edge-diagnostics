# -*- coding: utf-8 -*-
"""代码替代执行器：接口 + 规则输入/输出校验 + 测试规则。

当前规则标记为 edge_rule_test_v1：只证明降级流程可用，不代表业务规则正确。
以下内容确认后才能冻结为正式规则版本：
- 使用哪些 PerceptionResult / 窗口字段；
- normal / warning / fault 阈值；
- 多特征如何合并；
- 缺失特征如何处理；
- 规则分数算法；
- confidence 如何映射；
- 标准测试案例和期望输出。

正式代码先实现：接口、规则输入校验、规则输出校验、规则版本读取。
"""
from __future__ import annotations

from typing import Optional

from .contracts import (
    EDGE_RESULT_VALUES,
    EDGE_RISK_VALUES,
    EdgeResult,
    WindowAggregate,
)


class CodeFallbackRunner:
    """代码替代执行器接口。相同输入必须得到相同输出；输出结构与模型路线一致。"""

    rule_version: str

    def run(self, window: WindowAggregate) -> EdgeResult:
        raise NotImplementedError

    def _validate_input(self, window: WindowAggregate):
        if not isinstance(window.payload, dict) or "features" not in window.payload:
            raise ValueError("code_fallback: payload 缺少 'features'")

    def _validate_output(self, edge: EdgeResult):
        if edge.edge_result not in EDGE_RESULT_VALUES:
            raise ValueError("code_fallback: 非法 edge_result=%s" % edge.edge_result)
        if edge.edge_risk_level not in EDGE_RISK_VALUES:
            raise ValueError("code_fallback: 非法 edge_risk_level=%s" % edge.edge_risk_level)
        if not (isinstance(edge.confidence, (int, float)) and 0.0 <= edge.confidence <= 1.0):
            raise ValueError("code_fallback: 非法 confidence=%s" % edge.confidence)
        if not edge.model_version:
            raise ValueError("code_fallback: model_version 为空")


def _num(x: object) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and x == x else None


def _classify_test_rule(features: dict, flags: list):
    """测试规则分类：(edge_result, edge_risk_level)。确定性、版本化，阈值与
    tests 输入三类对应，仅供降级流程验证，不构成业务规则。"""
    if "DEVICE_NOT_RUNNING" in flags:
        return "normal", "low"

    vib = features.get("vibration") or {}
    kurt, peak = _num(vib.get("kurtosis")), _num(vib.get("absolute_peak"))
    imbal = _num((features.get("current_relationship") or {}).get("current_imbalance_ratio"))

    if kurt is None and peak is None and imbal is None:
        return "normal", "low"

    fault = (kurt is not None and kurt >= 8.0) or (peak is not None and peak >= 6.0) \
        or (imbal is not None and imbal >= 0.25)
    warning = (kurt is not None and kurt >= 4.5) or (peak is not None and peak >= 3.0) \
        or (imbal is not None and imbal >= 0.08)
    if fault:
        return "fault", "high"
    if warning:
        return "warning", "medium"
    return "normal", "low"


class TestRuleRunner(CodeFallbackRunner):
    """edge_rule_test_v1：确定性占位规则，仅供流程验证。"""

    __test__ = False  # 不是 pytest 测试类

    def __init__(self, rule_version: str = "edge_rule_test_v1"):
        self.rule_version = rule_version

    def run(self, window: WindowAggregate) -> EdgeResult:
        self._validate_input(window)
        payload = window.payload
        features = payload.get("features") or {}
        flags = (payload.get("perception_quality") or {}).get("flags") or []
        edge, risk = _classify_test_rule(features, flags)
        score = {"normal": 0.6, "warning": 0.7, "fault": 0.85}[edge]
        if "DEVICE_NOT_RUNNING" in flags:
            score = round(score * 0.9, 3)
        result = EdgeResult(edge_result=edge, confidence=score,
                            edge_risk_level=risk, model_version=self.rule_version)
        self._validate_output(result)
        return result
