# -*- coding: utf-8 -*-
"""模型适配器接口 + MockModel（无 torch，Windows 上可跑窗口/故障测试）。

MockModel 的故障注入由 fail_mode 控制：
- none:      正常，输出与输入一致的合法 JSON
- exception: 抛错 → MODEL_INFERENCE_FAILED
- slow:      长 sleep（slow_sleep_s）→ 超时
- invalid_json: 输出非 JSON 文本 → MODEL_OUTPUT_INVALID
- blank:       输出空文本 → MODEL_OUTPUT_INVALID

RandomModel 是另一种注入：按 fail_rate 随机失败，用于统计性场景。
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .code_fallback import classify_bearing

MOCK_MODEL_VERSION = "edge_mock_v1.0"


@dataclass
class InferenceOutcome:
    success: bool
    timed_out: bool = False
    text: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class InferenceAdapter:
    """模型推理接口。infer 返回 InferenceOutcome，超时由调用方（worker）用子线程判定。"""

    model_version: str = "unknown"  # 记录到 RunRecord.model_version

    def infer(self, model_input: dict) -> InferenceOutcome:
        raise NotImplementedError


def mock_json_output(model_input: dict) -> str:
    """MockModel 正常路径：根据输入特征生成与 edge-model-output/1.0 一致的 JSON。"""
    edge, risk = classify_bearing(model_input)
    flags = (model_input.get("perception_quality") or {}).get("flags") or []
    score = {"normal": 0.6, "warning": 0.7, "fault": 0.85}[edge]
    if "DEVICE_NOT_RUNNING" in flags:
        score = round(score * 0.9, 3)
    return json.dumps({
        "edge_result": edge,
        "edge_risk_level": risk,
        "confidence": score,
        "reason_codes": ["MOCK_OK"],
    }, ensure_ascii=False)


class MockModel(InferenceAdapter):
    model_version = MOCK_MODEL_VERSION

    def __init__(self, latency_ms: float = 5.0, jitter_ms: float = 0.0,
                 fail_mode: str = "none", fail_rate: float = 0.0,
                 slow_sleep_s: float = 30.0, seed: Optional[int] = None):
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.fail_mode = fail_mode
        self.fail_rate = fail_rate
        self.slow_sleep_s = slow_sleep_s
        self._rng = random.Random(seed) if seed is not None else random.Random()
        if fail_mode not in ("none", "exception", "slow", "invalid_json", "blank"):
            raise ValueError("unknown fail_mode: %s" % fail_mode)

    def _sleep(self):
        if self.fail_mode == "slow":
            time.sleep(self.slow_sleep_s)
            return
        latency_s = self.latency_ms / 1000.0
        if self.jitter_ms:
            latency_s += self._rng.uniform(-self.jitter_ms, self.jitter_ms) / 1000.0
            latency_s = max(0.0, latency_s)
        time.sleep(latency_s)

    def infer(self, model_input: dict) -> InferenceOutcome:
        t0 = time.monotonic()
        self._sleep()
        latency_ms = (time.monotonic() - t0) * 1000.0

        if self.fail_mode == "exception":
            return InferenceOutcome(success=False, latency_ms=latency_ms, error="mock_forced_exception")
        if self.fail_rate and self._rng.random() < self.fail_rate:
            return InferenceOutcome(success=False, latency_ms=latency_ms, error="mock_random_failure")
        if self.fail_mode == "invalid_json":
            return InferenceOutcome(success=True, text="这是模型给的解释文字，不是 JSON", latency_ms=latency_ms)
        if self.fail_mode == "blank":
            return InferenceOutcome(success=True, text="   ", latency_ms=latency_ms)

        return InferenceOutcome(success=True, text=mock_json_output(model_input), latency_ms=latency_ms)
