# -*- coding: utf-8 -*-
"""闭环验证工具的共享数据结构。

与《边缘模型运行实现流程.md》保持一致：
- EdgeResult 为对外统一扁平 4 字段；
- execution_mode / fallback_reason 等属于内部运行记录，不进 EdgeResult。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 枚举值对齐已协商接口（《目前没有做到的事.md》要求封闭枚举）
EDGE_RESULT_VALUES = ("normal", "warning", "fault")
EDGE_RISK_VALUES = ("low", "medium", "high")

EXECUTION_LOCAL_MODEL = "LOCAL_MODEL"
EXECUTION_CODE_FALLBACK = "CODE_FALLBACK"
EXECUTION_NONE = "NONE"  # 空窗口：不调用任何路线

# 内部降级原因（不进入外部 EdgeResult）
REASON_QUEUE_FULL = "QUEUE_FULL"
REASON_QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
REASON_TOTAL_TIMEOUT = "TOTAL_TIMEOUT"
REASON_INFERENCE_TIMEOUT = "TIMEOUT"
REASON_MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
REASON_MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
REASON_BREAKER_OPEN = "BREAKER_OPEN"
REASON_CODE_FALLBACK_FAILED = "CODE_FALLBACK_FAILED"
REASON_NONE = None


@dataclass
class EdgeResult:
    """对外统一结果，扁平 4 字段，与《边缘部分需要的接口.md》一致。"""
    edge_result: str            # normal | warning | fault
    confidence: float           # 0~1；模型路线为未校准诊断分数，规则路线为规则分数
    edge_risk_level: str        # low | medium | high
    model_version: str          # 模型版本 或 edge_rule_xxx

    def as_dict(self) -> Dict[str, Any]:
        return {
            "edge_result": self.edge_result,
            "confidence": self.confidence,
            "edge_risk_level": self.edge_risk_level,
            "model_version": self.model_version,
        }


@dataclass
class WindowAggregate:
    """一个发送方、一个时间窗口内的聚合结果。

    窗口归属按到达时刻（单调时钟）划分，不按 PerceptionResult 的时间戳划分；
    迟到/乱序数据在窗口关闭后到达的按 late_dropped 计数并丢弃。
    """
    sender_id: str
    window_id: int
    window_start_ns: int
    window_end_ns: int
    close_ts_ns: int
    expected_samples: int
    sample_count: int = 0
    first_sample_ts_ns: Optional[int] = None
    last_sample_ts_ns: Optional[int] = None
    missing_ratio: float = 0.0
    quality_status: str = "good"          # good | warning
    quality_flags: List[str] = field(default_factory=list)
    late_dropped_count: int = 0
    is_empty: bool = False                # sample_count == 0
    sparse: bool = False                  # sample_count < min_samples_for_full
    payload: Dict[str, Any] = field(default_factory=dict)  # 交给模型/规则的 PerceptionResult 形输入
    # 内部：提交到队列的时间戳（单调秒）
    submit_ts: Optional[float] = None

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "window_id": self.window_id,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "sample_count": self.sample_count,
            "expected_samples": self.expected_samples,
            "missing_ratio": self.missing_ratio,
            "quality_status": self.quality_status,
            "quality_flags": self.quality_flags,
            "sparse": self.sparse,
            "is_empty": self.is_empty,
            "late_dropped_count": self.late_dropped_count,
        }


@dataclass
class RunRecord:
    """一个窗口从关闭到产出结果的完整事件记录，供测试断言与结果落盘。"""
    sender_id: str
    window_id: int
    window_start_ns: int
    window_end_ns: int
    sample_count: int
    missing_ratio: float
    sparse: bool
    is_empty: bool
    execution_mode: str
    fallback_reason: Optional[str]
    output_valid: bool
    edge_result: Optional[str] = None
    edge_risk_level: Optional[str] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    queue_wait_ms: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    exceeded_total_timeout: bool = False
    late_dropped_count: int = 0
    breaker_state: Optional[str] = None
    note: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "window_id": self.window_id,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "sample_count": self.sample_count,
            "missing_ratio": self.missing_ratio,
            "sparse": self.sparse,
            "is_empty": self.is_empty,
            "execution_mode": self.execution_mode,
            "fallback_reason": self.fallback_reason,
            "output_valid": self.output_valid,
            "edge_result": self.edge_result,
            "edge_risk_level": self.edge_risk_level,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "queue_wait_ms": self.queue_wait_ms,
            "inference_latency_ms": self.inference_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "exceeded_total_timeout": self.exceeded_total_timeout,
            "late_dropped_count": self.late_dropped_count,
            "breaker_state": self.breaker_state,
            "note": self.note,
        }
