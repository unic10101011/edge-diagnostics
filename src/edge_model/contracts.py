# -*- coding: utf-8 -*-
"""公共契约：外部接口、窗口聚合、内部运行记录。

对外（与《边缘部分需要的接口.md》保持一致，不新增字段）：
    EdgeResult      扁平 4 字段
    PacketResult    每包输出 = 包身份 + EdgeResult

内部（只写运行记录，不进外部接口）：
    WindowAggregate 窗口聚合结果与元数据
    RunRecord       一次窗口诊断的完整执行记录
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 外部枚举（封闭）
EDGE_RESULT_VALUES = ("normal", "warning", "fault")
EDGE_RISK_VALUES = ("low", "medium", "high")

EXECUTION_LOCAL_MODEL = "LOCAL_MODEL"
EXECUTION_CODE_FALLBACK = "CODE_FALLBACK"
EXECUTION_NONE = "NONE"  # 空窗口：不调用任何路线

# 内部降级原因
REASON_QUEUE_FULL = "QUEUE_FULL"
REASON_QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
REASON_TOTAL_TIMEOUT = "TOTAL_TIMEOUT"
REASON_INFERENCE_TIMEOUT = "TIMEOUT"
REASON_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
REASON_MODEL_ERROR = "MODEL_ERROR"
REASON_MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
REASON_BREAKER_OPEN = "BREAKER_OPEN"
REASON_CODE_FALLBACK_FAILED = "CODE_FALLBACK_FAILED"


@dataclass
class EdgeResult:
    """对外统一结果（扁平 4 字段，接口冻结）。"""
    edge_result: str            # normal | warning | fault
    confidence: float           # 0~1；模型路线为未校准诊断分数，规则路线为规则分数
    edge_risk_level: str        # low | medium | high
    model_version: str          # 模型版本 或 edge_rule_* 规则版本

    def as_dict(self) -> Dict[str, Any]:
        return {
            "edge_result": self.edge_result,
            "confidence": self.confidence,
            "edge_risk_level": self.edge_risk_level,
            "model_version": self.model_version,
        }


@dataclass
class PacketResult:
    """每包输出：包身份 + 窗口级 EdgeResult（诊断映射给窗口内所有数据包）。

    窗口元数据（window_id/window_start_ns/window_end_ns）只进内部记录，
    不加入外部接口。
    """
    task_id: str
    packet_id: str
    sender_id: str
    sequence_number: int
    edge: EdgeResult

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "packet_id": self.packet_id,
            "sender_id": self.sender_id,
            "sequence_number": self.sequence_number,
            **self.edge.as_dict(),
        }


@dataclass
class WindowAggregate:
    """一个发送方、一个时间窗口内的聚合结果（内部）。"""
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
    quality_status: str = "good"
    quality_flags: List[str] = field(default_factory=list)
    late_dropped_count: int = 0
    is_empty: bool = False
    sparse: bool = False
    # 窗口内全部包的身份（用于把窗口诊断映射回每个数据包）
    included_packets: List[Dict[str, Any]] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)  # 交给模型/规则的聚合输入
    submit_ts: Optional[float] = None  # 提交到模型队列的单调秒

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
            "included_packet_ids": [p["packet_id"] for p in self.included_packets],
        }


@dataclass
class RunRecord:
    """一次窗口诊断的执行记录（内部，只写日志/指标）。"""
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
    packet_count: int = 0
    output_valid: bool = True   # 是否产出合法 EdgeResult（两条路线都失败时为 False）
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
            "packet_count": self.packet_count,
            "output_valid": self.output_valid,
            "execution_mode": self.execution_mode,
            "fallback_reason": self.fallback_reason,
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
