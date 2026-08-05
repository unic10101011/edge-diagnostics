# -*- coding: utf-8 -*-
"""边缘模型运行模块配置。

参数为【第一阶段实测默认值】，数据来源 Qwen2.5-1.5B-Instruct + Transformers
+ RTX 5060 Laptop（见 docs/01-本地实时闭环/边缘模型运行实现流程.md）。不是
永久标准：更换模型后需复用 tests/performance/closed_loop 重测模型相关参数。

命名说明：
    max_waiting_requests  队列中「等待中」任务数上限，不含正在推理的 1 条。
                          语义 = 1 条正在推理 + 最多 N 条等待。
    full_policy           reject：队列满 → 新窗口立即走代码降级；
                          replace：用新窗口替换尚未开始的旧窗口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WindowConfig:
    length_seconds: float = 1.0            # 第一阶段实测默认
    expected_samples_per_window: int = 20  # 发送器 20 包/秒 → 每窗期望 20 条
    min_samples_for_full: int = 15         # 稀疏窗口判定（低于此值标记 WINDOW_SPARSE）
    late_rule: str = "drop_and_count"      # 已关窗口后到达：丢弃并计数（当前唯一实现）
    submit_sparse_window: bool = True


@dataclass
class QueueConfig:
    max_waiting_requests: int = 1          # 等待中任务上限（不含正在推理的 1 条）
    worker_count: int = 1
    full_policy: str = "reject"            # reject | replace


@dataclass
class TimeoutConfig:
    # 超时语义（文档冻结）：
    #   queue_wait_ms   从进入模型队列到获得执行资格
    #   inference_ms    从调用模型服务到等待响应超时（逻辑超时，不终止已开始的 generate）
    #   total_ms        从窗口诊断任务创建到最终模型或降级结果完成
    queue_wait_ms: int = 250               # 第一阶段实测默认
    inference_ms: int = 1500               # 第一阶段实测默认
    total_ms: int = 2000                   # 第一阶段实测默认


@dataclass
class BreakerConfig:
    enabled: bool = True
    consecutive_failure_threshold: int = 5    # 连续失败熔断（Windows 调用侧维护）
    recovery_probe_interval_s: float = 60.0   # 恢复探测周期
    probe_inference_timeout_ms: int = 2000


@dataclass
class ModelClientConfig:
    base_url: str = "http://127.0.0.1:8001"
    infer_path: str = "/infer"
    health_path: str = "/health"
    readiness_path: str = "/readiness"
    connect_timeout_ms: int = 500
    read_timeout_ms: int = 1500            # 与 inference_ms 对齐


@dataclass
class FallbackConfig:
    rule_version: str = "edge_rule_test_v1"   # 待业务规则确认后才冻结为正式版本


@dataclass
class EdgeModelConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    model_client: ModelClientConfig = field(default_factory=ModelClientConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    def validate(self) -> List[str]:
        """启动校验：参数不合法时返回错误列表，由调用方决定是否停止启动。"""
        errors: List[str] = []
        if self.window.length_seconds <= 0:
            errors.append("window.length_seconds 必须为正数")
        if self.window.expected_samples_per_window <= 0:
            errors.append("window.expected_samples_per_window 必须为正数")
        if not (0 < self.window.min_samples_for_full <= self.window.expected_samples_per_window):
            errors.append("min_samples_for_full 必须在 (0, expected_samples_per_window]")
        if self.window.late_rule not in ("drop_and_count",):
            errors.append("late_rule 仅支持 drop_and_count（当前阶段）")
        if self.queue.max_waiting_requests < 1:
            errors.append("queue.max_waiting_requests 必须 >= 1")
        if self.queue.full_policy not in ("reject", "replace"):
            errors.append("queue.full_policy 必须是 reject 或 replace")
        if not (0 < self.timeout.queue_wait_ms < self.timeout.total_ms):
            errors.append("timeout.queue_wait_ms 必须为正数且小于 total_ms")
        if not (0 < self.timeout.inference_ms <= self.timeout.total_ms):
            errors.append("timeout.inference_ms 必须为正数且不超过 total_ms")
        if self.breaker.consecutive_failure_threshold < 1:
            errors.append("breaker.consecutive_failure_threshold 必须 >= 1")
        if self.breaker.recovery_probe_interval_s <= 0:
            errors.append("breaker.recovery_probe_interval_s 必须为正数")
        if not self.fallback.rule_version:
            errors.append("fallback.rule_version 不能为空")
        return errors

    def as_dict(self) -> Dict:
        return {
            "window": _asdict(self.window),
            "queue": _asdict(self.queue),
            "timeout": _asdict(self.timeout),
            "breaker": _asdict(self.breaker),
            "model_client": _asdict(self.model_client),
            "fallback": _asdict(self.fallback),
        }


def _asdict(obj) -> Dict:
    import dataclasses
    return dataclasses.asdict(obj)
