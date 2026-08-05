# -*- coding: utf-8 -*-
"""T3：模型变慢 / 不可用 / 队列满 时的降级路径验证。

全部用 MockModel 故障注入，无 torch、Windows 可直接 pytest。每种故障都断言：
- 及时进入 CODE_FALLBACK；
- 返回符合接口的 EdgeResult（output_valid、字段齐全、版本标记）；
- 记录内部 fallback_reason；
- 不阻塞后续感知窗口。
"""
from __future__ import annotations

import time

import pytest

from .code_fallback import CodeFallbackRunner
from .config import ClosedLoopConfig
from .jitter_source import Emission, run_schedule
from .model import (
    EXECUTION_CODE_FALLBACK,
    EXECUTION_LOCAL_MODEL,
    REASON_BREAKER_OPEN,
    REASON_CODE_FALLBACK_FAILED,
    REASON_INFERENCE_TIMEOUT,
    REASON_MODEL_INFERENCE_FAILED,
    REASON_MODEL_OUTPUT_INVALID,
    REASON_QUEUE_FULL,
    WindowAggregate,
)
from .model_adapter import MOCK_MODEL_VERSION, MockModel
from .pipeline import ClosedLoopPipeline
from .test_support import RecordCollector, sample_perceptions

PERCS = sample_perceptions(per_category=3, seed=42)


def _fault_cfg():
    cfg = ClosedLoopConfig()
    cfg.window.length_seconds = 0.2
    cfg.window.expected_samples = 4
    cfg.window.min_samples_for_full = 3
    # 默认容量 2：队列满相关的测试会显式改回 1；其余测试避免 flush 竞态误伤
    cfg.queue.capacity = 2
    return cfg


def _emissions(n_windows: int, window: float = 0.2):
    # 每个窗口在起点 +0.05 处给一条样本，保证各窗口独立
    return [Emission(round(w * window + 0.05, 3), PERCS[i % len(PERCS)])
            for i, w in enumerate(range(n_windows))]


def _run(cfg, adapter, schedule):
    collector = RecordCollector()
    pipeline = ClosedLoopPipeline(cfg, adapter, collector)
    pipeline.start()
    run_schedule(schedule, pipeline)
    pipeline.flush()
    pipeline.wait_idle(timeout_s=10)
    pipeline.stop()
    return collector.get(), pipeline


def _assert_clean_fallback(rec, expected_reason):
    assert rec.execution_mode == EXECUTION_CODE_FALLBACK
    assert rec.fallback_reason == expected_reason
    assert rec.output_valid is True
    assert rec.edge_result in ("normal", "warning", "fault")
    assert rec.edge_risk_level in ("low", "medium", "high")
    assert rec.confidence is not None and 0.0 <= rec.confidence <= 1.0
    assert rec.model_version.startswith("edge_rule_")


# ---------- 正常路径（正对照） ----------

def test_healthy_model_local_path():
    cfg = _fault_cfg()
    adapter = MockModel(latency_ms=2.0)
    records, pipeline = _run(cfg, adapter, _emissions(3))
    non_empty = [r for r in records if not r.is_empty]
    assert len(non_empty) == 3
    assert all(r.execution_mode == EXECUTION_LOCAL_MODEL for r in non_empty)
    assert all(r.output_valid for r in non_empty)
    assert all(r.model_version == MOCK_MODEL_VERSION for r in non_empty)
    assert all(r.fallback_reason is None for r in non_empty)


# ---------- 五种故障路径 ----------

def test_inference_timeout_falls_back():
    cfg = _fault_cfg()
    cfg.timeout.inference_ms = 50
    adapter = MockModel(fail_mode="slow", slow_sleep_s=30.0)
    records, _p = _run(cfg, adapter, _emissions(3))
    recs = [r for r in records if not r.is_empty]
    assert len(recs) == 3
    for r in recs:
        _assert_clean_fallback(r, REASON_INFERENCE_TIMEOUT)


def test_exception_falls_back():
    cfg = _fault_cfg()
    adapter = MockModel(latency_ms=2.0, fail_mode="exception")
    records, _p = _run(cfg, adapter, _emissions(3))
    recs = [r for r in records if not r.is_empty]
    assert len(recs) == 3
    for r in recs:
        _assert_clean_fallback(r, REASON_MODEL_INFERENCE_FAILED)


def test_invalid_json_falls_back():
    cfg = _fault_cfg()
    adapter = MockModel(latency_ms=2.0, fail_mode="invalid_json")
    records, _p = _run(cfg, adapter, _emissions(3))
    recs = [r for r in records if not r.is_empty]
    for r in recs:
        _assert_clean_fallback(r, REASON_MODEL_OUTPUT_INVALID)


def test_blank_output_falls_back():
    cfg = _fault_cfg()
    adapter = MockModel(latency_ms=2.0, fail_mode="blank")
    records, _p = _run(cfg, adapter, _emissions(3))
    recs = [r for r in records if not r.is_empty]
    for r in recs:
        _assert_clean_fallback(r, REASON_MODEL_OUTPUT_INVALID)


def test_inference_failure_does_not_block_following_windows():
    # 交替：健康 → 异常 → 健康，失败的窗口不影响后续窗口
    from .model_adapter import InferenceOutcome

    class Alternating(MockModel):
        def __init__(self):
            super().__init__(latency_ms=2.0)
            self._call = 0

        def infer(self, model_input):
            self._call += 1
            if self._call % 2 == 0:
                return super().infer(model_input)
            return InferenceOutcome(success=False, error="injected")

    cfg = _fault_cfg()
    records, _p = _run(cfg, Alternating(), _emissions(4))
    recs = [r for r in records if not r.is_empty]
    assert len(recs) == 4
    modes = [r.execution_mode for r in recs]
    assert modes.count(EXECUTION_CODE_FALLBACK) >= 1
    assert modes.count(EXECUTION_LOCAL_MODEL) >= 2  # 失败后恢复，后续窗口仍走模型
    # 最后一个窗口是 LOCAL_MODEL（不被前面的失败长期阻塞）
    assert recs[-1].execution_mode == EXECUTION_LOCAL_MODEL


# ---------- 队列满：两种策略 ----------

def test_queue_full_drop_falls_back_immediately():
    cfg = _fault_cfg()
    cfg.queue.capacity = 1
    cfg.queue.full_policy = "drop_current_to_fallback"
    # 服务 0.6s > 到达 0.2s → 必然积压
    adapter = MockModel(latency_ms=600.0)
    records, pipeline = _run(cfg, adapter, _emissions(5))
    recs = [r for r in records if not r.is_empty]
    assert len(recs) == 5
    full = [r for r in recs if r.fallback_reason == REASON_QUEUE_FULL]
    assert len(full) >= 1
    for r in full:
        _assert_clean_fallback(r, REASON_QUEUE_FULL)
    assert pipeline.max_observed_queued <= cfg.queue.capacity


def test_replace_oldest_pending_keeps_newest():
    cfg = _fault_cfg()
    cfg.queue.capacity = 1
    cfg.queue.full_policy = "replace_oldest_pending"
    adapter = MockModel(latency_ms=600.0)
    records, _p = _run(cfg, adapter, _emissions(5))
    recs = [r for r in records if not r.is_empty]
    full = [r for r in recs if r.fallback_reason == REASON_QUEUE_FULL]
    local = [r for r in recs if r.execution_mode == EXECUTION_LOCAL_MODEL]
    assert len(full) >= 1 and len(local) >= 1
    # 被替换的一定是更老的窗口（最新窗口被保留排队）
    max_local = max(r.window_id for r in local)
    assert any(r.window_id < max_local for r in full), "应替换旧的、保留新的"


# ---------- 熔断与恢复 ----------

def test_circuit_breaker_opens_then_recovers_after_probe():
    cfg = _fault_cfg()
    cfg.breaker.enabled = True
    cfg.breaker.consecutive_failure_threshold = 2
    cfg.breaker.recovery_probe_interval_s = 0.3
    adapter = MockModel(latency_ms=2.0, fail_mode="exception")

    collector = RecordCollector()
    pipeline = ClosedLoopPipeline(cfg, adapter, collector)
    pipeline.start()
    # 阶段1：前三个窗口（w0,w1 失败触发熔断；w2 立即 flush，使其在熔断开启期内
    # 被处理 → BREAKER_OPEN，而不是等探测期过后变成成功探测）
    run_schedule(_emissions(3), pipeline)
    pipeline.flush()
    assert pipeline.wait_idle(timeout_s=5)
    # 阶段2：等熔断探测期过后，恢复为健康模型，用同一发送方投递一个新窗口作为探测
    time.sleep(0.5)
    adapter.fail_mode = "none"
    pipeline.ingest("sender_benchmark_01", PERCS[0])
    pipeline.flush()
    pipeline.wait_idle(timeout_s=5)
    pipeline.stop()
    records = collector.get()

    recs = [r for r in records if not r.is_empty]
    opened = [r for r in recs if r.fallback_reason == REASON_BREAKER_OPEN]
    assert len(opened) >= 1, "熔断打开后应产生 BREAKER_OPEN 降级"
    recovered = [r for r in recs if r.execution_mode == EXECUTION_LOCAL_MODEL]
    assert len(recovered) >= 1, "熔断恢复探测成功后应回到模型路线"
    # 恢复的 LOCAL_MODEL 记录出现在 BREAKER_OPEN 之后
    last_open_idx = max(i for i, r in enumerate(recs) if r in opened)
    assert any(r.execution_mode == EXECUTION_LOCAL_MODEL for r in recs[last_open_idx + 1:])


# ---------- 两条路线都失败 ----------

def test_both_routes_failed_marks_code_fallback_failed():
    cfg = _fault_cfg()
    adapter = MockModel(latency_ms=2.0)
    collector = RecordCollector()
    pipeline = ClosedLoopPipeline(cfg, adapter, collector)
    malformed = WindowAggregate(
        sender_id="s1", window_id=99, window_start_ns=0, window_end_ns=int(1e9),
        close_ts_ns=int(1e9), expected_samples=20, sample_count=5,
        missing_ratio=0.75, sparse=True, is_empty=False,
        payload={"no_features": 1},  # 代码规则无法判断 → 抛错
    )
    pipeline._run_fallback(malformed, REASON_MODEL_INFERENCE_FAILED)
    rec = collector.get()[0]
    assert rec.fallback_reason == REASON_CODE_FALLBACK_FAILED
    assert rec.output_valid is False
    assert "MODEL_INFERENCE_FAILED" in rec.note


def test_fallback_runner_unit_deterministic_and_guarded():
    runner = CodeFallbackRunner("edge_rule_v1.0")
    a = runner.run(PERCS[0])
    b = runner.run(PERCS[0])
    assert a.as_dict() == b.as_dict()  # 相同输入 → 相同输出
    assert a.edge_result in ("normal", "warning", "fault")
    assert a.model_version == "edge_rule_v1.0"
    with pytest.raises(ValueError):
        runner.run({"no_features": 1})
