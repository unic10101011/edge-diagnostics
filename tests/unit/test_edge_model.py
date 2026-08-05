# -*- coding: utf-8 -*-
"""src/edge_model 防回归单元测试（Windows 可跑，无 torch/真实 HTTP）。

重点覆盖之前验证阶段发现的 4 个 bug，确保迁移后不回归：
1. 浮点切窗漂移 → 整数纳秒切窗（边界样本归属确定）；
2. flush() 状态未重置 → flush 后同发送方再次 ingest 不崩溃；
3. worker 取队列参数错误导致线程退出 → worker 正常处理；
4. replace 策略任务静默丢失 → 被替换窗口产出 QUEUE_FULL 降级记录。

外加：窗口级诊断 → 每包映射、空窗口跳过、降级路径、熔断恢复、配置校验。
"""
from __future__ import annotations

import time

import pytest

from edge_model.code_fallback import TestRuleRunner
from edge_model.config import EdgeModelConfig
from edge_model.contracts import (
    EXECUTION_CODE_FALLBACK,
    EXECUTION_LOCAL_MODEL,
    EXECUTION_NONE,
    REASON_BREAKER_OPEN,
    REASON_INFERENCE_TIMEOUT,
    REASON_MODEL_UNAVAILABLE,
    REASON_QUEUE_FULL,
    EdgeResult,
)
from edge_model.model_client import ModelInferResult
from edge_model.pipeline import EdgeModelPipeline


# ---------- 工具 ----------

def _perception(packet_id: str, seq: int, sender: str = "s1"):
    return {
        "task_id": "task-1",
        "packet_id": packet_id,
        "sender_id": sender,
        "sequence_number": seq,
        "end_generate_timestamp_ns": 0,
        "perception_quality": {"status": "good", "flags": []},
        "features": {"vibration": {"rms": 0.3, "absolute_peak": 1.8, "kurtosis": 3.1}},
    }


class FakeModelClient:
    """模拟模型客户端：正常 / 超时 / 连接错误 / 输出非法。"""

    def __init__(self, fail_mode="none", latency_ms=1.0, model_version="fake/v1"):
        self.fail_mode = fail_mode
        self.latency_ms = latency_ms
        self.model_version = model_version
        self.calls = 0

    def infer(self, payload, timeout_ms=None):
        import time
        self.calls += 1
        time.sleep(self.latency_ms / 1000.0)
        if self.fail_mode == "timeout":
            return ModelInferResult(success=False, timed_out=True, latency_ms=self.latency_ms,
                                    error="http_timeout")
        if self.fail_mode == "conn_error":
            return ModelInferResult(success=False, timed_out=False, latency_ms=self.latency_ms,
                                    error="connection_error: refused")
        if self.fail_mode == "invalid":
            return ModelInferResult(success=False, timed_out=False, latency_ms=self.latency_ms,
                                    error="output_invalid")
        return ModelInferResult(success=True, latency_ms=self.latency_ms,
                                edge=EdgeResult("warning", 0.7, "medium", self.model_version))


class _Harness:
    """收集 RunRecord 与 PacketResult 的测试底座。"""

    def __init__(self, cfg=None, client=None, window_len=None, capacity=None):
        self.cfg = cfg or EdgeModelConfig()
        if window_len is not None:  # 仅在显式传参时覆盖，保留调用方 cfg 里的设置
            self.cfg.window.length_seconds = window_len
        if capacity is not None:
            self.cfg.queue.max_waiting_requests = capacity
        elif cfg is None:
            # 默认容量 2：瞬时喂入（无真实时间间隔）时避免 worker 尚未取走
            # 就连续 submit 导致的 flush 竞态误判为 QUEUE_FULL
            self.cfg.queue.max_waiting_requests = 2
        self.records = []
        self.packets = []
        self.pipeline = EdgeModelPipeline(
            self.cfg, client or FakeModelClient(), TestRuleRunner(self.cfg.fallback.rule_version),
            on_run_record=self.records.append, on_packet_result=self.packets.append,
        )
        self.pipeline.start()

    def feed_clean_window(self, n=20, sender="s1"):
        """喂一个完整窗口（n 条 50ms 间隔）再关闭。"""
        for i in range(n):
            self.pipeline.ingest(sender, _perception(f"p{i:03d}", i + 1, sender), 0.0 + i * 0.05)
        self.pipeline.ingest(sender, _perception(f"p{900}", 901, sender), 1.05)  # 关闭窗口0
        self.pipeline.wait_idle(timeout_s=10)
        self.pipeline.flush()
        self.pipeline.wait_idle(timeout_s=10)
        self.pipeline.stop()
        return self


def _run(cfg, client):
    h = _Harness(cfg, client)
    return h


# ---------- 回归：4 个已知 bug ----------

def test_regression_boundary_exact_integer_ns():
    # bug#1：浮点切窗漂移。样本恰好落在边界，整数纳秒切窗归属必须确定。
    h = _Harness(window_len=1.0)
    h.pipeline.ingest("s1", _perception("p0", 1), 0.0)
    h.pipeline.ingest("s1", _perception("p1", 2), 1.0)  # 恰好边界 → 属窗口1
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.stop()
    recs = [r for r in h.records if not r.is_empty]
    assert len(recs) == 2  # 窗口0(1条) + 窗口1(1条)，没有因浮点合并
    assert recs[0].sample_count == 1 and recs[1].sample_count == 1
    assert recs[1].window_start_ns == recs[0].window_end_ns  # 边界连续


def test_regression_flush_then_reingest_no_crash():
    # bug#2：flush 后 _active=None 但 epoch 未清，同发送方再 ingest 会崩。
    h = _Harness(window_len=1.0)
    h.pipeline.ingest("s1", _perception("p0", 1), 0.0)
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    # 再次 ingest：不应崩溃，且能正常产出
    h.pipeline.ingest("s1", _perception("p1", 2), 0.0)
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.stop()
    non_empty = [r for r in h.records if not r.is_empty]
    assert len(non_empty) == 2  # flush 前后各一个窗口


def test_regression_worker_processes_all_windows():
    # bug#3：worker 取队列参数错误会线程退出 → 后续窗口全部 QUEUE_FULL/无记录。
    h = _Harness(window_len=0.2, client=FakeModelClient(latency_ms=1.0))
    h.cfg.window.expected_samples_per_window = 4
    for i in range(8):
        h.pipeline.ingest("s1", _perception(f"p{i}", i), 0.0 + i * 0.05)
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.stop()
    non_empty = [r for r in h.records if not r.is_empty]
    assert len(non_empty) == 2  # [0,0.2) / [0.2,0.4) 各 4 条
    assert all(r.execution_mode == EXECUTION_LOCAL_MODEL for r in non_empty)


def test_regression_replace_policy_not_silent_drop():
    # bug#4：replace 策略下被替换的旧窗口曾被静默丢弃，必须产出 QUEUE_FULL 降级记录。
    cfg = EdgeModelConfig()
    cfg.queue.max_waiting_requests = 1
    cfg.queue.full_policy = "replace"
    cfg.window.length_seconds = 0.2
    cfg.window.expected_samples_per_window = 4
    h = _Harness(cfg=cfg, client=FakeModelClient(latency_ms=300.0))
    # 连续喂 6 个窗口的样本：慢模型下队列必然满 → 旧窗口被替换
    for i in range(24):
        h.pipeline.ingest("s1", _perception(f"p{i}", i), 0.0 + i * 0.05)
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.stop()
    recs = [r for r in h.records if not r.is_empty]
    replaced = [r for r in recs if r.fallback_reason == REASON_QUEUE_FULL]
    assert len(replaced) >= 1, "被替换的旧窗口必须产出 QUEUE_FULL 记录"
    local = [r for r in recs if r.execution_mode == EXECUTION_LOCAL_MODEL]
    assert len(local) >= 1
    # 被替换的一定是更老的窗口（最新被保留）
    max_local = max(r.window_id for r in local)
    assert any(r.window_id < max_local for r in replaced)


# ---------- 窗口 → 每包映射 ----------

def test_window_diagnosis_maps_to_all_packets():
    h = _Harness()
    h.feed_clean_window(n=20)
    non_empty = [r for r in h.records if not r.is_empty and not r.is_empty]
    w0 = next(r for r in non_empty if r.window_id == 0)
    # 窗口0 的 20 个包 → 20 个 PacketResult
    w0_packets = [p for p in h.packets if p.sequence_number <= 20]
    assert len(w0_packets) == 20
    # 每个包保留自己的身份
    assert [p.sequence_number for p in w0_packets] == list(range(1, 21))
    assert len({p.packet_id for p in w0_packets}) == 20
    # 共享同一窗口级 EdgeResult（同一次模型结果）
    edges = {(p.edge.edge_result, p.edge.edge_risk_level, p.edge.confidence) for p in w0_packets}
    assert len(edges) == 1
    # 窗口元数据不进 PacketResult（外部接口 4 字段 + 包身份）
    assert all(p.edge.model_version == "fake/v1" for p in w0_packets)
    # 内部记录带窗口元数据
    assert w0.packet_count == 20


def test_empty_window_skips_model():
    h = _Harness()
    # 只有 1 条样本，然后跨一个整窗
    h.pipeline.ingest("s1", _perception("p0", 1), 0.0)
    h.pipeline.ingest("s1", _perception("p1", 2), 2.5)  # 空出窗口0
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=10)
    h.pipeline.stop()
    empty = [r for r in h.records if r.is_empty]
    assert len(empty) == 1
    assert empty[0].execution_mode == EXECUTION_NONE
    # 空窗口不产生任何 PacketResult（不调模型、不映射）
    # 非空窗口正常：窗口0有p0，窗口2有p1
    non_empty = [r for r in h.records if not r.is_empty]
    assert sum(1 for r in non_empty if r.execution_mode == EXECUTION_LOCAL_MODEL) == 2


# ---------- 降级路径 ----------

def test_model_timeout_falls_back():
    cfg = EdgeModelConfig()
    cfg.timeout.inference_ms = 50
    h = _Harness(cfg=cfg, client=FakeModelClient(fail_mode="timeout"))
    h.feed_clean_window(n=4)
    recs = [r for r in h.records if not r.is_empty]
    assert all(r.execution_mode == EXECUTION_CODE_FALLBACK for r in recs)
    assert all(r.fallback_reason == REASON_INFERENCE_TIMEOUT for r in recs)
    assert all(r.output_valid for r in recs)
    assert all(r.model_version == "edge_rule_test_v1" for r in recs)


def test_model_unavailable_falls_back():
    cfg = EdgeModelConfig()
    cfg.breaker.enabled = False  # 只看连接错误路径
    h = _Harness(cfg=cfg, client=FakeModelClient(fail_mode="conn_error"))
    h.feed_clean_window(n=4)
    recs = [r for r in h.records if not r.is_empty]
    assert all(r.execution_mode == EXECUTION_CODE_FALLBACK for r in recs)
    assert all(r.fallback_reason == REASON_MODEL_UNAVAILABLE for r in recs)
    assert all(r.output_valid for r in recs)


def test_model_invalid_output_falls_back():
    cfg = EdgeModelConfig()
    cfg.breaker.enabled = False
    h = _Harness(cfg=cfg, client=FakeModelClient(fail_mode="invalid"))
    h.feed_clean_window(n=4)
    recs = [r for r in h.records if not r.is_empty]
    assert all(r.execution_mode == EXECUTION_CODE_FALLBACK for r in recs)
    assert all(r.output_valid for r in recs)


def _ingest_spaced(h, perception, gap_s=0.25):
    """按真实时间喂入并等待，让 worker 在处理完上一窗口后再提交下一个。"""
    h.pipeline.ingest("s1", perception, time.monotonic())
    time.sleep(gap_s)


def test_circuit_breaker_open_and_recover():
    cfg = EdgeModelConfig()
    cfg.breaker.enabled = True
    cfg.breaker.consecutive_failure_threshold = 2
    cfg.breaker.recovery_probe_interval_s = 2.0  # 长探测窗，确保 w2 落在熔断期内 → BREAKER_OPEN
    cfg.window.length_seconds = 0.2
    client = FakeModelClient(fail_mode="conn_error", latency_ms=1.0)
    h = _Harness(cfg=cfg, client=client, capacity=2)
    # 阶段1：真实时间间隔 0.25s > 窗口 0.2s，样本各自落入独立窗口
    _ingest_spaced(h, _perception("p0", 0))   # 开窗0
    _ingest_spaced(h, _perception("p1", 1))   # 关窗0 → w0 fail (1)
    _ingest_spaced(h, _perception("p2", 2))   # 关窗1 → w1 fail (2 → 熔断打开)
    h.pipeline.flush()                        # 关窗2 → w2 在熔断期内 → BREAKER_OPEN
    h.pipeline.wait_idle(timeout_s=5)
    # 阶段2：过探测期后恢复为正常模型，投递探测 → 应回到模型路线
    time.sleep(2.3)
    client.fail_mode = "none"
    h.pipeline.ingest("s1", _perception("recover", 99), time.monotonic())
    h.pipeline.flush()
    h.pipeline.wait_idle(timeout_s=5)
    h.pipeline.stop()
    recs = [r for r in h.records if not r.is_empty]
    opened = [r for r in recs if r.fallback_reason == REASON_BREAKER_OPEN]
    assert len(opened) >= 1, "熔断打开后应有 BREAKER_OPEN 降级"
    assert any(r.execution_mode == EXECUTION_LOCAL_MODEL for r in recs[-3:])  # 恢复


# ---------- 配置校验 ----------

def test_config_validate_rejects_bad_values():
    cfg = EdgeModelConfig()
    cfg.timeout.queue_wait_ms = 3000  # 超过 total_ms=2000
    errors = cfg.validate()
    assert any("queue_wait_ms" in e for e in errors)
    cfg2 = EdgeModelConfig()
    assert cfg2.validate() == []


def test_rule_runner_output_contract():
    runner = TestRuleRunner("edge_rule_test_v1")
    from edge_model.contracts import WindowAggregate
    w = WindowAggregate(sender_id="s", window_id=0, window_start_ns=0, window_end_ns=int(1e9),
                        close_ts_ns=int(1e9), expected_samples=20, sample_count=1,
                        payload=_perception("p", 1))
    edge = runner.run(w)
    assert edge.edge_result in ("normal", "warning", "fault")
    assert edge.edge_risk_level in ("low", "medium", "high")
    assert 0.0 <= edge.confidence <= 1.0
    assert edge.model_version == "edge_rule_test_v1"
