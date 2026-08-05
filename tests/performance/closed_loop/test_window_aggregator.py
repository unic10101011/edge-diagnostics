# -*- coding: utf-8 -*-
"""T1：窗口聚合抖动验证。

聚合器级测试用脚本化到达时刻，完全确定、无 sleep；
管线级测试用真实时间跑 20Hz 计划，验证「推理不阻塞接收 / 每窗口至多一个模型任务 /
空窗口不调模型」等行为。全部无 torch，Windows 可直接 pytest。
"""
from __future__ import annotations

import time

import pytest

from .config import ClosedLoopConfig, WindowConfig
from .jitter_source import Emission, make_schedule, run_schedule
from .model import EXECUTION_LOCAL_MODEL, EXECUTION_NONE
from .model_adapter import MockModel
from .pipeline import ClosedLoopPipeline
from .test_support import RecordCollector, sample_perceptions
from .window_aggregator import WINDOW_SPARSE_FLAG, WindowAggregator

# 脚本化到达用小的相对秒值：聚合器内部转整数纳秒做窗口归属，小值转换精确、
# 边界确定（大单调基数 + 小偏移在浮点下会有 ~1e-9 量级漂移，会让边界样本归属漂移）
BASE = 0.0
FAKE_CLOCK = lambda: 1_000_000.0  # noqa: E731  # 仅用于 close_ts_ns 元数据


def _arrivals(n, start=0.0, step=0.05):
    return [BASE + start + i * step for i in range(n)]


def _scripted_aggregator(expected=20, min_full=15, length=1.0):
    cfg = WindowConfig(length_seconds=length, expected_samples=expected,
                       min_samples_for_full=min_full)
    return WindowAggregator("s1", cfg, clock=FAKE_CLOCK)


# ---------- 聚合器级：窗口元数据与边界 ----------

def test_clean_stream_window_metadata_and_boundaries():
    agg = _scripted_aggregator()
    # 20 条正常 50ms 到达
    for a in _arrivals(20):
        assert agg.ingest(_p(), a) == []
    # 第 21 条越过 1s 边界 → 关闭窗口 0
    closed = agg.ingest(_p(), BASE + 1.0)
    assert len(closed) == 1
    w0 = closed[0]
    assert w0.window_id == 0
    assert w0.sample_count == 20
    assert w0.missing_ratio == 0.0
    assert w0.is_empty is False and w0.sparse is False
    assert w0.quality_status == "good" and w0.quality_flags == []
    assert w0.window_start_ns == int(BASE * 1e9)
    assert w0.window_end_ns == int((BASE + 1.0) * 1e9)
    assert w0.first_sample_ts_ns == int(BASE * 1e9)
    assert abs(w0.last_sample_ts_ns - int((BASE + 0.95) * 1e9)) < 1_000_000  # 浮点容差
    # flush 关闭最后的局部窗口
    last = agg.flush()
    assert last is not None and last.window_id == 1
    assert last.sample_count == 1
    assert last.sparse is True
    assert last.missing_ratio == pytest.approx(0.95, abs=0.001)


def test_sparse_window_marked_quality_not_faked():
    agg = _scripted_aggregator(min_full=15)
    # 只有 10 条（缺失一半）→ 稀疏窗口
    for a in _arrivals(10):
        agg.ingest(_p(), a)
    closed = agg.ingest(_p(), BASE + 1.0)
    w0 = closed[0]
    assert w0.sample_count == 10
    assert w0.sparse is True
    assert w0.missing_ratio == pytest.approx(0.5, abs=0.001)
    assert w0.quality_status == "warning"
    assert WINDOW_SPARSE_FLAG in w0.quality_flags
    # 稀疏窗口 payload 仍带完整 features（可继续走模型/规则），只是被标记
    assert "features" in w0.payload


def test_empty_gap_produces_empty_windows():
    agg = _scripted_aggregator()
    for a in _arrivals(20):
        agg.ingest(_p(), a)
    # 中间空 1.5s，下一次到达落在窗口 2
    closed = agg.ingest(_p(), BASE + 2.5)
    wids = [w.window_id for w in closed]
    assert wids == [0, 1]
    assert closed[0].sample_count == 20
    empty = closed[1]
    assert empty.is_empty is True
    assert empty.sample_count == 0
    assert empty.missing_ratio == 1.0
    assert empty.payload == {}


def test_late_and_out_of_order_arrival_dropped_counted():
    agg = _scripted_aggregator()
    for a in _arrivals(20):
        agg.ingest(_p(), a)
    agg.ingest(_p(), BASE + 1.0)  # 关闭窗口 0
    # 到达时刻属于已关闭窗口 0 → 丢弃并计数
    assert agg.ingest(_p(), BASE + 0.5) == []
    assert agg.total_late_dropped == 1
    # 乱序数据内部时间戳不影响窗口归属：给一个时间戳远在未来的样本，
    # 仍按到达时刻进窗口 1（含此前 t=1.0 的边界样本，共 2 条）
    p = _p()
    p["end_generate_timestamp_ns"] = 9_999_999_999_999
    agg.ingest(p, BASE + 1.05)
    last = agg.flush()
    assert last.window_id == 1
    assert last.sample_count == 2
    assert last.first_sample_ts_ns == int((BASE + 1.0) * 1e9)
    assert last.last_sample_ts_ns == int((BASE + 1.05) * 1e9)


# ---------- 管线级：行为验证（真实时间，短窗口） ----------

def _p():
    return sample_perceptions(per_category=3, seed=42)[0]


def _run(cfg, adapter, schedule):
    collector = RecordCollector()
    pipeline = ClosedLoopPipeline(cfg, adapter, collector)
    pipeline.start()
    t_ingest = run_schedule(schedule, pipeline)
    pipeline.flush()
    pipeline.wait_idle(timeout_s=10)
    pipeline.stop()
    return collector.get(), pipeline, t_ingest


def test_pipeline_at_most_one_model_task_per_window():
    cfg = ClosedLoopConfig()
    cfg.queue.capacity = 2  # 避免 flush 时与刚入队的下一窗口撞满（本测试验证的是窗口化，不是背压）
    adapter = MockModel(latency_ms=1.0)
    schedule = make_schedule(_perceptions(), duration_s=2.05, rate_hz=20,
                             jitter_s=0.0, drop_prob=0.0, burst_prob=0.0,
                             gap_prob=0.0, seed=1)
    records, _pipeline, _t = _run(cfg, adapter, schedule)
    non_empty = [r for r in records if not r.is_empty]
    empty = [r for r in records if r.is_empty]
    assert empty == []  # 无空窗口
    # 每个非空窗口恰好一条记录（至多一个模型任务/窗口）
    assert len(non_empty) == 3  # [0,1) / [1,2) / flush 的 [2,...)
    assert all(r.execution_mode == EXECUTION_LOCAL_MODEL for r in non_empty)
    assert all(r.output_valid for r in non_empty)
    assert len({r.window_id for r in non_empty}) == 3
    # 边界不重叠不遗漏
    by_id = {r.window_id: r for r in non_empty}
    assert by_id[1].window_start_ns == by_id[0].window_end_ns
    assert by_id[0].sample_count == 20
    assert by_id[1].sample_count == 20
    # 平均每个窗口 ≤1 个模型任务（此处即每条非空记录一次）
    local_model_calls = sum(1 for r in non_empty if r.execution_mode == EXECUTION_LOCAL_MODEL)
    assert local_model_calls == len(non_empty)


def test_inference_does_not_block_ingest():
    # 慢模型（0.35s）vs 0.2s 窗口：worker 必然积压，但 ingest 不能阻塞
    cfg = ClosedLoopConfig()
    cfg.window.length_seconds = 0.2
    cfg.window.expected_samples = 4
    cfg.window.min_samples_for_full = 3
    cfg.queue.capacity = 6  # 足够容纳全部窗口，避免降级干扰断言
    cfg.timeout.queue_wait_ms = 5000  # 本测试验证 ingest 不阻塞，放开排队超时，避免 QUEUE_TIMEOUT 干扰
    adapter = MockModel(latency_ms=350.0)
    schedule = make_schedule(_perceptions(), duration_s=1.2, rate_hz=20,
                             jitter_s=0.0, drop_prob=0.0, burst_prob=0.0,
                             gap_prob=0.0, seed=2)
    records, pipeline, t_ingest = _run(cfg, adapter, schedule)
    # ingest 墙钟时间 ≈ 计划时长（1.2s），而不是被 6×0.35≈2.1s 推理拖慢
    assert t_ingest < 1.6, "ingest 被推理阻塞了: %.2fs" % t_ingest
    assert t_ingest >= 1.1
    non_empty = [r for r in records if not r.is_empty]
    assert len(non_empty) == 6
    assert all(r.execution_mode == EXECUTION_LOCAL_MODEL for r in non_empty)
    assert pipeline.max_observed_queued <= cfg.queue.capacity


def test_empty_window_skips_model():
    cfg = ClosedLoopConfig()
    cfg.queue.capacity = 2
    adapter = MockModel(latency_ms=1.0)
    # 20 条正常到达后空 1.5s，再补一条
    percs = _perceptions()
    schedule = [Emission(round(i * 0.05, 3), percs[i % len(percs)]) for i in range(20)]
    schedule.append(Emission(2.5, percs[0]))
    records, _pipeline, _t = _run(cfg, adapter, schedule)
    empty = [r for r in records if r.is_empty]
    assert len(empty) == 1
    assert empty[0].execution_mode == EXECUTION_NONE
    assert empty[0].output_valid is False
    non_empty = [r for r in records if not r.is_empty]
    assert len(non_empty) == 2  # 窗口0(20条) + flush 的窗口2(1条)
    # 空窗口没有触发模型调用：LOCAL_MODEL 记录数 == 非空窗口数
    assert sum(1 for r in non_empty if r.execution_mode == EXECUTION_LOCAL_MODEL) == 2


def _perceptions():
    return sample_perceptions(per_category=3, seed=42)
