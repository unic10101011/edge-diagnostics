# -*- coding: utf-8 -*-
"""闭环管线：把窗口聚合器、有界队列、推理 worker、代码降级串成一条可验证链路。

调用方只做两件事：
    pipeline.ingest(sender_id, perception, arrival_ts)   # 20Hz 感知到达
    pipeline.flush() / pipeline.stop()                    # 收尾

所有事件经 sink（callable(RunRecord)）输出；Runner 负责把事件落盘。
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

from .bounded_queue import InferenceWorker, ModelTaskQueue
from .code_fallback import CodeFallbackRunner
from .config import ClosedLoopConfig
from .model import (
    EXECUTION_CODE_FALLBACK,
    EXECUTION_LOCAL_MODEL,
    EXECUTION_NONE,
    REASON_CODE_FALLBACK_FAILED,
    RunRecord,
    WindowAggregate,
)
from .model_adapter import InferenceAdapter
from .window_aggregator import WindowAggregator


class ClosedLoopPipeline:
    def __init__(self, cfg: ClosedLoopConfig, adapter: InferenceAdapter,
                 sink: Callable[[RunRecord], None], clock=time.monotonic):
        self.cfg = cfg
        self.fallback = CodeFallbackRunner(cfg.fallback.rule_version)
        self.queue = ModelTaskQueue(cfg.queue.capacity, cfg.queue.full_policy, clock=clock)
        self.worker = InferenceWorker(
            self.queue, adapter, cfg,
            emit_local=self._emit_local, emit_fallback=self._emit_fallback, clock=clock,
        )
        self._aggregators: Dict[str, WindowAggregator] = {}
        self._agg_lock = threading.Lock()
        self.sink = sink
        self._clock = clock
        self.started = False

    # ---- 生命周期 ----
    def start(self):
        self.worker.start()
        self.started = True

    def stop(self, join_s: float = 5.0):
        self.worker.stop(join_s=join_s)

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        return self.queue.wait_until_idle(timeout_s)

    @property
    def max_observed_queued(self) -> int:
        return self.queue.max_observed_queued

    # ---- 感知入口 ----
    def ingest(self, sender_id: str, perception: dict, arrival_ts: Optional[float] = None):
        agg = self._get_aggregator(sender_id)
        for window in agg.ingest(perception, arrival_ts):
            self._handle_closed(window)

    def flush(self):
        """关闭所有发送方的 active 窗口（收尾）。"""
        with self._agg_lock:
            aggs = list(self._aggregators.values())
        for agg in aggs:
            w = agg.flush()
            if w is not None:
                self._handle_closed(w)

    def _get_aggregator(self, sender_id: str) -> WindowAggregator:
        with self._agg_lock:
            agg = self._aggregators.get(sender_id)
            if agg is None:
                agg = WindowAggregator(sender_id, self.cfg.window, clock=self._clock)
                self._aggregators[sender_id] = agg
            return agg

    # ---- 窗口处理 ----
    def _handle_closed(self, window: WindowAggregate):
        if window.is_empty:
            # 空窗口：不调模型，只记录
            self.sink(RunRecord(
                sender_id=window.sender_id, window_id=window.window_id,
                window_start_ns=window.window_start_ns, window_end_ns=window.window_end_ns,
                sample_count=0, missing_ratio=1.0, sparse=False, is_empty=True,
                execution_mode=EXECUTION_NONE, fallback_reason=None, output_valid=False,
                late_dropped_count=window.late_dropped_count,
            ))
            return
        res = self.queue.submit(window)
        # fallback_windows 涵盖两种情形：drop 策略下的当前窗口（accepted=False），
        # 以及 replace 策略下被替换的旧窗口（accepted=True）。都必须在提交线程走代码规则。
        for w, reason in res.fallback_windows:
            self._run_fallback(w, reason)

    def _run_fallback(self, window: WindowAggregate, reason: Optional[str],
                      queue_wait_ms: Optional[float] = None,
                      inference_ms: Optional[float] = None,
                      breaker_state: Optional[str] = None,
                      note: Optional[str] = None):
        try:
            edge = self.fallback.run(window.payload)
            self.sink(RunRecord(
                sender_id=window.sender_id, window_id=window.window_id,
                window_start_ns=window.window_start_ns, window_end_ns=window.window_end_ns,
                sample_count=window.sample_count, missing_ratio=window.missing_ratio,
                sparse=window.sparse, is_empty=False,
                execution_mode=EXECUTION_CODE_FALLBACK, fallback_reason=reason,
                output_valid=True,
                edge_result=edge.edge_result, edge_risk_level=edge.edge_risk_level,
                confidence=edge.confidence, model_version=edge.model_version,
                queue_wait_ms=queue_wait_ms, inference_latency_ms=inference_ms,
                late_dropped_count=window.late_dropped_count, breaker_state=breaker_state,
                note=note,
            ))
        except Exception as exc:  # noqa: BLE001
            # 两条路线都失败
            self.sink(RunRecord(
                sender_id=window.sender_id, window_id=window.window_id,
                window_start_ns=window.window_start_ns, window_end_ns=window.window_end_ns,
                sample_count=window.sample_count, missing_ratio=window.missing_ratio,
                sparse=window.sparse, is_empty=False,
                execution_mode=EXECUTION_CODE_FALLBACK, fallback_reason=REASON_CODE_FALLBACK_FAILED,
                output_valid=False, late_dropped_count=window.late_dropped_count,
                breaker_state=breaker_state,
                note="model_route_reason=%s; fallback_error=%r" % (reason, exc),
            ))

    # ---- worker 回调（worker 线程调用） ----
    def _emit_local(self, window: WindowAggregate, parsed: dict,
                    queue_wait_ms: float, inference_ms: float, total_ms: float,
                    exceeded_total_timeout: bool, model_version: str):
        self.sink(RunRecord(
            sender_id=window.sender_id, window_id=window.window_id,
            window_start_ns=window.window_start_ns, window_end_ns=window.window_end_ns,
            sample_count=window.sample_count, missing_ratio=window.missing_ratio,
            sparse=window.sparse, is_empty=False,
            execution_mode=EXECUTION_LOCAL_MODEL, fallback_reason=None,
            output_valid=True,
            edge_result=(parsed or {}).get("edge_result"),
            edge_risk_level=(parsed or {}).get("edge_risk_level"),
            confidence=(parsed or {}).get("confidence"),
            model_version=model_version,
            queue_wait_ms=round(queue_wait_ms, 2),
            inference_latency_ms=round(inference_ms, 2),
            total_latency_ms=round(total_ms, 2),
            exceeded_total_timeout=exceeded_total_timeout,
            late_dropped_count=window.late_dropped_count,
            breaker_state=self.worker.breaker_state,
        ))

    def _emit_fallback(self, window: WindowAggregate, reason: Optional[str],
                       queue_wait_ms: float, inference_ms: Optional[float],
                       breaker_state: Optional[str]):
        self._run_fallback(window, reason, queue_wait_ms=queue_wait_ms,
                           inference_ms=inference_ms, breaker_state=breaker_state)
