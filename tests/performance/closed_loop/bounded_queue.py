# -*- coding: utf-8 -*-
"""有界模型任务队列 + 单推理 worker + 分层超时 + 熔断。

队列语义（对应设计：「1 条正在推理 + 最多 capacity 条等待」）：
- capacity 只统计「等待中」的任务，正在推理的不占 capacity；
- 队列满策略：
  - drop_current_to_fallback：新窗口直接返回降级指令（当前窗口走代码规则）；
  - replace_oldest_pending：替换「尚未开始」的最老等待任务，正在推理的绝不打断；
- 分层超时（从窗口 submit 时刻起算）：
  - queue_wait_ms 超限 → QUEUE_TIMEOUT 降级（开始推理前检查）；
  - inference_ms    超限 → TIMEOUT 降级（子线程 join 判定）；
  - total_ms        超限 → 开始前触发 TOTAL_TIMEOUT；推理已完成但迟到则标记
    exceeded_total_timeout（默认仍返回模型结果，见 on_total_timeout_after_completion）；
- 熔断：连续失败 >= 阈值 → 打开熔断，一律走 BREAKER_OPEN 降级；
  恢复探测周期过后放行一次真实推理作为探测，成功即恢复。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .config import ClosedLoopConfig
from .model import (
    REASON_BREAKER_OPEN,
    REASON_INFERENCE_TIMEOUT,
    REASON_MODEL_INFERENCE_FAILED,
    REASON_MODEL_OUTPUT_INVALID,
    REASON_QUEUE_FULL,
    REASON_QUEUE_TIMEOUT,
    REASON_TOTAL_TIMEOUT,
    WindowAggregate,
)
from .model_adapter import InferenceAdapter, InferenceOutcome

FULL_POLICY_DROP = "drop_current_to_fallback"
FULL_POLICY_REPLACE = "replace_oldest_pending"


@dataclass
class SubmitResult:
    accepted: bool
    # 未接受（或被替换）的窗口及其降级原因，由调用方负责走代码规则
    fallback_windows: List[Tuple[WindowAggregate, str]] = field(default_factory=list)


class ModelTaskQueue:
    def __init__(self, capacity: int, full_policy: str, clock=time.monotonic):
        self.capacity = capacity
        self.full_policy = full_policy
        self._clock = clock
        self._cond = threading.Condition()
        self._pending: List[WindowAggregate] = []
        self._inflight = 0
        self.stopped = False
        self.max_observed_queued = 0

    def submit(self, window: WindowAggregate) -> SubmitResult:
        with self._cond:
            if self.stopped:
                return SubmitResult(accepted=False,
                                    fallback_windows=[(window, "STOPPED")])
            if len(self._pending) < self.capacity:
                window.submit_ts = self._clock()
                self._pending.append(window)
                self.max_observed_queued = max(self.max_observed_queued, len(self._pending))
                self._cond.notify()
                return SubmitResult(accepted=True)
            # 队列已满
            if self.full_policy == FULL_POLICY_REPLACE and self._pending:
                replaced = self._pending.pop(0)  # 最老的「等待中」任务
                window.submit_ts = self._clock()
                self._pending.append(window)
                self._cond.notify()
                return SubmitResult(accepted=True,
                                    fallback_windows=[(replaced, REASON_QUEUE_FULL)])
            # drop_current_to_fallback；或 replace 但无可替换的等待任务 → 同样丢弃当前
            return SubmitResult(accepted=False,
                                fallback_windows=[(window, REASON_QUEUE_FULL)])

    def get(self, timeout_s: float = 0.2) -> Optional[WindowAggregate]:
        with self._cond:
            while not self._pending and not self.stopped:
                self._cond.wait(timeout_s)
            if not self._pending:
                return None
            w = self._pending.pop(0)
            self._inflight += 1
            return w

    def done(self):
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    def pending_count(self) -> int:
        with self._cond:
            return len(self._pending)

    def inflight_count(self) -> int:
        with self._cond:
            return self._inflight

    def idle(self) -> bool:
        with self._cond:
            return not self._pending and self._inflight == 0

    def wait_until_idle(self, timeout_s: float = 5.0) -> bool:
        deadline = self._clock() + timeout_s
        with self._cond:
            while (self._pending or self._inflight > 0) and self._clock() < deadline:
                self._cond.wait(0.05)
            return not self._pending and self._inflight == 0

    def stop(self):
        with self._cond:
            self.stopped = True
            self._cond.notify_all()


class InferenceWorker:
    """单 worker 线程：从队列取窗口，执行推理并产生 RunRecord。"""

    def __init__(self, queue: ModelTaskQueue, adapter: InferenceAdapter, cfg: ClosedLoopConfig,
                 emit_local: Callable[[WindowAggregate, object, float, float, float, bool, str], None],
                 emit_fallback: Callable[[WindowAggregate, str, float, Optional[float], Optional[str]], None],
                 clock=time.monotonic, poll_s: float = 0.05):
        self.queue = queue
        self.adapter = adapter
        self.cfg = cfg
        self.emit_local = emit_local
        self.emit_fallback = emit_fallback
        self._clock = clock
        self._poll_s = poll_s
        self._consecutive_failures = 0
        self._breaker_open_until: Optional[float] = None
        self.breaker_state = "closed"
        self._thread = threading.Thread(target=self._loop, daemon=True, name="closed-loop-worker")
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self, join_s: float = 5.0):
        self._running = False
        self.queue.stop()
        self._thread.join(timeout=join_s)

    # ---- 熔断状态 ----
    def _note_failure(self):
        if not self.cfg.breaker.enabled:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.cfg.breaker.consecutive_failure_threshold:
            self._breaker_open_until = self._clock() + self.cfg.breaker.recovery_probe_interval_s
            self.breaker_state = "open"

    def _note_success(self):
        self._consecutive_failures = 0
        if self.breaker_state == "open":
            self.breaker_state = "closed"
        self._breaker_open_until = None

    def _breaker_allows_model(self) -> bool:
        if not self.cfg.breaker.enabled:
            return True
        if self._breaker_open_until is None:
            return True
        if self._clock() < self._breaker_open_until:
            return False
        # 探测期到：放行一次真实推理作为恢复探测
        self._breaker_open_until = None
        return True

    # ---- 主循环 ----
    def _loop(self):
        while self._running or not self.queue.idle():
            window = self.queue.get(timeout_s=self._poll_s)
            if window is None:
                continue
            try:
                self._process(window)
            finally:
                self.queue.done()

    def _process(self, window: WindowAggregate):
        cfg = self.cfg
        now = self._clock()
        submit_ts = window.submit_ts if window.submit_ts is not None else now
        queue_wait_ms = (now - submit_ts) * 1000.0
        breaker_state = self.breaker_state

        if not self._breaker_allows_model():
            self.emit_fallback(window, REASON_BREAKER_OPEN, queue_wait_ms, None, breaker_state)
            return

        # 排队超时 / 总处理超时（开始前检查）
        if queue_wait_ms > cfg.timeout.queue_wait_ms:
            self.emit_fallback(window, REASON_QUEUE_TIMEOUT, queue_wait_ms, None, breaker_state)
            return
        if (now - submit_ts) * 1000.0 > cfg.timeout.total_ms:
            self.emit_fallback(window, REASON_TOTAL_TIMEOUT, queue_wait_ms, None, breaker_state)
            return

        outcome = self._run_inference(window)
        total_ms = (self._clock() - submit_ts) * 1000.0

        if outcome.timed_out:
            self._note_failure()
            self.emit_fallback(window, REASON_INFERENCE_TIMEOUT, queue_wait_ms,
                               outcome.latency_ms, breaker_state)
            return
        if not outcome.success:
            self._note_failure()
            self.emit_fallback(window, REASON_MODEL_INFERENCE_FAILED, queue_wait_ms,
                               outcome.latency_ms, breaker_state)
            return

        # 校验模型输出
        from .output_validator_import import validate_model_output
        validation = validate_model_output(outcome.text or "")
        if not validation["valid"]:
            self._note_failure()
            self.emit_fallback(window, REASON_MODEL_OUTPUT_INVALID, queue_wait_ms,
                               outcome.latency_ms, breaker_state)
            return

        self._note_success()
        exceeded = total_ms > cfg.timeout.total_ms
        self.emit_local(window, validation["parsed"], queue_wait_ms,
                        outcome.latency_ms or 0.0, total_ms, exceeded,
                        self.adapter.model_version)

    def _run_inference(self, window: WindowAggregate):
        """在子线程执行 adapter.infer，join 超时判定。"""
        timeout_s = self.cfg.timeout.inference_ms / 1000.0
        holder: dict = {}
        t0 = self._clock()

        def _run():
            try:
                holder["outcome"] = self.adapter.infer(window.payload)
                holder["ok"] = True
            except Exception as exc:  # noqa: BLE001
                holder["ok"] = False
                holder["err"] = exc

        t = threading.Thread(target=_run, daemon=True, name="infer-call")
        t.start()
        t.join(timeout_s + 0.05)
        if t.is_alive():
            return InferenceOutcome(success=False, timed_out=True,
                                    latency_ms=(self._clock() - t0) * 1000.0)
        if not holder.get("ok"):
            return InferenceOutcome(success=False, timed_out=False,
                                    latency_ms=(self._clock() - t0) * 1000.0,
                                    error=repr(holder.get("err")))
        return holder["outcome"]
