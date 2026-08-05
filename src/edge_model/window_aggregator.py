# -*- coding: utf-8 -*-
"""窗口聚合器：每发送方双缓冲，按到达时刻切 1s 窗口。

逻辑已通过 tests/performance/closed_loop 验证（整数纳秒切窗，避免大单调基数
浮点漂移）。生产版在此之上记录窗口内每包身份（included_packets），供把窗口级
诊断结果映射回每个数据包使用。

- 窗口归属只按到达时刻，不使用 PerceptionResult 内部时间戳；
- 已关窗口后的迟到/乱序到达：丢弃并计数（late_rule = drop_and_count）；
- 空窗口产出 is_empty=True，调用方决定不调模型；
- 稀疏窗口标记 quality_status=warning + WINDOW_SPARSE，不伪装正常；
- 特征聚合用均值占位；聚合算法正确性不在本模块范围（由感知/业务规则负责）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import WindowConfig
from .contracts import WindowAggregate

WINDOW_SPARSE_FLAG = "WINDOW_SPARSE"


def to_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def _walk_path(obj: Dict, path: List[str]) -> object:
    for p in path:
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    return obj


def merge_features(samples: List[dict]) -> Dict:
    """按第一份样本的结构逐叶取均值（占位聚合）。缺字段的样本不参与该叶平均。"""
    if not samples:
        return {}
    template = samples[0].get("features", {})

    def rec(node: object, path: List[str]) -> object:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                out[k] = rec(v, path + [k])
            return out
        if isinstance(node, (int, float)):
            vals = []
            for s in samples:
                v = _walk_path(s.get("features", {}), path)
                if isinstance(v, (int, float)) and v == v:  # 排除 NaN/Inf
                    vals.append(float(v))
            if not vals:
                return None
            return round(sum(vals) / len(vals), 6)
        return node

    return rec(template, [])


def merge_quality(samples: List[dict]) -> Tuple[str, List[str]]:
    """窗口内质量：status 取最差（任一 warning 即 warning），flags 取并集。"""
    status = "good"
    flags: List[str] = []
    for s in samples:
        q = s.get("perception_quality") or {}
        if q.get("status") == "warning":
            status = "warning"
        for f in q.get("flags", []):
            if f not in flags:
                flags.append(f)
    return status, flags


def packet_identity(perception: dict) -> Dict[str, Any]:
    """从 PerceptionResult 提取包身份（用于结果映射回数据包）。"""
    return {
        "task_id": perception.get("task_id"),
        "packet_id": perception.get("packet_id"),
        "sender_id": perception.get("sender_id"),
        "sequence_number": perception.get("sequence_number"),
    }


@dataclass
class _WindowBuffer:
    wid: int
    start_ns: int
    end_ns: int
    samples: List[Tuple[float, dict]] = field(default_factory=list)  # (arrival_ts_sec, perception)


class WindowAggregator:
    """单发送方的窗口聚合器。"""

    def __init__(self, sender_id: str, cfg: WindowConfig, clock=time.monotonic):
        self.sender_id = sender_id
        self.cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._epoch_ns: Optional[int] = None
        self._active: Optional[_WindowBuffer] = None
        self.total_late_dropped = 0
        self.total_windows_closed = 0

    def _length_ns(self) -> int:
        return to_ns(self.cfg.length_seconds)

    def ingest(self, perception: dict, arrival_ts: Optional[float] = None) -> List[WindowAggregate]:
        """接收一条感知结果，返回因此关闭的窗口聚合列表（通常 0 或 1，间隔后可能多个）。"""
        if arrival_ts is None:
            arrival_ts = self._clock()
        arrival_ns = to_ns(arrival_ts)
        closed_buffers: List[_WindowBuffer] = []
        with self._lock:
            if self._epoch_ns is None:
                self._epoch_ns = arrival_ns
                length_ns = self._length_ns()
                self._active = _WindowBuffer(0, arrival_ns, arrival_ns + length_ns)
            if arrival_ns < self._active.start_ns:
                # 属于已关闭窗口的迟到/乱序数据：丢弃但计数
                self.total_late_dropped += 1
                return []
            while arrival_ns >= self._active.end_ns:
                closed_buffers.append(self._active)
                self._active = _WindowBuffer(
                    self._active.wid + 1, self._active.end_ns,
                    self._active.end_ns + self._length_ns(),
                )
            self._active.samples.append((arrival_ts, perception))
        out = [self._aggregate(b) for b in closed_buffers]
        self.total_windows_closed += len(out)
        return out

    def flush(self) -> Optional[WindowAggregate]:
        """关闭当前 active 窗口（收尾）。flush 后视为一段流结束，再次 ingest 重新起算。"""
        with self._lock:
            if self._epoch_ns is None or self._active is None:
                return None
            buf = self._active
            self._active = None
            self._epoch_ns = None
        agg = self._aggregate(buf)
        self.total_windows_closed += 1
        return agg

    def _aggregate(self, buf: _WindowBuffer) -> WindowAggregate:
        cfg = self.cfg
        samples = buf.samples
        n = len(samples)
        is_empty = n == 0
        expected = cfg.expected_samples_per_window
        missing_ratio = round(max(0.0, 1.0 - n / expected), 6) if expected else 0.0
        sparse = (not is_empty) and n < cfg.min_samples_for_full

        q_status, q_flags = ("good", []) if is_empty else merge_quality([s for _, s in samples])
        if sparse:
            q_status = "warning"
            if WINDOW_SPARSE_FLAG not in q_flags:
                q_flags.append(WINDOW_SPARSE_FLAG)

        first_s = samples[0][0] if samples else None
        last_s = samples[-1][0] if samples else None

        payload: Dict = {}
        included: List[Dict[str, Any]] = []
        if not is_empty:
            base = samples[0][1]
            payload = dict(base)
            payload["features"] = merge_features([s for _, s in samples])
            payload["perception_quality"] = {"status": q_status, "flags": list(q_flags)}
            payload["sequence_number"] = base.get("sequence_number")
            payload["end_generate_timestamp_ns"] = to_ns(last_s or 0)
            payload["feature_generated_at_ns"] = to_ns(self._clock())
            for _, s in samples:
                included.append(packet_identity(s))

        return WindowAggregate(
            sender_id=self.sender_id,
            window_id=buf.wid,
            window_start_ns=buf.start_ns,
            window_end_ns=buf.end_ns,
            close_ts_ns=to_ns(self._clock()),
            expected_samples=expected,
            sample_count=n,
            first_sample_ts_ns=to_ns(first_s) if first_s is not None else None,
            last_sample_ts_ns=to_ns(last_s) if last_s is not None else None,
            missing_ratio=missing_ratio,
            quality_status=q_status,
            quality_flags=q_flags,
            late_dropped_count=0,
            is_empty=is_empty,
            sparse=sparse,
            included_packets=included,
            payload=payload,
        )
