# -*- coding: utf-8 -*-
"""20 Hz 感知输入调度器：模拟真实网络抖动，供窗口聚合验证（T1）使用。

不生成感知数据本身（由调用方提供 PerceptionResult 列表），只负责排程：
- 正常 50 ms 间隔；
- 随机提前/延迟（jitter）；
- 偶发丢包（drop_prob，直接不产生该条）；
- 短时间突发（burst_prob，一个时刻连发多条）；
- 偶发空窗口（gap_prob，一段时间没有任何样本）；
- 到达时刻与样本内部时间戳天然不一致（乱序语义，窗口归属只看到达时刻）。

迟到/乱序到达（到达时所属窗口已关闭）不在这里合成，由测试直接以历史
arrival_ts 调用 pipeline.ingest 显式验证（见 test_window_aggregator.py）。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DEFAULT_SENDER = "sender_benchmark_01"


@dataclass
class Emission:
    t_s: float        # 相对时间（秒）
    perception: dict  # 完整 PerceptionResult
    sender_id: str = DEFAULT_SENDER


def make_schedule(perceptions: List[dict], duration_s: float,
                  rate_hz: float = 20.0,
                  jitter_s: float = 0.008,
                  drop_prob: float = 0.05,
                  burst_prob: float = 0.03,
                  burst_extra: Tuple[int, int] = (2, 4),
                  gap_prob: float = 0.03,
                  gap_len_s: Tuple[float, float] = (1.2, 2.5),
                  sender_id: str = DEFAULT_SENDER,
                  seed: Optional[int] = None) -> List[Emission]:
    """生成一个发送方的到达计划（已按到达时刻排序）。"""
    rng = random.Random(seed)
    period = 1.0 / rate_hz
    events: List[Emission] = []
    t = 0.0
    i = 0
    while t < duration_s:
        if rng.random() < gap_prob:
            t += rng.uniform(*gap_len_s)  # 无样本区间 → 空窗口
            continue
        base = t
        count = 1
        if rng.random() < burst_prob:
            count += rng.randint(*burst_extra)
        for _ in range(count):
            if rng.random() < drop_prob:
                continue  # 该条被丢弃（未产生到达）
            arrival = base
            if jitter_s > 0:
                arrival += rng.uniform(-jitter_s, jitter_s)
            if arrival < 0:
                arrival = 0.0
            events.append(Emission(round(arrival, 6), perceptions[i % len(perceptions)], sender_id))
            i += 1
        t += period
    events.sort(key=lambda e: e.t_s)
    return events


def make_multi_sender_schedule(perceptions_by_sender: Dict[str, List[dict]],
                               duration_s: float, seed: Optional[int] = None,
                               **kwargs) -> List[Emission]:
    """多发送方：各自独立排程后合并排序（T4 用）。"""
    events: List[Emission] = []
    for i, (sid, percs) in enumerate(perceptions_by_sender.items()):
        events.extend(make_schedule(percs, duration_s, sender_id=sid,
                                    seed=(seed if seed is None else seed + i), **kwargs))
    events.sort(key=lambda e: e.t_s)
    return events


def run_schedule(schedule: List[Emission], pipeline, real_time: bool = True,
                 on_emit: Optional[callable] = None) -> float:
    """按计划以真实时间推进，逐个调用 pipeline.ingest。返回总耗时（秒）。

    窗口归属使用「计划到达时刻」（e.t_s，小相对值）而不是实际 sleep 完成时刻：
    real_time 的 sleep 只负责节奏，保证窗口划分对计划完全确定（聚合器内部用
    整数纳秒计算，不受系统调度抖动或大单调基数浮点精度影响）；真实线程并发
    （worker/推理）仍被完整测试。
    """
    t0 = time.monotonic()
    for e in schedule:
        if real_time:
            target = t0 + e.t_s
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            arrival = e.t_s  # 计划到达时刻（相对值），确定性窗口归属
        else:
            arrival = None
        pipeline.ingest(e.sender_id, e.perception, arrival)
        if on_emit is not None:
            on_emit(e)
    return time.monotonic() - t0


def schedule_stats(schedule: List[Emission]) -> dict:
    """汇总计划：总条数、按发送方计数、相邻到达最小/最大间隔、是否出现空窗间隔。"""
    by_sender: Dict[str, int] = {}
    gaps = []
    prev: Optional[float] = None
    for e in schedule:
        by_sender[e.sender_id] = by_sender.get(e.sender_id, 0) + 1
        if prev is not None:
            gaps.append(e.t_s - prev)
        prev = e.t_s
    return {
        "total": len(schedule),
        "by_sender": by_sender,
        "min_gap_s": round(min(gaps), 6) if gaps else None,
        "max_gap_s": round(max(gaps), 6) if gaps else None,
        "duration_s": round(prev, 6) if prev is not None else 0.0,
    }
