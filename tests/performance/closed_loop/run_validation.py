#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 / T4 闭环验证运行器（真实模型在 WSL/GPU 下运行；也可 --adapter mock 快速冒烟）。

- T2 单发送方稳定性：20Hz 感知输入 → 1s 窗口 → ~1 req/s 模型调用，跑至少 30 分钟，
  记录 P50/P95/P99 总延迟、最大队列、超时/降级/非法输出数、显存/内存、窗口遗漏。
- T4 双发送方过载：2 发送方 × 1 窗口/s = 2 req/s（超过 μ≈1.44），验证队列有界、
  过载快速降级、发送方不被长期饿死、降级结果不冒充模型结果。

用法（WSL，真实模型）：
    source ~/.venvs/edge-bench/bin/activate
    python -m tests.performance.closed_loop.run_validation \
        --config configs/closed_loop.validation.yaml --adapter real --scenario t2

用法（Windows，mock 冒烟）：
    python -m tests.performance.closed_loop.run_validation \
        --adapter mock --scenario t2 --duration 30

输出：
    var/closed_loop/results/<run_id>/events.jsonl     逐窗口事件
    var/closed_loop/results/<run_id>/aggregate.json   汇总指标
    var/closed_loop/results/<run_id>/aggregate.csv
    var/closed_loop/results/<run_id>/env.json         环境与配置
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
_PERF = Path(__file__).resolve().parents[1]
for _p in (_REPO, _PERF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from closed_loop.config import ClosedLoopConfig, load_config  # noqa: E402
from closed_loop.jitter_source import (  # noqa: E402
    make_multi_sender_schedule,
    make_schedule,
    run_schedule,
)
from closed_loop.model import EXECUTION_CODE_FALLBACK, EXECUTION_LOCAL_MODEL, RunRecord  # noqa: E402
from closed_loop.model_adapter import MockModel  # noqa: E402
from closed_loop.pipeline import ClosedLoopPipeline  # noqa: E402
from closed_loop.test_support import sample_perceptions  # noqa: E402

DEFAULT_CONFIG = "configs/closed_loop.validation.yaml"


def _now_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _pct(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return round(sorted_vals[f], 2)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 2)


def make_adapter(kind: str, cfg: ClosedLoopConfig):
    if kind == "mock":
        return MockModel(latency_ms=cfg.timeout.inference_ms / 2, jitter_ms=10.0)
    if kind == "real":
        from closed_loop.real_model import RealModel  # 惰性导入 torch
        m = cfg.model or {}
        adapter = RealModel(
            model_path=m.get("path", "/home/unic/models/Qwen2.5-1.5B-Instruct"),
            dtype=m.get("dtype", "bfloat16"),
            device=m.get("device", "auto"),
            max_new_tokens=int(m.get("max_new_tokens", 64)),
        )
        adapter.warmup(calls=2)  # 启动最小可用性检查，排除首推预热
        return adapter
    raise ValueError("adapter must be 'mock' or 'real'")


def _rss_mb() -> Optional[float]:
    """主机内存（RSS）MB。仅 Linux/WSL 可用，无第三方依赖。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        return None
    return None


def _latency_trend_p95(records: List[RunRecord], bucket_s: int = 60) -> Dict[str, float]:
    """按窗口时间桶（window_id//bucket_s）统计总延迟 P95，用于观察延迟是否随时间上升。

    窗口 ~1/s，window_id 可近似视为运行秒数；桶宽默认 60s。
    """
    buckets: Dict[int, List[float]] = {}
    for r in records:
        if r.is_empty or r.total_latency_ms is None:
            continue
        b = r.window_id // bucket_s
        buckets.setdefault(b, []).append(r.total_latency_ms)
    out = {}
    for b in sorted(buckets):
        v = sorted(buckets[b])
        out[str(b)] = _pct(v, 0.95)
    return out


def aggregate_records(records: List[RunRecord], duration_s: float,
                      max_queued: int, mem_samples: List[float],
                      rss_samples: Optional[List[float]] = None,
                      worker_alive_at_stop: Optional[bool] = None) -> dict:
    non_empty = [r for r in records if not r.is_empty]
    total = len(non_empty)
    local = sum(1 for r in non_empty if r.execution_mode == EXECUTION_LOCAL_MODEL)
    fallback = sum(1 for r in non_empty if r.execution_mode == EXECUTION_CODE_FALLBACK)
    valid = sum(1 for r in non_empty if r.output_valid)

    reasons: Dict[str, int] = {}
    for r in non_empty:
        if r.fallback_reason:
            reasons[r.fallback_reason] = reasons.get(r.fallback_reason, 0) + 1
    edge_hist: Dict[str, int] = {}
    for r in non_empty:
        if r.edge_result:
            edge_hist[r.edge_result] = edge_hist.get(r.edge_result, 0) + 1

    tot = sorted(r.total_latency_ms for r in non_empty if r.total_latency_ms is not None)
    qw = sorted(r.queue_wait_ms for r in non_empty if r.queue_wait_ms is not None)
    inf = sorted(r.inference_latency_ms for r in non_empty if r.inference_latency_ms is not None)

    return {
        "windows_total": len(records),
        "windows_non_empty": total,
        "windows_empty": len(records) - total,
        "local_model": local,
        "code_fallback": fallback,
        "fallback_rate": round(fallback / total, 4) if total else None,
        "output_valid_rate": round(valid / total, 4) if total else None,
        "fallback_reasons": reasons,
        "edge_results": edge_hist,
        "throughput_window_per_s": round(total / duration_s, 4) if duration_s else None,
        "total_latency_ms_p50": _pct(tot, 0.5),
        "total_latency_ms_p95": _pct(tot, 0.95),
        "total_latency_ms_p99": _pct(tot, 0.99),
        "queue_wait_ms_p50": _pct(qw, 0.5),
        "queue_wait_ms_p95": _pct(qw, 0.95),
        "inference_latency_ms_p50": _pct(inf, 0.5),
        "inference_latency_ms_p95": _pct(inf, 0.95),
        "exceeded_total_timeout_count": sum(1 for r in non_empty if r.exceeded_total_timeout),
        "max_observed_queued": max_queued,
        "late_dropped_total": sum(r.late_dropped_count for r in records),
        "latency_trend_p95_by_60s": _latency_trend_p95(non_empty, bucket_s=60),
        "mem_min_mb": round(min(mem_samples), 1) if mem_samples else None,
        "mem_max_mb": round(max(mem_samples), 1) if mem_samples else None,
        "rss_min_mb": round(min(rss_samples), 1) if rss_samples else None,
        "rss_max_mb": round(max(rss_samples), 1) if rss_samples else None,
        "worker_alive_at_stop": worker_alive_at_stop,
        "actual_duration_s": round(duration_s, 3),
    }


def run_scenario(cfg: ClosedLoopConfig, adapter, scenario: str, duration_s: float,
                 sender_count: int, rate_hz: int, out_dir: Path, schedule_mode: str = "clean"):
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    lock = threading.Lock()
    records: List[RunRecord] = []

    def sink(rec: RunRecord):
        with lock:
            records.append(rec)
            events_path.open("a", encoding="utf-8").write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")

    pipeline = ClosedLoopPipeline(cfg, adapter, sink)
    pipeline.start()

    # 显存 + 主机 RSS 采样（每 2s；RSS 无第三方依赖，仅 Linux/WSL）
    mem_samples: List[float] = []
    rss_samples: List[float] = []
    stop_mem = threading.Event()

    def _mem_sampler():
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False
        while not stop_mem.is_set():
            if has_torch:
                try:
                    mem_samples.append(torch.cuda.memory_allocated() / 1024 ** 2)
                except Exception:
                    pass
            rss = _rss_mb()
            if rss is not None:
                rss_samples.append(rss)
            time.sleep(2)

    mem_t = threading.Thread(target=_mem_sampler, daemon=True)
    mem_t.start()

    percs = sample_perceptions(per_category=5, seed=42)
    if scenario == "t2":
        if schedule_mode == "clean":
            # 干净 20Hz 固定节拍：稳态稳定性测试用（无丢包/突发/空窗）
            schedule = make_schedule(percs, duration_s=duration_s, rate_hz=rate_hz,
                                     sender_id="sender_t2", seed=42,
                                     jitter_s=0.0, drop_prob=0.0, burst_prob=0.0, gap_prob=0.0)
        else:
            schedule = make_schedule(percs, duration_s=duration_s, rate_hz=rate_hz,
                                     sender_id="sender_t2", seed=42)
    else:  # t4
        by_sender = {f"sender_a{i}": percs for i in range(sender_count)}
        schedule = make_multi_sender_schedule(by_sender, duration_s=duration_s, seed=42)

    t0 = time.monotonic()
    run_schedule(schedule, pipeline)
    pipeline.flush()
    pipeline.wait_idle(timeout_s=30)
    worker_alive = pipeline.worker._thread.is_alive()  # stop 前捕获
    pipeline.stop()
    stop_mem.set()
    elapsed = time.monotonic() - t0

    agg = aggregate_records(records, elapsed, pipeline.max_observed_queued, mem_samples,
                            rss_samples=rss_samples, worker_alive_at_stop=worker_alive)
    agg["scenario"] = scenario
    agg["sender_count"] = sender_count
    return agg


def main():
    ap = argparse.ArgumentParser(description="闭环验证 T2/T4")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--adapter", choices=["mock", "real"], default="real")
    ap.add_argument("--schedule", choices=["clean", "jittery"], default="clean",
                    help="clean=固定20Hz无空窗（稳态稳定测试）；jittery=模拟丢包/突发/空窗")
    ap.add_argument("--scenario", choices=["t2", "t4"], required=True)
    ap.add_argument("--duration", type=float, default=None, help="覆盖场景时长（秒），冒烟用")
    ap.add_argument("--sender-count", type=int, default=None)
    ap.add_argument("--inference-ms", type=int, default=None, help="覆盖推理超时（第4步用）")
    ap.add_argument("--queue-wait-ms", type=int, default=None, help="覆盖排队超时")
    ap.add_argument("--breaker-threshold", type=int, default=None, help="覆盖熔断阈值")
    ap.add_argument("--probe-interval", type=float, default=None, help="覆盖恢复探测周期（秒）")
    ap.add_argument("--out", default=None, help="输出目录（默认 var/closed_loop/results/<run_id>）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.inference_ms is not None:
        cfg.timeout.inference_ms = args.inference_ms
    if args.queue_wait_ms is not None:
        cfg.timeout.queue_wait_ms = args.queue_wait_ms
    if args.breaker_threshold is not None:
        cfg.breaker.consecutive_failure_threshold = args.breaker_threshold
    if args.probe_interval is not None:
        cfg.breaker.recovery_probe_interval_s = args.probe_interval
    sc = cfg.scenarios or {}
    if args.scenario == "t2":
        duration = args.duration if args.duration else (sc.get("t2_stability", {}).get("duration_seconds", 1800))
        sender_count = args.sender_count or sc.get("t2_stability", {}).get("sender_count", 1)
        rate = sc.get("t2_stability", {}).get("perception_rate_hz", 20)
    else:
        duration = args.duration if args.duration else (sc.get("t4_two_senders", {}).get("duration_seconds", 600))
        sender_count = args.sender_count or sc.get("t4_two_senders", {}).get("sender_count", 2)
        rate = sc.get("t4_two_senders", {}).get("perception_rate_hz", 20)

    adapter = make_adapter(args.adapter, cfg)

    base_dir = _REPO / (args.out or ("var/closed_loop/results/" + _now_run_id()))
    agg = run_scenario(cfg, adapter, args.scenario, duration, sender_count, rate, base_dir,
                       schedule_mode=args.schedule)

    (base_dir / "aggregate.json").write_text(
        json.dumps({"run_id": base_dir.name, "scenario": args.scenario,
                    "adapter": args.adapter, "config": cfg.as_dict(),
                    "aggregate": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(base_dir / "aggregate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in agg.items():
            w.writerow([k, json.dumps(v, ensure_ascii=False) if not isinstance(v, (int, float)) else v])
    (base_dir / "env.json").write_text(json.dumps(
        {"python": sys.version.split()[0], "adapter": args.adapter,
         "config": cfg.as_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("run_id:", base_dir.name)
    for k, v in agg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
