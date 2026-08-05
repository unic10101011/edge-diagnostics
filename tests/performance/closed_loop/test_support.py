# -*- coding: utf-8 -*-
"""测试公共支撑：复用 generate_test_inputs 生成感知数据 + 线程安全的记录收集器。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_PERF_DIR = Path(__file__).resolve().parents[1]
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

from generate_test_inputs import generate_inputs  # noqa: E402


def sample_perceptions(per_category: int = 3, seed: int = 42) -> list:
    """返回打平后的 PerceptionResult 列表（去掉内部 _category 标记）。"""
    data = generate_inputs(per_category=per_category, seed=seed)
    out = []
    for _cat, items in data.items():
        for it in items:
            out.append({k: v for k, v in it.items() if k != "_category"})
    return out


class RecordCollector:
    """sink：收集 RunRecord，线程安全。"""

    def __init__(self):
        self._records = []
        self._lock = threading.Lock()

    def __call__(self, record):
        with self._lock:
            self._records.append(record)

    def get(self) -> list:
        with self._lock:
            return list(self._records)

    def reset(self):
        with self._lock:
            self._records.clear()
