# -*- coding: utf-8 -*-
"""闭环验证工具配置。

所有参数均为『待压测后冻结』，不是最终生产值。默认值集中在 DEFAULTS，
`configs/closed_loop.validation.yaml` 是对外的文档化覆盖入口。测试直接用
`default_config()`，不依赖 yaml 文件，保证 Windows 上无额外依赖也能跑。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional

from .model import REASON_QUEUE_TIMEOUT


@dataclass
class WindowConfig:
    length_seconds: float = 1.0          # TEST: 窗口长度
    expected_samples: int = 20           # 每发送方 20 包/秒
    min_samples_for_full: int = 15       # TEST: 稀疏判定阈值
    late_tolerance_seconds: float = 0.25  # TEST: 迟到容忍（窗口边界归属）
    submit_sparse_window: bool = True    # 稀疏窗口仍提交模型/规则，仅标记质量
    on_empty_window: str = "record_only"  # record_only | skip_model（空窗口不调模型）


@dataclass
class QueueConfig:
    capacity: int = 1                    # TEST: 1 或 2
    worker_count: int = 1
    # drop_current_to_fallback | replace_oldest_pending
    full_policy: str = "drop_current_to_fallback"


@dataclass
class TimeoutConfig:
    queue_wait_ms: int = 250             # TEST: 200~300
    inference_ms: int = 1500             # TEST:
    total_ms: int = 2000                 # TEST:
    # 推理已完成但总时长超限时：deliver_late_model_result（返回模型结果并标记迟到）
    # 或 fallback（丢弃模型结果，改走规则）
    on_total_timeout_after_completion: str = "deliver_late_model_result"


@dataclass
class BreakerConfig:
    enabled: bool = True
    consecutive_failure_threshold: int = 5   # TEST:
    recovery_probe_interval_s: float = 60.0  # TEST:
    probe_inference_timeout_ms: int = 2000


@dataclass
class FallbackConfig:
    rule_version: str = "edge_rule_v1.0"


@dataclass
class ClosedLoopConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    # 透传 yaml 中的 model / scenarios 段（T2/T4 运行器用），不参与 dataclass 校验
    model: Dict = field(default_factory=dict)
    scenarios: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        if self.model:
            d["model"] = self.model
        if self.scenarios:
            d["scenarios"] = self.scenarios
        return d


def default_config() -> ClosedLoopConfig:
    return ClosedLoopConfig()


def load_config(path: Optional[str] = None) -> ClosedLoopConfig:
    """从 yaml 覆盖默认值；文件不存在或 yaml 不可用时返回默认配置。"""
    cfg = default_config()
    if not path:
        return cfg
    try:
        import yaml
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cfg
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return cfg
    return apply_dict(cfg, raw)


def apply_dict(cfg: ClosedLoopConfig, raw: Dict) -> ClosedLoopConfig:
    """把 yaml dict 按嵌套结构覆盖到 dataclass（只覆盖已存在的字段）。"""
    for section, sub in {
        "window": cfg.window, "queue": cfg.queue, "timeout": cfg.timeout,
        "breaker": cfg.breaker, "fallback": cfg.fallback,
    }.items():
        data = raw.get(section)
        if isinstance(data, dict):
            for k, v in data.items():
                if hasattr(sub, k) and v is not None:
                    setattr(sub, k, v)
    # 透传 model / scenarios 原始 dict
    if isinstance(raw.get("model"), dict):
        cfg.model = raw["model"]
    if isinstance(raw.get("scenarios"), dict):
        cfg.scenarios = raw["scenarios"]
    return cfg
