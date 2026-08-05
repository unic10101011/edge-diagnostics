# -*- coding: utf-8 -*-
"""WSL 模型服务 HTTP 客户端（stdlib urllib，零额外依赖）。

调用 src/model_service 的 /infer、/health、/readiness。
inference_timeout 在两层生效：HTTP 读取超时（本客户端）＋ worker 侧 join 兜底。
超时是逻辑超时：WSL 侧已开始的 Transformers generate 不会被中止，推理锁保证
同一模型不并发进入 generate()。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import ModelClientConfig
from .contracts import EdgeResult


@dataclass
class ModelInferResult:
    success: bool
    timed_out: bool = False
    edge: Optional[EdgeResult] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class HealthResult:
    ok: bool
    detail: str = ""


class ModelClient:
    def __init__(self, cfg: ModelClientConfig, clock=time.monotonic):
        self.cfg = cfg
        self._clock = clock

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + path

    def _request_json(self, path: str, payload: Optional[dict] = None,
                      read_timeout_s: Optional[float] = None) -> Dict[str, Any]:
        url = self._url(path)
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
        timeout = read_timeout_s if read_timeout_s is not None else self.cfg.read_timeout_ms / 1000.0
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    # ---- 健康检查 ----
    def health(self) -> HealthResult:
        try:
            body = self._request_json(self.cfg.health_path, read_timeout_s=self.cfg.connect_timeout_ms / 1000.0)
            return HealthResult(ok=body.get("status") == "ok", detail=str(body))
        except Exception as exc:  # noqa: BLE001
            return HealthResult(ok=False, detail="%s: %s" % (type(exc).__name__, exc))

    def readiness(self) -> HealthResult:
        try:
            body = self._request_json(self.cfg.readiness_path, read_timeout_s=self.cfg.connect_timeout_ms / 1000.0)
            return HealthResult(ok=body.get("ready") is True, detail=str(body))
        except Exception as exc:  # noqa: BLE001
            return HealthResult(ok=False, detail="%s: %s" % (type(exc).__name__, exc))

    # ---- 推理 ----
    def infer(self, model_input: dict, inference_timeout_ms: Optional[int] = None) -> ModelInferResult:
        t0 = self._clock()
        read_timeout = (inference_timeout_ms or self.cfg.read_timeout_ms) / 1000.0
        try:
            body = self._request_json(self.cfg.infer_path, {"input": model_input},
                                      read_timeout_s=read_timeout)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            # socket.timeout 是 TimeoutError（py3.10+）→ timed_out
            if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                return ModelInferResult(success=False, timed_out=True,
                                        latency_ms=(self._clock() - t0) * 1000.0,
                                        error="http_timeout")
            return ModelInferResult(success=False, timed_out=False,
                                    latency_ms=(self._clock() - t0) * 1000.0,
                                    error="connection_error: %s" % reason)
        except Exception as exc:  # noqa: BLE001
            return ModelInferResult(success=False, timed_out=False,
                                    latency_ms=(self._clock() - t0) * 1000.0,
                                    error="%s: %s" % (type(exc).__name__, exc))
        latency_ms = (self._clock() - t0) * 1000.0

        if body.get("valid") is not True:
            return ModelInferResult(success=False, timed_out=False, latency_ms=latency_ms,
                                    error=body.get("error") or "service_invalid")
        edge = EdgeResult(
            edge_result=body["edge_result"],
            confidence=float(body["confidence"]),
            edge_risk_level=body["edge_risk_level"],
            model_version=body["model_version"],
        )
        return ModelInferResult(success=True, edge=edge, latency_ms=latency_ms,
                                raw_text=body.get("raw_text"))
