# -*- coding: utf-8 -*-
"""Qwen 模型运行器（WSL 侧，torch/transformers 惰性导入）。

职责：加载模型、warmup、串行调用 generate()、解析校验输出、返回结果。
推理锁保证任何时刻只有一个 generate()；HTTP 超时后推理可能仍在执行（逻辑超时
不中止 generate），锁确保不并发进入 generate()。永久卡死当前阶段无法处理，
文档需写明该限制。
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from .output_validator import validate_model_output
from .prompt import build_prompt


class ModelRunner:
    def __init__(self, model_path: str, model_version: str = "qwen2.5-1.5b-instruct/phase1",
                 dtype: str = "bfloat16", device: str = "auto",
                 max_new_tokens: int = 64, streamer_timeout_s: float = 30.0,
                 low_cpu_mem_usage: bool = True, warmup_calls: int = 2):
        self.model_path = model_path
        self.model_version = model_version
        self.max_new_tokens = max_new_tokens
        self.streamer_timeout_s = streamer_timeout_s
        self._infer_lock = threading.Lock()  # 串行化 generate（超时后也绝不并发）
        self._ready = False
        self._load_error: Optional[str] = None

        self._torch, self._AutoModelForCausalLM, self._AutoTokenizer, self._TextIteratorStreamer = self._import_runtime()
        self.tokenizer, self.model = self._load(dtype, device, low_cpu_mem_usage)
        self.pad_token_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                             else self.tokenizer.eos_token_id)
        self._warmup(warmup_calls)
        self._ready = True

    @staticmethod
    def _import_runtime():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        return torch, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    def _load(self, dtype, device, low_cpu_mem_usage):
        try:
            tokenizer = self._AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            dtype_obj = getattr(self._torch, dtype) if isinstance(dtype, str) else dtype
            model = self._AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=dtype_obj, device_map=device,
                low_cpu_mem_usage=low_cpu_mem_usage, trust_remote_code=True)
            model.eval()
            return tokenizer, model
        except Exception as exc:  # noqa: BLE001
            self._load_error = "%s: %s" % (type(exc).__name__, exc)
            raise

    def _warmup(self, calls: int):
        """启动最小可用性检查：短推理，排除首次编译/预热。"""
        probe = {
            "perception_quality": {"status": "good", "flags": []},
            "features": {"vibration": {"rms": 0.3, "absolute_peak": 1.8, "kurtosis": 3.1,
                                       "dominant_frequency_hz": 120.0,
                                       "band_power_ratio_500_2000": 0.3, "spectral_entropy": 0.6}},
        }
        saved = self.max_new_tokens
        self.max_new_tokens = 8
        try:
            for _ in range(calls):
                self.infer(probe)
        finally:
            self.max_new_tokens = saved

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ---- 推理（HTTP 处理器调用） ----
    def infer(self, model_input: dict) -> Dict:
        t_start = time.monotonic()
        with self._infer_lock:
            try:
                return self._infer_locked(model_input, t_start)
            except Exception as exc:  # noqa: BLE001
                return {"valid": False, "error": "%s: %s" % (type(exc).__name__, exc),
                        "latency_ms": round((time.monotonic() - t_start) * 1000.0, 2)}

    def _infer_locked(self, model_input: dict, t_start: float) -> Dict:
        torch = self._torch
        prompt = build_prompt(self.tokenizer, model_input)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": "tokenize_error: %s" % exc,
                    "latency_ms": round((time.monotonic() - t_start) * 1000.0, 2)}

        streamer = self._TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=self.streamer_timeout_s)
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.pad_token_id,
            "streamer": streamer,
            "return_dict_in_generate": True,
        }
        holder: Dict = {}

        def _generate():
            try:
                with torch.inference_mode():
                    t0 = time.monotonic()
                    holder["out"] = self.model.generate(**gen_kwargs)
                    holder["latency_ms"] = (time.monotonic() - t0) * 1000.0
                holder["ok"] = True
            except Exception as exc:  # noqa: BLE001
                holder["ok"] = False
                holder["err"] = exc

        t_gen = threading.Thread(target=_generate, daemon=True)
        t_gen.start()
        chunks = []
        streamer_timeout = False
        try:
            for chunk in streamer:
                chunks.append(chunk)
        except StopIteration:
            streamer_timeout = True
        except Exception:  # noqa: BLE001
            pass
        t_gen.join(timeout=self.streamer_timeout_s + 5)
        latency_ms = (time.monotonic() - t_start) * 1000.0

        if t_gen.is_alive() or streamer_timeout:
            return {"valid": False, "error": "stream_timeout",
                    "latency_ms": round(latency_ms, 2)}
        if not holder.get("ok"):
            return {"valid": False, "error": "%s: %s" % (type(holder.get("err")).__name__, holder.get("err")),
                    "latency_ms": round(latency_ms, 2)}

        text = "".join(chunks).strip()
        validation = validate_model_output(text)
        if not validation["valid"]:
            return {"valid": False, "error": "output_invalid:%s" % ",".join(validation["errors"]),
                    "latency_ms": round(latency_ms, 2), "raw_text": text}

        parsed = validation["parsed"]
        return {
            "valid": True,
            "edge_result": parsed["edge_result"],
            "edge_risk_level": parsed["edge_risk_level"],
            "confidence": float(parsed.get("confidence") or 0.0),
            "model_version": self.model_version,
            "raw_text": text,
            "latency_ms": round(holder.get("latency_ms") or latency_ms, 2),
        }
