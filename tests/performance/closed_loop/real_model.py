# -*- coding: utf-8 -*-
"""真实模型适配器：包装 Transformers 推理，复用现有提示词构造。

只在 WSL/GPU 场景使用（T2 稳定性 / T4 过载）。torch/transformers 全部惰性导入，
Windows 上跑 closed_loop 的 pytest 时不会被本模块拉入。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .model_adapter import InferenceAdapter, InferenceOutcome

_PERF_DIR = Path(__file__).resolve().parents[1]


def _import_runtime():
    import torch  # noqa: F401
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    if str(_PERF_DIR) not in sys.path:
        sys.path.insert(0, str(_PERF_DIR))
    from benchmark_deepseek import build_prompt
    return torch, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, build_prompt


class RealModel(InferenceAdapter):
    def __init__(self, model_path: str, dtype: str = "bfloat16", device: str = "auto",
                 max_new_tokens: int = 64, streamer_timeout_s: float = 30.0,
                 low_cpu_mem_usage: bool = True):
        torch, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, build_prompt = _import_runtime()
        self._torch = torch
        self._TextIteratorStreamer = TextIteratorStreamer
        self.max_new_tokens = max_new_tokens
        self.streamer_timeout_s = streamer_timeout_s

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype_obj, device_map=device,
            low_cpu_mem_usage=low_cpu_mem_usage, trust_remote_code=True,
        )
        self.model.eval()
        self.pad_token_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                             else self.tokenizer.eos_token_id)
        self._build_prompt = build_prompt
        # 推理锁：worker 超时后仍会启动下一个 infer 线程，此时上一个 generate
        # 可能还在后台运行。串行化保证任何时刻只有一个 model.generate，
        # 避免原生 Transformers 并发调用同一模型（benchmark 中 serialize_inference 的等价物）。
        self._infer_lock = threading.Lock()

    def warmup(self, calls: int = 2, max_new_tokens: int = 8):
        """启动时最小可用性检查：跑 2 次极短推理，排除首次编译/预热对超时统计的污染。

        对应文档「启动时加载模型后执行一次最小可用性检查」；预热不参与推理超时判定。
        """
        probe = {
            "perception_quality": {"status": "good", "flags": []},
            "features": {
                "vibration": {"rms": 0.3, "absolute_peak": 1.8, "kurtosis": 3.1,
                              "dominant_frequency_hz": 120.0, "band_power_ratio_500_2000": 0.3,
                              "spectral_entropy": 0.6},
            },
        }
        saved = self.max_new_tokens
        self.max_new_tokens = max_new_tokens
        try:
            for _ in range(calls):
                self.infer(probe)
        finally:
            self.max_new_tokens = saved

    def infer(self, model_input: dict) -> InferenceOutcome:
        torch = self._torch
        t_start = time.monotonic()
        with self._infer_lock:  # 串行化：超时线程未结束前，后续 infer 等待锁，绝不并发 generate
            return self._infer_locked(model_input, torch, t_start)

    def _infer_locked(self, model_input: dict, torch, t_start: float) -> InferenceOutcome:
        prompt = self._build_prompt(self.tokenizer, model_input)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        except Exception as exc:  # noqa: BLE001
            return InferenceOutcome(success=False, latency_ms=(time.monotonic() - t_start) * 1000.0,
                                    error="tokenize_error: %s" % exc)

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
        holder: dict = {}

        def _generate():
            try:
                with torch.inference_mode():
                    t0 = time.monotonic()
                    holder["latency_ms"] = None
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
        if t_gen.is_alive() or streamer_timeout:
            return InferenceOutcome(success=False, timed_out=True,
                                    latency_ms=(time.monotonic() - t_start) * 1000.0)
        if not holder.get("ok"):
            return InferenceOutcome(success=False, timed_out=False,
                                    latency_ms=(time.monotonic() - t_start) * 1000.0,
                                    error="%s: %s" % (type(holder.get("err")).__name__, holder.get("err")))

        text = "".join(chunks).strip()
        return InferenceOutcome(success=True, text=text,
                                latency_ms=holder.get("latency_ms") or (time.monotonic() - t_start) * 1000.0)
