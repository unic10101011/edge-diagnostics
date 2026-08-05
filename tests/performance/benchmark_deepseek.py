#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek 边缘模型单体压测（第一阶段）。

链路：模拟 PerceptionResult -> 固定提示词 -> DeepSeek 本地推理 ->
      提取 JSON -> 校验输出字段 -> 记录耗时/吞吐/显存。

不依赖感知模块、编排器、HTTP、正式任务队列、代码降级和云端。

运行（WSL）：
    source ~/.venvs/edge-bench/bin/activate
    python3 tests/performance/benchmark_deepseek.py --config configs/benchmark.deepseek.yaml

输出：
    var/benchmark/results/aggregate_<run_id>.json   各场景汇总
    var/benchmark/results/aggregate_<run_id>.csv
    var/benchmark/results/requests_<run_id>.jsonl   单请求明细
    var/benchmark/results/env_<run_id>.json         环境与加载信息
    var/benchmark/logs/benchmark_<run_id>.log       运行日志
"""

import argparse
import csv
import datetime
import itertools
import json
import logging
import queue
import statistics
import threading
import time
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from generate_test_inputs import ensure_inputs_file
from output_validator import OUTPUT_SCHEMA_VERSION, validate_model_output

REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_run_id():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(logs_dir: Path, run_id: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logs_dir / f"benchmark_{run_id}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def build_prompt(tokenizer, perception):
    """构造固定结构提示词：只要求输出三个核心判断字段的 JSON。

    DeepSeek 与 Qwen 使用同一提示词，保证 A/B 同口径。
    """
    body = json.dumps(perception, ensure_ascii=False)
    system = (
        "你是轴承设备状态诊断助手。把给定的感知结果转换为一个 JSON 诊断结论。\n"
        "严格要求：\n"
        "1. 只输出一个 JSON 对象，禁止输出 <think>、任何解释、分析或思考过程，禁止 Markdown 代码块。\n"
        "2. JSON 只包含以下三个字段，不允许增加或缺失：\n"
        '   edge_result: 只能是 "normal" | "warning" | "fault"\n'
        '   edge_risk_level: 只能是 "low" | "medium" | "high"\n'
        "   confidence: [0,1] 的浮点数，表示诊断分数\n"
        "3. 立即输出 JSON，不要有任何前言。"
    )
    user = "感知结果：\n" + body + "\n\n请输出诊断 JSON："
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # 极少数 tokenizer 没有 chat_template 时的兜底
        return f"{system}\n\n{user}"


class InferenceLock:
    """可关闭的全局推理锁：serialize_inference=true 时并发请求串行化。"""

    def __init__(self, enabled: bool):
        self._lock = threading.Lock() if enabled else None

    def acquire(self):
        if self._lock is None:
            return None
        return self._lock


def run_one(model, tokenizer, prompt, gen_cfg, timeout_s, serialize_lock, logger,
            request_seq, arrival_ts=None):
    """执行一次推理并返回单请求记录。

    arrival_ts：请求到达的 monotonic 时间（秒）。用于计算排队等待和总延迟；
    为 None 时以本函数开始时刻为基准。
    """
    rec = {
        "request_id": "benchmark-%06d" % request_seq,
        "arrival_ts": round(arrival_ts, 6) if arrival_ts is not None else None,
        "queue_wait_ms": 0.0,
        "inference_latency_ms": None,
        "total_latency_ms": 0.0,
        "first_token_ms": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "output_valid": False,
        "validation_errors": [],
        "timed_out": False,
        "truncated": False,
        "error_type": None,
        "category": "unknown",
    }
    t_start = time.monotonic()
    ref_ts = arrival_ts if arrival_ts is not None else t_start

    # 分词
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    except Exception as exc:
        rec["error_type"] = "tokenize_error: %s" % exc
        rec["total_latency_ms"] = round((time.monotonic() - ref_ts) * 1000.0, 2)
        return rec
    rec["input_tokens"] = int(inputs["input_ids"].shape[1])

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout_s
    )
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": int(gen_cfg["max_new_tokens"]),
        "do_sample": bool(gen_cfg["do_sample"]),
        "pad_token_id": gen_cfg["pad_token_id"],
        "streamer": streamer,
        "return_dict_in_generate": True,
    }
    if gen_cfg["do_sample"]:
        gen_kwargs["temperature"] = float(gen_cfg["temperature"])

    holder = {}

    def _generate():
        try:
            with torch.inference_mode():
                lock = serialize_lock.acquire() if serialize_lock else None
                if lock is not None:
                    with lock:
                        t_inf = time.monotonic()
                        holder["start_gen_ts"] = t_inf
                        out = model.generate(**gen_kwargs)
                        holder["inference_latency_ms"] = (time.monotonic() - t_inf) * 1000.0
                else:
                    t_inf = time.monotonic()
                    holder["start_gen_ts"] = t_inf
                    out = model.generate(**gen_kwargs)
                    holder["inference_latency_ms"] = (time.monotonic() - t_inf) * 1000.0
                holder["out"] = out
                holder["ok"] = True
        except Exception as exc:
            holder["ok"] = False
            holder["err"] = exc

    t_gen = threading.Thread(target=_generate, daemon=True)
    t_gen.start()

    chunks = []
    first_ms = None
    streamer_timeout = False
    try:
        for chunk in streamer:
            if first_ms is None and chunk:
                first_ms = (time.monotonic() - t_start) * 1000.0
            chunks.append(chunk)
    except StopIteration:
        streamer_timeout = True  # 两次 token 间隔超过 timeout
    except Exception:
        pass

    t_gen.join(timeout=timeout_s + 5)
    timed_out = t_gen.is_alive() or streamer_timeout
    rec["first_token_ms"] = round(first_ms, 2) if first_ms is not None else None

    if timed_out:
        rec["timed_out"] = True
        rec["error_type"] = "timeout"
        rec["output_tokens"] = 0
        rec["total_latency_ms"] = round((time.monotonic() - ref_ts) * 1000.0, 2)
        return rec

    if not holder.get("ok"):
        err = holder.get("err")
        if isinstance(err, torch.cuda.OutOfMemoryError):
            rec["error_type"] = "cuda_oom"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            rec["error_type"] = "%s: %s" % (type(err).__name__, err)
        rec["total_latency_ms"] = round((time.monotonic() - ref_ts) * 1000.0, 2)
        return rec

    out = holder["out"]
    sequences = out.sequences[0] if hasattr(out, "sequences") else out[0]
    gen_ids = sequences[rec["input_tokens"]:]
    rec["output_tokens"] = int(gen_ids.numel())
    rec["truncated"] = rec["output_tokens"] >= int(gen_cfg["max_new_tokens"])

    text = "".join(chunks).strip()
    if not text:
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    validation = validate_model_output(text)
    rec["output_valid"] = validation["valid"]
    rec["validation_errors"] = validation["errors"]
    rec["had_preamble"] = validation["had_preamble"]

    start_gen_ts = holder.get("start_gen_ts")
    if start_gen_ts is not None:
        rec["queue_wait_ms"] = round((start_gen_ts - ref_ts) * 1000.0, 2)
        rec["infer_start_ts"] = round(start_gen_ts, 6)
        rec["infer_end_ts"] = round(
            start_gen_ts + holder.get("inference_latency_ms", 0.0) / 1000.0, 6)
    rec["inference_latency_ms"] = round(holder.get("inference_latency_ms", 0.0), 2)
    rec["total_latency_ms"] = round((time.monotonic() - ref_ts) * 1000.0, 2)
    return rec


def run_plain(model, tokenizer, inputs, gen_cfg, concurrency, count, duration_s,
              timeout_s, serialize_lock, logger, scenario, base_seq):
    """并发线程拉取任务，直到达到 count 或 duration。"""
    state = {"issued": 0}
    issue_lock = threading.Lock()
    deadline = time.monotonic() + duration_s if duration_s else None
    results = []
    results_lock = threading.Lock()
    seq = itertools.count(base_seq + 1)

    # 时长场景每 2s 采样一次已分配显存，用于观察是否随时间增长
    mem_samples = []
    mem_thread = None
    if duration_s and torch.cuda.is_available():
        def _mem_sampler():
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                mem_samples.append(torch.cuda.memory_allocated() / 1024 ** 2)
                time.sleep(2)
        mem_thread = threading.Thread(target=_mem_sampler, daemon=True)
        mem_thread.start()

    def worker():
        while True:
            with issue_lock:
                if count is not None and state["issued"] >= count:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                i = state["issued"]
                state["issued"] += 1
            arrival_ts = time.monotonic()
            perception = inputs[i % len(inputs)]
            rec = run_one(
                model, tokenizer, build_prompt(tokenizer, _clean_input(perception)),
                gen_cfg, timeout_s, serialize_lock, logger, next(seq), arrival_ts,
            )
            rec["scenario"] = scenario
            rec["category"] = perception.get("_category", "unknown")
            with results_lock:
                results.append(rec)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if mem_thread is not None:
        mem_thread.join(timeout=5)
    return results, base_seq + len(results), {"mem_samples": mem_samples}


def run_paced(model, tokenizer, inputs, gen_cfg, concurrency, rate, duration_s,
              timeout_s, serialize_lock, logger, scenario, base_seq):
    """固定速率压测：生产线程按 1/rate 节奏投递，N 个工作线程消费。"""
    jobs = queue.Queue()
    deadline = time.monotonic() + duration_s
    results = []
    results_lock = threading.Lock()
    seq = itertools.count(base_seq + 1)
    qstate = {"max_qsize": 0}
    qlock = threading.Lock()

    def worker():
        while True:
            try:
                item = jobs.get(timeout=1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    break
                continue
            if item is None:
                break
            arrival_ts, perception = item
            rec = run_one(
                model, tokenizer, build_prompt(tokenizer, _clean_input(perception)),
                gen_cfg, timeout_s, serialize_lock, logger, next(seq), arrival_ts,
            )
            rec["scenario"] = scenario
            rec["category"] = perception.get("_category", "unknown")
            rec["rate_per_second"] = rate
            with results_lock:
                results.append(rec)

    def producer():
        i = 0
        period = 1.0 / rate
        next_issue = time.monotonic()
        while time.monotonic() < deadline:
            wait = next_issue - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            perception = inputs[i % len(inputs)]
            i += 1
            jobs.put((time.monotonic(), perception))
            with qlock:
                qs = jobs.qsize()
                if qs > qstate["max_qsize"]:
                    qstate["max_qsize"] = qs
            next_issue += period
        for _ in range(concurrency):
            jobs.put(None)

    producer_t = threading.Thread(target=producer, daemon=True)
    producer_t.start()
    workers = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for w in workers:
        w.start()
    producer_t.join()
    for w in workers:
        w.join()
    return results, base_seq + len(results), {"mem_samples": [], "max_qsize": qstate["max_qsize"]}


def _clean_input(perception):
    """去掉内部字段 _category，保持交给模型的就是标准 PerceptionResult。"""
    return {k: v for k, v in perception.items() if k != "_category"}


def _pct(sorted_values, p):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def aggregate(records, meta, duration_s):
    ok = [r for r in records if r["error_type"] is None and not r["timed_out"]]
    total = len(records)
    valid = sum(1 for r in records if r["output_valid"])
    lat = sorted(r["inference_latency_ms"] for r in ok if r["inference_latency_ms"] is not None)
    tot = sorted(r["total_latency_ms"] for r in ok if r["total_latency_ms"] is not None)
    ft = sorted(r["first_token_ms"] for r in ok if r["first_token_ms"] is not None)
    out_tokens = sum(r["output_tokens"] or 0 for r in ok)

    return {
        "scenario": meta["scenario"],
        "concurrency": meta.get("concurrency"),
        "rate_per_second": meta.get("rate_per_second"),
        "requests_total": total,
        "success": len(ok),
        "failed": total - len(ok),
        "timeout": sum(1 for r in records if r["timed_out"]),
        "cuda_oom": sum(1 for r in records if r["error_type"] == "cuda_oom"),
        "json_valid_rate": round(valid / total, 4) if total else None,
        "throughput_req_per_s": round(len(ok) / duration_s, 3) if duration_s else None,
        "output_tokens_per_s": round(out_tokens / duration_s, 1) if duration_s else None,
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p50_latency_ms": round(_pct(lat, 0.5), 1) if lat else None,
        "p95_latency_ms": round(_pct(lat, 0.95), 1) if lat else None,
        "p99_latency_ms": round(_pct(lat, 0.99), 1) if lat else None,
        "avg_total_latency_ms": round(statistics.mean(tot), 1) if tot else None,
        "p95_total_latency_ms": round(_pct(tot, 0.95), 1) if tot else None,
        "avg_first_token_ms": round(statistics.mean(ft), 1) if ft else None,
        "p95_first_token_ms": round(_pct(ft, 0.95), 1) if ft else None,
        "avg_input_tokens": round(sum(r["input_tokens"] or 0 for r in ok) / len(ok), 1) if ok else None,
        "avg_output_tokens": round(sum(r["output_tokens"] or 0 for r in ok) / len(ok), 1) if ok else None,
        "truncated_count": sum(1 for r in records if r["truncated"]),
        "max_gpu_mem_mb": meta.get("max_gpu_mem_mb"),
        "mem_min_mb": meta.get("mem_min_mb"),
        "mem_max_mb": meta.get("mem_max_mb"),
        "max_queue_length": meta.get("max_queue_length"),
        "actual_duration_s": round(duration_s, 3),
        "template_version": meta.get("template_version"),
    }


def write_results(run_id, results_dir, env_info, aggregates, all_requests):
    results_dir.mkdir(parents=True, exist_ok=True)
    agg_path = results_dir / f"aggregate_{run_id}.json"
    agg_path.write_text(json.dumps({"run_id": run_id, "environment": env_info,
                                    "scenarios": aggregates}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    csv_path = results_dir / f"aggregate_{run_id}.csv"
    keys = list(aggregates[0].keys()) if aggregates else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(aggregates)

    req_path = results_dir / f"requests_{run_id}.jsonl"
    with open(req_path, "w", encoding="utf-8") as f:
        for r in all_requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    env_path = results_dir / f"env_{run_id}.json"
    env_path.write_text(json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return agg_path, csv_path, req_path, env_path


def collect_env_info(model_path, dtype, load_s, mem_before, mem_after, logger):
    info = {
        "model_path": model_path,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mem_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 0)
        if torch.cuda.is_available() else None,
        "mem_before_load_mb": round(mem_before / 1024**2, 1) if mem_before else None,
        "mem_after_load_mb": round(mem_after / 1024**2, 1) if mem_after else None,
        "model_load_seconds": round(load_s, 2),
        "input_template_version": None,  # 场景运行时填入
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }
    return info


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 边缘模型单体压测")
    parser.add_argument("--config", default="configs/benchmark.deepseek.yaml")
    parser.add_argument("--scenario", default=None,
                        help="只运行指定场景：warmup/single_baseline/low_concurrency/"
                             "medium_concurrency/high_concurrency/fixed_rate")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    inputs_jsonl = _resolve(cfg["input"]["inputs_jsonl"])
    results_dir = _resolve(cfg["output"]["results_dir"])
    logs_dir = _resolve(cfg["output"]["logs_dir"])

    run_id = _now_run_id()
    logger = setup_logging(logs_dir, run_id)

    # 输入准备
    created, template_version = ensure_inputs_file(
        inputs_jsonl,
        per_category=cfg["input"].get("per_category", 20),
        seed=cfg["input"].get("seed", 42),
    )
    inputs = []
    for line in inputs_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            inputs.append(json.loads(line))
    logger.info("inputs: %d items, template_version=%s, generated_now=%s",
                len(inputs), template_version, created)

    # 模型加载（只加载一次）
    model_cfg = cfg["model"]
    dtype = getattr(torch, model_cfg["dtype"]) if isinstance(model_cfg["dtype"], str) else model_cfg["dtype"]
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["path"],
        dtype=dtype,  # transformers>=5 已用 dtype 取代 torch_dtype
        device_map=model_cfg.get("device", "auto"),
        low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
        trust_remote_code=True,
    )
    model.eval()
    load_s = time.time() - t0
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    logger.info("model loaded in %.2fs", load_s)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_cfg = {
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "do_sample": cfg["generation"]["do_sample"],
        "temperature": cfg["generation"].get("temperature", 0),
        "pad_token_id": pad_token_id,
    }
    timeout_s = cfg["request"]["timeout_seconds"]
    serialize_lock = InferenceLock(cfg["request"].get("serialize_inference", True))

    env_info = collect_env_info(model_cfg["path"], str(dtype), load_s, mem_before, mem_after, logger)
    env_info["input_template_version"] = template_version

    # 场景执行
    sc = cfg["scenarios"]
    only = args.scenario
    aggregates = []
    all_requests = []
    base_seq = 0

    def _run_scenario(scenario, concurrency, count, duration_s, rate):
        nonlocal base_seq
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        if rate:
            records, base_seq, extra = run_paced(
                model, tokenizer, inputs, gen_cfg, concurrency, rate, duration_s,
                timeout_s, serialize_lock, logger, scenario, base_seq)
        else:
            records, base_seq, extra = run_plain(
                model, tokenizer, inputs, gen_cfg, concurrency, count, duration_s,
                timeout_s, serialize_lock, logger, scenario, base_seq)
        mem_samples = extra.get("mem_samples", [])
        elapsed = time.monotonic() - t0
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
        meta = {
            "scenario": scenario,
            "concurrency": concurrency,
            "rate_per_second": rate,
            "max_gpu_mem_mb": round(peak, 1) if peak else None,
            "mem_min_mb": round(min(mem_samples), 1) if mem_samples else None,
            "mem_max_mb": round(max(mem_samples), 1) if mem_samples else None,
            "max_queue_length": extra.get("max_qsize"),
            "template_version": template_version,
        }
        agg = aggregate(records, meta, elapsed)
        aggregates.append(agg)
        all_requests.extend(records)
        logger.info("scenario %s done: %d req, %.3f req/s, valid_rate=%.3f, p95=%.1fms, peak_mem=%sMB",
                    scenario, agg["requests_total"], agg["throughput_req_per_s"] or 0,
                    agg["json_valid_rate"] or 0, agg["p95_latency_ms"] or 0, agg["max_gpu_mem_mb"])
        return agg

    # 显式 --scenario 覆盖 enabled 标志；未指定时按 enabled 决定是否纳入全量运行
    def _sel(name, default_enabled=True):
        if only is not None:
            return only == name
        return sc.get(name, {}).get("enabled", default_enabled)

    if _sel("warmup"):
        _run_scenario("warmup", sc["warmup"]["concurrency"], sc["warmup"]["requests"], None, None)
    if _sel("single_baseline"):
        _run_scenario("single_baseline", sc["single_baseline"]["concurrency"],
                      sc["single_baseline"]["requests"], None, None)
    if _sel("stability", False) and "stability" in sc:
        _run_scenario("stability", sc["stability"]["concurrency"], None,
                      sc["stability"]["duration_seconds"], None)
    if _sel("low_concurrency"):
        _run_scenario("low_concurrency", sc["low_concurrency"]["concurrency"], None,
                      sc["low_concurrency"]["duration_seconds"], None)
    if _sel("medium_concurrency"):
        _run_scenario("medium_concurrency", sc["medium_concurrency"]["concurrency"], None,
                      sc["medium_concurrency"]["duration_seconds"], None)
    if _sel("high_concurrency"):
        _run_scenario("high_concurrency", sc["high_concurrency"]["concurrency"], None,
                      sc["high_concurrency"]["duration_seconds"], None)
    if _sel("fixed_rate"):
        fr = sc["fixed_rate"]
        concurrency = fr.get("concurrency", 1)
        steps = fr.get("steps")
        if steps:
            # 精简版：每档 (rate, duration_seconds) 独立时长
            for step in steps:
                _run_scenario("fixed_rate", concurrency, None,
                              step["duration_seconds"], step["rate"])
        else:
            # 兼容旧结构：concurrency_list × rates_per_second
            for c in fr.get("concurrency_list", [1]):
                for r in fr.get("rates_per_second", []):
                    _run_scenario("fixed_rate", c, None, fr.get("duration_seconds", 60), r)

    paths = write_results(run_id, results_dir, env_info, aggregates, all_requests)
    print("run_id:", run_id)
    for name, p in zip(["aggregate_json", "aggregate_csv", "requests_jsonl", "env_json"], paths):
        print(f"{name}: {p}")
    print("results_dir:", results_dir)


if __name__ == "__main__":
    main()
