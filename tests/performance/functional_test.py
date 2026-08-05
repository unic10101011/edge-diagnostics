#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20 条功能测试：验证模型能否稳定、快速输出短 JSON。

覆盖：5 正常 / 5 低风险 / 5 中高风险 / 5 边界异常。
检查：合法 JSON、缺字段、多字段、枚举合法、confidence 数值、Markdown 代码块、
额外解释、64 token 是否截断、判断是否基本合理。

进入性能测试的最低门槛：
    JSON 合法率 = 100%
    无输出截断
    平均输出 <= max_new_tokens（默认 64）
    字段校验通过率 = 100%

用法（WSL）：
    source ~/.venvs/edge-bench/bin/activate
    python3 tests/performance/functional_test.py \
        --model /home/unic/models/Qwen2.5-1.5B-Instruct \
        --inputs var/benchmark/inputs.jsonl --max-new-tokens 64
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_deepseek import build_prompt  # noqa: E402
from output_validator import validate_model_output  # noqa: E402

# 每类的期望 edge_result（启发式，用于提示“判断是否基本合理”，不构成硬性门槛）
EXPECTED = {
    "normal": "normal",
    "low": ["normal", "warning"],
    "mid_high": ["warning", "fault"],
    "anomaly": "fault",
}


def pick20(lines):
    buckets = {"normal": [], "low": [], "mid_high": [], "anomaly": []}
    for line in lines:
        p = json.loads(line)
        cat = p.get("_category")
        if cat == "normal" and len(buckets["normal"]) < 5:
            buckets["normal"].append(p)
        elif cat == "risk":
            kurt = p["features"]["vibration"]["kurtosis"]
            key = "low" if kurt < 6.0 else "mid_high"
            if len(buckets[key]) < 5:
                buckets[key].append(p)
        elif cat == "anomaly" and len(buckets["anomaly"]) < 5:
            buckets["anomaly"].append(p)
        if all(len(v) >= 5 for v in buckets.values()):
            break
    out = []
    for k in ("normal", "low", "mid_high", "anomaly"):
        out.append((k, buckets[k]))
    return out


def main():
    ap = argparse.ArgumentParser(description="20条功能测试")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", default="var/benchmark/inputs.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id
    print("model ready:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    lines = [l for l in Path(args.inputs).read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = pick20(lines)

    rows = []
    for category, items in groups:
        for idx, perception in enumerate(items):
            inp = {k: v for k, v in perception.items() if k != "_category"}
            prompt = build_prompt(tok, inp)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            t0 = time.monotonic()
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad)
            gen = out[0][inputs["input_ids"].shape[1]:]
            latency_ms = (time.monotonic() - t0) * 1000.0
            text = tok.decode(gen, skip_special_tokens=True)
            v = validate_model_output(text)
            output_tokens = int(gen.numel())
            truncated = output_tokens >= args.max_new_tokens
            actual = v["parsed"].get("edge_result") if isinstance(v["parsed"], dict) else None
            expected = EXPECTED[category]
            reasonable = actual in expected if isinstance(expected, list) else actual == expected
            rows.append({
                "case": f"{category}#{idx + 1}", "category": category,
                "valid": v["valid"], "errors": v["errors"], "warnings": v["warnings"],
                "output_tokens": output_tokens, "truncated": truncated,
                "latency_ms": round(latency_ms, 1),
                "edge_result": actual, "reasonable": reasonable,
            })

    # 汇总
    n = len(rows)
    valid_n = sum(r["valid"] for r in rows)
    trunc_n = sum(r["truncated"] for r in rows)
    fenced = sum(1 for r in rows if any("```" in w for w in r["warnings"]) or
                 any(w.startswith("non_json_wrapper") for w in r["warnings"]))
    field_ok = sum(1 for r in rows if r["valid"] and not any(
        w.startswith("extra_fields") for w in r["warnings"]))
    avg_out = statistics.mean(r["output_tokens"] for r in rows)
    avg_lat = statistics.mean(r["latency_ms"] for r in rows)
    reasonable_n = sum(r["reasonable"] for r in rows)

    print("\n%-10s %-7s %-6s %-7s %-5s %-8s %-9s %-11s %-9s" % (
        "case", "valid", "tok", "trunc", "enum", "fieldOK", "latency_ms", "edge_result", "reasonable"))
    for r in rows:
        field_ok_f = r["valid"] and not any(w.startswith("extra_fields") for w in r["warnings"])
        print("%-10s %-7s %-6s %-7s %-5s %-8s %-9s %-11s %-9s" % (
            r["case"], r["valid"], r["output_tokens"], r["truncated"],
            "ok" if r["valid"] else ",".join(r["errors"]),
            field_ok_f, r["latency_ms"], r["edge_result"], r["reasonable"]))

    print("\n===== 汇总 =====")
    print(f"JSON合法率: {valid_n}/{n} = {valid_n / n:.0%}")
    print(f"无截断: {n - trunc_n}/{n}")
    print(f"字段严格(无多余字段): {field_ok}/{n}")
    print(f"平均输出token: {avg_out:.1f}")
    print(f"平均延迟: {avg_lat:.0f} ms")
    print(f"判断基本合理: {reasonable_n}/{n}")

    print("\n===== 进入性能测试的门槛判定 =====")
    gates = {
        "JSON合法率=100%": valid_n == n,
        "无输出截断": trunc_n == 0,
        "平均输出<=%d" % args.max_new_tokens: avg_out <= args.max_new_tokens,
        "字段校验通过率=100%": field_ok == n,
    }
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if all(gates.values()):
        print("\n结论：通过，可进入单请求基准测试。")
    else:
        print("\n结论：未通过，先调整提示词或输出解析策略，不要进入并发测试。")


if __name__ == "__main__":
    main()
