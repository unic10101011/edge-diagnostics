# -*- coding: utf-8 -*-
"""模型提示词构造。

与 tests/performance/benchmark_deepseek.build_prompt 同源（该提示词已通过
Qwen2.5-1.5B 实测：JSON 合法率 100%）。任何提示词改动都必须更新 prompt 版本
并重跑 tests/performance/closed_loop。
"""
from __future__ import annotations

import json

PROMPT_VERSION = "edge-model-prompt/1.0"


def build_prompt(tokenizer, model_input: dict) -> str:
    """构造固定结构提示词：只要求输出三个核心判断字段的 JSON。"""
    body = json.dumps(model_input, ensure_ascii=False)
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
    except Exception:  # noqa: BLE001 极少数 tokenizer 没有 chat_template 时兜底
        return f"{system}\n\n{user}"
