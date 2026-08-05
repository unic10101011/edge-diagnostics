#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型输出 JSON 校验逻辑（不依赖 torch，可单独测试）。

校验目标：DeepSeek 根据轴承感知结果生成的诊断 JSON。
期望结构（edge-model-output/1.0）：

    {
      "edge_result": "normal" | "warning" | "fault",
      "edge_risk_level": "low" | "medium" | "high",
      "confidence": float(0~1) | null,       // 未校准诊断分数，不是真实概率
      "reason_codes": [str, ...],             // 可选
      "evidence": [ {...}, ... ],             // 可选，预留
      "recommended_actions": [str, ...]       // 可选
    }

返回 dict：
  valid      是否通过（errors 为空）
  errors     致命错误（JSON 解析失败、缺必填字段、枚举非法、confidence 越界）
  warnings   非致命提示（如 JSON 前后带 <think>/说明文字，能提取但费吞吐）
  parsed     解析出的 JSON 对象（可为 None）
  had_preamble / had_trailing
"""

import json

OUTPUT_SCHEMA_VERSION = "edge-model-output/1.0"

ALLOWED_EDGE_RESULT = {"normal", "warning", "fault"}
ALLOWED_RISK_LEVEL = {"low", "medium", "high"}
# 允许出现在模型输出中的字段；其余一律作为“多余字段”警告记录
ALLOWED_FIELDS = {"edge_result", "edge_risk_level", "confidence", "reason_codes", "recommended_actions"}


def extract_json_span(text):
    """从模型原始输出中截取 JSON 对象区间。

    用最后一个 { 到最后一个 }：模型通常先输出 <think> 推理（可能含花括号/代码块），
    JSON 是最后出现的目标对象，因此取最后一个 { 更稳。

    返回 (substring, had_preamble, had_trailing)。
    """
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, bool(text.strip()), False
    preamble = text[:start].strip()
    trailing = text[end + 1:].strip()
    return text[start:end + 1], bool(preamble), bool(trailing)


def _is_list_of_str(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def validate_model_output(text):
    """校验模型原始输出文本。"""
    errors = []
    warnings = []
    parsed = None

    if not text or not text.strip():
        return {
            "valid": False,
            "errors": ["empty_output"],
            "warnings": [],
            "parsed": None,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "had_preamble": False,
            "had_trailing": False,
        }

    span, had_preamble, had_trailing = extract_json_span(text)
    if span is None:
        # 区分“根本没有 JSON”和“JSON 被截断未闭合”（max_new_tokens 截断的典型形态）
        if "{" in text:
            errors = ["invalid_json:unterminated"]
        else:
            errors = ["no_json_object"]
        return {
            "valid": False,
            "errors": errors,
            "warnings": [],
            "parsed": None,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "had_preamble": bool(text.strip()),
            "had_trailing": False,
        }

    try:
        parsed = json.loads(span)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "errors": ["invalid_json: %s" % exc.msg],
            "warnings": [],
            "parsed": None,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "had_preamble": had_preamble,
            "had_trailing": had_trailing,
        }

    if not isinstance(parsed, dict):
        return {
            "valid": False,
            "errors": ["not_object"],
            "warnings": [],
            "parsed": parsed,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "had_preamble": had_preamble,
            "had_trailing": had_trailing,
        }

    # 必填字段
    if "edge_result" not in parsed:
        errors.append("missing_field:edge_result")
    elif parsed["edge_result"] not in ALLOWED_EDGE_RESULT:
        errors.append("invalid_edge_result:%s" % parsed["edge_result"])

    if "edge_risk_level" not in parsed:
        errors.append("missing_field:edge_risk_level")
    elif parsed["edge_risk_level"] not in ALLOWED_RISK_LEVEL:
        errors.append("invalid_edge_risk_level:%s" % parsed["edge_risk_level"])

    if "confidence" not in parsed:
        errors.append("missing_field:confidence")
    else:
        conf = parsed["confidence"]
        if conf is not None:
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                errors.append("invalid_confidence_type")
            elif not (0.0 <= float(conf) <= 1.0):
                errors.append("invalid_confidence_range:%s" % conf)

    # 可选字段：出现时校验类型
    if "reason_codes" in parsed and not _is_list_of_str(parsed["reason_codes"]):
        errors.append("invalid_reason_codes_type")
    if "recommended_actions" in parsed and not _is_list_of_str(parsed["recommended_actions"]):
        errors.append("invalid_recommended_actions_type")

    # 多余字段：提示词要求字段不能增加，出现即记录
    extra = sorted(set(parsed.keys()) - ALLOWED_FIELDS)
    if extra:
        warnings.append("extra_fields:%s" % ",".join(extra))

    # 模型输出 <think> 等推理文字：JSON 本身合法，能提取，但拖慢吞吐 → 非致命提示
    if had_preamble or had_trailing:
        warnings.append("non_json_wrapper")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "parsed": parsed,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "had_preamble": had_preamble,
        "had_trailing": had_trailing,
    }
