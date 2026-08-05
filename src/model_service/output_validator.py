# -*- coding: utf-8 -*-
"""模型输出 JSON 校验（与 tests/performance/output_validator 同源，无 torch 依赖）。

期望结构（edge-model-output/1.0）：
    {
      "edge_result": "normal" | "warning" | "fault",
      "edge_risk_level": "low" | "medium" | "high",
      "confidence": float(0~1) | null,
      "reason_codes": [str, ...],              // 可选
      "recommended_actions": [str, ...]        // 可选
    }
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

OUTPUT_SCHEMA_VERSION = "edge-model-output/1.0"

ALLOWED_EDGE_RESULT = {"normal", "warning", "fault"}
ALLOWED_RISK_LEVEL = {"low", "medium", "high"}
ALLOWED_FIELDS = {"edge_result", "edge_risk_level", "confidence", "reason_codes", "recommended_actions"}


def extract_json_span(text: str):
    """截取模型原始输出中的 JSON 对象区间（取最后一个 { 到最后一个 }）。"""
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, bool(text.strip()), False
    preamble = text[:start].strip()
    trailing = text[end + 1:].strip()
    return text[start:end + 1], bool(preamble), bool(trailing)


def _is_list_of_str(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def validate_model_output(text: str) -> Dict:
    """校验模型输出文本，返回 valid / errors / warnings / parsed。"""
    errors: List[str] = []
    warnings: List[str] = []
    parsed = None

    if not text or not text.strip():
        return {"valid": False, "errors": ["empty_output"], "warnings": [],
                "parsed": None, "schema_version": OUTPUT_SCHEMA_VERSION,
                "had_preamble": False, "had_trailing": False}

    span, had_preamble, had_trailing = extract_json_span(text)
    if span is None:
        if "{" in text:
            errors = ["invalid_json:unterminated"]
        else:
            errors = ["no_json_object"]
        return {"valid": False, "errors": errors, "warnings": [],
                "parsed": None, "schema_version": OUTPUT_SCHEMA_VERSION,
                "had_preamble": bool(text.strip()), "had_trailing": False}

    try:
        parsed = json.loads(span)
    except json.JSONDecodeError as exc:
        return {"valid": False, "errors": ["invalid_json: %s" % exc.msg], "warnings": [],
                "parsed": None, "schema_version": OUTPUT_SCHEMA_VERSION,
                "had_preamble": had_preamble, "had_trailing": had_trailing}

    if not isinstance(parsed, dict):
        return {"valid": False, "errors": ["not_object"], "warnings": [],
                "parsed": parsed, "schema_version": OUTPUT_SCHEMA_VERSION,
                "had_preamble": had_preamble, "had_trailing": had_trailing}

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

    if "reason_codes" in parsed and not _is_list_of_str(parsed["reason_codes"]):
        errors.append("invalid_reason_codes_type")
    if "recommended_actions" in parsed and not _is_list_of_str(parsed["recommended_actions"]):
        errors.append("invalid_recommended_actions_type")

    extra = sorted(set(parsed.keys()) - ALLOWED_FIELDS)
    if extra:
        warnings.append("extra_fields:%s" % ",".join(extra))
    if had_preamble or had_trailing:
        warnings.append("non_json_wrapper")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings,
            "parsed": parsed, "schema_version": OUTPUT_SCHEMA_VERSION,
            "had_preamble": had_preamble, "had_trailing": had_trailing}
