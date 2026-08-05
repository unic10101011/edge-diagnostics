#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输出校验逻辑测试（无 torch 依赖）。

单独运行，不跟随默认全量测试自动执行：
    python3 -m pytest tests/performance/test_output_validator.py -q
"""

import json

import pytest

from output_validator import (
    ALLOWED_EDGE_RESULT,
    ALLOWED_RISK_LEVEL,
    OUTPUT_SCHEMA_VERSION,
    validate_model_output,
)


def _ok_json(edge_result="normal", risk="low", confidence=0.85, extra=None):
    obj = {
        "edge_result": edge_result,
        "edge_risk_level": risk,
        "confidence": confidence,
        "reason_codes": ["BASELINE_OK"],
    }
    if extra:
        obj.update(extra)
    return json.dumps(obj, ensure_ascii=False)


def test_valid_normal():
    r = validate_model_output(_ok_json())
    assert r["valid"] is True
    assert r["errors"] == []
    assert r["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert r["parsed"]["edge_result"] == "normal"


@pytest.mark.parametrize("value", sorted(ALLOWED_EDGE_RESULT))
def test_all_edge_result_values_valid(value):
    assert validate_model_output(_ok_json(edge_result=value))["valid"] is True


@pytest.mark.parametrize("value", sorted(ALLOWED_RISK_LEVEL))
def test_all_risk_level_values_valid(value):
    assert validate_model_output(_ok_json(risk=value))["valid"] is True


def test_invalid_edge_result():
    r = validate_model_output(_ok_json(edge_result="severe"))
    assert r["valid"] is False
    assert any(e.startswith("invalid_edge_result:") for e in r["errors"])


def test_invalid_risk_level():
    r = validate_model_output(_ok_json(risk="critical"))
    assert r["valid"] is False
    assert any(e.startswith("invalid_edge_risk_level:") for e in r["errors"])


def test_missing_required_fields():
    r = validate_model_output(json.dumps({"edge_result": "warning"}))
    assert r["valid"] is False
    assert any(e == "missing_field:edge_risk_level" for e in r["errors"])
    assert any(e == "missing_field:confidence" for e in r["errors"])


def test_confidence_null_is_allowed():
    r = validate_model_output(_ok_json(confidence=None))
    assert r["valid"] is True


def test_confidence_out_of_range():
    r = validate_model_output(_ok_json(confidence=1.5))
    assert r["valid"] is False
    assert any(e.startswith("invalid_confidence_range") for e in r["errors"])


def test_confidence_negative():
    r = validate_model_output(_ok_json(confidence=-0.1))
    assert r["valid"] is False


def test_bad_json():
    r = validate_model_output('{"edge_result": "warning", broken')
    assert r["valid"] is False
    assert any(e.startswith("invalid_json") for e in r["errors"])
    assert r["parsed"] is None


def test_no_json_object():
    r = validate_model_output("I cannot answer this.")
    assert r["valid"] is False
    assert "no_json_object" in r["errors"]


def test_empty_output():
    r = validate_model_output("   ")
    assert r["valid"] is False
    assert "empty_output" in r["errors"]


def test_think_preamble_is_wrapped_not_fatal():
    # R1 风格：先输出 <think> 再给 JSON。JSON 本身合法但带包装，需记录影响吞吐
    text = "<think>Let me analyze the features...</think>\n" + _ok_json(edge_result="warning")
    r = validate_model_output(text)
    assert r["valid"] is True
    assert r["had_preamble"] is True
    assert r["errors"] == []
    assert "non_json_wrapper" in r["warnings"]


def test_markdown_code_fence():
    text = "```json\n" + _ok_json() + "\n```"
    r = validate_model_output(text)
    assert r["valid"] is True


def test_extra_fields_flagged_as_warning():
    r = validate_model_output(_ok_json(extra={"unexpected_field": 1}))
    assert r["valid"] is True
    assert any(w.startswith("extra_fields:") for w in r["warnings"])


def test_wrong_reason_codes_type():
    r = validate_model_output(_ok_json(extra={"reason_codes": "not_a_list"}))
    assert r["valid"] is False
    assert "invalid_reason_codes_type" in r["errors"]


def test_confidence_bool_rejected():
    # bool 是 int 子类，应被当作非法类型
    r = validate_model_output(_ok_json(confidence=True))
    assert r["valid"] is False
    assert "invalid_confidence_type" in r["errors"]
