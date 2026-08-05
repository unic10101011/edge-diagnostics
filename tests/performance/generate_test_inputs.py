#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成模拟感知结果 PerceptionResult（轴承场景，与感知接口文档一致的最小字段集）。

三类输入：normal（正常）、risk（风险/预警）、anomaly（边界/异常）。
使用固定 seed，保证每次生成内容一致、结果可比。
输出 JSONL：每行一个 PerceptionResult 对象。

用法：
    python3 generate_test_inputs.py --output var/benchmark/inputs.jsonl \
        --per-category 20 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

# 输入模板版本：压测结果必须记录此值，后续改动字段需升版本
INPUT_TEMPLATE_VERSION = "bearing-perception-result/1.0"

# 与《降采样和感知实现流程.md》《边缘部分需要的接口.md》一致的固定通道元信息
_VIB_META = {"source_sample_rate_hz": 64000, "analysis_sample_rate_hz": 16000, "unit": "mm/s"}
_CUR_META = {"source_sample_rate_hz": 64000, "analysis_sample_rate_hz": 16000, "unit": "A"}


def _vib(rms, peak, kurtosis, dom_freq, band, entropy):
    return {
        **_VIB_META,
        "rms": round(rms, 4),
        "absolute_peak": round(peak, 4),
        "kurtosis": round(kurtosis, 4),
        "dominant_frequency_hz": round(dom_freq, 1),
        "band_power_ratio_500_2000": round(band, 4),
        "spectral_entropy": round(entropy, 4),
    }


def _cur(rms_a, peak_a):
    return {**_CUR_META, "rms_a": round(rms_a, 4), "absolute_peak_a": round(peak_a, 4)}


def _ctx(speed_mean, speed_sd, torque_mean, torque_sd, load_mean, load_sd, temp):
    return {
        "shaft_speed_rpm": {
            "mean": round(speed_mean, 2),
            "last": round(speed_mean + random.uniform(-0.5, 0.5), 2),
            "minimum": round(speed_mean - speed_sd, 2),
            "maximum": round(speed_mean + speed_sd, 2),
            "standard_deviation": round(speed_sd, 3),
        },
        "load_torque_nm": {
            "mean": round(torque_mean, 3),
            "last": round(torque_mean + random.uniform(-0.02, 0.02), 3),
            "minimum": round(torque_mean - torque_sd, 3),
            "maximum": round(torque_mean + torque_sd, 3),
            "standard_deviation": round(torque_sd, 4),
        },
        "bearing_radial_load_n": {
            "mean": round(load_mean, 1),
            "last": round(load_mean + random.uniform(-1, 1), 1),
            "minimum": round(load_mean - load_sd, 1),
            "maximum": round(load_mean + load_sd, 1),
            "standard_deviation": round(load_sd, 3),
        },
        "bearing_module_temperature_c": round(temp, 1),
    }


def _base(idx, status, flags):
    ts = 1784784400000000000 + idx * 50000000
    return {
        "task_id": "benchmark-000001",
        "packet_id": "benchmark_pkt_%06d" % idx,
        "sender_id": "sender_benchmark_01",
        "sequence_number": (idx % 80) + 1,
        "end_generate_timestamp_ns": ts,
        "feature_generated_at_ns": ts + 10000000,
        "perception_quality": {"status": status, "flags": flags},
    }


def make_normal(idx):
    """健康：低峭度、低峰值、电流平衡、转速稳定、质量 good。"""
    p = _base(idx, "good", [])
    p["features"] = {
        "vibration": _vib(
            rms=random.uniform(0.25, 0.45),
            peak=random.uniform(1.4, 2.0),
            kurtosis=random.uniform(2.8, 3.4),
            dom_freq=random.choice([120.0, 240.0, 360.0]),
            band=random.uniform(0.2, 0.35),
            entropy=random.uniform(0.55, 0.7),
        ),
        "phase_current_1": _cur(rms_a=random.uniform(2.3, 2.5), peak_a=random.uniform(3.3, 3.6)),
        "phase_current_2": _cur(rms_a=random.uniform(2.3, 2.5), peak_a=random.uniform(3.3, 3.6)),
        "current_relationship": {"current_imbalance_ratio": round(random.uniform(0.01, 0.05), 4)},
        "operating_context": _ctx(
            speed_mean=900.0, speed_sd=random.uniform(0.2, 0.5),
            torque_mean=0.70, torque_sd=0.01,
            load_mean=1000.0, load_sd=1.2,
            temp=random.uniform(44.0, 48.0),
        ),
    }
    return p


def make_risk(idx):
    """风险/预警：峭度和峰值升高、500-2000Hz 频带能量占比偏高、电流不平衡、质量 warning。"""
    p = _base(idx, "warning", [])
    p["features"] = {
        "vibration": _vib(
            rms=random.uniform(0.7, 1.2),
            peak=random.uniform(3.0, 4.5),
            kurtosis=random.uniform(5.0, 7.0),
            dom_freq=random.choice([240.0, 480.0, 720.0]),
            band=random.uniform(0.5, 0.7),
            entropy=random.uniform(0.4, 0.55),
        ),
        "phase_current_1": _cur(rms_a=random.uniform(2.5, 3.0), peak_a=random.uniform(3.8, 4.6)),
        "phase_current_2": _cur(rms_a=random.uniform(2.0, 2.3), peak_a=random.uniform(3.0, 3.5)),
        "current_relationship": {"current_imbalance_ratio": round(random.uniform(0.10, 0.18), 4)},
        "operating_context": _ctx(
            speed_mean=random.uniform(880.0, 920.0), speed_sd=random.uniform(1.5, 3.0),
            torque_mean=0.75, torque_sd=0.03,
            load_mean=1050.0, load_sd=3.0,
            temp=random.uniform(50.0, 56.0),
        ),
    }
    return p


def make_anomaly(idx):
    """边界/异常：两种变体。

    - 强冲击：峭度极高、峰值极高、电流严重不平衡（fault 特征）。
    - 停机/近零：转速低于运行阈值，DEVICE_NOT_RUNNING 标志，振动接近零。
    """
    variant = idx % 2
    if variant == 0:
        p = _base(idx, "warning", [])
        p["features"] = {
            "vibration": _vib(
                rms=random.uniform(2.0, 3.5),
                peak=random.uniform(8.0, 14.0),
                kurtosis=random.uniform(10.0, 14.0),
                dom_freq=random.choice([720.0, 1200.0, 2400.0]),
                band=random.uniform(0.75, 0.9),
                entropy=random.uniform(0.25, 0.4),
            ),
            "phase_current_1": _cur(rms_a=random.uniform(3.2, 4.0), peak_a=random.uniform(5.5, 7.0)),
            "phase_current_2": _cur(rms_a=random.uniform(1.2, 1.8), peak_a=random.uniform(2.0, 2.8)),
            "current_relationship": {"current_imbalance_ratio": round(random.uniform(0.30, 0.55), 4)},
            "operating_context": _ctx(
                speed_mean=random.uniform(820.0, 880.0), speed_sd=random.uniform(8.0, 15.0),
                torque_mean=0.85, torque_sd=0.08,
                load_mean=1100.0, load_sd=8.0,
                temp=random.uniform(62.0, 70.0),
            ),
        }
    else:
        # 停机/近零：运行状态判断为未运行，振动接近零
        p = _base(idx, "warning", ["DEVICE_NOT_RUNNING"])
        p["features"] = {
            "vibration": _vib(
                rms=random.uniform(0.001, 0.01),
                peak=random.uniform(0.005, 0.03),
                kurtosis=3.0,
                dom_freq=20.0,
                band=0.05,
                entropy=0.9,
            ),
            "phase_current_1": _cur(rms_a=random.uniform(0.05, 0.15), peak_a=random.uniform(0.1, 0.3)),
            "phase_current_2": _cur(rms_a=random.uniform(0.05, 0.15), peak_a=random.uniform(0.1, 0.3)),
            "current_relationship": {"current_imbalance_ratio": 0.0},
            "operating_context": _ctx(
                speed_mean=random.uniform(5.0, 40.0), speed_sd=random.uniform(0.5, 2.0),
                torque_mean=0.0, torque_sd=0.0,
                load_mean=0.0, load_sd=0.0,
                temp=random.uniform(30.0, 40.0),
            ),
        }
    return p


_GENERATORS = {
    "normal": make_normal,
    "risk": make_risk,
    "anomaly": make_anomaly,
}


def generate_inputs(per_category=20, seed=42):
    """返回按类别划分的输入列表：{category: [PerceptionResult, ...]}。"""
    rng = random.Random(seed)
    # 让随机序列可复现（各生成函数内部使用全局 random）
    random.seed(seed)
    out = {}
    idx = 0
    for category, gen in _GENERATORS.items():
        items = []
        for _ in range(per_category):
            items.append(gen(idx))
            idx += 1
        out[category] = items
    return out


def write_jsonl(path, per_category=20, seed=42):
    data = generate_inputs(per_category=per_category, seed=seed)
    rows = []
    for category, items in data.items():
        for item in items:
            item["_category"] = category  # 记录类别，便于结果按类统计
            rows.append(item)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), INPUT_TEMPLATE_VERSION


def ensure_inputs_file(path, per_category=20, seed=42):
    """压测入口调用：输入文件不存在时自动生成。"""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return False, INPUT_TEMPLATE_VERSION
    count, ver = write_jsonl(path, per_category=per_category, seed=seed)
    return True, ver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成轴承感知结果模拟输入 JSONL")
    parser.add_argument("--output", default="var/benchmark/inputs.jsonl")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, ver = write_jsonl(path, per_category=args.per_category, seed=args.seed)
    print(f"template_version={ver}")
    print(f"wrote {count} inputs -> {path}")
