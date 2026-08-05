# -*- coding: utf-8 -*-
"""复用 tests/performance/output_validator.py 的 sys.path 引导。

与现有 benchmark 脚本的约定一致：tests/performance 目录本身不是包，模块靠
「脚本所在目录在 sys.path 上」互相导入。这里显式把 tests/performance 加进
sys.path 再导入 validate_model_output，保证 closed_loop 作为包也能复用。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PERF_DIR = Path(__file__).resolve().parents[1]
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

from output_validator import (  # noqa: E402
    OUTPUT_SCHEMA_VERSION,
    validate_model_output,
)

__all__ = ["OUTPUT_SCHEMA_VERSION", "validate_model_output"]
