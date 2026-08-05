# -*- coding: utf-8 -*-
"""让 tests/unit 能以顶层包形式 import src/edge_model、src/model_service。"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
