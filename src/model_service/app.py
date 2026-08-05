# -*- coding: utf-8 -*-
"""WSL 模型服务 HTTP 入口（stdlib http.server，零额外依赖）。

端点：
    GET  /health     存活检查（进程活着即 ok）
    GET  /readiness  就绪检查（模型加载 + warmup 完成）
    POST /infer      窗口聚合输入 → 模型诊断结果（或错误）

模型加载 + warmup 在启动时完成；infer 由 ThreadingHTTPServer 多线程处理，
generate() 由 ModelRunner 内部推理锁串行化。

用法（WSL，edge-bench venv）：
    python -m src.model_service.app \
        --model /home/unic/models/Qwen2.5-1.5B-Instruct --port 8001
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model_service.model_runner import ModelRunner  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    runner: Optional[ModelRunner] = None  # 由 make_server 注入

    def log_message(self, fmt, *args):  # 简化日志
        sys.stderr.write("[model_service] %s\n" % (fmt % args))

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path.rstrip("/") == "/readiness":
            runner = self.runner
            ready = bool(runner and runner.ready)
            self._send_json(200, {"ready": ready,
                                  "load_error": runner.load_error if runner else None})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._send_json(404, {"error": "not_found"})
            return
        if self.runner is None or not self.runner.ready:
            self._send_json(503, {"valid": False, "error": "model_not_ready"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model_input = payload.get("input")
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"valid": False, "error": "bad_request: %s" % exc})
            return
        if not isinstance(model_input, dict):
            self._send_json(400, {"valid": False, "error": "bad_request: input 必须是对象"})
            return
        result = self.runner.infer(model_input)
        status = 200 if result.get("valid") else 502
        self._send_json(status, result)


def make_server(host: str, port: int, runner: ModelRunner) -> ThreadingHTTPServer:
    _Handler.runner = runner
    return ThreadingHTTPServer((host, port), _Handler)


def main():
    ap = argparse.ArgumentParser(description="WSL 模型服务")
    ap.add_argument("--model", default="/home/unic/models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    print("加载模型:", args.model)
    runner = ModelRunner(model_path=args.model, dtype=args.dtype,
                         max_new_tokens=args.max_new_tokens)
    server = make_server(args.host, args.port, runner)
    print("模型服务就绪:", "http://%s:%d" % (args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
