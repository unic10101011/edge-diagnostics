# -*- coding: utf-8 -*-
"""WSL 模型服务侧（src/model_service）。

只负责：加载 Qwen、warmup、串行调用 generate()、解析和校验模型输出、返回结果。
不负责窗口聚合 / 有界队列 / 超时 / 熔断 / 代码降级——那些在 Windows 侧
src/edge_model。即使本服务完全不可用，Windows 侧仍能走代码规则完成闭环。
"""
