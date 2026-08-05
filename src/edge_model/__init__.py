# -*- coding: utf-8 -*-
"""边缘模型运行模块（Windows 边缘主程序侧）。

只依赖 Python 标准库（不依赖 torch/transformers/requests），通过 HTTP 调用
WSL 侧的 src/model_service。即使模型服务完全不可用，本侧仍能走代码规则降级
完成闭环。

职责：
- 感知结果窗口聚合（1s，整数纳秒切窗）；
- 有界模型队列 + 三层超时 + 熔断；
- HTTP 调用 WSL 模型服务；
- 代码规则降级（当前为 edge_rule_test_v1，待业务规则确认）；
- 把窗口级诊断结果映射给窗口内所有数据包。
"""
