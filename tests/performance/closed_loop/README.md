# 闭环验证：窗口聚合＋有界队列＋超时＋降级（第一阶段）

独立验证工具，不属于正式业务代码。验证「20 Hz 感知 → 1s 窗口 → ≤1 模型调用/s/发送方」
这条第一阶段闭环的**行为**（窗口化 / 队列 / 超时 / 降级），并产出冻结参数所需的数据。

组件逻辑写清楚、可复用；参数冻结后再平移到 `src/` 正式实现。

## 链路

```text
20 Hz PerceptionResult
        ↓ 每发送方 1s 双缓冲窗口
WindowAggregate（sample_count/missing_ratio/quality/late_dropped）
        ↓ 有界队列（capacity=1：1条推理+最多1条等待）
单推理 Worker（排队/推理/总处理三层超时 + 熔断）
        ↓
LocalModelRunner ──成功──→ 统一 EdgeResult
        ↓ 超时/异常/输出非法/队列满/熔断
CodeFallbackRunner ──→ 统一 EdgeResult（edge_rule_* 版本标记）
```

## 组件

| 模块 | 职责 |
|---|---|
| `window_aggregator.py` | 每发送方双缓冲窗口，整数纳秒切窗，空窗/稀疏窗标记 |
| `bounded_queue.py` | 有界队列 + 满队列策略（drop / replace_oldest）+ 推理 worker + 熔断 |
| `pipeline.py` | 串起聚合器→队列→worker→降级，产出 RunRecord |
| `code_fallback.py` | 确定性规则降级（版本化，不编造测量值） |
| `model_adapter.py` | InferenceAdapter + MockModel（无 torch，故障注入） |
| `real_model.py` | 真实 Transformers 模型（惰性导入 torch，仅 WSL/GPU） |
| `jitter_source.py` | 20Hz 抖动输入调度（jitter/丢包/突发/空窗） |
| `run_validation.py` | T2 稳定性 / T4 过载 运行器（WSL 真实模型） |

## 测试

无 torch，Windows 直接跑：

```bash
python -m pytest tests/performance/closed_loop/ -v
```

- `test_window_aggregator.py`（T1 窗口抖动）：边界不重不漏、sample_count/missing_ratio
  正确、稀疏标记质量、空窗口不调模型、迟到/乱序丢弃并计数、推理不阻塞感知接收。
- `test_closed_loop.py`（T3 故障注入）：5 种故障路径各自进入降级并返回合规 EdgeResult；
  队列满两种策略；熔断开启→探测→恢复；两条路线都失败。

## 长跑场景（T2/T4，WSL 真实模型）

```bash
source ~/.venvs/edge-bench/bin/activate
# 单发送方 30 分钟稳定性
python tests/performance/closed_loop/run_validation.py \
    --config configs/closed_loop.validation.yaml --adapter real --scenario t2
# 双发送方过载（10 分钟）
python tests/performance/closed_loop/run_validation.py \
    --config configs/closed_loop.validation.yaml --adapter real --scenario t4
```

Windows 上可用 `--adapter mock --duration 30` 冒烟（不需要模型）。

## 输出

`var/closed_loop/results/<run_id>/`：
- `events.jsonl`：逐窗口事件（execution_mode / fallback_reason / 延迟 / 质量 / 是否迟到）
- `aggregate.json` / `aggregate.csv`：P50/P95/P99、降级率、最大队列、超时数、显存等
- `env.json`：环境与配置

## 待冻结参数

窗口长度、迟到容忍、稀疏阈值、队列容量、满队列策略、排队/推理/总超时、
熔断阈值、恢复探测周期——全部标 `# TEST:` 在
`configs/closed_loop.validation.yaml`，由 T2/T4 长跑数据决定后冻结并写入执行计划文档。
