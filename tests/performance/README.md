# DeepSeek 边缘模型单体压测（第一阶段）

验证方案可行性的独立压测工具，不属于正式业务代码。只测试一条链路：

```text
模拟 PerceptionResult
        → 固定提示词
        → DeepSeek 本地推理
        → 提取 JSON
        → 校验输出字段
        → 记录耗时 / 吞吐 / 显存
```

不依赖感知模块、编排器、HTTP、正式任务队列、代码降级和云端。

## 环境

- 运行环境：**WSL**（模型在 `/home/unic/models/DeepSeek-R1-Distill-Qwen-1.5B`）
- GPU：RTX 5060 Laptop（8 GiB，Blackwell，需 torch cu128）
- 仓库在 Windows：`D:\Projects\edge`，WSL 内路径 `/mnt/d/Projects/edge`

## 安装（首次）

```bash
cd /mnt/d/Projects/edge
bash tests/performance/setup_wsl.sh        # 建 venv + 装 torch(cu128)/transformers
source ~/.venvs/edge-bench/bin/activate
```

## 运行

```bash
# 1. 生成模拟输入（三类各 N 条，固定 seed，输出 JSONL）
python3 tests/performance/generate_test_inputs.py \
    --output var/benchmark/inputs.jsonl --per-category 20 --seed 42

# 2. 输出校验逻辑单测（无 torch 依赖）
python3 -m pytest tests/performance/test_output_validator.py -q

# 3. 跑压测（输入文件不存在时会自动生成）
python3 tests/performance/benchmark_deepseek.py --config configs/benchmark.deepseek.yaml

# 只跑某个场景（快速调试）
python3 tests/performance/benchmark_deepseek.py --scenario warmup
```

## 输出

| 文件 | 内容 |
|---|---|
| `var/benchmark/results/aggregate_<run_id>.json` | 各场景汇总指标 |
| `var/benchmark/results/aggregate_<run_id>.csv` | 同上，CSV |
| `var/benchmark/results/requests_<run_id>.jsonl` | 单请求明细 |
| `var/benchmark/results/env_<run_id>.json` | 环境信息 + 模型加载耗时/显存 |
| `var/benchmark/logs/benchmark_<run_id>.log` | 运行日志 |

单请求记录字段：`request_id / queue_wait_ms / inference_latency_ms / total_latency_ms /
first_token_ms / input_tokens / output_tokens / output_valid / validation_errors /
timed_out / truncated / error_type / category / scenario`。

汇总指标：成功/失败数、JSON 合法率、`req/s`、输出 `tokens/s`、平均/P50/P95/P99 延迟、
首 token 延迟、最大显存、CUDA OOM 数、输入/输出平均 token 数、超时数、实际耗时。

## 模拟输入

模板版本 `bearing-perception-result/1.0`（记录进结果）。字段与《降采样和感知实现流程.md》
的 `PerceptionResult` 一致，三类：

- `normal`：健康（低峭度、电流平衡、质量 good）
- `risk`：预警（峭度/峰值升高、电流不平衡、质量 warning）
- `anomaly`：边界/异常（强冲击变体 + 停机/近零变体，含 `DEVICE_NOT_RUNNING` 标志）

> 注意：压测输入用的是**轴承场景** schema。若计划中的车辆示例（`objects/vehicle`）才是目标，
> 请先改 `generate_test_inputs.py`，并升级 `INPUT_TEMPLATE_VERSION`。

## 关键参数（`configs/benchmark.deepseek.yaml`）

- `generation.max_new_tokens: 64`：限制输出，避免 `<think>` 拉低吞吐。若 JSON 合法率低且为截断导致，调大。
- `request.serialize_inference: true`：原生 Transformers 并发调用同一模型不安全，先加锁串行建立基线；
  之后用完全相同的输入和指标跑 vLLM 时改为 `false` 以启用真实批处理。
- `scenarios.fixed_rate.duration_seconds`：每档速率时长，调试时可缩短到 10。

## 与 vLLM 对比

用相同输入文件、相同场景矩阵和相同指标跑 vLLM，直接对比 `req/s`、P95/P99、显存、合法率。
对比前把 `serialize_inference` 改为 `false`（vLLM 本身支持并发批处理）。
