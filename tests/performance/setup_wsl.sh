#!/usr/bin/env bash
# 在 WSL 中安装 DeepSeek 单体压测所需 Python 环境。
# RTX 5060 (Blackwell, sm_120) 需要 CUDA 12.8 的 torch，即 cu128 轮子。
# 虚拟环境放在 WSL home 下，避免在 /mnt/d 挂载点建 venv 的符号链接/权限问题。
#
# 免 sudo：若系统 Python 缺 ensurepip（未装 python3-venv），自动降级为
#   venv --without-pip + get-pip.py 引导，无需管理员权限。
#
# 用法：bash tests/performance/setup_wsl.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.venvs/edge-bench}"
PYTHON="${PYTHON:-python3}"
CUR_ARCH="cu128"

echo "==> 虚拟环境目录: $VENV_DIR"

if "$PYTHON" -c "import ensurepip" >/dev/null 2>&1; then
  echo "==> 创建虚拟环境（含 pip）"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "==> ensurepip 缺失，使用 venv --without-pip + get-pip.py 引导"
  rm -rf "$VENV_DIR"
  "$PYTHON" -m venv --without-pip "$VENV_DIR"
  BOOT="$VENV_DIR/get-pip.py"
  curl -sS https://bootstrap.pypa.io/get-pip.py -o "$BOOT"
  "$VENV_DIR/bin/python" "$BOOT"
  rm -f "$BOOT"
fi

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

echo "==> 升级 pip"
"$PIP" install --upgrade pip

echo "==> 安装 torch（$CUR_ARCH，适配 RTX 5060）"
"$PIP" install torch --index-url "https://download.pytorch.org/whl/$CUR_ARCH"

echo "==> 安装其余依赖"
"$PIP" install transformers accelerate pyyaml pytest

echo "==> 校验 torch 是否可用 CUDA"
"$PY" - <<'PY'
import torch, transformers
print("torch         :", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version  :", torch.version.cuda if torch.cuda.is_available() else None)
print("device        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("transformers  :", transformers.__version__)
PY

echo ""
echo "完成。接下来："
echo "  source $VENV_DIR/bin/activate"
echo "  python3 tests/performance/generate_test_inputs.py --output var/benchmark/inputs.jsonl"
echo "  python3 -m pytest tests/performance/test_output_validator.py -q"
echo "  python3 tests/performance/benchmark_deepseek.py --config configs/benchmark.deepseek.yaml"
