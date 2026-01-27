#!/bin/bash
set -euo pipefail

trap 'echo "[run_llm_server] shutting down"; kill -- -$$ >/dev/null 2>&1 || true' EXIT INT TERM

MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-Omni-AWQ-4bit}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen3-omni-thinker \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 2048 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8
