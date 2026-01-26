#!/bin/bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-Omni-AWQ-4bit}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen3-omni-thinker \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8
