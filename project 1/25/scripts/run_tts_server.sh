#!/bin/bash
set -euo pipefail

trap 'echo "[run_tts_server] shutting down"; kill -- -$$ >/dev/null 2>&1 || true' EXIT INT TERM

VLLM_OMNI_PATH="/workspace/vllm-omni"
if [ ! -d "$VLLM_OMNI_PATH" ]; then
  VLLM_OMNI_PATH="/workspace/project 1/25/third_party/vllm-omni"
fi
export PYTHONPATH="${VLLM_OMNI_PATH}:${PYTHONPATH:-}"

GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-8000}
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d " ")
  if [ -n "$FREE_MEM" ] && [ "$FREE_MEM" -lt "$GPU_MIN_FREE_MIB" ]; then
    echo "[run_tts_server] GPU free ${FREE_MEM} MiB < ${GPU_MIN_FREE_MIB} MiB; aborting start"
    exit 1
  fi
fi

TTS_MODEL_DIR=${TTS_MODEL_DIR:-/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice}
TTS_STAGE_CONFIG=${TTS_STAGE_CONFIG:-/workspace/project 1/25/artifacts/qwen3_tts_l40s.yaml}
TTS_HOST=${TTS_HOST:-0.0.0.0}
TTS_PORT=${TTS_PORT:-9000}

export TTS_MODEL_DIR TTS_STAGE_CONFIG TTS_HOST TTS_PORT

python3 "/workspace/project 1/25/clients/tts_server.py"
