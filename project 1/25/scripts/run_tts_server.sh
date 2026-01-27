#!/bin/bash
set -euo pipefail

trap 'echo "[run_tts_server] shutting down"; kill -- -$$ >/dev/null 2>&1 || true' EXIT INT TERM

export PYTHONPATH="/workspace/project 1/25/third_party/vllm-omni:${PYTHONPATH:-}"

TTS_MODEL_DIR=${TTS_MODEL_DIR:-/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice}
TTS_STAGE_CONFIG=${TTS_STAGE_CONFIG:-/workspace/project 1/25/artifacts/qwen3_tts_l40s.yaml}
TTS_HOST=${TTS_HOST:-0.0.0.0}
TTS_PORT=${TTS_PORT:-9000}

export TTS_MODEL_DIR TTS_STAGE_CONFIG TTS_HOST TTS_PORT

python3 "/workspace/project 1/25/clients/tts_server.py"
