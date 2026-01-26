#!/bin/bash
set -euo pipefail

export PYTHONPATH="/workspace/project 1/25/third_party/vllm-omni:${PYTHONPATH:-}"

pip install -U fastapi uvicorn soundfile librosa onnxruntime sox "huggingface_hub==0.36.0"

TTS_MODEL_ID=${TTS_MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
TTS_MODEL_DIR=${TTS_MODEL_DIR:-/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}
DOWNLOAD_TTS=${DOWNLOAD_TTS:-1}

if [ "$DOWNLOAD_TTS" = "1" ]; then
  python3 - <<PY
from huggingface_hub import snapshot_download
print("[TTS-SETUP] downloading", "${TTS_MODEL_ID}")
snapshot_download(repo_id="${TTS_MODEL_ID}", local_dir="${TTS_MODEL_DIR}", local_dir_use_symlinks=False)
print("[TTS-SETUP] done")
PY
fi

python3 - <<'PY'
import vllm_omni
print("[TTS-SETUP] vllm_omni import OK")
PY
