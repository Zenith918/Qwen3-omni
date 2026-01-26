#!/bin/bash
set -euo pipefail

echo "[LLM-SETUP] Python: $(python3 -V)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

set +e
python3 - <<'PY'
try:
    import vllm
    print(f"[LLM-SETUP] vLLM version: {vllm.__version__}")
    raise SystemExit(0)
except Exception:
    print("[LLM-SETUP] vLLM not found, installing...")
    raise SystemExit(1)
PY
NEED_INSTALL=$?
set -e

if [ "$NEED_INSTALL" -ne 0 ]; then
  pip install -U vllm
fi

python3 - <<'PY'
import vllm
print(f"[LLM-SETUP] vLLM ready: {vllm.__version__}")
PY
