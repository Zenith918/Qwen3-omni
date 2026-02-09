#!/bin/bash
set -euo pipefail

ROOT="/workspace/project 1/25"
MODE="${REGRESSION_MODE:-fast}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-fast}"
      shift 2
      ;;
    *)
      echo "[run_ci_regression] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# 黄金基线路径 (20260208_200725, 0.6B + CP/Decoder CUDA Graph, 全 PASS)
GOLDEN_BASELINE="${TTS_REGRESSION_BASELINE:-$ROOT/output/regression/20260208_200725/summary.json}"
export TTS_REGRESSION_BASELINE="$GOLDEN_BASELINE"
export TTS_GATE_SNR_BASELINE_DB="${TTS_GATE_SNR_BASELINE_DB:-15}"

if [ "$MODE" = "fast" ]; then
  export REGRESSION_MODE=fast
  export RUN_DETERMINISM="${RUN_DETERMINISM:-1}"
  export SAVE_WAV="${SAVE_WAV:-0}"
  export TTS_DETERMINISM_RUNS="${TTS_DETERMINISM_RUNS:-3}"
  export TTS_DETERMINISM_TEXTS="${TTS_DETERMINISM_TEXTS:-short_01}"
elif [ "$MODE" = "full" ]; then
  export REGRESSION_MODE=full
  export RUN_DETERMINISM="${RUN_DETERMINISM:-1}"
  export SAVE_WAV="${SAVE_WAV:-1}"
  export TTS_DETERMINISM_RUNS="${TTS_DETERMINISM_RUNS:-10}"
  export TTS_DETERMINISM_TEXTS="${TTS_DETERMINISM_TEXTS:-long_03,short_01}"
else
  echo "[run_ci_regression] invalid mode: $MODE" >&2
  exit 1
fi

TEXTS_PATH="${TTS_TEXTS_PATH:-$ROOT/clients/texts_p0_base.json}"
VOICES_PATH="${TTS_VOICES_PATH:-$ROOT/clients/voices_base.json}"
OUT_ROOT="${TTS_REGRESSION_OUT:-$ROOT/output/regression}"
STREAM_URL="${TTS_STREAM_URL:-http://127.0.0.1:9000/tts/stream}"
OFFLINE_URL="${TTS_OFFLINE_URL:-http://127.0.0.1:9000/synthesize}"

ARGS=(--texts "$TEXTS_PATH" --out-root "$OUT_ROOT" --stream-url "$STREAM_URL" --offline-url "$OFFLINE_URL")
if [ -n "$VOICES_PATH" ]; then
  ARGS+=(--voices "$VOICES_PATH")
fi

echo "[run_ci_regression] mode=$MODE baseline=$GOLDEN_BASELINE"
python3 "$ROOT/clients/tts_regression_suite.py" "${ARGS[@]}"
