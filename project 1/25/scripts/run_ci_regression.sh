#!/bin/bash
set -euo pipefail

ROOT="/workspace/project 1/25"
TEXTS_PATH="${TTS_TEXTS_PATH:-$ROOT/clients/texts.json}"
VOICES_PATH="${TTS_VOICES_PATH:-}"
OUT_ROOT="${TTS_REGRESSION_OUT:-$ROOT/output/regression}"
STREAM_URL="${TTS_STREAM_URL:-http://127.0.0.1:9000/tts/stream}"
OFFLINE_URL="${TTS_OFFLINE_URL:-http://127.0.0.1:9000/synthesize}"

ARGS=(--texts "$TEXTS_PATH" --out-root "$OUT_ROOT" --stream-url "$STREAM_URL" --offline-url "$OFFLINE_URL")
if [ -n "$VOICES_PATH" ]; then
  ARGS+=(--voices "$VOICES_PATH")
fi

python3 "$ROOT/clients/tts_regression_suite.py" "${ARGS[@]}"


