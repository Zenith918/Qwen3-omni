#!/bin/bash
set -euo pipefail

LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:8000}
TTS_URL=${TTS_URL:-http://127.0.0.1:9000/tts/stream}

export LLM_BASE_URL TTS_URL
python3 "/workspace/project 1/25/clients/bridge_demo.py"
