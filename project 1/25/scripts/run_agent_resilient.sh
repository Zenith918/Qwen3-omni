#!/bin/bash
# D9: Resilient Agent runner — auto-restarts on crash, logs to /tmp/agent_d9.log
# Usage: bash scripts/run_agent_resilient.sh &

cd "/workspace/project 1/25"

export LIVEKIT_URL="${LIVEKIT_URL:-wss://renshenghehuoren-mpdsjfwe.livekit.cloud}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-API7fj35wGLumtc}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B}"
export PYTHONPATH="${PYTHONPATH:-/workspace/vllm-omni}"
export VAD_SILENCE_MS="${VAD_SILENCE_MS:-200}"
export TTS_FRAME_MS="${TTS_FRAME_MS:-20}"
export MIN_ENDPOINTING="${MIN_ENDPOINTING:-0.3}"
export ENABLE_CONTINUATION="${ENABLE_CONTINUATION:-1}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-150}"
export LLM_HISTORY_TURNS="${LLM_HISTORY_TURNS:-10}"
export CAPTURE_PRE_RTC="${CAPTURE_PRE_RTC:-1}"

LOG="/tmp/agent_d9.log"
MAX_RESTARTS=10
restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "[$(date)] Starting Agent (attempt $((restart_count+1))/$MAX_RESTARTS)" >> "$LOG"
    python3 runtime/livekit_agent.py start >> "$LOG" 2>&1
    exit_code=$?
    echo "[$(date)] Agent exited with code $exit_code" >> "$LOG"

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Agent exited cleanly, stopping" >> "$LOG"
        break
    fi

    restart_count=$((restart_count + 1))
    echo "[$(date)] Restarting in 5s..." >> "$LOG"
    sleep 5
done

echo "[$(date)] Agent wrapper exiting (restarts=$restart_count)" >> "$LOG"

