#!/bin/bash
# D14 P0-3: Silence matrix experiment
# Tests TURN_TAKING_MIN_SILENCE_MS values: 200 400 600 900 1200
# For each value: restart agent, run 4 cases, collect results

set -e
cd "/workspace/project 1/25"

CASES_JSON="tools/autortc/cases/d14_matrix_cases.json"
TOKEN_API="http://127.0.0.1:9090/api/token"
OUTPUT_ROOT="/tmp/d14_matrix"
SILENCE_VALUES="200 400 600 900 1200"

mkdir -p "$OUTPUT_ROOT"

for SILENCE_MS in $SILENCE_VALUES; do
    echo ""
    echo "========================================"
    echo "TURN_TAKING_MIN_SILENCE_MS = ${SILENCE_MS}ms"
    echo "========================================"

    # Stop supervisor-managed agent
    supervisorctl stop voice-agent:livekit-agent 2>/dev/null || true
    sleep 2

    # Start agent with new silence value
    export TURN_TAKING_MIN_SILENCE_MS=$SILENCE_MS
    export LIVEKIT_URL="wss://renshenghehuoren-mpdsjfwe.livekit.cloud"
    export LIVEKIT_API_KEY="API7fj35wGLumtc"
    export LIVEKIT_API_SECRET="WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B"
    export PYTHONPATH="/workspace/vllm-omni"

    python3 runtime/livekit_agent.py start &
    AGENT_PID=$!
    echo "Agent PID=$AGENT_PID with silence=${SILENCE_MS}ms"
    sleep 8

    # Run 4 cases
    python3 -u tools/autobrowser/run_suite.py \
        --cases_json "$CASES_JSON" \
        --token_api "$TOKEN_API" \
        --record_s 25 \
        --p0_only 1 \
        --output_root "${OUTPUT_ROOT}/silence_${SILENCE_MS}" \
        --inter_case_wait_s 10 \
        2>&1 | tee "${OUTPUT_ROOT}/silence_${SILENCE_MS}.log"

    # Kill manual agent
    kill $AGENT_PID 2>/dev/null || true
    wait $AGENT_PID 2>/dev/null || true
    sleep 3
done

# Restart supervisor agent with default
echo ""
echo "Restoring supervisor agent..."
supervisorctl start voice-agent:livekit-agent 2>/dev/null || true
sleep 5

# Collect results
echo ""
echo "========================================"
echo "MATRIX RESULTS"
echo "========================================"
python3 -c "
import json, glob
print(f'{'Silence_MS':>12} {'TT_P95':>8} {'TT_Count':>8} {'TO_Rate':>8} {'TO_Count':>8}')
print('-' * 60)
for ms in [200, 400, 600, 900, 1200]:
    fs = glob.glob(f'/tmp/d14_matrix/silence_{ms}/*/summary.json')
    if not fs:
        print(f'{ms:>12} (no data)')
        continue
    s = json.load(open(fs[0]))
    tt_p95 = s.get('tt_p95_ms', 'N/A')
    tt_count = s.get('tt_count', 0)
    to_rate = s.get('talk_over_rate', 0)
    to_count = s.get('talk_over_count', 0)
    tt_str = f'{tt_p95:.0f}' if isinstance(tt_p95, (int, float)) else tt_p95
    print(f'{ms:>12} {tt_str:>8} {tt_count:>8} {to_rate:>8.1%} {to_count:>8}')
"
