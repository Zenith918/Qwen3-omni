#!/bin/bash
# D15 P0-5: Freeze GT-based USER_KPI baseline
# Run 5x mini stability + 1x full 16 with optimal turn_taking config
# Then compute FAIL threshold = baseline GT_TT_P95 + 50ms

set -e
cd "/workspace/project 1/25"

CASES_MINI="tools/autortc/cases/mini_cases.json"
CASES_FULL="tools/autortc/cases/all_cases.json"
TOKEN_API="http://127.0.0.1:9090/api/token"
BASELINE_DIR="golden/d15_userkpi_gt_baseline"
OUTPUT_ROOT="/tmp/d15_baseline"

# Load optimal config if available
if [ -f "/tmp/d15_grid/optimal_env.sh" ]; then
    source /tmp/d15_grid/optimal_env.sh
    echo "[Baseline] Using grid-optimal config"
fi

export MODE="${MODE:-turn_taking}"
echo "[Baseline] MODE=$MODE"

mkdir -p "$OUTPUT_ROOT" "$BASELINE_DIR"

echo ""
echo "========================================="
echo "Phase 1: 5x mini stability runs"
echo "========================================="
for i in 1 2 3 4 5; do
    echo ""
    echo "--- Mini run $i/5 ---"
    python3 -u tools/autobrowser/run_suite.py \
        --cases_json "$CASES_MINI" \
        --token_api "$TOKEN_API" \
        --record_s 25 \
        --p0_only 1 \
        --output_root "${OUTPUT_ROOT}/mini_${i}" \
        --inter_case_wait_s 15 \
        2>&1 | tee "${OUTPUT_ROOT}/mini_${i}.log"

    # Copy summary to baseline
    LATEST=$(ls -td "${OUTPUT_ROOT}/mini_${i}/"*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "${LATEST}summary.json" ]; then
        cp "${LATEST}summary.json" "${BASELINE_DIR}/mini_run_${i}_summary.json"
    fi
    sleep 10
done

echo ""
echo "========================================="
echo "Phase 2: 1x full 16-case run"
echo "========================================="
python3 -u tools/autobrowser/run_suite.py \
    --cases_json "$CASES_FULL" \
    --token_api "$TOKEN_API" \
    --record_s 25 \
    --output_root "${OUTPUT_ROOT}/full16" \
    --inter_case_wait_s 15 \
    2>&1 | tee "${OUTPUT_ROOT}/full16.log"

LATEST_FULL=$(ls -td "${OUTPUT_ROOT}/full16/"*/ 2>/dev/null | head -1)
if [ -n "$LATEST_FULL" ] && [ -f "${LATEST_FULL}summary.json" ]; then
    cp "${LATEST_FULL}summary.json" "${BASELINE_DIR}/full16_summary.json"
fi

echo ""
echo "========================================="
echo "Phase 3: Compute baseline stats"
echo "========================================="
python3 -c "
import json, glob, os
import numpy as np

baseline_dir = '${BASELINE_DIR}'
mini_files = sorted(glob.glob(os.path.join(baseline_dir, 'mini_run_*_summary.json')))
print(f'Found {len(mini_files)} mini summaries')

gt_tt_p95_vals = []
gt_to_counts = []
gt_tt_counts = []

for f in mini_files:
    s = json.load(open(f))
    v = s.get('gt_tt_p95_ms')
    if v is not None:
        gt_tt_p95_vals.append(v)
    gt_to_counts.append(s.get('talk_over_gt_count', 0))
    gt_tt_counts.append(s.get('gt_tt_count', 0))
    print(f'  {os.path.basename(f)}: GT_TT_P95={v}, TO_GT={s.get(\"talk_over_gt_count\", 0)}')

if gt_tt_p95_vals:
    arr = np.array(gt_tt_p95_vals)
    baseline_p95 = float(np.percentile(arr, 95))
    fail_threshold = baseline_p95 + 50
    stats = {
        'gt_tt_p95_values': gt_tt_p95_vals,
        'gt_tt_p95_mean': float(np.mean(arr)),
        'gt_tt_p95_std': float(np.std(arr)),
        'gt_tt_p95_p95': baseline_p95,
        'fail_threshold_ms': fail_threshold,
        'total_gt_to': sum(gt_to_counts),
        'total_gt_tt_cases': sum(gt_tt_counts),
        'mini_runs': len(mini_files),
    }
    out_path = os.path.join(baseline_dir, 'baseline_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\\nBaseline GT_TT_P95 (P95 of runs): {baseline_p95:.0f}ms')
    print(f'FAIL threshold: {fail_threshold:.0f}ms')
    print(f'Total talk-over (GT): {sum(gt_to_counts)}')
    print(f'Saved: {out_path}')
else:
    print('ERROR: No GT TT P95 data collected')
"

echo ""
echo "Baseline frozen to: $BASELINE_DIR"
echo "Done."
