# D13 Cloud Agent Task: WYSIWYG upgrade to production-consistent

## Role

You are an AI engineer taking over a real-time voice call project. The system uses Qwen3-Omni + Qwen3-TTS + LiveKit for browser-based real-time voice, with AutoRTC + AutoBrowser regression testing.

## Environment (auto-configured, no manual Secrets needed)

SSH and LiveKit credentials are injected via `.cursor/setup_ssh.sh` at VM boot:
- `ssh gpu` connects directly to GPU server (RunPod L40S)
- LiveKit env vars are globally exported (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`)

## Required Reading (priority order)

1. **`SKILL.md`** - Engineering SOP, all technical decisions, lessons learned, Gates
2. **`DEV_LOG.md`** - Detailed dev log and acceptance results per stage
3. **`tools/autobrowser/run_suite.py`** - D12 AutoBrowser orchestrator (main file to modify)
4. **`runtime/webrtc_test.html`** - Product webpage (browser-side instrumentation, main file to modify)
5. **`tools/autortc/audio_metrics.py`** - Metrics analysis + Gates

## Current State (D12 done, D13 code partially modified but UNVERIFIED)

D12 results:
- AutoBrowser 16/16 cases PASS, USER_KPI P50=201ms P95=207ms
- But USER_KPI values are "too neat" due to 30ms polling + mic mute + resetForMeasurement cheating

D13 code changes already made but NOT verified:
- `run_suite.py`: `_prepare_chromium_wav()` replaces `_convert_wav_for_chromium()`, appends 10s silence
- `run_suite.py`: Removed mic mute + resetForMeasurement calls
- `webrtc_test.html`: `finalizeTrace()` now keeps raw + clamped + is_talk_over
- `webrtc_test.html`: Polling precision changed to PLAYOUT_POLL_MS=5ms, MIC_POLL_MS=10ms
- `run_suite.py`: summary now outputs raw/clamped/talk_over stats

## D13 Goals

### P0-1: Fix USER_KPI definition (code done, needs verification)

- `user_kpi_raw_ms` = raw value (can be negative = talk-over)
- `user_kpi_ms` = max(0, raw) (clamped, for turn-taking gate)
- `is_talk_over` = raw < 0
- report.md adds: `talk_over_count`, `talk_over_ms_p95`
- Report header shows Turn-taking KPI and Duplex KPI tables

**Acceptance**: Run mini 4 cases, confirm browser_trace.json has raw/clamped/is_talk_over fields

### P0-2: Padded WAV replaces mic mute (code done, needs verification)

- `_prepare_chromium_wav()` appends 10s silence after speech (48kHz zeros)
- No more `setMicrophoneEnabled(false)` or `resetForMeasurement()`
- `monitorMic` detects EoT naturally via energy drop

**Acceptance**:
- Run mini 4 cases, confirm `t_user_eot_browser` comes from natural energy drop (not mute)
- Turn-taking cases raw USER_KPI should NOT all be ~200ms (should have more variance)
- Should NOT have many negative values (if so, silence padding insufficient or WAV looped)

### P0-3: Playout detection precision (code done, needs verification)

- `PLAYOUT_POLL_MS` from 30ms to 5ms
- `MIC_POLL_MS` from 30ms to 10ms
- `agentAnalyser.fftSize` from 512 to 256 (faster processing)
- Each trace records `playout_resolution_ms` and `mic_resolution_ms`

**Acceptance**: USER_KPI variance should be larger than D12 (no longer stuck at 200 plus/minus 8ms)

### P0-4: USER_KPI gate from WARN to FAIL-ready

- Run 3x mini suite (repeat 3), collect USER_KPI fluctuation data
- Use `tools/autortc/baseline_stability.py` approach for USER_KPI stats (min/med/P95/P99/max/sigma)
- Suggested threshold: FAIL gate = baseline_P95 + 50ms
- In `audio_metrics.py` upgrade WARN to FAIL (or prepare the switch)

**Acceptance**: Have USER_KPI fluctuation stats and suggested threshold

### P1-1: Hearing calibration report

- Sample 4 cases, compare AutoBrowser USER_KPI vs engineering estimate
- Since no real human listening test, compare `user_kpi_raw_ms` (browser) vs `eot_to_first_audio_ms` (probe, from autortc summary)
- Output `output/autobrowser/calibration_report.md`

### P1-2: netem effectiveness

- Check if container has NET_ADMIN (`tc qdisc add dev eth0 root netem delay 1ms` test)
- If not, try `toxiproxy` or Python socket proxy for application-layer impairment
- If nothing works, document clearly in report

## GPU Server Remote Access

GPU server (RunPod L40S) runs the full stack:
- TTS Server (:9000)
- LLM Server (vLLM)
- LiveKit Agent (:8089)
- Token Server (:9090)
- Playwright + Chromium

**SSH is pre-configured**, just use:
```bash
ssh gpu 'echo OK && hostname'
```

**Project path**: `/workspace/project 1/25/`

**Run commands on GPU server**:
```bash
# Single command
ssh gpu 'cd "/workspace/project 1/25" && python3 -u tools/autobrowser/run_suite.py --cases_json tools/autortc/cases/mini_cases.json --token_api http://127.0.0.1:9090/api/token --record_s 25 --p0_only 1 --output_root /tmp/d13_test --inter_case_wait_s 10 2>&1'

# Read file
ssh gpu 'cat "/workspace/project 1/25/output/autobrowser/latest/report.md"'

# Check service status
ssh gpu 'curl -s http://127.0.0.1:9090/api/token?room=health&identity=test | head -1'
```

**Code sync flow** (local edit then GPU runs):
```bash
# 1. Local commit + push
git add -A && git commit -m "..." && git push origin main

# 2. GPU server pull
ssh gpu 'cd "/workspace/project 1/25" && git pull origin main'

# 3. Run test on GPU
ssh gpu 'cd "/workspace/project 1/25" && python3 -u tools/autobrowser/run_suite.py ...'
```

## Execution Steps

1. **Verify SSH**: `ssh gpu 'echo OK && hostname'`
2. **Verify services**: `ssh gpu 'curl -s http://127.0.0.1:9090/api/token?room=test&identity=test | python3 -m json.tool | head -3'`
3. **Run mini 4 cases on GPU to verify D13 changes**
4. **If issues: fix code locally, push, GPU pull, rerun**
5. **Read results**: `ssh gpu 'cat /tmp/d13_test/*/report.md'`
6. **Run full 16 cases**
7. **P0-4 stability sampling (repeat 3)**
8. **Update report template, SKILL.md, DEV_LOG.md**
9. **Final commit + push**

## Critical Caveats

- **DO NOT** use `requestAnimationFrame` - does not fire in headless Chromium, must use `setInterval`
- **DO NOT** re-introduce mic mute + resetForMeasurement (removing it IS the D13 core change)
- `user_kpi_ms=0` is a valid value (Python falsy trap), must use `is not None`
- Token server on GPU at `http://127.0.0.1:9090/api/token`
- WAV file paths relative to `/workspace/project 1/25/`
- Wait 10-15s between cases for LiveKit Agent process pool recycling
- SSH paths have spaces, must quote: `"/workspace/project 1/25/"`

## When Done

1. Update `DEV_LOG.md` with D13 record (what was done, results, lessons)
2. Update `SKILL.md` if new lessons learned
3. git commit + push
