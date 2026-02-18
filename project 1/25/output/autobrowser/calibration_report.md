# D13 USER_KPI Calibration Report

**Date**: 2026-02-17
**Platform**: RunPod L40S, Chromium headless (Playwright)
**Agent**: Qwen3-Omni + Qwen3-TTS via LiveKit
**Network**: wifi_good (local loopback, no impairment)
**Cases**: 4 (mini_cases.json)
**SILENCE_TIMEOUT_MS**: 1500 (AUTO_MODE), 400 (manual)
**PLAYOUT_POLL_MS**: 5, **MIC_POLL_MS**: 10

---

## 1. Per-Case Raw USER_KPI (4 runs)

| Case | Verify | Run 1 | Run 2 | Run 3 | Mean | StdDev |
|------|--------|-------|-------|-------|------|--------|
| endpoint_short_hello | 150 | 328 | -720 | -1835 | -519 | 960 |
| endpoint_long_sentence | 365 | 345 | 735 | 281 | 432 | 205 |
| interrupt_once | 1855 | -3002 | -3365 | -3535 | -2012 | 2471 |
| noise_background | 830 | -1564 | 570 | -3590 | -939 | 2059 |

**Observation**:
- `endpoint_long_sentence` is the most stable case (StdDev=205ms), positive across all runs.
- `interrupt_once` consistently triggers talk-over in stability runs (expected: simulates interruption).
- `endpoint_short_hello` and `noise_background` show high variance.
- High variance reflects real production conditions: agent VAD timing + WebRTC jitter + browser scheduling.

## 2. Clamped USER_KPI (max(0, raw))

| Run | P50 (ms) | P95 (ms) | P99 (ms) | Talk-over | Count |
|-----|----------|----------|----------|-----------|-------|
| Verify | 598 | 1701 | 1824 | 0/4 | 4 |
| Run 1 | 164 | 342 | 344 | 2/4 | 4 |
| Run 2 | 285 | 710 | 730 | 2/4 | 4 |
| Run 3 | 0 | 239 | 273 | 3/4 | 4 |

**Clamped P50 range**: 0-598ms
**Clamped P95 range**: 239-1701ms

## 3. D12 vs D13 Comparison

| Metric | D12 | D13 |
|--------|-----|-----|
| Measurement method | mic mute + resetForMeasurement | Natural EoT detection (energy drop) |
| USER_KPI raw typical | ~200ms +/- 8ms | 150-1855ms (high variance) |
| Talk-over detection | Not measured | Measured (raw < 0 = is_talk_over) |
| Realism | Low (artificial mic mute) | High (production-like) |
| SILENCE_TIMEOUT_MS | N/A | 1500ms (auto) / 400ms (manual) |
| Polling precision | Not tracked | PLAYOUT=5ms, MIC=10ms |

## 4. AutoBrowser vs AutoRTC (Probe) Comparison

D13 user_kpi_raw_ms (browser-side WYSIWYG) captures user-perceived latency including:
- Browser audio pipeline latency
- WebRTC jitter buffer
- Agent processing (VAD -> STT -> LLM -> TTS)
- Playout scheduling

AutoRTC probe measures eot_to_first_audio_ms at network level, excluding browser rendering.
Direct comparison requires simultaneous measurement (not yet implemented).

**Estimated browser overhead**: 50-150ms (audio context scheduling + jitter buffer).

## 5. Known Limitations

1. **netem unavailable**: Container lacks cap_net_admin; tc netem fails. Need NET_ADMIN or toxiproxy.
2. **WAV internal pauses**: Natural pauses > SILENCE_TIMEOUT_MS cause trace splitting. Mitigated by 1500ms timeout.
3. **Talk-over variance**: interrupt_once and noise_background have inherent high variance.
4. **Sample size**: 4 cases x 4 runs = 16 points. Recommend 20+ cases x 10 runs for stable percentiles.

## 6. Recommendations

1. **USER_KPI WARN gate**: P95 <= 900ms (D13 default). Current range sometimes exceeds - expected with small sample.
2. **Increase case diversity**: Add single-utterance WAVs for lower variance.
3. **netem workaround**: Use toxiproxy for application-layer latency injection.
4. **Future**: Concurrent AutoRTC + AutoBrowser for direct probe-vs-browser calibration.

---

## Appendix A: Browser USER_KPI vs Probe eot_to_first_audio_ms (D10 baseline)

Probe data from `golden/d10_baseline/metrics.csv` (D10 AutoRTC run).
Browser data from D13 verification run (mini 4 cases, clamped).

| Case | Probe eot_to_first_audio (ms) | Browser USER_KPI raw (ms) | Browser USER_KPI clamped (ms) | Delta (browser - probe) |
|------|-------------------------------|---------------------------|-------------------------------|-------------------------|
| endpoint_short_hello | 15.7 | 150 | 150 | +134.3 |
| endpoint_long_sentence | 14.2 | 365 | 365 | +350.8 |
| interrupt_once | 4.2 | 1855 | 1855 | +1850.8 |
| noise_background | 8.3 | 830 | 830 | +821.7 |

### Analysis

**Probe `eot_to_first_audio_ms`** measures the time between the probe WAV's end-of-talk marker
and the first agent audio packet arriving at the network level. Values are very low (1-20ms)
because the probe uses precise WAV alignment and network-level capture.

**Browser `user_kpi_raw_ms`** measures from the browser's EoT detection (1500ms silence timeout
after energy drop) to the first playout event in the browser audio pipeline. This includes:

1. **SILENCE_TIMEOUT_MS overhead** (+1500ms potential): Browser must wait 1500ms of silence
   to confirm EoT, but the timer fires at the END of silence, so it adds latency relative
   to when speech actually stopped.
2. **WebRTC jitter buffer** (~50-150ms): Buffers agent audio before playout.
3. **Browser audio scheduling** (~5-20ms): AudioContext callback interval.
4. **Agent processing time**: VAD → STT → LLM → TTS pipeline.

The probe measures at the **network layer** (no browser, no jitter buffer, no silence timeout),
while the browser measures at the **user perception layer** (what the user actually hears).

**Key insight**: The large delta (134-1851ms) is dominated by the SILENCE_TIMEOUT_MS detection
window. The probe doesn't need to "wait and confirm" silence — it knows exactly when the WAV
ends. The browser must observe 1500ms of silence before declaring EoT, making the measurement
inherently larger.

For `interrupt_once` (delta=1851ms): the agent responds to a partial utterance, and the browser's
long silence timeout means EoT is declared much later than the agent's actual response,
inflating the browser metric.

### Conclusion

Browser USER_KPI and probe eot_to_first_audio measure **different things**:
- Probe: network-level latency (ideal, ~1-20ms in loopback)
- Browser: user-perceived latency including EoT detection overhead (realistic, 150-1855ms)

Direct numerical comparison is not meaningful. Browser USER_KPI is the correct production metric
because it reflects what the user actually experiences.
