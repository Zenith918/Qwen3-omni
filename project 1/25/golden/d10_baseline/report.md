# AutoRTC Report (D11)

## PRIMARY KPI

- **EoT→FirstAudio P95**: `17.23` ms
- Baseline (D10): `None` ms
- Δ: `N/A` ms

- run_id: `20260216_071946`
- total_cases: `16`
- ok_cases: `16`
- P0 cases: `12`, P1 cases: `4`

## Capture Status

- OK: `16`
- POST_SILENT: `0`
- PRE_MISSING: `0`
- POST_MISSING: `0`

## Aggregates (reply segment)

- EoT->FirstAudio P95 (P0 valid): `17.229999999999997` ms
- tts_first->publish P95: `0.3` ms
- fast lane TTFT P95: `71.355` ms
- inter_arrival P95: `21.025466918945312` ms
- max clipping_ratio: `0.0`
- P0 audible_dropout total (reply): `0`
- P0 max_gap_ms (reply): `220.0` (from 12 cases with reply_wav)
- P0 audio valid: `12/12`
- P0 with reply_wav: `12/12`
- pre_rtc coverage: `16/16`
- mel valid (capture=OK): `16/16`

## Gates (8/8)

- PASS: EoT->FirstAudio P95 <= 650ms
- PASS: tts_first->publish P95 <= 120ms
- PASS: audible_dropout == 0 (P0 reply)
- FAIL: max_gap < 200ms (P0 reply)
- PASS: clipping_ratio < 0.1%
- PASS: fast lane TTFT P95 <= 80ms
- PASS: P0 audio valid rate = 100%
- PASS: inter_arrival_p95 <= 30ms

**Result: 7/8 PASS**

## P1 Anomaly Fingerprints (WARN, not gated)

- `boom_trigger`: input_spike_count=1 (boom verified in input), input_peak_deriv_max=1.9999, input_max_abs_peak=1.0000 (hard-clipped)
- `speed_drift`: drift_ratio=4.7143 (>2% deviation), duration_diff_ms=10400.0
- `distortion_sibilant`: hf_ratio_drop=0.011199
- `stutter_long_pause`: expected_silence_coverage=0.0

## Per-Case Detail

| case_id | tier | capture | reply_rms | reply_max_gap | audible | mel | spike | deriv_max | hf_drop | drift |
|---------|------|---------|-----------|---------------|---------|-----|-------|-----------|---------|-------|
| endpoint_short_hello | P0 | OK | 0.038096 | 140.0 | 0 | 14.1991 | 0 | 0.2455 | 0.015806 | 1.3125 |
| endpoint_fast_speech | P0 | OK | 0.042606 | 120.0 | 0 | 11.1361 | 0 | 0.2101 | 0.003031 | 0.75 |
| endpoint_long_sentence | P0 | OK | 0.014024 | 0.0 | 0 | 18.1536 | 0 | 0.1626 | 0.040854 | 4.3587 |
| endpoint_low_volume_like | P0 | OK | 0.042875 | 0.0 | 0 | 14.3666 | 0 | 0.1553 | 0.008242 | 0.8019 |
| interrupt_once | P0 | OK | 0.0339 | 220.0 | 0 | 9.504 | 0 | 0.1414 | 0.02415 | 1.4592 |
| interrupt_twice | P0 | OK | 0.037563 | 100.0 | 0 | 9.708 | 0 | 0.1866 | 0.016186 | 0.7094 |
| noise_background | P0 | OK | 0.01479 | 0.0 | 0 | 14.4842 | 0 | 0.053 | 0.019978 | 3.9773 |
| noise_cough_laugh | P0 | OK | 0.027449 | 0.0 | 0 | 12.8931 | 0 | 0.1083 | -0.002671 | 0.7417 |
| stress_20_turns_01 | P0 | OK | 0.024179 | 20.0 | 0 | 14.3574 | 0 | 0.0849 | 0.008714 | 3.0167 |
| stress_20_turns_02 | P0 | OK | 0.046935 | 0.0 | 0 | 10.7498 | 0 | 0.1699 | 0.020017 | 0.7077 |
| quality_short_text_guard | P0 | OK | 0.024038 | 160.0 | 0 | 12.4125 | 0 | 0.2112 | 0.091981 | 5.0125 |
| quality_continuation_trigger | P0 | OK | 0.039424 | 60.0 | 0 | 12.5054 | 0 | 0.1454 | 0.039779 | 2.2609 |
| boom_trigger | P1 | OK | 0.036706 | 0.0 | 0 | 10.4007 | 0 | 0.1163 | 0.007651 | 1.2149 |
| speed_drift | P1 | OK | 0.001224 | 0.0 | 0 | 11.5378 | 0 | 0.0539 | -0.283567 | 4.7143 |
| distortion_sibilant | P1 | OK | 0.055915 | 140.0 | 0 | 14.9879 | 0 | 0.1725 | 0.011199 | 1.5982 |
| stutter_long_pause | P1 | OK | 0.033713 | 0.0 | 0 | 13.3151 | 0 | 0.1461 | 0.067267 | 1.537 |

## Suggested Fixes

| case_id | issue | action |
|---------|-------|--------|
| endpoint_short_hello | drift_ratio=1.3125 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| endpoint_fast_speech | drift_ratio=0.7500 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| endpoint_long_sentence | mel_distance=18.2 | High mel_distance → check resample chain, bit-width scaling, or repeated resample. |
| endpoint_long_sentence | drift_ratio=4.3587 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| endpoint_low_volume_like | drift_ratio=0.8019 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| interrupt_once | drift_ratio=1.4592 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| interrupt_twice | drift_ratio=0.7094 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| noise_background | drift_ratio=3.9773 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| noise_cough_laugh | drift_ratio=0.7417 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| stress_20_turns_01 | drift_ratio=3.0167 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| stress_20_turns_02 | drift_ratio=0.7077 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| quality_short_text_guard | drift_ratio=5.0125 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| quality_continuation_trigger | drift_ratio=2.2609 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| boom_trigger | drift_ratio=1.2149 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| speed_drift | drift_ratio=4.7143 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| distortion_sibilant | drift_ratio=1.5982 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
| stutter_long_pause | drift_ratio=1.5370 | Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing. |
