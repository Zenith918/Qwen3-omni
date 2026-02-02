


## Notes

- vLLM and vLLM-Omni spawn worker processes; killing only the parent leaves GPU allocations behind.
- Use the built-in process-group trap in `scripts/run_llm_server.sh` and `scripts/run_tts_server.sh` to avoid orphan workers.
- Avoid running multiple GPU servers concurrently unless needed; memory fragmentation can prevent startup.

## TTS Overlap Debug SOP (Q31-Q36)

Purpose: reproduce deterministic TTFA/RTF runs, isolate drift, and compare overlap
schemes for TTS streaming (long_03, packet_tokens=4, left_context=72).

### Common setup

1) Stop existing servers and workers:
   - `ps -C python3 -o pid,cmd`
   - `kill <pid>` for any `tts_server.py` parent processes
2) Start a single server with explicit env (examples below).
3) Wait until the server is ready:
   - Look for "Application startup complete" in terminal logs.
4) Warm up once:
   - `python3 "clients/tts_codes_dump.py" --text-id long_03`
5) Run timing stats:
   - `python3 "tmp_ttfa_runs.py" --text-id long_03 --count 30 --out "output/debug/<name>.json"`

Tip: if you see "Connection refused", the server is not ready. Wait and retry.

### Q31/Q32: process=0 no overlap

Use `TTS_DEEP_STREAM_PROCESS=0` and vary packet_tokens:
  - packet=4: baseline TTFA lower bound
  - packet=2 / packet=1: check if TTFA can approach 350ms

### Q33: overlap A vs B (hybrid)

Start with process=1 and only change prefill:
  - A (immediate overlap): `TTS_DEEP_STREAM_PREFILL_PACKETS=0`
  - B (first packet then overlap): `TTS_DEEP_STREAM_PREFILL_PACKETS=1`

After each run, collect first-packet stability:
  - `python3 "clients/tts_codes_dump.py" --text-id long_03 --count 2 --manifest "output/debug/q33_tags.json"`
  - `python3 "tmp_codes_analysis.py" --packet-tokens 4 --compare --tags <tag1> <tag2>`

### Q34: dummy decoder trigger source

Use `TTS_DEEP_STREAM_DUMMY_DECODER` to isolate decoder stages:
  - `pre_transformer` (only pre_transformer runs)
  - `conv_only` (skip pre_transformer, run conv/upsample)
  - unset (full decoder)

Then use `tmp_codes_analysis.py` to compare full hash and first diff frame.

### Q35: precision sensitivity

Run 20x each with process=1 overlap:
  - D0: bf16 (default)
  - D1: `TTS_DEEP_STREAM_CODEGEN_FP32=1`
  - D2: `TTS_DEEP_STREAM_DECODER_FP32=1`
  - D3: both env vars above

### Q36: topK drift localization

Enable topK logging in server:
  - `TTS_CODEGEN_DEBUG_TOPK=1`
  - `TTS_CODEGEN_DEBUG_STEP_START=0`
  - `TTS_CODEGEN_DEBUG_STEP_END=100`
  - `TTS_CODEGEN_DEBUG_TOPK_N=2`

Run two identical requests and compare the first step where top1/top2 differ.

## Q37–Q43 Minimal Repro SOP

Use the same fixed texts (short_01 + long_03), and set `temperature=0` / `top_p=1`
via deterministic greedy mode.

### Backend flag toggles (Q37)

Set flags before server start (empty means default):
  - `TTS_CUDNN_BENCHMARK=0|1`
  - `TTS_CUDNN_DETERMINISTIC=0|1`
  - `TTS_CUDNN_ALLOW_TF32=0|1`
  - `TTS_CUDA_MATMUL_ALLOW_TF32=0|1`
  - `TTS_CUDNN_TRACE=1` to print cudnn enabled/available/version

Q37.1 (confirm cudnn availability):
- Quick check:
  - `python3 - <<'PY' ...` (print cudnn enabled/version and run conv1d/convtranspose1d)

### Dummy decoder modes (Q38b)

Use `TTS_DEEP_STREAM_DUMMY_DECODER=noop` to skip conv/upsample while keeping
pre_transformer on GPU.

### Stream sync experiments (Q39)

Note: CUDA events/streams can only be synchronized within the same process.
If you need event-based sync, run with `TTS_DEEP_STREAM_PROCESS=0` so codegen
and decoder share a process.

Q39b (event-based minimal sync):
- Start server with `TTS_DEEP_STREAM_SYNC_MODE=event` and `process=0`
- Run: `python3 "tmp_ttfa_runs.py" --text-id long_03 --count 10 --out "output/debug/q39b_event_long.json"`
- Optional: `python3 "tmp_codes_analysis.py" --runs-json "output/debug/q39b_event_long.json" --packet-tokens 4 --compare`
- Expectation: hash_unique=1 and first_diff_frame=-1 if event sync eliminates drift

### Phase overlap (SYNC_MODE=phase)

Goal: keep pre_transformer overlapped, but add a narrow barrier around conv/upsample.

Requirements:
- `TTS_DEEP_STREAM_PROCESS=0` (same process)
- `TTS_DEEP_STREAM_CODEGEN_BLOCKING=0` (non-blocking codegen)
- `TTS_DEEP_STREAM_SYNC_MODE=phase`
- Same GPU for codegen/decoder (`TTS_DEEP_STREAM_DEVICE` == `TTS_DEEP_STREAM_CODEGEN_DEVICE`)

Start server (example):
- `TTS_DEEP_STREAM_SYNC_MODE=phase TTS_DEEP_STREAM_PROCESS=0 TTS_DEEP_STREAM_CODEGEN_BLOCKING=0 ... python3 "clients/tts_server.py"`
- Confirm log does NOT print: `phase sync disabled ...`

Run tests:
- Warm-up: `python3 "clients/tts_codes_dump.py" --text-id long_03 --count 1`
- 10-run timing:
  - `python3 "tmp_ttfa_runs.py" --text-id long_03 --count 10 --out "output/debug/phase_long.json"`
  - `python3 "tmp_ttfa_runs.py" --text-id short_01 --count 10 --out "output/debug/phase_short.json"`
- Hash check:
  - `python3 "tmp_codes_analysis.py" --runs-json "output/debug/phase_long.json" --packet-tokens 4 --compare"`
  - `python3 "tmp_codes_analysis.py" --runs-json "output/debug/phase_short.json" --packet-tokens 4 --compare"`
- Optional wav: `python3 "clients/tts_regression_suite.py" --texts "clients/texts_p0_base.json" --voices "clients/voices_base.json" --out-root "output/regression_phase"`

Acceptance:
- `hash_unique=1` for 10-run
- TTFA P50 better than event sync (target ~350ms level)

### Packet=1 anomaly (Q41–Q43)

Q41 (packet trace):
- Start server with `TTS_DEEP_STREAM_PROCESS=0`, `TTS_DEEP_STREAM_PACKET_TOKENS=1`,
  `TTS_DEEP_STREAM_PACKET_TRACE=1`.
- Run: `python3 "tmp_ttfa_runs.py" --text-id long_03 --count 2 --out "output/debug/q41_packet1_long.json"`.
- Inspect `meta_<tag>.json` for `queue_wait_ms`, `decode_calls`, `pcm_samples_total`.

Q42 (codegen-only, no decoder):
- Run: `python3 "tmp_codegen_only_stream.py" --text-id long_03 --count 2 --out "output/debug/q42_codegen_only_packet1.json"`.
- This uses `_iter_deep_codes` directly (decoder bypassed); compare first_packet_ms/total_ms.

Q43 (decode-only, fixed codes):
- Use a stable tag from packet=2 runs (example: `1769915670616_7828eb21`).
- Inline run (cache mode):
  - `python3 - <<'PY' ...` (see DEV_LOG Q43 for the exact snippet)
- Capture: `samples_total` vs `samples_expected`, `bad_len_calls`, decode_ms_p50.
