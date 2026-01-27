#!/usr/bin/env python3
import os
import time
import requests
import soundfile as sf
import wave

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
OUT_WAV = os.environ.get("TTS_OUT", "/workspace/project 1/25/output/tts_smoke.wav")
TTS_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "60"))

payload = {
    "text": "你好，我是你的语音助手。",
    "task_type": "CustomVoice",
    "language": "Chinese",
    "speaker": "Vivian",
    "instruct": "",
    "max_new_tokens": 2048,
    "non_streaming_mode": False,
}

COLD_RUNS = int(os.environ.get("TTS_COLD_RUNS", "1"))
WARMUP_RUNS = int(os.environ.get("TTS_WARMUP_RUNS", "2"))
WARM_RUNS = int(os.environ.get("TTS_WARM_RUNS", "20"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return -1.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def run_once(out_path: str) -> dict:
    start = time.time()
    first_chunk = None
    headers = {}
    with requests.post(
        TTS_URL,
        json=payload,
        stream=True,
        timeout=(10, TTS_TIMEOUT_S),
    ) as resp:
        resp.raise_for_status()
        headers = dict(resp.headers)
        sr = int(resp.headers.get("x-sample-rate", "24000"))
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = time.time()
                wf.writeframes(chunk)

    end = time.time()
    dur_s = -1.0
    try:
        info = sf.info(out_path)
        dur_s = info.duration
    except Exception:
        pass

    rtf = (end - start) / dur_s if dur_s and dur_s > 0 else -1.0
    ttfa = (first_chunk - start) if first_chunk else -1.0
    return {
        "first_chunk_s": ttfa,
        "total_s": end - start,
        "dur_s": dur_s,
        "rtf": rtf,
        "headers": headers,
    }


print("[TTS] cold runs:", COLD_RUNS)
cold_runs = []
for i in range(COLD_RUNS):
    cold_runs.append(run_once(OUT_WAV))

cold_ttfa_vals = [r["first_chunk_s"] for r in cold_runs]
cold_total_vals = [r["total_s"] for r in cold_runs]
cold_rtf_vals = [r["rtf"] for r in cold_runs if r["rtf"] > 0]
print("[TTS] cold headers:", cold_runs[-1]["headers"] if cold_runs else {})
print(
    "[TTS] cold TTFA p50=%.3f p95=%.3f"
    % (percentile(cold_ttfa_vals, 0.5), percentile(cold_ttfa_vals, 0.95))
)
print(
    "[TTS] cold total p50=%.3f p95=%.3f"
    % (percentile(cold_total_vals, 0.5), percentile(cold_total_vals, 0.95))
)
print(
    "[TTS] cold rtf p50=%.3f p95=%.3f"
    % (percentile(cold_rtf_vals, 0.5), percentile(cold_rtf_vals, 0.95))
)

print("[TTS] warmup runs:", WARMUP_RUNS)
for i in range(WARMUP_RUNS):
    run_once(OUT_WAV)

warm_runs = []
for i in range(WARM_RUNS):
    warm_runs.append(run_once(OUT_WAV))

ttfa_vals = [r["first_chunk_s"] for r in warm_runs]
total_vals = [r["total_s"] for r in warm_runs]
rtf_vals = [r["rtf"] for r in warm_runs if r["rtf"] > 0]

print("[TTS] warm headers:", warm_runs[-1]["headers"] if warm_runs else {})
print(
    "[TTS] warm TTFA p50=%.3f p95=%.3f"
    % (percentile(ttfa_vals, 0.5), percentile(ttfa_vals, 0.95))
)
print(
    "[TTS] warm total p50=%.3f p95=%.3f"
    % (percentile(total_vals, 0.5), percentile(total_vals, 0.95))
)
print(
    "[TTS] warm rtf p50=%.3f p95=%.3f"
    % (percentile(rtf_vals, 0.5), percentile(rtf_vals, 0.95))
)
