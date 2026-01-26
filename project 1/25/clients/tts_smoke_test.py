#!/usr/bin/env python3
import os
import time
import requests
import soundfile as sf

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/synthesize")
OUT_WAV = os.environ.get("TTS_OUT", "/workspace/project 1/25/output/tts_smoke.wav")

payload = {
    "text": "你好，我是你的语音助手。",
    "task_type": "CustomVoice",
    "language": "Chinese",
    "speaker": "Vivian",
    "instruct": "",
    "max_new_tokens": 2048,
}

start = time.time()
first_chunk = None

with requests.post(TTS_URL, json=payload, stream=True, timeout=600) as resp:
    resp.raise_for_status()
    with open(OUT_WAV, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first_chunk is None:
                first_chunk = time.time()
            f.write(chunk)

end = time.time()

dur_s = -1.0
try:
    info = sf.info(OUT_WAV)
    dur_s = info.duration
except Exception:
    pass

rtf = (end - start) / dur_s if dur_s and dur_s > 0 else -1.0

print("[TTS] saved:", OUT_WAV)
print("[TTS] first_chunk_s=%.3f total_s=%.3f dur_s=%.3f rtf=%.3f" % (
    (first_chunk - start) if first_chunk else -1,
    end - start,
    dur_s,
    rtf,
))
print("[TTS] headers:", dict(resp.headers))
