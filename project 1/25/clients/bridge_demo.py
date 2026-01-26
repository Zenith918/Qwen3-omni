#!/usr/bin/env python3
import json
import os
import time
import requests
import soundfile as sf

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/synthesize")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/project 1/25/output/bridge")

os.makedirs(OUT_DIR, exist_ok=True)

user_prompt = "请用中文介绍你能做什么，长度不超过三句话。"


def sse_stream(resp):
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            yield data


def should_flush(text: str) -> bool:
    if len(text) >= 40:
        return True
    for ch in "。！？!?\n":
        if ch in text:
            return True
    return False


def tts_call(text: str, idx: int) -> tuple[float, str]:
    payload = {
        "text": text,
        "task_type": "CustomVoice",
        "language": "Chinese",
        "speaker": "Vivian",
        "instruct": "",
        "max_new_tokens": 2048,
    }
    start = time.time()
    first_chunk = None
    out_path = os.path.join(OUT_DIR, f"chunk_{idx:02d}.wav")
    with requests.post(TTS_URL, json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = time.time()
                f.write(chunk)
    return (first_chunk - start) if first_chunk else -1.0, out_path


def main():
    start = time.time()
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are Qwen."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 256,
        "stream": True,
    }

    buffer = ""
    idx = 0
    first_audio = None
    out_paths: list[str] = []

    with requests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for data in sse_stream(resp):
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                buffer += delta["content"]
                if should_flush(buffer):
                    idx += 1
                    tts_latency, out_path = tts_call(buffer.strip(), idx)
                    out_paths.append(out_path)
                    if first_audio is None and tts_latency >= 0:
                        first_audio = time.time()
                    buffer = ""

    if buffer.strip():
        idx += 1
        tts_latency, out_path = tts_call(buffer.strip(), idx)
        out_paths.append(out_path)
        if first_audio is None and tts_latency >= 0:
            first_audio = time.time()

    total = time.time() - start
    first_audio_s = (first_audio - start) if first_audio else -1.0
    total_audio_s = 0.0
    for path in out_paths:
        try:
            total_audio_s += sf.info(path).duration
        except Exception:
            pass
    print(f"[BRIDGE] chunks={idx} first_audio_s={first_audio_s:.3f} total_s={total:.3f} audio_s={total_audio_s:.3f}")


if __name__ == "__main__":
    main()
