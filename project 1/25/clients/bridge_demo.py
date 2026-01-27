#!/usr/bin/env python3
import json
import os
import random
import signal
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Event, Thread
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import soundfile as sf

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/project 1/25/output/bridge")
BRIDGE_STOP_PORT = int(os.environ.get("BRIDGE_STOP_PORT", "0"))
BRIDGE_STARTER_TEXT = os.environ.get("BRIDGE_STARTER_TEXT", "").strip()
BRIDGE_QUEUE_MAX = int(os.environ.get("BRIDGE_QUEUE_MAX", "0"))
BRIDGE_STREAM_TIMEOUT_S = float(os.environ.get("BRIDGE_STREAM_TIMEOUT_S", "25"))
BRIDGE_STREAM_READ_TIMEOUT_S = float(os.environ.get("BRIDGE_STREAM_READ_TIMEOUT_S", "10"))
BRIDGE_TTS_MAX_NEW_TOKENS = int(os.environ.get("BRIDGE_TTS_MAX_NEW_TOKENS", "512"))
BRIDGE_TTS_TIMEOUT_S = float(os.environ.get("BRIDGE_TTS_TIMEOUT_S", "60"))
BRIDGE_TTS_WORKERS = int(os.environ.get("BRIDGE_TTS_WORKERS", "2"))
BRIDGE_MAX_SEGMENTS = int(os.environ.get("BRIDGE_MAX_SEGMENTS", "0"))
BRIDGE_OVERALL_TIMEOUT_S = float(os.environ.get("BRIDGE_OVERALL_TIMEOUT_S", "180"))
BRIDGE_WARMUP_RUNS = int(os.environ.get("BRIDGE_WARMUP_RUNS", "0"))
BRIDGE_WARM_RUNS = int(os.environ.get("BRIDGE_WARM_RUNS", "1"))

os.makedirs(OUT_DIR, exist_ok=True)

user_prompt = os.environ.get(
    "BRIDGE_PROMPT",
    "请用中文介绍你能做什么，长度不超过三句话。",
)
BRIDGE_MAX_TOKENS = int(os.environ.get("BRIDGE_MAX_TOKENS", "256"))
starter_candidates = ["嗯…", "我在", "好的", "我听到了", "请说", "在的"]
stop_event = Event()


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


def sse_stream(resp):
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            yield data


def should_flush(text: str, min_chars: int = 8, max_chars: int = 12) -> tuple[bool, str]:
    clean = text.strip()
    if not clean:
        return False, ""
    for ch in "，。！？!?":
        if ch in clean:
            return True, "punct"
    if min_chars <= len(clean) <= max_chars:
        return True, "len"
    return False, ""


def pick_starter() -> str:
    if BRIDGE_STARTER_TEXT:
        return BRIDGE_STARTER_TEXT
    return random.choice(starter_candidates)


class _StopHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/bridge/stop":
            self.send_response(404)
            self.end_headers()
            return
        stop_event.set()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        return


def _run_stop_server() -> None:
    if BRIDGE_STOP_PORT <= 0:
        return
    server = HTTPServer(("0.0.0.0", BRIDGE_STOP_PORT), _StopHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()


def tts_call(text: str, idx: int) -> tuple[float, str]:
    payload = {
        "text": text,
        "task_type": "CustomVoice",
        "language": "Chinese",
        "speaker": "Vivian",
        "instruct": "",
        "max_new_tokens": BRIDGE_TTS_MAX_NEW_TOKENS,
        "non_streaming_mode": False,
    }
    start = time.time()
    first_chunk = None
    out_path = os.path.join(OUT_DIR, f"chunk_{idx:02d}.wav")
    try:
        with requests.post(
            TTS_URL,
            json=payload,
            stream=True,
            timeout=(10, BRIDGE_TTS_TIMEOUT_S),
        ) as resp:
            resp.raise_for_status()
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
    except requests.exceptions.RequestException:
        return -1.0, out_path
    return first_chunk if first_chunk else -1.0, out_path


def run_once(run_idx: int | None = None) -> dict:
    stop_event.clear()
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
        "max_tokens": BRIDGE_MAX_TOKENS,
        "stream": True,
    }

    buffer = ""
    idx = 0
    out_paths: list[str] = []
    futures = []
    flush_stats = {"punct": 0, "len": 0, "starter": 0}
    segment_queue = Queue(maxsize=BRIDGE_QUEUE_MAX) if BRIDGE_QUEUE_MAX > 0 else Queue()

    def _handle_sigint(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    def producer():
        nonlocal buffer, idx
        if stop_event.is_set():
            segment_queue.put(None)
            return

        stream_start = time.time()
        starter = pick_starter()
        if starter:
            idx += 1
            flush_stats["starter"] += 1
            segment_queue.put((starter, idx))
            if BRIDGE_MAX_SEGMENTS > 0 and idx >= BRIDGE_MAX_SEGMENTS:
                stop_event.set()

        try:
            with requests.post(
                f"{LLM_BASE_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(10, BRIDGE_STREAM_READ_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                for data in sse_stream(resp):
                    if stop_event.is_set():
                        break
                    if time.time() - stream_start > BRIDGE_STREAM_TIMEOUT_S:
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        buffer += delta["content"]
                        should, reason = should_flush(buffer)
                        if should:
                            idx += 1
                            flush_stats[reason] += 1
                            segment_queue.put((buffer.strip(), idx))
                            buffer = ""
                            if BRIDGE_MAX_SEGMENTS > 0 and idx >= BRIDGE_MAX_SEGMENTS:
                                stop_event.set()
                                break
        except requests.exceptions.ReadTimeout:
            pass
        except requests.exceptions.RequestException:
            pass

        if not stop_event.is_set() and buffer.strip():
            idx += 1
            flush_stats["len"] += 1
            segment_queue.put((buffer.strip(), idx))
        segment_queue.put(None)

    def consumer(pool: ThreadPoolExecutor):
        while True:
            try:
                item = segment_queue.get(timeout=0.2)
            except Empty:
                continue
            if item is None:
                segment_queue.task_done()
                break
            text, seg_idx = item
            futures.append(pool.submit(tts_call, text, seg_idx))
            segment_queue.task_done()

    with ThreadPoolExecutor(max_workers=BRIDGE_TTS_WORKERS) as pool:
        prod_thread = Thread(target=producer, daemon=True)
        cons_thread = Thread(target=consumer, args=(pool,), daemon=True)
        prod_thread.start()
        cons_thread.start()
        prod_thread.join()
        cons_thread.join()

        first_audio = None
        deadline = start + BRIDGE_OVERALL_TIMEOUT_S
        try:
            for fut in as_completed(futures, timeout=max(0.1, BRIDGE_OVERALL_TIMEOUT_S)):
                if time.time() > deadline:
                    break
                if stop_event.is_set() and fut.cancel():
                    continue
                first_chunk_ts, out_path = fut.result()
                out_paths.append(out_path)
                if isinstance(first_chunk_ts, float) and first_chunk_ts > 0:
                    if first_audio is None or first_chunk_ts < first_audio:
                        first_audio = first_chunk_ts
        except Exception:
            pass

        for fut in futures:
            if not fut.done():
                fut.cancel()

    total = time.time() - start
    first_audio_s = (first_audio - start) if first_audio else -1.0
    total_audio_s = 0.0
    for path in out_paths:
        try:
            total_audio_s += sf.info(path).duration
        except Exception:
            pass
    prefix = f"[BRIDGE][RUN {run_idx:02d}] " if run_idx is not None else "[BRIDGE] "
    print(
        prefix
        + "chunks=%d first_audio_s=%.3f total_s=%.3f audio_s=%.3f "
        "flush_punct=%d flush_len=%d starter=%d stop=%s"
        % (
            idx,
            first_audio_s,
            total,
            total_audio_s,
            flush_stats["punct"],
            flush_stats["len"],
            flush_stats["starter"],
            str(stop_event.is_set()).lower(),
        )
    )
    return {
        "first_audio_s": first_audio_s,
        "total_s": total,
        "audio_s": total_audio_s,
        "chunks": idx,
        "flush_punct": flush_stats["punct"],
        "flush_len": flush_stats["len"],
        "starter": flush_stats["starter"],
        "stopped": stop_event.is_set(),
    }


def main():
    _run_stop_server()
    if BRIDGE_WARMUP_RUNS > 0:
        for i in range(BRIDGE_WARMUP_RUNS):
            run_once(i + 1)

    warm_results = []
    for i in range(BRIDGE_WARM_RUNS):
        warm_results.append(run_once(i + 1))

    ok_runs = [r for r in warm_results if r["first_audio_s"] > 0]
    first_audio_vals = [r["first_audio_s"] for r in ok_runs]
    total_vals = [r["total_s"] for r in ok_runs]
    audio_vals = [r["audio_s"] for r in ok_runs]
    chunks_vals = [r["chunks"] for r in ok_runs]
    flush_punct_vals = [r["flush_punct"] for r in ok_runs]
    flush_len_vals = [r["flush_len"] for r in ok_runs]
    starter_vals = [r["starter"] for r in ok_runs]

    print(
        "[BRIDGE] first_audio_s p50=%.3f p95=%.3f (n=%d)"
        % (percentile(first_audio_vals, 0.5), percentile(first_audio_vals, 0.95), len(ok_runs))
    )
    print(
        "[BRIDGE] total_s p50=%.3f p95=%.3f (n=%d)"
        % (percentile(total_vals, 0.5), percentile(total_vals, 0.95), len(ok_runs))
    )
    if audio_vals:
        print("[BRIDGE] audio_s avg=%.3f" % (sum(audio_vals) / len(audio_vals)))
    if chunks_vals:
        print("[BRIDGE] chunks avg=%.2f" % (sum(chunks_vals) / len(chunks_vals)))
    if flush_punct_vals:
        print("[BRIDGE] flush_punct avg=%.2f" % (sum(flush_punct_vals) / len(flush_punct_vals)))
    if flush_len_vals:
        print("[BRIDGE] flush_len avg=%.2f" % (sum(flush_len_vals) / len(flush_len_vals)))
    if starter_vals:
        print("[BRIDGE] starter avg=%.2f" % (sum(starter_vals) / len(starter_vals)))


if __name__ == "__main__":
    main()
