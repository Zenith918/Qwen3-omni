#!/usr/bin/env python3
import json
import os
import random
import signal
import time
import wave
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
BRIDGE_TIMEOUT_PROFILE = os.environ.get("BRIDGE_TIMEOUT_PROFILE", "sanity").lower()
if BRIDGE_TIMEOUT_PROFILE == "long":
    BRIDGE_TIMEOUT_CONNECT_S = float(os.environ.get("BRIDGE_TIMEOUT_CONNECT_S", "10"))
    BRIDGE_TIMEOUT_READ_S = float(os.environ.get("BRIDGE_TIMEOUT_READ_S", "180"))
else:
    BRIDGE_TIMEOUT_CONNECT_S = float(os.environ.get("BRIDGE_TIMEOUT_CONNECT_S", "5"))
    BRIDGE_TIMEOUT_READ_S = float(os.environ.get("BRIDGE_TIMEOUT_READ_S", "60"))

BRIDGE_TTS_WORKERS = int(os.environ.get("BRIDGE_TTS_WORKERS", "1"))
BRIDGE_MAX_SEGMENTS = int(os.environ.get("BRIDGE_MAX_SEGMENTS", "0"))
BRIDGE_OVERALL_TIMEOUT_S = float(os.environ.get("BRIDGE_OVERALL_TIMEOUT_S", "180"))
BRIDGE_WARMUP_RUNS = int(os.environ.get("BRIDGE_WARMUP_RUNS", "0"))
BRIDGE_WARM_RUNS = int(os.environ.get("BRIDGE_WARM_RUNS", "1"))
BRIDGE_ABORT_ON_FAIL = os.environ.get("BRIDGE_ABORT_ON_FAIL", "0").lower() in ("1", "true", "yes")
BRIDGE_MODE = os.environ.get("BRIDGE_MODE", "llm").lower()
BRIDGE_FORCE_FLUSH_LEN = os.environ.get("BRIDGE_FORCE_FLUSH_LEN", "0").lower() in ("1", "true", "yes")
BRIDGE_FLUSH_MIN_CHARS = int(os.environ.get("BRIDGE_FLUSH_MIN_CHARS", "8"))
BRIDGE_FLUSH_MAX_CHARS = int(os.environ.get("BRIDGE_FLUSH_MAX_CHARS", "12"))
BRIDGE_STATIC_SEGMENTS = os.environ.get("BRIDGE_STATIC_SEGMENTS", "").strip()

os.makedirs(OUT_DIR, exist_ok=True)

user_prompt = os.environ.get(
    "BRIDGE_PROMPT",
    "请用中文介绍你能做什么，长度不超过三句话。",
)
BRIDGE_MAX_TOKENS = int(os.environ.get("BRIDGE_MAX_TOKENS", "256"))
starter_candidates = ["嗯…", "我在", "好的", "我听到了", "请说", "在的"]
stop_event = Event()


def _record_failure(seg_idx: int, text: str, queue_depth: int) -> None:
    failure_path = os.path.join(OUT_DIR, "bridge_failures.jsonl")
    record = {
        "ts": time.time(),
        "segment_idx": seg_idx,
        "text": text,
        "length": len(text),
        "queue_depth": queue_depth,
    }
    try:
        with open(failure_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

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


def tts_call(
    session: requests.Session,
    text: str,
    idx: int,
    seg_type: str,
    t_seg_created: float,
    preview: str,
) -> tuple[float, str, float, float, float]:
    payload = {
        "text": text,
        "task_type": "CustomVoice",
        "language": "Chinese",
        "speaker": "Vivian",
        "instruct": "",
        "max_new_tokens": BRIDGE_TTS_MAX_NEW_TOKENS,
        "non_streaming_mode": False,
    }
    t_tts_req_sent = time.time()
    first_chunk = None
    t_tts_done = None
    out_path = os.path.join(OUT_DIR, f"chunk_{idx:02d}.wav")
    try:
        with session.post(
            TTS_URL,
            json=payload,
            stream=True,
            timeout=(BRIDGE_TIMEOUT_CONNECT_S, BRIDGE_TIMEOUT_READ_S),
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
            t_tts_done = time.time()
    except requests.exceptions.RequestException as e:
        t_tts_done = time.time()
        print(
            "[BRIDGE_SEG] idx=%d type=%s chars=%d preview=%s "
            "t_seg_created=%.6f t_tts_req_sent=%.6f t_tts_first_byte=-1.000000 t_tts_done=%.6f "
            "queue_wait=%.3f tts_ttfa_client=-1.000 err=%s"
            % (
                idx,
                seg_type,
                len(text),
                preview,
                t_seg_created,
                t_tts_req_sent,
                t_tts_done,
                t_tts_req_sent - t_seg_created,
                str(e),
            )
        )
        return -1.0, out_path, t_tts_req_sent, -1.0, t_tts_done

    t_tts_first_byte = first_chunk if first_chunk else -1.0
    t_tts_done = t_tts_done if t_tts_done else time.time()
    queue_wait = t_tts_req_sent - t_seg_created
    tts_ttfa_client = (t_tts_first_byte - t_tts_req_sent) if t_tts_first_byte > 0 else -1.0
    print(
        "[BRIDGE_SEG] idx=%d type=%s chars=%d preview=%s "
        "t_seg_created=%.6f t_tts_req_sent=%.6f t_tts_first_byte=%.6f t_tts_done=%.6f "
        "queue_wait=%.3f tts_ttfa_client=%.3f"
        % (
            idx,
            seg_type,
            len(text),
            preview,
            t_seg_created,
            t_tts_req_sent,
            t_tts_first_byte,
            t_tts_done,
            queue_wait,
            tts_ttfa_client,
        )
    )
    return t_tts_first_byte if t_tts_first_byte else -1.0, out_path, t_tts_req_sent, t_tts_first_byte, t_tts_done


def run_once(run_idx: int | None = None) -> dict:
    stop_event.clear()
    start = time.time()
    if BRIDGE_TTS_WORKERS != 1:
        print(f"[BRIDGE] warning: BRIDGE_TTS_WORKERS={BRIDGE_TTS_WORKERS} ignored; forced sequential")
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
    results: list[tuple[int, float, str, float, float, float, str]] = []
    nonlocal_failed = [0]
    queue_len_peak = 0
    flush_stats = {"punct": 0, "len": 0, "starter": 0}
    segment_queue = Queue(maxsize=BRIDGE_QUEUE_MAX) if BRIDGE_QUEUE_MAX > 0 else Queue()

    def _handle_sigint(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    def producer():
        nonlocal buffer, idx, queue_len_peak
        if stop_event.is_set():
            segment_queue.put(None)
            return

        stream_start = time.time()
        starter = pick_starter()
        if starter:
            idx += 1
            flush_stats["starter"] += 1
            qdepth = segment_queue.qsize()
            segment_queue.put((starter, idx, qdepth, "starter", time.time()))
            queue_len_peak = max(queue_len_peak, qdepth)
            if BRIDGE_MAX_SEGMENTS > 0 and idx >= BRIDGE_MAX_SEGMENTS:
                stop_event.set()

        if BRIDGE_MODE == "static":
            segments = []
            if BRIDGE_STATIC_SEGMENTS:
                segments = [s.strip() for s in BRIDGE_STATIC_SEGMENTS.split("|") if s.strip()]
            else:
                segments = ["你好。", "我在这里。", "请继续说。"]
            for seg in segments:
                idx += 1
                flush_stats["len"] += 1
                qdepth = segment_queue.qsize()
                segment_queue.put((seg, idx, qdepth, "main", time.time()))
                queue_len_peak = max(queue_len_peak, qdepth)
            segment_queue.put(None)
            return

        try:
            with requests.post(
                f"{LLM_BASE_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(BRIDGE_TIMEOUT_CONNECT_S, BRIDGE_TIMEOUT_READ_S),
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
                        should, reason = should_flush(buffer, BRIDGE_FLUSH_MIN_CHARS, BRIDGE_FLUSH_MAX_CHARS)
                        if BRIDGE_FORCE_FLUSH_LEN and len(buffer.strip()) >= BRIDGE_FLUSH_MIN_CHARS:
                            should = True
                            reason = "len"
                        if should:
                            idx += 1
                            flush_stats[reason] += 1
                            qdepth = segment_queue.qsize()
                            segment_queue.put((buffer.strip(), idx, qdepth, "main", time.time()))
                            queue_len_peak = max(queue_len_peak, qdepth)
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
            qdepth = segment_queue.qsize()
            segment_queue.put((buffer.strip(), idx, qdepth, "main", time.time()))
            queue_len_peak = max(queue_len_peak, qdepth)
        segment_queue.put(None)

    def consumer():
        while True:
            try:
                item = segment_queue.get(timeout=0.2)
            except Empty:
                continue
            if item is None:
                segment_queue.task_done()
                break
            text, seg_idx, qdepth, seg_type, t_seg_created = item
            preview = text[:20].replace("\n", " ")
            first_chunk_ts, out_path, t_req_sent, t_first, t_done = tts_call(
                session, text, seg_idx, seg_type, t_seg_created, preview
            )
            results.append((seg_idx, first_chunk_ts, out_path, t_req_sent, t_first, t_done, seg_type))
            if first_chunk_ts <= 0:
                nonlocal_failed[0] += 1
                _record_failure(seg_idx, text, qdepth)
            segment_queue.task_done()

    with requests.Session() as session:
        prod_thread = Thread(target=producer, daemon=True)
        cons_thread = Thread(target=consumer, daemon=True)
        prod_thread.start()
        cons_thread.start()
        prod_thread.join()
        cons_thread.join()

    results.sort(key=lambda x: x[0])
    first_audio = None
    for _, first_chunk_ts, out_path, _, _, _, _ in results:
        out_paths.append(out_path)
        if first_audio is None and isinstance(first_chunk_ts, float) and first_chunk_ts > 0:
            first_audio = first_chunk_ts

    total = time.time() - start
    first_audio_s = (first_audio - start) if first_audio else -1.0
    total_audio_s = 0.0
    for path in out_paths:
        try:
            total_audio_s += sf.info(path).duration
        except Exception:
            pass
    failed = nonlocal_failed[0]
    prefix = f"[BRIDGE][RUN {run_idx:02d}] " if run_idx is not None else "[BRIDGE] "
    print(
        prefix
        + "ttfa_s=%.3f first_audio_s=%.3f total_s=%.3f audio_s=%.3f "
        "chunks=%d queue_len_peak=%d flush_punct=%d flush_len=%d starter=%d stop=%s failed=%d"
        % (
            first_audio_s,
            first_audio_s,
            total,
            total_audio_s,
            idx,
            queue_len_peak,
            flush_stats["punct"],
            flush_stats["len"],
            flush_stats["starter"],
            str(stop_event.is_set()).lower(),
            failed,
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
        "failed": failed,
        "queue_len_peak": queue_len_peak,
    }


def main():
    _run_stop_server()
    if BRIDGE_WARMUP_RUNS > 0:
        for i in range(BRIDGE_WARMUP_RUNS):
            run_once(i + 1)

    warm_results = []
    for i in range(BRIDGE_WARM_RUNS):
        result = run_once(i + 1)
        warm_results.append(result)
        if BRIDGE_ABORT_ON_FAIL and result["failed"] > 0:
            print("[BRIDGE] abort_on_fail=true; stopping further runs")
            break

    ok_runs = [r for r in warm_results if r["first_audio_s"] > 0]
    first_audio_vals = [r["first_audio_s"] for r in ok_runs]
    total_vals = [r["total_s"] for r in ok_runs]
    audio_vals = [r["audio_s"] for r in ok_runs]
    chunks_vals = [r["chunks"] for r in ok_runs]
    flush_punct_vals = [r["flush_punct"] for r in ok_runs]
    flush_len_vals = [r["flush_len"] for r in ok_runs]
    starter_vals = [r["starter"] for r in ok_runs]
    failed_total = sum(r["failed"] for r in warm_results)
    failed_runs = sum(1 for r in warm_results if r["failed"] > 0 or r["first_audio_s"] <= 0)

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
    print("[BRIDGE] failed_runs=%d failed_segments=%d" % (failed_runs, failed_total))


if __name__ == "__main__":
    main()
