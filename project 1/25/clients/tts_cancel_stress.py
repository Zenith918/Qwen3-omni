#!/usr/bin/env python3
import os
import time
import json
import requests

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TOTAL_RUNS = int(os.environ.get("TTS_TOTAL_RUNS", "30"))
CANCEL_RUNS = int(os.environ.get("TTS_CANCEL_RUNS", "10"))
CANCEL_AFTER_CHUNKS = int(os.environ.get("TTS_CANCEL_AFTER_CHUNKS", "1"))
TTS_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "60"))
TEXT = os.environ.get("TTS_TEXT", "你好，我在听。")
MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "256"))

payload = {
    "text": TEXT,
    "task_type": "CustomVoice",
    "language": "Chinese",
    "speaker": "Vivian",
    "instruct": "",
    "max_new_tokens": MAX_NEW_TOKENS,
    "non_streaming_mode": False,
}


def build_cancel_set(total: int, cancel_count: int) -> set[int]:
    if cancel_count <= 0:
        return set()
    stride = max(1, total // cancel_count)
    picks = list(range(stride, total + 1, stride))[:cancel_count]
    return set(picks)


def run_once(idx: int, cancel: bool) -> dict:
    start = time.time()
    first_chunk_ts = None
    chunk_count = 0
    headers = {}
    try:
        with requests.post(
            TTS_URL,
            json=payload,
            stream=True,
            timeout=(10, TTS_TIMEOUT_S),
        ) as resp:
            resp.raise_for_status()
            headers = dict(resp.headers)
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                chunk_count += 1
                if first_chunk_ts is None:
                    first_chunk_ts = time.time()
                if cancel and chunk_count >= CANCEL_AFTER_CHUNKS:
                    break
        ok = True
        error = ""
    except Exception as e:
        ok = False
        error = str(e)
    total_s = time.time() - start
    ttfa_ms = (first_chunk_ts - start) * 1000.0 if first_chunk_ts else -1.0
    if cancel and first_chunk_ts is None:
        ok = False
        error = "cancel_no_chunk"
    return {
        "idx": idx,
        "cancel": cancel,
        "ok": ok,
        "error": error,
        "ttfa_ms": ttfa_ms,
        "total_s": total_s,
        "headers": headers,
    }


def main() -> None:
    cancel_set = build_cancel_set(TOTAL_RUNS, CANCEL_RUNS)
    results = []
    for i in range(1, TOTAL_RUNS + 1):
        cancel = i in cancel_set
        res = run_once(i, cancel)
        results.append(res)
        print(
            "[CANCEL] idx=%d cancel=%s ok=%s ttfa_ms=%.2f total_s=%.2f err=%s"
            % (
                i,
                str(cancel).lower(),
                str(res["ok"]).lower(),
                res["ttfa_ms"],
                res["total_s"],
                res["error"],
            )
        )

    failed = [r for r in results if not r["ok"]]
    summary = {
        "total": TOTAL_RUNS,
        "cancelled": sum(1 for r in results if r["cancel"]),
        "failed": len(failed),
        "failures": failed,
    }
    print("[CANCEL] summary:", json.dumps(summary, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


