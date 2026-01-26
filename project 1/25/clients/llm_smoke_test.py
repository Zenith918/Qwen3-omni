#!/usr/bin/env python3
import json
import os
import time
import requests

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")

prompt = (
    "你现在是产品工程师。请用中文简要介绍你能做什么。"
)


def sse_stream(resp):
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            yield data


def main():
    models = requests.get(f"{BASE_URL}/v1/models", timeout=30).json()
    print("[LLM] models:", models)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are Qwen."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 256,
        "stream": True,
    }

    start = time.time()
    first_tok = None
    text = ""
    tok_count = 0

    with requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for data in sse_stream(resp):
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                if first_tok is None:
                    first_tok = time.time()
                text += delta["content"]
                tok_count += 1

    end = time.time()
    ttft = (first_tok - start) if first_tok else None
    tps = tok_count / (end - (first_tok or start) + 1e-6)

    print("[LLM] output:", text)
    print(f"[LLM] ttft_s={ttft:.3f} tok_s={tps:.2f} tokens={tok_count}")


if __name__ == "__main__":
    main()
