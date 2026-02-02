#!/usr/bin/env python3
import argparse
import json
import time

import requests


def load_text(texts_path: str, text_id: str) -> str:
    with open(texts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("texts", []):
        if item.get("id") == text_id:
            return item.get("text", "")
    raise ValueError(f"text_id not found: {text_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-id", default="long_03")
    parser.add_argument("--texts", default="/workspace/project 1/25/clients/texts_p0_base.json")
    parser.add_argument("--stream-url", default="http://127.0.0.1:9000/tts/stream")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    text = load_text(args.texts, args.text_id)
    payload = {
        "text": text,
        "task_type": "Base",
        "language": "Chinese",
        "speaker": "serena",
        "instruct": "",
        "max_new_tokens": 2048,
        "non_streaming_mode": False,
    }

    tags = []
    for idx in range(args.count):
        with requests.post(args.stream_url, json=payload, stream=True, timeout=(10, 600)) as resp:
            resp.raise_for_status()
            tag = resp.headers.get("X-Code-Dump-Tag", "")
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
            tags.append(tag)
            print(f"[codes_dump] idx={idx} tag={tag}")

    if args.manifest:
        manifest = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "text_id": args.text_id,
            "count": args.count,
            "tags": tags,
        }
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
