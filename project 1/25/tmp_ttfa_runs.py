#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any

import requests


def _percentile(values, p: float) -> float:
    if not values:
        return -1.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(p * (len(values) - 1)))))
    return float(values[k])


def _load_meta(tag: str, dump_dir: str, retry_s: float = 30.0) -> dict[str, Any]:
    path = os.path.join(dump_dir, f"meta_{tag}.json")
    deadline = time.time() + retry_s
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        time.sleep(0.1)
    raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-id", default="long_03")
    parser.add_argument("--texts", default="/workspace/project 1/25/clients/texts_p0_base.json")
    parser.add_argument("--stream-url", default="http://127.0.0.1:9000/tts/stream")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--dump-dir", default="/workspace/project 1/25/output/code_dumps")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    with open(args.texts, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = next(item["text"] for item in data.get("texts", []) if item.get("id") == args.text_id)

    payload = {
        "text": text,
        "task_type": "Base",
        "language": "Chinese",
        "speaker": "serena",
        "instruct": "",
        "max_new_tokens": 2048,
        "non_streaming_mode": False,
    }

    rows = []
    for idx in range(args.count):
        t0 = time.time()
        first_chunk_t = None
        total_bytes = 0
        with requests.post(args.stream_url, json=payload, stream=True, timeout=(10, 600)) as resp:
            resp.raise_for_status()
            tag = resp.headers.get("X-Code-Dump-Tag", "")
            sr = int(resp.headers.get("X-Sample-Rate", "24000"))
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_chunk_t is None:
                    first_chunk_t = time.time()
                total_bytes += len(chunk)
        t1 = time.time()
        if first_chunk_t is None:
            first_chunk_t = t1

        meta = {}
        meta_missing = False
        if not tag:
            meta_missing = True
        else:
            try:
                meta = _load_meta(tag, args.dump_dir)
            except FileNotFoundError:
                meta_missing = True
        frames = int(meta.get("frames", 0))
        # duration from frames if available
        duration_s = None
        if frames > 0 and sr > 0:
            try:
                # 12Hz -> 24000/upsample; fallback using frames*upsample/sr not stored. Use total bytes if present.
                pass
            except Exception:
                pass
        if total_bytes > 0 and sr > 0:
            samples = total_bytes // 2
            duration_s = samples / float(sr)

        row = {
            "idx": idx,
            "tag": tag,
            "sha256": meta.get("sha256", ""),
            "code_ms": float(meta.get("code_ms", -1.0)),
            "decode_first_ms": float(meta.get("decode_first_ms", -1.0)),
            "ttfa_ms": float(meta.get("ttfa_ms", -1.0)),
            "client_ttfa_ms": (first_chunk_t - t0) * 1000.0,
            "total_s": t1 - t0,
            "duration_s": duration_s if duration_s is not None else -1.0,
            "meta_missing": meta_missing,
        }
        if duration_s and duration_s > 0:
            row["rtf"] = (t1 - t0) / duration_s
        else:
            row["rtf"] = -1.0
        rows.append(row)
        print(f"[run] idx={idx} tag={tag} ttfa_ms={row['ttfa_ms']:.3f} rtf={row['rtf']:.3f}")

    fail_count = sum(1 for r in rows if r.get("meta_missing"))
    summary = {
        "count": args.count,
        "ttfa_ms": {
            "p50": _percentile([r["ttfa_ms"] for r in rows], 0.5),
            "p95": _percentile([r["ttfa_ms"] for r in rows], 0.95),
        },
        "code_ms": {
            "p50": _percentile([r["code_ms"] for r in rows], 0.5),
            "p95": _percentile([r["code_ms"] for r in rows], 0.95),
        },
        "decode_first_ms": {
            "p50": _percentile([r["decode_first_ms"] for r in rows], 0.5),
            "p95": _percentile([r["decode_first_ms"] for r in rows], 0.95),
        },
        "rtf": {
            "p50": _percentile([r["rtf"] for r in rows], 0.5),
            "p95": _percentile([r["rtf"] for r in rows], 0.95),
        },
        "hash_unique": len(set(r["sha256"] for r in rows if r.get("sha256"))),
        "fail_count": fail_count,
    }

    out = {"rows": rows, "summary": summary}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
