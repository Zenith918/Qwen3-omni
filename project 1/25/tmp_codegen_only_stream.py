#!/usr/bin/env python3
import json
import os
import sys
import time
from threading import Event

import torch


def _load_text(texts_path: str, text_id: str) -> str:
    with open(texts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("texts", []):
        if item.get("id") == text_id:
            return item.get("text", "")
    raise ValueError(f"text_id not found: {text_id}")


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def _run_once(ts, req):
    cancel_event = Event()
    frames: list[torch.Tensor] = []
    t0 = time.time()
    t_first = None
    for codes in ts._iter_deep_codes(req, cancel_event):
        if not isinstance(codes, torch.Tensor):
            continue
        if t_first is None:
            t_first = time.time()
        codes = ts._normalize_codes_tensor(codes.to(torch.long))
        for frame in codes:
            frames.append(frame.detach().cpu())
    t1 = time.time()
    if t_first is None:
        t_first = t1
    if not frames:
        raise RuntimeError("No audio codes produced")
    codes_tensor = torch.stack(frames, dim=0)
    return {
        "first_packet_ms": (t_first - t0) * 1000.0,
        "total_ms": (t1 - t0) * 1000.0,
        "frames": int(codes_tensor.shape[0]),
        "sha256": _sha256_tensor(codes_tensor),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-id", default="long_03")
    parser.add_argument("--texts", default="/workspace/project 1/25/clients/texts_p0_base.json")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    os.environ.setdefault("TTS_DEEP_STREAM_ENABLE", "1")
    os.environ.setdefault("TTS_DEEP_STREAM_PROCESS", "0")
    os.environ.setdefault("TTS_DEEP_STREAM_DEVICE", "cuda:0")
    os.environ.setdefault("TTS_DEEP_STREAM_CODEGEN_DEVICE", "cuda:0")
    os.environ.setdefault("TTS_DEEP_STREAM_MODEL_DIR", "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    os.environ.setdefault("TTS_DEEP_STREAM_PACKET_TOKENS", "1")
    os.environ.setdefault("TTS_DEEP_STREAM_DETERMINISTIC", "1")
    os.environ.setdefault("TTS_DEEP_STREAM_DETERMINISTIC_POLICY", "greedy")
    os.environ.setdefault("TTS_DEEP_STREAM_SEED_MODE", "fixed")
    os.environ.setdefault("TTS_DEEP_STREAM_SEED", "42")

    sys.path.append("/workspace/project 1/25/clients")
    import tts_server as ts

    ts._init_deep_stream_backend()
    text = _load_text(args.texts, args.text_id)
    req = ts.TTSRequest(
        text=text,
        task_type="Base",
        language="Chinese",
        speaker="serena",
        instruct="",
        max_new_tokens=2048,
        non_streaming_mode=False,
    )

    rows = []
    for idx in range(args.count):
        row = {"idx": idx}
        row.update(_run_once(ts, req))
        rows.append(row)
        print(
            f"[codegen_only] idx={idx} first_packet_ms={row['first_packet_ms']:.3f} "
            f"total_ms={row['total_ms']:.3f} frames={row['frames']} sha256={row['sha256']}"
        )

    summary = {
        "count": args.count,
        "hash_unique": len(set(r["sha256"] for r in rows if r.get("sha256"))),
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
