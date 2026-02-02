#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import torch


def _load_codes(path: str) -> torch.Tensor:
    codes = torch.load(path, map_location="cpu")
    if isinstance(codes, torch.Tensor):
        return codes
    return torch.as_tensor(codes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dump-dir", default="/workspace/project 1/25/output/code_dumps")
    parser.add_argument("--mode", default="cache")
    args = parser.parse_args()

    os.environ.setdefault("TTS_DEEP_STREAM_ENABLE", "1")
    os.environ.setdefault("TTS_DEEP_STREAM_PROCESS", "0")
    os.environ.setdefault("TTS_DEEP_STREAM_DEVICE", "cuda:0")
    os.environ.setdefault("TTS_DEEP_STREAM_MODEL_DIR", "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    sys.path.append("/workspace/project 1/25/clients")
    import tts_server as ts
    from tts_incremental_decoder import IncrementalDecoder

    ts._init_deep_stream_backend()
    tokenizer = ts._deep_tokenizer
    if tokenizer is None:
        raise RuntimeError("tokenizer not initialized")

    mode = args.mode.strip().lower()
    if mode not in ("cache", "window", "full"):
        mode = "cache"
    decoder = IncrementalDecoder(tokenizer, device=os.environ["TTS_DEEP_STREAM_DEVICE"], transformer_mode=mode)
    state = decoder.reset_state()

    path = os.path.join(args.dump_dir, f"codes_{args.tag}.pt")
    codes = _load_codes(path)
    if codes.dim() == 3 and codes.shape[0] == 1:
        codes = codes.squeeze(0)
    if codes.dim() != 2:
        raise ValueError(f"unexpected codes shape: {tuple(codes.shape)}")

    try:
        upsample = int(tokenizer.get_decode_upsample_rate())
    except Exception:
        upsample = -1

    samples_total = 0
    samples_max = 0
    samples_min = None
    decode_times_ms = []
    zero_calls = 0
    bad_len_calls = 0

    for idx in range(codes.shape[0]):
        frame = codes[idx : idx + 1]
        t0 = time.time()
        audio_np, state = decoder.decode_incremental(frame, state)
        t1 = time.time()
        decode_times_ms.append((t1 - t0) * 1000.0)
        n = int(audio_np.size)
        samples_total += n
        samples_max = max(samples_max, n)
        samples_min = n if samples_min is None else min(samples_min, n)
        if n == 0:
            zero_calls += 1
        if upsample > 0 and n != upsample:
            bad_len_calls += 1

    expected_samples = codes.shape[0] * upsample if upsample > 0 else -1
    result = {
        "tag": args.tag,
        "frames": int(codes.shape[0]),
        "upsample": upsample,
        "samples_total": samples_total,
        "samples_expected": expected_samples,
        "samples_max": samples_max,
        "samples_min": samples_min if samples_min is not None else 0,
        "zero_calls": zero_calls,
        "bad_len_calls": bad_len_calls,
        "decode_ms_sum": sum(decode_times_ms),
        "decode_ms_p50": float(sorted(decode_times_ms)[len(decode_times_ms) // 2]) if decode_times_ms else -1.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
