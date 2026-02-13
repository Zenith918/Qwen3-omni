#!/usr/bin/env python3
"""
P4.C4: End-to-end throughput benchmark for /tts/stream.

Runs short_01 and long_03 requests for a fixed duration,
measuring RTF_total (P50/P95), TTFA P95, and fallback rate.

Usage:
    python3 throughput_benchmark.py [--duration 30] [--server http://localhost:9000]
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import requests

TEXTS = {
    "short_01": "你好",
    "long_03": "如果系统在长句播放时出现越来越慢的情况，请先记录当时的时间戳、首包时间以及整体音频时长，并把日志打包。随后尝试相同文本重复三次，观察 RTF 是否呈线性上升，这能帮助我们定位是解码还是拼接导致的问题，同时也便于对比修复效果。",
}

SPEAKER = "serena"
SAMPLE_RATE = 24000  # Qwen3-TTS output sample rate


def run_single_request(server_url: str, text: str, speaker: str):
    """Run a single TTS stream request and measure timing."""
    url = f"{server_url}/tts/stream"
    payload = {"text": text, "speaker": speaker}

    t0 = time.perf_counter()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=60)
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                total_bytes += len(chunk)
                chunk_count += 1
    except Exception as e:
        return {"error": str(e)}

    t1 = time.perf_counter()

    if first_chunk_time is None:
        return {"error": "no_audio_data"}

    # Calculate metrics
    wall_time = t1 - t0
    ttfa = (first_chunk_time - t0)

    # Estimate audio duration from bytes (16-bit PCM, mono, 24kHz)
    # WAV header is 44 bytes
    audio_bytes = max(0, total_bytes - 44)
    audio_samples = audio_bytes // 2  # 16-bit = 2 bytes per sample
    audio_duration = audio_samples / SAMPLE_RATE

    rtf = wall_time / audio_duration if audio_duration > 0 else float('inf')

    return {
        "wall_time": wall_time,
        "ttfa": ttfa,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "total_bytes": total_bytes,
        "chunk_count": chunk_count,
    }


def run_throughput_benchmark(
    server_url: str,
    text_id: str,
    duration_s: float = 30.0,
    warmup_runs: int = 2,
):
    """Run throughput benchmark for a specific text."""
    text = TEXTS[text_id]

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        run_single_request(server_url, text, SPEAKER)

    # Benchmark
    print(f"  Running benchmark for {duration_s}s...")
    results = []
    t_start = time.perf_counter()

    while time.perf_counter() - t_start < duration_s:
        r = run_single_request(server_url, text, SPEAKER)
        if "error" not in r:
            results.append(r)
        else:
            print(f"    Error: {r['error']}")

    if not results:
        return {"error": "no_successful_requests"}

    rtfs = [r["rtf"] for r in results]
    ttfas = [r["ttfa"] * 1000 for r in results]  # Convert to ms

    summary = {
        "text_id": text_id,
        "num_requests": len(results),
        "duration_s": time.perf_counter() - t_start,
        "rtf_p50": float(np.percentile(rtfs, 50)),
        "rtf_p95": float(np.percentile(rtfs, 95)),
        "rtf_min": float(np.min(rtfs)),
        "rtf_max": float(np.max(rtfs)),
        "ttfa_p50_ms": float(np.percentile(ttfas, 50)),
        "ttfa_p95_ms": float(np.percentile(ttfas, 95)),
        "ttfa_max_ms": float(np.max(ttfas)),
        "audio_duration_avg": float(np.mean([r["audio_duration"] for r in results])),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="TTS Throughput Benchmark")
    parser.add_argument("--server", default="http://localhost:9000")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--texts", default="long_03,short_01")
    parser.add_argument("--label", default="default", help="Label for this config")
    args = parser.parse_args()

    text_ids = [t.strip() for t in args.texts.split(",")]

    print(f"=== Throughput Benchmark: {args.label} ===")
    print(f"Server: {args.server}")
    print(f"Duration: {args.duration}s per text")
    print(f"Texts: {text_ids}")
    print()

    all_results = {}
    for text_id in text_ids:
        if text_id not in TEXTS:
            print(f"  Skipping unknown text: {text_id}")
            continue
        print(f"[{text_id}]")
        summary = run_throughput_benchmark(
            args.server, text_id, args.duration, args.warmup
        )
        all_results[text_id] = summary
        if "error" not in summary:
            print(f"  Requests: {summary['num_requests']}")
            print(f"  RTF P50: {summary['rtf_p50']:.4f}")
            print(f"  RTF P95: {summary['rtf_p95']:.4f}")
            print(f"  TTFA P95: {summary['ttfa_p95_ms']:.1f}ms")
            print(f"  Audio avg: {summary['audio_duration_avg']:.2f}s")
        else:
            print(f"  Error: {summary['error']}")
        print()

    # Save results
    output = {
        "label": args.label,
        "server": args.server,
        "duration_s": args.duration,
        "results": all_results,
    }
    outfile = f"/tmp/throughput_{args.label}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {outfile}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Text':>10} | {'N':>4} | {'RTF P50':>8} | {'RTF P95':>8} | {'TTFA P95':>10}")
    print("-" * 60)
    for tid, r in all_results.items():
        if "error" not in r:
            print(f"{tid:>10} | {r['num_requests']:>4} | {r['rtf_p50']:>8.4f} | {r['rtf_p95']:>8.4f} | {r['ttfa_p95_ms']:>8.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()





