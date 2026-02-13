#!/usr/bin/env python3
"""
Day 2 — 音频输入 → Qwen3-Omni → 结构化输出

支持三种模式：
  --mode fast   : 只要 reply_text（最小延迟，用于首响路径）
  --mode slow   : 完整 {transcript, paralinguistic, reply_text}（用于日志/评测）
  --mode dual   : fast + slow 并发，fast 走首响，slow 异步落盘

支持流式输出：
  --stream 1    : 开启 Omni text streaming（配合 Bridge 分段喂 TTS）

用法：
  python3 demo_audio_to_omni.py --wav input.wav --mode fast --stream 1
  python3 demo_audio_to_omni.py --wav input.wav --mode dual --runs 3
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Generator

import numpy as np
import requests

# ── 配置 ──────────────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

PROMPT_FAST = (
    "听完这段音频后，直接给出一句简短自然的口语回复（<=20字）。"
    "只输出回复文字，不要任何分析、JSON 或多余内容。"
)

PROMPT_SLOW = (
    "你是一个语音分析助手。用户会给你一段音频，请你完成以下任务：\n"
    "1) 转写音频内容（transcript）\n"
    "2) 提取副语言线索（paralinguistic），包括但不限于：情绪(emotion)、语速(speed)、"
    "犹豫/停顿(hesitation)、语气强度(intensity)、语调(intonation)\n"
    "3) 给出一句简短自然回复（reply_text），不超过30字\n\n"
    "仅输出 JSON，不要包含其他文字。格式：\n"
    '{"transcript": "...", "paralinguistic": {"emotion": "...", "speed": "...", '
    '"hesitation": "...", "intensity": "...", "intonation": "..."}, "reply_text": "..."}'
)


# ── 音频工具 ──────────────────────────────────────────────────
def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 wav 文件，返回 (samples_float32, sample_rate)"""
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sampwidth == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def resample_to_16k(samples: np.ndarray, orig_rate: int) -> np.ndarray:
    if orig_rate == 16000:
        return samples
    target_len = int(len(samples) * 16000 / orig_rate)
    return np.interp(np.linspace(0, len(samples) - 1, target_len),
                     np.arange(len(samples)), samples)


def wav_to_base64(wav_path: str) -> tuple[str, float]:
    """读取 wav → 重采样 16kHz → base64，返回 (b64_str, duration_s)"""
    samples, rate = read_wav(wav_path)
    duration = len(samples) / rate
    samples_16k = resample_to_16k(samples, rate)
    pcm = (samples_16k * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii"), duration


def extract_json(text: str) -> dict | None:
    """从模型回复中提取第一个 JSON 对象"""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Omni 调用（非流式） ─────────────────────────────────────
def call_omni(audio_b64: str, mode: str = "fast") -> dict:
    """
    非流式调用 Omni。
    mode='fast' → 只要 reply_text
    mode='slow' → 完整 JSON
    返回 {content, parsed_json, timings}
    """
    prompt = PROMPT_FAST if mode == "fast" else PROMPT_SLOW
    max_tokens = 64 if mode == "fast" else 512

    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }

    t0 = time.time()
    resp = requests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    t1 = time.time()

    raw = resp.json()
    content = raw["choices"][0]["message"]["content"].strip()

    if mode == "fast":
        parsed = {"reply_text": content}
    else:
        parsed = extract_json(content)

    return {
        "raw_response": raw,
        "content": content,
        "parsed_json": parsed,
        "mode": mode,
        "timings": {
            "llm_latency_ms": round((t1 - t0) * 1000, 1),
        },
    }


# ── Omni 调用（流式） ───────────────────────────────────────
def stream_omni(audio_b64: str, mode: str = "fast") -> Generator[dict, None, None]:
    """
    流式调用 Omni，逐 token yield。
    每次 yield: {"token": str, "accumulated": str, "time_ms": float, "is_first": bool}
    最后一个 yield 带 "done": True 和 timings
    """
    prompt = PROMPT_FAST if mode == "fast" else PROMPT_SLOW
    max_tokens = 64 if mode == "fast" else 512

    payload = {
        "model": LLM_MODEL,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    }

    t0 = time.time()
    resp = requests.post(f"{LLM_BASE_URL}/v1/chat/completions",
                         json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    accumulated = ""
    first_token_time = None
    token_count = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            tok = delta["content"]
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            accumulated += tok
            token_count += 1
            yield {
                "token": tok,
                "accumulated": accumulated,
                "time_ms": round((now - t0) * 1000, 1),
                "is_first": token_count == 1,
            }

    t_end = time.time()
    yield {
        "done": True,
        "accumulated": accumulated,
        "token_count": token_count,
        "timings": {
            "ttft_ms": round((first_token_time - t0) * 1000, 1) if first_token_time else -1,
            "total_ms": round((t_end - t0) * 1000, 1),
            "llm_latency_ms": round((first_token_time - t0) * 1000, 1) if first_token_time else -1,
        },
    }


# ── 主程序 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Audio → Qwen3-Omni")
    parser.add_argument("--wav", required=True, help="Input wav file")
    parser.add_argument("--mode", default="fast", choices=["fast", "slow", "dual"])
    parser.add_argument("--stream", type=int, default=0, help="Enable Omni streaming (0/1)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.wav):
        print(f"ERROR: wav not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    # 预处理音频（一次编码，多次复用）
    audio_b64, wav_duration = wav_to_base64(args.wav)
    print(f"Input: {args.wav} ({wav_duration:.2f}s)")

    all_results = []
    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx+1}/{args.runs}  mode={args.mode}  stream={args.stream}")
        print(f"{'='*60}")

        if args.mode == "dual":
            # Fast + Slow 并发
            with ThreadPoolExecutor(max_workers=2) as pool:
                fast_fut = pool.submit(call_omni, audio_b64, "fast")
                slow_fut = pool.submit(call_omni, audio_b64, "slow")
                fast_res = fast_fut.result()
                slow_res = slow_fut.result()

            print(f"  [fast] {fast_res['timings']['llm_latency_ms']:.0f}ms → \"{fast_res['content'][:60]}\"")
            print(f"  [slow] {slow_res['timings']['llm_latency_ms']:.0f}ms → JSON parse {'✅' if slow_res['parsed_json'] else '❌'}")
            if slow_res["parsed_json"]:
                print(f"         transcript: {slow_res['parsed_json'].get('transcript', 'N/A')[:60]}")

            all_results.append({
                "run": run_idx,
                "fast": {"content": fast_res["content"], "timings": fast_res["timings"]},
                "slow": {"content": slow_res["content"], "parsed_json": slow_res["parsed_json"],
                          "timings": slow_res["timings"]},
            })

        elif args.stream:
            # 流式模式
            tokens = []
            final = None
            for item in stream_omni(audio_b64, args.mode):
                if item.get("done"):
                    final = item
                else:
                    tokens.append(item)
                    if item["is_first"]:
                        print(f"  TTFT: {item['time_ms']:.0f}ms  first=\"{item['token']}\"")

            text = final["accumulated"] if final else ""
            print(f"  Total: {final['timings']['total_ms']:.0f}ms  tokens={final['token_count']}")
            print(f"  Text: \"{text[:80]}\"")

            all_results.append({
                "run": run_idx,
                "mode": args.mode,
                "stream": True,
                "content": text,
                "timings": final["timings"] if final else {},
                "token_trace": [{"t_ms": t["time_ms"], "tok": t["token"]} for t in tokens],
            })

        else:
            # 非流式
            res = call_omni(audio_b64, args.mode)
            print(f"  Latency: {res['timings']['llm_latency_ms']:.0f}ms")
            print(f"  Text: \"{res['content'][:80]}\"")
            if args.mode == "slow" and res["parsed_json"]:
                print(f"  transcript: {res['parsed_json'].get('transcript', 'N/A')[:60]}")

            all_results.append({
                "run": run_idx,
                "mode": args.mode,
                "content": res["content"],
                "parsed_json": res.get("parsed_json"),
                "timings": res["timings"],
            })

    # ── 保存 ──
    if args.mode == "dual":
        # 保存 slow lane 结果
        slow_data = [r["slow"] for r in all_results if "slow" in r]
        with open(os.path.join(out_dir, "day2_slowlane.json"), "w", encoding="utf-8") as f:
            json.dump(slow_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Slow lane → {out_dir}/day2_slowlane.json")

    # ── 统计 ──
    if args.runs > 1:
        print(f"\n{'='*60}")
        print(f"  Stability ({args.runs} runs, mode={args.mode})")
        print(f"{'='*60}")
        if args.mode == "dual":
            fast_lats = [r["fast"]["timings"]["llm_latency_ms"] for r in all_results]
            slow_lats = [r["slow"]["timings"]["llm_latency_ms"] for r in all_results]
            print(f"  fast: min={min(fast_lats):.0f} p50={sorted(fast_lats)[len(fast_lats)//2]:.0f} max={max(fast_lats):.0f} ms")
            print(f"  slow: min={min(slow_lats):.0f} p50={sorted(slow_lats)[len(slow_lats)//2]:.0f} max={max(slow_lats):.0f} ms")
        else:
            key = "ttft_ms" if args.stream else "llm_latency_ms"
            lats = [r["timings"][key] for r in all_results if key in r.get("timings", {})]
            if lats:
                lats_sorted = sorted(lats)
                p50 = lats_sorted[len(lats)//2]
                p95_idx = min(int(len(lats) * 0.95), len(lats)-1)
                p95 = lats_sorted[p95_idx]
                print(f"  {key}: min={min(lats):.0f} p50={p50:.0f} p95={p95:.0f} max={max(lats):.0f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
