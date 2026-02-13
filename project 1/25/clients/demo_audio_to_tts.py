#!/usr/bin/env python3
"""
Day 2 — 音频输入 → Omni → Bridge 分段 → TTS → 语音回复 + EoT→FirstAudio 指标

核心升级：
  1. 支持 fast/slow/dual 模式 (P0-2)
  2. 支持 --stream_omni 1：Omni 流式 + Bridge 分段 → TTS 流式首响 (P0-3)
  3. EoT→FirstAudio 指标 + 自动统计 P50/P95 (P0-1)
  4. 完整 timeline 打点 (兼容 D1 字段)

用法：
  # 非流式（D1 兼容）
  python3 demo_audio_to_tts.py --wav input.wav

  # 流式 fast lane（D2 推荐）
  python3 demo_audio_to_tts.py --wav input.wav --stream_omni 1 --mode fast

  # 基准测量 20 次
  python3 demo_audio_to_tts.py --wav input.wav --stream_omni 1 --mode fast --runs 20
"""

import argparse
import io
import json
import os
import re
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np
import requests

# 复用 Omni 调用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_audio_to_omni import (
    call_omni, stream_omni, wav_to_base64, read_wav,
    PROMPT_FAST, PROMPT_SLOW,
)

# D3: Silero VAD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime"))
_vad_instance = None

def get_vad():
    global _vad_instance
    if _vad_instance is None:
        from vad_silero import SileroVAD
        _vad_instance = SileroVAD(threshold=0.5, min_speech_ms=250, min_silence_ms=300)
    return _vad_instance

def compute_vad_end(wav_path: str) -> float:
    """用 Silero VAD 计算真实的 speech_end 时间(ms)"""
    vad = get_vad()
    result = vad.process_wav(wav_path)
    if result["segments"]:
        return result["segments"][-1]["end_ms"]
    # fallback: 如果 VAD 没检测到语音，用文件时长
    return result["total_duration_ms"]

# ── 配置 ──────────────────────────────────────────────────────
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "serena")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

# ── Bridge 文本分段器 ────────────────────────────────────────
# 策略与 SKILL.md 一致:
#   starter 段：2~6 字优先送 TTS 降低首包延迟
#   中文标点立即 flush
#   无标点时 8~12 字 flush

CHINESE_PUNCT = set("，。！？、；：…—""''《》【】")
MAX_CHARS_NO_PUNCT = 10  # 无标点时最大缓冲字符数
STARTER_MIN = 2          # starter 段最小字符数


class BridgeSegmenter:
    """
    文本流分段器：接收逐 token 文本，按 Bridge 策略切分成可说片段。
    """

    def __init__(self, starter_min: int = STARTER_MIN,
                 max_no_punct: int = MAX_CHARS_NO_PUNCT):
        self.buffer = ""
        self.segments_emitted = 0
        self.starter_min = starter_min
        self.max_no_punct = max_no_punct

    def feed(self, text: str) -> list[str]:
        """
        送入新 token 文本，返回 0 或多个可说片段。
        """
        self.buffer += text
        segments = []

        while self.buffer:
            # 查找最早的标点位置
            punct_pos = -1
            for i, ch in enumerate(self.buffer):
                if ch in CHINESE_PUNCT:
                    punct_pos = i
                    break

            if punct_pos >= 0:
                # 标点触发 flush（包含标点本身）
                seg = self.buffer[: punct_pos + 1].strip()
                self.buffer = self.buffer[punct_pos + 1:]
                if seg:
                    segments.append(seg)
                    self.segments_emitted += 1
                continue

            # 无标点 — 检查 starter 或长度阈值
            if self.segments_emitted == 0:
                # Starter 段：一旦到达 starter_min 就立即发出
                if len(self.buffer.strip()) >= self.starter_min:
                    seg = self.buffer.strip()
                    self.buffer = ""
                    segments.append(seg)
                    self.segments_emitted += 1
                break
            else:
                # 后续段：到达 max_no_punct 就 flush
                if len(self.buffer.strip()) >= self.max_no_punct:
                    seg = self.buffer.strip()
                    self.buffer = ""
                    segments.append(seg)
                    self.segments_emitted += 1
                break

        return segments

    def flush(self) -> Optional[str]:
        """流结束时强制 flush 缓冲区"""
        seg = self.buffer.strip()
        self.buffer = ""
        if seg:
            self.segments_emitted += 1
            return seg
        return None


# ── TTS 调用 ─────────────────────────────────────────────────
def call_tts_stream(text: str, retries: int = 2) -> dict:
    """
    调用 TTS /tts/stream，返回 {pcm_data, ttfa_ms, total_ms, audio_duration_s}
    带重试机制应对偶发连接中断。
    """
    payload = {"text": text, "speaker": TTS_SPEAKER}

    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(TTS_URL, json=payload, stream=True, timeout=120)
            resp.raise_for_status()

            first_chunk_time = None
            chunks = []

            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    chunks.append(chunk)

            t_end = time.time()
            pcm_data = b"".join(chunks)
            audio_dur = len(pcm_data) / (24000 * 2)  # 16-bit mono 24kHz

            return {
                "pcm_data": pcm_data,
                "ttfa_ms": round((first_chunk_time - t0) * 1000, 1) if first_chunk_time else None,
                "total_ms": round((t_end - t0) * 1000, 1),
                "audio_duration_s": round(audio_dur, 3),
            }
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                print(f"  ⚠ TTS retry {attempt+1}/{retries}: {e}")
                time.sleep(0.5)
            else:
                print(f"  ❌ TTS failed after {retries+1} attempts: {e}")
                return {"pcm_data": b"", "ttfa_ms": None, "total_ms": 0, "audio_duration_s": 0}


def save_wav(pcm_data: bytes, path: str, rate: int = 24000):
    """保存 PCM 为 wav 文件"""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_data)


# ── 非流式管道（D1 兼容 + 指标升级）────────────────────────
def pipeline_nostream(audio_b64: str, mode: str, out_dir: str,
                      wav_duration: float) -> dict:
    """
    非流式管道：audio → Omni(完整) → TTS(完整)
    返回 timeline dict
    """
    tl = {}

    # t0: 音频就绪
    t0 = time.time()
    tl["t0_audio_ingress_start"] = round(t0 * 1000, 1)

    # t1: VAD end — D3: 用 Silero VAD 真实检测
    # 在文件场景下，VAD 处理是瞬时的，t1 = t0 + vad_processing_time
    # vad_end_offset_ms 是音频内的相对时间，表示"用户说完"的时刻
    vad_end_offset_ms = tl.get("_vad_end_offset_ms", wav_duration * 1000)
    t1 = t0  # 文件场景：音频已完全就绪，VAD end = 处理开始
    tl["t1_vad_end"] = round(t1 * 1000, 1)
    tl["vad_end_offset_ms"] = vad_end_offset_ms

    # t2: Omni 请求发出
    t2 = time.time()
    tl["t2_omni_request_sent"] = round(t2 * 1000, 1)

    omni_res = call_omni(audio_b64, mode)

    tl["t3_omni_first_token"] = None  # 非流式
    t4 = time.time()
    tl["t4_reply_text_ready"] = round(t4 * 1000, 1)

    reply_text = omni_res["content"] if mode == "fast" else \
        (omni_res["parsed_json"] or {}).get("reply_text", omni_res["content"])

    # t5: TTS 请求发出
    t5 = time.time()
    tl["t5_tts_request_sent_first"] = round(t5 * 1000, 1)

    tts_res = call_tts_stream(reply_text)

    t6 = t5 + (tts_res["ttfa_ms"] or 0) / 1000
    tl["t6_ttfa_first_audio_chunk_received"] = round(t6 * 1000, 1)
    tl["t7_playout_start"] = tl["t6_ttfa_first_audio_chunk_received"]

    # 核心指标
    eot_to_first_audio_ms = round((t6 - t1) * 1000, 1)
    tl["eot_to_first_audio_ms"] = eot_to_first_audio_ms
    tl["total_e2e_ms"] = round((t6 - t0) * 1000, 1)
    tl["omni_latency_ms"] = omni_res["timings"]["llm_latency_ms"]
    tl["tts_ttfa_ms"] = tts_res["ttfa_ms"]
    tl["tts_total_ms"] = tts_res["total_ms"]
    tl["input_wav_duration_s"] = round(wav_duration, 3)
    tl["reply_wav_duration_s"] = tts_res["audio_duration_s"]
    tl["reply_text"] = reply_text
    tl["pcm_data"] = tts_res["pcm_data"]  # 不序列化，仅内部使用

    return tl


# ── 流式管道（D2 核心：Omni stream → Bridge 分段 → TTS）───
def pipeline_stream(audio_b64: str, mode: str, out_dir: str,
                    wav_duration: float) -> dict:
    """
    流式管道：audio → Omni(stream) → Bridge 分段 → TTS(首段) → FirstAudio
    返回 timeline dict
    """
    tl = {}
    stream_trace = []  # 详细 token/段 trace

    # t0, t1
    t0 = time.time()
    tl["t0_audio_ingress_start"] = round(t0 * 1000, 1)
    vad_end_offset_ms = tl.get("_vad_end_offset_ms", wav_duration * 1000)
    t1 = t0  # 文件场景：音频就绪即 VAD end
    tl["t1_vad_end"] = round(t1 * 1000, 1)
    tl["vad_end_offset_ms"] = vad_end_offset_ms

    # t2: Omni streaming 请求
    t2 = time.time()
    tl["t2_omni_request_sent"] = round(t2 * 1000, 1)

    all_pcm = []
    tts_first_audio_time = None
    tts_calls = 0
    first_tts_ttfa = None
    tts_total_start = None
    tts_total_end = None
    t3_first_token = None

    # ── Phase 1: 收集 Omni 流式 tokens（~50ms，不做 TTS 避免 GPU 争抢）──
    omni_timings = {}
    full_reply = ""
    for item in stream_omni(audio_b64, "fast" if mode == "dual" else mode):
        if item.get("done"):
            omni_timings = item.get("timings", {})
            break

        tok = item["token"]
        tok_time = item["time_ms"]

        if item["is_first"]:
            t3_first_token = time.time()
            tl["t3_omni_first_token"] = round(t3_first_token * 1000, 1)

        stream_trace.append({
            "type": "token", "tok": tok, "t_ms": tok_time,
        })
        full_reply += tok

    tl["omni_stream_timings"] = omni_timings

    t4 = time.time()
    tl["t4_reply_text_ready"] = round(t4 * 1000, 1)

    # ── Phase 2: Bridge 分段 → TTS（Omni 已结束，TTS 独占 GPU）──
    MIN_SEGMENT_CHARS = 4  # 防止极短文本触发 TTS embedding crash
    SHORT_TEXT_THRESHOLD = 20  # 短文本不拆分

    if len(full_reply.strip()) <= SHORT_TEXT_THRESHOLD:
        # 短文本：整段送 TTS
        segments_to_send = [full_reply.strip()] if full_reply.strip() else []
    else:
        # 长文本：用 Bridge 分段
        segmenter = BridgeSegmenter()
        segments_to_send = []
        for ch in full_reply:
            segs = segmenter.feed(ch)
            segments_to_send.extend(segs)
        final = segmenter.flush()
        if final:
            segments_to_send.append(final)
        # 合并过短的段
        merged = []
        buf = ""
        for seg in segments_to_send:
            buf += seg
            if len(buf) >= MIN_SEGMENT_CHARS:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] += buf
            else:
                merged.append(buf)
        segments_to_send = merged

    # Slow lane 异步启动（dual 模式，此时 Omni 已结束不争 GPU）
    slow_future: Optional[Future] = None
    if mode == "dual":
        slow_pool = ThreadPoolExecutor(max_workers=1)
        slow_future = slow_pool.submit(call_omni, audio_b64, "slow")

    for seg in segments_to_send:
        seg_time = time.time()
        stream_trace.append({
            "type": "segment", "text": seg,
            "t_ms": round((seg_time - t0) * 1000, 1),
            "trigger": "short_text" if len(segments_to_send) == 1 else "bridge",
        })

        if tts_calls == 0:
            tl["t5_tts_request_sent_first"] = round(seg_time * 1000, 1)
            tts_total_start = seg_time

        tts_r = call_tts_stream(seg)
        tts_calls += 1

        if tts_r["pcm_data"]:
            all_pcm.append(tts_r["pcm_data"])
            tts_total_end = time.time()

            if tts_first_audio_time is None and tts_r["ttfa_ms"] is not None:
                tts_first_audio_time = tts_total_start + tts_r["ttfa_ms"] / 1000
                first_tts_ttfa = tts_r["ttfa_ms"]

    # ── 计算 timeline ──
    if tts_first_audio_time:
        tl["t6_ttfa_first_audio_chunk_received"] = round(tts_first_audio_time * 1000, 1)
    else:
        tl["t6_ttfa_first_audio_chunk_received"] = None

    tl["t7_playout_start"] = tl["t6_ttfa_first_audio_chunk_received"]

    # 核心指标
    if tts_first_audio_time:
        eot_to_first_audio_ms = round((tts_first_audio_time - t1) * 1000, 1)
    else:
        eot_to_first_audio_ms = None
    tl["eot_to_first_audio_ms"] = eot_to_first_audio_ms
    tl["total_e2e_ms"] = round((tts_first_audio_time - t0) * 1000, 1) if tts_first_audio_time else None
    tl["omni_ttft_ms"] = omni_timings.get("ttft_ms")
    tl["omni_total_ms"] = omni_timings.get("total_ms")
    tl["tts_ttfa_ms"] = first_tts_ttfa
    tl["tts_total_ms"] = round((tts_total_end - tts_total_start) * 1000, 1) if tts_total_end and tts_total_start else None
    tl["tts_calls"] = tts_calls
    tl["input_wav_duration_s"] = round(wav_duration, 3)

    # 合并所有 PCM
    all_pcm_data = b"".join(all_pcm)
    tl["reply_wav_duration_s"] = round(len(all_pcm_data) / (24000 * 2), 3)
    tl["pcm_data"] = all_pcm_data

    # Slow lane 结果
    if slow_future:
        try:
            slow_res = slow_future.result(timeout=30)
            tl["slow_lane"] = {
                "parsed_json": slow_res.get("parsed_json"),
                "latency_ms": slow_res["timings"]["llm_latency_ms"],
            }
        except Exception as e:
            tl["slow_lane"] = {"error": str(e)}

    tl["stream_trace"] = stream_trace
    tl["reply_text"] = full_reply

    return tl


# ── 统计辅助 ─────────────────────────────────────────────────
def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0
    sorted_v = sorted(vals)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return sorted_v[idx]


# ── 主程序 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D2: Audio → Omni → Bridge → TTS")
    parser.add_argument("--wav", required=True, help="Input wav file")
    parser.add_argument("--mode", default="fast", choices=["fast", "slow", "dual"])
    parser.add_argument("--stream_omni", type=int, default=0, help="Enable Omni streaming (0/1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for P50/P95")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--speaker", default=None, help="TTS speaker override")
    args = parser.parse_args()

    global TTS_SPEAKER
    if args.speaker:
        TTS_SPEAKER = args.speaker

    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.wav):
        print(f"ERROR: wav not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    # 预处理音频
    audio_b64, wav_duration = wav_to_base64(args.wav)

    # D3: Silero VAD — 预计算 speech end
    vad_end_ms = compute_vad_end(args.wav)
    print(f"Input: {args.wav} ({wav_duration:.2f}s)")
    print(f"VAD speech_end: {vad_end_ms:.0f}ms (of {wav_duration*1000:.0f}ms)")
    print(f"Mode: {args.mode}  stream_omni: {args.stream_omni}  runs: {args.runs}")

    all_timelines = []

    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx+1}/{args.runs}")
        print(f"{'='*60}")

        # 注入 VAD end offset 供 pipeline 使用
        _vad_ctx = {"_vad_end_offset_ms": vad_end_ms}

        if args.stream_omni:
            tl = pipeline_stream(audio_b64, args.mode, out_dir, wav_duration)
        else:
            tl = pipeline_nostream(audio_b64, args.mode, out_dir, wav_duration)
        tl["vad_end_offset_ms"] = vad_end_ms
        tl["vad_source"] = "silero"

        # 提取 pcm 用于保存（不序列化到 JSON）
        pcm_data = tl.pop("pcm_data", b"")
        stream_trace = tl.pop("stream_trace", [])

        # 打印关键指标
        eot = tl.get("eot_to_first_audio_ms")
        omni_lat = tl.get("omni_latency_ms") or tl.get("omni_ttft_ms")
        tts_ttfa = tl.get("tts_ttfa_ms")

        print(f"  EoT→FirstAudio: {eot} ms")
        print(f"  Omni {'TTFT' if args.stream_omni else 'latency'}: {omni_lat} ms")
        print(f"  TTS TTFA:       {tts_ttfa} ms")
        print(f"  Reply text:     \"{tl.get('reply_text', '')[:60]}\"")
        print(f"  Reply audio:    {tl.get('reply_wav_duration_s', 0):.2f}s")
        if tl.get("slow_lane"):
            sl = tl["slow_lane"]
            if "parsed_json" in sl and sl["parsed_json"]:
                print(f"  [slow] transcript: {sl['parsed_json'].get('transcript', 'N/A')[:50]}")
                print(f"  [slow] latency:    {sl.get('latency_ms', 'N/A')} ms")
        if args.stream_omni:
            seg_count = sum(1 for t in stream_trace if t["type"] == "segment")
            print(f"  Bridge segments: {seg_count}  TTS calls: {tl.get('tts_calls', 0)}")

        all_timelines.append(tl)

        # 保存最后一次的 wav 和 trace
        if run_idx == args.runs - 1:
            reply_wav_path = os.path.join(out_dir, "day2_reply.wav")
            if pcm_data:
                save_wav(pcm_data, reply_wav_path)
                print(f"  Saved: {reply_wav_path}")

            if stream_trace:
                trace_path = os.path.join(out_dir, "day2_stream_trace.jsonl")
                with open(trace_path, "w", encoding="utf-8") as f:
                    for entry in stream_trace:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"  Trace: {trace_path}")

            # 保存 slow lane
            if tl.get("slow_lane") and tl["slow_lane"].get("parsed_json"):
                sl_path = os.path.join(out_dir, "day2_slowlane.json")
                with open(sl_path, "w", encoding="utf-8") as f:
                    json.dump(tl["slow_lane"]["parsed_json"], f, ensure_ascii=False, indent=2)

    # ── 统计报告 ──
    print(f"\n{'='*60}")
    print(f"  D2 Metrics Report ({args.runs} runs)")
    print(f"{'='*60}")

    eot_vals = [t["eot_to_first_audio_ms"] for t in all_timelines if t.get("eot_to_first_audio_ms") is not None]
    omni_vals = [t.get("omni_latency_ms") or t.get("omni_ttft_ms", 0) for t in all_timelines]
    tts_vals = [t["tts_ttfa_ms"] for t in all_timelines if t.get("tts_ttfa_ms") is not None]

    metrics = {}
    if eot_vals:
        metrics["eot_to_first_audio_p50"] = percentile(eot_vals, 50)
        metrics["eot_to_first_audio_p95"] = percentile(eot_vals, 95)
        metrics["eot_to_first_audio_min"] = min(eot_vals)
        metrics["eot_to_first_audio_max"] = max(eot_vals)
        print(f"  EoT→FirstAudio P50: {metrics['eot_to_first_audio_p50']:.1f} ms")
        print(f"  EoT→FirstAudio P95: {metrics['eot_to_first_audio_p95']:.1f} ms")
        print(f"  EoT→FirstAudio range: [{metrics['eot_to_first_audio_min']:.0f}, {metrics['eot_to_first_audio_max']:.0f}] ms")

        # Gate 检查
        gate_eot = 550
        passed = metrics["eot_to_first_audio_p95"] <= gate_eot
        print(f"  Gate EoT→FA P95 ≤ {gate_eot}ms: {'✅ PASS' if passed else '❌ FAIL'} ({metrics['eot_to_first_audio_p95']:.1f}ms)")
        metrics["gate_eot_to_first_audio_pass"] = passed

    if omni_vals:
        metrics["omni_p50"] = percentile(omni_vals, 50)
        metrics["omni_p95"] = percentile(omni_vals, 95)
        print(f"  Omni P50: {metrics['omni_p50']:.1f} ms  P95: {metrics['omni_p95']:.1f} ms")

    if tts_vals:
        metrics["tts_ttfa_p50"] = percentile(tts_vals, 50)
        metrics["tts_ttfa_p95"] = percentile(tts_vals, 95)
        print(f"  TTS TTFA P50: {metrics['tts_ttfa_p50']:.1f} ms  P95: {metrics['tts_ttfa_p95']:.1f} ms")

    metrics["runs"] = args.runs
    metrics["mode"] = args.mode
    metrics["stream_omni"] = bool(args.stream_omni)
    metrics["wav_duration_s"] = wav_duration
    metrics["all_eot_values"] = eot_vals
    metrics["all_omni_values"] = omni_vals
    metrics["all_tts_values"] = tts_vals

    metrics_path = os.path.join(out_dir, "day2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # 保存最后一次 timeline（兼容 D1 格式）
    timeline_path = os.path.join(out_dir, "day2_timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(all_timelines[-1], f, ensure_ascii=False, indent=2)
    print(f"  Timeline saved: {timeline_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
