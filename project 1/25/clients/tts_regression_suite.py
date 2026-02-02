#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import time
import wave
from datetime import datetime

import numpy as np
import requests
import soundfile as sf

DEFAULT_TEXTS_PATH = os.path.join(os.path.dirname(__file__), "texts_p0_base.json")
DEFAULT_OUT_ROOT = "/workspace/project 1/25/output/regression"
DEFAULT_STREAM_URL = os.environ.get("TTS_STREAM_URL", "http://127.0.0.1:9000/tts/stream")
DEFAULT_OFFLINE_URL = os.environ.get("TTS_OFFLINE_URL", "http://127.0.0.1:9000/synthesize")
DEFAULT_STREAM_TIMEOUT_S = float(os.environ.get("TTS_STREAM_TIMEOUT_S", os.environ.get("TTS_TIMEOUT_S", "120")))
DEFAULT_OFFLINE_TIMEOUT_S = float(os.environ.get("TTS_OFFLINE_TIMEOUT_S", "600"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "2048"))
DEFAULT_ALIGN_MAX_MS = int(os.environ.get("TTS_ALIGN_MAX_MS", "500"))
DEFAULT_POP_FRAME_MS = int(os.environ.get("TTS_POP_FRAME_MS", "20"))
DEFAULT_RMS_CLIP = float(os.environ.get("TTS_RMS_CLIP_THRESHOLD", "0.4"))
EXPECT_DEEP_STREAM = os.environ.get("TTS_EXPECT_DEEP_STREAM", "1").lower() in ("1", "true", "yes")
EXPECTED_PACKET_TOKENS = int(os.environ.get("TTS_EXPECT_PACKET_TOKENS", "4"))
WARMUP_RUNS = int(os.environ.get("TTS_WARMUP_RUNS", "2"))
BASELINE_PATH = os.environ.get("TTS_REGRESSION_BASELINE", "")
REPORT_PATH = os.environ.get("TTS_REPORT_PATH", "/workspace/project 1/25/REPORT.md")
LONG_RUN_SECONDS = float(os.environ.get("TTS_LONG_RUN_SECONDS", "0"))


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


def load_texts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("texts") if isinstance(data, dict) else data
    texts = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            texts.append({"id": f"text_{idx + 1:02d}", "text": item})
        else:
            text_id = item.get("id") or f"text_{idx + 1:02d}"
            texts.append({"id": text_id, "text": item.get("text", ""), "category": item.get("category", "")})
    return texts


def load_voices(voices_path) -> list[dict]:
    if voices_path:
        with open(voices_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("voices") if isinstance(data, dict) else data
    env_json = os.environ.get("TTS_VOICES_JSON", "").strip()
    if env_json:
        data = json.loads(env_json)
        return data.get("voices") if isinstance(data, dict) else data
    return [
        {"id": "base", "task_type": "Base", "speaker": "serena", "language": "Chinese", "instruct": ""},
    ]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    if audio.size == 0:
        return audio
    ratio = sr_out / float(sr_in)
    n_out = max(1, int(round(len(audio) * ratio)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def estimate_offset(ref: np.ndarray, target: np.ndarray, sr: int, max_offset_ms: int) -> int:
    if ref.size == 0 or target.size == 0:
        return 0
    max_offset = int(sr * max_offset_ms / 1000)
    if max_offset <= 0:
        return 0
    window = min(len(ref), len(target), int(sr * 1.0))
    if window <= 0:
        return 0
    step = max(1, window // 4000)
    ref_seg = ref[:window:step]
    tgt_seg = target[:window:step]
    if ref_seg.size == 0 or tgt_seg.size == 0:
        return 0
    ref_seg = ref_seg - np.mean(ref_seg)
    tgt_seg = tgt_seg - np.mean(tgt_seg)
    corr = np.correlate(tgt_seg, ref_seg, mode="full")
    max_offset_ds = int(max_offset / step)
    center = len(ref_seg) - 1
    start = max(0, center - max_offset_ds)
    end = min(len(corr), center + max_offset_ds + 1)
    if start >= end:
        return 0
    lag_ds = int(np.argmax(corr[start:end]) + start - center)
    return lag_ds * step


def align_signals(ref: np.ndarray, target: np.ndarray, sr: int, max_offset_ms: int) -> tuple[np.ndarray, np.ndarray, int]:
    lag = estimate_offset(ref, target, sr, max_offset_ms)
    if lag > 0:
        target_aligned = target[lag:]
        ref_aligned = ref[: len(target_aligned)]
    elif lag < 0:
        ref_aligned = ref[-lag:]
        target_aligned = target[: len(ref_aligned)]
    else:
        min_len = min(len(ref), len(target))
        ref_aligned = ref[:min_len]
        target_aligned = target[:min_len]
    min_len = min(len(ref_aligned), len(target_aligned))
    return ref_aligned[:min_len], target_aligned[:min_len], lag


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    if ref.size == 0 or test.size == 0:
        return -1.0
    noise = test - ref
    sig_pow = float(np.mean(ref ** 2))
    noise_pow = float(np.mean(noise ** 2))
    if noise_pow <= 1e-12:
        return 120.0
    if sig_pow <= 1e-12:
        return -1.0
    return 10.0 * np.log10(sig_pow / noise_pow)


def pop_click_score(audio: np.ndarray, sr: int, frame_ms: int) -> float:
    if audio.size == 0:
        return -1.0
    frame = max(1, int(sr * frame_ms / 1000))
    if len(audio) < frame * 2:
        return 0.0
    rms = []
    for idx in range(0, len(audio) - frame + 1, frame):
        frame_audio = audio[idx : idx + frame]
        rms.append(np.sqrt(np.mean(frame_audio ** 2)))
    if len(rms) < 2:
        return 0.0
    diffs = np.abs(np.diff(np.array(rms, dtype=np.float32)))
    return float(np.percentile(diffs, 95))


def audio_health(audio: np.ndarray, rms_clip: float) -> tuple[bool, dict]:
    if audio.size == 0:
        return False, {"rms": 0.0, "peak": 0.0, "reason": "empty"}
    if not np.isfinite(audio).all():
        return False, {"rms": 0.0, "peak": 0.0, "reason": "nan"}
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-5:
        return False, {"rms": rms, "peak": peak, "reason": "all_zero"}
    if rms > rms_clip or peak >= 0.99:
        return False, {"rms": rms, "peak": peak, "reason": "clipping"}
    return True, {"rms": rms, "peak": peak, "reason": "ok"}


def _parse_float_header(headers: dict, key: str) -> float:
    raw = headers.get(key)
    if raw is None:
        return -1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return -1.0


def _parse_int_header(headers: dict, key: str) -> int:
    raw = headers.get(key)
    if raw is None:
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def run_offline(offline_url: str, payload: dict, out_path: str) -> tuple[np.ndarray, int, float]:
    t_start = time.time()
    resp = requests.post(offline_url, json=payload, timeout=(10, DEFAULT_OFFLINE_TIMEOUT_S))
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    audio, sr = sf.read(out_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    t_total = time.time() - t_start
    return audio.astype(np.float32), int(sr), t_total


def run_stream(stream_url: str, payload: dict, out_path: str) -> dict:
    t_start = time.time()
    first_chunk = None
    pcm_chunks: list[bytes] = []
    headers = {}
    with requests.post(stream_url, json=payload, stream=True, timeout=(10, DEFAULT_STREAM_TIMEOUT_S)) as resp:
        resp.raise_for_status()
        headers = dict(resp.headers)
        headers_lower = {k.lower(): v for k, v in headers.items()}
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
                pcm_chunks.append(chunk)
                wf.writeframes(chunk)
    t_end = time.time()
    pcm = b"".join(pcm_chunks)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    dur_s = len(audio) / sr if sr > 0 else 0.0
    ttfa_ms = (first_chunk - t_start) * 1000.0 if first_chunk else -1.0
    rtf = (t_end - t_start) / dur_s if dur_s > 0 else -1.0
    return {
        "audio": audio,
        "sr": sr,
        "ttfa_ms": ttfa_ms,
        "rtf": rtf,
        "total_s": t_end - t_start,
        "headers": headers,
        "headers_lower": headers_lower,
        "model_ttfp_ms": _parse_float_header(headers_lower, "x-model-ttfp-ms"),
        "model_ttf_ms": _parse_float_header(headers_lower, "x-model-ttf-ms"),
        "server_ttfa_ms": _parse_float_header(headers_lower, "x-server-ttfa-ms"),
        "deep_stream_impl": str(headers_lower.get("x-deep-stream-impl", "")).lower(),
        "packet_tokens": _parse_int_header(headers_lower, "x-packet-tokens"),
        "left_context": _parse_int_header(headers_lower, "x-left-context"),
    }


def calc_stats(values: list[float], allow_negative: bool = False) -> dict:
    if allow_negative:
        clean = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    else:
        clean = [v for v in values if isinstance(v, (int, float)) and v >= 0 and np.isfinite(v)]
    return {
        "p50": float(percentile(clean, 0.5)) if clean else -1.0,
        "p95": float(percentile(clean, 0.95)) if clean else -1.0,
        "max": float(max(clean)) if clean else -1.0,
        "n": len(clean),
    }


def sync_latest(run_dir: str, latest_dir: str) -> None:
    if os.path.islink(latest_dir):
        os.unlink(latest_dir)
    if os.path.isdir(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def append_report_line(report_path: str, line: str) -> None:
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", default=DEFAULT_TEXTS_PATH)
    parser.add_argument("--voices", default="")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    parser.add_argument("--offline-url", default=DEFAULT_OFFLINE_URL)
    args = parser.parse_args()

    texts = load_texts(args.texts)
    voices = load_voices(args.voices or None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, timestamp)
    latest_dir = os.path.join(args.out_root, "latest")
    ensure_dir(run_dir)

    if WARMUP_RUNS > 0 and texts:
        warm_payload = {
            "text": texts[0]["text"],
            "task_type": voices[0].get("task_type", "CustomVoice"),
            "language": voices[0].get("language", "Chinese"),
            "speaker": voices[0].get("speaker", "Vivian"),
            "instruct": voices[0].get("instruct", ""),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "non_streaming_mode": False,
        }
        for _ in range(WARMUP_RUNS):
            try:
                run_stream(args.stream_url, warm_payload, os.path.join(run_dir, "_warmup.wav"))
            except Exception:
                pass

    cases = []
    failures = []
    manual_candidates = []

    for text in texts:
        for voice in voices:
            text_id = text["id"]
            voice_id = voice.get("id", "voice")
            case_dir = os.path.join(run_dir, voice_id, text_id)
            ensure_dir(case_dir)
            offline_path = os.path.join(case_dir, "offline.wav")
            stream_path = os.path.join(case_dir, "stream.wav")
            metrics_path = os.path.join(case_dir, "metrics.json")

            payload = {
                "text": text["text"],
                "task_type": voice.get("task_type", "CustomVoice"),
                "language": voice.get("language", "Chinese"),
                "speaker": voice.get("speaker", "Vivian"),
                "instruct": voice.get("instruct", ""),
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "non_streaming_mode": False,
            }

            try:
                offline_audio, offline_sr, offline_total = run_offline(
                    args.offline_url, payload, offline_path
                )
            except Exception as e:
                failures.append(f"{voice_id}/{text_id}: offline_failed={e}")
                continue

            try:
                stream_result = run_stream(args.stream_url, payload, stream_path)
            except Exception as e:
                failures.append(f"{voice_id}/{text_id}: stream_failed={e}")
                continue

            stream_audio = stream_result["audio"]
            stream_sr = stream_result["sr"]
            e2e_ttfa_ms = stream_result["ttfa_ms"]
            ttfa_ms = e2e_ttfa_ms
            rtf = stream_result["rtf"]
            headers = stream_result["headers"]
            headers_lower = stream_result["headers_lower"]
            model_ttfp_ms = stream_result["model_ttfp_ms"]
            model_ttf_ms = stream_result["model_ttf_ms"]
            server_ttfa_ms = stream_result["server_ttfa_ms"]
            deep_stream_impl = stream_result["deep_stream_impl"]
            packet_tokens = stream_result["packet_tokens"]
            left_context = stream_result["left_context"]

            if EXPECT_DEEP_STREAM:
                deep_header = str(headers_lower.get("x-deep-stream", "")).lower()
                if deep_header not in ("true", "1", "yes"):
                    failures.append(f"{voice_id}/{text_id}: deep_stream_header_missing")
                if not deep_stream_impl:
                    failures.append(f"{voice_id}/{text_id}: deep_stream_impl_missing")
                elif deep_stream_impl != "paper":
                    failures.append(
                        f"{voice_id}/{text_id}: deep_stream_impl={deep_stream_impl}"
                    )
                if packet_tokens <= 0:
                    failures.append(f"{voice_id}/{text_id}: packet_tokens_missing")
                elif packet_tokens != EXPECTED_PACKET_TOKENS:
                    failures.append(
                        f"{voice_id}/{text_id}: packet_tokens={packet_tokens} expected={EXPECTED_PACKET_TOKENS}"
                    )
                if model_ttf_ms < 0:
                    failures.append(
                        f"{voice_id}/{text_id}: model_ttf_missing"
                    )

            if offline_sr != stream_sr:
                stream_audio = resample_linear(stream_audio, stream_sr, offline_sr)
                stream_sr = offline_sr

            stream_ok, stream_health = audio_health(stream_audio, DEFAULT_RMS_CLIP)
            offline_ok, offline_health = audio_health(offline_audio, DEFAULT_RMS_CLIP)
            if not stream_ok:
                failures.append(
                    f"{voice_id}/{text_id}: stream_bad_audio={stream_health.get('reason')}"
                )
            if not offline_ok:
                failures.append(
                    f"{voice_id}/{text_id}: offline_bad_audio={offline_health.get('reason')}"
                )

            aligned_offline, aligned_stream, offset = align_signals(
                offline_audio, stream_audio, stream_sr, DEFAULT_ALIGN_MAX_MS
            )
            mae_waveform = (
                float(np.mean(np.abs(aligned_offline - aligned_stream)))
                if aligned_offline.size > 0
                else -1.0
            )
            snr = snr_db(aligned_offline, aligned_stream)
            duration_diff_ms = (len(stream_audio) - len(offline_audio)) * 1000.0 / stream_sr
            pop_score = pop_click_score(stream_audio, stream_sr, DEFAULT_POP_FRAME_MS)

            metrics = {
                "mae_waveform": mae_waveform,
                "snr_db": snr,
                "pop_click_score": pop_score,
                "duration_diff_ms": duration_diff_ms,
                "ttfa_ms": ttfa_ms,
                "e2e_ttfa_ms": e2e_ttfa_ms,
                "model_ttfp_ms": model_ttfp_ms,
                "model_ttf_ms": model_ttf_ms,
                "server_ttfa_ms": server_ttfa_ms,
                "rtf": rtf,
                "offset_samples": offset,
                "stream_rms": stream_health.get("rms", 0.0),
                "stream_peak": stream_health.get("peak", 0.0),
                "offline_rms": offline_health.get("rms", 0.0),
                "offline_peak": offline_health.get("peak", 0.0),
                "offline_total_s": offline_total,
                "deep_stream_impl": deep_stream_impl,
                "packet_tokens": packet_tokens,
                "left_context": left_context,
            }
            write_json(metrics_path, metrics)

            if mae_waveform > 1e-3:
                failures.append(f"{voice_id}/{text_id}: mae_waveform={mae_waveform:.6f}")

            cases.append(
                {
                    "text_id": text_id,
                    "voice_id": voice_id,
                    "category": text.get("category", ""),
                    "paths": {"offline": offline_path, "stream": stream_path},
                    "metrics": metrics,
                }
            )
            if len(manual_candidates) < 2:
                manual_candidates.append({"voice_id": voice_id, "text_id": text_id, "path": stream_path})

    metrics_summary = {
        "mae_waveform": calc_stats([c["metrics"]["mae_waveform"] for c in cases]),
        "snr_db": calc_stats([c["metrics"]["snr_db"] for c in cases], allow_negative=True),
        "pop_click_score": calc_stats([c["metrics"]["pop_click_score"] for c in cases]),
        "duration_diff_ms": calc_stats([c["metrics"]["duration_diff_ms"] for c in cases], allow_negative=True),
        "ttfa_ms": calc_stats([c["metrics"]["ttfa_ms"] for c in cases]),
        "e2e_ttfa_ms": calc_stats([c["metrics"]["e2e_ttfa_ms"] for c in cases]),
        "model_ttfp_ms": calc_stats([c["metrics"]["model_ttfp_ms"] for c in cases]),
        "model_ttf_ms": calc_stats([c["metrics"]["model_ttf_ms"] for c in cases]),
        "server_ttfa_ms": calc_stats([c["metrics"]["server_ttfa_ms"] for c in cases]),
        "rtf": calc_stats([c["metrics"]["rtf"] for c in cases]),
    }

    baseline = None
    baseline_path = BASELINE_PATH.strip()
    if not baseline_path:
        baseline_path = os.path.join(args.out_root, "latest", "summary.json")
    if os.path.isfile(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception:
            baseline = None

    if baseline:
        base_pop = baseline.get("metrics", {}).get("pop_click_score", {}).get("p95", -1.0)
        if base_pop > 0:
            if metrics_summary["pop_click_score"]["p95"] > base_pop * 1.2:
                failures.append(
                    f"pop_click_score_p95_regress: {metrics_summary['pop_click_score']['p95']:.6f} > {base_pop * 1.2:.6f}"
                )
        base_ttfa = baseline.get("metrics", {}).get("e2e_ttfa_ms", {}).get("p50", -1.0)
        if base_ttfa <= 0:
            base_ttfa = baseline.get("metrics", {}).get("ttfa_ms", {}).get("p50", -1.0)
        if base_ttfa > 0:
            if metrics_summary["e2e_ttfa_ms"]["p50"] > base_ttfa + 80.0:
                failures.append(
                    f"ttfa_p50_regress_ms: {metrics_summary['e2e_ttfa_ms']['p50']:.2f} > {base_ttfa + 80.0:.2f}"
                )

    status = "PASS" if not failures else "FAIL"
    long_run = None
    if LONG_RUN_SECONDS > 0 and texts and voices:
        long_dir = os.path.join(run_dir, "long_run")
        ensure_dir(long_dir)
        long_texts = [t for t in texts if t.get("category") == "long"]
        long_text = long_texts[0] if long_texts else texts[-1]
        voice = voices[0]
        payload = {
            "text": long_text["text"],
            "task_type": voice.get("task_type", "CustomVoice"),
            "language": voice.get("language", "Chinese"),
            "speaker": voice.get("speaker", "Vivian"),
            "instruct": voice.get("instruct", ""),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "non_streaming_mode": False,
        }
        total_audio = 0.0
        series = []
        for idx in range(1, 21):
            out_path = os.path.join(long_dir, f"stream_{idx:02d}.wav")
            try:
                result = run_stream(args.stream_url, payload, out_path)
            except Exception as e:
                failures.append(f"long_run_failed: {e}")
                break
            dur_s = len(result["audio"]) / result["sr"] if result["sr"] > 0 else 0.0
            if dur_s <= 0:
                failures.append("long_run_failed: zero_duration")
                break
            total_audio += dur_s
            series.append({"idx": idx, "rtf": result["rtf"], "dur_s": dur_s})
            if total_audio >= LONG_RUN_SECONDS:
                break
        slope = 0.0
        if len(series) >= 2:
            x = np.array([s["idx"] for s in series], dtype=np.float32)
            y = np.array([s["rtf"] for s in series], dtype=np.float32)
            slope = float(np.polyfit(x, y, 1)[0])
        long_run = {
            "text_id": long_text.get("id", ""),
            "voice_id": voice.get("id", "voice"),
            "total_audio_s": total_audio,
            "series": series,
            "rtf_slope": slope,
        }
        write_json(os.path.join(long_dir, "long_run.json"), long_run)

    summary = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "stream_url": args.stream_url,
        "offline_url": args.offline_url,
        "metrics": metrics_summary,
        "definitions": {
            "model_ttfp_ms": "LM time to first packet tokens (server-side, before decode).",
            "model_ttf_ms": "model_ttfp_ms + first packet decode time (server-side, paper-style).",
            "server_ttfa_ms": "server request-in to first audio bytes written.",
            "e2e_ttfa_ms": "client request start to first audio bytes received.",
        },
        "cases": cases,
        "long_run": long_run,
        "status": status,
        "failures": failures,
        "manual_candidates": manual_candidates,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary)
    sync_latest(run_dir, latest_dir)

    print(f"[REGRESSION] status={status} cases={len(cases)} failures={len(failures)}")
    print("[REGRESSION] metrics:", json.dumps(metrics_summary, ensure_ascii=False))
    if manual_candidates:
        print("[REGRESSION] manual_check:", manual_candidates)
    if failures:
        for f in failures:
            print("[REGRESSION] FAIL:", f)

    report_line = (
        f"- 回归 {timestamp}: {status} "
        f"E2E_TTFA_P50={metrics_summary['e2e_ttfa_ms']['p50']:.2f}ms "
        f"E2E_TTFA_P95={metrics_summary['e2e_ttfa_ms']['p95']:.2f}ms "
        f"MODEL_TTF_P50={metrics_summary['model_ttf_ms']['p50']:.2f}ms "
        f"RTF_P50={metrics_summary['rtf']['p50']:.3f} "
        f"MAE_P50={metrics_summary['mae_waveform']['p50']:.6f} "
        f"SNR_P50={metrics_summary['snr_db']['p50']:.2f}dB"
    )
    append_report_line(REPORT_PATH, report_line)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())




