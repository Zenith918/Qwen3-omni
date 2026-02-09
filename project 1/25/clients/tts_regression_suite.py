#!/usr/bin/env python3
import argparse
import io
import hashlib
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
import torch

DEFAULT_TEXTS_PATH = os.path.join(os.path.dirname(__file__), "texts_p0_base.json")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
EXPECTED_PACKET_TOKENS = int(os.environ.get("TTS_EXPECT_PACKET_TOKENS", "2"))
TTS_CODE_DUMP_DIR = os.environ.get("TTS_CODE_DUMP_DIR", "/workspace/project 1/25/output/code_dumps")
DETERMINISM_RUNS = int(os.environ.get("TTS_DETERMINISM_RUNS", "10"))
DETERMINISM_TEXTS = os.environ.get("TTS_DETERMINISM_TEXTS", "long_03,short_01")
GATE_TTFA_P95_MS = float(os.environ.get("TTS_GATE_TTFA_P95_MS", "350"))
GATE_ABS_DURATION_DIFF_MS = float(os.environ.get("TTS_GATE_ABS_DURATION_DIFF_MS", "50"))
GATE_SNR_BASELINE_DB = float(os.environ.get("TTS_GATE_SNR_BASELINE_DB", "15"))
REPEAT_WINDOW_MS = int(os.environ.get("TTS_REPEAT_WINDOW_MS", "500"))
REPEAT_CORR_THRESHOLD = float(os.environ.get("TTS_REPEAT_CORR_THRESHOLD", "0.995"))
REPEAT_COUNT_MAX = int(os.environ.get("TTS_REPEAT_COUNT_MAX", "0"))
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


def write_json_compact(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: str, data: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


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


def normalize_decode_mode(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw.startswith("incremental"):
        return "incremental"
    if raw == "windowed":
        return "windowed"
    return "unknown"


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


def repeat_segment_metrics(
    audio: np.ndarray, sr: int, window_ms: int, corr_threshold: float, rms_floor: float = 1e-4
) -> tuple[int, float]:
    if audio.size == 0 or sr <= 0:
        return 0, 0.0
    window = max(1, int(sr * window_ms / 1000))
    if len(audio) < window * 2:
        return 0, 0.0
    segments = len(audio) // window
    if segments < 2:
        return 0, 0.0
    repeats = 0
    max_corr = 0.0
    for idx in range(1, segments):
        a = audio[(idx - 1) * window : idx * window]
        b = audio[idx * window : (idx + 1) * window]
        if a.size == 0 or b.size == 0:
            continue
        rms_a = float(np.sqrt(np.mean(a**2)))
        rms_b = float(np.sqrt(np.mean(b**2)))
        if rms_a < rms_floor or rms_b < rms_floor:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            continue
        corr = float(np.dot(a, b) / denom)
        if corr > max_corr:
            max_corr = corr
        if corr >= corr_threshold:
            repeats += 1
    return repeats, max_corr


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_codes(tag: str, dump_dir: str):
    path = os.path.join(dump_dir, f"codes_{tag}.pt")
    codes = torch.load(path, map_location="cpu")
    if isinstance(codes, torch.Tensor):
        return codes.cpu().numpy()
    return codes


def _first_diff_frame(a, b) -> int:
    if a is None or b is None:
        return -1
    frames = min(a.shape[0], b.shape[0])
    if frames <= 0:
        return -1
    a = a[:frames]
    b = b[:frames]
    if a.ndim == 1:
        diff = a != b
        return int(diff.argmax()) if diff.any() else -1
    diff = (a != b).any(axis=1)
    return int(diff.argmax()) if diff.any() else -1


def run_offline(offline_url: str, payload: dict, out_path: str, save_wav: bool = True) -> tuple[np.ndarray, int, float]:
    t_start = time.time()
    resp = requests.post(offline_url, json=payload, timeout=(10, DEFAULT_OFFLINE_TIMEOUT_S))
    resp.raise_for_status()
    if save_wav:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        audio, sr = sf.read(out_path, dtype="float32")
    else:
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    t_total = time.time() - t_start
    return audio.astype(np.float32), int(sr), t_total


def run_stream(stream_url: str, payload: dict, out_path: str, save_wav: bool = True) -> dict:
    t_start = time.time()
    first_chunk = None
    pcm_chunks: list[bytes] = []
    headers = {}
    headers_lower = {}
    save_wav = save_wav and bool(out_path)
    wf = None
    with requests.post(stream_url, json=payload, stream=True, timeout=(10, DEFAULT_STREAM_TIMEOUT_S)) as resp:
        resp.raise_for_status()
        headers = dict(resp.headers)
        headers_lower = {k.lower(): v for k, v in headers.items()}
        sr = int(resp.headers.get("x-sample-rate", "24000"))
        if save_wav:
            wf = wave.open(out_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first_chunk is None:
                first_chunk = time.time()
            pcm_chunks.append(chunk)
            if wf is not None:
                wf.writeframes(chunk)
    if wf is not None:
        wf.close()
    t_end = time.time()
    pcm = b"".join(pcm_chunks)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    dur_s = len(audio) / sr if sr > 0 else 0.0
    ttfa_ms = (first_chunk - t_start) * 1000.0 if first_chunk else -1.0
    rtf = (t_end - t_start) / dur_s if dur_s > 0 else -1.0
    decode_mode_raw = str(headers_lower.get("x-deep-decode-mode", ""))
    decode_mode = normalize_decode_mode(decode_mode_raw)
    if decode_mode != "incremental" and os.environ.get("ALLOW_WINDOWED_BASELINE", "") != "1":
        raise RuntimeError(f"decode_mode_not_incremental:{decode_mode_raw}")
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
        "code_dump_tag": str(headers_lower.get("x-code-dump-tag", "")),
        "decode_mode_raw": decode_mode_raw,
        "decode_mode": decode_mode,
        "deep_process": str(headers_lower.get("x-deep-process", "")).lower(),
        "sdpa_mode": str(headers_lower.get("x-sdpa-mode", "")),
        "sdpa_flash_attempts": _parse_int_header(headers_lower, "x-sdpa-flash-attempts"),
        "sdpa_flash_success": _parse_int_header(headers_lower, "x-sdpa-flash-success"),
        "sdpa_flash_fallbacks": _parse_int_header(headers_lower, "x-sdpa-flash-fallbacks"),
    }


def run_determinism_checks(
    stream_url: str,
    payload: dict,
    run_dir: str,
    text_id: str,
    runs: int,
    dump_dir: str,
    save_wav: bool = True,
) -> dict:
    det_dir = os.path.join(run_dir, "determinism")
    ensure_dir(det_dir)
    tags = []
    audio_hashes = []
    for idx in range(runs):
        out_path = os.path.join(det_dir, f"{text_id}_{idx:02d}.wav")
        payload = dict(payload)
        payload["dump_codes"] = True
        try:
            result = run_stream(stream_url, payload, out_path, save_wav=save_wav)
        except Exception as e:
            return {
                "text_id": text_id,
                "count": len(tags),
                "hash_unique": -1,
                "first_diff_frame": -2,
                "tags": tags,
                "error": f"stream_failed:{e}",
            }
        audio = result.get("audio")
        if isinstance(audio, np.ndarray) and audio.size:
            audio_hashes.append(_sha256_bytes(audio.tobytes()))
        tag = result.get("code_dump_tag", "")
        tags.append(tag)
    tags = [t for t in tags if t]
    if not tags:
        hash_unique = len(set(audio_hashes)) if audio_hashes else -1
        return {
            "text_id": text_id,
            "count": len(audio_hashes),
            "hash_unique": hash_unique,
            "first_diff_frame": -1,
            "tags": tags,
            "error": "" if hash_unique == 1 else "missing_code_dump_tags",
        }
    codes_cache = {}
    full_hashes = []
    for tag in tags:
        try:
            codes = _load_codes(tag, dump_dir)
        except Exception as e:
            if audio_hashes:
                hash_unique = len(set(audio_hashes))
                return {
                    "text_id": text_id,
                    "count": len(audio_hashes),
                    "hash_unique": hash_unique,
                    "first_diff_frame": -1,
                    "tags": tags,
                    "error": "" if hash_unique == 1 else f"load_codes_failed:{e}",
                }
            return {
                "text_id": text_id,
                "count": len(tags),
                "hash_unique": -1,
                "first_diff_frame": -2,
                "tags": tags,
                "error": f"load_codes_failed:{e}",
            }
        codes_cache[tag] = codes
        full_hashes.append(_sha256_bytes(codes.tobytes()))
    first_diff = -1
    if len(tags) >= 2:
        first_diff = _first_diff_frame(codes_cache.get(tags[0]), codes_cache.get(tags[1]))
    return {
        "text_id": text_id,
        "count": len(tags),
        "hash_unique": len(set(full_hashes)),
        "first_diff_frame": first_diff,
        "tags": tags,
        "error": "",
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

    mode = os.environ.get("REGRESSION_MODE", "fast").strip().lower()
    if mode not in ("fast", "full"):
        mode = "fast"
    run_determinism = os.environ.get("RUN_DETERMINISM", "1").lower() in ("1", "true", "yes")
    save_wav_env = os.environ.get("SAVE_WAV", "").strip().lower()
    if save_wav_env in ("1", "true", "yes"):
        save_wav = True
    elif save_wav_env in ("0", "false", "no"):
        save_wav = False
    else:
        save_wav = mode == "full"
    det_runs_env = os.environ.get("TTS_DETERMINISM_RUNS", "").strip()
    det_runs = int(det_runs_env) if det_runs_env else (3 if mode == "fast" else 10)
    det_texts_env = os.environ.get("TTS_DETERMINISM_TEXTS", "").strip()
    det_texts = det_texts_env if det_texts_env else ("short_01" if mode == "fast" else DETERMINISM_TEXTS)

    texts = load_texts(args.texts)
    if mode == "fast":
        allow_ids = {"short_01", "medium_01"}
        texts = [t for t in texts if t.get("id") in allow_ids]
    voices = load_voices(args.voices or None)
    if mode == "fast" and len(voices) > 1:
        voices = [voices[0]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, timestamp)
    latest_dir = os.path.join(args.out_root, "latest")
    ensure_dir(run_dir)

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
    baseline_cases = {}
    if baseline and isinstance(baseline.get("cases"), list):
        for case in baseline.get("cases", []):
            key = (case.get("voice_id", ""), case.get("text_id", ""))
            if key not in baseline_cases:
                baseline_cases[key] = case

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
                run_stream(args.stream_url, warm_payload, os.path.join(run_dir, "_warmup.wav"), save_wav=save_wav)
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
            offline_path = os.path.join(case_dir, "offline.wav") if save_wav else ""
            stream_path = os.path.join(case_dir, "stream.wav") if save_wav else ""
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
                    args.offline_url, payload, offline_path, save_wav=save_wav
                )
            except Exception as e:
                failures.append(f"{voice_id}/{text_id}: offline_failed={e}")
                continue

            try:
                stream_result = run_stream(args.stream_url, payload, stream_path, save_wav=save_wav)
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
            abs_duration_diff_ms = abs(duration_diff_ms)
            pop_score = pop_click_score(stream_audio, stream_sr, DEFAULT_POP_FRAME_MS)
            repeat_count, repeat_corr_max = repeat_segment_metrics(
                stream_audio, stream_sr, REPEAT_WINDOW_MS, REPEAT_CORR_THRESHOLD
            )

            baseline_snr_db = None
            baseline_offset = None
            baseline_stream_path = ""
            baseline_case = baseline_cases.get((voice_id, text_id))
            if baseline_case:
                baseline_stream_path = str(baseline_case.get("paths", {}).get("stream", ""))
                if baseline_stream_path and not os.path.isabs(baseline_stream_path):
                    baseline_stream_path = os.path.join(PROJECT_ROOT, baseline_stream_path)
                if baseline_stream_path and os.path.isfile(baseline_stream_path):
                    try:
                        base_audio, base_sr = sf.read(baseline_stream_path, dtype="float32")
                        if base_audio.ndim > 1:
                            base_audio = np.mean(base_audio, axis=1)
                        stream_for_base = stream_audio
                        stream_for_base_sr = stream_sr
                        if base_sr != stream_for_base_sr:
                            stream_for_base = resample_linear(
                                stream_for_base, stream_for_base_sr, base_sr
                            )
                            stream_for_base_sr = base_sr
                        aligned_base, aligned_stream_base, baseline_offset = align_signals(
                            base_audio, stream_for_base, stream_for_base_sr, DEFAULT_ALIGN_MAX_MS
                        )
                        baseline_snr_db = snr_db(aligned_base, aligned_stream_base)
                    except Exception:
                        baseline_snr_db = None
                        baseline_offset = None

            metrics = {
                "mae_waveform": mae_waveform,
                "snr_db": snr,
                "snr_baseline_db": baseline_snr_db,
                "pop_click_score": pop_score,
                "duration_diff_ms": duration_diff_ms,
                "abs_duration_diff_ms": abs_duration_diff_ms,
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
                "decode_mode": stream_result.get("decode_mode", ""),
                "decode_mode_raw": stream_result.get("decode_mode_raw", ""),
                "deep_process": stream_result.get("deep_process", ""),
                "sdpa_mode": stream_result.get("sdpa_mode", ""),
                "sdpa_flash_attempts": stream_result.get("sdpa_flash_attempts", -1),
                "sdpa_flash_success": stream_result.get("sdpa_flash_success", -1),
                "sdpa_flash_fallbacks": stream_result.get("sdpa_flash_fallbacks", -1),
                "repeat_count": repeat_count,
                "repeat_corr_max": repeat_corr_max,
                "baseline_offset_samples": baseline_offset,
                "baseline_stream_path": baseline_stream_path,
            }
            write_json(metrics_path, metrics)

            if mae_waveform > 1e-3:
                failures.append(f"{voice_id}/{text_id}: mae_waveform={mae_waveform:.6f}")
            if abs_duration_diff_ms > GATE_ABS_DURATION_DIFF_MS:
                failures.append(
                    f"{voice_id}/{text_id}: abs_duration_diff_ms={abs_duration_diff_ms:.3f}"
                )
            if repeat_count > REPEAT_COUNT_MAX:
                failures.append(
                    f"{voice_id}/{text_id}: repeat_count={repeat_count} corr_max={repeat_corr_max:.4f}"
                )
            if baseline_snr_db is not None and baseline_snr_db < GATE_SNR_BASELINE_DB:
                failures.append(
                    f"{voice_id}/{text_id}: snr_vs_baseline_db={baseline_snr_db:.2f} < {GATE_SNR_BASELINE_DB:.2f}"
                )

            cases.append(
                {
                    "text_id": text_id,
                    "voice_id": voice_id,
                    "category": text.get("category", ""),
                    "paths": {"offline": offline_path, "stream": stream_path},
                    "metrics": metrics,
                }
            )
            if save_wav and len(manual_candidates) < 2:
                manual_candidates.append({"voice_id": voice_id, "text_id": text_id, "path": stream_path})

    if cases:
        for case in cases:
            if case.get("text_id") == "long_03":
                candidate = {
                    "voice_id": case.get("voice_id", ""),
                    "text_id": case.get("text_id", ""),
                    "path": case.get("paths", {}).get("stream", ""),
                }
                if save_wav and candidate["path"] and candidate not in manual_candidates:
                    manual_candidates.append(candidate)
                break

    metrics_summary = {
        "mae_waveform": calc_stats([c["metrics"]["mae_waveform"] for c in cases]),
        "snr_db": calc_stats([c["metrics"]["snr_db"] for c in cases], allow_negative=True),
        "snr_baseline_db": calc_stats(
            [c["metrics"].get("snr_baseline_db") for c in cases], allow_negative=True
        ),
        "pop_click_score": calc_stats([c["metrics"]["pop_click_score"] for c in cases]),
        "duration_diff_ms": calc_stats([c["metrics"]["duration_diff_ms"] for c in cases], allow_negative=True),
        "abs_duration_diff_ms": calc_stats([c["metrics"]["abs_duration_diff_ms"] for c in cases]),
        "ttfa_ms": calc_stats([c["metrics"]["ttfa_ms"] for c in cases]),
        "e2e_ttfa_ms": calc_stats([c["metrics"]["e2e_ttfa_ms"] for c in cases]),
        "model_ttfp_ms": calc_stats([c["metrics"]["model_ttfp_ms"] for c in cases]),
        "model_ttf_ms": calc_stats([c["metrics"]["model_ttf_ms"] for c in cases]),
        "server_ttfa_ms": calc_stats([c["metrics"]["server_ttfa_ms"] for c in cases]),
        "rtf": calc_stats([c["metrics"]["rtf"] for c in cases]),
        "repeat_count": calc_stats([c["metrics"]["repeat_count"] for c in cases], allow_negative=True),
        "repeat_corr_max": calc_stats([c["metrics"]["repeat_corr_max"] for c in cases], allow_negative=True),
    }

    def _unique(values: list) -> list:
        seen = []
        for v in values:
            if v not in seen:
                seen.append(v)
        return seen

    def _sum_positive(values: list) -> int:
        nums = [int(v) for v in values if isinstance(v, (int, float)) and v >= 0]
        return int(sum(nums)) if nums else -1

    decode_modes = _unique([c["metrics"].get("decode_mode", "") for c in cases if c["metrics"].get("decode_mode", "")])
    packet_tokens_vals = _unique([c["metrics"].get("packet_tokens", -1) for c in cases])
    left_context_vals = _unique([c["metrics"].get("left_context", -1) for c in cases])
    deep_process_vals = _unique([c["metrics"].get("deep_process", "") for c in cases])
    sdpa_modes = _unique([c["metrics"].get("sdpa_mode", "") for c in cases if c["metrics"].get("sdpa_mode", "")])
    sdpa_flash_attempts = _sum_positive([c["metrics"].get("sdpa_flash_attempts", -1) for c in cases])
    sdpa_flash_success = _sum_positive([c["metrics"].get("sdpa_flash_success", -1) for c in cases])
    sdpa_flash_fallbacks = _sum_positive([c["metrics"].get("sdpa_flash_fallbacks", -1) for c in cases])
    sdpa_flash_success_rate = (
        float(sdpa_flash_success) / float(sdpa_flash_attempts)
        if sdpa_flash_attempts and sdpa_flash_attempts > 0
        else -1.0
    )

    def _normalize_process(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in ("1", "true", "yes"):
            return "1"
        if raw in ("0", "false", "no"):
            return "0"
        return raw

    expected = {
        "packet_tokens": int(os.environ.get("TTS_BASELINE_PACKET_TOKENS", "2")),
        "left_context": int(os.environ.get("TTS_BASELINE_LEFT_CONTEXT", "72")),
        "process": _normalize_process(os.environ.get("TTS_BASELINE_PROCESS", "0")),
    }
    actual = {
        "packet_tokens": packet_tokens_vals,
        "left_context": left_context_vals,
        "process": deep_process_vals,
    }
    baseline_checks = {}
    baseline_changed = []
    for key, exp in expected.items():
        values = actual.get(key, [])
        if key == "process":
            norm_values = [_normalize_process(v) for v in values]
            changed = not (len(norm_values) == 1 and norm_values[0] == exp)
            baseline_checks[key] = {"expected": exp, "actual": norm_values, "changed": changed}
        else:
            changed = not (len(values) == 1 and str(values[0]).lower() == str(exp).lower())
            baseline_checks[key] = {"expected": exp, "actual": values, "changed": changed}
        if changed:
            baseline_changed.append(key)

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
    if BASELINE_PATH.strip() and not baseline_cases:
        failures.append("baseline_missing_or_no_cases")

    if metrics_summary["ttfa_ms"]["p95"] > GATE_TTFA_P95_MS:
        failures.append(
            f"ttfa_p95_gate_ms: {metrics_summary['ttfa_ms']['p95']:.2f} > {GATE_TTFA_P95_MS:.2f}"
        )
    if metrics_summary["abs_duration_diff_ms"]["p95"] > GATE_ABS_DURATION_DIFF_MS:
        failures.append(
            f"abs_duration_diff_p95_ms: {metrics_summary['abs_duration_diff_ms']['p95']:.2f} > {GATE_ABS_DURATION_DIFF_MS:.2f}"
        )
    if metrics_summary["repeat_count"]["max"] > REPEAT_COUNT_MAX:
        failures.append(
            f"repeat_count_gate: {metrics_summary['repeat_count']['max']:.0f} > {REPEAT_COUNT_MAX}"
        )

    snr_baseline_vals = [
        c["metrics"].get("snr_baseline_db")
        for c in cases
        if isinstance(c["metrics"].get("snr_baseline_db"), (int, float))
    ]
    snr_baseline_min = min(snr_baseline_vals) if snr_baseline_vals else -1.0

    determinism = []
    if run_determinism and det_runs > 0 and texts and voices:
        det_text_ids = [t.strip() for t in det_texts.split(",") if t.strip()]
        text_map = {t.get("id"): t for t in texts}
        voice = voices[0]
        for text_id in det_text_ids:
            text_obj = text_map.get(text_id)
            if not text_obj:
                determinism.append(
                    {
                        "text_id": text_id,
                        "count": 0,
                        "hash_unique": -1,
                        "first_diff_frame": -2,
                        "tags": [],
                        "error": "text_not_found",
                    }
                )
                failures.append(f"determinism_{text_id}: text_not_found")
                continue
            payload = {
                "text": text_obj.get("text", ""),
                "task_type": voice.get("task_type", "CustomVoice"),
                "language": voice.get("language", "Chinese"),
                "speaker": voice.get("speaker", "Vivian"),
                "instruct": voice.get("instruct", ""),
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                "non_streaming_mode": False,
            }
            result = run_determinism_checks(
                args.stream_url,
                payload,
                run_dir,
                text_id,
                det_runs,
                TTS_CODE_DUMP_DIR,
                save_wav,
            )
            determinism.append(result)
            if result.get("error"):
                failures.append(f"determinism_{text_id}: {result['error']}")
            elif result.get("hash_unique") != 1 or result.get("first_diff_frame") != -1:
                failures.append(
                    f"determinism_{text_id}: hash_unique={result.get('hash_unique')} first_diff_frame={result.get('first_diff_frame')}"
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
                result = run_stream(args.stream_url, payload, out_path, save_wav=save_wav)
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

    sdpa_summary = {
        "mode_values": sdpa_modes,
        "flash_attempts": sdpa_flash_attempts,
        "flash_success": sdpa_flash_success,
        "flash_fallbacks": sdpa_flash_fallbacks,
        "flash_success_rate": sdpa_flash_success_rate,
    }

    summary_full = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "stream_url": args.stream_url,
        "offline_url": args.offline_url,
        "regression_mode": mode,
        "run_determinism": run_determinism,
        "save_wav": save_wav,
        "determinism_runs": det_runs if run_determinism else 0,
        "determinism_texts": det_texts if run_determinism else "",
        "metrics": metrics_summary,
        "decode_modes": decode_modes,
        "sdpa_flash": sdpa_summary,
        "baseline_path": baseline_path,
        "baseline_loaded": bool(baseline),
        "baseline_cases_count": len(baseline_cases),
        "baseline_checks": baseline_checks,
        "baseline_changed_params": baseline_changed,
        "gates": {
            "ttfa_p95_ms": {"value": metrics_summary["ttfa_ms"]["p95"], "threshold": GATE_TTFA_P95_MS},
            "abs_duration_diff_p95_ms": {
                "value": metrics_summary["abs_duration_diff_ms"]["p95"],
                "threshold": GATE_ABS_DURATION_DIFF_MS,
            },
            "repeat_count_max": {"value": metrics_summary["repeat_count"]["max"], "threshold": REPEAT_COUNT_MAX},
        },
        "determinism": determinism,
        "definitions": {
            "model_ttfp_ms": "LM time to first packet tokens (server-side, before decode).",
            "model_ttf_ms": "model_ttfp_ms + first packet decode time (server-side, paper-style).",
            "server_ttfa_ms": "server request-in to first audio bytes written.",
            "e2e_ttfa_ms": "client request start to first audio bytes received.",
            "abs_duration_diff_ms": "Absolute duration diff between stream and offline.",
            "repeat_count": "Count of adjacent window repeats above correlation threshold.",
            "repeat_corr_max": "Max adjacent window correlation.",
            "snr_baseline_db": "SNR between current stream and baseline stream (aligned).",
        },
        "cases": cases,
        "long_run": long_run,
        "status": status,
        "failures": failures,
        "manual_candidates": manual_candidates,
    }
    summary_brief = {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "regression_mode": mode,
        "run_determinism": run_determinism,
        "save_wav": save_wav,
        "determinism_runs": det_runs if run_determinism else 0,
        "determinism_texts": det_texts if run_determinism else "",
        "metrics": metrics_summary,
        "decode_modes": decode_modes,
        "sdpa_flash": sdpa_summary,
        "baseline_path": baseline_path,
        "baseline_loaded": bool(baseline),
        "baseline_cases_count": len(baseline_cases),
        "baseline_checks": baseline_checks,
        "baseline_changed_params": baseline_changed,
        "gates": summary_full["gates"],
        "determinism": determinism,
        "cases_count": len(cases),
        "status": status,
        "failures": failures,
    }
    if mode == "fast":
        write_json_compact(os.path.join(run_dir, "summary.json"), summary_brief)
    else:
        write_json(os.path.join(run_dir, "summary.json"), summary_full)
        write_json_compact(os.path.join(run_dir, "summary_brief.json"), summary_brief)
    append_jsonl(os.path.join(args.out_root, "summary.jsonl"), summary_brief)
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
