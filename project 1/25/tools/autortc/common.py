#!/usr/bin/env python3
import json
import time
import wave
from pathlib import Path

import numpy as np


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_wav_mono_int16(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]
    return audio, sample_rate


def resample_linear_int16(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if len(audio) == 0:
        return audio
    n_dst = int(round(len(audio) * float(dst_sr) / float(src_sr)))
    xp = np.arange(len(audio), dtype=np.float64)
    xnew = np.linspace(0, len(audio) - 1, n_dst, dtype=np.float64)
    y = np.interp(xnew, xp, audio.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)


def wav_duration_s(path: str) -> float:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
    return float(n) / float(sr) if sr > 0 else 0.0


def write_wav_int16(path: str, pcm: bytes, sample_rate: int, num_channels: int = 1) -> None:
    ensure_parent(path)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def write_json(path: str, data: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_epoch_s() -> float:
    return time.time()

