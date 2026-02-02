#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def load_codes(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        arr = np.load(path)
        return torch.from_numpy(arr)
    return torch.load(path)


def pop_click_score(audio: np.ndarray, sr: int, frame_ms: int = 20) -> float:
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


def decode_full(tokenizer: Qwen3TTSTokenizer, codes: torch.Tensor) -> tuple[np.ndarray, int]:
    wavs, sr = tokenizer.decode([{"audio_codes": codes}])
    audio = wavs[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.float().detach().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.flatten()
    return audio.astype(np.float32), int(sr)


def decode_streaming(
    tokenizer: Qwen3TTSTokenizer, codes: torch.Tensor, packet_tokens: int, left_context: int
) -> tuple[np.ndarray, int]:
    total = codes.shape[0]
    out_chunks = []
    idx = 0
    sr = None
    while idx < total:
        end = min(idx + packet_tokens, total)
        ctx = min(left_context, idx)
        codes_chunk = codes[idx - ctx : end]
        wavs, sr = tokenizer.decode_streaming(codes_chunk, left_context_size=ctx)
        chunk = wavs[0]
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.float().detach().cpu().numpy()
        out_chunks.append(chunk)
        idx = end
    if out_chunks:
        audio = np.concatenate(out_chunks, axis=0)
    else:
        audio = np.zeros((0,), dtype=np.float32)
        sr = tokenizer.get_output_sample_rate()
    if audio.ndim > 1:
        audio = audio.flatten()
    return audio.astype(np.float32), int(sr)


def spike_times(audio: np.ndarray, sr: int, percentile: float = 99.9) -> tuple[np.ndarray, float]:
    diff = np.abs(np.diff(audio))
    thr = float(np.percentile(diff, percentile))
    idx = np.where(diff >= thr)[0]
    return idx / sr, thr


def pick_top_spikes(audio: np.ndarray, sr: int, percentile: float = 99.9, min_gap_s: float = 0.05) -> list[float]:
    diff = np.abs(np.diff(audio))
    thr = np.percentile(diff, percentile)
    idx_sorted = np.argsort(diff)[::-1]
    picked = []
    for idx in idx_sorted:
        t = idx / sr
        if all(abs(t - p) > min_gap_s for p in picked):
            picked.append(t)
        if len(picked) >= 20:
            break
    return picked


def codes_jump_stats(codes: torch.Tensor, token_index: int) -> dict:
    if token_index <= 0 or token_index >= codes.shape[0]:
        return {"token_index": token_index, "delta_max": 0.0, "delta_mean": 0.0, "deltas": []}
    prev = codes[token_index - 1]
    curr = codes[token_index]
    deltas = torch.abs(curr - prev).tolist()
    return {
        "token_index": int(token_index),
        "delta_max": float(max(deltas)) if deltas else 0.0,
        "delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "deltas": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out-dir", default="/workspace/project 1/25/output/debug")
    parser.add_argument("--tokenizer-dir", default="/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--packet-tokens", type=int, default=4)
    parser.add_argument("--left-context", type=int, default=72)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_codes(args.codes)
    if isinstance(codes, torch.Tensor) and codes.dtype != torch.long:
        codes = codes.to(torch.long)
    if codes.dim() == 3 and codes.shape[0] == 1:
        codes = codes.squeeze(0)

    tokenizer_dir = args.tokenizer_dir
    candidate = Path(tokenizer_dir) / "speech_tokenizer"
    if candidate.is_dir():
        tokenizer_dir = str(candidate)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    tok_kwargs = {"device_map": args.device, "dtype": dtype}
    tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_dir, attn_implementation="eager", **tok_kwargs)

    a_audio, a_sr = decode_full(tokenizer, codes)
    b_audio, b_sr = decode_streaming(tokenizer, codes, args.packet_tokens, args.left_context)
    c_audio, c_sr = decode_streaming(tokenizer, codes, args.packet_tokens, 0)

    base = Path(args.codes).stem
    a_path = out_dir / f"{base}_A_full.wav"
    b_path = out_dir / f"{base}_B_stream.wav"
    c_path = out_dir / f"{base}_C_ctx0.wav"
    sf.write(a_path, a_audio, a_sr)
    sf.write(b_path, b_audio, b_sr)
    sf.write(c_path, c_audio, c_sr)

    metrics = {
        "A_full": {"pop_click_score": pop_click_score(a_audio, a_sr)},
        "B_stream": {"pop_click_score": pop_click_score(b_audio, b_sr)},
        "C_ctx0": {"pop_click_score": pop_click_score(c_audio, c_sr)},
    }
    with (out_dir / f"{base}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # P2: map pop times to code jumps (use B_stream)
    try:
        sr = tokenizer.get_output_sample_rate()
        upsample = tokenizer.get_decode_upsample_rate()
        frames_per_sec = sr / float(upsample)
    except Exception:
        frames_per_sec = 12.5

    top_times = pick_top_spikes(b_audio, b_sr)
    pop_stats = []
    for t in top_times:
        token_idx = int(round(t * frames_per_sec))
        pop_stats.append({"time_s": t, **codes_jump_stats(codes, token_idx)})

    with (out_dir / f"{base}_pop_code_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frames_per_sec": frames_per_sec,
                "packet_tokens": args.packet_tokens,
                "left_context": args.left_context,
                "top_spikes": pop_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
