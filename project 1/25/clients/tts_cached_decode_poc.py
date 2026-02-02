#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

sys.path.append(os.path.dirname(__file__))
from tts_incremental_decoder import IncrementalDecoder


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


def decode_full_true(tokenizer: Qwen3TTSTokenizer, codes: torch.Tensor) -> tuple[np.ndarray, int]:
    codes_t = codes.to(torch.long)
    if codes_t.dim() == 3 and codes_t.shape[0] == 1:
        codes_t = codes_t.squeeze(0)
    if codes_t.dim() != 2:
        raise ValueError("decode_full_true expects shape (T, Q)")
    expected_len = int((codes_t[:, 0] > 0).sum().item()) * int(tokenizer.model.get_decode_upsample_rate())

    codes_t = codes_t.unsqueeze(0).transpose(1, 2)  # (1, Q, T)
    device = tokenizer.model.decoder.pre_conv.conv.weight.device
    codes_t = codes_t.to(device)
    cache_position = torch.arange(0, codes_t.shape[-1], device=device)
    position_ids = cache_position.unsqueeze(0)
    with torch.inference_mode():
        wav = tokenizer.model.decoder(codes_t, position_ids=position_ids, cache_position=cache_position)
    audio = wav.squeeze(1).to(torch.float32).detach().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.flatten()
    if expected_len > 0 and len(audio) > expected_len:
        audio = audio[:expected_len]
    return audio.astype(np.float32), int(tokenizer.get_output_sample_rate())


def decode_windowed(
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
        start_position = max(0, idx - ctx)
        wavs, sr = tokenizer.decode_streaming(
            codes_chunk, left_context_size=ctx, start_position=start_position
        )
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


def decode_incremental_stateful(
    tokenizer: Qwen3TTSTokenizer,
    codes: torch.Tensor,
    packet_tokens: int,
    device: str,
    transformer_mode: str,
) -> tuple[np.ndarray, int]:
    incremental = IncrementalDecoder(tokenizer, device=device, transformer_mode=transformer_mode)
    state = incremental.reset_state()
    total = codes.shape[0]
    idx = 0
    chunks = []
    sr = int(tokenizer.get_output_sample_rate())
    while idx < total:
        end = min(idx + packet_tokens, total)
        pcm, state = incremental.decode_incremental(codes[idx:end], state)
        if pcm.size > 0:
            chunks.append(pcm)
        idx = end
    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
    if state.expected_samples > 0 and len(audio) > state.expected_samples:
        audio = audio[: state.expected_samples]
    if audio.ndim > 1:
        audio = audio.flatten()
    return audio.astype(np.float32), sr


def calc_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return -1.0
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out-dir", default="/workspace/project 1/25/output/debug")
    parser.add_argument("--tokenizer-dir", default="/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--packet-tokens", type=int, default=4)
    parser.add_argument("--left-context", type=int, default=72)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--transformer-mode",
        default="cache",
        choices=("cache", "window", "full"),
        help="pre_transformer mode: cache/window/full",
    )
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

    t0 = time.perf_counter()
    a_audio, a_sr = decode_full_true(tokenizer, codes)
    t1 = time.perf_counter()
    b_audio, b_sr = decode_incremental_stateful(
        tokenizer, codes, args.packet_tokens, args.device, args.transformer_mode
    )
    t2 = time.perf_counter()
    c_audio, c_sr = decode_windowed(tokenizer, codes, args.packet_tokens, args.left_context)
    t3 = time.perf_counter()

    base = Path(args.codes).stem
    a_path = out_dir / f"{base}_A_full_true.wav"
    b_path = out_dir / f"{base}_B_incremental.wav"
    c_path = out_dir / f"{base}_C_windowed.wav"
    sf.write(a_path, a_audio, a_sr)
    sf.write(b_path, b_audio, b_sr)
    sf.write(c_path, c_audio, c_sr)

    metrics = {
        "A_full_true": {
            "pop_click_score": pop_click_score(a_audio, a_sr),
            "decode_ms": (t1 - t0) * 1000.0,
            "length": len(a_audio),
        },
        "B_incremental": {
            "pop_click_score": pop_click_score(b_audio, b_sr),
            "decode_ms": (t2 - t1) * 1000.0,
            "length": len(b_audio),
            "mae_vs_full": calc_mae(a_audio, b_audio),
        },
        "C_windowed": {
            "pop_click_score": pop_click_score(c_audio, c_sr),
            "decode_ms": (t3 - t2) * 1000.0,
            "length": len(c_audio),
            "mae_vs_full": calc_mae(a_audio, c_audio),
        },
    }
    with (out_dir / f"{base}_incremental_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
