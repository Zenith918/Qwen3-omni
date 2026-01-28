#!/usr/bin/env python3
import os
import time
import threading
from queue import Queue, Empty
import wave

import numpy as np
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TOKENIZER_DIR = os.environ.get("TTS_TOKENIZER_DIR", "")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/project 1/25/output/poc_stream")
TEXT = os.environ.get("TTS_TEXT", "你好，我在。今天我们测试流式语音。")
SPEAKER = os.environ.get("TTS_SPEAKER", "Vivian")
LANGUAGE = os.environ.get("TTS_LANGUAGE", "Chinese")
INSTRUCT = os.environ.get("TTS_INSTRUCT", "")
PACKET_TOKENS = int(os.environ.get("TTS_PACKET_TOKENS", "4"))
RUNS = int(os.environ.get("TTS_POC_RUNS", "3"))
MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "512"))
DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")

os.makedirs(OUT_DIR, exist_ok=True)


def pcm16_bytes(audio_np: np.ndarray) -> bytes:
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767.0).astype(np.int16)
    return pcm.tobytes()


class CodecStreamQueue:
    def __init__(self):
        self._q: Queue = Queue()
        self._closed = False

    def put(self, codes: torch.Tensor):
        if self._closed:
            return
        self._q.put(codes)

    def close(self):
        self._closed = True
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item


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


def run_once(model: Qwen3TTSModel, tokenizer: Qwen3TTSTokenizer, run_idx: int) -> dict:
    streamer = CodecStreamQueue()
    full_codes_holder: dict[str, list[torch.Tensor]] = {}

    def _generate():
        try:
            codes_list, _ = model.generate_custom_voice_codes(
                text=TEXT,
                speaker=SPEAKER,
                language=LANGUAGE,
                instruct=INSTRUCT,
                codec_streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            full_codes_holder["codes"] = codes_list
        finally:
            streamer.close()

    thread = threading.Thread(target=_generate, daemon=True)
    t_req_in = time.time()
    thread.start()

    codes_accum: list[torch.Tensor] = []
    stream_audio_chunks: list[np.ndarray] = []
    sent_samples = 0
    first_pcm_ts = None
    sr = None

    stream_path = os.path.join(OUT_DIR, f"stream_{run_idx:02d}.wav")
    with wave.open(stream_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        for codes in streamer:
            if isinstance(codes, torch.Tensor):
                if codes.dim() == 2 and codes.shape[0] == 1:
                    codes = codes.squeeze(0)
                codes_accum.append(codes.to(torch.long))
            else:
                continue

            if len(codes_accum) % PACKET_TOKENS != 0:
                continue

            codes_tensor = torch.stack(codes_accum, dim=0)
            wavs, sr = tokenizer.decode([{"audio_codes": codes_tensor}])
            audio_np = wavs[0]
            new_audio = audio_np[sent_samples:]
            if new_audio.size > 0:
                if first_pcm_ts is None:
                    first_pcm_ts = time.time()
                stream_audio_chunks.append(new_audio)
                wf.writeframes(pcm16_bytes(new_audio))
                sent_samples = len(audio_np)

        if codes_accum and (len(codes_accum) % PACKET_TOKENS != 0):
            codes_tensor = torch.stack(codes_accum, dim=0)
            wavs, sr = tokenizer.decode([{"audio_codes": codes_tensor}])
            audio_np = wavs[0]
            new_audio = audio_np[sent_samples:]
            if new_audio.size > 0:
                if first_pcm_ts is None:
                    first_pcm_ts = time.time()
                stream_audio_chunks.append(new_audio)
                wf.writeframes(pcm16_bytes(new_audio))
                sent_samples = len(audio_np)

    thread.join()
    t_done = time.time()

    stream_audio = np.concatenate(stream_audio_chunks, axis=0) if stream_audio_chunks else np.zeros((0,), dtype=np.float32)

    offline_audio = np.zeros((0,), dtype=np.float32)
    if full_codes_holder.get("codes"):
        offline_codes = full_codes_holder["codes"][0]
        wavs, _ = tokenizer.decode([{"audio_codes": offline_codes}])
        offline_audio = wavs[0]
        offline_path = os.path.join(OUT_DIR, f"offline_{run_idx:02d}.wav")
        with wave.open(offline_path, "wb") as wf_off:
            wf_off.setnchannels(1)
            wf_off.setsampwidth(2)
            wf_off.setframerate(sr or 24000)
            wf_off.writeframes(pcm16_bytes(offline_audio))

    min_len = min(len(stream_audio), len(offline_audio))
    if min_len > 0:
        diff = np.abs(stream_audio[:min_len] - offline_audio[:min_len])
        mean_abs_err = float(np.mean(diff))
        max_abs_err = float(np.max(diff))
    else:
        mean_abs_err = -1.0
        max_abs_err = -1.0

    t_first_pcm_out = (first_pcm_ts - t_req_in) if first_pcm_ts else -1.0
    result = {
        "t_first_pcm_out": t_first_pcm_out,
        "t_total": t_done - t_req_in,
        "packet_tokens": PACKET_TOKENS,
        "stream_len": len(stream_audio),
        "offline_len": len(offline_audio),
        "mean_abs_err": mean_abs_err,
        "max_abs_err": max_abs_err,
    }
    print(
        f"[POC][RUN {run_idx:02d}] t_first_pcm_out={t_first_pcm_out:.3f} "
        f"t_total={t_done - t_req_in:.3f} packet_tokens={PACKET_TOKENS} "
        f"len_stream={len(stream_audio)} len_offline={len(offline_audio)} "
        f"mae={mean_abs_err:.6f} maxe={max_abs_err:.6f}"
    )
    return result


def main():
    tokenizer_dir = TOKENIZER_DIR.strip()
    if not tokenizer_dir:
        candidate = os.path.join(MODEL_DIR, "speech_tokenizer")
        if os.path.isdir(candidate):
            tokenizer_dir = candidate
        else:
            tokenizer_dir = MODEL_DIR

    dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
    model_kwargs = {"device_map": DEVICE, "dtype": dtype}
    try:
        model = Qwen3TTSModel.from_pretrained(MODEL_DIR, attn_implementation="flash_attention_2", **model_kwargs)
    except Exception:
        model = Qwen3TTSModel.from_pretrained(MODEL_DIR, attn_implementation="eager", **model_kwargs)

    tok_kwargs = {"device_map": DEVICE, "dtype": dtype}
    try:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir, attn_implementation="flash_attention_2", **tok_kwargs
        )
    except Exception:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_dir, attn_implementation="eager", **tok_kwargs)

    results = []
    for i in range(RUNS):
        results.append(run_once(model, tokenizer, i + 1))

    ok = [r for r in results if r["t_first_pcm_out"] > 0]
    vals = [r["t_first_pcm_out"] for r in ok]
    print(
        f"[POC] t_first_pcm_out p50={percentile(vals, 0.5):.3f} "
        f"p95={percentile(vals, 0.95):.3f} n={len(ok)}"
    )


if __name__ == "__main__":
    main()
