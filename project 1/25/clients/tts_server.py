#!/usr/bin/env python3
import io
import os
import sys
import time
from typing import Generator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vllm import SamplingParams
from vllm_omni import Omni

app = FastAPI()

TTS_MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
STAGE_CONFIG = os.environ.get(
    "TTS_STAGE_CONFIG",
    "/workspace/project 1/25/artifacts/qwen3_tts_l40s.yaml",
)

_omni = None
_sampling_params = None
_warmup_done = False
_first_request_done = False


def _build_inputs(req: "TTSRequest") -> dict:
    prompt = f"<|im_start|>assistant\n{req.text}<|im_end|>\n<|im_start|>assistant\n"
    return {
        "prompt": prompt,
        "additional_information": {
            "task_type": [req.task_type],
            "text": [req.text],
            "language": [req.language],
            "speaker": [req.speaker],
            "instruct": [req.instruct],
            "max_new_tokens": [req.max_new_tokens],
            "non_streaming_mode": [req.non_streaming_mode],
        },
    }


def _audio_to_pcm16(audio_np: np.ndarray) -> bytes:
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _iter_audio_chunks(
    audio_np: np.ndarray,
    chunk_samples: int,
) -> Generator[bytes, None, None]:
    total = len(audio_np)
    idx = 0
    while idx < total:
        end = min(idx + chunk_samples, total)
        yield _audio_to_pcm16(audio_np[idx:end])
        idx = end


def _split_text_stream(
    text: str,
    starter_min: int = 2,
    starter_max: int = 6,
    main_min: int = 20,
    main_max: int = 60,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= starter_max:
        return [text]

    punct = "，。！？!?"
    split_idx = None
    for i in range(starter_min, min(starter_max, len(text)) + 1):
        if text[i - 1] in punct:
            split_idx = i
            break
    if split_idx is None:
        split_idx = starter_max

    segments: list[str] = [text[:split_idx].strip()]
    buf = ""
    for ch in text[split_idx:]:
        buf += ch
        if ch in punct and len(buf) >= main_min:
            segments.append(buf.strip())
            buf = ""
        elif len(buf) >= main_max:
            segments.append(buf.strip())
            buf = ""
    if buf.strip():
        segments.append(buf.strip())
    return segments


class TTSRequest(BaseModel):
    text: str
    task_type: str = "CustomVoice"
    language: str = "Chinese"
    speaker: str = "Vivian"
    instruct: str = ""
    max_new_tokens: int = 2048
    non_streaming_mode: bool = False


@app.on_event("startup")
def _init_omni():
    global _omni, _sampling_params, _warmup_done
    _omni = Omni(
        model=TTS_MODEL_DIR,
        stage_configs_path=STAGE_CONFIG,
        stage_init_timeout=600,
    )
    _sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        seed=42,
        detokenize=False,
        repetition_penalty=1.05,
    )
    _warmup()
    _warmup_done = True


def _warmup() -> None:
    assert _omni is not None
    warm_req = TTSRequest(text="warmup")
    inputs = _build_inputs(warm_req)
    for _ in _omni.generate(inputs, [_sampling_params]):
        pass


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    assert _omni is not None
    inputs = _build_inputs(req)

    start = time.time()
    audio_tensor = None
    sr = 24000
    for stage_outputs in _omni.generate(inputs, [_sampling_params]):
        for output in stage_outputs.request_output:
            audio_tensor = output.multimodal_output["audio"]
            sr = int(output.multimodal_output["sr"].item())
    if audio_tensor is None:
        raise RuntimeError("No audio output produced")

    audio_np = audio_tensor.float().detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()

    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)

    headers = {
        "X-Gen-Latency-Ms": str(int((time.time() - start) * 1000)),
        "X-Sample-Rate": str(sr),
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


@app.post("/tts/stream")
def synthesize_stream(req: TTSRequest):
    assert _omni is not None
    global _first_request_done
    t_req_in = time.time()
    first_out: Optional[float] = None
    sr = 24000
    chunk_ms = int(os.environ.get("TTS_STREAM_CHUNK_MS", "30"))
    chunk_ms = max(20, min(40, chunk_ms))
    sent_samples = 0
    segments = _split_text_stream(req.text)
    if not segments:
        segments = [req.text]
    warm_request = _warmup_done and _first_request_done

    def _gen() -> Generator[bytes, None, None]:
        global _first_request_done
        nonlocal first_out, sr, sent_samples
        for seg_idx, seg in enumerate(segments):
            seg_req = TTSRequest(
                text=seg,
                task_type=req.task_type,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct,
                max_new_tokens=req.max_new_tokens,
                non_streaming_mode=req.non_streaming_mode,
            )
            inputs = _build_inputs(seg_req)
            sent_samples = 0
            for stage_outputs in _omni.generate(inputs, [_sampling_params]):
                for output in stage_outputs.request_output:
                    audio_tensor = output.multimodal_output["audio"]
                    sr = int(output.multimodal_output["sr"].item())
                    audio_np = audio_tensor.float().detach().cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()

                    if len(audio_np) <= sent_samples:
                        continue

                    new_audio = audio_np[sent_samples:]
                    chunk_samples = max(1, int(sr * chunk_ms / 1000))
                    for chunk in _iter_audio_chunks(new_audio, chunk_samples):
                        if first_out is None:
                            first_out = time.time()
                            print(
                                f"[TTS] t_req_in={t_req_in:.6f} "
                                f"t_first_audio_out={first_out:.6f} "
                                f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                                f"segments={len(segments)} chunk_ms={chunk_ms}"
                            )
                            sys.stdout.flush()
                        yield chunk
                    sent_samples = len(audio_np)

        t_done = time.time()
        print(
            f"[TTS] t_req_in={t_req_in:.6f} t_done={t_done:.6f} "
            f"total={(t_done - t_req_in):.3f} warm={warm_request} "
            f"segments={len(segments)} chunk_ms={chunk_ms}"
        )
        sys.stdout.flush()
        _first_request_done = True

    headers = {
        "X-Sample-Rate": str(sr),
        "X-PCM-Format": "s16le",
        "X-Chunk-Ms": str(chunk_ms),
        "X-Warmup-Done": str(_warmup_done).lower(),
        "X-Warm-Request": str(warm_request).lower(),
        "X-Segments": str(len(segments)),
    }
    return StreamingResponse(_gen(), media_type="audio/pcm", headers=headers)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
