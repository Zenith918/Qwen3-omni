#!/usr/bin/env python3
import io
import os
import time
from typing import Optional

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


class TTSRequest(BaseModel):
    text: str
    task_type: str = "CustomVoice"
    language: str = "Chinese"
    speaker: str = "Vivian"
    instruct: str = ""
    max_new_tokens: int = 2048


@app.on_event("startup")
def _init_omni():
    global _omni, _sampling_params
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


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    assert _omni is not None
    prompt = f"<|im_start|>assistant\n{req.text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = {
        "prompt": prompt,
        "additional_information": {
            "task_type": [req.task_type],
            "text": [req.text],
            "language": [req.language],
            "speaker": [req.speaker],
            "instruct": [req.instruct],
            "max_new_tokens": [req.max_new_tokens],
        },
    }

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


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
