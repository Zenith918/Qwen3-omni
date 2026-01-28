#!/usr/bin/env python3
import io
import os
import sys
import time
from collections import deque
import multiprocessing as mp
from typing import Generator, Optional
from threading import Event, Lock, Thread
from queue import Queue, Empty

import numpy as np
import soundfile as sf
import torch
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
_tts_lock = Lock()
_starter_cache_ready = False
_starter_cache: dict[str, dict[str, object]] = {}

TTS_DEEP_STREAM_ENABLE = os.environ.get("TTS_DEEP_STREAM_ENABLE", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_PACKET_TOKENS = int(os.environ.get("TTS_DEEP_STREAM_PACKET_TOKENS", "4"))
TTS_DEEP_STREAM_DEVICE = os.environ.get("TTS_DEEP_STREAM_DEVICE", "cuda:0")
TTS_DEEP_STREAM_MODEL_DIR = os.environ.get("TTS_DEEP_STREAM_MODEL_DIR", TTS_MODEL_DIR)
TTS_DEEP_STREAM_TOKENIZER_DIR = os.environ.get("TTS_DEEP_STREAM_TOKENIZER_DIR", "")
TTS_DEEP_STREAM_WINDOW_PACKETS = int(os.environ.get("TTS_DEEP_STREAM_WINDOW_PACKETS", "12"))
TTS_DEEP_STREAM_OVERLAP_MS = int(os.environ.get("TTS_DEEP_STREAM_OVERLAP_MS", "200"))
TTS_DEEP_STREAM_PROCESS = os.environ.get("TTS_DEEP_STREAM_PROCESS", "1").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_METRICS = os.environ.get("TTS_DEEP_STREAM_METRICS", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_REQUEST_TIMEOUT_S = float(os.environ.get("TTS_DEEP_STREAM_REQUEST_TIMEOUT_S", "120"))
TTS_DEEP_STREAM_CODE_TIMEOUT_S = float(os.environ.get("TTS_DEEP_STREAM_CODE_TIMEOUT_S", "120"))

_deep_model = None
_deep_tokenizer = None
_deep_worker = None

TTS_STARTER_CACHE_ENABLE = os.environ.get("TTS_STARTER_CACHE_ENABLE", "1").lower() in ("1", "true", "yes")
TTS_STARTER_CACHE_TEXTS = os.environ.get("TTS_STARTER_CACHE_TEXTS", "嗯|好的|我在|我听到了|请说")
TTS_STARTER_CACHE_MAX_NEW_TOKENS = int(os.environ.get("TTS_STARTER_CACHE_MAX_NEW_TOKENS", "256"))


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


def _iter_pcm16_chunks(pcm: bytes, sr: int, chunk_ms: int) -> Generator[bytes, None, None]:
    chunk_samples = max(1, int(sr * chunk_ms / 1000))
    chunk_bytes = chunk_samples * 2
    total = len(pcm)
    idx = 0
    while idx < total:
        end = min(idx + chunk_bytes, total)
        yield pcm[idx:end]
        idx = end


def _percentile(values: list[float], p: float) -> float:
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


class _CodecStreamQueue:
    def __init__(self, cancel_event: Event):
        self._q: Queue = Queue()
        self._closed = False
        self._cancel_event = cancel_event

    def put(self, codes: torch.Tensor):
        if self._closed:
            return
        self._q.put(codes)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._q.put(None)

    def __iter__(self):
        while True:
            if self._cancel_event.is_set():
                break
            try:
                item = self._q.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break
            yield item


class _DeepCodeWorker:
    def __init__(self, model_dir: str, device: str):
        self.model_dir = model_dir
        self.device = device
        self._ctx = mp.get_context("spawn")
        self._cmd_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue(maxsize=16)
        self._process = None
        self._request_id = 0
        self._start()

    def _start(self) -> None:
        self._process = self._ctx.Process(
            target=_deep_code_worker_main,
            args=(self._cmd_q, self._out_q, self.model_dir, self.device),
            daemon=True,
        )
        self._process.start()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def send_generate(self, request_id: int, payload: dict) -> None:
        self._cmd_q.put({"type": "generate", "request_id": request_id, "payload": payload})

    def read_out(self, timeout: float = 0.1) -> Optional[dict]:
        try:
            return self._out_q.get(timeout=timeout)
        except Empty:
            return None

    def drain_out(self) -> None:
        while True:
            try:
                self._out_q.get_nowait()
            except Empty:
                break

    def shutdown(self, force: bool = False) -> None:
        if self._process is None:
            return
        if self._process.is_alive() and not force:
            try:
                self._cmd_q.put({"type": "shutdown"})
                self._process.join(timeout=2.0)
            except Exception:
                pass
        if self._process.is_alive():
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process.join(timeout=2.0)
        self._process = None
        try:
            self._cmd_q.close()
            self._out_q.close()
        except Exception:
            pass

    def abort(self) -> None:
        self.shutdown(force=True)


def _deep_code_worker_main(cmd_q, out_q, model_dir: str, device: str) -> None:
    import torch
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model_kwargs = {"device_map": device, "dtype": dtype}
    try:
        model = Qwen3TTSModel.from_pretrained(model_dir, attn_implementation="flash_attention_2", **model_kwargs)
    except Exception:
        model = Qwen3TTSModel.from_pretrained(model_dir, attn_implementation="eager", **model_kwargs)

    class _MPCodecStreamer:
        def __init__(self, request_id: int):
            self._request_id = request_id

        def put(self, codes: torch.Tensor):
            if not isinstance(codes, torch.Tensor):
                return
            try:
                codes_cpu = codes.detach().cpu()
            except Exception:
                codes_cpu = codes
            out_q.put({"type": "codes", "request_id": self._request_id, "codes": codes_cpu})

        def close(self):
            return

    while True:
        cmd = cmd_q.get()
        if not isinstance(cmd, dict):
            continue
        cmd_type = cmd.get("type")
        if cmd_type == "shutdown":
            break
        if cmd_type != "generate":
            continue
        request_id = cmd.get("request_id", 0)
        payload = cmd.get("payload", {})
        try:
            streamer = _MPCodecStreamer(request_id)
            codes_list, _ = model.generate_custom_voice_codes(
                text=payload.get("text", ""),
                speaker=payload.get("speaker", "Vivian"),
                language=payload.get("language", "Chinese"),
                instruct=payload.get("instruct", ""),
                codec_streamer=streamer,
                max_new_tokens=payload.get("max_new_tokens", 2048),
                non_streaming_mode=payload.get("non_streaming_mode", False),
            )
            codes_cpu_list = []
            for codes in codes_list:
                if isinstance(codes, torch.Tensor):
                    codes_cpu_list.append(codes.detach().cpu())
                else:
                    codes_cpu_list.append(codes)
            out_q.put({"type": "done", "request_id": request_id, "codes": codes_cpu_list})
        except Exception as e:
            out_q.put({"type": "error", "request_id": request_id, "error": str(e)})


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


def _ensure_deep_worker() -> Optional[_DeepCodeWorker]:
    global _deep_worker
    if not TTS_DEEP_STREAM_PROCESS:
        return None
    if _deep_worker is None or not _deep_worker.is_alive():
        _deep_worker = _DeepCodeWorker(TTS_DEEP_STREAM_MODEL_DIR, TTS_DEEP_STREAM_DEVICE)
    return _deep_worker


def _iter_deep_codes(req: "TTSRequest", cancel_event: Event):
    if not TTS_DEEP_STREAM_PROCESS:
        streamer = _CodecStreamQueue(cancel_event)

        def _generate_codes():
            assert _deep_model is not None
            try:
                _deep_model.generate_custom_voice_codes(
                    text=req.text,
                    speaker=req.speaker,
                    language=req.language,
                    instruct=req.instruct,
                    codec_streamer=streamer,
                    max_new_tokens=req.max_new_tokens,
                    non_streaming_mode=req.non_streaming_mode,
                )
            finally:
                streamer.close()

        thread = Thread(target=_generate_codes, daemon=True)
        thread.start()
        yield from streamer
        thread.join(timeout=0.2)
        return

    worker = _ensure_deep_worker()
    if worker is None:
        return
    worker.drain_out()
    request_id = worker.next_request_id()
    payload = {
        "text": req.text,
        "speaker": req.speaker,
        "language": req.language,
        "instruct": req.instruct,
        "max_new_tokens": req.max_new_tokens,
        "non_streaming_mode": req.non_streaming_mode,
    }
    worker.send_generate(request_id, payload)
    while True:
        if cancel_event.is_set():
            worker.abort()
            break
        msg = worker.read_out(timeout=0.1)
        if msg is None:
            continue
        if msg.get("request_id") != request_id:
            continue
        msg_type = msg.get("type")
        if msg_type == "codes":
            yield msg.get("codes")
        elif msg_type == "done":
            break
        elif msg_type == "error":
            raise RuntimeError(msg.get("error", "deep code worker error"))


def _generate_codes_blocking(req: "TTSRequest") -> list[torch.Tensor]:
    if not TTS_DEEP_STREAM_PROCESS:
        assert _deep_model is not None
        codes_list, _ = _deep_model.generate_custom_voice_codes(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens,
            non_streaming_mode=req.non_streaming_mode,
        )
        return codes_list

    worker = _ensure_deep_worker()
    if worker is None:
        raise RuntimeError("deep code worker not available")
    worker.drain_out()
    request_id = worker.next_request_id()
    payload = {
        "text": req.text,
        "speaker": req.speaker,
        "language": req.language,
        "instruct": req.instruct,
        "max_new_tokens": req.max_new_tokens,
        "non_streaming_mode": req.non_streaming_mode,
    }
    worker.send_generate(request_id, payload)
    codes_list = []
    t_start = time.time()
    while True:
        if time.time() - t_start > TTS_DEEP_STREAM_CODE_TIMEOUT_S:
            worker.abort()
            raise RuntimeError("deep code worker timeout")
        msg = worker.read_out(timeout=0.1)
        if msg is None:
            continue
        if msg.get("request_id") != request_id:
            continue
        msg_type = msg.get("type")
        if msg_type == "done":
            codes_list = msg.get("codes", [])
            break
        if msg_type == "error":
            raise RuntimeError(msg.get("error", "deep code worker error"))
    return codes_list


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
    if TTS_DEEP_STREAM_ENABLE:
        _init_deep_stream_backend()
        if TTS_STARTER_CACHE_ENABLE:
            _prime_starter_cache()
        _warmup_done = True
        return
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
    if TTS_STARTER_CACHE_ENABLE:
        _prime_starter_cache()
    _warmup_done = True


@app.on_event("shutdown")
def _close_omni():
    global _omni, _deep_model, _deep_tokenizer, _deep_worker
    try:
        if _omni is not None:
            _omni.close()
    except Exception:
        pass
    _omni = None
    _deep_model = None
    _deep_tokenizer = None
    if _deep_worker is not None:
        try:
            _deep_worker.shutdown(force=True)
        except Exception:
            pass
    _deep_worker = None
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _warmup() -> None:
    assert _omni is not None
    warm_req = TTSRequest(text="warmup")
    inputs = _build_inputs(warm_req)
    for _ in _omni.generate(inputs, [_sampling_params]):
        pass


def _init_deep_stream_backend() -> None:
    global _deep_model, _deep_tokenizer
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    tokenizer_dir = TTS_DEEP_STREAM_TOKENIZER_DIR.strip()
    if not tokenizer_dir:
        candidate = os.path.join(TTS_DEEP_STREAM_MODEL_DIR, "speech_tokenizer")
        tokenizer_dir = candidate if os.path.isdir(candidate) else TTS_DEEP_STREAM_MODEL_DIR

    dtype = torch.float32 if TTS_DEEP_STREAM_DEVICE == "cpu" else torch.bfloat16
    model_kwargs = {"device_map": TTS_DEEP_STREAM_DEVICE, "dtype": dtype}
    tok_kwargs = {"device_map": TTS_DEEP_STREAM_DEVICE, "dtype": dtype}
    if not TTS_DEEP_STREAM_PROCESS:
        try:
            _deep_model = Qwen3TTSModel.from_pretrained(
                TTS_DEEP_STREAM_MODEL_DIR, attn_implementation="flash_attention_2", **model_kwargs
            )
        except Exception:
            _deep_model = Qwen3TTSModel.from_pretrained(
                TTS_DEEP_STREAM_MODEL_DIR, attn_implementation="eager", **model_kwargs
            )
    try:
        _deep_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir, attn_implementation="flash_attention_2", **tok_kwargs
        )
    except Exception:
        _deep_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir, attn_implementation="eager", **tok_kwargs
        )
    if TTS_DEEP_STREAM_PROCESS:
        _ensure_deep_worker()


def _generate_audio_np(req: TTSRequest) -> tuple[np.ndarray, int]:
    if TTS_DEEP_STREAM_ENABLE:
        return _generate_audio_np_deep(req)
    assert _omni is not None
    inputs = _build_inputs(req)
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
    return audio_np, sr


def _generate_audio_np_deep(req: TTSRequest) -> tuple[np.ndarray, int]:
    assert _deep_tokenizer is not None
    if _deep_model is not None:
        wavs, sr = _deep_model.generate_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens,
            non_streaming_mode=req.non_streaming_mode,
        )
        audio_np = wavs[0]
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.float().detach().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        return audio_np, sr

    codes_list = _generate_codes_blocking(req)
    if not codes_list:
        raise RuntimeError("No audio codes produced")
    codes = codes_list[0]
    if isinstance(codes, torch.Tensor):
        codes = codes.to(torch.long)
    wavs, sr = _deep_tokenizer.decode([{"audio_codes": codes}])
    audio_np = wavs[0]
    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.float().detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    return audio_np, sr


def _prime_starter_cache() -> None:
    global _starter_cache_ready
    texts = [t.strip() for t in TTS_STARTER_CACHE_TEXTS.split("|") if t.strip()]
    if not texts:
        _starter_cache_ready = True
        return
    try:
        with _tts_lock:
            for text in texts:
                try:
                    req = TTSRequest(
                        text=text,
                        max_new_tokens=TTS_STARTER_CACHE_MAX_NEW_TOKENS,
                        non_streaming_mode=False,
                    )
                    audio_np, sr = _generate_audio_np(req)
                    pcm = _audio_to_pcm16(audio_np)
                    _starter_cache[text] = {
                        "pcm": pcm,
                        "sr": sr,
                        "samples": len(audio_np),
                    }
                except Exception as e:
                    print(f"[TTS_CACHE] failed text={text} err={e}")
                    sys.stdout.flush()
    finally:
        _starter_cache_ready = True
        print(f"[TTS_CACHE] primed={len(_starter_cache)}")
        sys.stdout.flush()


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    start = time.time()
    if TTS_DEEP_STREAM_ENABLE:
        audio_np, sr = _generate_audio_np_deep(req)
    else:
        assert _omni is not None
        inputs = _build_inputs(req)

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
    if TTS_DEEP_STREAM_ENABLE:
        return _synthesize_stream_deep(req)
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
    cancel_event = Event()
    cache_hit = False
    cached_pcm = None
    cached_sr = None
    if _starter_cache_ready and segments:
        key = segments[0].strip()
        cached = _starter_cache.get(key)
        if cached is not None:
            cache_hit = True
            cached_pcm = cached.get("pcm")
            cached_sr = cached.get("sr")
            if isinstance(cached_sr, int):
                sr = cached_sr

    def _gen() -> Generator[bytes, None, None]:
        global _first_request_done
        nonlocal first_out, sr, sent_samples
        t_after_lock = None
        t_before_generate = None
        try:
            if cache_hit and cached_pcm and cached_sr:
                for chunk in _iter_pcm16_chunks(cached_pcm, cached_sr, chunk_ms):
                    if first_out is None:
                        first_out = time.time()
                        print(
                            f"[TTS_CACHE] t_req_in={t_req_in:.6f} "
                            f"t_first_audio_out={first_out:.6f} "
                            f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                            f"segment=starter cache_hit=true chunk_ms={chunk_ms}"
                        )
                        sys.stdout.flush()
                    yield chunk

            with _tts_lock:
                t_after_lock = time.time()
                for seg_idx, seg in enumerate(segments):
                    if cancel_event.is_set():
                        break
                    if cache_hit and seg_idx == 0:
                        continue
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
                    if t_before_generate is None:
                        t_before_generate = time.time()
                    for stage_outputs in _omni.generate(inputs, [_sampling_params]):
                        if cancel_event.is_set():
                            break
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
                                if cancel_event.is_set():
                                    break
                                if first_out is None:
                                    first_out = time.time()
                                    print(
                                        f"[TTS] t_req_in={t_req_in:.6f} "
                                        f"t_after_lock={t_after_lock:.6f} "
                                        f"t_before_generate={t_before_generate:.6f} "
                                        f"t_first_audio_out={first_out:.6f} "
                                        f"lock_wait={(t_after_lock - t_req_in):.3f} "
                                        f"gen_to_first={(first_out - t_before_generate):.3f} "
                                        f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                                        f"segments={len(segments)} chunk_ms={chunk_ms} cache_hit={str(cache_hit).lower()}"
                                    )
                                    sys.stdout.flush()
                                yield chunk
                            sent_samples = len(audio_np)
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            cancel_event.set()
            raise
        finally:
            if cancel_event.is_set():
                print(f"[TTS] request_cancelled=true t_req_in={t_req_in:.6f}")
                sys.stdout.flush()

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
        "X-Cache-Starter": str(cache_hit).lower(),
    }
    return StreamingResponse(_gen(), media_type="audio/pcm", headers=headers)


def _synthesize_stream_deep(req: TTSRequest):
    assert _deep_tokenizer is not None
    global _first_request_done
    t_req_in = time.time()
    first_out: Optional[float] = None
    sr = 24000
    chunk_ms = int(os.environ.get("TTS_STREAM_CHUNK_MS", "30"))
    chunk_ms = max(20, min(40, chunk_ms))
    segments = _split_text_stream(req.text)
    if not segments:
        segments = [req.text]
    warm_request = _warmup_done and _first_request_done
    cancel_event = Event()
    cache_hit = False
    cached_pcm = None
    cached_sr = None
    packet_tokens = max(1, TTS_DEEP_STREAM_PACKET_TOKENS)
    if _starter_cache_ready and segments:
        key = segments[0].strip()
        cached = _starter_cache.get(key)
        if cached is not None:
            cache_hit = True
            cached_pcm = cached.get("pcm")
            cached_sr = cached.get("sr")
            if isinstance(cached_sr, int):
                sr = cached_sr

    def _gen() -> Generator[bytes, None, None]:
        global _first_request_done
        nonlocal first_out, sr
        t_after_lock = None
        t_before_generate = None
        decode_ms_list: list[float] = []
        deadline = t_req_in + TTS_DEEP_STREAM_REQUEST_TIMEOUT_S

        def _check_deadline() -> bool:
            if TTS_DEEP_STREAM_REQUEST_TIMEOUT_S <= 0:
                return False
            if time.time() > deadline:
                cancel_event.set()
                return True
            return False

        def _record_decode(ms: float) -> None:
            if TTS_DEEP_STREAM_METRICS:
                decode_ms_list.append(ms)
        try:
            if cache_hit and cached_pcm and cached_sr:
                for chunk in _iter_pcm16_chunks(cached_pcm, cached_sr, chunk_ms):
                    if first_out is None:
                        first_out = time.time()
                        print(
                            f"[TTS_CACHE] t_req_in={t_req_in:.6f} "
                            f"t_first_audio_out={first_out:.6f} "
                            f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                            f"segment=starter cache_hit=true chunk_ms={chunk_ms}"
                        )
                        sys.stdout.flush()
                    yield chunk

            with _tts_lock:
                t_after_lock = time.time()
                for seg_idx, seg in enumerate(segments):
                    if _check_deadline():
                        break
                    if cancel_event.is_set():
                        break
                    if cache_hit and seg_idx == 0:
                        continue
                    pending_tail = np.zeros((0,), dtype=np.float32)
                    overlap_samples = None
                    window_packets = max(1, TTS_DEEP_STREAM_WINDOW_PACKETS)
                    window_tokens = max(packet_tokens, window_packets * packet_tokens)
                    codes_accum: deque[torch.Tensor] = deque(maxlen=window_tokens + packet_tokens)
                    if t_before_generate is None:
                        t_before_generate = time.time()

                    def _emit_with_overlap(new_audio: np.ndarray) -> list[np.ndarray]:
                        nonlocal pending_tail, overlap_samples
                        if new_audio.size == 0:
                            return []
                        if overlap_samples is None or overlap_samples <= 0:
                            return [new_audio]
                        out_chunks: list[np.ndarray] = []
                        if pending_tail.size == 0:
                            if len(new_audio) <= overlap_samples:
                                pending_tail = new_audio
                                return []
                            out_chunks.append(new_audio[:-overlap_samples])
                            pending_tail = new_audio[-overlap_samples:]
                            return out_chunks
                        n = min(overlap_samples, len(pending_tail), len(new_audio))
                        if n > 0:
                            fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                            cross = pending_tail[-n:] * (1.0 - fade) + new_audio[:n] * fade
                            if len(pending_tail) > n:
                                out_chunks.append(np.concatenate([pending_tail[:-n], cross], axis=0))
                            else:
                                out_chunks.append(cross)
                            remain = new_audio[n:]
                        else:
                            out_chunks.append(pending_tail)
                            remain = new_audio
                        if len(remain) <= overlap_samples:
                            pending_tail = remain
                        else:
                            out_chunks.append(remain[:-overlap_samples])
                            pending_tail = remain[-overlap_samples:]
                        return out_chunks

                    def _align_window_audio(full_audio: np.ndarray) -> np.ndarray:
                        nonlocal pending_tail, overlap_samples
                        if full_audio.size == 0:
                            return full_audio
                        if overlap_samples is None or overlap_samples <= 0:
                            return full_audio
                        if pending_tail.size == 0:
                            return full_audio
                        if len(full_audio) <= overlap_samples:
                            return full_audio
                        search_len = min(len(full_audio), max(overlap_samples * 3, overlap_samples + 1))
                        if search_len <= overlap_samples:
                            return full_audio
                        tail = pending_tail
                        if len(tail) > search_len:
                            return full_audio
                        search = full_audio[:search_len]
                        tail_z = tail - np.mean(tail)
                        search_z = search - np.mean(search)
                        corr = np.correlate(search_z, tail_z, mode="valid")
                        offset = int(np.argmax(corr)) if corr.size > 0 else 0
                        return full_audio[offset:]

                    for codes in _iter_deep_codes(
                        TTSRequest(
                            text=seg,
                            task_type=req.task_type,
                            language=req.language,
                            speaker=req.speaker,
                            instruct=req.instruct,
                            max_new_tokens=req.max_new_tokens,
                            non_streaming_mode=req.non_streaming_mode,
                        ),
                        cancel_event,
                    ):
                        if _check_deadline():
                            break
                        if cancel_event.is_set():
                            break
                        if isinstance(codes, torch.Tensor):
                            if codes.dim() == 2 and codes.shape[0] == 1:
                                codes = codes.squeeze(0)
                            codes = codes.to(torch.long)
                            if TTS_DEEP_STREAM_DEVICE != "cpu":
                                codes = codes.to(TTS_DEEP_STREAM_DEVICE)
                            codes_accum.append(codes)
                        else:
                            continue

                        if len(codes_accum) % packet_tokens != 0:
                            continue

                        window_list = list(codes_accum)[-window_tokens:]
                        codes_tensor = torch.stack(window_list, dim=0)
                        t_decode = time.time()
                        wavs, sr = _deep_tokenizer.decode([{"audio_codes": codes_tensor}])
                        _record_decode((time.time() - t_decode) * 1000.0)
                        audio_np = wavs[0]
                        if isinstance(audio_np, torch.Tensor):
                            audio_np = audio_np.float().detach().cpu().numpy()
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()

                        if overlap_samples is None:
                            overlap_samples = int(sr * TTS_DEEP_STREAM_OVERLAP_MS / 1000)
                        aligned_audio = _align_window_audio(audio_np)
                        window_token_count = max(1, codes_tensor.shape[0])
                        new_samples = int(round(len(aligned_audio) * packet_tokens / window_token_count))
                        new_samples = max(1, min(len(aligned_audio), new_samples))
                        new_audio = aligned_audio[-new_samples:]
                        chunk_samples = max(1, int(sr * chunk_ms / 1000))
                        for emit_audio in _emit_with_overlap(new_audio):
                            for chunk in _iter_audio_chunks(emit_audio, chunk_samples):
                                if _check_deadline():
                                    break
                                if cancel_event.is_set():
                                    break
                                if first_out is None:
                                    first_out = time.time()
                                    print(
                                        f"[TTS_DEEP] t_req_in={t_req_in:.6f} "
                                        f"t_after_lock={t_after_lock:.6f} "
                                        f"t_before_generate={t_before_generate:.6f} "
                                        f"t_first_audio_out={first_out:.6f} "
                                        f"lock_wait={(t_after_lock - t_req_in):.3f} "
                                        f"gen_to_first={(first_out - t_before_generate):.3f} "
                                        f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                                        f"segments={len(segments)} chunk_ms={chunk_ms} "
                                        f"packet_tokens={packet_tokens} window_packets={window_packets} "
                                        f"overlap_ms={TTS_DEEP_STREAM_OVERLAP_MS} cache_hit={str(cache_hit).lower()}"
                                    )
                                    sys.stdout.flush()
                                yield chunk

                    remaining_tokens = len(codes_accum) % packet_tokens
                    if codes_accum and remaining_tokens:
                        window_list = list(codes_accum)[-window_tokens:]
                        codes_tensor = torch.stack(window_list, dim=0)
                        t_decode = time.time()
                        wavs, sr = _deep_tokenizer.decode([{"audio_codes": codes_tensor}])
                        _record_decode((time.time() - t_decode) * 1000.0)
                        audio_np = wavs[0]
                        if isinstance(audio_np, torch.Tensor):
                            audio_np = audio_np.float().detach().cpu().numpy()
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                        if overlap_samples is None:
                            overlap_samples = int(sr * TTS_DEEP_STREAM_OVERLAP_MS / 1000)
                        aligned_audio = _align_window_audio(audio_np)
                        window_token_count = max(1, codes_tensor.shape[0])
                        new_samples = int(round(len(aligned_audio) * remaining_tokens / window_token_count))
                        new_samples = max(1, min(len(aligned_audio), new_samples))
                        new_audio = aligned_audio[-new_samples:]
                        chunk_samples = max(1, int(sr * chunk_ms / 1000))
                        for emit_audio in _emit_with_overlap(new_audio):
                            for chunk in _iter_audio_chunks(emit_audio, chunk_samples):
                                if _check_deadline():
                                    break
                                if cancel_event.is_set():
                                    break
                                if first_out is None:
                                    first_out = time.time()
                                yield chunk

                    if pending_tail.size > 0:
                        chunk_samples = max(1, int(sr * chunk_ms / 1000))
                        for chunk in _iter_audio_chunks(pending_tail, chunk_samples):
                            if _check_deadline():
                                break
                            if cancel_event.is_set():
                                break
                            if first_out is None:
                                first_out = time.time()
                            yield chunk

                    if cancel_event.is_set():
                        break
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            cancel_event.set()
            raise
        finally:
            if cancel_event.is_set():
                print(f"[TTS] request_cancelled=true t_req_in={t_req_in:.6f}")
                sys.stdout.flush()

        t_done = time.time()
        if TTS_DEEP_STREAM_METRICS and decode_ms_list:
            if len(decode_ms_list) >= 2:
                x = np.arange(1, len(decode_ms_list) + 1, dtype=np.float32)
                y = np.array(decode_ms_list, dtype=np.float32)
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = 0.0
            print(
                f"[TTS_DEEP] decode_ms p50={_percentile(decode_ms_list, 0.5):.3f} "
                f"p95={_percentile(decode_ms_list, 0.95):.3f} "
                f"max={max(decode_ms_list):.3f} slope={slope:.6f} n={len(decode_ms_list)}"
            )
            sys.stdout.flush()
        print(
            f"[TTS_DEEP] t_req_in={t_req_in:.6f} t_done={t_done:.6f} "
            f"total={(t_done - t_req_in):.3f} warm={warm_request} "
            f"segments={len(segments)} chunk_ms={chunk_ms} packet_tokens={packet_tokens}"
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
        "X-Cache-Starter": str(cache_hit).lower(),
        "X-Deep-Stream": "true",
        "X-Packet-Tokens": str(packet_tokens),
        "X-Window-Packets": str(TTS_DEEP_STREAM_WINDOW_PACKETS),
        "X-Overlap-Ms": str(TTS_DEEP_STREAM_OVERLAP_MS),
        "X-Deep-Process": str(TTS_DEEP_STREAM_PROCESS).lower(),
    }
    return StreamingResponse(_gen(), media_type="audio/pcm", headers=headers)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
