#!/usr/bin/env python3
import io
import os
import sys
import time
import hashlib
import json
import random
import uuid

from tts_incremental_decoder import IncrementalDecoder
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
TTS_DEEP_STREAM_CODEGEN_DEVICE = os.environ.get("TTS_DEEP_STREAM_CODEGEN_DEVICE", TTS_DEEP_STREAM_DEVICE)
TTS_DEEP_STREAM_MODEL_DIR = os.environ.get("TTS_DEEP_STREAM_MODEL_DIR", TTS_MODEL_DIR)
TTS_DEEP_STREAM_TOKENIZER_DIR = os.environ.get("TTS_DEEP_STREAM_TOKENIZER_DIR", "")
TTS_DEEP_STREAM_PROCESS = os.environ.get("TTS_DEEP_STREAM_PROCESS", "1").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_METRICS = os.environ.get("TTS_DEEP_STREAM_METRICS", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_GLUE_ONLY_SEC = float(os.environ.get("TTS_DEEP_STREAM_GLUE_ONLY_SEC", "1.0"))
TTS_DEEP_STREAM_REQUEST_TIMEOUT_S = float(os.environ.get("TTS_DEEP_STREAM_REQUEST_TIMEOUT_S", "120"))
TTS_DEEP_STREAM_CODE_TIMEOUT_S = float(os.environ.get("TTS_DEEP_STREAM_CODE_TIMEOUT_S", "120"))
TTS_DEEP_STREAM_IDLE_TIMEOUT_S = float(
    os.environ.get("TTS_DEEP_STREAM_IDLE_TIMEOUT_S", str(TTS_DEEP_STREAM_CODE_TIMEOUT_S))
)
TTS_DEEP_STREAM_LEFT_CONTEXT = int(os.environ.get("TTS_DEEP_STREAM_LEFT_CONTEXT", "25"))
TTS_DEEP_STREAM_DETERMINISTIC = os.environ.get("TTS_DEEP_STREAM_DETERMINISTIC", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_DETERMINISTIC_POLICY = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_POLICY", "seeded"
).strip().lower()
TTS_DEEP_STREAM_SEED_MODE = os.environ.get("TTS_DEEP_STREAM_SEED_MODE", "content").strip().lower()
TTS_DEEP_STREAM_DETERMINISTIC_STRICT = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_STRICT", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_DETERMINISTIC_STRICT_DECODER = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_STRICT_DECODER", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_DETERMINISTIC_SOFT = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_SOFT", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_DETERMINISTIC_SOFT_DECODER = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_SOFT_DECODER", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_TRACE_TIMING = os.environ.get("TTS_DEEP_STREAM_TRACE_TIMING", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_VALIDATE_CODES = os.environ.get("TTS_DEEP_STREAM_VALIDATE_CODES", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_CLAMP_CODES = os.environ.get("TTS_DEEP_STREAM_CLAMP_CODES", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD_DECODER = os.environ.get(
    "TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD_DECODER", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_CODEGEN_STRICT = os.environ.get("TTS_DEEP_STREAM_CODEGEN_STRICT", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_CODEGEN_CUBLAS = os.environ.get("TTS_DEEP_STREAM_CODEGEN_CUBLAS", "").strip()
TTS_DEEP_STREAM_CODEGEN_STRICT_HARD = os.environ.get(
    "TTS_DEEP_STREAM_CODEGEN_STRICT_HARD", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_CODEGEN_FP32 = os.environ.get("TTS_DEEP_STREAM_CODEGEN_FP32", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_DECODER_FP32 = os.environ.get("TTS_DEEP_STREAM_DECODER_FP32", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_PREFILL_PACKETS = int(os.environ.get("TTS_DEEP_STREAM_PREFILL_PACKETS", "0"))
TTS_CODEGEN_DEBUG_TOPK = os.environ.get("TTS_CODEGEN_DEBUG_TOPK", "0").lower() in ("1", "true", "yes")
TTS_CODEGEN_DEBUG_STEP_START = int(os.environ.get("TTS_CODEGEN_DEBUG_STEP_START", "4"))
TTS_CODEGEN_DEBUG_STEP_END = int(
    os.environ.get("TTS_CODEGEN_DEBUG_STEP_END", str(TTS_CODEGEN_DEBUG_STEP_START))
)
TTS_CODEGEN_DEBUG_TOPK_N = int(os.environ.get("TTS_CODEGEN_DEBUG_TOPK_N", "2"))
TTS_DEEP_STREAM_CODEGEN_GENERATOR = os.environ.get("TTS_DEEP_STREAM_CODEGEN_GENERATOR", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_DECODE_MODE = os.environ.get("TTS_DEEP_STREAM_DECODE_MODE", "").strip().lower()
if TTS_DEEP_STREAM_DECODE_MODE in ("incremental", "windowed"):
    TTS_DEEP_STREAM_INCREMENTAL = TTS_DEEP_STREAM_DECODE_MODE == "incremental"
else:
    TTS_DEEP_STREAM_INCREMENTAL = os.environ.get("TTS_DEEP_STREAM_INCREMENTAL", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    TTS_DEEP_STREAM_DECODE_MODE = "incremental" if TTS_DEEP_STREAM_INCREMENTAL else "windowed"
TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER = os.environ.get(
    "TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER", "cache"
).strip().lower()
TTS_DEEP_STREAM_INCREMENTAL_HOLDBACK = int(os.environ.get("TTS_DEEP_STREAM_INCREMENTAL_HOLDBACK", "-1"))
TTS_DEEP_STREAM_INCREMENTAL_TRIM_OFFLINE = os.environ.get(
    "TTS_DEEP_STREAM_INCREMENTAL_TRIM_OFFLINE", "0"
).lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_SYNC_MODE = os.environ.get("TTS_DEEP_STREAM_SYNC_MODE", "").strip().lower()
TTS_DEEP_STREAM_DECODE_EVERY = int(os.environ.get("TTS_DEEP_STREAM_DECODE_EVERY", "1"))
TTS_DEEP_STREAM_PACKET_TRACE = os.environ.get("TTS_DEEP_STREAM_PACKET_TRACE", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_PACKET1_TRACE = os.environ.get("TTS_DEEP_STREAM_PACKET1_TRACE", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_PACKET_SCHEDULE = os.environ.get("TTS_DEEP_STREAM_PACKET_SCHEDULE", "fixed2").strip().lower()
TTS_DEEP_STREAM_DECODE_EVERY_N = int(os.environ.get("TTS_DEEP_STREAM_DECODE_EVERY_N", "1"))
TTS_DEEP_STREAM_DECODER_SYNC = os.environ.get("TTS_DEEP_STREAM_DECODER_SYNC", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_PACKET_DEBUG = os.environ.get("TTS_PACKET_DEBUG", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_DECODE_EVERY_N = int(os.environ.get("TTS_DEEP_STREAM_DECODE_EVERY_N", "1"))
TTS_DEEP_STREAM_DECODER_SYNC = os.environ.get("TTS_DEEP_STREAM_DECODER_SYNC", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_PACKET_DEBUG = os.environ.get("TTS_PACKET_DEBUG", "0").lower() in ("1", "true", "yes")
TTS_CODEGEN_CUDAGRAPH_TALKER = os.environ.get("TTS_CODEGEN_CUDAGRAPH_TALKER", "0").lower() in ("1", "true", "yes")
TTS_CODEGEN_CUDAGRAPH_CP = os.environ.get("TTS_CODEGEN_CUDAGRAPH_CP", "0").lower() in ("1", "true", "yes")
TTS_DECODER_CUDAGRAPH = os.environ.get("TTS_DECODER_CUDAGRAPH", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_CODEGEN_BLOCKING = os.environ.get("TTS_DEEP_STREAM_CODEGEN_BLOCKING", "0").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_OFFLINE_FROM_CODES = os.environ.get("TTS_DEEP_STREAM_OFFLINE_FROM_CODES", "1").lower() in (
    "1",
    "true",
    "yes",
)
TTS_DEEP_STREAM_SILENCE_RMS = float(os.environ.get("TTS_DEEP_STREAM_SILENCE_RMS", "0.003"))
TTS_DEEP_STREAM_SILENCE_PACKETS = int(os.environ.get("TTS_DEEP_STREAM_SILENCE_PACKETS", "6"))
TTS_DEEP_STREAM_SILENCE_PACKETS_P1 = int(os.environ.get("TTS_DEEP_STREAM_SILENCE_PACKETS_P1", "0"))
TTS_DEEP_STREAM_SEGMENT = os.environ.get("TTS_DEEP_STREAM_SEGMENT", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_MAX_SEC_PER_CHAR = float(os.environ.get("TTS_DEEP_STREAM_MAX_SEC_PER_CHAR", "0.35"))
TTS_DEEP_STREAM_MAX_SEC_MIN = float(os.environ.get("TTS_DEEP_STREAM_MAX_SEC_MIN", "4.0"))
TTS_DEEP_STREAM_SEED = int(os.environ.get("TTS_DEEP_STREAM_SEED", "0"))
TTS_CODE_DUMP_ENABLE = os.environ.get("TTS_CODE_DUMP_ENABLE", "0").lower() in ("1", "true", "yes")
TTS_CODE_DUMP_DIR = os.environ.get("TTS_CODE_DUMP_DIR", "/workspace/project 1/25/output/code_dumps")

_deep_model = None
_deep_tokenizer = None
_deep_worker = None

TTS_STARTER_CACHE_ENABLE = os.environ.get("TTS_STARTER_CACHE_ENABLE", "1").lower() in ("1", "true", "yes")
TTS_STARTER_CACHE_TEXTS = os.environ.get("TTS_STARTER_CACHE_TEXTS", "嗯|好的|我在|我听到了|请说")
TTS_STARTER_CACHE_MAX_NEW_TOKENS = int(os.environ.get("TTS_STARTER_CACHE_MAX_NEW_TOKENS", "256"))


def _apply_backend_flags() -> None:
    def _opt_bool(name: str) -> Optional[bool]:
        raw = os.environ.get(name, "").strip().lower()
        if raw == "":
            return None
        if raw in ("1", "true", "yes", "y"):
            return True
        if raw in ("0", "false", "no", "n"):
            return False
        return None

    cudnn_benchmark = _opt_bool("TTS_CUDNN_BENCHMARK")
    if cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = cudnn_benchmark
    cudnn_deterministic = _opt_bool("TTS_CUDNN_DETERMINISTIC")
    if cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = cudnn_deterministic
    cudnn_allow_tf32 = _opt_bool("TTS_CUDNN_ALLOW_TF32")
    if cudnn_allow_tf32 is not None:
        try:
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        except Exception:
            pass
    matmul_allow_tf32 = _opt_bool("TTS_CUDA_MATMUL_ALLOW_TF32")
    if matmul_allow_tf32 is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
        except Exception:
            pass

    if _opt_bool("TTS_CUDNN_TRACE"):
        try:
            print(
                "[CUDNN] enabled="
                f"{torch.backends.cudnn.enabled} "
                f"available={torch.backends.cudnn.is_available()} "
                f"version={torch.backends.cudnn.version()} "
                f"benchmark={torch.backends.cudnn.benchmark} "
                f"deterministic={torch.backends.cudnn.deterministic} "
                f"allow_tf32={getattr(torch.backends.cudnn, 'allow_tf32', 'n/a')}"
            )
            sys.stdout.flush()
        except Exception:
            pass


def _set_global_seed(
    seed: int,
    strict: bool = False,
    soft: bool = False,
    single_thread: bool = False,
    strict_hard: bool = False,
) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if single_thread:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    if strict or soft:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict_hard)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    _apply_backend_flags()


def _ensure_code_dump_dir() -> None:
    if not TTS_CODE_DUMP_ENABLE:
        return
    os.makedirs(TTS_CODE_DUMP_DIR, exist_ok=True)


def _dump_codes(request_tag: str, codes_tensor: torch.Tensor, meta: dict) -> None:
    if not TTS_CODE_DUMP_ENABLE:
        return
    _ensure_code_dump_dir()
    codes_np = codes_tensor.detach().cpu().numpy()
    sha256 = hashlib.sha256(codes_np.tobytes()).hexdigest()
    meta = dict(meta)
    if codes_np.size > 0:
        meta["min_code"] = int(codes_np.min())
        meta["max_code"] = int(codes_np.max())
        codebook_size = int(meta.get("codebook_size", 0) or 0)
        if codebook_size > 0:
            out_of_range = (codes_np < 0) | (codes_np >= codebook_size)
            meta["out_of_range_count"] = int(out_of_range.sum())
    meta.update(
        {
            "sha256": sha256,
            "frames": int(codes_np.shape[0]),
            "codebooks": int(codes_np.shape[1]) if codes_np.ndim > 1 else 1,
        }
    )
    codes_path = os.path.join(TTS_CODE_DUMP_DIR, f"codes_{request_tag}.pt")
    meta_path = os.path.join(TTS_CODE_DUMP_DIR, f"meta_{request_tag}.json")
    torch.save(torch.from_numpy(codes_np), codes_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


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


class _PhaseSync:
    def __init__(self) -> None:
        self._pause_request = Event()
        self._paused = Event()
        self._resume = Event()
        self._done = Event()
        self._lock = Lock()
        self._pause_event: Optional[torch.cuda.Event] = None
        self._resume_event: Optional[torch.cuda.Event] = None

    def request_pause(self) -> None:
        if self._done.is_set():
            return
        self._pause_request.set()

    def should_pause(self) -> bool:
        return self._pause_request.is_set() and not self._done.is_set()

    def mark_paused(self, pause_event: Optional[torch.cuda.Event]) -> None:
        if self._done.is_set():
            return
        with self._lock:
            self._pause_event = pause_event
        self._paused.set()

    def wait_for_pause(self, timeout: Optional[float] = None) -> bool:
        if self._done.is_set():
            return False
        return self._paused.wait(timeout)

    def get_pause_event(self) -> Optional[torch.cuda.Event]:
        with self._lock:
            return self._pause_event

    def resume(self, resume_event: Optional[torch.cuda.Event]) -> None:
        if self._done.is_set():
            return
        with self._lock:
            self._resume_event = resume_event
        self._pause_request.clear()
        self._resume.set()

    def wait_for_resume(self) -> Optional[torch.cuda.Event]:
        if self._done.is_set():
            return None
        self._resume.wait()
        with self._lock:
            resume_event = self._resume_event
            self._resume_event = None
            self._pause_event = None
        self._resume.clear()
        self._paused.clear()
        return resume_event

    def mark_done(self) -> None:
        self._done.set()
        self._pause_request.clear()
        self._resume.set()
        self._paused.set()


class _Packet1Sampler:
    def __init__(self, req_tag: str, log_interval_s: float = 1.0) -> None:
        self.req_tag = req_tag
        self.log_interval_s = log_interval_s
        self._lock = Lock()
        self._last_prod_ts = 0.0
        self._last_cons_ts = 0.0
        self._frames_emitted = 0
        self._decode_calls = 0
        self._packet_idx = 0

    def update_frames_emitted(self, value: int) -> None:
        with self._lock:
            self._frames_emitted = int(value)

    def increment_decode_calls(self) -> int:
        with self._lock:
            self._decode_calls += 1
            return self._decode_calls

    def next_packet_idx(self) -> int:
        with self._lock:
            self._packet_idx += 1
            return self._packet_idx

    def _should_log_prod(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_prod_ts < self.log_interval_s:
                return False
            self._last_prod_ts = now
        return True

    def _should_log_cons(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_cons_ts < self.log_interval_s:
                return False
            self._last_cons_ts = now
        return True

    def log_prod(self, put_ms: float, qsize: int) -> None:
        if not self._should_log_prod():
            return
        with self._lock:
            frames_emitted = self._frames_emitted
        print(
            f"[P1_PROD] req_tag={self.req_tag} put_ms={put_ms:.3f} "
            f"qsize={qsize} frames_emitted={frames_emitted}"
        )
        sys.stdout.flush()

    def log_cons(self, get_ms: float, qsize: int) -> None:
        if not self._should_log_cons():
            return
        with self._lock:
            decode_calls = self._decode_calls
        print(
            f"[P1_CONS] req_tag={self.req_tag} get_ms={get_ms:.3f} "
            f"qsize={qsize} decode_calls={decode_calls}"
        )
        sys.stdout.flush()

    def log_packet_ready(self, pkt_idx: int, t_packet_ready: float) -> None:
        print(
            f"[P1_PKT] req_tag={self.req_tag} pkt_idx={pkt_idx} "
            f"t_packet_ready={t_packet_ready:.6f}"
        )
        sys.stdout.flush()


class _CodecStreamQueue:
    def __init__(
        self,
        cancel_event: Event,
        idle_timeout_s: float = 0.0,
        sync_mode: str = "",
        phase_sync: Optional[_PhaseSync] = None,
        packet1_sampler: Optional[_Packet1Sampler] = None,
    ):
        self._q: Queue = Queue()
        self._closed = False
        self._cancel_event = cancel_event
        self._idle_timeout_s = idle_timeout_s
        self._last_activity = time.time()
        self._sync_mode = sync_mode
        self._phase_sync = phase_sync
        self._packet1_sampler = packet1_sampler

    def _safe_qsize(self) -> int:
        try:
            return int(self._q.qsize())
        except Exception:
            return -1

    def put(self, codes: torch.Tensor):
        if self._closed:
            return
        self._last_activity = time.time()
        codes_event = None
        if self._sync_mode in ("event", "phase") and torch.cuda.is_available():
            try:
                codes_event = torch.cuda.Event(enable_timing=False)
                codes_event.record(torch.cuda.current_stream())
            except Exception:
                codes_event = None
        payload = (codes, codes_event) if codes_event is not None else codes
        t_put0 = time.perf_counter()
        self._q.put(payload)
        put_ms = (time.perf_counter() - t_put0) * 1000.0
        if self._packet1_sampler is not None:
            self._packet1_sampler.log_prod(put_ms, self._safe_qsize())
        if self._phase_sync is not None and self._phase_sync.should_pause():
            if codes_event is None and torch.cuda.is_available():
                try:
                    codes_event = torch.cuda.Event(enable_timing=False)
                    codes_event.record(torch.cuda.current_stream())
                except Exception:
                    codes_event = None
            self._phase_sync.mark_paused(codes_event)
            resume_event = self._phase_sync.wait_for_resume()
            if resume_event is not None and torch.cuda.is_available():
                try:
                    torch.cuda.current_stream().wait_event(resume_event)
                except Exception:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._last_activity = time.time()
        if self._phase_sync is not None:
            self._phase_sync.mark_done()
        self._q.put(None)

    def __iter__(self):
        while True:
            if self._cancel_event.is_set():
                if self._phase_sync is not None:
                    self._phase_sync.mark_done()
                break
            if self._idle_timeout_s > 0 and (time.time() - self._last_activity) > self._idle_timeout_s:
                self._cancel_event.set()
                if self._phase_sync is not None:
                    self._phase_sync.mark_done()
                break
            try:
                t_get0 = time.perf_counter()
                item = self._q.get(timeout=0.1)
                get_ms = (time.perf_counter() - t_get0) * 1000.0
                if self._packet1_sampler is not None:
                    self._packet1_sampler.log_cons(get_ms, self._safe_qsize())
            except Empty:
                if self._packet1_sampler is not None:
                    get_ms = (time.perf_counter() - t_get0) * 1000.0
                    self._packet1_sampler.log_cons(get_ms, self._safe_qsize())
                continue
            self._last_activity = time.time()
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
    if TTS_DEEP_STREAM_CODEGEN_STRICT and TTS_DEEP_STREAM_CODEGEN_CUBLAS:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = TTS_DEEP_STREAM_CODEGEN_CUBLAS
    import torch
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import Qwen3TTSModel

    _set_global_seed(
        TTS_DEEP_STREAM_SEED,
        strict=TTS_DEEP_STREAM_CODEGEN_STRICT,
        strict_hard=TTS_DEEP_STREAM_CODEGEN_STRICT_HARD,
        soft=TTS_DEEP_STREAM_DETERMINISTIC_SOFT,
        single_thread=TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD,
    )
    dtype = torch.float32 if device == "cpu" or TTS_DEEP_STREAM_CODEGEN_FP32 else torch.bfloat16
    model_kwargs = {"device_map": device, "dtype": dtype}
    try:
        model = Qwen3TTSModel.from_pretrained(model_dir, attn_implementation="flash_attention_2", **model_kwargs)
    except Exception:
        model = Qwen3TTSModel.from_pretrained(model_dir, attn_implementation="eager", **model_kwargs)
    try:
        model.model.eval()
    except Exception:
        pass
    if TTS_CODEGEN_DEBUG_TOPK:
        try:
            import inspect
            from vllm_omni.model_executor.models.qwen3_tts import modeling_qwen3_tts as m

            orig_forward = m.Qwen3TTSTalkerForConditionalGeneration.forward

            def _debug_forward(self, *args, **kwargs):
                out = orig_forward(self, *args, **kwargs)
                try:
                    gen_step = kwargs.get("generation_step")
                    if gen_step is None:
                        return out
                    if gen_step < TTS_CODEGEN_DEBUG_STEP_START or gen_step > TTS_CODEGEN_DEBUG_STEP_END:
                        return out
                    req_id = getattr(self, "_debug_request_id", None)
                    if req_id is None:
                        return out
                    logged = getattr(self, "_debug_logged_steps", None)
                    if logged is None:
                        logged = set()
                        self._debug_logged_steps = logged
                    key = (req_id, int(gen_step))
                    if key in logged:
                        return out
                    logged.add(key)
                    logits = getattr(out, "logits", None)
                    if logits is None or logits.numel() == 0:
                        return out
                    vec = logits[0, -1].detach().float().cpu()
                    topk = torch.topk(vec, k=max(1, TTS_CODEGEN_DEBUG_TOPK_N))
                    top1_id = int(topk.indices[0].item())
                    top1_val = float(topk.values[0].item())
                    top2_id = int(topk.indices[1].item()) if topk.values.numel() > 1 else -1
                    top2_val = float(topk.values[1].item()) if topk.values.numel() > 1 else 0.0
                    gap = top1_val - top2_val if top2_id >= 0 else 0.0
                    seed = getattr(self, "_debug_seed", None)
                    text_hash = getattr(self, "_debug_text_hash", "")
                    autocast = torch.is_autocast_enabled()
                    try:
                        param_dtype = next(self.parameters()).dtype
                    except Exception:
                        param_dtype = "unknown"
                    print(
                        f"[CODEGEN_TOPK] req_id={req_id} step={int(gen_step)} "
                        f"top1={top1_id} {top1_val:.6f} top2={top2_id} {top2_val:.6f} "
                        f"gap={gap:.6f} seed={seed} text_hash={text_hash} "
                        f"dtype={param_dtype} autocast={autocast}"
                    )
                    sys.stdout.flush()
                except Exception as e:
                    print(f"[CODEGEN_TOPK] failed to log: {e}")
                    sys.stdout.flush()
                return out

            _debug_forward.__signature__ = inspect.signature(orig_forward)
            m.Qwen3TTSTalkerForConditionalGeneration.forward = _debug_forward
            try:
                model_dtype = next(model.parameters()).dtype
            except Exception:
                model_dtype = "unknown"
            print(f"[CODEGEN_DEBUG] model_dtype={model_dtype} device={device}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[CODEGEN_DEBUG] install_failed err={e}")
            sys.stdout.flush()

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
            seed = payload.get("seed")
            if seed is not None:
                _set_global_seed(
                    int(seed),
                    strict=TTS_DEEP_STREAM_CODEGEN_STRICT,
                    strict_hard=TTS_DEEP_STREAM_CODEGEN_STRICT_HARD,
                    soft=TTS_DEEP_STREAM_DETERMINISTIC_SOFT,
                    single_thread=TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD,
                )
            streamer = _MPCodecStreamer(request_id)
            gen_kwargs = payload.get("gen_kwargs") or {}
            if TTS_CODEGEN_DEBUG_TOPK:
                try:
                    text = payload.get("text", "")
                    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
                    model.model.talker._debug_request_id = request_id
                    model.model.talker._debug_seed = seed
                    model.model.talker._debug_text_hash = text_hash
                except Exception:
                    pass
            codes_list, _ = model.generate_custom_voice_codes(
                text=payload.get("text", ""),
                speaker=payload.get("speaker", "Vivian"),
                language=payload.get("language", "Chinese"),
                instruct=payload.get("instruct", ""),
                codec_streamer=streamer,
                max_new_tokens=payload.get("max_new_tokens", 2048),
                non_streaming_mode=payload.get("non_streaming_mode", False),
                **gen_kwargs,
            )
            codes_cpu_list = []
            for codes in codes_list:
                if isinstance(codes, torch.Tensor):
                    codes_cpu_list.append(codes.detach().cpu())
                else:
                    codes_cpu_list.append(codes)
            out_q.put({"type": "done", "request_id": request_id, "codes": codes_cpu_list})
            if TTS_CODEGEN_DEBUG_TOPK:
                try:
                    model.model.talker._debug_request_id = None
                    model.model.talker._debug_seed = None
                    model.model.talker._debug_text_hash = None
                except Exception:
                    pass
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


def _deep_gen_kwargs(req: "TTSRequest") -> dict:
    if not TTS_DEEP_STREAM_DETERMINISTIC:
        return {}
    if TTS_DEEP_STREAM_DETERMINISTIC_POLICY == "greedy":
        return {
            "do_sample": False,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "subtalker_dosample": False,
            "subtalker_top_p": 1.0,
        }
    return {}


def _stable_seed_from_text(req: "TTSRequest") -> int:
    hasher = hashlib.sha256()
    hasher.update(req.text.encode("utf-8"))
    hasher.update((req.task_type or "").encode("utf-8"))
    hasher.update((req.language or "").encode("utf-8"))
    hasher.update((req.speaker or "").encode("utf-8"))
    hasher.update((req.instruct or "").encode("utf-8"))
    hasher.update(str(req.max_new_tokens).encode("utf-8"))
    hasher.update(str(req.non_streaming_mode).encode("utf-8"))
    return int(hasher.hexdigest()[:8], 16)


def _get_request_seed(req: "TTSRequest") -> Optional[int]:
    if not TTS_DEEP_STREAM_DETERMINISTIC:
        return None
    if req.seed is not None:
        return int(req.seed)
    mode = TTS_DEEP_STREAM_SEED_MODE
    if mode == "fixed":
        return int(TTS_DEEP_STREAM_SEED)
    if mode == "random":
        return random.randint(0, 2**31 - 1)
    return _stable_seed_from_text(req)


def _make_request_generator(seed: Optional[int], device: str) -> Optional[torch.Generator]:
    if seed is None or not TTS_DEEP_STREAM_CODEGEN_GENERATOR:
        return None
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen
    except Exception:
        return None


def _normalize_codes_tensor(codes: torch.Tensor) -> torch.Tensor:
    if codes.dim() == 3 and codes.shape[0] == 1:
        codes = codes.squeeze(0)
    if codes.dim() == 1:
        codes = codes.unsqueeze(0)
    if codes.dim() != 2:
        raise ValueError(f"Unsupported codes shape: {tuple(codes.shape)}")
    return codes


def _get_codebook_size() -> int:
    if _deep_tokenizer is None:
        return 0
    try:
        return int(getattr(_deep_tokenizer.model.decoder.config, "codebook_size", 0) or 0)
    except Exception:
        return 0


def _sanitize_codes(codes: torch.Tensor) -> torch.Tensor:
    codebook_size = _get_codebook_size()
    if codebook_size <= 0:
        return codes
    try:
        min_code = int(codes.min().item())
        max_code = int(codes.max().item())
    except Exception:
        return codes
    if min_code < 0 or max_code >= codebook_size:
        if TTS_DEEP_STREAM_VALIDATE_CODES:
            out_of_range = ((codes < 0) | (codes >= codebook_size)).sum().item()
            print(
                f"[TTS_CODES] out_of_range={int(out_of_range)} "
                f"min={min_code} max={max_code} codebook={codebook_size}"
            )
            sys.stdout.flush()
        if TTS_DEEP_STREAM_CLAMP_CODES:
            codes = codes.clamp(0, codebook_size - 1)
    return codes


def _decode_codes_streaming(
    codes: torch.Tensor,
    packet_tokens: int,
    left_context_frames: int,
) -> tuple[np.ndarray, int]:
    assert _deep_tokenizer is not None
    codes = _normalize_codes_tensor(codes)
    total = codes.shape[0]
    out_chunks: list[np.ndarray] = []
    idx = 0
    while idx < total:
        end = min(idx + packet_tokens, total)
        ctx = min(left_context_frames, idx)
        codes_chunk = codes[idx - ctx : end]
        start_position = max(0, idx - ctx)
        wavs, sr = _deep_tokenizer.decode_streaming(
            codes_chunk, left_context_size=ctx, start_position=start_position
        )
        out_chunks.append(wavs[0])
        idx = end
    if out_chunks:
        audio_np = np.concatenate(out_chunks, axis=0)
    else:
        audio_np = np.zeros((0,), dtype=np.float32)
        sr = _deep_tokenizer.get_output_sample_rate()
    return audio_np, sr


def _decode_codes_incremental(
    codes: torch.Tensor,
    packet_tokens: int,
) -> tuple[np.ndarray, int]:
    assert _deep_tokenizer is not None
    mode = TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER
    if mode not in ("cache", "window", "full"):
        mode = "cache"
    decoder = IncrementalDecoder(_deep_tokenizer, device=TTS_DEEP_STREAM_DEVICE, transformer_mode=mode)
    state = decoder.reset_state()
    codes = _normalize_codes_tensor(codes)
    total = codes.shape[0]
    idx = 0
    chunks: list[np.ndarray] = []
    while idx < total:
        end = min(idx + packet_tokens, total)
        pcm, state = decoder.decode_incremental(codes[idx:end], state)
        if pcm.size > 0:
            chunks.append(pcm)
        idx = end
    if chunks:
        audio_np = np.concatenate(chunks, axis=0)
    else:
        audio_np = np.zeros((0,), dtype=np.float32)
    if state.expected_samples > 0 and len(audio_np) > state.expected_samples:
        audio_np = audio_np[: state.expected_samples]
    return audio_np, int(_deep_tokenizer.get_output_sample_rate())


def _trim_trailing_silence(audio_np: np.ndarray, sr: int) -> np.ndarray:
    if audio_np.size == 0:
        return audio_np
    if TTS_DEEP_STREAM_SILENCE_RMS <= 0:
        return audio_np
    win = max(1, int(sr * 0.2))
    end = len(audio_np)
    trimmed = 0
    while end > win:
        frame = audio_np[end - win : end]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms >= TTS_DEEP_STREAM_SILENCE_RMS:
            break
        end -= win
        trimmed += win
        if trimmed >= sr * 10:
            break
    return audio_np[:end]


def _estimate_max_frames(text: str) -> int:
    try:
        sr = _deep_tokenizer.get_output_sample_rate()
        upsample = _deep_tokenizer.get_decode_upsample_rate()
        frames_per_sec = sr / float(upsample)
    except Exception:
        frames_per_sec = 12.5
    max_sec = max(TTS_DEEP_STREAM_MAX_SEC_MIN, len(text.strip()) * TTS_DEEP_STREAM_MAX_SEC_PER_CHAR)
    return int(max_sec * frames_per_sec)


def _ensure_deep_worker() -> Optional[_DeepCodeWorker]:
    global _deep_worker
    if not TTS_DEEP_STREAM_PROCESS:
        return None
    if _deep_worker is None or not _deep_worker.is_alive():
        _deep_worker = _DeepCodeWorker(TTS_DEEP_STREAM_MODEL_DIR, TTS_DEEP_STREAM_CODEGEN_DEVICE)
    return _deep_worker


def _iter_deep_codes(
    req: "TTSRequest",
    cancel_event: Event,
    phase_sync: Optional[_PhaseSync] = None,
    packet1_sampler: Optional[_Packet1Sampler] = None,
):
    if not TTS_DEEP_STREAM_PROCESS:
        streamer = _CodecStreamQueue(
            cancel_event,
            idle_timeout_s=TTS_DEEP_STREAM_IDLE_TIMEOUT_S,
            sync_mode=TTS_DEEP_STREAM_SYNC_MODE,
            phase_sync=phase_sync,
            packet1_sampler=packet1_sampler,
        )
        gen_kwargs = _deep_gen_kwargs(req)
        seed = _get_request_seed(req)

        def _generate_codes():
            assert _deep_model is not None
            try:
                if seed is not None:
                    _set_global_seed(
                        seed,
                        strict=TTS_DEEP_STREAM_CODEGEN_STRICT,
                        strict_hard=TTS_DEEP_STREAM_CODEGEN_STRICT_HARD,
                        soft=TTS_DEEP_STREAM_DETERMINISTIC_SOFT,
                        single_thread=TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD,
                    )
                _deep_model.generate_custom_voice_codes(
                    text=req.text,
                    speaker=req.speaker,
                    language=req.language,
                    instruct=req.instruct,
                    codec_streamer=streamer,
                    max_new_tokens=req.max_new_tokens,
                    non_streaming_mode=req.non_streaming_mode,
                    **gen_kwargs,
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
    deadline = None
    if TTS_DEEP_STREAM_CODE_TIMEOUT_S > 0:
        deadline = time.time() + TTS_DEEP_STREAM_CODE_TIMEOUT_S
    seed = _get_request_seed(req)
    payload = {
        "text": req.text,
        "speaker": req.speaker,
        "language": req.language,
        "instruct": req.instruct,
        "max_new_tokens": req.max_new_tokens,
        "non_streaming_mode": req.non_streaming_mode,
        "gen_kwargs": _deep_gen_kwargs(req),
        "seed": seed,
    }
    worker.send_generate(request_id, payload)
    while True:
        if cancel_event.is_set():
            worker.abort()
            break
        if deadline is not None and time.time() > deadline:
            cancel_event.set()
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
        gen_kwargs = _deep_gen_kwargs(req)
        assert _deep_model is not None
        seed = _get_request_seed(req)
        if seed is not None:
            gen = _make_request_generator(seed, _deep_model.device if _deep_model is not None else "cpu")
            if gen is not None:
                gen_kwargs = dict(gen_kwargs)
                gen_kwargs["generator"] = gen
        if seed is not None:
            _set_global_seed(
                seed,
                strict=TTS_DEEP_STREAM_CODEGEN_STRICT,
                strict_hard=TTS_DEEP_STREAM_CODEGEN_STRICT_HARD,
                soft=TTS_DEEP_STREAM_DETERMINISTIC_SOFT,
                single_thread=TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD,
            )
        codes_list, _ = _deep_model.generate_custom_voice_codes(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens,
            non_streaming_mode=req.non_streaming_mode,
            **gen_kwargs,
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
        "gen_kwargs": _deep_gen_kwargs(req),
        "seed": _get_request_seed(req),
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


def _collect_codes_streaming(req: "TTSRequest", max_frames: int) -> list[torch.Tensor]:
    cancel_event = Event()
    frames: list[torch.Tensor] = []
    total_frames = 0
    for codes in _iter_deep_codes(req, cancel_event):
        if isinstance(codes, tuple) and len(codes) == 2:
            codes = codes[0]
        if not isinstance(codes, torch.Tensor):
            continue
        codes = _normalize_codes_tensor(codes.to(torch.long))
        for frame in codes:
            frames.append(frame)
            total_frames += 1
            if max_frames > 0 and total_frames >= max_frames:
                cancel_event.set()
                break
        if cancel_event.is_set():
            break
    if not frames:
        raise RuntimeError("No audio codes produced")
    return [torch.stack(frames, dim=0)]


class TTSRequest(BaseModel):
    text: str
    task_type: str = "CustomVoice"
    language: str = "Chinese"
    speaker: str = "Vivian"
    instruct: str = ""
    max_new_tokens: int = 2048
    non_streaming_mode: bool = False
    seed: Optional[int] = None


@app.on_event("startup")
def _init_omni():
    global _omni, _sampling_params, _warmup_done
    if TTS_DEEP_STREAM_ENABLE:
        _set_global_seed(
            TTS_DEEP_STREAM_SEED,
            strict=TTS_DEEP_STREAM_DETERMINISTIC_STRICT_DECODER,
            soft=TTS_DEEP_STREAM_DETERMINISTIC_SOFT_DECODER,
            single_thread=TTS_DEEP_STREAM_DETERMINISTIC_SINGLE_THREAD_DECODER,
        )
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

    dtype = (
        torch.float32
        if TTS_DEEP_STREAM_DEVICE == "cpu" or TTS_DEEP_STREAM_DECODER_FP32
        else torch.bfloat16
    )
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
            _deep_model.model.eval()
        except Exception:
            pass
    try:
        _deep_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir, attn_implementation="flash_attention_2", **tok_kwargs
        )
    except Exception:
        _deep_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir, attn_implementation="eager", **tok_kwargs
        )
    try:
        _deep_tokenizer.model.eval()
    except Exception:
        pass
    if TTS_DEEP_STREAM_PROCESS:
        _ensure_deep_worker()

    # ── CUDA Graph acceleration (flag-controlled, default off) ──
    if (TTS_CODEGEN_CUDAGRAPH_TALKER or TTS_CODEGEN_CUDAGRAPH_CP) and _deep_model is not None:
        try:
            from codegen_cudagraph import install_cudagraph_accelerator
            install_cudagraph_accelerator(
                _deep_model,
                talker_flag=TTS_CODEGEN_CUDAGRAPH_TALKER,
                cp_flag=TTS_CODEGEN_CUDAGRAPH_CP,
            )
        except Exception as _e:
            print(f"[WARNING] CUDA Graph install failed: {_e}")
            import traceback; traceback.print_exc()


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
    if _deep_model is not None and not TTS_DEEP_STREAM_OFFLINE_FROM_CODES:
        gen_kwargs = _deep_gen_kwargs(req)
        wavs, sr = _deep_model.generate_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct,
            max_new_tokens=req.max_new_tokens,
            non_streaming_mode=req.non_streaming_mode,
            **gen_kwargs,
        )
        audio_np = wavs[0]
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.float().detach().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        return audio_np, sr

    max_frames = _estimate_max_frames(req.text)
    if max_frames > 0:
        req = req.copy(update={"max_new_tokens": min(req.max_new_tokens, max_frames)})
    if TTS_DEEP_STREAM_INCREMENTAL:
        codes_list = _collect_codes_streaming(req, max_frames)
    else:
        codes_list = _generate_codes_blocking(req)
    if not codes_list:
        raise RuntimeError("No audio codes produced")
    codes = codes_list[0]
    if isinstance(codes, torch.Tensor):
        codes = codes.to(torch.long)
    codes = _sanitize_codes(codes if isinstance(codes, torch.Tensor) else torch.as_tensor(codes))
    if TTS_CODE_DUMP_ENABLE:
        req_seed = _get_request_seed(req)
        request_tag = f"offline_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        meta = {
            "request_tag": request_tag,
            "text": req.text,
            "task_type": req.task_type,
            "language": req.language,
            "speaker": req.speaker,
            "instruct": req.instruct,
            "max_new_tokens": req.max_new_tokens,
            "max_new_tokens_gen": req.max_new_tokens,
            "max_frames_cap": max_frames,
            "packet_tokens": max(1, TTS_DEEP_STREAM_PACKET_TOKENS),
            "left_context": TTS_DEEP_STREAM_LEFT_CONTEXT,
            "sample_rate": _deep_tokenizer.get_output_sample_rate(),
            "deterministic": TTS_DEEP_STREAM_DETERMINISTIC,
            "seed": req_seed if req_seed is not None else TTS_DEEP_STREAM_SEED,
            "seed_mode": TTS_DEEP_STREAM_SEED_MODE,
            "process": TTS_DEEP_STREAM_PROCESS,
            "impl": "paper",
            "mode": "offline",
        }
        try:
            meta["codebook_size"] = int(getattr(_deep_tokenizer.model.decoder.config, "codebook_size", 0) or 0)
        except Exception:
            meta["codebook_size"] = 0
        _dump_codes(request_tag, codes if isinstance(codes, torch.Tensor) else torch.as_tensor(codes), meta)
    if TTS_DEEP_STREAM_INCREMENTAL:
        audio_np, sr = _decode_codes_incremental(
            codes,
            packet_tokens=max(1, TTS_DEEP_STREAM_PACKET_TOKENS),
        )
    else:
        audio_np, sr = _decode_codes_streaming(
            codes,
            packet_tokens=max(1, TTS_DEEP_STREAM_PACKET_TOKENS),
            left_context_frames=TTS_DEEP_STREAM_LEFT_CONTEXT,
        )
    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.float().detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    if not TTS_DEEP_STREAM_INCREMENTAL or TTS_DEEP_STREAM_INCREMENTAL_TRIM_OFFLINE:
        audio_np = _trim_trailing_silence(audio_np, sr)
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
    if TTS_DEEP_STREAM_SEGMENT:
        segments = _split_text_stream(req.text)
    else:
        segments = [req.text]
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
                        _record_metrics()
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

    gen = _gen()
    first_chunk = b""
    try:
        first_chunk = next(gen)
    except StopIteration:
        gen = iter(())
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


@app.post("/tts/stream_glue")
def synthesize_stream_glue(req: TTSRequest):
    return _synthesize_stream_glue(req)


def _synthesize_stream_glue(req: TTSRequest):
    request_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    t_req_in = time.time()
    first_out: Optional[float] = None
    sr = 24000
    chunk_ms = int(os.environ.get("TTS_STREAM_CHUNK_MS", "30"))
    chunk_ms = max(20, min(40, chunk_ms))
    duration_s = max(0.0, TTS_DEEP_STREAM_GLUE_ONLY_SEC)
    total_samples = int(duration_s * sr)
    chunk_samples = max(1, int(sr * chunk_ms / 1000))
    q: Queue = Queue(maxsize=8)
    cancel_event = Event()
    queue_wait_ms_list: list[float] = []
    emitted_samples = 0

    def _producer():
        produced = 0
        while produced < total_samples and not cancel_event.is_set():
            n = min(chunk_samples, total_samples - produced)
            audio_np = np.zeros((n,), dtype=np.float32)
            q.put(audio_np)
            produced += n
        q.put(None)

    producer = Thread(target=_producer, daemon=True)
    producer.start()

    def _gen() -> Generator[bytes, None, None]:
        nonlocal first_out, emitted_samples
        try:
            while True:
                t_wait = time.time()
                try:
                    item = q.get(timeout=0.1)
                except Empty:
                    if cancel_event.is_set():
                        break
                    continue
                wait_ms = (time.time() - t_wait) * 1000.0
                queue_wait_ms_list.append(wait_ms)
                if item is None:
                    break
                for chunk in _iter_audio_chunks(item, chunk_samples):
                    if first_out is None:
                        first_out = time.time()
                    emitted_samples += len(chunk) // 2
                    yield chunk
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            cancel_event.set()
            raise
        finally:
            cancel_event.set()
            try:
                producer.join(timeout=1.0)
            except Exception:
                pass
            t_done = time.time()
            queue_wait_p95_ms = _percentile(queue_wait_ms_list, 0.95) if queue_wait_ms_list else -1.0
            pcm_seconds_emitted = float(emitted_samples) / float(sr) if sr > 0 else -1.0
            meta = {
                "request_tag": request_tag,
                "glue_only": True,
                "text": req.text,
                "task_type": req.task_type,
                "language": req.language,
                "speaker": req.speaker,
                "instruct": req.instruct,
                "max_new_tokens": req.max_new_tokens,
                "packet_tokens": TTS_DEEP_STREAM_PACKET_TOKENS,
                "sample_rate": sr,
                "process": TTS_DEEP_STREAM_PROCESS,
                "impl": "glue_only",
                "codegen_wall_ms": 0.0,
                "decode_wall_ms": 0.0,
                "glue_wall_ms": (t_done - t_req_in) * 1000.0,
                "queue_wait_p95_ms": queue_wait_p95_ms,
                "pcm_seconds_emitted": pcm_seconds_emitted,
            }
            if first_out is not None:
                meta["t_first_audio"] = first_out
                meta["ttfa_ms"] = (first_out - t_req_in) * 1000.0
            _dump_codes(request_tag, torch.zeros((0, 1), dtype=torch.long), meta)

    gen = _gen()
    first_chunk = b""
    try:
        first_chunk = next(gen)
    except StopIteration:
        gen = iter(())

    def _iter_full():
        if first_chunk:
            yield first_chunk
        yield from gen

    headers = {
        "X-Sample-Rate": str(sr),
        "X-PCM-Format": "s16le",
        "X-Chunk-Ms": str(chunk_ms),
        "X-Deep-Stream": "glue_only",
        "X-Code-Dump-Tag": request_tag,
    }
    return StreamingResponse(_iter_full(), media_type="audio/pcm", headers=headers)


def _synthesize_stream_deep(req: TTSRequest):
    assert _deep_tokenizer is not None
    global _first_request_done
    request_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    t_req_in = time.time()
    first_out: Optional[float] = None
    sr = 24000
    chunk_ms = int(os.environ.get("TTS_STREAM_CHUNK_MS", "30"))
    chunk_ms = max(20, min(40, chunk_ms))
    if TTS_DEEP_STREAM_SEGMENT:
        segments = _split_text_stream(req.text)
    else:
        segments = [req.text]
    if not segments:
        segments = [req.text]
    warm_request = _warmup_done and _first_request_done
    cancel_event = Event()
    cache_hit = False
    cached_pcm = None
    cached_sr = None
    packet_tokens = max(1, TTS_DEEP_STREAM_PACKET_TOKENS)
    packet_schedule = TTS_DEEP_STREAM_PACKET_SCHEDULE
    if packet_schedule not in ("fixed2", "adaptive2to8"):
        packet_schedule = "fixed2"
    if packet_schedule == "adaptive2to8" and packet_tokens != 2:
        print(
            f"[TTS_DEEP] packet_schedule=adaptive2to8 override packet_tokens={packet_tokens} -> 2"
        )
        sys.stdout.flush()
        packet_tokens = 2
    left_context_frames = max(0, TTS_DEEP_STREAM_LEFT_CONTEXT)
    req_seed = _get_request_seed(req)
    metrics = {
        "model_ttfp_ms": -1.0,
        "model_ttf_ms": -1.0,
        "server_ttfa_ms": -1.0,
    }
    codes_all: list[torch.Tensor] = []
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
        t_first_packet_ready: Optional[float] = None
        t_first_decode_done: Optional[float] = None
        t_codegen_done: Optional[float] = None
        t_last_audio: Optional[float] = None
        decode_ms_list: list[float] = []
        decode_wall_ms_total = 0.0
        codegen_iter_wall_ms = 0.0  # Q13: pure time in next(codes_iter)
        deadline = t_req_in + TTS_DEEP_STREAM_REQUEST_TIMEOUT_S
        use_incremental = TTS_DEEP_STREAM_INCREMENTAL
        incremental_decoder: Optional[IncrementalDecoder] = None
        incremental_state = None
        incremental_tail = np.zeros((0,), dtype=np.float32)
        incremental_emitted = 0
        holdback_samples = 0
        prefill_packets = max(0, TTS_DEEP_STREAM_PREFILL_PACKETS)
        prefill_buf: list[tuple[torch.Tensor, object]] = []
        prefill_done = prefill_packets == 0
        decode_every = max(1, TTS_DEEP_STREAM_DECODE_EVERY)
        decode_every_target = decode_every
        if packet_schedule == "adaptive2to8":
            decode_every = 1
            decode_every_target = 1
        decode_every_buf: list[torch.Tensor] = []
        decode_every_count = 0
        decoder_stream = None
        decoder_event_done = None
        decode_every_event = None
        packet_event = None
        packet_trace = {
            "queue_wait_ms": 0.0,
            "decode_calls": 0,
            "codes_frames_max": 0,
            "pcm_samples_total": 0,
            "pcm_samples_max": 0,
        }
        queue_wait_ms_list: list[float] = []
        silence_packets_limit = TTS_DEEP_STREAM_SILENCE_PACKETS
        if packet_tokens == 1:
            silence_packets_limit = TTS_DEEP_STREAM_SILENCE_PACKETS_P1
        packet1_sampler = None
        packet1_enabled = packet_tokens == 1 and (
            TTS_DEEP_STREAM_PACKET_TRACE or TTS_DEEP_STREAM_PACKET1_TRACE
        )
        if packet1_enabled:
            packet1_sampler = _Packet1Sampler(request_tag)
        packet_idx = 0
        packet_idx_total = 0
        phase_sync = None
        phase_supported = (
            TTS_DEEP_STREAM_SYNC_MODE == "phase"
            and not TTS_DEEP_STREAM_PROCESS
            and not TTS_DEEP_STREAM_CODEGEN_BLOCKING
            and TTS_DEEP_STREAM_DEVICE != "cpu"
            and torch.cuda.is_available()
            and TTS_DEEP_STREAM_CODEGEN_DEVICE == TTS_DEEP_STREAM_DEVICE
        )
        if TTS_DEEP_STREAM_SYNC_MODE == "phase" and not phase_supported:
            print(
                "[TTS_DEEP] phase sync disabled "
                "requires process=0, non-blocking codegen, same GPU, and CUDA"
            )
            sys.stdout.flush()
        if phase_supported and use_incremental:
            phase_sync = _PhaseSync()
        if use_incremental:
            mode = TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER
            if mode not in ("cache", "window", "full"):
                mode = "cache"
            incremental_decoder = IncrementalDecoder(
                _deep_tokenizer, device=TTS_DEEP_STREAM_DEVICE, transformer_mode=mode
            )
            # Install decoder CUDA Graph if enabled
            _decoder_graph_accel = None
            if TTS_DECODER_CUDAGRAPH:
                try:
                    from decoder_cudagraph import install_decoder_cudagraph
                    _decoder_graph_accel = install_decoder_cudagraph(
                        incremental_decoder,
                        packet_tokens=TTS_DEEP_STREAM_PACKET_TOKENS,
                    )
                    if _decoder_graph_accel:
                        print(f"[decoder_cudagraph] Installed decoder CUDA Graph accelerator "
                              f"(pre_captured={_decoder_graph_accel._pre_captured})")
                except Exception as e:
                    print(f"[decoder_cudagraph] Failed to install: {e}")
                    import traceback; traceback.print_exc()
            incremental_state = incremental_decoder.reset_state()
            # Reset decoder graph step counter for new request
            if hasattr(incremental_decoder, '_decoder_graph_step_count'):
                incremental_decoder._decoder_graph_step_count = 0
            holdback_samples = TTS_DEEP_STREAM_INCREMENTAL_HOLDBACK
            if holdback_samples < 0:
                try:
                    holdback_samples = int(_deep_tokenizer.get_decode_upsample_rate())
                except Exception:
                    holdback_samples = 0
            holdback_samples = max(0, holdback_samples)

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

        def _note_packet1_decode() -> None:
            if packet1_sampler is not None:
                packet1_sampler.increment_decode_calls()

        def _record_metrics() -> None:
            if metrics["server_ttfa_ms"] < 0 and first_out is not None:
                metrics["server_ttfa_ms"] = (first_out - t_req_in) * 1000.0
            if metrics["model_ttfp_ms"] < 0 and t_before_generate and t_first_packet_ready:
                metrics["model_ttfp_ms"] = (t_first_packet_ready - t_before_generate) * 1000.0
            if metrics["model_ttf_ms"] < 0 and t_before_generate and t_first_decode_done:
                metrics["model_ttf_ms"] = (t_first_decode_done - t_before_generate) * 1000.0

        def _maybe_update_schedule() -> None:
            nonlocal decode_every, decode_every_target, decode_every_count
            if packet_schedule != "adaptive2to8":
                return
            if decode_every_count != 0:
                return
            if decode_every != decode_every_target:
                decode_every = decode_every_target

        def _decode_with_sync(codes_tensor: torch.Tensor, codes_event=None) -> np.ndarray:
            nonlocal incremental_state, decoder_stream, decoder_event_done, phase_sync
            if incremental_decoder is None or incremental_state is None:
                raise RuntimeError("incremental decoder not initialized")
            if TTS_DEEP_STREAM_SYNC_MODE == "sync" and TTS_DEEP_STREAM_DEVICE != "cpu":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            use_event_stream = (
                TTS_DEEP_STREAM_SYNC_MODE in ("event", "phase")
                and TTS_DEEP_STREAM_DEVICE != "cpu"
                and torch.cuda.is_available()
            )
            pre_conv_hook = None
            post_conv_hook = None
            if TTS_DEEP_STREAM_SYNC_MODE == "phase" and phase_sync is not None:
                def _pre_conv() -> None:
                    phase_sync.request_pause()
                    if not phase_sync.wait_for_pause():
                        return
                    pause_event = phase_sync.get_pause_event()
                    if pause_event is None:
                        return
                    try:
                        torch.cuda.current_stream().wait_event(pause_event)
                    except Exception:
                        pass

                def _post_conv() -> None:
                    resume_event = None
                    if torch.cuda.is_available():
                        try:
                            resume_event = torch.cuda.Event(enable_timing=False)
                            resume_event.record(torch.cuda.current_stream())
                        except Exception:
                            resume_event = None
                    phase_sync.resume(resume_event)

                pre_conv_hook = _pre_conv
                post_conv_hook = _post_conv

            if use_event_stream:
                if decoder_stream is None:
                    decoder_stream = torch.cuda.Stream()
                if decoder_event_done is None:
                    decoder_event_done = torch.cuda.Event(enable_timing=False)
                if codes_event is None:
                    try:
                        codes_event = torch.cuda.Event(enable_timing=False)
                        codes_event.record(torch.cuda.current_stream())
                    except Exception:
                        codes_event = None
                with torch.cuda.stream(decoder_stream):
                    if codes_event is not None:
                        try:
                            decoder_stream.wait_event(codes_event)
                        except Exception:
                            pass
                    audio_np, _state = incremental_decoder.decode_incremental(
                        codes_tensor,
                        incremental_state,
                        pre_conv_hook=pre_conv_hook,
                        post_conv_hook=post_conv_hook,
                    )
                    decoder_event_done.record(decoder_stream)
                try:
                    torch.cuda.current_stream().wait_event(decoder_event_done)
                except Exception:
                    pass
            else:
                audio_np, _state = incremental_decoder.decode_incremental(
                    codes_tensor,
                    incremental_state,
                    pre_conv_hook=pre_conv_hook,
                    post_conv_hook=post_conv_hook,
                )
            incremental_state = _state
            return audio_np

        def _decode_incremental_codes(
            codes_tensor: torch.Tensor, codes_event=None
        ) -> Generator[bytes, None, None]:
            nonlocal t_first_packet_ready, t_first_decode_done, incremental_tail, incremental_emitted, silence_packets
            nonlocal incremental_state, first_out, decode_wall_ms_total, t_last_audio
            if t_first_packet_ready is None:
                t_first_packet_ready = time.time()
            t_decode = time.time()
            audio_np = _decode_with_sync(codes_tensor, codes_event=codes_event)
            t_decode_done = time.time()
            decode_wall_ms_total += (t_decode_done - t_decode) * 1000.0
            if t_first_decode_done is None:
                t_first_decode_done = t_decode_done
            _record_decode((t_decode_done - t_decode) * 1000.0)
            _note_packet1_decode()
            if TTS_DEEP_STREAM_PACKET_TRACE:
                packet_trace["decode_calls"] += 1
                packet_trace["codes_frames_max"] = max(packet_trace["codes_frames_max"], int(codes_tensor.shape[0]))
                packet_trace["pcm_samples_total"] += int(audio_np.size)
                packet_trace["pcm_samples_max"] = max(packet_trace["pcm_samples_max"], int(audio_np.size))
            if audio_np.size > 0:
                if incremental_tail.size == 0:
                    incremental_tail = audio_np
                else:
                    incremental_tail = np.concatenate([incremental_tail, audio_np], axis=0)
                max_emit = max(0, incremental_state.expected_samples - holdback_samples)
                can_emit = max_emit - incremental_emitted
                if can_emit > 0:
                    if len(incremental_tail) > can_emit:
                        emit_audio = incremental_tail[:can_emit]
                        incremental_tail = incremental_tail[can_emit:]
                    else:
                        emit_audio = incremental_tail
                        incremental_tail = np.zeros((0,), dtype=np.float32)
                    incremental_emitted += len(emit_audio)
                    if silence_packets_limit > 0:
                        rms = float(np.sqrt(np.mean(emit_audio**2))) if emit_audio.size > 0 else 0.0
                        if rms < TTS_DEEP_STREAM_SILENCE_RMS:
                            silence_packets += 1
                        else:
                            silence_packets = 0
                    chunk_samples = max(1, int(sr * chunk_ms / 1000))
                    for chunk in _iter_audio_chunks(emit_audio, chunk_samples):
                        if _check_deadline():
                            break
                        if cancel_event.is_set():
                            break
                        if first_out is None:
                            first_out = time.time()
                            _record_metrics()
                            print(
                                f"[TTS_DEEP] req_tag={request_tag} t_req_in={t_req_in:.6f} "
                                f"t_after_lock={t_after_lock:.6f} "
                                f"t_before_generate={t_before_generate:.6f} "
                                f"t_first_audio_out={first_out:.6f} "
                                f"lock_wait={(t_after_lock - t_req_in):.3f} "
                                f"gen_to_first={(first_out - t_before_generate):.3f} "
                                f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                                f"segments={len(segments)} chunk_ms={chunk_ms} "
                                f"packet_tokens={packet_tokens} left_ctx={left_context_frames} "
                                f"impl=paper cache_hit={str(cache_hit).lower()}"
                            )
                            sys.stdout.flush()
                        t_last_audio = time.time()
                        yield chunk
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
                    t_last_audio = time.time()
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
                    total_frames = 0
                    frames_emitted = 0
                    silence_packets = 0
                    max_frames = _estimate_max_frames(seg)
                    req_max_new_tokens = req.max_new_tokens
                    if max_frames > 0:
                        req_max_new_tokens = min(req.max_new_tokens, max_frames)
                    codes_buffer: deque[torch.Tensor] = deque(maxlen=left_context_frames + packet_tokens)
                    packet_buf: list[torch.Tensor] = []
                    if t_before_generate is None:
                        t_before_generate = time.time()

                    gen_req = TTSRequest(
                        text=seg,
                        task_type=req.task_type,
                        language=req.language,
                        speaker=req.speaker,
                        instruct=req.instruct,
                        max_new_tokens=req_max_new_tokens,
                        non_streaming_mode=req.non_streaming_mode,
                    )
                    if TTS_DEEP_STREAM_CODEGEN_BLOCKING:
                        codes_iter = _generate_codes_blocking(gen_req)
                    else:
                        codes_iter = _iter_deep_codes(
                            gen_req,
                            cancel_event,
                            phase_sync=phase_sync,
                            packet1_sampler=packet1_sampler,
                        )
                    codes_iter = iter(codes_iter)
                    while True:
                        t_wait = time.time()
                        try:
                            codes = next(codes_iter)
                        except StopIteration:
                            break
                        t_wait_done = time.time()
                        codegen_iter_wall_ms += (t_wait_done - t_wait) * 1000.0
                        if packet1_sampler is not None and TTS_DEEP_STREAM_PROCESS:
                            wait_ms = (t_wait_done - t_wait) * 1000.0
                            packet1_sampler.log_cons(wait_ms, -1)
                        if TTS_DEEP_STREAM_PACKET_TRACE:
                            wait_ms = (t_wait_done - t_wait) * 1000.0
                            packet_trace["queue_wait_ms"] += wait_ms
                            queue_wait_ms_list.append(wait_ms)
                        if _check_deadline():
                            break
                        if cancel_event.is_set():
                            break
                        codes_event = None
                        if isinstance(codes, tuple) and len(codes) == 2:
                            codes, codes_event = codes
                        if (
                            TTS_DEEP_STREAM_SYNC_MODE in ("event", "phase")
                            and codes_event is not None
                            and TTS_DEEP_STREAM_DEVICE != "cpu"
                            and torch.cuda.is_available()
                        ):
                            try:
                                torch.cuda.current_stream().wait_event(codes_event)
                            except Exception:
                                pass
                            codes_event = None
                        if not isinstance(codes, torch.Tensor):
                            continue
                        codes = _normalize_codes_tensor(codes.to(torch.long))
                        codes = _sanitize_codes(codes)
                        if TTS_CODE_DUMP_ENABLE:
                            codes_all.append(codes.detach().cpu())
                        if TTS_DEEP_STREAM_DEVICE != "cpu":
                            codes = codes.to(TTS_DEEP_STREAM_DEVICE)
                        if (
                            TTS_DEEP_STREAM_SYNC_MODE in ("event", "phase")
                            and TTS_DEEP_STREAM_DEVICE != "cpu"
                            and torch.cuda.is_available()
                        ):
                            try:
                                codes_event = torch.cuda.Event(enable_timing=False)
                                codes_event.record(torch.cuda.current_stream())
                            except Exception:
                                codes_event = None

                        packet_event = codes_event
                        for frame in codes:
                            total_frames += 1
                            if packet1_sampler is not None:
                                packet1_sampler.update_frames_emitted(total_frames)
                            if max_frames > 0 and total_frames > max_frames:
                                cancel_event.set()
                                break
                            if use_incremental:
                                packet_buf.append(frame)
                                if len(packet_buf) < packet_tokens:
                                    continue
                                codes_tensor = torch.stack(packet_buf, dim=0)
                                packet_buf.clear()
                                packet_idx_total += 1
                                if packet_schedule == "adaptive2to8" and first_out is not None:
                                    if packet_idx_total >= 4 and decode_every_target < 2:
                                        decode_every_target = 2
                                    if sr > 0 and incremental_emitted > 0 and t_before_generate is not None:
                                        rtf_rolling = (time.time() - t_before_generate) / (
                                            incremental_emitted / float(sr)
                                        )
                                        if rtf_rolling > 1.4 and decode_every_target < 8:
                                            decode_every_target = 8
                                        elif rtf_rolling > 1.2 and decode_every_target < 4:
                                            decode_every_target = 4
                                _maybe_update_schedule()
                                if packet1_sampler is not None:
                                    packet_idx += 1
                                    packet1_sampler.log_packet_ready(packet_idx, time.time())
                                packet_event_local = packet_event
                                if decode_every > 1:
                                    decode_every_buf.append(codes_tensor)
                                    decode_every_count += 1
                                    decode_every_event = packet_event_local
                                    if decode_every_count < decode_every:
                                        continue
                                    codes_tensor = torch.cat(decode_every_buf, dim=0)
                                    decode_every_buf.clear()
                                    decode_every_count = 0
                                    _maybe_update_schedule()
                                    packet_event_local = decode_every_event
                                    decode_every_event = None
                                if not prefill_done:
                                    prefill_buf.append((codes_tensor, packet_event_local))
                                    if len(prefill_buf) < prefill_packets:
                                        continue
                                    prefill_done = True
                                    for pre_codes, pre_event in prefill_buf:
                                        yield from _decode_incremental_codes(
                                            pre_codes, codes_event=pre_event
                                        )
                                    prefill_buf.clear()
                                    continue
                                yield from _decode_incremental_codes(
                                    codes_tensor, codes_event=packet_event_local
                                )
                                if silence_packets_limit > 0 and silence_packets >= silence_packets_limit:
                                    print(
                                        f"[TTS_DEEP] req_tag={request_tag} cancel_reason=silence "
                                        f"packets={silence_packets} limit={silence_packets_limit}"
                                    )
                                    sys.stdout.flush()
                                    cancel_event.set()
                                    break
                                continue

                            codes_buffer.append(frame)
                            if (total_frames - frames_emitted) < packet_tokens:
                                continue
                            new_frames = packet_tokens
                            ctx = min(left_context_frames, max(0, len(codes_buffer) - new_frames))
                            needed = ctx + new_frames
                            if needed <= 0:
                                continue
                            codes_list = list(codes_buffer)[-needed:]
                            codes_tensor = torch.stack(codes_list, dim=0)
                            if t_first_packet_ready is None:
                                t_first_packet_ready = time.time()
                            t_decode = time.time()
                            start_position = max(0, frames_emitted - ctx)
                            wavs, sr = _deep_tokenizer.decode_streaming(
                                codes_tensor, left_context_size=ctx, start_position=start_position
                            )
                            t_decode_done = time.time()
                            decode_wall_ms_total += (t_decode_done - t_decode) * 1000.0
                            if t_first_decode_done is None:
                                t_first_decode_done = t_decode_done
                            _record_decode((t_decode_done - t_decode) * 1000.0)
                            audio_np = wavs[0]
                            if silence_packets_limit > 0:
                                rms = float(np.sqrt(np.mean(audio_np**2))) if audio_np.size > 0 else 0.0
                                if rms < TTS_DEEP_STREAM_SILENCE_RMS:
                                    silence_packets += 1
                                else:
                                    silence_packets = 0
                            chunk_samples = max(1, int(sr * chunk_ms / 1000))
                            for chunk in _iter_audio_chunks(audio_np, chunk_samples):
                                if _check_deadline():
                                    break
                                if cancel_event.is_set():
                                    break
                                if first_out is None:
                                    first_out = time.time()
                                    _record_metrics()
                                    print(
                                        f"[TTS_DEEP] req_tag={request_tag} t_req_in={t_req_in:.6f} "
                                        f"t_after_lock={t_after_lock:.6f} "
                                        f"t_before_generate={t_before_generate:.6f} "
                                        f"t_first_audio_out={first_out:.6f} "
                                        f"lock_wait={(t_after_lock - t_req_in):.3f} "
                                        f"gen_to_first={(first_out - t_before_generate):.3f} "
                                        f"ttfa={first_out - t_req_in:.3f} warm={warm_request} "
                                        f"segments={len(segments)} chunk_ms={chunk_ms} "
                                        f"packet_tokens={packet_tokens} left_ctx={left_context_frames} "
                                        f"impl=paper cache_hit={str(cache_hit).lower()}"
                                    )
                                    sys.stdout.flush()
                                yield chunk
                            frames_emitted += new_frames
                            if packet1_sampler is not None:
                                packet1_sampler.update_frames_emitted(frames_emitted)
                            if silence_packets_limit > 0 and silence_packets >= silence_packets_limit:
                                print(
                                    f"[TTS_DEEP] req_tag={request_tag} cancel_reason=silence "
                                    f"packets={silence_packets} limit={silence_packets_limit}"
                                )
                                sys.stdout.flush()
                                cancel_event.set()
                                break
                            continue

                    if not cancel_event.is_set():
                        t_codegen_done = time.time()

                    if use_incremental and not cancel_event.is_set() and not _check_deadline():
                        if packet_buf:
                            codes_tensor = torch.stack(packet_buf, dim=0)
                            packet_buf.clear()
                            if decode_every > 1:
                                decode_every_buf.append(codes_tensor)
                                decode_every_count += 1
                                decode_every_event = packet_event
                            else:
                                decode_every_buf.append(codes_tensor)
                                decode_every_count += 1
                            if decode_every > 1 and decode_every_count < decode_every:
                                codes_tensor = None
                            else:
                                codes_tensor = torch.cat(decode_every_buf, dim=0)
                                decode_every_buf.clear()
                                decode_every_count = 0
                                packet_event = decode_every_event
                                decode_every_event = None
                            if t_first_packet_ready is None:
                                t_first_packet_ready = time.time()
                            t_decode = time.time()
                            audio_np = np.zeros((0,), dtype=np.float32)
                            if codes_tensor is not None:
                                audio_np = _decode_with_sync(
                                    codes_tensor, codes_event=packet_event
                                )
                            t_decode_done = time.time()
                            if t_first_decode_done is None:
                                t_first_decode_done = t_decode_done
                            _record_decode((t_decode_done - t_decode) * 1000.0)
                            if codes_tensor is not None:
                                _note_packet1_decode()
                            if TTS_DEEP_STREAM_PACKET_TRACE and codes_tensor is not None:
                                packet_trace["decode_calls"] += 1
                                packet_trace["codes_frames_max"] = max(
                                    packet_trace["codes_frames_max"], int(codes_tensor.shape[0])
                                )
                                packet_trace["pcm_samples_total"] += int(audio_np.size)
                                packet_trace["pcm_samples_max"] = max(packet_trace["pcm_samples_max"], int(audio_np.size))
                            if codes_tensor is not None and audio_np.size > 0:
                                if incremental_tail.size == 0:
                                    incremental_tail = audio_np
                                else:
                                    incremental_tail = np.concatenate([incremental_tail, audio_np], axis=0)
                        if decode_every_buf:
                            codes_tensor = torch.cat(decode_every_buf, dim=0)
                            decode_every_buf.clear()
                            decode_every_count = 0
                            _maybe_update_schedule()
                            if t_first_packet_ready is None:
                                t_first_packet_ready = time.time()
                            t_decode = time.time()
                            audio_np = _decode_with_sync(
                                codes_tensor, codes_event=decode_every_event
                            )
                            decode_every_event = None
                            t_decode_done = time.time()
                            if t_first_decode_done is None:
                                t_first_decode_done = t_decode_done
                            _record_decode((t_decode_done - t_decode) * 1000.0)
                            _note_packet1_decode()
                            if TTS_DEEP_STREAM_PACKET_TRACE:
                                packet_trace["decode_calls"] += 1
                                packet_trace["codes_frames_max"] = max(
                                    packet_trace["codes_frames_max"], int(codes_tensor.shape[0])
                                )
                                packet_trace["pcm_samples_total"] += int(audio_np.size)
                                packet_trace["pcm_samples_max"] = max(packet_trace["pcm_samples_max"], int(audio_np.size))
                            if audio_np.size > 0:
                                if incremental_tail.size == 0:
                                    incremental_tail = audio_np
                                else:
                                    incremental_tail = np.concatenate([incremental_tail, audio_np], axis=0)
                        if incremental_decoder is not None and incremental_state is not None:
                            t_decode = time.time()
                            flush_audio, incremental_state = incremental_decoder.finalize(incremental_state)
                            t_decode_done = time.time()
                            if flush_audio.size > 0:
                                _record_decode((t_decode_done - t_decode) * 1000.0)
                                _note_packet1_decode()
                                if TTS_DEEP_STREAM_PACKET_TRACE:
                                    packet_trace["decode_calls"] += 1
                                    packet_trace["pcm_samples_total"] += int(flush_audio.size)
                                    packet_trace["pcm_samples_max"] = max(
                                        packet_trace["pcm_samples_max"], int(flush_audio.size)
                                    )
                                if incremental_tail.size == 0:
                                    incremental_tail = flush_audio
                                else:
                                    incremental_tail = np.concatenate([incremental_tail, flush_audio], axis=0)
                        remaining = (
                            incremental_state.expected_samples - incremental_emitted
                            if incremental_state is not None
                            else 0
                        )
                        if remaining > 0 and incremental_tail.size > 0:
                            emit_audio = incremental_tail[:remaining]
                            chunk_samples = max(1, int(sr * chunk_ms / 1000))
                            for chunk in _iter_audio_chunks(emit_audio, chunk_samples):
                                if _check_deadline():
                                    break
                                if cancel_event.is_set():
                                    break
                                if first_out is None:
                                    first_out = time.time()
                                    _record_metrics()
                                yield chunk
                            incremental_tail = np.zeros((0,), dtype=np.float32)

                    if not use_incremental:
                        remaining = total_frames - frames_emitted
                        if remaining > 0 and not cancel_event.is_set() and not _check_deadline():
                            ctx = min(left_context_frames, max(0, len(codes_buffer) - remaining))
                            needed = ctx + remaining
                            codes_list = list(codes_buffer)[-needed:] if needed > 0 else []
                            if codes_list:
                                codes_tensor = torch.stack(codes_list, dim=0)
                                if t_first_packet_ready is None:
                                    t_first_packet_ready = time.time()
                                t_decode = time.time()
                                start_position = max(0, frames_emitted - ctx)
                                wavs, sr = _deep_tokenizer.decode_streaming(
                                    codes_tensor, left_context_size=ctx, start_position=start_position
                                )
                                t_decode_done = time.time()
                                decode_wall_ms_total += (t_decode_done - t_decode) * 1000.0
                                if t_first_decode_done is None:
                                    t_first_decode_done = t_decode_done
                                _record_decode((t_decode_done - t_decode) * 1000.0)
                                audio_np = wavs[0]
                                if silence_packets_limit > 0:
                                    rms = float(np.sqrt(np.mean(audio_np**2))) if audio_np.size > 0 else 0.0
                                    if rms < TTS_DEEP_STREAM_SILENCE_RMS:
                                        silence_packets += 1
                                    else:
                                        silence_packets = 0
                                chunk_samples = max(1, int(sr * chunk_ms / 1000))
                                for chunk in _iter_audio_chunks(audio_np, chunk_samples):
                                    if _check_deadline():
                                        break
                                    if cancel_event.is_set():
                                        break
                                    if first_out is None:
                                        first_out = time.time()
                                        _record_metrics()
                                    yield chunk
                                if silence_packets_limit > 0 and silence_packets >= silence_packets_limit:
                                    print(
                                        f"[TTS_DEEP] req_tag={request_tag} cancel_reason=silence "
                                        f"packets={silence_packets} limit={silence_packets_limit}"
                                    )
                                    sys.stdout.flush()
                                    cancel_event.set()
                            frames_emitted = total_frames

                    if cancel_event.is_set():
                        break
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            cancel_event.set()
            if phase_sync is not None:
                phase_sync.mark_done()
            raise
        finally:
            if phase_sync is not None:
                phase_sync.mark_done()
            if cancel_event.is_set():
                print(f"[TTS] request_cancelled=true t_req_in={t_req_in:.6f}")
                sys.stdout.flush()

        t_done = time.time()
        _record_metrics()
        # Q13: corrected wall time breakdown
        # loop_wall_ms: full generate loop (codegen + decode + glue interleaved)
        loop_wall_ms = -1.0
        if t_before_generate is not None and t_codegen_done is not None:
            loop_wall_ms = (t_codegen_done - t_before_generate) * 1000.0
        # codegen_wall_ms (LEGACY, kept for compat): same as loop_wall_ms
        codegen_wall_ms = loop_wall_ms
        # codegen_iter_wall_ms: pure time waiting for next(codes_iter) (accumulated above)
        # decode_wall_ms: always-on accumulated decode time (NOT gated on METRICS)
        decode_wall_ms = decode_wall_ms_total if decode_wall_ms_total > 0 else -1.0
        total_wall_ms = (t_done - t_before_generate) * 1000.0 if t_before_generate is not None else -1.0
        # glue_wall_ms: residual = loop - codegen_iter - decode
        glue_wall_ms = -1.0
        if loop_wall_ms >= 0 and codegen_iter_wall_ms >= 0 and decode_wall_ms_total >= 0:
            glue_wall_ms = max(0.0, loop_wall_ms - codegen_iter_wall_ms - decode_wall_ms_total)
        # tail_wall_ms: time after loop (flush + final emit)
        tail_wall_ms = -1.0
        if t_codegen_done is not None:
            tail_wall_ms = (t_done - t_codegen_done) * 1000.0
        queue_wait_p95_ms = _percentile(queue_wait_ms_list, 0.95) if queue_wait_ms_list else -1.0
        pcm_seconds_emitted = float(incremental_emitted) / float(sr) if sr > 0 else -1.0
        if (
            TTS_DEEP_STREAM_TRACE_TIMING
            and t_first_packet_ready
            and t_first_decode_done
            and t_codegen_done
            and first_out is not None
        ):
            overlap = t_first_decode_done < t_codegen_done
            print(
                f"[TTS_TIMING] req_tag={request_tag} "
                f"t_req_in={t_req_in:.6f} "
                f"t_code_first={t_first_packet_ready:.6f} "
                f"t_decode_first={t_first_decode_done:.6f} "
                f"t_first_audio={first_out:.6f} "
                f"t_codegen_done={t_codegen_done:.6f} "
                f"overlap={str(overlap).lower()} "
                f"code_to_decode={(t_first_decode_done - t_first_packet_ready):.3f} "
                f"code_to_done={(t_codegen_done - t_first_packet_ready):.3f}"
            )
            sys.stdout.flush()
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
            f"[TTS_DEEP] req_tag={request_tag} t_req_in={t_req_in:.6f} t_done={t_done:.6f} "
            f"total={(t_done - t_req_in):.3f} warm={warm_request} "
            f"segments={len(segments)} chunk_ms={chunk_ms} packet_tokens={packet_tokens} "
            f"codegen_iter={codegen_iter_wall_ms:.1f}ms decode={decode_wall_ms_total:.1f}ms "
            f"glue={glue_wall_ms:.1f}ms tail={tail_wall_ms:.1f}ms "
            f"loop={loop_wall_ms:.1f}ms"
        )
        sys.stdout.flush()
        _first_request_done = True
        if TTS_CODE_DUMP_ENABLE and codes_all:
            codes_tensor = torch.cat(codes_all, dim=0)
            meta = {
                "request_tag": request_tag,
                "model_dir": TTS_DEEP_STREAM_MODEL_DIR,
                "text": req.text,
                "task_type": req.task_type,
                "language": req.language,
                "speaker": req.speaker,
                "instruct": req.instruct,
                "max_new_tokens": req.max_new_tokens,
                "packet_tokens": packet_tokens,
                "left_context": left_context_frames,
                "sample_rate": _deep_tokenizer.get_output_sample_rate(),
                "deterministic": TTS_DEEP_STREAM_DETERMINISTIC,
                "seed": req_seed if req_seed is not None else TTS_DEEP_STREAM_SEED,
                "seed_mode": TTS_DEEP_STREAM_SEED_MODE,
                "process": TTS_DEEP_STREAM_PROCESS,
                "impl": "paper",
                # Q13: corrected timing breakdown (always-on, low-overhead)
                "codegen_wall_ms": codegen_wall_ms,  # LEGACY: = loop_wall_ms (full loop)
                "codegen_iter_wall_ms": codegen_iter_wall_ms,  # pure next(codes_iter) time
                "decode_wall_ms": decode_wall_ms,  # always-on decode accumulator
                "decode_wall_ms_total": decode_wall_ms_total,  # raw total (no -1 sentinel)
                "loop_wall_ms": loop_wall_ms,  # full generate loop
                "glue_wall_ms": glue_wall_ms,  # residual = loop - codegen_iter - decode
                "tail_wall_ms": tail_wall_ms,  # time after loop (flush + emit)
                "total_wall_ms": total_wall_ms,  # t_done - t_before_generate
                "queue_wait_p95_ms": queue_wait_p95_ms,
                "pcm_seconds_emitted": pcm_seconds_emitted,
                "packet_schedule": packet_schedule,
                "decode_every_final": decode_every,
            }
            if TTS_DEEP_STREAM_PACKET_TRACE:
                meta["queue_wait_ms"] = float(packet_trace["queue_wait_ms"])
                meta["decode_calls"] = int(packet_trace["decode_calls"])
                meta["codes_frames_max"] = int(packet_trace["codes_frames_max"])
                meta["pcm_samples_total"] = int(packet_trace["pcm_samples_total"])
                meta["pcm_samples_max"] = int(packet_trace["pcm_samples_max"])
            if t_first_packet_ready is not None:
                meta["t_code_first"] = t_first_packet_ready
                meta["code_ms"] = (t_first_packet_ready - t_req_in) * 1000.0
            if t_first_decode_done is not None:
                meta["t_decode_first"] = t_first_decode_done
                if t_first_packet_ready is not None:
                    meta["decode_first_ms"] = (t_first_decode_done - t_first_packet_ready) * 1000.0
            if first_out is not None:
                meta["t_first_audio"] = first_out
                meta["ttfa_ms"] = (first_out - t_req_in) * 1000.0
            if t_codegen_done is not None and t_first_packet_ready is not None:
                meta["t_codegen_done"] = t_codegen_done
                meta["code_total_ms"] = (t_codegen_done - t_first_packet_ready) * 1000.0
            try:
                meta["codebook_size"] = int(getattr(_deep_tokenizer.model.decoder.config, "codebook_size", 0) or 0)
            except Exception:
                meta["codebook_size"] = 0
            # CUDA Graph stats (if enabled)
            if TTS_CODEGEN_CUDAGRAPH_TALKER or TTS_CODEGEN_CUDAGRAPH_CP:
                try:
                    from codegen_cudagraph import get_cudagraph_stats, reset_cudagraph_stats
                    meta.update(get_cudagraph_stats())
                    reset_cudagraph_stats()
                except Exception:
                    pass
            # Decoder CUDA Graph stats
            if TTS_DECODER_CUDAGRAPH and _decoder_graph_accel is not None:
                try:
                    meta.update(_decoder_graph_accel.get_stats_dict())
                except Exception:
                    pass
            _dump_codes(request_tag, codes_tensor, meta)

    left_ctx_header = TTS_DEEP_STREAM_LEFT_CONTEXT
    if left_ctx_header < 0 and _deep_tokenizer is not None:
        try:
            left_ctx_header = int(getattr(_deep_tokenizer.model.config.decoder_config, "sliding_window", 72))
        except Exception:
            left_ctx_header = 25

    gen = _gen()
    first_chunk = b""
    try:
        first_chunk = next(gen)
    except StopIteration:
        gen = iter(())

    silence_packets_header = TTS_DEEP_STREAM_SILENCE_PACKETS
    if packet_tokens == 1:
        silence_packets_header = TTS_DEEP_STREAM_SILENCE_PACKETS_P1
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
        "X-Deep-Process": str(TTS_DEEP_STREAM_PROCESS).lower(),
        "X-Deep-Stream-Impl": "paper",
        "X-Deep-Decode-Mode": (
            f"incremental:{TTS_DEEP_STREAM_INCREMENTAL_TRANSFORMER}"
            if TTS_DEEP_STREAM_INCREMENTAL
            else "windowed"
        ),
        "X-Left-Context": str(left_ctx_header),
        "X-Silence-Rms": str(TTS_DEEP_STREAM_SILENCE_RMS),
        "X-Silence-Packets": str(silence_packets_header),
        "X-Model-TTFP-MS": f"{metrics['model_ttfp_ms']:.3f}",
        "X-Model-TTF-MS": f"{metrics['model_ttf_ms']:.3f}",
        "X-Server-TTFA-MS": f"{metrics['server_ttfa_ms']:.3f}",
        "X-Code-Dump-Tag": request_tag,
    }
    def _gen_with_first() -> Generator[bytes, None, None]:
        if first_chunk:
            yield first_chunk
        yield from gen

    return StreamingResponse(_gen_with_first(), media_type="audio/pcm", headers=headers)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)



if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "9000"))
    uvicorn.run(app, host=host, port=port)
