#!/usr/bin/env python3
"""
LiveKit Voice Agent v3 — D17 low-latency

D17 changes:
  P0-2: STT→LLM prefetch (overlap LLM request with framework processing)
  P0-3: Two-phase LLM prompt (fast first utterance ≤12 tokens)
  P0-4: TTS first-frame hot path (flush first chunk immediately)
  P0-5: Model warm-up on agent start
"""

import asyncio
import base64
import io
import json
import logging
import os
import queue as _queue
import sys
import time
import uuid
import wave
import functools
import threading
from pathlib import Path

import numpy as np
import requests

from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    WorkerOptions,
    WorkerType,
)
from livekit.agents import llm, tts, stt
from livekit.agents.types import APIConnectOptions
from livekit.plugins import silero as silero_plugin

# ── 路径 ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clients"))

from noise_robust_vad import NoiseRobustVAD
from endpointing_controller import EndpointingController

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "serena")

SAMPLE_RATE_TTS = int(os.environ.get("SAMPLE_RATE_TTS", "24000"))
TTS_FRAME_MS = int(os.environ.get("TTS_FRAME_MS", "20"))
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "200"))
VAD_PREFIX_MS = int(os.environ.get("VAD_PREFIX_MS", "300"))
MIN_ENDPOINTING = float(os.environ.get("MIN_ENDPOINTING", "0.3"))

# D15 P0-2: Mode-based configuration (turn_taking or duplex)
MODE = os.environ.get("MODE", "turn_taking")

# D17: Aggressive endpointing — base values lowered further, controller adds dynamic hold
#   Total base for clean short speech: 200ms + 100ms = 300ms (D16 was 500ms)
#   EndpointingController adds 0-300ms extra hold for noisy/long utterances
VAD_ENDPOINTING_SILENCE_MS = int(os.environ.get("VAD_ENDPOINTING_SILENCE_MS",
    "200" if MODE == "turn_taking" else "200"))
ENDPOINTING_DELAY_MS = int(os.environ.get("ENDPOINTING_DELAY_MS",
    "100" if MODE == "turn_taking" else "100"))
ADAPTIVE_ENDPOINTING = os.environ.get("ADAPTIVE_ENDPOINTING",
    "1" if MODE == "turn_taking" else "0") == "1"
BARGEIN_MIN_SPEECH_MS = int(os.environ.get("BARGEIN_MIN_SPEECH_MS", "120"))
BARGEIN_ACTIVATION_THRESHOLD = float(os.environ.get("BARGEIN_ACTIVATION_THRESHOLD",
    "0.7" if MODE == "turn_taking" else "0.5"))
NOISE_GATE_ENABLED = os.environ.get("NOISE_GATE_ENABLED",
    "1" if MODE == "turn_taking" else "0") == "1"

# D14 compat: TURN_TAKING_MIN_SILENCE_MS still works as override (sets total delay)
TURN_TAKING_MIN_SILENCE_MS = int(os.environ.get("TURN_TAKING_MIN_SILENCE_MS", "0"))
if TURN_TAKING_MIN_SILENCE_MS > 0:
    VAD_ENDPOINTING_SILENCE_MS = max(200, TURN_TAKING_MIN_SILENCE_MS // 2)
    ENDPOINTING_DELAY_MS = TURN_TAKING_MIN_SILENCE_MS - VAD_ENDPOINTING_SILENCE_MS

MIN_ENDPOINTING = ENDPOINTING_DELAY_MS / 1000.0

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_HISTORY_TURNS = int(os.environ.get("LLM_HISTORY_TURNS", "10"))
ENABLE_CONTINUATION = os.environ.get("ENABLE_CONTINUATION", "1") == "1"

# D17 P0-2: STT→LLM prefetch (fire LLM request immediately after STT completes)
FAST_LANE_ENABLED = os.environ.get("FAST_LANE_ENABLED",
    "1" if MODE == "turn_taking" else "0") == "1"
# D17 P0-3: Two-phase first-response — short first utterance then continuation
LLM_MAX_TOKENS_FIRST = int(os.environ.get("LLM_MAX_TOKENS_FIRST", "24"))
LLM_TEMPERATURE_FIRST = float(os.environ.get("LLM_TEMPERATURE_FIRST", "0.2"))

TRACE_DIR = os.environ.get("TRACE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))

# D17: Reusable HTTP sessions for connection pooling (avoids TCP handshake per call)
_http_llm = requests.Session()
_http_tts = requests.Session()

# D7/D8: Ring1 Pre-RTC 录音
CAPTURE_PRE_RTC = os.environ.get("CAPTURE_PRE_RTC", "1") == "1"
PRE_RTC_BASE_DIR = os.environ.get("PRE_RTC_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "pre_rtc"))

# ══════════════════════════════════════════════════════════════
# Trace 收集器（F5: 同时支持 AutoRTC 和手动连接）
# ══════════════════════════════════════════════════════════════
class TraceCollector:
    def __init__(self):
        self._traces = {}
        self._trace_file = os.path.join(TRACE_DIR, "day5_e2e_traces.jsonl")
        os.makedirs(TRACE_DIR, exist_ok=True)

    def new_trace(self, trace_id: str = None, extra: dict = None) -> str:
        tid = trace_id or str(uuid.uuid4())[:8]
        if tid not in self._traces:
            self._traces[tid] = {"trace_id": tid, "created_at": time.time()}
        if extra:
            self._traces[tid].update(extra)
        return tid

    def mark(self, trace_id: str, key: str, ts: float = None):
        if not trace_id:
            return
        if trace_id not in self._traces:
            self.new_trace(trace_id)
        # F5: 允许覆盖（不再 skip 已存在的 key），保证手动连接也能打点
        self._traces[trace_id][key] = ts or time.time()

    def get(self, trace_id: str) -> dict:
        return self._traces.get(trace_id, {})

    def is_autortc(self, trace_id: str) -> bool:
        """判断是否来自 AutoRTC DataChannel"""
        t = self._traces.get(trace_id, {})
        return t.get("trace_from") == "data_channel"

    def finalize(self, trace_id: str, extra: dict = None):
        if trace_id not in self._traces:
            return
        t = self._traces.pop(trace_id)
        if extra:
            t.update(extra)

        def delta(a, b):
            if a in t and b in t:
                return round((t[b] - t[a]) * 1000, 1)
            return None

        t["latency_ms"] = {
            "vad_end_to_stt_done": delta("t_agent_vad_end", "t_stt_done"),
            "stt_done_to_llm_first": delta("t_stt_done", "t_llm_first_token"),
            "llm_first_to_tts_first": delta("t_llm_first_token", "t_tts_first_chunk"),
            "tts_first_to_publish": delta("t_tts_first_chunk", "t_agent_publish_first_frame"),
            "vad_end_to_publish": delta("t_agent_vad_end", "t_agent_publish_first_frame"),
            "vad_end_to_tts_first": delta("t_agent_vad_end", "t_tts_first_chunk"),
        }

        # D17 P0-1: processing breakdown (all six segments for report)
        t["proc_breakdown"] = {
            "vad_to_stt_ms": t["latency_ms"]["vad_end_to_stt_done"],
            "stt_to_llm_ms": t["latency_ms"]["stt_done_to_llm_first"],
            "llm_to_tts_ms": t["latency_ms"]["llm_first_to_tts_first"],
            "tts_to_pub_ms": t["latency_ms"]["tts_first_to_publish"],
            "vad_to_pub_ms": t["latency_ms"]["vad_end_to_publish"],
        }

        try:
            with open(self._trace_file, "a") as f:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[Trace] Write error: {e}")

        logger.info(f"[Trace] {trace_id} | "
                     f"vad→stt={t['latency_ms'].get('vad_end_to_stt_done')}ms "
                     f"stt→llm={t['latency_ms'].get('stt_done_to_llm_first')}ms "
                     f"llm→tts={t['latency_ms'].get('llm_first_to_tts_first')}ms "
                     f"tts→pub={t['latency_ms'].get('tts_first_to_publish')}ms "
                     f"E2E={t['latency_ms'].get('vad_end_to_publish')}ms")


_tracer = TraceCollector()


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def audio_frames_to_wav_base64(frames, target_sr=16000):
    if isinstance(frames, rtc.AudioFrame):
        frames = [frames]
    all_samples = []
    src_sr = None
    for f in frames:
        src_sr = f.sample_rate
        data = np.frombuffer(f.data, dtype=np.int16)
        if f.num_channels > 1:
            data = data[::f.num_channels]
        all_samples.append(data)
    if not all_samples:
        return None, 0
    pcm = np.concatenate(all_samples)
    if src_sr and src_sr != target_sr:
        num_samples = int(len(pcm) * target_sr / src_sr)
        pcm = np.interp(
            np.linspace(0, len(pcm) - 1, num_samples),
            np.arange(len(pcm)), pcm.astype(np.float32),
        ).astype(np.int16)
    duration = len(pcm) / target_sr
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8"), duration


# ══════════════════════════════════════════════════════════════
# STT
# ══════════════════════════════════════════════════════════════
class OmniSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._current_trace_id = None
        self._llm_ref = None
        self._tts_ref = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def set_siblings(self, llm_instance, tts_instance):
        self._llm_ref = llm_instance
        self._tts_ref = tts_instance

    async def _recognize_impl(self, buffer, *, language="zh", conn_options=None):
        tid = self._current_trace_id
        if not tid:
            tid = _tracer.new_trace()
            self._current_trace_id = tid
        _tracer.mark(tid, "t_agent_vad_end")
        if self._llm_ref:
            self._llm_ref.set_trace_id(tid)
        if self._tts_ref:
            self._tts_ref.set_trace_id(tid)

        frames = buffer if isinstance(buffer, list) else [buffer]
        wav_b64, duration = await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(audio_frames_to_wav_base64, frames, 16000)
        )

        if not wav_b64 or duration < 0.2:
            logger.warning(f"[STT] Audio too short: {duration:.2f}s")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT, request_id="stt",
                alternatives=[stt.SpeechData(language="zh", text="", confidence=0.0)],
            )

        logger.info(f"[STT] Transcribing {duration:.1f}s...")
        text = await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(self._call_omni_stt, wav_b64)
        )
        if tid:
            _tracer.mark(tid, "t_stt_done")
        logger.info(f"[STT] → '{text[:60]}'")

        # D17 P0-2: Immediately prefetch LLM reply — overlaps with framework processing
        if FAST_LANE_ENABLED and self._llm_ref and text:
            self._llm_ref.prefetch_reply(text, tid)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT, request_id="stt",
            alternatives=[stt.SpeechData(language="zh", text=text, confidence=0.9 if text else 0.0)],
        )

    def _call_omni_stt(self, wav_b64):
        payload = {
            "model": LLM_MODEL, "stream": False, "max_tokens": 200, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}},
                {"type": "text", "text": "请直接转写上面的语音内容，只输出转写文字，不要任何解释。"},
            ]}],
        }
        try:
            resp = _http_llm.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            logger.error(f"[STT] Error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════
# LLM — F1: 带对话历史
# ══════════════════════════════════════════════════════════════
_LLM_SYS_PROMPT_FAST = (
    "你是一个友好的语音助手。用中文回复，自然口语化。"
    "回复规则：第一句必须3-5个字（如'好的'、'嗯嗯'、'明白了'、'没问题'），"
    "用逗号分隔后再补充详细内容。结合对话上下文。"
)
_LLM_SYS_PROMPT_SHORT = (
    "你是一个友好的语音助手。用中文简短回复（不超过30字），自然口语化。注意结合对话上下文。"
)


def _extract_history(chat_ctx) -> list:
    """Extract message history from LiveKit chat context."""
    items = []
    for item in chat_ctx.items:
        if not hasattr(item, 'role'):
            continue
        role = item.role
        if role not in ("user", "assistant"):
            continue
        text = ""
        content_list = item.content if isinstance(item.content, list) else [item.content]
        for c in content_list:
            if isinstance(c, str) and c.strip():
                text = c.strip()
                break
        if text:
            items.append({"role": role, "content": text})
    max_items = LLM_HISTORY_TURNS * 2
    if len(items) > max_items:
        items = items[-max_items:]
    return items


def _build_llm_messages(history: list, sys_prompt: str = None) -> list:
    """Build complete messages array for LLM API call."""
    if sys_prompt is None:
        sys_prompt = _LLM_SYS_PROMPT_FAST if ENABLE_CONTINUATION else _LLM_SYS_PROMPT_SHORT
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": "你好"})
    return messages


class OmniLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._current_trace_id = None
        self._chat_history = []
        # D17 P0-2: prefetch state
        self._prefetch_queue = None
        self._prefetch_started = False

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def prefetch_reply(self, user_text: str, tid: str):
        """D17 P0-2: Start LLM request immediately (called from STT, before framework)."""
        q = _queue.Queue()
        self._prefetch_queue = q
        self._prefetch_started = True
        history = list(self._chat_history)
        history.append({"role": "user", "content": user_text})
        threading.Thread(
            target=self._run_prefetch, args=(history, tid, q), daemon=True
        ).start()

    def _run_prefetch(self, history, tid, q):
        messages = _build_llm_messages(history)
        payload = {
            "model": LLM_MODEL, "stream": True,
            "max_tokens": LLM_MAX_TOKENS, "temperature": LLM_TEMPERATURE,
            "messages": messages,
        }
        try:
            resp = _http_llm.post(f"{LLM_BASE_URL}/v1/chat/completions",
                                 json=payload, stream=True, timeout=30)
            resp.raise_for_status()
            first_token = True
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta_data = chunk["choices"][0].get("delta", {})
                    if "content" in delta_data and delta_data["content"]:
                        token = delta_data["content"]
                        if first_token and tid:
                            _tracer.mark(tid, "t_llm_first_token")
                            first_token = False
                        q.put(token)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        except Exception as e:
            logger.error(f"[LLM Prefetch] Error: {e}")
        finally:
            q.put(None)

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        self._chat_history = _extract_history(chat_ctx)
        if self._prefetch_started and self._prefetch_queue is not None:
            q = self._prefetch_queue
            self._prefetch_queue = None
            self._prefetch_started = False
            return PrefetchedLLMStream(
                prefetch_queue=q, llm_instance=self, chat_ctx=chat_ctx,
                tools=tools or [], conn_options=conn_options or APIConnectOptions(),
                trace_id=self._current_trace_id,
            )
        return OmniLLMStream(
            llm_instance=self, chat_ctx=chat_ctx,
            tools=tools or [], conn_options=conn_options or APIConnectOptions(),
            trace_id=self._current_trace_id,
        )


class PrefetchedLLMStream(llm.LLMStream):
    """D17 P0-2: Reads tokens from a queue that was pre-filled by the prefetch thread."""
    def __init__(self, prefetch_queue, llm_instance, chat_ctx, tools, conn_options, trace_id=None):
        super().__init__(llm=llm_instance, chat_ctx=chat_ctx,
                         tools=tools, conn_options=conn_options)
        self._q = prefetch_queue
        self._trace_id = trace_id

    async def _run(self):
        full_text = ""
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, self._q.get)
            if token is None:
                break
            full_text += token
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="omni-prefetch",
                    delta=llm.ChoiceDelta(role="assistant", content=token)
                )
            )
        logger.info(f"[LLM Prefetch] → '{full_text[:80]}'")


class OmniLLMStream(llm.LLMStream):
    def __init__(self, llm_instance, chat_ctx, tools, conn_options, trace_id=None):
        super().__init__(llm=llm_instance, chat_ctx=chat_ctx,
                         tools=tools, conn_options=conn_options)
        self._chat_ctx = chat_ctx
        self._trace_id = trace_id

    async def _run(self):
        history = _extract_history(self._chat_ctx)
        messages = _build_llm_messages(history)
        user_msg = messages[-1]["content"] if messages else "你好"
        logger.info(f"[LLM] User: '{user_msg[:80]}' (history={len(messages)-2} msgs)")

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self._stream_omni, messages)
            )
        except Exception as e:
            logger.error(f"[LLM] Error: {e}")

    def _stream_omni(self, messages):
        payload = {
            "model": LLM_MODEL,
            "stream": True,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "messages": messages,
        }

        resp = _http_llm.post(f"{LLM_BASE_URL}/v1/chat/completions",
                             json=payload, stream=True, timeout=30)
        resp.raise_for_status()

        full_text = ""
        first_token = True
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta_data = chunk["choices"][0].get("delta", {})
                if "content" in delta_data and delta_data["content"]:
                    token = delta_data["content"]
                    full_text += token

                    if first_token and self._trace_id:
                        _tracer.mark(self._trace_id, "t_llm_first_token")
                        first_token = False

                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id="omni",
                            delta=llm.ChoiceDelta(role="assistant", content=token)
                        )
                    )
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        logger.info(f"[LLM] → '{full_text[:80]}'")


# ══════════════════════════════════════════════════════════════
# TTS — 边收边推 + F3: 安全错误处理
# ══════════════════════════════════════════════════════════════
class QwenTTS(tts.TTS):
    _room_ref: "rtc.Room | None" = None  # D9: room reference for DataChannel events
    _reply_seq: int = 0  # D9: reply sequence counter
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE_TTS, num_channels=1,
        )
        self._current_trace_id = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def synthesize(self, text: str, *, conn_options=None):
        return QwenTTSStream(
            text, self, conn_options=conn_options or APIConnectOptions(),
            trace_id=self._current_trace_id,
        )


class QwenTTSStream(tts.ChunkedStream):
    def __init__(self, text, tts_instance, conn_options=None, trace_id=None):
        super().__init__(tts=tts_instance, input_text=text,
                         conn_options=conn_options or APIConnectOptions())
        self._text = text
        self._trace_id = trace_id
        self._tts_instance = tts_instance

    async def _run(self, output_emitter):
        text = self._text
        tid = self._trace_id

        output_emitter.initialize(
            request_id=f"tts-{id(self)}", sample_rate=SAMPLE_RATE_TTS,
            num_channels=1, mime_type="audio/pcm", frame_size_ms=TTS_FRAME_MS,
        )

        if not text or len(text.strip()) < 2:
            logger.warning(f"[TTS] Text too short: '{text}'")
            output_emitter.push(b'\x00' * 960)
            output_emitter.flush()
            return

        logger.info(f"[TTS] Synth: '{text[:40]}'")

        stop_event = threading.Event()
        pre_rtc_chunks = []  # D7 Ring1: 收集 push 的帧用于落盘
        try:
            frame_bytes = int(SAMPLE_RATE_TTS * TTS_FRAME_MS / 1000) * 2
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue(maxsize=0)
            total_bytes = 0
            first_push = True

            worker = asyncio.create_task(asyncio.to_thread(
                self._stream_tts_worker, text, tid, frame_bytes, loop, q, stop_event,
            ))

            got_audio = False
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.warning(f"[TTS] Worker error (non-fatal): {item}")
                    break
                if not item:
                    continue

                got_audio = True
                total_bytes += len(item)
                output_emitter.push(item)
                if CAPTURE_PRE_RTC:
                    pre_rtc_chunks.append(item)
                if first_push and tid:
                    _tracer.mark(tid, "t_agent_publish_first_frame")
                    first_push = False
                    # D9 P0-1: 发 reply_start DataChannel 事件
                    self._send_reply_event(tid, "reply_start")

            await worker

            if not got_audio:
                logger.warning("[TTS] No audio")
                output_emitter.push(b"\x00" * 960)

            output_emitter.flush()
            if got_audio:
                duration = total_bytes / 2 / SAMPLE_RATE_TTS
                logger.info(f"[TTS] Done: {total_bytes}B ({duration:.2f}s) "
                            f"frames={total_bytes // frame_bytes}")
                # D9 P0-1: 发 reply_end DataChannel 事件
                self._send_reply_event(tid, "reply_end")

        except Exception as e:
            logger.error(f"[TTS] Error: {e}")
            try:
                output_emitter.push(b'\x00' * 960)
                output_emitter.flush()
            except Exception:
                pass
        finally:
            # D10 P0-1: Ring1 落盘移到 finally — 即使 TTS 中断/异常也保存已收集的 chunks
            if CAPTURE_PRE_RTC and pre_rtc_chunks and tid:
                try:
                    pcm = b"".join(pre_rtc_chunks)
                    import wave as _wave
                    pre_dir = os.path.join(PRE_RTC_BASE_DIR, tid)
                    os.makedirs(pre_dir, exist_ok=True)
                    pre_rtc_path = os.path.join(pre_dir, "pre_rtc.wav")
                    with _wave.open(pre_rtc_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE_TTS)
                        wf.writeframes(pcm)
                    logger.info(f"[TTS] Ring1: {pre_rtc_path} ({len(pcm)}B)")
                except Exception as e:
                    logger.warning(f"[TTS] Ring1 error: {e}")
            stop_event.set()
            # D7: 立刻 finalize（在 finally 里，确保即使 disconnect 也能写入）
            if tid:
                _tracer.finalize(tid)

    def _send_reply_event(self, tid: str, event_name: str):
        """D9: send reply_start / reply_end via DataChannel to probe_bot"""
        room = getattr(self._tts_instance, "_room_ref", None)
        if not room:
            return
        try:
            trace_data = _tracer.get(tid) if tid in _tracer._traces else {}
            seq = getattr(self._tts_instance, "_reply_seq", 0)
            if event_name == "reply_end":
                self._tts_instance._reply_seq = seq + 1  # increment AFTER end, so start/end share same seq
            payload = json.dumps({
                "type": "autortc_reply",
                "event": event_name,
                "trace_id": tid,
                "case_id": trace_data.get("case_id", ""),
                "reply_seq": seq,
                "t_ms": time.time(),
            }, ensure_ascii=False)
            # fire-and-forget via event loop
            loop = asyncio.get_running_loop()
            loop.create_task(
                room.local_participant.publish_data(
                    payload, reliable=True, topic="autortc.reply",
                )
            )
        except Exception as e:
            logger.debug(f"[TTS] reply event send err: {e}")

    def _stream_tts_worker(self, text, trace_id, frame_bytes, loop, out_queue, stop_event):
        """D17 P0-4: First-frame hot path — flush first chunk immediately."""
        payload = {"text": text, "speaker": TTS_SPEAKER}
        carry = bytearray()
        first_chunk = True
        resp = None
        # D17 P0-4: First chunk threshold — push as soon as we have 10ms of audio
        first_frame_bytes = max(frame_bytes // 2, 480)
        try:
            resp = _http_tts.post(TTS_URL, json=payload, stream=True, timeout=(5, 15))
            resp.raise_for_status()

            for chunk in resp.iter_content(chunk_size=4096):
                if stop_event.is_set():
                    break
                if not chunk:
                    continue
                if first_chunk and trace_id:
                    _tracer.mark(trace_id, "t_tts_first_chunk")
                    first_chunk = False

                carry.extend(chunk)

                # D17 P0-4: For the very first push, use smaller threshold
                if first_chunk is False and len(carry) >= first_frame_bytes and first_frame_bytes < frame_bytes:
                    frame = bytes(carry[:])
                    carry.clear()
                    loop.call_soon_threadsafe(out_queue.put_nowait, frame)
                    first_frame_bytes = frame_bytes  # switch to normal after first push

                while len(carry) >= frame_bytes:
                    frame = bytes(carry[:frame_bytes])
                    del carry[:frame_bytes]
                    loop.call_soon_threadsafe(out_queue.put_nowait, frame)

            if carry and not stop_event.is_set():
                loop.call_soon_threadsafe(out_queue.put_nowait, bytes(carry))
        except Exception as e:
            if not stop_event.is_set():
                logger.warning(f"[TTS] Stream error: {e}")
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            loop.call_soon_threadsafe(out_queue.put_nowait, None)


# ══════════════════════════════════════════════════════════════
# Voice Agent
# ══════════════════════════════════════════════════════════════
class QwenVoiceAgent(Agent):
    def __init__(self):
        self._stt_instance = OmniSTT()
        self._llm_instance = OmniLLM()
        self._tts_instance = QwenTTS()

        # D7: 让 STT 能在 recognize 开始时 propagate trace_id 给 LLM/TTS
        self._stt_instance.set_siblings(self._llm_instance, self._tts_instance)

        inner_vad = silero_plugin.VAD.load(
            min_silence_duration=VAD_ENDPOINTING_SILENCE_MS / 1000.0,
            prefix_padding_duration=VAD_PREFIX_MS / 1000.0,
            min_speech_duration=BARGEIN_MIN_SPEECH_MS / 1000.0,
            activation_threshold=BARGEIN_ACTIVATION_THRESHOLD,
        )
        # D16 P0-2: Adaptive endpointing controller
        ep_controller = None
        if ADAPTIVE_ENDPOINTING:
            ep_controller = EndpointingController()
            logger.info("[Agent] EndpointingController enabled (adaptive hold)")
        self._ep_controller = ep_controller
        vad_instance = NoiseRobustVAD(
            inner_vad,
            noise_gate_enabled=NOISE_GATE_ENABLED,
            endpointing_controller=ep_controller,
        )

        super().__init__(
            instructions="你是一个友好的语音助手。用中文回复，自然口语化。注意结合对话上下文。",
            stt=self._stt_instance,
            llm=self._llm_instance,
            tts=self._tts_instance,
            vad=vad_instance,
            allow_interruptions=(MODE == "duplex"),
            min_endpointing_delay=MIN_ENDPOINTING,
            max_endpointing_delay=2.0 if MODE == "turn_taking" else 1.5,
        )
        self._active_trace_id = None

    def apply_trace(self, trace_id: str, extra: dict = None):
        if not trace_id:
            return
        _tracer.new_trace(trace_id, extra=extra or {})
        self._active_trace_id = trace_id
        self._stt_instance.set_trace_id(trace_id)
        self._llm_instance.set_trace_id(trace_id)
        self._tts_instance.set_trace_id(trace_id)

    async def on_enter(self):
        logger.info(f"[Agent] on_enter | VAD={VAD_SILENCE_MS}ms FRAME={TTS_FRAME_MS}ms "
                     f"CONT={ENABLE_CONTINUATION} HIST={LLM_HISTORY_TURNS} "
                     f"FAST_LANE={FAST_LANE_ENABLED}")
        # D17 P0-5: warm up STT/LLM/TTS in background to avoid cold-start spikes
        asyncio.create_task(self._warmup())
        try:
            await self.session.say("你好，我是语音助手，有什么可以帮你的？")
            logger.info("[Agent] Welcome sent")
        except Exception as e:
            logger.error(f"[Agent] on_enter fail: {e}")

    async def _warmup(self):
        """D17 P0-5: Pre-warm model endpoints to eliminate cold-start latency."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._warmup_sync)
        except Exception as e:
            logger.warning(f"[Agent] Warm-up error: {e}")

    def _warmup_sync(self):
        t0 = time.time()
        try:
            _http_llm.post(f"{LLM_BASE_URL}/v1/chat/completions", json={
                "model": LLM_MODEL, "stream": False, "max_tokens": 5,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "hi"}],
            }, timeout=10)
        except Exception:
            pass
        try:
            _http_tts.post(TTS_URL, json={"text": "嗯", "speaker": TTS_SPEAKER}, timeout=10)
        except Exception:
            pass
        logger.info(f"[Agent] Warm-up done ({time.time()-t0:.1f}s)")

    async def on_user_turn_completed(self, turn_ctx, **kwargs):
        new_msg = kwargs.get("new_message", None)
        text = ""
        if new_msg and hasattr(new_msg, 'content'):
            for c in (new_msg.content if isinstance(new_msg.content, list) else [new_msg.content]):
                if isinstance(c, str) and c.strip():
                    text = c.strip()
                    break

        # D7: trace 已在 STT._recognize_impl 开始时创建并 propagate
        tid = self._stt_instance._current_trace_id or self._active_trace_id or "unknown"
        logger.info(f"[Agent] Turn [{tid}] text='{text[:50]}'")

        # D7 修复：不用 call_later（子进程退出后会丢失）
        # 改为在 TTS 完成时由 TTS 触发 finalize
        # 这里只记录 user_text，finalize 交给 TTS._run 末尾
        _tracer.mark(tid, "_user_text_ready")
        if tid in _tracer._traces:
            _tracer._traces[tid]["user_text"] = text
            # D16 P0-3: Record endpointing params used for this turn
            if self._ep_controller and self._ep_controller.last_decision:
                _tracer._traces[tid]["endpointing_params"] = \
                    self._ep_controller.last_decision.to_dict()

        # 重置，下一轮 STT 会创建新的
        self._stt_instance._current_trace_id = None
        self._active_trace_id = None


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
async def entrypoint(ctx):
    logger.info(f"[Entry] JOB RECEIVED | Room: {ctx.room.name}")
    try:
        await ctx.connect()
        logger.info("[Entry] Room connected")

        # D7: 等待 user（排除 probe bot 和其他观察者）
        participant = None
        for _ in range(60):
            for pid, p in ctx.room.remote_participants.items():
                if not p.identity.startswith("autortc-probe"):
                    participant = p
                    break
            if participant:
                break
            try:
                participant = await asyncio.wait_for(
                    ctx.wait_for_participant(), timeout=1.0)
                if participant and participant.identity.startswith("autortc-probe"):
                    participant = None  # 跳过 probe
                    continue
            except asyncio.TimeoutError:
                continue
        if not participant:
            participant = await ctx.wait_for_participant()
        logger.info(f"[Entry] Participant: {participant.identity}")

        agent = QwenVoiceAgent()
        session = AgentSession()

        # DataChannel 监听（AutoRTC trace 透传 + probe_ready/agent_ready 双向 ACK）
        def _on_data_received(data_packet):
            try:
                topic = getattr(data_packet, "topic", "")
                payload = json.loads(data_packet.data.decode("utf-8"))

                # D10 P0-2: probe_ready → agent 回 agent_ready ACK
                # probe sends topic="autortc.probe" type="autortc_probe" event="probe_ready"
                if (topic == "autortc.probe_ready" or topic == "autortc.probe"
                        or payload.get("type") in ("autortc_probe_ready", "autortc_probe")):
                    trace_id = payload.get("trace_id", "")
                    logger.info(f"[Barrier] Received probe_ready, trace={trace_id}")
                    try:
                        ack = json.dumps({
                            "type": "autortc_agent_ready",
                            "trace_id": trace_id,
                            "t_ms": time.time(),
                        }, ensure_ascii=False)
                        import asyncio as _aio
                        _aio.ensure_future(
                            ctx.room.local_participant.publish_data(
                                ack, reliable=True, topic="autortc.agent_ready"
                            )
                        )
                        logger.info(f"[Barrier] Sent agent_ready ACK, trace={trace_id}")
                    except Exception as e:
                        logger.warning(f"[Barrier] agent_ready send error: {e}")
                    return

                if topic != "autortc.trace":
                    return
                if payload.get("type") != "autortc_trace":
                    return
                trace_id = payload.get("trace_id", "")
                if not trace_id:
                    return
                sender = getattr(data_packet.participant, "identity", None) if getattr(data_packet, "participant", None) else None
                extra = {
                    "trace_from": "data_channel",
                    "trace_sender_identity": sender,
                    "case_id": payload.get("case_id"),
                    "turn_id": payload.get("turn_id"),
                }
                if payload.get("event") == "user_send_start":
                    extra["t_user_send_start"] = payload.get("t_user_send_start")
                if payload.get("event") == "user_send_end":
                    extra["t_user_send_end"] = payload.get("t_user_send_end")
                agent.apply_trace(trace_id, extra=extra)
                logger.info(f"[Trace] DC {trace_id} event={payload.get('event')}")
            except Exception as e:
                logger.warning(f"[Trace] DC parse error: {e}")

        ctx.room.on("data_received", _on_data_received)

        # D9: 让 TTS 能发 DataChannel reply 事件
        agent._tts_instance._room_ref = ctx.room

        await session.start(agent, room=ctx.room)
        logger.info("[Entry] Session started")
    except Exception as e:
        logger.error(f"[Entry] FAIL: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from livekit.agents import cli

    logger.info(f"[Config] MODE={MODE} "
                f"VAD_SILENCE={VAD_ENDPOINTING_SILENCE_MS}ms ENDP_DELAY={ENDPOINTING_DELAY_MS}ms "
                f"(base_total={VAD_ENDPOINTING_SILENCE_MS + ENDPOINTING_DELAY_MS}ms) "
                f"ADAPTIVE_EP={ADAPTIVE_ENDPOINTING} FAST_LANE={FAST_LANE_ENABLED} "
                f"BARGEIN_MIN_SPEECH={BARGEIN_MIN_SPEECH_MS}ms "
                f"BARGEIN_THRESH={BARGEIN_ACTIVATION_THRESHOLD} "
                f"NOISE_GATE={NOISE_GATE_ENABLED} "
                f"FRAME={TTS_FRAME_MS}ms CONT={ENABLE_CONTINUATION}")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
            api_key=os.environ.get("LIVEKIT_API_KEY"),
            api_secret=os.environ.get("LIVEKIT_API_SECRET"),
            ws_url=os.environ.get("LIVEKIT_URL"),
            port=8089,
            num_idle_processes=8,  # D8: 增大进程池，防止 16 case 耗尽
        ),
    )
