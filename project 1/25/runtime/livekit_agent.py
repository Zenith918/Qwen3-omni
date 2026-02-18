#!/usr/bin/env python3
"""
LiveKit Voice Agent v3 — D6 修复版

修复:
  F1: LLM 加对话历史（chat_ctx → messages 完整传递）
  F3: TTS "Response ended prematurely" 安全处理
  F5: Trace 回退逻辑（非 AutoRTC 也能正常打点）
  保留 D6 工程师的: 边收边推、DataChannel trace、防污染
"""

import asyncio
import base64
import io
import json
import logging
import os
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
# D14 P0-3: Turn-taking minimum silence knob (ms) — overrides MIN_ENDPOINTING if set
TURN_TAKING_MIN_SILENCE_MS = int(os.environ.get("TURN_TAKING_MIN_SILENCE_MS", "0"))
if TURN_TAKING_MIN_SILENCE_MS > 0:
    MIN_ENDPOINTING = TURN_TAKING_MIN_SILENCE_MS / 1000.0

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_HISTORY_TURNS = int(os.environ.get("LLM_HISTORY_TURNS", "10"))  # F1: 保留最近N轮
ENABLE_CONTINUATION = os.environ.get("ENABLE_CONTINUATION", "1") == "1"

TRACE_DIR = os.environ.get("TRACE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))

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
        self._llm_ref = None  # D7: LLM 引用，用于在 STT 开始时 propagate trace_id
        self._tts_ref = None  # D7: TTS 引用

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def set_siblings(self, llm_instance, tts_instance):
        """D7: 让 STT 能在 recognize 开始时就 propagate trace_id 到 LLM/TTS"""
        self._llm_ref = llm_instance
        self._tts_ref = tts_instance

    async def _recognize_impl(self, buffer, *, language="zh", conn_options=None):
        # D7: 在 STT 开始时就创建/确认 trace，立刻 propagate 给 LLM/TTS
        tid = self._current_trace_id
        if not tid:
            tid = _tracer.new_trace()
            self._current_trace_id = tid
        _tracer.mark(tid, "t_agent_vad_end")
        # 立刻让 LLM 和 TTS 用同一个 trace_id（在 LLM 开始前！）
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
            resp = requests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=payload, timeout=30)
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
class OmniLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._current_trace_id = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        return OmniLLMStream(
            llm_instance=self, chat_ctx=chat_ctx,
            tools=tools or [], conn_options=conn_options or APIConnectOptions(),
            trace_id=self._current_trace_id,
        )


class OmniLLMStream(llm.LLMStream):
    def __init__(self, llm_instance, chat_ctx, tools, conn_options, trace_id=None):
        super().__init__(llm=llm_instance, chat_ctx=chat_ctx,
                         tools=tools, conn_options=conn_options)
        self._chat_ctx = chat_ctx
        self._trace_id = trace_id

    def _build_messages(self):
        """F1: 从 chat_ctx 构建完整 messages（含历史），而非只取最后一句"""
        if ENABLE_CONTINUATION:
            sys_prompt = (
                "你是一个友好的语音助手。用中文回复，自然口语化。"
                "先用一句短话（10字以内）回应，然后再适当补充。"
                "注意结合之前的对话上下文回复。"
            )
        else:
            sys_prompt = "你是一个友好的语音助手。用中文简短回复（不超过30字），自然口语化。注意结合对话上下文。"

        messages = [{"role": "system", "content": sys_prompt}]

        # 从 chat_ctx.items 提取历史（最近 N 轮）
        history_items = []
        for item in self._chat_ctx.items:
            if not hasattr(item, 'role'):
                continue
            role = item.role
            if role not in ("user", "assistant"):
                continue
            # 提取文本内容
            text = ""
            content_list = item.content if isinstance(item.content, list) else [item.content]
            for c in content_list:
                if isinstance(c, str) and c.strip():
                    text = c.strip()
                    break
            if text:
                history_items.append({"role": role, "content": text})

        # 只保留最近 N 轮（每轮 = 1 user + 1 assistant = 2 条）
        max_items = LLM_HISTORY_TURNS * 2
        if len(history_items) > max_items:
            history_items = history_items[-max_items:]

        messages.extend(history_items)

        # 确保最后一条是 user
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": "你好"})

        return messages

    async def _run(self):
        messages = self._build_messages()
        user_msg = messages[-1]["content"] if messages else "你好"
        logger.info(f"[LLM] User: '{user_msg[:80]}' (history={len(messages)-2} msgs)")

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self._stream_omni, messages)
            )
        except Exception as e:
            logger.error(f"[LLM] Error: {e}")

    def _stream_omni(self, messages):
        """F1: 接受完整 messages list（含历史）"""
        payload = {
            "model": LLM_MODEL,
            "stream": True,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "messages": messages,
        }

        resp = requests.post(f"{LLM_BASE_URL}/v1/chat/completions",
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

                    # F5: 无条件记录 llm_first_token（不再要求 t_stt_done 存在）
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
        """F3: 边收边推 worker，安全处理所有异常"""
        payload = {"text": text, "speaker": TTS_SPEAKER}
        carry = bytearray()
        first_chunk = True
        resp = None
        try:
            resp = requests.post(TTS_URL, json=payload, stream=True, timeout=(5, 15))
            resp.raise_for_status()

            for chunk in resp.iter_content(chunk_size=4096):
                if stop_event.is_set():
                    break
                if not chunk:
                    continue
                if first_chunk and trace_id:
                    # F5: 无条件记录
                    _tracer.mark(trace_id, "t_tts_first_chunk")
                    first_chunk = False

                carry.extend(chunk)
                while len(carry) >= frame_bytes:
                    frame = bytes(carry[:frame_bytes])
                    del carry[:frame_bytes]
                    loop.call_soon_threadsafe(out_queue.put_nowait, frame)

            if carry and not stop_event.is_set():
                loop.call_soon_threadsafe(out_queue.put_nowait, bytes(carry))
        except Exception as e:
            # F3: 不抛异常到主协程，只 warning
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

        super().__init__(
            instructions="你是一个友好的语音助手。用中文回复，自然口语化。注意结合对话上下文。",
            stt=self._stt_instance,
            llm=self._llm_instance,
            tts=self._tts_instance,
            vad=silero_plugin.VAD.load(
                min_silence_duration=VAD_SILENCE_MS / 1000.0,
                prefix_padding_duration=VAD_PREFIX_MS / 1000.0,
                min_speech_duration=0.05,
                activation_threshold=0.5,
            ),
            allow_interruptions=True,
            min_endpointing_delay=MIN_ENDPOINTING,
            max_endpointing_delay=1.5,
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
                     f"CONT={ENABLE_CONTINUATION} HIST={LLM_HISTORY_TURNS}")
        try:
            await self.session.say("你好，我是语音助手，有什么可以帮你的？")
            logger.info("[Agent] Welcome sent")
        except Exception as e:
            logger.error(f"[Agent] on_enter fail: {e}")

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

    logger.info(f"[Config] VAD={VAD_SILENCE_MS}ms FRAME={TTS_FRAME_MS}ms "
                f"ENDP={MIN_ENDPOINTING}s CONT={ENABLE_CONTINUATION} "
                f"TOKENS={LLM_MAX_TOKENS} HIST={LLM_HISTORY_TURNS}")

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
