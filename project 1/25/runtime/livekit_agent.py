#!/usr/bin/env python3
"""
LiveKit Voice Agent v2 — D5 端到端可观测 + 低延迟优化

架构：
  浏览器 (WebRTC) ↔ LiveKit Cloud ↔ 本 Agent
  Agent 内部: VAD → STT(Omni) → LLM(Omni) → TTS → 流式发布

D5 改进：
  1. 全链路 trace 打点（9 个时间戳，输出 JSONL）
  2. TTS 流式 push 小帧（20ms），不再一次性塞大块
  3. VAD hangover 可配（A/B: 550ms → 200ms）
  4. LLM continuation 机制（短首句 + 延续句）
  5. 所有重活 offload 到线程池
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
# 配置（全部可 env 覆盖，不写死）
# ══════════════════════════════════════════════════════════════
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-omni-thinker")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "serena")

SAMPLE_RATE_TTS = int(os.environ.get("SAMPLE_RATE_TTS", "24000"))
TTS_FRAME_MS = int(os.environ.get("TTS_FRAME_MS", "20"))          # 发布帧粒度 ms
VAD_SILENCE_MS = int(os.environ.get("VAD_SILENCE_MS", "200"))     # hangover ms (D5 A/B)
VAD_PREFIX_MS = int(os.environ.get("VAD_PREFIX_MS", "300"))       # prefix padding ms
MIN_ENDPOINTING = float(os.environ.get("MIN_ENDPOINTING", "0.3"))  # 最小 endpointing delay

LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
ENABLE_CONTINUATION = os.environ.get("ENABLE_CONTINUATION", "1") == "1"

TRACE_DIR = os.environ.get("TRACE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))

# ══════════════════════════════════════════════════════════════
# Trace 收集器
# ══════════════════════════════════════════════════════════════
class TraceCollector:
    """收集每轮对话的端到端时间戳，输出 JSONL"""

    def __init__(self):
        self._traces = {}  # trace_id -> dict
        self._trace_file = os.path.join(TRACE_DIR, "day5_e2e_traces.jsonl")
        os.makedirs(TRACE_DIR, exist_ok=True)

    def new_trace(self) -> str:
        tid = str(uuid.uuid4())[:8]
        self._traces[tid] = {
            "trace_id": tid,
            "created_at": time.time(),
        }
        return tid

    def mark(self, trace_id: str, key: str, ts: float = None):
        if trace_id not in self._traces:
            return
        self._traces[trace_id][key] = ts or time.time()

    def get(self, trace_id: str) -> dict:
        return self._traces.get(trace_id, {})

    def finalize(self, trace_id: str, extra: dict = None):
        if trace_id not in self._traces:
            return
        t = self._traces.pop(trace_id)
        if extra:
            t.update(extra)

        # 计算延迟段
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


# 全局 trace 收集器
_tracer = TraceCollector()


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def audio_frames_to_wav_base64(frames, target_sr=16000):
    """将 AudioFrame(s) 转为 base64 WAV（16kHz mono for Omni）"""
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
            np.arange(len(pcm)),
            pcm.astype(np.float32),
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
# STT — Omni 转写（带 trace 打点）
# ══════════════════════════════════════════════════════════════
class OmniSTT(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._current_trace_id = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    async def _recognize_impl(self, buffer, *, language="zh", conn_options=None):
        tid = self._current_trace_id
        if tid:
            _tracer.mark(tid, "t_agent_vad_end")

        frames = buffer if isinstance(buffer, list) else [buffer]

        # offload 音频编码到线程
        wav_b64, duration = await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(audio_frames_to_wav_base64, frames, 16000)
        )

        if not wav_b64 or duration < 0.2:
            logger.warning(f"[STT] Audio too short: {duration:.2f}s")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id="stt",
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
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id="stt",
            alternatives=[stt.SpeechData(
                language="zh", text=text, confidence=0.9 if text else 0.0,
            )],
        )

    def _call_omni_stt(self, wav_b64):
        payload = {
            "model": LLM_MODEL,
            "stream": False,
            "max_tokens": 200,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}},
                    {"type": "text", "text": "请直接转写上面的语音内容，只输出转写文字，不要任何解释。"},
                ],
            }],
        }
        try:
            resp = requests.post(f"{LLM_BASE_URL}/v1/chat/completions",
                                 json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as e:
            logger.error(f"[STT] Error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════
# LLM — Omni fast lane（带 trace + continuation）
# ══════════════════════════════════════════════════════════════
class OmniLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self._current_trace_id = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        return OmniLLMStream(
            llm_instance=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or APIConnectOptions(),
            trace_id=self._current_trace_id,
        )


class OmniLLMStream(llm.LLMStream):
    def __init__(self, llm_instance, chat_ctx, tools, conn_options, trace_id=None):
        super().__init__(llm=llm_instance, chat_ctx=chat_ctx,
                         tools=tools, conn_options=conn_options)
        self._chat_ctx = chat_ctx
        self._trace_id = trace_id

    async def _run(self):
        user_msg = ""
        for item in reversed(self._chat_ctx.items):
            if hasattr(item, 'role') and item.role == "user":
                content_list = item.content if isinstance(item.content, list) else [item.content]
                for c in content_list:
                    if isinstance(c, str) and c.strip():
                        user_msg = c.strip()
                        break
                if user_msg:
                    break

        if not user_msg:
            user_msg = "你好"

        logger.info(f"[LLM] User: '{user_msg[:80]}'")

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self._stream_omni, user_msg)
            )
        except Exception as e:
            logger.error(f"[LLM] Error: {e}")

    def _stream_omni(self, user_msg):
        # 如果启用 continuation，prompt 鼓励先短后长
        if ENABLE_CONTINUATION:
            sys_prompt = (
                "你是一个语音助手。用中文回复，自然口语化。"
                "先用一句短话（10字以内）回应，然后再补充详细内容。"
                "两句之间用句号分隔。"
            )
        else:
            sys_prompt = "你是一个语音助手。用中文简短回复（不超过30字），自然口语化。"

        payload = {
            "model": LLM_MODEL,
            "stream": True,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
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
# TTS — 流式小帧 push（20ms 粒度，不一次性塞大块）
# ══════════════════════════════════════════════════════════════
class QwenTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE_TTS,
            num_channels=1,
        )
        self._current_trace_id = None

    def set_trace_id(self, tid):
        self._current_trace_id = tid

    def synthesize(self, text: str, *, conn_options=None):
        return QwenTTSStream(
            text, self,
            conn_options=conn_options or APIConnectOptions(),
            trace_id=self._current_trace_id,
        )


class QwenTTSStream(tts.ChunkedStream):
    def __init__(self, text, tts_instance, conn_options=None, trace_id=None):
        super().__init__(tts=tts_instance, input_text=text,
                         conn_options=conn_options or APIConnectOptions())
        self._text = text
        self._trace_id = trace_id

    async def _run(self, output_emitter):
        text = self._text
        tid = self._trace_id

        # 必须先 initialize — StreamAdapter 需要，否则 end_input 崩溃
        output_emitter.initialize(
            request_id=f"tts-{id(self)}",
            sample_rate=SAMPLE_RATE_TTS,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=TTS_FRAME_MS,
        )

        if not text or len(text.strip()) < 2:
            logger.warning(f"[TTS] Text too short: '{text}'")
            output_emitter.push(b'\x00' * 960)  # 20ms 静音
            output_emitter.flush()
            return

        logger.info(f"[TTS] Synth: '{text[:40]}'")

        try:
            pcm_data = await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(self._call_tts_streaming, text, tid)
            )

            if not pcm_data:
                logger.warning("[TTS] No audio returned")
                output_emitter.push(b'\x00' * 960)
                output_emitter.flush()
                return

            # 分 20ms 小帧 push
            frame_bytes = int(SAMPLE_RATE_TTS * TTS_FRAME_MS / 1000) * 2
            first_push = True
            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset + frame_bytes]
                output_emitter.push(chunk)
                if first_push and tid:
                    _tracer.mark(tid, "t_agent_publish_first_frame")
                    first_push = False
                offset += frame_bytes

            output_emitter.flush()

            duration = len(pcm_data) / 2 / SAMPLE_RATE_TTS
            logger.info(f"[TTS] Done: {len(pcm_data)}B ({duration:.2f}s) "
                         f"frames={len(pcm_data)//frame_bytes}")

        except Exception as e:
            logger.error(f"[TTS] Error: {e}")
            try:
                output_emitter.push(b'\x00' * 960)
                output_emitter.flush()
            except Exception:
                pass

    def _call_tts_streaming(self, text, trace_id):
        """流式调 TTS server，记录 TTFA"""
        payload = {"text": text, "speaker": TTS_SPEAKER}
        resp = requests.post(TTS_URL, json=payload, stream=True, timeout=60)
        resp.raise_for_status()

        chunks = []
        first_chunk = True
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if first_chunk and trace_id:
                    _tracer.mark(trace_id, "t_tts_first_chunk")
                    first_chunk = False
                chunks.append(chunk)

        return b"".join(chunks)


# ══════════════════════════════════════════════════════════════
# Voice Agent — 带 trace 生命周期
# ══════════════════════════════════════════════════════════════
class QwenVoiceAgent(Agent):
    """
    D5 Agent: VAD(Silero低hangover) → STT(Omni) → LLM(Omni+continuation) → TTS(流式小帧)
    """

    def __init__(self):
        self._stt_instance = OmniSTT()
        self._llm_instance = OmniLLM()
        self._tts_instance = QwenTTS()

        super().__init__(
            instructions="你是一个语音助手。用中文回复，自然口语化。",
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

    async def on_enter(self):
        logger.info(f"[Agent] on_enter | VAD_SILENCE={VAD_SILENCE_MS}ms "
                     f"FRAME={TTS_FRAME_MS}ms CONTINUATION={ENABLE_CONTINUATION}")
        try:
            await self.session.say("你好，我是语音助手，请说话。")
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

        # 创建 trace 并设置到各插件
        tid = _tracer.new_trace()
        self._stt_instance.set_trace_id(tid)
        self._llm_instance.set_trace_id(tid)
        self._tts_instance.set_trace_id(tid)

        logger.info(f"[Agent] Turn done [{tid}] text='{text[:50]}'")

        # 异步等待本轮结束后 finalize trace
        asyncio.get_event_loop().call_later(
            15.0, lambda: _tracer.finalize(tid, {"user_text": text})
        )


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
async def entrypoint(ctx):
    logger.info(f"[Entry] JOB RECEIVED | Room: {ctx.room.name}")
    try:
        await ctx.connect()
        logger.info("[Entry] Room connected")

        participant = await ctx.wait_for_participant()
        logger.info(f"[Entry] Participant: {participant.identity}")

        agent = QwenVoiceAgent()
        session = AgentSession()

        await session.start(agent, room=ctx.room)
        logger.info("[Entry] Session started")
    except Exception as e:
        logger.error(f"[Entry] FAIL: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from livekit.agents import cli

    logger.info(f"[Config] VAD_SILENCE={VAD_SILENCE_MS}ms TTS_FRAME={TTS_FRAME_MS}ms "
                f"MIN_ENDPOINT={MIN_ENDPOINTING}s CONTINUATION={ENABLE_CONTINUATION} "
                f"LLM_MAX_TOKENS={LLM_MAX_TOKENS}")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
            api_key=os.environ.get("LIVEKIT_API_KEY"),
            api_secret=os.environ.get("LIVEKIT_API_SECRET"),
            ws_url=os.environ.get("LIVEKIT_URL"),
            port=8089,
        ),
    )
