#!/usr/bin/env python3
"""
Day 2 — Duplex Controller v0

产品级双工控制平面：
  - 状态机：LISTENING → THINKING → SPEAKING → INTERRUPTING
  - cancel() 级联：LLM → Bridge → TTS → 播放
  - barge-in 触发（键盘模拟 / VAD stub）
  - cancel→停声 P95 ≤ 200ms

架构：
  DuplexController 是一个状态机，管理整个对话回合。
  它不直接做 audio I/O，而是提供控制 API 供上层调用。

用法（独立 demo）：
  python3 runtime/duplex_controller.py --wav input.wav --keyboard_bargein 1
"""

import argparse
import enum
import json
import os
import sys
import threading
import time
from typing import Optional, Callable

import requests

# ── 状态定义 ─────────────────────────────────────────────────
class DuplexState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"


# ── 事件日志 ─────────────────────────────────────────────────
class EventLog:
    """线程安全的事件日志"""
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def log(self, event_type: str, **kwargs):
        with self._lock:
            entry = {
                "t_ms": round(time.time() * 1000, 1),
                "type": event_type,
                **kwargs,
            }
            self.events.append(entry)
            return entry

    def dump(self):
        with self._lock:
            return list(self.events)


# ── Cancel Token ─────────────────────────────────────────────
class CancelToken:
    """
    可传播的取消令牌。
    一旦 cancel() 被调用，所有持有该 token 的组件应立即停止。
    """
    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def reset(self):
        self._cancelled.clear()


# ── Duplex Controller ───────────────────────────────────────
class DuplexController:
    """
    产品级双工控制器 v0

    状态机转换：
      IDLE → LISTENING（开始收音）
      LISTENING → THINKING（VAD end / 用户说完）
      THINKING → SPEAKING（首段 TTS 音频就绪）
      SPEAKING → INTERRUPTING（检测到 barge-in）
      INTERRUPTING → LISTENING（cancel 完成，重新收音）
      SPEAKING → IDLE（播放结束，等待下一轮）
    """

    def __init__(self,
                 tts_url: str = "http://127.0.0.1:9000/tts/stream",
                 llm_url: str = "http://127.0.0.1:8000",
                 on_state_change: Optional[Callable] = None):
        self.tts_url = tts_url
        self.llm_url = llm_url
        self.on_state_change = on_state_change

        self._state = DuplexState.IDLE
        self._state_lock = threading.Lock()
        self._cancel_token = CancelToken()
        self.event_log = EventLog()

        # 活跃请求追踪（用于 cancel 级联）
        self._active_tts_response: Optional[requests.Response] = None
        self._active_llm_response: Optional[requests.Response] = None
        self._active_threads: list[threading.Thread] = []

        # 性能计量
        self._interrupt_start_time: Optional[float] = None
        self._silence_achieved_time: Optional[float] = None

    @property
    def state(self) -> DuplexState:
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: DuplexState, reason: str = ""):
        with self._state_lock:
            old = self._state
            self._state = new_state
        self.event_log.log("state_change", old=old.value, new=new_state.value, reason=reason)
        if self.on_state_change:
            self.on_state_change(old, new_state, reason)

    # ── Cancel 级联 ──────────────────────────────────────────
    def cancel(self, reason: str = "barge-in") -> float:
        """
        级联取消所有活跃操作。
        返回 cancel→停声 延迟（ms）。
        只测量连接关闭时间（实际停声时刻），/tts/cancel 异步发送。
        """
        self._interrupt_start_time = time.time()
        self._set_state(DuplexState.INTERRUPTING, reason)
        self.event_log.log("cancel_start", reason=reason)

        # 1. 设置取消令牌（所有组件应检查此令牌 — 立即停止音频消费）
        self._cancel_token.cancel()

        # 2. 关闭活跃 TTS 响应连接（立即停止音频流入）
        if self._active_tts_response:
            try:
                self._active_tts_response.close()
            except Exception:
                pass
            self._active_tts_response = None

        # 3. 关闭活跃 LLM 响应连接
        if self._active_llm_response:
            try:
                self._active_llm_response.close()
            except Exception:
                pass
            self._active_llm_response = None

        # ── 到此为止音频已停止，记录停声时刻 ──
        self._silence_achieved_time = time.time()
        cancel_latency_ms = (self._silence_achieved_time - self._interrupt_start_time) * 1000

        # 4. 异步通知 TTS cancel（fire-and-forget，不计入延迟）
        def _notify_tts_cancel():
            try:
                requests.post(
                    f"{self.tts_url.rsplit('/', 1)[0]}/tts/cancel",
                    timeout=0.3)
            except Exception:
                pass

        threading.Thread(target=_notify_tts_cancel, daemon=True).start()

        self.event_log.log("cancel_done", latency_ms=round(cancel_latency_ms, 1))
        return cancel_latency_ms

    def reset(self):
        """重置为 IDLE 状态，准备下一轮对话"""
        self._cancel_token = CancelToken()
        self._active_tts_response = None
        self._active_llm_response = None
        self._set_state(DuplexState.IDLE, "reset")

    # ── 对话回合 ─────────────────────────────────────────────
    def start_listening(self):
        self._cancel_token.reset()
        self._set_state(DuplexState.LISTENING, "start")

    def end_listening(self):
        """VAD end - 用户说完"""
        self._set_state(DuplexState.THINKING, "vad_end")

    def start_speaking(self):
        self._set_state(DuplexState.SPEAKING, "tts_first_audio")

    def end_speaking(self):
        self._set_state(DuplexState.IDLE, "playout_done")

    # ── 带 cancel 支持的 TTS 调用 ────────────────────────────
    def call_tts_stream(self, text: str, speaker: str = "serena") -> dict:
        """
        调用 TTS，支持通过 cancel_token 中途终止。
        """
        if self._cancel_token.is_cancelled:
            return {"pcm_data": b"", "ttfa_ms": None, "total_ms": 0, "cancelled": True}

        payload = {"text": text, "speaker": speaker}
        t0 = time.time()

        try:
            resp = requests.post(self.tts_url, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            self._active_tts_response = resp

            first_chunk_time = None
            chunks = []

            for chunk in resp.iter_content(chunk_size=4096):
                if self._cancel_token.is_cancelled:
                    resp.close()
                    break
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        self.start_speaking()  # 首音到达 → SPEAKING
                    chunks.append(chunk)

            self._active_tts_response = None
            t_end = time.time()
            pcm_data = b"".join(chunks)

            return {
                "pcm_data": pcm_data,
                "ttfa_ms": round((first_chunk_time - t0) * 1000, 1) if first_chunk_time else None,
                "total_ms": round((t_end - t0) * 1000, 1),
                "audio_duration_s": round(len(pcm_data) / (24000 * 2), 3),
                "cancelled": self._cancel_token.is_cancelled,
            }
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            self._active_tts_response = None
            if self._cancel_token.is_cancelled:
                return {"pcm_data": b"", "ttfa_ms": None, "total_ms": 0, "cancelled": True}
            raise

    # ── 带 cancel 支持的 LLM 流式调用 ────────────────────────
    def stream_llm(self, audio_b64: str, prompt: str,
                   model: str = "qwen3-omni-thinker") -> str:
        """
        流式调用 LLM，收集完整回复文本。支持中途取消。
        """
        if self._cancel_token.is_cancelled:
            return ""

        payload = {
            "model": model,
            "stream": True,
            "max_tokens": 64,
            "temperature": 0.2,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "audio_url",
                     "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        try:
            resp = requests.post(
                f"{self.llm_url}/v1/chat/completions",
                json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            self._active_llm_response = resp

            text = ""
            for line in resp.iter_lines(decode_unicode=True):
                if self._cancel_token.is_cancelled:
                    resp.close()
                    break
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        text += delta["content"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            self._active_llm_response = None
            return text

        except Exception as e:
            self._active_llm_response = None
            if self._cancel_token.is_cancelled:
                return ""
            raise


# ── 独立 Demo：键盘 Barge-in ─────────────────────────────────
def demo_bargein(wav_path: str, keyboard_mode: bool = True):
    """
    演示完整对话回合 + barge-in 中断。
    keyboard_mode=True：播放中按 Enter 触发 interrupt。
    """
    import wave as wave_mod

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clients"))
    from demo_audio_to_omni import wav_to_base64, PROMPT_FAST

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def on_state_change(old, new, reason):
        print(f"  📡 [{old.value}] → [{new.value}]  ({reason})")

    ctrl = DuplexController(on_state_change=on_state_change)
    cancel_latencies = []

    print(f"\n{'='*60}")
    print(f"  Duplex Controller v0 Demo")
    print(f"  Input: {wav_path}")
    if keyboard_mode:
        print(f"  ⌨  Press ENTER during SPEAKING to trigger barge-in")
    print(f"{'='*60}")

    # 预处理音频
    audio_b64, wav_duration = wav_to_base64(wav_path)

    # ── 回合 1：正常播放（不中断）──
    print(f"\n--- Round 1: Normal playback (no interrupt) ---")
    ctrl.reset()
    ctrl.start_listening()
    time.sleep(0.1)  # 模拟收音
    ctrl.end_listening()

    # LLM
    reply_text = ctrl.stream_llm(audio_b64, PROMPT_FAST)
    print(f"  Reply: \"{reply_text}\"")

    if not reply_text.strip():
        print("  ❌ LLM returned empty reply")
        return

    # TTS
    tts_result = ctrl.call_tts_stream(reply_text.strip())
    if tts_result["pcm_data"]:
        ctrl.end_speaking()
        print(f"  TTS: {tts_result['audio_duration_s']:.2f}s, TTFA={tts_result['ttfa_ms']}ms")

    # ── 回合 2: Barge-in 中断测试 ──
    print(f"\n--- Round 2: Barge-in interrupt test ---")
    ctrl.reset()
    ctrl.start_listening()
    time.sleep(0.05)
    ctrl.end_listening()

    reply_text = ctrl.stream_llm(audio_b64, PROMPT_FAST)
    print(f"  Reply: \"{reply_text}\"")

    if not reply_text.strip():
        print("  ❌ LLM returned empty reply")
        return

    if keyboard_mode:
        # 在后台线程中开始 TTS
        tts_done = threading.Event()
        tts_result_holder = [None]

        def tts_thread():
            try:
                result = ctrl.call_tts_stream(reply_text.strip())
                tts_result_holder[0] = result
            except Exception as e:
                tts_result_holder[0] = {"error": str(e), "cancelled": True}
            tts_done.set()

        t = threading.Thread(target=tts_thread, daemon=True)
        t.start()

        # 等待 SPEAKING 状态或超时
        for _ in range(50):  # 5 秒超时
            if ctrl.state == DuplexState.SPEAKING:
                break
            time.sleep(0.1)

        if ctrl.state == DuplexState.SPEAKING:
            print(f"  🔊 Now SPEAKING. Press ENTER to interrupt (or wait 3s for auto-interrupt)...")

            # 使用超时等待 stdin（非阻塞方式）
            interrupted = threading.Event()

            def wait_enter():
                try:
                    import select
                    ready, _, _ = select.select([sys.stdin], [], [], 3.0)
                    if ready:
                        sys.stdin.readline()
                        interrupted.set()
                    else:
                        # 超时：自动 interrupt
                        interrupted.set()
                except Exception:
                    time.sleep(1.5)  # 非交互环境，1.5 秒后自动中断
                    interrupted.set()

            enter_thread = threading.Thread(target=wait_enter, daemon=True)
            enter_thread.start()
            interrupted.wait(timeout=5)

            # 触发 barge-in
            cancel_ms = ctrl.cancel("keyboard_bargein")
            cancel_latencies.append(cancel_ms)
            print(f"  ⚡ Cancel latency: {cancel_ms:.1f} ms")

        tts_done.wait(timeout=10)
        r = tts_result_holder[0]
        if r:
            print(f"  TTS cancelled={r.get('cancelled', False)}, "
                  f"partial audio={r.get('audio_duration_s', 0):.2f}s")

    else:
        # 自动模式：TTS 开始后 200ms 自动触发 interrupt
        tts_done = threading.Event()
        tts_result_holder = [None]

        def tts_thread():
            try:
                result = ctrl.call_tts_stream(reply_text.strip())
                tts_result_holder[0] = result
            except Exception as e:
                tts_result_holder[0] = {"error": str(e), "cancelled": True}
            tts_done.set()

        t = threading.Thread(target=tts_thread, daemon=True)
        t.start()

        # 等 SPEAKING，然后 200ms 后 cancel
        for _ in range(50):
            if ctrl.state == DuplexState.SPEAKING:
                break
            time.sleep(0.1)

        if ctrl.state == DuplexState.SPEAKING:
            time.sleep(0.2)  # 播放 200ms
            cancel_ms = ctrl.cancel("auto_bargein")
            cancel_latencies.append(cancel_ms)
            print(f"  ⚡ Cancel latency: {cancel_ms:.1f} ms")

        tts_done.wait(timeout=10)
        r = tts_result_holder[0]
        if r:
            print(f"  TTS cancelled={r.get('cancelled', False)}, "
                  f"partial audio={r.get('audio_duration_s', 0):.2f}s")

    # ── 多轮 cancel 延迟测量 ──
    print(f"\n--- Cancel latency benchmark (10 rounds, auto) ---")
    for i in range(10):
        ctrl.reset()
        ctrl.start_listening()
        ctrl.end_listening()

        reply = ctrl.stream_llm(audio_b64, PROMPT_FAST)
        if not reply.strip():
            continue

        tts_done_ev = threading.Event()
        result_h = [None]

        def tts_work():
            try:
                result_h[0] = ctrl.call_tts_stream(reply.strip())
            except Exception:
                result_h[0] = {"cancelled": True}
            tts_done_ev.set()

        t = threading.Thread(target=tts_work, daemon=True)
        t.start()

        # 等 SPEAKING
        for _ in range(50):
            if ctrl.state == DuplexState.SPEAKING:
                break
            time.sleep(0.1)

        if ctrl.state == DuplexState.SPEAKING:
            time.sleep(0.15)  # 播放 150ms
            c_ms = ctrl.cancel(f"bench_{i}")
            cancel_latencies.append(c_ms)
            sys.stdout.write(f"  [{i+1}/10] cancel={c_ms:.1f}ms  ")
            sys.stdout.flush()

        tts_done_ev.wait(timeout=10)

    print()

    # ── 报告 ──
    if cancel_latencies:
        sorted_lats = sorted(cancel_latencies)
        p50 = sorted_lats[len(sorted_lats) // 2]
        p95_idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)
        p95 = sorted_lats[p95_idx]

        print(f"\n{'='*60}")
        print(f"  Duplex Controller Report")
        print(f"{'='*60}")
        print(f"  Cancel→Silence P50: {p50:.1f} ms")
        print(f"  Cancel→Silence P95: {p95:.1f} ms")
        print(f"  Cancel→Silence range: [{min(sorted_lats):.0f}, {max(sorted_lats):.0f}] ms")
        gate_pass = p95 <= 200
        print(f"  Gate P95 ≤ 200ms: {'✅ PASS' if gate_pass else '❌ FAIL'}")

        # 保存结果
        report = {
            "cancel_latencies_ms": cancel_latencies,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "min_ms": round(min(sorted_lats), 1),
            "max_ms": round(max(sorted_lats), 1),
            "gate_cancel_200ms_pass": gate_pass,
            "event_log": ctrl.event_log.dump(),
        }
        report_path = os.path.join(OUTPUT_DIR, "day2_bargein_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Report: {report_path}")

    return cancel_latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplex Controller v0 Demo")
    parser.add_argument("--wav", required=True, help="Input wav for demo")
    parser.add_argument("--keyboard_bargein", type=int, default=0,
                        help="1=keyboard mode (press Enter), 0=auto mode")
    args = parser.parse_args()

    demo_bargein(args.wav, keyboard_mode=bool(args.keyboard_bargein))

