#!/usr/bin/env python3
"""
Day 3 P0-3: Live Duplex Demo

模拟真实通话场景：
  - 用 wav 文件模拟麦克风输入（流式送帧给 VAD）
  - VAD 检测 speech_end → Omni fast lane → TTS → 播放
  - 播放期间持续 VAD，检测到 barge-in → cancel 级联 → 重新 LISTENING
  - Slow lane 延后到回合结束后 2-5s 运行

架构：
  SimulatedMic → VAD → DuplexController → Omni → TTS → (simulated playback)
                  ↑
          (barge-in detection during playback)

用法：
  # 多轮对话模拟（用多个 wav 文件）
  python3 runtime/live_duplex.py --wavs input1.wav input2.wav input3.wav --loops 3

  # 带 barge-in 测试（第 N 轮中途插入新语音）
  python3 runtime/live_duplex.py --wavs input1.wav --loops 5 --bargein_at 2,4
"""

import argparse
import json
import os
import sys
import threading
import time
import wave
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clients"))

from vad_silero import SileroVAD
from duplex_controller import DuplexController, DuplexState, CancelToken
from gpu_scheduler import GPUScheduler
from demo_audio_to_omni import wav_to_base64, PROMPT_FAST

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ── 配置 ──────────────────────────────────────────────────────
SLOW_LANE_DELAY_S = 3.0  # slow lane 延后秒数


class LiveDuplexSession:
    """
    Live duplex 对话会话。
    模拟真实麦克风输入 + VAD + Omni + TTS + barge-in。
    """

    def __init__(self):
        self.vad = SileroVAD(threshold=0.5, min_speech_ms=250, min_silence_ms=300)
        self.ctrl = DuplexController(on_state_change=self._on_state_change)
        self.metrics = []
        self.slow_lane_log = []
        self._slow_lane_thread: Optional[threading.Thread] = None
        # D4: 硬优先级 GPU 调度器
        self.gpu = GPUScheduler(
            slow_budget_ms=800,
            bargein_cooldown_ms=5000,       # barge-in 后 5s 冷却
            fast_baseline_ms=150,           # fast 基线（含 TTS queue wait）
            min_idle_before_slow_ms=5000,   # 至少 5s 空闲才跑 slow
        )

    def _on_state_change(self, old, new, reason):
        print(f"  📡 [{old.value}] → [{new.value}]  ({reason})")

    def process_turn(self, wav_path: str, turn_idx: int,
                     bargein_after_ms: int = 0) -> dict:
        """
        处理一个对话回合：
        1. 流式送帧给 VAD
        2. VAD speech_end → 发 Omni
        3. Omni 回复 → TTS
        4. 如果 bargein_after_ms > 0, TTS 播放 N ms 后模拟 barge-in

        返回 turn metrics dict
        """
        turn = {
            "turn_idx": turn_idx,
            "wav": wav_path,
            "bargein": bargein_after_ms > 0,
        }
        t_turn_start = time.time()

        # ── Step 1: 模拟麦克风输入 + VAD ──
        self.ctrl.reset()
        self.ctrl.start_listening()
        self.vad.reset()

        # 读取 wav
        audio_b64, wav_duration = wav_to_base64(wav_path)

        with wave.open(wav_path, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            orig_rate = wf.getframerate()
            n_ch = wf.getnchannels()

        audio = np.frombuffer(raw, dtype=np.int16)
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch)[:, 0]
        if orig_rate != 16000:
            from scipy import signal
            num = int(len(audio) * 16000 / orig_rate)
            audio = signal.resample(audio, num).astype(np.int16)

        # 流式送帧
        vad_end_time = None
        frame_samples = self.vad.frame_samples
        for start in range(0, len(audio), frame_samples):
            frame = audio[start:start + frame_samples]
            if len(frame) < frame_samples // 2:
                break
            result = self.vad.process_frame(frame)
            if result["event"] == "speech_end":
                vad_end_time = time.time()
                turn["vad_end_offset_ms"] = result["event_ms"]
                break

        # 如果 VAD 没检测到 end，用文件末尾
        if vad_end_time is None:
            vad_end_time = time.time()
            turn["vad_end_offset_ms"] = wav_duration * 1000

        self.ctrl.end_listening()
        t1 = vad_end_time
        turn["t1_vad_end_epoch"] = round(t1 * 1000, 1)

        # ── Step 2: Omni fast lane (stream) ──
        t2 = time.time()
        turn["t2_omni_start_epoch"] = round(t2 * 1000, 1)

        # D4: Fast lane 通过硬优先级调度器独占 GPU
        with self.gpu.fast_lane():
            reply_text = self.ctrl.stream_llm(audio_b64, PROMPT_FAST)
        t4 = time.time()
        turn["t4_reply_ready_epoch"] = round(t4 * 1000, 1)
        turn["omni_ttft_ms"] = round((t4 - t2) * 1000, 1)
        turn["reply_text"] = reply_text.strip()

        if not reply_text.strip():
            print(f"  ⚠ Empty reply, skipping TTS")
            turn["status"] = "empty_reply"
            self.metrics.append(turn)
            return turn

        print(f"  Reply: \"{reply_text.strip()[:50]}\"  (Omni {turn['omni_ttft_ms']:.0f}ms)")

        # ── Step 3: TTS ──
        t5 = time.time()
        turn["t5_tts_start_epoch"] = round(t5 * 1000, 1)

        if bargein_after_ms > 0:
            # Barge-in 模式：在后台跑 TTS，N ms 后触发 cancel
            tts_result_holder = [None]
            tts_done = threading.Event()

            def tts_work():
                try:
                    tts_result_holder[0] = self.ctrl.call_tts_stream(reply_text.strip())
                except Exception as e:
                    tts_result_holder[0] = {"cancelled": True, "error": str(e)}
                tts_done.set()

            t = threading.Thread(target=tts_work, daemon=True)
            t.start()

            # 等待 SPEAKING
            for _ in range(50):
                if self.ctrl.state == DuplexState.SPEAKING:
                    break
                time.sleep(0.05)

            if self.ctrl.state == DuplexState.SPEAKING:
                time.sleep(bargein_after_ms / 1000.0)
                cancel_ms = self.ctrl.cancel("simulated_bargein")
                self.gpu.notify_bargein()  # D4: 通知调度器启动冷却
                turn["bargein_cancel_ms"] = round(cancel_ms, 1)
                print(f"  ⚡ Barge-in cancel: {cancel_ms:.1f}ms")

            tts_done.wait(timeout=30)
            tts_r = tts_result_holder[0] or {}
            turn["tts_cancelled"] = tts_r.get("cancelled", False)
            turn["tts_ttfa_ms"] = tts_r.get("ttfa_ms")
        else:
            # 正常模式
            tts_r = self.ctrl.call_tts_stream(reply_text.strip())
            self.ctrl.end_speaking()
            turn["tts_ttfa_ms"] = tts_r.get("ttfa_ms")
            turn["tts_total_ms"] = tts_r.get("total_ms")
            turn["reply_audio_s"] = tts_r.get("audio_duration_s", 0)

        t6_epoch = t5 + (turn.get("tts_ttfa_ms") or 0) / 1000
        turn["t6_first_audio_epoch"] = round(t6_epoch * 1000, 1)

        # 核心指标
        eot_to_first_audio = round((t6_epoch - t1) * 1000, 1)
        turn["eot_to_first_audio_ms"] = eot_to_first_audio
        print(f"  EoT→FirstAudio: {eot_to_first_audio:.0f}ms  "
              f"TTS TTFA: {turn.get('tts_ttfa_ms', 'N/A')}ms")

        # ── Step 4: Slow lane (延后) ──
        def _slow_lane():
            time.sleep(SLOW_LANE_DELAY_S)
            from demo_audio_to_omni import call_omni
            try:
                # D4: 通过 GPU 调度器非阻塞尝试获取
                with self.gpu.slow_lane() as ctx:
                    if not ctx.result.acquired:
                        self.slow_lane_log.append({
                            "turn": turn_idx, "skipped": True,
                            "reason": ctx.result.reason,
                        })
                        return
                    # 拿到 GPU，执行 slow lane
                    t_start = time.time()
                    res = call_omni(audio_b64, "slow")
                    lat = round((time.time() - t_start) * 1000, 1)
                self.slow_lane_log.append({
                    "turn": turn_idx,
                    "latency_ms": lat,
                    "delayed_s": SLOW_LANE_DELAY_S,
                    "parsed": res.get("parsed_json") is not None,
                })
                print(f"  [slow lane] turn {turn_idx}: {lat:.0f}ms")
            except Exception as e:
                self.slow_lane_log.append({
                    "turn": turn_idx,
                    "error": str(e),
                })

        self._slow_lane_thread = threading.Thread(target=_slow_lane, daemon=True)
        self._slow_lane_thread.start()

        turn["total_turn_ms"] = round((time.time() - t_turn_start) * 1000, 1)
        turn["status"] = "ok"
        self.metrics.append(turn)
        return turn

    def run_session(self, wav_paths: list[str], loops: int = 1,
                    bargein_at: set[int] = set(),
                    bargein_ms: int = 200) -> dict:
        """
        运行多轮对话会话。
        """
        print(f"\n{'='*60}")
        print(f"  Live Duplex Session")
        print(f"  WAVs: {len(wav_paths)}, Loops: {loops}")
        print(f"  Barge-in at turns: {bargein_at or 'none'}")
        print(f"{'='*60}")

        total_turns = 0
        total_bargeins = 0

        for loop_idx in range(loops):
            for wav_idx, wav_path in enumerate(wav_paths):
                turn_idx = loop_idx * len(wav_paths) + wav_idx
                do_bargein = turn_idx in bargein_at
                if do_bargein:
                    total_bargeins += 1

                print(f"\n--- Turn {turn_idx} (loop {loop_idx+1}, "
                      f"wav {wav_idx+1}/{len(wav_paths)})"
                      f"{' [BARGEIN]' if do_bargein else ''} ---")

                try:
                    self.process_turn(
                        wav_path, turn_idx,
                        bargein_after_ms=bargein_ms if do_bargein else 0)
                    total_turns += 1
                except Exception as e:
                    print(f"  ❌ Turn {turn_idx} error: {e}")
                    self.metrics.append({
                        "turn_idx": turn_idx,
                        "status": "error",
                        "error": str(e),
                    })
                    total_turns += 1

                # Brief pause between turns
                time.sleep(0.3)

        # Wait for any pending slow lane
        if self._slow_lane_thread and self._slow_lane_thread.is_alive():
            self._slow_lane_thread.join(timeout=10)

        return self._generate_report(total_turns, total_bargeins)

    def _generate_report(self, total_turns: int, total_bargeins: int) -> dict:
        """生成会话报告"""
        ok_turns = [m for m in self.metrics if m.get("status") == "ok"]
        error_turns = [m for m in self.metrics if m.get("status") == "error"]

        eot_vals = [m["eot_to_first_audio_ms"] for m in ok_turns
                    if m.get("eot_to_first_audio_ms") is not None]
        omni_vals = [m["omni_ttft_ms"] for m in ok_turns
                     if m.get("omni_ttft_ms") is not None]
        cancel_vals = [m["bargein_cancel_ms"] for m in ok_turns
                       if m.get("bargein_cancel_ms") is not None]

        def pct(vals, p):
            if not vals:
                return 0
            s = sorted(vals)
            return s[min(int(len(s) * p / 100), len(s) - 1)]

        report = {
            "total_turns": total_turns,
            "ok_turns": len(ok_turns),
            "error_turns": len(error_turns),
            "total_bargeins": total_bargeins,
            "crashes": len(error_turns),

            "eot_to_first_audio_p50": pct(eot_vals, 50),
            "eot_to_first_audio_p95": pct(eot_vals, 95),
            "omni_ttft_p50": pct(omni_vals, 50),
            "omni_ttft_p95": pct(omni_vals, 95),
            "bargein_cancel_p50": pct(cancel_vals, 50),
            "bargein_cancel_p95": pct(cancel_vals, 95),

            "slow_lane_log": self.slow_lane_log,
            "all_turns": self.metrics,
        }

        # Gates
        # D4: 调度器统计
        report["gpu_scheduler"] = self.gpu.stats.to_dict()

        report["gate_zero_crash"] = len(error_turns) == 0
        report["gate_bargein_p95_le_150ms"] = pct(cancel_vals, 95) <= 150
        report["gate_omni_ttft_p95_le_80ms"] = pct(omni_vals, 95) <= 80
        report["gate_eot_fa_p95_le_450ms"] = pct(eot_vals, 95) <= 450
        report["gate_zero_interference"] = self.gpu.stats.interference_count == 0

        print(f"\n{'='*60}")
        print(f"  Live Duplex Session Report")
        print(f"{'='*60}")
        print(f"  Turns:     {total_turns} ({len(ok_turns)} ok, {len(error_turns)} errors)")
        print(f"  Bargeins:  {total_bargeins}")
        print(f"  Crashes:   {len(error_turns)}")
        if eot_vals:
            print(f"  EoT→FA P50: {report['eot_to_first_audio_p50']:.0f}ms  "
                  f"P95: {report['eot_to_first_audio_p95']:.0f}ms")
        if omni_vals:
            print(f"  Omni TTFT P50: {report['omni_ttft_p50']:.0f}ms  "
                  f"P95: {report['omni_ttft_p95']:.0f}ms")
        if cancel_vals:
            print(f"  Cancel P50: {report['bargein_cancel_p50']:.1f}ms  "
                  f"P95: {report['bargein_cancel_p95']:.1f}ms")
        print(f"  Slow lane runs: {len(self.slow_lane_log)}")
        sl_delayed = sum(1 for s in self.slow_lane_log if s.get("delayed_s"))
        print(f"  Slow lane delayed: {sl_delayed}")

        gs = self.gpu.stats
        print(f"\n  GPU Scheduler:")
        print(f"    Fast blocked by slow: {gs.fast_blocked_count} (total {gs.fast_blocked_total_ms:.0f}ms)")
        print(f"    Slow completed/skipped: {gs.slow_completed}/{gs.slow_skipped + gs.slow_skipped_cooldown}")
        print(f"    Slow skipped (cooldown): {gs.slow_skipped_cooldown}")
        print(f"    Interference count: {gs.interference_count}")

        print(f"\n  Gates:")
        print(f"    0 crash:            {'✅' if report['gate_zero_crash'] else '❌'}")
        print(f"    Bargein P95≤150ms:  {'✅' if report['gate_bargein_p95_le_150ms'] else '❌'}")
        print(f"    Omni TTFT P95≤80ms: {'✅' if report['gate_omni_ttft_p95_le_80ms'] else '❌'}")
        print(f"    EoT→FA P95≤450ms:   {'✅' if report['gate_eot_fa_p95_le_450ms'] else '❌'}")
        print(f"    0 interference:     {'✅' if report['gate_zero_interference'] else '❌'}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Live Duplex Demo")
    parser.add_argument("--wavs", nargs="+", required=True,
                        help="Input wav files (simulated mic turns)")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of loops through the wav list")
    parser.add_argument("--bargein_at", type=str, default="",
                        help="Comma-separated turn indices to trigger barge-in")
    parser.add_argument("--bargein_ms", type=int, default=200,
                        help="Milliseconds into TTS playback to trigger barge-in")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bargein_set = set()
    if args.bargein_at:
        bargein_set = set(int(x) for x in args.bargein_at.split(","))

    session = LiveDuplexSession()
    report = session.run_session(
        args.wavs,
        loops=args.loops,
        bargein_at=bargein_set,
        bargein_ms=args.bargein_ms,
    )

    report_path = os.path.join(OUTPUT_DIR, "day3_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()

