#!/usr/bin/env python3
"""
Day 3 P0-2: Silero VAD 集成

功能:
  - 从 wav 文件/实时音频帧检测 speech segments
  - 输出 speech_start / speech_end (endpointer)
  - 支持 hangover (说完后等待 N ms 确认结束)
  - 支持误判/漏判评估

用法:
  # 从 wav 文件检测 VAD
  python3 runtime/vad_silero.py --wav input.wav

  # 评估测试集
  python3 runtime/vad_silero.py --eval_dir test_wavs/
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

# ── Silero VAD 包装器 ───────────────────────────────────────
class SileroVAD:
    """
    Silero VAD v5 包装器，支持帧级别和文件级别 VAD。

    参数:
        threshold: VAD 概率阈值 (默认 0.5)
        min_speech_ms: 最短语音段 (ms)
        min_silence_ms: hangover — 语音结束后等待确认的时间 (ms)
        sample_rate: 采样率 (必须 16000 或 8000)
        frame_ms: 每帧时长 (ms)，Silero 支持 30/60/100ms
    """

    def __init__(self,
                 threshold: float = 0.5,
                 min_speech_ms: int = 250,
                 min_silence_ms: int = 300,
                 sample_rate: int = 16000,
                 frame_ms: int = 32):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        # Silero VAD v5 requires exactly 512 samples @16kHz or 256 @8kHz
        self.frame_samples = 512 if sample_rate == 16000 else 256
        self.frame_ms = self.frame_samples * 1000.0 / sample_rate  # actual: 32ms @16kHz

        # Load Silero VAD model (CPU only — no GPU needed)
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.model.eval()

        # Streaming state
        self.reset()

    def reset(self):
        """重置流式状态"""
        self.model.reset_states()
        self._is_speech = False
        self._speech_start_ms: Optional[float] = None
        self._silence_start_ms: Optional[float] = None
        self._current_ms: float = 0.0
        self._segments: list[dict] = []

    def process_frame(self, audio_frame: np.ndarray) -> dict:
        """
        处理一帧音频 (numpy int16 或 float32)。

        返回:
            {
                "is_speech": bool,
                "probability": float,
                "event": None | "speech_start" | "speech_end",
                "event_ms": float (如果有事件),
            }
        """
        # 转换格式
        if audio_frame.dtype == np.int16:
            audio_f32 = audio_frame.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_frame.astype(np.float32)

        # Ensure correct frame size
        if len(audio_f32) != self.frame_samples:
            # Pad or truncate
            if len(audio_f32) < self.frame_samples:
                audio_f32 = np.pad(audio_f32, (0, self.frame_samples - len(audio_f32)))
            else:
                audio_f32 = audio_f32[:self.frame_samples]

        tensor = torch.from_numpy(audio_f32)

        # Run Silero VAD
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()

        frame_end_ms = self._current_ms + self.frame_ms
        result = {
            "is_speech": False,
            "probability": prob,
            "event": None,
            "event_ms": None,
            "frame_ms": self._current_ms,
        }

        if prob >= self.threshold:
            # Speech detected
            if not self._is_speech:
                # Speech start
                self._is_speech = True
                self._speech_start_ms = self._current_ms
                self._silence_start_ms = None
                result["event"] = "speech_start"
                result["event_ms"] = self._current_ms
            else:
                # Continuing speech — reset silence counter
                self._silence_start_ms = None
            result["is_speech"] = True
        else:
            # Silence
            if self._is_speech:
                # Was speaking, now silence
                if self._silence_start_ms is None:
                    self._silence_start_ms = self._current_ms

                silence_duration = frame_end_ms - self._silence_start_ms
                if silence_duration >= self.min_silence_ms:
                    # Hangover exceeded — speech ended
                    speech_duration = self._silence_start_ms - (self._speech_start_ms or 0)
                    if speech_duration >= self.min_speech_ms:
                        segment = {
                            "start_ms": self._speech_start_ms,
                            "end_ms": self._silence_start_ms,
                            "duration_ms": speech_duration,
                        }
                        self._segments.append(segment)
                        result["event"] = "speech_end"
                        result["event_ms"] = self._silence_start_ms
                    self._is_speech = False
                    self._speech_start_ms = None
                    self._silence_start_ms = None
                else:
                    # Still in hangover
                    result["is_speech"] = True

        self._current_ms = frame_end_ms
        return result

    def process_wav(self, wav_path: str) -> list[dict]:
        """
        处理整个 wav 文件，返回 VAD segments。
        """
        import wave

        self.reset()

        with wave.open(wav_path, 'rb') as wf:
            assert wf.getsampwidth() == 2, "Only 16-bit PCM supported"
            orig_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw_bytes = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw_bytes, dtype=np.int16)

        # Convert to mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

        # Resample if needed
        if orig_rate != self.sample_rate:
            from scipy import signal as sig
            num_samples = int(len(audio) * self.sample_rate / orig_rate)
            audio = sig.resample(audio, num_samples).astype(np.int16)

        # Process frame by frame
        events = []
        for start in range(0, len(audio), self.frame_samples):
            frame = audio[start:start + self.frame_samples]
            if len(frame) < self.frame_samples // 2:
                break  # Skip very short final frame
            result = self.process_frame(frame)
            if result["event"]:
                events.append(result)

        # Flush: if still speaking at end of file
        if self._is_speech and self._speech_start_ms is not None:
            duration = self._current_ms - self._speech_start_ms
            if duration >= self.min_speech_ms:
                segment = {
                    "start_ms": self._speech_start_ms,
                    "end_ms": self._current_ms,
                    "duration_ms": duration,
                }
                self._segments.append(segment)
                events.append({
                    "event": "speech_end",
                    "event_ms": self._current_ms,
                    "is_speech": False,
                    "probability": 0.0,
                    "frame_ms": self._current_ms,
                })

        return {
            "segments": self._segments,
            "events": events,
            "total_duration_ms": self._current_ms,
            "speech_duration_ms": sum(s["duration_ms"] for s in self._segments),
        }

    def get_last_speech_end(self) -> Optional[float]:
        """返回最后一个 speech_end 时间 (ms)"""
        if self._segments:
            return self._segments[-1]["end_ms"]
        return None


# ── VAD 评估 ────────────────────────────────────────────────
def evaluate_vad(vad: SileroVAD, test_cases: list[dict]) -> dict:
    """
    评估 VAD 在一组测试用例上的表现。

    test_cases: [{"wav": path, "expected": "speech"|"silence"|"mixed",
                  "has_speech": bool}]
    """
    results = []
    false_positives = 0  # 噪声/安静被误判为语音
    false_negatives = 0  # 语音被漏判
    total = len(test_cases)

    for tc in test_cases:
        wav_path = tc["wav"]
        expected_has_speech = tc.get("has_speech", True)

        vad_result = vad.process_wav(wav_path)
        detected_speech = len(vad_result["segments"]) > 0

        correct = detected_speech == expected_has_speech
        if not correct:
            if detected_speech and not expected_has_speech:
                false_positives += 1
            elif not detected_speech and expected_has_speech:
                false_negatives += 1

        results.append({
            "wav": wav_path,
            "expected_speech": expected_has_speech,
            "detected_speech": detected_speech,
            "correct": correct,
            "segments": vad_result["segments"],
            "speech_duration_ms": vad_result["speech_duration_ms"],
        })

    fp_rate = false_positives / max(1, sum(1 for t in test_cases if not t.get("has_speech", True)))
    fn_rate = false_negatives / max(1, sum(1 for t in test_cases if t.get("has_speech", True)))

    return {
        "total": total,
        "correct": sum(1 for r in results if r["correct"]),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fp_rate": round(fp_rate, 4),
        "fn_rate": round(fn_rate, 4),
        "gate_fp_le_2pct": fp_rate <= 0.02,
        "gate_fn_le_2pct": fn_rate <= 0.02,
        "details": results,
    }


# ── 生成测试音频 ─────────────────────────────────────────────
def generate_test_audio(output_dir: str) -> list[dict]:
    """生成 VAD 测试用例音频"""
    import wave
    os.makedirs(output_dir, exist_ok=True)
    test_cases = []
    sr = 16000

    def save_wav(path, audio_int16):
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio_int16.tobytes())

    # 1. 纯安静 (2 秒)
    silence = np.zeros(sr * 2, dtype=np.int16)
    p = os.path.join(output_dir, "silence_2s.wav")
    save_wav(p, silence)
    test_cases.append({"wav": p, "has_speech": False, "label": "pure_silence"})

    # 2. 低噪声 (2 秒)
    noise = (np.random.randn(sr * 2) * 100).astype(np.int16)
    p = os.path.join(output_dir, "low_noise_2s.wav")
    save_wav(p, noise)
    test_cases.append({"wav": p, "has_speech": False, "label": "low_noise"})

    # 3. 中等噪声 (2 秒)
    noise_med = (np.random.randn(sr * 2) * 800).astype(np.int16)
    p = os.path.join(output_dir, "med_noise_2s.wav")
    save_wav(p, noise_med)
    test_cases.append({"wav": p, "has_speech": False, "label": "medium_noise"})

    # 4. 短脉冲（模拟咳嗽/点击，50ms）
    click = np.zeros(sr * 2, dtype=np.int16)
    click_start = int(sr * 0.5)
    click[click_start:click_start + int(sr * 0.05)] = (np.random.randn(int(sr * 0.05)) * 5000).astype(np.int16)
    p = os.path.join(output_dir, "click_50ms.wav")
    save_wav(p, click)
    test_cases.append({"wav": p, "has_speech": False, "label": "short_click"})

    # 5. 使用已有的测试音频（真实语音）
    real_wav = "/workspace/project 1/25/output/day1_test_input.wav"
    if os.path.exists(real_wav):
        test_cases.append({"wav": real_wav, "has_speech": True, "label": "real_speech"})

    # 6. 使用 TTS 生成的回复（真实语音）
    reply_wav = "/workspace/project 1/25/output/day2_reply.wav"
    if os.path.exists(reply_wav):
        test_cases.append({"wav": reply_wav, "has_speech": True, "label": "tts_speech"})

    return test_cases


# ── CLI ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Silero VAD integration")
    parser.add_argument("--wav", type=str, help="Process a single wav file")
    parser.add_argument("--eval", action="store_true", help="Run VAD evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_speech_ms", type=int, default=250)
    parser.add_argument("--min_silence_ms", type=int, default=300)
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

    vad = SileroVAD(
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
    )
    print(f"Silero VAD initialized (threshold={args.threshold}, "
          f"min_speech={args.min_speech_ms}ms, hangover={args.min_silence_ms}ms)")

    if args.wav:
        result = vad.process_wav(args.wav)
        print(f"\nVAD result for: {args.wav}")
        print(f"  Total duration: {result['total_duration_ms']:.0f} ms")
        print(f"  Speech duration: {result['speech_duration_ms']:.0f} ms")
        print(f"  Segments: {len(result['segments'])}")
        for seg in result["segments"]:
            print(f"    [{seg['start_ms']:.0f} - {seg['end_ms']:.0f}] "
                  f"({seg['duration_ms']:.0f}ms)")
        print(f"  Events:")
        for ev in result["events"]:
            print(f"    {ev['event']} at {ev['event_ms']:.0f}ms "
                  f"(prob={ev['probability']:.3f})")

        # 保存结果
        result_path = os.path.join(OUTPUT_DIR, "day3_vad_result.json")
        # Remove numpy types for JSON serialization
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Saved: {result_path}")

    if args.eval:
        print(f"\n{'='*60}")
        print(f"  VAD Evaluation")
        print(f"{'='*60}")

        test_dir = os.path.join(OUTPUT_DIR, "vad_test_audio")
        test_cases = generate_test_audio(test_dir)
        print(f"  Generated {len(test_cases)} test cases")

        eval_result = evaluate_vad(vad, test_cases)

        print(f"  Correct:         {eval_result['correct']}/{eval_result['total']}")
        print(f"  False positives: {eval_result['false_positives']} "
              f"(rate: {eval_result['fp_rate']*100:.1f}%)")
        print(f"  False negatives: {eval_result['false_negatives']} "
              f"(rate: {eval_result['fn_rate']*100:.1f}%)")
        print(f"  Gate FP ≤ 2%:    {'✅ PASS' if eval_result['gate_fp_le_2pct'] else '❌ FAIL'}")
        print(f"  Gate FN ≤ 2%:    {'✅ PASS' if eval_result['gate_fn_le_2pct'] else '❌ FAIL'}")

        for d in eval_result["details"]:
            status = "✅" if d["correct"] else "❌"
            print(f"    {status} {os.path.basename(d['wav'])}: "
                  f"expected={d['expected_speech']} detected={d['detected_speech']} "
                  f"segs={len(d['segments'])}")

        eval_path = os.path.join(OUTPUT_DIR, "day3_vad_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2, default=str)
        print(f"\n  Eval saved: {eval_path}")


if __name__ == "__main__":
    main()

