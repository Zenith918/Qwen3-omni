#!/usr/bin/env python3
"""
D7 P1-1: 生成多样性测试音频

生成 4 类触发音频：
  1. 爆音触发（高能量短促 + 语音）
  2. 语速漂移触发（长元音/数字串）
  3. 失真触发（sibilant 高频）
  4. 卡顿触发（长句 + 中间停顿）

使用 TTS 生成 + numpy 后处理
"""

import os
import sys
import wave
import numpy as np
import requests

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "serena")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "output", "autortc_cases")


def tts_synthesize(text: str) -> bytes:
    """调 TTS server 合成"""
    resp = requests.post(TTS_URL, json={"text": text, "speaker": TTS_SPEAKER},
                         stream=True, timeout=60)
    resp.raise_for_status()
    chunks = []
    for c in resp.iter_content(chunk_size=8192):
        if c:
            chunks.append(c)
    return b"".join(chunks)


def save_wav(path: str, pcm: bytes, sr: int = 24000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print(f"  Saved: {path} ({len(pcm)} bytes, {len(pcm)/2/sr:.1f}s)")


def gen_boom_trigger():
    """爆音触发：高能量尖峰 + 正常语音"""
    pcm = tts_synthesize("你好，今天天气真不错。")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    # 在开头插入 10ms 高能量尖峰
    spike = (np.random.randn(int(24000 * 0.01)) * 30000).astype(np.int16)
    result = np.concatenate([spike, np.frombuffer(pcm, dtype=np.int16)])
    path = os.path.join(OUTPUT_DIR, "boom_trigger.wav")
    save_wav(path, result.tobytes())
    return path


def gen_speed_drift():
    """语速漂移触发：长数字串 + 长元音"""
    pcm = tts_synthesize("一二三四五六七八九十，啊啊啊啊啊，请问这个数字是多少？")
    path = os.path.join(OUTPUT_DIR, "speed_drift.wav")
    save_wav(path, pcm)
    return path


def gen_distortion_trigger():
    """失真触发：高频 sibilant"""
    pcm = tts_synthesize("师傅说这首诗是上世纪四十四年写的，属实是神作。")
    path = os.path.join(OUTPUT_DIR, "distortion_sibilant.wav")
    save_wav(path, pcm)
    return path


def gen_stutter_trigger():
    """卡顿触发：长句 + 中间 2s 静音（模拟停顿后继续）"""
    pcm1 = tts_synthesize("我今天去了超市，买了很多东西。")
    pcm2 = tts_synthesize("然后又去了公园散步，遇到了一个老朋友。")
    silence = np.zeros(int(24000 * 2), dtype=np.int16)  # 2s 静音
    result = np.concatenate([
        np.frombuffer(pcm1, dtype=np.int16),
        silence,
        np.frombuffer(pcm2, dtype=np.int16),
    ])
    path = os.path.join(OUTPUT_DIR, "stutter_long_pause.wav")
    save_wav(path, result.tobytes())
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating D7 P1-1 test cases...")

    paths = {}
    print("\n1. Boom trigger:")
    paths["boom_trigger"] = gen_boom_trigger()

    print("\n2. Speed drift:")
    paths["speed_drift"] = gen_speed_drift()

    print("\n3. Distortion sibilant:")
    paths["distortion_sibilant"] = gen_distortion_trigger()

    print("\n4. Stutter long pause:")
    paths["stutter_long_pause"] = gen_stutter_trigger()

    # 生成 cases JSON
    import json
    cases = {"cases": [
        {"case_id": k, "wav": v} for k, v in paths.items()
    ]}
    cases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cases", "p1_cases.json")
    with open(cases_path, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nCases JSON: {cases_path}")
    print("Done!")


if __name__ == "__main__":
    main()

