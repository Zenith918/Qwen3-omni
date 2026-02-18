"""
D15 P0-3: Noise-Robust VAD wrapper.

Wraps Silero VAD and adds:
1. Spectral noise gate — suppresses START_OF_SPEECH when audio is pure noise
   (high HF energy ratio or high spectral entropy → not human speech)
2. The underlying Silero VAD handles endpointing silence, min speech duration,
   and activation threshold independently.
"""

import asyncio
import logging
import time
from typing import Literal

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad
from livekit.plugins import silero as silero_plugin

logger = logging.getLogger("noise-robust-vad")

HF_CUTOFF_HZ = 3000
HF_ENERGY_RATIO_THRESHOLD = 0.35
SPECTRAL_ENTROPY_THRESHOLD = 0.85
NOISE_FRAME_SAMPLE_RATE = 16000


class NoiseRobustVAD(agents_vad.VAD):
    """VAD wrapper that adds spectral noise filtering on top of Silero."""

    def __init__(self, inner_vad: silero_plugin.VAD, *, noise_gate_enabled: bool = True):
        super().__init__(capabilities=inner_vad.capabilities)
        self._inner_vad = inner_vad
        self._noise_gate_enabled = noise_gate_enabled

    @property
    def model(self) -> str:
        return f"noise_robust({self._inner_vad.model})"

    @property
    def provider(self) -> str:
        return self._inner_vad.provider

    def stream(self) -> "NoiseRobustVADStream":
        inner_stream = self._inner_vad.stream()
        return NoiseRobustVADStream(
            self, inner_stream, noise_gate_enabled=self._noise_gate_enabled
        )


class NoiseRobustVADStream(agents_vad.VADStream):
    """VAD stream that wraps inner Silero stream and applies spectral filtering."""

    def __init__(self, vad: NoiseRobustVAD, inner_stream, *, noise_gate_enabled: bool):
        super().__init__(vad)
        self._inner = inner_stream
        self._noise_gate_enabled = noise_gate_enabled
        self._suppressed_count = 0

    async def _main_task(self) -> None:
        forward_task = asyncio.create_task(self._forward_frames())
        filter_task = asyncio.create_task(self._filter_events())
        try:
            done, pending = await asyncio.wait(
                [forward_task, filter_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            for t in done:
                if t.exception():
                    raise t.exception()
        except asyncio.CancelledError:
            forward_task.cancel()
            filter_task.cancel()

    async def _forward_frames(self):
        """Forward audio frames from our input to the inner Silero stream."""
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                self._inner.push_frame(item)
            elif isinstance(item, self._FlushSentinel):
                self._inner.flush()

    async def _filter_events(self):
        """Read events from inner stream, apply noise gate, re-emit."""
        async for event in self._inner:
            if (
                self._noise_gate_enabled
                and event.type == agents_vad.VADEventType.START_OF_SPEECH
                and event.frames
                and _is_pure_noise(event.frames)
            ):
                self._suppressed_count += 1
                if self._suppressed_count <= 3 or self._suppressed_count % 10 == 0:
                    logger.info(
                        f"[NoiseGate] Suppressed noise-triggered START_OF_SPEECH "
                        f"(total suppressed: {self._suppressed_count})"
                    )
                continue

            self._event_ch.send_nowait(event)

    async def aclose(self) -> None:
        await self._inner.aclose()
        await super().aclose()


def _is_pure_noise(frames: list[rtc.AudioFrame]) -> bool:
    """
    Check if audio frames contain pure noise rather than human speech.

    Uses two spectral features:
    1. HF energy ratio (>3kHz / total) — noise is broad-spectrum, speech is 300-3kHz
    2. Spectral entropy — noise has flat spectrum (high entropy), speech is peaky (low entropy)
    """
    try:
        all_data = []
        sr = NOISE_FRAME_SAMPLE_RATE
        for f in frames:
            data = np.frombuffer(f.data, dtype=np.int16)
            sr = f.sample_rate
            all_data.append(data)

        if not all_data:
            return False

        audio = np.concatenate(all_data).astype(np.float32) / 32768.0
        if len(audio) < 256:
            return False

        fft = np.abs(np.fft.rfft(audio))
        if np.sum(fft) < 1e-8:
            return False

        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

        # 1. HF energy ratio
        power = fft ** 2
        total_power = np.sum(power)
        if total_power < 1e-12:
            return False

        hf_mask = freqs > HF_CUTOFF_HZ
        hf_ratio = np.sum(power[hf_mask]) / total_power

        # 2. Spectral entropy (normalized)
        power_norm = power / total_power
        power_norm = power_norm[power_norm > 1e-12]
        entropy = -np.sum(power_norm * np.log2(power_norm))
        max_entropy = np.log2(len(power))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        is_noise = hf_ratio > HF_ENERGY_RATIO_THRESHOLD and norm_entropy > SPECTRAL_ENTROPY_THRESHOLD

        return is_noise

    except Exception as e:
        logger.debug(f"[NoiseGate] Spectral analysis error: {e}")
        return False
