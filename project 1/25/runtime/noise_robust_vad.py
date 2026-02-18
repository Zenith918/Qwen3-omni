"""
D15 P0-3 + D16 P0-2: Noise-Robust VAD wrapper with adaptive endpointing.

Wraps Silero VAD and adds:
1. Spectral noise gate — suppresses START_OF_SPEECH when audio is pure noise
2. D16: Adaptive endpointing hold — delays END_OF_SPEECH based on signal quality
   (clean/short → fast, noisy → conservative)
3. Per-turn EndpointingParams tracking for trace reporting
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
    """VAD wrapper that adds spectral noise filtering + adaptive endpointing."""

    def __init__(self, inner_vad: silero_plugin.VAD, *,
                 noise_gate_enabled: bool = True,
                 endpointing_controller=None):
        super().__init__(capabilities=inner_vad.capabilities)
        self._inner_vad = inner_vad
        self._noise_gate_enabled = noise_gate_enabled
        self._endpointing_controller = endpointing_controller

    @property
    def model(self) -> str:
        return f"noise_robust({self._inner_vad.model})"

    @property
    def provider(self) -> str:
        return self._inner_vad.provider

    def stream(self) -> "NoiseRobustVADStream":
        inner_stream = self._inner_vad.stream()
        return NoiseRobustVADStream(
            self, inner_stream,
            noise_gate_enabled=self._noise_gate_enabled,
            endpointing_controller=self._endpointing_controller,
        )


class NoiseRobustVADStream(agents_vad.VADStream):
    """VAD stream: spectral noise gate + D16 adaptive endpointing hold."""

    def __init__(self, vad: NoiseRobustVAD, inner_stream, *,
                 noise_gate_enabled: bool,
                 endpointing_controller=None):
        super().__init__(vad)
        self._inner = inner_stream
        self._noise_gate_enabled = noise_gate_enabled
        self._controller = endpointing_controller
        self._suppressed_count = 0
        self._hold_task: asyncio.Task | None = None
        self._in_speech = False
        self.last_endpointing_params: dict | None = None

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
        """Forward audio frames to inner Silero stream + feed controller."""
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                self._inner.push_frame(item)
                if self._controller:
                    try:
                        pcm = np.frombuffer(item.data, dtype=np.int16)
                        self._controller.on_audio_frame(pcm, item.sample_rate)
                    except Exception:
                        pass
            elif isinstance(item, self._FlushSentinel):
                self._inner.flush()

    async def _filter_events(self):
        """Read events from inner stream, apply noise gate + adaptive hold."""
        async for event in self._inner:
            if event.type == agents_vad.VADEventType.START_OF_SPEECH:
                # Cancel any pending held END_OF_SPEECH
                if self._hold_task and not self._hold_task.done():
                    self._hold_task.cancel()
                    self._hold_task = None
                    self._in_speech = True
                    if self._controller:
                        self._controller.on_speech_start()
                    continue  # suppress START since we never emitted END

                # Noise gate
                if (self._noise_gate_enabled
                        and event.frames
                        and _is_pure_noise(event.frames)):
                    self._suppressed_count += 1
                    if self._suppressed_count <= 3 or self._suppressed_count % 10 == 0:
                        logger.info(
                            f"[NoiseGate] Suppressed START_OF_SPEECH "
                            f"(total: {self._suppressed_count})"
                        )
                    continue

                self._in_speech = True
                if self._controller:
                    self._controller.on_speech_start()
                self._event_ch.send_nowait(event)

            elif event.type == agents_vad.VADEventType.END_OF_SPEECH:
                # Cancel any previous pending END
                if self._hold_task and not self._hold_task.done():
                    self._hold_task.cancel()

                if self._controller:
                    decision = self._controller.compute_hold()
                    self.last_endpointing_params = decision.to_dict()
                    hold_ms = decision.extra_hold_ms
                    if hold_ms > 0:
                        self._hold_task = asyncio.create_task(
                            self._delayed_emit_end(event, hold_ms / 1000.0)
                        )
                        continue

                self._in_speech = False
                self._event_ch.send_nowait(event)

            else:
                self._event_ch.send_nowait(event)

    async def _delayed_emit_end(self, event, hold_s: float):
        """Hold END_OF_SPEECH, then emit if not cancelled by new speech."""
        try:
            await asyncio.sleep(hold_s)
            self._in_speech = False
            self._event_ch.send_nowait(event)
        except asyncio.CancelledError:
            logger.debug("[AdaptiveHold] END cancelled — speech resumed during hold")

    async def aclose(self) -> None:
        if self._hold_task and not self._hold_task.done():
            self._hold_task.cancel()
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
