"""
D16 P0-2: Adaptive Endpointing Controller.

Dynamically adjusts extra hold time after VAD END_OF_SPEECH based on
input signal quality (SNR, noise floor) and utterance characteristics.

Integration: Used by NoiseRobustVADStream to hold END_OF_SPEECH events
for a variable duration — fast for clean speech, conservative for noisy.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("endpointing-controller")

FRAME_ANALYSIS_WINDOW_S = 2.0
DEFAULT_SAMPLE_RATE = 48000


@dataclass
class EndpointingDecision:
    extra_hold_ms: int
    reason: str
    snr_est_db: float
    noise_floor_rms: float
    utterance_dur_s: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "extra_hold_ms": self.extra_hold_ms,
            "reason": self.reason,
            "snr_est_db": round(self.snr_est_db, 1),
            "noise_floor_rms": round(self.noise_floor_rms, 6),
            "utterance_dur_s": round(self.utterance_dur_s, 2),
        }


class EndpointingController:
    """
    Per-session adaptive endpointing.

    Maintains a rolling buffer of audio energy estimates. When VAD fires
    END_OF_SPEECH, compute_hold() returns how many extra ms to wait before
    forwarding the event downstream.

    Rules (turn_taking mode):
      - SNR >= 15 dB AND utterance < 2.0s  → 0 ms  (fast path)
      - SNR >= 15 dB AND utterance >= 2.0s → 100 ms (medium)
      - SNR < 15 dB  OR noise detected     → 300 ms (conservative)

    The VAD-level silence (300ms) + extra_hold + pipeline delay (200ms)
    gives total endpointing latency of 500–800ms.
    """

    def __init__(self, *, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 snr_fast_threshold_db: float = 15.0,
                 short_utterance_s: float = 2.0,
                 fast_hold_ms: int = 0,
                 medium_hold_ms: int = 100,
                 conservative_hold_ms: int = 300):
        self._sample_rate = sample_rate
        self._snr_fast_threshold = snr_fast_threshold_db
        self._short_utterance_s = short_utterance_s
        self._fast_hold = fast_hold_ms
        self._medium_hold = medium_hold_ms
        self._conservative_hold = conservative_hold_ms

        self._energy_buffer: deque[tuple[float, float]] = deque(maxlen=500)
        self._speech_start_time: float | None = None
        self._last_decision: EndpointingDecision | None = None

    def on_audio_frame(self, pcm_int16: np.ndarray, sample_rate: int):
        """Record energy from an audio frame for rolling SNR estimation."""
        if len(pcm_int16) == 0:
            return
        audio_f = pcm_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_f ** 2)))
        self._energy_buffer.append((time.time(), rms))
        self._sample_rate = sample_rate

    def on_speech_start(self):
        """Called when START_OF_SPEECH fires."""
        self._speech_start_time = time.time()

    def compute_hold(self) -> EndpointingDecision:
        """
        Called when END_OF_SPEECH fires. Returns the extra hold duration.
        """
        now = time.time()
        utterance_dur = (now - self._speech_start_time) if self._speech_start_time else 0.0

        snr_db, noise_floor = self._estimate_snr(now)

        if snr_db >= self._snr_fast_threshold and utterance_dur < self._short_utterance_s:
            hold = self._fast_hold
            reason = "clean_short"
        elif snr_db >= self._snr_fast_threshold:
            hold = self._medium_hold
            reason = "clean_long"
        else:
            hold = self._conservative_hold
            reason = "noisy"

        decision = EndpointingDecision(
            extra_hold_ms=hold,
            reason=reason,
            snr_est_db=snr_db,
            noise_floor_rms=noise_floor,
            utterance_dur_s=utterance_dur,
        )
        self._last_decision = decision

        if logger.isEnabledFor(logging.DEBUG) or hold > 0:
            logger.info(
                f"[EndpCtrl] hold={hold}ms reason={reason} "
                f"SNR={snr_db:.1f}dB noise={noise_floor:.5f} "
                f"utt={utterance_dur:.2f}s"
            )

        return decision

    @property
    def last_decision(self) -> EndpointingDecision | None:
        return self._last_decision

    def _estimate_snr(self, now: float) -> tuple[float, float]:
        """
        Estimate SNR from the energy buffer.

        Uses the bottom 10% of frame energies as noise floor estimate,
        and the top 50% as signal estimate.
        """
        window_start = now - FRAME_ANALYSIS_WINDOW_S
        recent = [rms for ts, rms in self._energy_buffer if ts >= window_start]

        if len(recent) < 5:
            return 30.0, 0.001  # assume clean if too few samples

        sorted_e = sorted(recent)
        n = len(sorted_e)

        noise_floor = float(np.mean(sorted_e[:max(1, n // 10)]))
        signal_level = float(np.mean(sorted_e[n // 2:]))

        if noise_floor < 1e-6:
            noise_floor = 1e-6

        snr_db = 20.0 * np.log10(max(signal_level, 1e-6) / noise_floor)
        snr_db = min(snr_db, 60.0)

        return snr_db, noise_floor
