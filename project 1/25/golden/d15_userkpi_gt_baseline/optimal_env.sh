#!/bin/bash
# D15 P0-4: Auto-selected optimal endpointing config
export VAD_ENDPOINTING_SILENCE_MS=500
export ENDPOINTING_DELAY_MS=300
export BARGEIN_ACTIVATION_THRESHOLD=0.7
export BARGEIN_MIN_SPEECH_MS=120
export NOISE_GATE_ENABLED=1
export MODE=turn_taking
