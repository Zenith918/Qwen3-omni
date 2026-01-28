Qwen3-Omni Technical Report (saved copy)
Source:
- https://arxiv.org/abs/2509.17765
- https://arxiv.org/html/2509.17765v1

Key details referenced for engineering:
- Talker predicts multi-codebook codec frames autoregressively.
- MTP module outputs residual codebooks for each frame.
- Code2Wav is a lightweight causal ConvNet; streaming from the first codec frame.
- Theoretical first-packet latency (audio) reported as low as 234 ms (cold start, concurrency 1).
- Output codec runs at 12.5 Hz; 1 token ~= 80 ms audio.

Streaming notes from the report (paraphrased):
- "Left context only" multi-codebook generation allows immediate waveform output.
- Code2Wav + MTP are designed for low-latency and batched inference.
- Latency components: prefill, TTPT (Talker/Thinker), MTP + codec decode.

Why this matters in our PoC:
- We should expose per-frame multi-codebook tokens to decode incrementally.
- Packet size should aggregate 4 frames (4 tokens) to reach ~320 ms per packet.
