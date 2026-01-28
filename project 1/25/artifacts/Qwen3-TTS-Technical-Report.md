Qwen3-TTS Technical Report (saved copy)
Source:
- https://arxiv.org/abs/2601.15621
- https://arxiv.org/html/2601.15621v1

Key details referenced for engineering:
- Tokenizer-12Hz is a 12.5 Hz multi-codebook tokenizer with fully causal encoder/decoder.
- Left-context-only decoding supports immediate waveform emission once codes are available.
- Packetization in the report uses 4 tokens ~= 320 ms per packet (12.5 Hz token rate).
- Multi-codebook: first codebook is semantic, residual codebooks add acoustic detail.
- Reported first-packet latency is the sum of LM time-to-first packet tokens (TTFP) + tokenizer decode time per packet.
- 12Hz variants report first-packet latency ~97 ms (0.6B) / ~101 ms (1.7B) at concurrency 1.

Streaming notes from the report (paraphrased):
- 25Hz tokenizer uses block-wise DiT with look-ahead; needs enough future tokens.
- 12Hz tokenizer uses causal ConvNet decoder; no right-context look-ahead needed.
- A "packet" is defined as 4 tokens to avoid scheduling overhead.

Why this matters in our PoC:
- We should emit PCM every 4 code frames (4 tokens) to match 320 ms packets.
- Decoder must be left-context-only and causal to avoid waiting for future frames.
