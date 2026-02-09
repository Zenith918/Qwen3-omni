#!/usr/bin/env python3
"""
Decoder CUDA Graph acceleration for Qwen3-TTS IncrementalDecoder.

Flag-controlled: TTS_DECODER_CUDAGRAPH=0|1 (default 0).

Design:
  - Only graphs the conv path (_decode_conv_path) which is 90%+ of decode time.
  - Pre-transformer, quantizer, pre_conv remain eager (tiny cost).
  - Uses pre-allocated static state buffers with copy_() for in-place updates.
  - Automatic fallback to eager on shape mismatch / capture failure.
  - Bit-exact: graph replay produces identical PCM to eager execution.

Integration:
  Called from IncrementalDecoder.decode_incremental() when enabled.
"""

import os
import sys
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Graph-friendly conv operations ──────────────────────────────────────────
# These replace the original _stream_* functions with in-place state updates
# so CUDA Graph can capture and replay them correctly.


def _graph_causal_conv(
    conv_module, x: torch.Tensor, state_buf: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Graph-friendly causal conv. Updates state_buf IN-PLACE via copy_().

    Args:
        conv_module: The CausalConvNet module.
        x: Input tensor (1, C, T).
        state_buf: Pre-allocated static buffer for conv state, or None if k<=1.

    Returns:
        y: Output tensor.
    """
    weight = conv_module.conv.weight
    bias = conv_module.conv.bias
    dilation = conv_module.dilation
    groups = conv_module.conv.groups
    stride = conv_module.stride
    k = int(conv_module.kernel_size)
    x_cast = x.to(weight.dtype)

    if k <= 1:
        y = F.conv1d(x_cast, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)
        return y

    if state_buf is None:
        # Defensive: state_buf should always be set for k > 1 after warmup
        raise RuntimeError(f"state_buf is None for conv with kernel_size={k}")

    # state_buf shape: (B, C, k-1)
    x_cat = torch.cat([state_buf, x_cast], dim=-1)
    y = F.conv1d(x_cat, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)
    # In-place update: write new state back to the SAME buffer
    state_buf.copy_(x_cat[:, :, -(k - 1) :])
    return y


def _graph_transpose_conv(
    conv_module, x: torch.Tensor, prev_buf: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Graph-friendly transpose conv. Updates prev_buf IN-PLACE via copy_().

    For kernel==stride: no prev handling needed.
    For kernel==2*stride: prev_buf is updated each frame.
    """
    weight = conv_module.conv.weight
    bias = conv_module.conv.bias
    stride = int(conv_module.conv.stride[0])
    kernel = int(conv_module.conv.kernel_size[0])
    padding = kernel - stride
    x = x.to(weight.dtype)

    if kernel == stride:
        # No overlap, no prev state needed
        y = F.conv_transpose1d(x, weight, bias=bias, stride=stride, padding=padding)
        return y

    if kernel != 2 * stride:
        raise ValueError(f"unsupported transposed conv kernel={kernel} stride={stride}")

    if prev_buf is None:
        raise RuntimeError(f"prev_buf is None for transposed conv with kernel={kernel} stride={stride}")

    # kernel == 2*stride: process frame by frame with prev_buf
    # prev_buf must be set (we only capture after warmup)
    out_chunks = []
    for idx in range(x.shape[-1]):
        frame = x[..., idx : idx + 1]
        combo = torch.cat([prev_buf, frame], dim=-1)
        y = F.conv_transpose1d(combo, weight, bias=bias, stride=stride, padding=padding)
        out_chunks.append(y)
        prev_buf.copy_(frame)  # In-place update!

    if out_chunks:
        out = torch.cat(out_chunks, dim=-1)
    else:
        out = x.new_zeros((x.shape[0], weight.shape[1], 0))
    return out


def _graph_convnext(
    block, x: torch.Tensor, state_buf: torch.Tensor
) -> torch.Tensor:
    """Graph-friendly ConvNeXt block. Updates state_buf in-place."""
    y = _graph_causal_conv(block.dwconv, x, state_buf)
    y = y.permute(0, 2, 1)
    y = block.norm(y)
    y = block.pwconv1(y)
    y = block.act(y)
    y = block.pwconv2(y)
    y = block.gamma * y
    y = y.permute(0, 2, 1)
    y = x + y
    return y


def _graph_residual_unit(
    unit, x: torch.Tensor, conv1_buf: Optional[torch.Tensor], conv2_buf: Optional[torch.Tensor]
) -> torch.Tensor:
    """Graph-friendly residual unit. Updates conv buffers in-place."""
    residual = x
    y = unit.act1(x)
    y = _graph_causal_conv(unit.conv1, y, conv1_buf)
    y = unit.act2(y)
    y = _graph_causal_conv(unit.conv2, y, conv2_buf)
    return y + residual


def _graph_decode_conv_path(
    decoder_module, hidden: torch.Tensor, state_bufs: dict
) -> torch.Tensor:
    """
    Full conv path using graph-friendly operations with in-place state updates.

    Args:
        decoder_module: The decoder sub-module (tokenizer.model.decoder).
        hidden: Input tensor (1, C, T) from pre_transformer output.
        state_bufs: Dict of pre-allocated static state buffers.

    Returns:
        Output PCM-level tensor (1, 1, samples).
    """
    # Upsample stages
    for idx, blocks in enumerate(decoder_module.upsample):
        trans = blocks[0]
        hidden = _graph_transpose_conv(
            trans, hidden, state_bufs.get(f"upsample_trans_{idx}")
        )
        convnext = blocks[1]
        hidden = _graph_convnext(convnext, hidden, state_bufs[f"upsample_dw_{idx}"])

    # Decoder conv blocks
    db = decoder_module.decoder
    hidden = _graph_causal_conv(db[0], hidden, state_bufs["decoder_pre"])

    block_idx = 0
    for block in db[1:-2]:
        if not hasattr(block, "block"):
            continue
        # SnakeBeta activation
        hidden = block.block[0](hidden)
        # Transposed conv
        hidden = _graph_transpose_conv(
            block.block[1], hidden, state_bufs[f"block_{block_idx}_trans"]
        )
        # Residual units
        for unit_idx, unit in enumerate(block.block[2:]):
            hidden = _graph_residual_unit(
                unit,
                hidden,
                state_bufs.get(f"block_{block_idx}_res_{unit_idx}_conv1"),
                state_bufs.get(f"block_{block_idx}_res_{unit_idx}_conv2"),
            )
        block_idx += 1

    # Final activation + conv
    hidden = db[-2](hidden)
    hidden = _graph_causal_conv(db[-1], hidden, state_bufs["decoder_post"])
    hidden = hidden.clamp(min=-1, max=1).to(torch.float32)
    return hidden


# ── DecoderConvGraphAccelerator ─────────────────────────────────────────────


@dataclass
class DecoderGraphStats:
    """Runtime statistics for decoder CUDA Graph."""
    graph_replays: int = 0
    eager_steps: int = 0
    fallback_count: int = 0
    fallback_reasons: list = field(default_factory=list)
    capture_time_ms: float = 0.0


class DecoderConvGraphAccelerator:
    """
    Manages CUDA Graph capture and replay for the decoder conv path.

    Supports two modes:
      A) **Pre-capture** (server mode): Graph is captured once during init,
         before any concurrent codegen threads. At request time only
         copy_() + replay() is used, avoiding the process-wide CUDA
         capture lock that would kill concurrent codegen threads.
      B) **Lazy capture** (microbench mode): Graph is captured on the 2nd
         decode step. Only safe when no concurrent GPU work happens.

    Usage (server):
        accel = DecoderConvGraphAccelerator(decoder_module)
        accel.pre_capture(inc_decoder)  # during init, no concurrent work
        # ... per request ...
        step 1: eager (initialise state)
        step 2+: accel.bind_and_replay(hidden, state)

    Usage (microbench):
        accel = DecoderConvGraphAccelerator(decoder_module)
        step 1: eager
        step 2: accel.capture(hidden, state)
        step 3+: accel.replay(hidden, state)
    """

    def __init__(self, decoder_module):
        """
        Args:
            decoder_module: The decoder (tokenizer.model.decoder).
        """
        self.decoder_module = decoder_module
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_hidden: Optional[torch.Tensor] = None
        self.static_out: Optional[torch.Tensor] = None
        self.state_bufs: dict = {}
        self.captured = False
        self._expected_shape: Optional[tuple] = None
        self.stats = DecoderGraphStats()
        self._warmup_done = False
        self._pre_captured = False  # True if graph was pre-captured at init
        # Dedicated CUDA stream for graph capture/replay isolation
        self._stream = torch.cuda.Stream()

    def _collect_and_freeze_states(self, state) -> dict:
        """
        Collect all state tensors from DecoderState and replace with
        pre-allocated static buffers. Returns the buffer dict.
        """
        bufs = {}

        # Upsample stages
        for idx in range(len(state.upsample_trans_prev)):
            # Transpose conv prev (only for kernel != stride)
            if state.upsample_trans_prev[idx] is not None:
                bufs[f"upsample_trans_{idx}"] = state.upsample_trans_prev[idx].clone()
                state.upsample_trans_prev[idx] = bufs[f"upsample_trans_{idx}"]
            # ConvNeXt dwconv state
            if state.upsample_dwconv[idx] is not None:
                bufs[f"upsample_dw_{idx}"] = state.upsample_dwconv[idx].clone()
                state.upsample_dwconv[idx] = bufs[f"upsample_dw_{idx}"]

        # Decoder pre-conv
        if state.decoder_pre is not None:
            bufs["decoder_pre"] = state.decoder_pre.clone()
            state.decoder_pre = bufs["decoder_pre"]

        # Decoder blocks
        for bi, bs in enumerate(state.decoder_blocks):
            if bs.trans_prev is not None:
                bufs[f"block_{bi}_trans"] = bs.trans_prev.clone()
                bs.trans_prev = bufs[f"block_{bi}_trans"]
            for ri, rs in enumerate(bs.residuals):
                if rs.conv1 is not None:
                    bufs[f"block_{bi}_res_{ri}_conv1"] = rs.conv1.clone()
                    rs.conv1 = bufs[f"block_{bi}_res_{ri}_conv1"]
                if rs.conv2 is not None:
                    bufs[f"block_{bi}_res_{ri}_conv2"] = rs.conv2.clone()
                    rs.conv2 = bufs[f"block_{bi}_res_{ri}_conv2"]

        # Decoder post-conv
        if state.decoder_post is not None:
            bufs["decoder_post"] = state.decoder_post.clone()
            state.decoder_post = bufs["decoder_post"]

        return bufs

    def _sync_state_from_bufs(self, state):
        """
        After graph replay, the static buffers contain updated state.
        Ensure state references still point to the same buffers.
        (This is a no-op if references weren't changed, but guards against issues.)
        """
        for idx in range(len(state.upsample_trans_prev)):
            k = f"upsample_trans_{idx}"
            if k in self.state_bufs:
                state.upsample_trans_prev[idx] = self.state_bufs[k]
            k = f"upsample_dw_{idx}"
            if k in self.state_bufs:
                state.upsample_dwconv[idx] = self.state_bufs[k]

        if "decoder_pre" in self.state_bufs:
            state.decoder_pre = self.state_bufs["decoder_pre"]

        for bi, bs in enumerate(state.decoder_blocks):
            k = f"block_{bi}_trans"
            if k in self.state_bufs:
                bs.trans_prev = self.state_bufs[k]
            for ri, rs in enumerate(bs.residuals):
                k1 = f"block_{bi}_res_{ri}_conv1"
                k2 = f"block_{bi}_res_{ri}_conv2"
                if k1 in self.state_bufs:
                    rs.conv1 = self.state_bufs[k1]
                if k2 in self.state_bufs:
                    rs.conv2 = self.state_bufs[k2]

        if "decoder_post" in self.state_bufs:
            state.decoder_post = self.state_bufs["decoder_post"]

    def capture(self, hidden: torch.Tensor, state) -> torch.Tensor:
        """
        Capture the conv path as a CUDA Graph on a dedicated stream.

        Uses stream isolation to prevent capturing operations from
        concurrent codegen threads running on the default stream.

        Includes mandatory warmup to initialize cuDNN/cuBLAS workspace
        allocations which are not capturable by CUDA Graphs.

        Args:
            hidden: Input tensor (1, C, T) for the conv path.
            state: DecoderState with initialized state buffers.

        Returns:
            Output tensor from the capture run, or None on failure.
        """
        self._expected_shape = tuple(hidden.shape)

        # Freeze states: replace with clones that will be static buffers
        self.state_bufs = self._collect_and_freeze_states(state)
        logger.info(
            f"[DecoderGraph] Froze {len(self.state_bufs)} state buffers. "
            f"Shapes: { {k: tuple(v.shape) for k, v in self.state_bufs.items()} }"
        )

        # Create static input
        self.static_hidden = hidden.clone()

        t0 = time.perf_counter()

        # Save state buffer values (needed to restore after warmup + capture)
        saved_states = {k: v.clone() for k, v in self.state_bufs.items()}

        # ── Stream isolation ──
        # Synchronize default stream first, then do all work on dedicated
        # stream so concurrent codegen ops are NOT captured into our graph.
        torch.cuda.current_stream().synchronize()

        with torch.cuda.stream(self._stream):
            # ── WARMUP (mandatory for cuDNN/cuBLAS workspace allocation) ──
            with torch.no_grad():
                for _ in range(3):
                    for k, sv in saved_states.items():
                        self.state_bufs[k].copy_(sv)
                    self.static_hidden.copy_(hidden)
                    _graph_decode_conv_path(
                        self.decoder_module, self.static_hidden, self.state_bufs
                    )
            self._stream.synchronize()

            # ── CAPTURE ──
            for k, sv in saved_states.items():
                self.state_bufs[k].copy_(sv)
            self.static_hidden.copy_(hidden)

            graph = torch.cuda.CUDAGraph()
            try:
                with torch.no_grad():
                    with torch.cuda.graph(graph):
                        out = _graph_decode_conv_path(
                            self.decoder_module, self.static_hidden, self.state_bufs
                        )
            except Exception as e:
                logger.warning(f"[DecoderGraph] Capture failed: {e}")
                self.stats.fallback_count += 1
                self.stats.fallback_reasons.append(f"capture_failed: {e}")
                for k, sv in saved_states.items():
                    self.state_bufs[k].copy_(sv)
                self._sync_state_from_bufs(state)
                return None

            self.graph = graph
            self.static_out = out
            self.captured = True

            # ── FIRST REPLAY for correct output ──
            for k, sv in saved_states.items():
                self.state_bufs[k].copy_(sv)
            self.static_hidden.copy_(hidden)
            self.graph.replay()

        # Wait for dedicated stream to finish
        self._stream.synchronize()

        t1 = time.perf_counter()
        self.stats.capture_time_ms = (t1 - t0) * 1000.0

        logger.info(
            f"[DecoderGraph] Captured + first replay on dedicated stream. "
            f"Input={tuple(hidden.shape)} Output={tuple(self.static_out.shape)} "
            f"capture_time={self.stats.capture_time_ms:.1f}ms "
            f"state_bufs={len(self.state_bufs)}"
        )

        self._sync_state_from_bufs(state)
        return self.static_out

    def replay(self, hidden: torch.Tensor, state) -> Optional[torch.Tensor]:
        """
        Replay the captured graph on the dedicated stream.

        Args:
            hidden: Input tensor (must match captured shape).
            state: DecoderState (state buffers are updated in-place by graph).

        Returns:
            Output tensor, or None on fallback.
        """
        if not self.captured:
            self.stats.fallback_count += 1
            self.stats.fallback_reasons.append("not_captured")
            return None

        if tuple(hidden.shape) != self._expected_shape:
            self.stats.fallback_count += 1
            reason = f"shape_mismatch:{tuple(hidden.shape)}vs{self._expected_shape}"
            self.stats.fallback_reasons.append(reason)
            return None

        # Wait for any pending default-stream work before we touch buffers
        self._stream.wait_stream(torch.cuda.current_stream())

        # Copy input to static buffer and replay on our stream
        self.static_hidden.copy_(hidden)
        with torch.cuda.stream(self._stream):
            self.graph.replay()

        # Block default stream until replay finishes so caller sees results
        torch.cuda.current_stream().wait_stream(self._stream)

        self._sync_state_from_bufs(state)
        self.stats.graph_replays += 1
        return self.static_out

    def _bind_state_to_bufs(self, state):
        """
        Copy real request state values into pre-captured static buffers
        and replace state references with those buffers.

        Used at request time (step 2) after eager step 1 has initialized
        the real state. After this call, the state object's conv buffers
        point to the graph's static buffers, so graph replay will update
        them in-place correctly.
        """
        for idx in range(len(state.upsample_trans_prev)):
            k = f"upsample_trans_{idx}"
            if k in self.state_bufs and state.upsample_trans_prev[idx] is not None:
                self.state_bufs[k].copy_(state.upsample_trans_prev[idx])
                state.upsample_trans_prev[idx] = self.state_bufs[k]
            k = f"upsample_dw_{idx}"
            if k in self.state_bufs and state.upsample_dwconv[idx] is not None:
                self.state_bufs[k].copy_(state.upsample_dwconv[idx])
                state.upsample_dwconv[idx] = self.state_bufs[k]

        if "decoder_pre" in self.state_bufs and state.decoder_pre is not None:
            self.state_bufs["decoder_pre"].copy_(state.decoder_pre)
            state.decoder_pre = self.state_bufs["decoder_pre"]

        for bi, bs in enumerate(state.decoder_blocks):
            k = f"block_{bi}_trans"
            if k in self.state_bufs and bs.trans_prev is not None:
                self.state_bufs[k].copy_(bs.trans_prev)
                bs.trans_prev = self.state_bufs[k]
            for ri, rs in enumerate(bs.residuals):
                k1 = f"block_{bi}_res_{ri}_conv1"
                k2 = f"block_{bi}_res_{ri}_conv2"
                if k1 in self.state_bufs and rs.conv1 is not None:
                    self.state_bufs[k1].copy_(rs.conv1)
                    rs.conv1 = self.state_bufs[k1]
                if k2 in self.state_bufs and rs.conv2 is not None:
                    self.state_bufs[k2].copy_(rs.conv2)
                    rs.conv2 = self.state_bufs[k2]

        if "decoder_post" in self.state_bufs and state.decoder_post is not None:
            self.state_bufs["decoder_post"].copy_(state.decoder_post)
            state.decoder_post = self.state_bufs["decoder_post"]

    def bind_and_replay(self, hidden: torch.Tensor, state) -> Optional[torch.Tensor]:
        """
        Bind real request state to pre-captured graph buffers, then replay.

        Used at request time step 2 (first step after eager warmup).
        Copies real state values into graph's static buffers, replaces
        state references, then replays the graph.

        Args:
            hidden: Input tensor (must match captured shape).
            state: DecoderState with real values from eager step 1.

        Returns:
            Output tensor, or None on fallback.
        """
        if not self.captured:
            self.stats.fallback_count += 1
            self.stats.fallback_reasons.append("not_captured")
            return None

        if tuple(hidden.shape) != self._expected_shape:
            self.stats.fallback_count += 1
            reason = f"shape_mismatch:{tuple(hidden.shape)}vs{self._expected_shape}"
            self.stats.fallback_reasons.append(reason)
            return None

        # Copy real state into pre-captured static buffers
        self._bind_state_to_bufs(state)

        # Now replay (copies hidden, replays graph, syncs state refs)
        return self.replay(hidden, state)

    def pre_capture(self, inc_decoder, packet_tokens: int = 2):
        """
        Pre-capture the decoder CUDA Graph during server initialization.

        Must be called BEFORE any concurrent GPU work (codegen threads)
        starts. This avoids the process-wide CUDA capture lock that would
        kill concurrent operations.

        Runs two synthetic decode steps:
          Step 1: eager (initializes conv state buffers)
          Step 2: capture (captures the graph)

        Args:
            inc_decoder: IncrementalDecoder instance.
            packet_tokens: Number of tokens per packet (default 2).
        """
        from tts_incremental_decoder import _stream_causal_conv

        device = inc_decoder.device
        state = inc_decoder.reset_state()

        # Determine number of codebooks from quantizer
        try:
            n_q = inc_decoder.decoder.quantizer.max_n_q
        except AttributeError:
            n_q = 16  # default for Qwen3-TTS

        logger.info(
            f"[DecoderGraph] Pre-capture: packet_tokens={packet_tokens}, "
            f"n_q={n_q}, device={device}"
        )

        # --- Helper: run non-conv pipeline to get hidden for conv path ---
        def _run_pre_conv_pipeline(codes_input, current_state):
            """Run quantizer -> pre_conv -> pre_transformer -> permute."""
            codes_t = codes_input.transpose(1, 2)  # (1, Q, T)
            h = inc_decoder.decoder.quantizer.decode(codes_t)
            h, current_state.pre_conv = _stream_causal_conv(
                inc_decoder.decoder.pre_conv, h, current_state.pre_conv
            )
            h = h.transpose(1, 2)  # (1, T, C)

            t_new = h.shape[1]
            if inc_decoder.transformer_mode == "cache":
                cache_position = torch.arange(
                    current_state.pos, current_state.pos + t_new, device=device
                )
                out = inc_decoder.decoder.pre_transformer(
                    inputs_embeds=h,
                    use_cache=True,
                    past_key_values=current_state.kv_cache,
                    cache_position=cache_position,
                )
                current_state.kv_cache = out.past_key_values
                current_state.pos += t_new
                h = out.last_hidden_state
            else:
                if current_state.transformer_context is None:
                    context = h
                else:
                    context = torch.cat([current_state.transformer_context, h], dim=1)
                start_pos = max(0, current_state.pos - (context.shape[1] - t_new))
                position_ids = torch.arange(
                    start_pos, start_pos + context.shape[1], device=device
                ).unsqueeze(0)
                out = inc_decoder.decoder.pre_transformer(
                    inputs_embeds=context, use_cache=False, position_ids=position_ids,
                )
                h = out.last_hidden_state[:, -t_new:, :]
                current_state.pos += t_new
                current_state.transformer_context = context

            h = h.permute(0, 2, 1).contiguous()  # (1, C, T)
            return h

        dummy_codes = torch.ones(1, packet_tokens, n_q, dtype=torch.long, device=device)

        with torch.no_grad():
            # Step 1: eager conv path (initializes state buffers)
            hidden1 = _run_pre_conv_pipeline(dummy_codes, state)
            inc_decoder._decode_conv_path(hidden1, state)
            logger.info(
                f"[DecoderGraph] Pre-capture step 1 done. "
                f"hidden_shape={tuple(hidden1.shape)}, "
                f"state_bufs_initialized=True"
            )

            # Step 2: capture conv path graph
            hidden2 = _run_pre_conv_pipeline(dummy_codes, state)

        result = self.capture(hidden2, state)
        if result is not None:
            self._pre_captured = True
            logger.info(
                f"[DecoderGraph] Pre-capture SUCCESS. "
                f"Graph ready for replay. "
                f"capture_time={self.stats.capture_time_ms:.1f}ms"
            )
        else:
            logger.warning("[DecoderGraph] Pre-capture FAILED, will fall back to eager")

    def get_stats_dict(self) -> dict:
        """Return stats as a JSON-serializable dict."""
        reasons_counter = Counter(self.stats.fallback_reasons)
        return {
            "decoder_cudagraph_used": self.captured,
            "decoder_graph_replays": self.stats.graph_replays,
            "decoder_eager_steps": self.stats.eager_steps,
            "decoder_fallback_count": self.stats.fallback_count,
            "decoder_fallback_reasons_topk": dict(reasons_counter.most_common(5)),
            "decoder_capture_time_ms": round(self.stats.capture_time_ms, 1),
            "decoder_state_bufs_count": len(self.state_bufs),
            "decoder_pre_captured": self._pre_captured,
        }


_cached_decoder_graph_accel: Optional[DecoderConvGraphAccelerator] = None
_cached_decoder_module_id: Optional[int] = None


def install_decoder_cudagraph(inc_decoder, packet_tokens: int = 2) -> Optional[DecoderConvGraphAccelerator]:
    """
    Install decoder CUDA Graph acceleration into an IncrementalDecoder.

    Monkey-patches decode_incremental to use graph for the conv path.
    Pre-captures the graph on first call; subsequent calls for the same
    decoder model reuse the cached graph (avoiding repeated 500ms capture).

    Args:
        inc_decoder: IncrementalDecoder instance.
        packet_tokens: Number of tokens per packet (default 2).

    Returns:
        DecoderConvGraphAccelerator instance, or None if disabled.
    """
    global _cached_decoder_graph_accel, _cached_decoder_module_id

    enabled = os.environ.get("TTS_DECODER_CUDAGRAPH", "0").lower() in ("1", "true", "yes")
    if not enabled:
        return None

    if not torch.cuda.is_available():
        logger.warning("[DecoderGraph] CUDA not available, skipping")
        return None

    # Reuse cached accelerator if same decoder module (model weights are shared)
    if (_cached_decoder_graph_accel is not None
            and _cached_decoder_graph_accel._pre_captured
            and _cached_decoder_module_id == id(inc_decoder.decoder)):
        accel = _cached_decoder_graph_accel
        # Reset per-request stats
        accel.stats = DecoderGraphStats()
        accel.stats.capture_time_ms = 0.0  # not re-captured
        inc_decoder._decoder_graph_accel = accel
        inc_decoder._decoder_graph_step_count = 0
        logger.debug("[DecoderGraph] Reusing cached graph accelerator")
    else:
        accel = DecoderConvGraphAccelerator(inc_decoder.decoder)
        inc_decoder._decoder_graph_accel = accel
        inc_decoder._decoder_graph_step_count = 0

        # Pre-capture graph (BEFORE concurrent codegen starts)
        try:
            accel.pre_capture(inc_decoder, packet_tokens=packet_tokens)
            _cached_decoder_graph_accel = accel
            _cached_decoder_module_id = id(inc_decoder.decoder)
        except Exception as e:
            logger.warning(f"[DecoderGraph] Pre-capture failed: {e}")
            import traceback
            traceback.print_exc()

    # Save original method
    orig_decode = inc_decoder.decode_incremental

    def _graphed_decode_incremental(audio_codes, state, pre_conv_hook=None, post_conv_hook=None):
        """Monkey-patched decode_incremental with CUDA Graph for conv path."""
        if audio_codes is None:
            import numpy as np
            return np.zeros((0,), dtype=np.float32), state

        import numpy as np

        codes = audio_codes
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        if codes.numel() == 0:
            return np.zeros((0,), dtype=np.float32), state
        if codes.dim() == 3 and codes.shape[0] == 1:
            codes = codes.squeeze(0)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.dim() != 3 or codes.shape[0] != 1:
            raise ValueError("decode_incremental expects shape (T, Q) or (1, T, Q)")

        state.codebook_dim = int(codes.shape[-1])
        valid_frames = int((codes[:, :, 0] > 0).sum().item())
        state.expected_samples += valid_frames * inc_decoder.decode_upsample_rate
        codes = codes.to(inc_decoder.device).to(torch.long)
        codes = codes.transpose(1, 2)  # (1, Q, T)

        # Use no_grad instead of inference_mode for CUDA Graph compatibility
        with torch.no_grad():
            hidden = inc_decoder.decoder.quantizer.decode(codes)
            from tts_incremental_decoder import _stream_causal_conv
            hidden, state.pre_conv = _stream_causal_conv(
                inc_decoder.decoder.pre_conv, hidden, state.pre_conv
            )
            hidden = hidden.transpose(1, 2)  # (1, T, C)

            t_new = hidden.shape[1]
            if inc_decoder.transformer_mode == "cache":
                cache_position = torch.arange(
                    state.pos, state.pos + t_new, device=hidden.device
                )
                out = inc_decoder.decoder.pre_transformer(
                    inputs_embeds=hidden,
                    use_cache=True,
                    past_key_values=state.kv_cache,
                    cache_position=cache_position,
                )
                state.kv_cache = out.past_key_values
                state.pos += t_new
                hidden = out.last_hidden_state
            else:
                # Window mode - same as original
                if state.transformer_context is None:
                    context = hidden
                else:
                    context = torch.cat([state.transformer_context, hidden], dim=1)
                if inc_decoder.transformer_mode == "window" and context.shape[1] > inc_decoder.window_size:
                    context = context[:, -inc_decoder.window_size :]
                start_pos = max(0, state.pos - (context.shape[1] - t_new))
                position_ids = torch.arange(
                    start_pos, start_pos + context.shape[1], device=hidden.device
                ).unsqueeze(0)
                out = inc_decoder.decoder.pre_transformer(
                    inputs_embeds=context, use_cache=False, position_ids=position_ids,
                )
                hidden = out.last_hidden_state[:, -t_new:, :]
                state.pos += t_new
                state.transformer_context = context

            if pre_conv_hook is not None:
                pre_conv_hook()

            hidden = hidden.permute(0, 2, 1).contiguous()

            inc_decoder._decoder_graph_step_count += 1

            if inc_decoder._decoder_graph_step_count == 1:
                # Step 1: always eager (initializes conv state buffers)
                hidden = inc_decoder._decode_conv_path(hidden, state)
                accel.stats.eager_steps += 1
            elif accel._pre_captured:
                # Pre-captured mode (server): bind state + replay from step 2
                if inc_decoder._decoder_graph_step_count == 2:
                    result = accel.bind_and_replay(hidden, state)
                else:
                    result = accel.replay(hidden, state)
                if result is None:
                    hidden = inc_decoder._decode_conv_path(hidden, state)
                    accel.stats.eager_steps += 1
                else:
                    hidden = result
            else:
                # Lazy capture mode (microbench): capture on step 2
                if inc_decoder._decoder_graph_step_count == 2:
                    result = accel.capture(hidden, state)
                    if result is None:
                        hidden = inc_decoder._decode_conv_path(hidden, state)
                        accel.stats.eager_steps += 1
                    else:
                        hidden = result
                else:
                    result = accel.replay(hidden, state)
                    if result is None:
                        hidden = inc_decoder._decode_conv_path(hidden, state)
                        accel.stats.eager_steps += 1
                    else:
                        hidden = result

            if post_conv_hook is not None:
                post_conv_hook()

        audio = hidden.squeeze(0).squeeze(0).detach().cpu().numpy()
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)
        state.emitted_samples += len(audio)
        return audio, state

    inc_decoder.decode_incremental = _graphed_decode_incremental
    logger.info("[DecoderGraph] Installed decoder CUDA Graph accelerator")
    return accel

