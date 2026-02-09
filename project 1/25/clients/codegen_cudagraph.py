#!/usr/bin/env python3
"""
CUDA Graph acceleration for Qwen3-TTS codegen.

Flag-controlled, default off, one-click reversible.

Flags:
  TTS_CODEGEN_CUDAGRAPH_TALKER=0|1   (Talker backbone decode)
  TTS_CODEGEN_CUDAGRAPH_CP=0|1       (Code Predictor decode)

Design:
  - Only decode steps (q_len=1) use CUDA Graphs; prefill stays eager.
  - graph capture under torch.no_grad() (NOT inference_mode).
  - Static buffers + copy_() for replay.
  - Scatter-based KV cache (GraphFriendlyCache) for position-agnostic replay.
  - Automatic fallback to eager on shape mismatch / capture failure.
"""

import os
import types
import logging
import time

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Any

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

log = logging.getLogger("codegen_cudagraph")


# ======================================================================
# Stats
# ======================================================================
@dataclass
class CUDAGraphStats:
    talker_graph_used: int = 0
    cp_graph_used: int = 0
    talker_eager_fallback: int = 0
    cp_eager_fallback: int = 0
    prefill_eager_steps: int = 0
    fallback_count: int = 0
    fallback_reasons: list = field(default_factory=list)

    def to_dict(self):
        reasons = {}
        for r in self.fallback_reasons[-200:]:
            reasons[r] = reasons.get(r, 0) + 1
        topk = sorted(reasons.items(), key=lambda x: -x[1])[:5]
        return {
            "cudagraph_talker_used": self.talker_graph_used,
            "cudagraph_cp_used": self.cp_graph_used,
            "prefill_eager_steps": self.prefill_eager_steps,
            "fallback_count": self.fallback_count,
            "fallback_reasons_topk": dict(topk),
        }

    def reset_per_request(self):
        self.talker_graph_used = 0
        self.cp_graph_used = 0
        self.talker_eager_fallback = 0
        self.cp_eager_fallback = 0
        self.prefill_eager_steps = 0
        self.fallback_count = 0
        self.fallback_reasons.clear()


_stats = CUDAGraphStats()


# ======================================================================
# GraphFriendlyCache
# ======================================================================
class GraphFriendlyCache(DynamicCache):
    """
    DynamicCache subclass with pre-allocated static KV buffers and
    scatter-based update() for CUDA Graph compatibility.

    Key properties:
    - layers[i].keys / .values are always (B, num_kv_heads, max_seq_len, head_dim)
    - update() uses scatter_(2, idx, val) so the write position comes from
      a *tensor* (cache_position), not a Python int — graph-replayable.
    - get_seq_length() returns a manually-tracked counter (_seen_tokens).
    """

    def __init__(self, num_layers: int, max_seq_len: int,
                 num_kv_heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.max_seq_len = max_seq_len
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # Pre-allocate full-size buffers and register them via DynamicCache API
        for i in range(num_layers):
            k = torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            v = torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                            device=device, dtype=dtype)
            # DynamicCache.update creates internal layer entries
            super().update(k[:, :, :1, :], v[:, :, :1, :], i)
            # Overwrite with full-size buffer
            self.layers[i].keys = k
            self.layers[i].values = v

        self._seen_tokens = 0

    # -- core ----------------------------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Scatter-based update: write position from cache_position tensor."""
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is not None:
            q_len = key_states.shape[2]
            idx = cache_position.view(1, 1, q_len, 1).expand_as(key_states)
            self.layers[layer_idx].keys.scatter_(2, idx, key_states)
            self.layers[layer_idx].values.scatter_(2, idx, value_states)
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        # fallback: in-place copy
        pos = self._seen_tokens
        sl = key_states.shape[2]
        self.layers[layer_idx].keys[:, :, pos:pos + sl, :].copy_(key_states)
        self.layers[layer_idx].values[:, :, pos:pos + sl, :].copy_(value_states)
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    # -- helpers -------------------------------------------------------
    def copy_from_dynamic(self, dyn_cache: DynamicCache, cur_len: int | None = None):
        if cur_len is None:
            cur_len = dyn_cache.get_seq_length()
        for i in range(self._num_layers):
            k, v = dyn_cache[i]
            self.layers[i].keys[:, :, :cur_len, :].copy_(k[:, :, :cur_len, :])
            self.layers[i].values[:, :, :cur_len, :].copy_(v[:, :, :cur_len, :])
        self._seen_tokens = cur_len

    def zero_buffers(self):
        for i in range(self._num_layers):
            self.layers[i].keys.zero_()
            self.layers[i].values.zero_()
        self._seen_tokens = 0


# ======================================================================
# Frozen-cache helpers (from P2, bit-exact proven)
# ======================================================================
import types as _types


def _make_frozen_cache(src_cache, num_layers, max_seq_len,
                       num_kv_heads, head_dim, device, dtype,
                       shared_bufs=None):
    """
    Build a DynamicCache with pre-allocated static buffers and a
    monkey-patched update() that does in-place copy (not cat).
    If *shared_bufs* is given, re-use the same (key, value) buffer list.
    """
    fc = DynamicCache()
    cur_len = src_cache.get_seq_length() if src_cache is not None else 0

    pos_holder = [cur_len]

    for i in range(num_layers):
        if shared_bufs is not None:
            k_buf, v_buf = shared_bufs[i]
        else:
            k_buf = torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                                device=device, dtype=dtype)
            v_buf = torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                                device=device, dtype=dtype)
        if src_cache is not None and cur_len > 0:
            sk, sv = src_cache[i]
            k_buf[:, :, :cur_len, :].copy_(sk[:, :, :cur_len, :])
            v_buf[:, :, :cur_len, :].copy_(sv[:, :, :cur_len, :])
        fc.update(k_buf[:, :, :1, :], v_buf[:, :, :1, :], i)
        fc.layers[i].keys = k_buf
        fc.layers[i].values = v_buf

        # CRITICAL FIX: Override layer.get_seq_length() so that
        # create_causal_mask / get_mask_sizes sees the actual sequence
        # length, NOT the full-buffer size (MAX_SEQ_LEN).
        layer = fc.layers[i]
        def _layer_gsl(self_layer, _ph=pos_holder):
            return _ph[0]
        layer.get_seq_length = _types.MethodType(_layer_gsl, layer)

    def _upd(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = pos_holder[0]
        sl = key_states.shape[2]
        self.layers[layer_idx].keys[:, :, pos:pos + sl, :].copy_(key_states)
        self.layers[layer_idx].values[:, :, pos:pos + sl, :].copy_(value_states)
        return (self.layers[layer_idx].keys[:, :, :pos + sl, :],
                self.layers[layer_idx].values[:, :, :pos + sl, :])

    def _gsl(self, layer_idx=0):
        return pos_holder[0]

    fc.update = _types.MethodType(_upd, fc)
    fc.get_seq_length = _types.MethodType(_gsl, fc)
    fc._seen_tokens = cur_len
    return fc, pos_holder


def _reset_frozen_cache(fc, pos_holder, reset_to):
    pos_holder[0] = reset_to
    fc._seen_tokens = reset_to


# ======================================================================
# CP (Code Predictor) Graph Accelerator
# ======================================================================
class CPGraphAccelerator:
    """
    Replaces code_predictor.generate() with a CUDA-Graph-based loop.

    Uses the P2-proven frozen-cache approach:
    - Shared KV buffers across all 14 step graphs
    - Each graph uses a view-returning update() (pos-based slicing)
      so tensor shapes match eager mode exactly → bit-exact.
    """

    def __init__(self, code_predictor, talker_config, device, dtype):
        self.cp = code_predictor
        self.cp_config = code_predictor.config
        self.talker_config = talker_config
        self.device = device
        self.dtype = dtype

        self.num_layers = self.cp_config.num_hidden_layers
        self.num_kv_heads = self.cp_config.num_key_value_heads
        self.head_dim = self.cp_config.head_dim
        self.hidden_size = self.cp_config.hidden_size
        self.n_decode_steps = self.cp_config.num_code_groups - 2  # 14
        self.max_cp_len = self.n_decode_steps + 4  # 18

        # Shared KV buffers (one set for all 14 graphs)
        self.shared_bufs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.num_layers):
            k = torch.zeros(1, self.num_kv_heads, self.max_cp_len, self.head_dim,
                            device=device, dtype=dtype)
            v = torch.zeros(1, self.num_kv_heads, self.max_cp_len, self.head_dim,
                            device=device, dtype=dtype)
            self.shared_bufs.append((k, v))

        # Static input buffers (shared across ALL graphs)
        self.s_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
        self.s_cp = torch.tensor([0], device=device, dtype=torch.long)

        # ONE frozen cache shared by all graphs (critical for multi-step KV flow)
        self.fc: Any = None
        self.pos_holder: list[int] | None = None

        # Per-step graphs and output refs
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: dict[int, Any] = {}
        self._captured = False

    def _capture_all_graphs(self):
        """
        Capture one CUDA Graph per decode step (gs=1..n_decode_steps).

        CRITICAL: all graphs share ONE frozen cache (fc) and ONE
        cache_position tensor (s_cp) so that KV written by graph N
        is visible to graph N+1 during sequential replay.
        """
        log.info("CPGraph: capturing %d graphs (ONE shared frozen-cache) ...",
                 self.n_decode_steps)
        t0 = time.time()

        # Build ONE frozen cache backed by shared_bufs
        self.fc, self.pos_holder = _make_frozen_cache(
            src_cache=None, num_layers=self.num_layers,
            max_seq_len=self.max_cp_len,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            device=self.device, dtype=self.dtype,
            shared_bufs=self.shared_bufs,
        )

        for gs in range(1, self.n_decode_steps + 1):
            cp_len = gs + 1  # cache already has positions 0..(gs)

            # Fill shared bufs with dummy KV
            for i in range(self.num_layers):
                kb, vb = self.shared_bufs[i]
                kb.zero_()
                vb.zero_()
                kb[:, :, :cp_len, :].normal_(0, 0.01)
                vb[:, :, :cp_len, :].normal_(0, 0.01)

            # Set position state for this step
            self.pos_holder[0] = cp_len
            self.fc._seen_tokens = cp_len
            self.s_cp.fill_(cp_len)

            # Warmup
            with torch.no_grad():
                self.cp(input_ids=self.s_ids, past_key_values=self.fc,
                        cache_position=self.s_cp, use_cache=True,
                        generation_steps=gs)
            torch.cuda.synchronize()

            # Restore dummy KV and position state
            for i in range(self.num_layers):
                kb, vb = self.shared_bufs[i]
                kb[:, :, :cp_len, :].normal_(0, 0.01)
                vb[:, :, :cp_len, :].normal_(0, 0.01)
            self.pos_holder[0] = cp_len
            self.fc._seen_tokens = cp_len
            self.s_cp.fill_(cp_len)

            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                g_out = self.cp(input_ids=self.s_ids, past_key_values=self.fc,
                                cache_position=self.s_cp, use_cache=True,
                                generation_steps=gs)

            self.graphs[gs] = g
            self.graph_outputs[gs] = g_out

        self._captured = True
        log.info("CPGraph: %d graphs captured in %.1fs",
                 len(self.graphs), time.time() - t0)

    # ------------------------------------------------------------------
    def predict_codes(self, inputs_embeds, do_sample, top_p, top_k,
                      temperature, generator=None, **_kw):
        """
        Drop-in replacement for code_predictor.generate().
        Returns object with .sequences (1, n_decode_steps).
        """
        if not self._captured:
            self._capture_all_graphs()

        # ── 1. Prefill (eager, fresh DynamicCache) ───────────────────
        pf_cache = DynamicCache()
        pf_len = inputs_embeds.shape[1]
        with torch.no_grad():
            pf_out = self.cp(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=pf_cache,
                cache_position=torch.arange(pf_len, device=self.device),
                position_ids=torch.arange(pf_len, device=self.device).unsqueeze(0),
                use_cache=True,
                generation_steps=0,
            )

        # ── 2. Copy prefill KV → shared buffers ─────────────────────
        for i in range(self.num_layers):
            k, v = pf_cache[i]
            kb, vb = self.shared_bufs[i]
            kb[:, :, :pf_len, :].copy_(k[:, :, :pf_len, :])
            vb[:, :, :pf_len, :].copy_(v[:, :, :pf_len, :])

        # Sample first token
        pf_logits = pf_out.logits
        next_tok = self._sample(pf_logits[:, -1, :],
                                do_sample, top_k, top_p, temperature, generator)

        # ── 3. Decode steps with graph replay (shared fc) ────────────
        all_tokens = [next_tok]

        for gs in range(1, self.n_decode_steps + 1):
            cp_len = gs + 1  # positions already in cache

            # Set cache state for this step
            self.pos_holder[0] = cp_len
            self.fc._seen_tokens = cp_len
            self.s_cp.fill_(cp_len)

            # Copy input ids
            self.s_ids.copy_(next_tok.view(1, 1))

            # Replay
            self.graphs[gs].replay()
            torch.cuda.synchronize()

            # Sample
            logits = self.graph_outputs[gs].logits
            next_tok = self._sample(logits[:, -1, :],
                                    do_sample, top_k, top_p, temperature, generator)
            all_tokens.append(next_tok)

        _stats.cp_graph_used += 1

        # Build (1, 14) sequences
        sequences = torch.cat(
            [t.view(1, 1) if t.dim() == 1 else t for t in all_tokens],
            dim=-1,
        )

        class _R:
            pass
        r = _R()
        r.sequences = sequences
        r.hidden_states = None
        return r

    # ------------------------------------------------------------------
    @staticmethod
    def _sample(logits, do_sample, top_k, top_p, temperature, generator):
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        if temperature is not None and temperature > 0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            tk = min(top_k, logits.size(-1))
            topk_vals, topk_idx = torch.topk(logits, k=tk, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            return topk_idx.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        if top_p is not None and top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            probs = torch.softmax(sl, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sl = sl.masked_fill(mask, -float("inf"))
            probs = torch.softmax(sl, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            return si.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# ======================================================================
# Talker Graph Accelerator
# ======================================================================
class TalkerGraphAccelerator:
    """
    CUDA Graph for Talker backbone (Qwen3TTSTalkerModel) decode steps.

    Uses a GraphFriendlyCache with scatter-based update so that one
    captured graph works for *any* cache position (the position comes
    from the cache_position tensor, not Python ints).

    A 4D attention mask (static buffer, updated before replay) bypasses
    create_causal_mask inside the model.
    """

    MAX_SEQ_LEN = 512

    def __init__(self, talker_model, config, device, dtype):
        self.model = talker_model
        self.config = config
        self.device = device
        self.dtype = dtype
        self.original_forward = talker_model.forward

        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Static cache
        self.static_cache = GraphFriendlyCache(
            self.num_layers, self.MAX_SEQ_LEN,
            self.num_kv_heads, self.head_dim,
            device, dtype,
        )

        # Static input buffers
        self.s_embeds = torch.zeros(1, 1, self.hidden_size, device=device, dtype=dtype)
        self.s_cache_pos = torch.zeros(1, device=device, dtype=torch.long)
        self.s_pos_ids = torch.zeros(3, 1, 1, device=device, dtype=torch.long)
        self.s_4d_mask = torch.full(
            (1, 1, 1, self.MAX_SEQ_LEN), float("-inf"),
            device=device, dtype=dtype,
        )

        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_output: BaseModelOutputWithPast | None = None
        self._captured = False

    # ------------------------------------------------------------------
    def _capture_graph(self):
        log.info("TalkerGraph: capturing (MAX_SEQ_LEN=%d) ...", self.MAX_SEQ_LEN)
        DUMMY = 10

        # fill cache
        self.static_cache.zero_buffers()
        for i in range(self.num_layers):
            self.static_cache.layers[i].keys[:, :, :DUMMY, :].normal_(0, 0.01)
            self.static_cache.layers[i].values[:, :, :DUMMY, :].normal_(0, 0.01)
        self.static_cache._seen_tokens = DUMMY

        self.s_embeds.normal_(0, 0.01)
        self.s_cache_pos.fill_(DUMMY)
        self.s_pos_ids.fill_(DUMMY)
        self.s_4d_mask.fill_(float("-inf"))
        self.s_4d_mask[0, 0, 0, :DUMMY + 1] = 0.0

        # warm-up (output_hidden_states=True because generate() needs it)
        with torch.no_grad():
            self.original_forward(
                input_ids=None,
                attention_mask=self.s_4d_mask,
                position_ids=self.s_pos_ids,
                past_key_values=self.static_cache,
                inputs_embeds=self.s_embeds,
                use_cache=True,
                cache_position=self.s_cache_pos,
                output_attentions=False,
                output_hidden_states=True,
            )
        torch.cuda.synchronize()

        # restore dummy KV
        for i in range(self.num_layers):
            self.static_cache.layers[i].keys[:, :, DUMMY, :].zero_()
            self.static_cache.layers[i].keys[:, :, :DUMMY, :].normal_(0, 0.01)
            self.static_cache.layers[i].values[:, :, DUMMY, :].zero_()
            self.static_cache.layers[i].values[:, :, :DUMMY, :].normal_(0, 0.01)
        self.static_cache._seen_tokens = DUMMY

        self.s_4d_mask.fill_(float("-inf"))
        self.s_4d_mask[0, 0, 0, :DUMMY + 1] = 0.0

        # capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.graph_output = self.original_forward(
                input_ids=None,
                attention_mask=self.s_4d_mask,
                position_ids=self.s_pos_ids,
                past_key_values=self.static_cache,
                inputs_embeds=self.s_embeds,
                use_cache=True,
                cache_position=self.s_cache_pos,
                output_attentions=False,
                output_hidden_states=True,
            )

        self._captured = True
        log.info("TalkerGraph: capture OK")

    # ------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                cache_position=None, **kwargs):
        """Patched inner-model forward with CUDA Graph decode."""

        q_len = inputs_embeds.shape[1] if inputs_embeds is not None else 1

        # ── prefill → always eager ───────────────────────────────────
        if q_len != 1:
            _stats.prefill_eager_steps += 1
            return self.original_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, **kwargs,
            )

        # ── current position ─────────────────────────────────────────
        if cache_position is not None:
            cur_pos = cache_position[0].item()
        elif past_key_values is not None:
            cur_pos = past_key_values.get_seq_length()
        else:
            cur_pos = 0

        # ── check bounds ─────────────────────────────────────────────
        if cur_pos >= self.MAX_SEQ_LEN - 1:
            _stats.fallback_count += 1
            _stats.fallback_reasons.append(f"seq_exceeded:{cur_pos}")
            _stats.talker_eager_fallback += 1
            return self.original_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, **kwargs,
            )

        # ── lazy capture ─────────────────────────────────────────────
        if not self._captured:
            try:
                self._capture_graph()
            except Exception as e:
                log.warning("TalkerGraph: capture failed (%s), falling back", e)
                _stats.fallback_count += 1
                _stats.fallback_reasons.append(f"capture_fail:{e}")
                _stats.talker_eager_fallback += 1
                return self.original_forward(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache,
                    cache_position=cache_position, **kwargs,
                )

        # ── first decode of a request: migrate DynamicCache → static ─
        if not isinstance(past_key_values, GraphFriendlyCache):
            self.static_cache.zero_buffers()
            self.static_cache.copy_from_dynamic(past_key_values, cur_len=cur_pos)
            # reset the 4D mask for this request
            self.s_4d_mask.fill_(float("-inf"))
            self.s_4d_mask[0, 0, 0, :cur_pos] = 0.0

        # ── copy inputs → static buffers ─────────────────────────────
        self.s_embeds.copy_(inputs_embeds)
        self.s_cache_pos.copy_(cache_position)
        if position_ids is not None:
            self.s_pos_ids.copy_(position_ids)

        # unmask current position
        self.s_4d_mask[0, 0, 0, cur_pos] = 0.0

        # ── replay ───────────────────────────────────────────────────
        self.graph.replay()
        torch.cuda.synchronize()

        _stats.talker_graph_used += 1
        self.static_cache._seen_tokens = cur_pos + 1

        return BaseModelOutputWithPast(
            last_hidden_state=self.graph_output.last_hidden_state,
            past_key_values=self.static_cache,
            hidden_states=self.graph_output.hidden_states,
            attentions=self.graph_output.attentions,
        )


# ======================================================================
# Installation
# ======================================================================
def install_cudagraph_accelerator(model, talker_flag: bool, cp_flag: bool):
    """
    Monkey-patch a Qwen3TTSModel to use CUDA Graph for codegen.
    Both flags default to False (off).

    Args:
        model:  Qwen3TTSModel (the wrapper, not the HF model directly)
        talker_flag:  enable graph for talker backbone
        cp_flag:      enable graph for code predictor
    """
    global _stats
    _stats = CUDAGraphStats()

    talker = model.model.talker
    device = next(talker.parameters()).device
    dtype = next(talker.parameters()).dtype

    log.info("install_cudagraph_accelerator: talker=%s  cp=%s  device=%s  dtype=%s",
             talker_flag, cp_flag, device, dtype)

    if cp_flag:
        log.info("── CP CUDA Graph ──")
        cp_accel = CPGraphAccelerator(
            code_predictor=talker.code_predictor,
            talker_config=talker.config,
            device=device, dtype=dtype,
        )
        cp_accel._capture_all_graphs()

        # monkey-patch code_predictor.generate
        _orig_cp_generate = talker.code_predictor.generate

        def _graphed_cp_generate(inputs_embeds=None, max_new_tokens=None,
                                 do_sample=False, top_p=1.0, top_k=50,
                                 temperature=1.0, **kwargs):
            generator = kwargs.get("generator", None)
            try:
                return cp_accel.predict_codes(
                    inputs_embeds=inputs_embeds,
                    do_sample=do_sample, top_p=top_p, top_k=top_k,
                    temperature=temperature, generator=generator,
                )
            except Exception as e:
                log.warning("CPGraph predict_codes failed (%s), eager fallback", e)
                _stats.cp_eager_fallback += 1
                _stats.fallback_count += 1
                _stats.fallback_reasons.append(f"cp_predict_fail:{e}")
                return _orig_cp_generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample, top_p=top_p, top_k=top_k,
                    temperature=temperature, **kwargs,
                )

        talker.code_predictor.generate = _graphed_cp_generate
        log.info("── CP CUDA Graph installed ──")

    if talker_flag:
        log.info("── Talker CUDA Graph ──")
        talker_accel = TalkerGraphAccelerator(
            talker_model=talker.model,
            config=talker.config,
            device=device, dtype=dtype,
        )
        talker_accel._capture_graph()

        # monkey-patch talker.model.forward
        _orig_talker_fwd = talker.model.forward

        def _graphed_talker_fwd(input_ids=None, attention_mask=None,
                                position_ids=None, past_key_values=None,
                                inputs_embeds=None, use_cache=None,
                                cache_position=None, **kwargs):
            try:
                return talker_accel.forward(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache,
                    cache_position=cache_position, **kwargs,
                )
            except Exception as e:
                log.warning("TalkerGraph forward failed (%s), eager fallback", e)
                _stats.talker_eager_fallback += 1
                _stats.fallback_count += 1
                _stats.fallback_reasons.append(f"talker_fwd_fail:{e}")
                return _orig_talker_fwd(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=use_cache,
                    cache_position=cache_position, **kwargs,
                )

        talker.model.forward = _graphed_talker_fwd
        log.info("── Talker CUDA Graph installed ──")

    return _stats


def get_cudagraph_stats() -> dict:
    """Return current CUDA Graph stats as a dict."""
    return _stats.to_dict()


def reset_cudagraph_stats():
    """Reset per-request CUDA Graph stats."""
    _stats.reset_per_request()

