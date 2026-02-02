#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

TTS_DEEP_STREAM_TRACE_POS = os.environ.get("TTS_DEEP_STREAM_TRACE_POS", "0").lower() in ("1", "true", "yes")
TTS_DEEP_STREAM_TRACE_POS_LIMIT = int(os.environ.get("TTS_DEEP_STREAM_TRACE_POS_LIMIT", "8"))
TTS_DEEP_STREAM_DUMMY_DECODER = os.environ.get("TTS_DEEP_STREAM_DUMMY_DECODER", "full").strip().lower()


@dataclass
class ResidualUnitState:
    conv1: Optional[torch.Tensor] = None
    conv2: Optional[torch.Tensor] = None


@dataclass
class DecoderBlockState:
    trans_prev: Optional[torch.Tensor] = None
    residuals: list[ResidualUnitState] = field(default_factory=list)


@dataclass
class DecoderState:
    pos: int = 0
    kv_cache: Optional[Cache] = None
    transformer_context: Optional[torch.Tensor] = None
    pre_conv: Optional[torch.Tensor] = None
    upsample_trans_prev: list[Optional[torch.Tensor]] = field(default_factory=list)
    upsample_dwconv: list[Optional[torch.Tensor]] = field(default_factory=list)
    decoder_pre: Optional[torch.Tensor] = None
    decoder_blocks: list[DecoderBlockState] = field(default_factory=list)
    decoder_post: Optional[torch.Tensor] = None
    expected_samples: int = 0
    emitted_samples: int = 0


def _ensure_buffer(
    buffer: Optional[torch.Tensor], shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if buffer is None or buffer.shape != shape or buffer.device != device or buffer.dtype != dtype:
        return torch.zeros(shape, device=device, dtype=dtype)
    return buffer


def _stream_causal_conv(
    conv_module, x: torch.Tensor, state: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    weight = conv_module.conv.weight
    bias = conv_module.conv.bias
    dilation = conv_module.dilation
    groups = conv_module.conv.groups
    stride = conv_module.stride
    if stride != 1:
        raise ValueError("streaming conv only supports stride=1")
    k = int(conv_module.kernel_size)
    x_cast = x.to(weight.dtype)
    if k <= 1:
        y = F.conv1d(x_cast, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)
        return y, state
    state_len = k - 1
    buf = _ensure_buffer(state, (x_cast.shape[0], x_cast.shape[1], state_len), x_cast.device, x_cast.dtype)
    x_cat = torch.cat([buf, x_cast], dim=-1)
    y = F.conv1d(x_cat, weight, bias=bias, stride=stride, dilation=dilation, groups=groups)
    new_state = x_cat[:, :, -state_len:]
    return y, new_state


def _stream_transpose_conv(
    conv_module, x: torch.Tensor, prev: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    weight = conv_module.conv.weight
    bias = conv_module.conv.bias
    stride = int(conv_module.conv.stride[0])
    kernel = int(conv_module.conv.kernel_size[0])
    padding = kernel - stride
    x = x.to(weight.dtype)
    if prev is not None:
        prev = prev.to(x.device).to(x.dtype)
    if kernel == stride:
        y = F.conv_transpose1d(x, weight, bias=bias, stride=stride, padding=padding)
        return y, prev
    if kernel != 2 * stride:
        raise ValueError(f"unsupported transposed conv kernel={kernel} stride={stride}")
    out_chunks = []
    for idx in range(x.shape[-1]):
        frame = x[..., idx : idx + 1]
        if prev is None:
            prev = frame
            continue
        combo = torch.cat([prev, frame], dim=-1)
        y = F.conv_transpose1d(combo, weight, bias=bias, stride=stride, padding=padding)
        out_chunks.append(y)
        prev = frame
    if out_chunks:
        out = torch.cat(out_chunks, dim=-1)
    else:
        out = x.new_zeros((x.shape[0], weight.shape[1], 0))
    return out, prev


def _stream_convnext(block, x: torch.Tensor, state: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    y, new_state = _stream_causal_conv(block.dwconv, x, state)
    y = y.permute(0, 2, 1)
    y = block.norm(y)
    y = block.pwconv1(y)
    y = block.act(y)
    y = block.pwconv2(y)
    y = block.gamma * y
    y = y.permute(0, 2, 1)
    y = x + y
    return y, new_state


def _stream_residual_unit(unit, x: torch.Tensor, state: ResidualUnitState) -> tuple[torch.Tensor, ResidualUnitState]:
    residual = x
    y = unit.act1(x)
    y, state.conv1 = _stream_causal_conv(unit.conv1, y, state.conv1)
    y = unit.act2(y)
    y, state.conv2 = _stream_causal_conv(unit.conv2, y, state.conv2)
    return y + residual, state


class IncrementalDecoder:
    def __init__(self, tokenizer, device: str = "cuda", transformer_mode: str = "cache"):
        self.tokenizer = tokenizer
        self.model = tokenizer.model
        self.decoder = self.model.decoder
        self.device = torch.device(device)
        self.decode_upsample_rate = int(self.model.get_decode_upsample_rate())
        self.transformer_mode = transformer_mode
        self.window_size = int(getattr(self.decoder.pre_transformer, "window_size", 72))

    def reset_state(self) -> DecoderState:
        state = DecoderState()
        state.upsample_trans_prev = [None for _ in self.decoder.upsample]
        state.upsample_dwconv = [None for _ in self.decoder.upsample]
        state.decoder_blocks = []
        for block in self.decoder.decoder[1:-2]:
            if not hasattr(block, "block"):
                continue
            unit_states = [ResidualUnitState() for _ in block.block[2:]]
            state.decoder_blocks.append(DecoderBlockState(trans_prev=None, residuals=unit_states))
        return state

    def decode_incremental(
        self, audio_codes: torch.Tensor, state: DecoderState
    ) -> tuple[np.ndarray, DecoderState]:
        if audio_codes is None:
            return np.zeros((0,), dtype=np.float32), state
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
        valid_frames = int((codes[:, :, 0] > 0).sum().item())
        state.expected_samples += valid_frames * self.decode_upsample_rate
        codes = codes.to(self.device).to(torch.long)
        codes = codes.transpose(1, 2)  # (1, Q, T)

        with torch.inference_mode():
            hidden = self.decoder.quantizer.decode(codes)
            hidden, state.pre_conv = _stream_causal_conv(self.decoder.pre_conv, hidden, state.pre_conv)
            hidden = hidden.transpose(1, 2)  # (1, T, C)

            t_new = hidden.shape[1]
            if TTS_DEEP_STREAM_DUMMY_DECODER == "conv_only":
                # Skip pre_transformer; still advance pos to keep consistent indexing.
                state.pos += t_new
            else:
                if self.transformer_mode == "cache":
                    cache_position = torch.arange(state.pos, state.pos + t_new, device=hidden.device)
                    if TTS_DEEP_STREAM_TRACE_POS and state.pos <= TTS_DEEP_STREAM_TRACE_POS_LIMIT:
                        try:
                            print(
                                f"[DECODER_POS] mode=cache pos={state.pos} t_new={t_new} "
                                f"cache_min={int(cache_position.min().item())} "
                                f"cache_max={int(cache_position.max().item())} "
                                f"codes_shape={tuple(audio_codes.shape)}"
                            )
                            sys.stdout.flush()
                        except Exception:
                            pass
                    out = self.decoder.pre_transformer(
                        inputs_embeds=hidden,
                        use_cache=True,
                        past_key_values=state.kv_cache,
                        cache_position=cache_position,
                    )
                    state.kv_cache = out.past_key_values
                    state.pos += t_new
                    hidden = out.last_hidden_state
                else:
                    if state.transformer_context is None:
                        context = hidden
                    else:
                        context = torch.cat([state.transformer_context, hidden], dim=1)
                    if self.transformer_mode == "window" and context.shape[1] > self.window_size:
                        context = context[:, -self.window_size :]
                    start_pos = max(0, state.pos - (context.shape[1] - t_new))
                    position_ids = torch.arange(start_pos, start_pos + context.shape[1], device=hidden.device).unsqueeze(0)
                    if TTS_DEEP_STREAM_TRACE_POS and state.pos <= TTS_DEEP_STREAM_TRACE_POS_LIMIT:
                        try:
                            print(
                                f"[DECODER_POS] mode={self.transformer_mode} pos={state.pos} t_new={t_new} "
                                f"pos_min={int(position_ids.min().item())} "
                                f"pos_max={int(position_ids.max().item())} "
                                f"codes_shape={tuple(audio_codes.shape)}"
                            )
                            sys.stdout.flush()
                        except Exception:
                            pass
                    out = self.decoder.pre_transformer(
                        inputs_embeds=context,
                        use_cache=False,
                        position_ids=position_ids,
                    )
                    hidden = out.last_hidden_state[:, -t_new:, :]
                    state.pos += t_new
                    state.transformer_context = context

            if TTS_DEEP_STREAM_DUMMY_DECODER == "pre_transformer":
                return np.zeros((0,), dtype=np.float32), state

            if TTS_DEEP_STREAM_DUMMY_DECODER == "noop":
                # Skip all conv/upsample stages but keep transformer on GPU.
                out_len = int(t_new * self.decode_upsample_rate)
                audio = np.zeros((out_len,), dtype=np.float32)
                state.emitted_samples += out_len
                return audio, state

            hidden = hidden.permute(0, 2, 1)

            for idx, blocks in enumerate(self.decoder.upsample):
                trans = blocks[0]
                hidden, state.upsample_trans_prev[idx] = _stream_transpose_conv(
                    trans, hidden, state.upsample_trans_prev[idx]
                )
                convnext = blocks[1]
                hidden, state.upsample_dwconv[idx] = _stream_convnext(convnext, hidden, state.upsample_dwconv[idx])

            decoder = self.decoder.decoder
            hidden, state.decoder_pre = _stream_causal_conv(decoder[0], hidden, state.decoder_pre)

            block_states = state.decoder_blocks
            block_idx = 0
            for block in decoder[1:-2]:
                if not hasattr(block, "block"):
                    continue
                blk_state = block_states[block_idx]
                block_idx += 1
                # SnakeBeta then transposed conv
                hidden = block.block[0](hidden)
                hidden, blk_state.trans_prev = _stream_transpose_conv(block.block[1], hidden, blk_state.trans_prev)
                for unit_idx, unit in enumerate(block.block[2:]):
                    hidden, blk_state.residuals[unit_idx] = _stream_residual_unit(
                        unit, hidden, blk_state.residuals[unit_idx]
                    )

            hidden = decoder[-2](hidden)
            hidden, state.decoder_post = _stream_causal_conv(decoder[-1], hidden, state.decoder_post)
            hidden = hidden.clamp(min=-1, max=1).to(torch.float32)

        audio = hidden.squeeze(0).squeeze(0).detach().cpu().numpy()
        if audio.ndim > 1:
            audio = audio.flatten()
        audio = audio.astype(np.float32)
        state.emitted_samples += len(audio)
        return audio, state
