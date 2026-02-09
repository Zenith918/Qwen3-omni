#!/usr/bin/env python3
"""
P4.A1: Decoder-only microbench.

Feeds real codes (from code_dump) through IncrementalDecoder's decode path only.
No codegen. Measures RTF_decode_only, launches/frame, avg/percentile latencies.

Usage:
    cd /workspace/project\ 1/25/clients
    PYTHONPATH=/workspace/vllm-omni python3 decoder_microbench.py
"""

import os, sys, time, hashlib, json, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace/vllm-omni")

# ── Configuration ──
MODEL_DIR = os.environ.get(
    "TTS_DEEP_STREAM_MODEL_DIR",
    "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)
PACKET_TOKENS = int(os.environ.get("TTS_DEEP_STREAM_PACKET_TOKENS", "2"))
USE_CUDAGRAPH = os.environ.get("TTS_DECODER_CUDAGRAPH", "0").lower() in ("1", "true", "yes")
WARMUP_RUNS = 3
TIMED_RUNS = 5
DEVICE = "cuda:0"

SAMPLE_RATE = 24000  # Qwen3-TTS output sample rate

# ── Text IDs to test ──
# We'll try to find real code dumps, or generate them on the fly
CODE_DUMP_DIR = os.environ.get(
    "TTS_CODE_DUMP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "output", "code_dumps"),
)


def _decode_with_new_graph(decoder, chunk, state, accel):
    """
    Decode one chunk using the new graph accelerator.
    Manually performs the decode steps with graph support for conv path.
    """
    from tts_incremental_decoder import _stream_causal_conv

    codes = torch.as_tensor(chunk)
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)
    if codes.dim() != 3 or codes.shape[0] != 1:
        raise ValueError("Expected (1, T, Q)")

    state.codebook_dim = int(codes.shape[-1])
    valid_frames = int((codes[:, :, 0] > 0).sum().item())
    state.expected_samples += valid_frames * decoder.decode_upsample_rate
    codes = codes.to(decoder.device).to(torch.long)
    codes = codes.transpose(1, 2)  # (1, Q, T)

    with torch.no_grad():
        hidden = decoder.decoder.quantizer.decode(codes)
        hidden, state.pre_conv = _stream_causal_conv(
            decoder.decoder.pre_conv, hidden, state.pre_conv
        )
        hidden = hidden.transpose(1, 2)  # (1, T, C)

        t_new = hidden.shape[1]
        cache_position = torch.arange(state.pos, state.pos + t_new, device=hidden.device)
        out = decoder.decoder.pre_transformer(
            inputs_embeds=hidden, use_cache=True,
            past_key_values=state.kv_cache, cache_position=cache_position,
        )
        state.kv_cache = out.past_key_values
        state.pos += t_new
        hidden = out.last_hidden_state

        hidden = hidden.permute(0, 2, 1).contiguous()

        decoder._decoder_graph_step_count += 1
        step = decoder._decoder_graph_step_count

        if step == 1:
            # First step: eager warmup
            hidden = decoder._decode_conv_path(hidden, state)
            accel.stats.eager_steps += 1
        elif step == 2:
            # Second step: capture graph
            result = accel.capture(hidden, state)
            if result is None:
                hidden = decoder._decode_conv_path(hidden, state)
                accel.stats.eager_steps += 1
            else:
                hidden = result
        else:
            # Subsequent steps: replay
            result = accel.replay(hidden, state)
            if result is None:
                hidden = decoder._decode_conv_path(hidden, state)
                accel.stats.eager_steps += 1
            else:
                hidden = result

    audio = hidden.squeeze(0).squeeze(0).detach().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.flatten()
    audio = audio.astype(np.float32)
    state.emitted_samples += len(audio)
    return audio


def find_code_dumps():
    """Find real codes tensors from code_dump directory."""
    codes_files = sorted(glob.glob(os.path.join(CODE_DUMP_DIR, "codes_offline_*.pt")))
    if not codes_files:
        codes_files = sorted(glob.glob(os.path.join(CODE_DUMP_DIR, "codes_*.pt")))

    results = {}
    # Try to find short and long codes
    for f in codes_files:
        try:
            codes = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(codes, torch.Tensor) and codes.dim() == 2:
                n_frames = codes.shape[0]
                tag = os.path.basename(f)
                if n_frames < 30:
                    if "short" not in results:
                        results["short"] = (tag, codes)
                elif n_frames > 200:
                    if "long" not in results:
                        results["long"] = (tag, codes)
        except Exception:
            continue

    if not results:
        # Generate synthetic codes
        print("[WARN] No code dumps found, using synthetic codes")
        results["short"] = ("synthetic_short", torch.randint(1, 1000, (14, 16)))
        results["long"] = ("synthetic_long", torch.randint(1, 1000, (305, 16)))

    return results


def load_model_and_decoder():
    """Load model and create IncrementalDecoder."""
    os.environ["TTS_DEEP_STREAM_ENABLE"] = "1"
    os.environ["TTS_DEEP_STREAM_PROCESS"] = "0"
    os.environ["TTS_DEEP_STREAM_MODEL_DIR"] = MODEL_DIR

    import tts_server as ts
    ts._init_deep_stream_backend()

    from tts_incremental_decoder import IncrementalDecoder

    decoder = IncrementalDecoder(ts._deep_tokenizer, device=DEVICE, transformer_mode="cache")
    return decoder


def benchmark_decode(decoder, codes_2d, label, use_graph=False, use_new_graph=False):
    """
    Run decode-only benchmark on a codes tensor.

    Args:
        decoder: IncrementalDecoder instance
        codes_2d: shape (T, Q=16) int64 tensor
        label: descriptive label
        use_graph: whether to enable old CUDA Graph for conv path
        use_new_graph: whether to enable new proper CUDA Graph
    """
    n_frames = codes_2d.shape[0]
    n_codebooks = codes_2d.shape[1]
    audio_seconds = n_frames / 12.0  # 12 Hz codec rate

    # Temporarily set graph flag
    old_use_graph = decoder.use_cudagraph
    decoder.use_cudagraph = use_graph

    # Install/uninstall new graph accelerator
    _new_accel = None
    if use_new_graph:
        from decoder_cudagraph import DecoderConvGraphAccelerator
        _new_accel = DecoderConvGraphAccelerator(decoder.decoder)
        decoder._decoder_graph_accel = _new_accel
        decoder._decoder_graph_step_count = 0

    all_step_ms = []
    all_pcm_hashes = []
    all_wall_s = []

    for run_idx in range(WARMUP_RUNS + TIMED_RUNS):
        state = decoder.reset_state()
        # Reset new graph accelerator per run
        if _new_accel is not None:
            _new_accel.__init__(decoder.decoder)
            decoder._decoder_graph_accel = _new_accel
            decoder._decoder_graph_step_count = 0
        pcm_all = []
        step_times = []

        # Feed codes in packets of PACKET_TOKENS
        for start in range(0, n_frames, PACKET_TOKENS):
            end = min(start + PACKET_TOKENS, n_frames)
            chunk = codes_2d[start:end]  # (packet, Q)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if _new_accel is not None:
                audio_np = _decode_with_new_graph(decoder, chunk, state, _new_accel)
            else:
                audio_np, state = decoder.decode_incremental(chunk, state)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_times.append((t1 - t0) * 1000.0)  # ms
            pcm_all.append(audio_np)

        # Finalize
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        final_audio, state = decoder.finalize(state)
        torch.cuda.synchronize()
        tf1 = time.perf_counter()
        step_times.append((tf1 - tf0) * 1000.0)
        pcm_all.append(final_audio)

        total_pcm = np.concatenate(pcm_all)
        wall_s = sum(step_times) / 1000.0
        pcm_hash = hashlib.md5(total_pcm.tobytes()).hexdigest()[:12]

        if run_idx >= WARMUP_RUNS:
            all_step_ms.append(step_times)
            all_pcm_hashes.append(pcm_hash)
            all_wall_s.append(wall_s)

    # Compute stats
    all_steps_flat = []
    for run_steps in all_step_ms:
        # Skip first step (warmup/prefill) and last step (finalize)
        decode_steps = run_steps[1:-1] if len(run_steps) > 2 else run_steps
        all_steps_flat.extend(decode_steps)

    step_arr = np.array(all_steps_flat)
    wall_arr = np.array(all_wall_s)

    pcm_seconds = len(total_pcm) / SAMPLE_RATE
    rtf_arr = wall_arr / pcm_seconds

    unique_hashes = set(all_pcm_hashes)
    bit_exact = len(unique_hashes) == 1

    result = {
        "label": label,
        "use_graph": use_graph,
        "n_frames": n_frames,
        "audio_seconds": round(pcm_seconds, 2),
        "pcm_samples": len(total_pcm),
        "n_steps": len(all_step_ms[0]) if all_step_ms else 0,
        "wall_s_p50": round(float(np.percentile(wall_arr, 50)), 4),
        "wall_s_p95": round(float(np.percentile(wall_arr, 95)), 4),
        "rtf_p50": round(float(np.percentile(rtf_arr, 50)), 4),
        "rtf_p95": round(float(np.percentile(rtf_arr, 95)), 4),
        "step_ms_avg": round(float(step_arr.mean()), 3),
        "step_ms_p50": round(float(np.percentile(step_arr, 50)), 3),
        "step_ms_p95": round(float(np.percentile(step_arr, 95)), 3),
        "step_ms_p99": round(float(np.percentile(step_arr, 99)), 3),
        "step_ms_max": round(float(step_arr.max()), 3),
        "bit_exact": bit_exact,
        "pcm_hash": list(unique_hashes)[0] if bit_exact else str(unique_hashes),
    }

    # Restore
    decoder.use_cudagraph = old_use_graph
    return result


def profile_launches_new_graph(decoder, codes_2d):
    """Profile CUDA kernel launches with new graph for steady-state steps."""
    from decoder_cudagraph import DecoderConvGraphAccelerator

    accel = DecoderConvGraphAccelerator(decoder.decoder)
    decoder._decoder_graph_accel = accel
    decoder._decoder_graph_step_count = 0

    state = decoder.reset_state()
    n_frames = codes_2d.shape[0]

    # Run enough steps to capture graph (steps 1=warmup, 2=capture, 3+=replay)
    for start in range(0, min(n_frames, PACKET_TOKENS * 6), PACKET_TOKENS):
        end = min(start + PACKET_TOKENS, n_frames)
        chunk = codes_2d[start:end]
        _decode_with_new_graph(decoder, chunk, state, accel)
    torch.cuda.synchronize()

    # Profile steady-state replay steps
    profile_steps = min(10, (n_frames - PACKET_TOKENS * 6) // PACKET_TOKENS)
    start_offset = PACKET_TOKENS * 6
    launch_counts = []

    for step_idx in range(profile_steps):
        start = start_offset + step_idx * PACKET_TOKENS
        end = min(start + PACKET_TOKENS, n_frames)
        if end <= start:
            break
        chunk = codes_2d[start:end]

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            _decode_with_new_graph(decoder, chunk, state, accel)
            torch.cuda.synchronize()

        events = prof.key_averages()
        total_kernels = sum(e.count for e in events if "cuda" in e.key.lower())
        launch_counts.append(total_kernels)

    # Cleanup
    if hasattr(decoder, '_decoder_graph_accel'):
        del decoder._decoder_graph_accel
    if hasattr(decoder, '_decoder_graph_step_count'):
        del decoder._decoder_graph_step_count

    if launch_counts:
        return {
            "launches_per_step_avg": round(np.mean(launch_counts), 1),
            "launches_per_step_min": int(min(launch_counts)),
            "launches_per_step_max": int(max(launch_counts)),
            "profiled_steps": len(launch_counts),
        }
    return {"launches_per_step_avg": -1, "profiled_steps": 0}


def profile_launches(decoder, codes_2d, use_graph=False):
    """Profile CUDA kernel launches for one decode pass."""
    old_use_graph = decoder.use_cudagraph
    decoder.use_cudagraph = use_graph

    state = decoder.reset_state()
    n_frames = codes_2d.shape[0]

    # Warmup
    for start in range(0, min(n_frames, PACKET_TOKENS * 4), PACKET_TOKENS):
        end = min(start + PACKET_TOKENS, n_frames)
        chunk = codes_2d[start:end]
        _, state = decoder.decode_incremental(chunk, state)
    torch.cuda.synchronize()

    # Profile a few steady-state steps
    profile_steps = min(10, (n_frames - PACKET_TOKENS * 4) // PACKET_TOKENS)
    start_offset = PACKET_TOKENS * 4
    launch_counts = []

    for step_idx in range(profile_steps):
        start = start_offset + step_idx * PACKET_TOKENS
        end = min(start + PACKET_TOKENS, n_frames)
        if end <= start:
            break
        chunk = codes_2d[start:end]

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            _, state = decoder.decode_incremental(chunk, state)
            torch.cuda.synchronize()

        events = prof.key_averages()
        launches = sum(1 for e in events if "cudaLaunchKernel" in e.key)
        total_kernels = sum(e.count for e in events if "cuda" in e.key.lower())
        launch_counts.append(total_kernels)

    decoder.use_cudagraph = old_use_graph

    if launch_counts:
        return {
            "launches_per_step_avg": round(np.mean(launch_counts), 1),
            "launches_per_step_min": int(min(launch_counts)),
            "launches_per_step_max": int(max(launch_counts)),
            "profiled_steps": len(launch_counts),
        }
    return {"launches_per_step_avg": -1, "profiled_steps": 0}


def main():
    print("=" * 60)
    print("P4.A1: Decoder-Only Microbench")
    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Packet tokens: {PACKET_TOKENS}")
    print(f"Warmup runs: {WARMUP_RUNS}, Timed runs: {TIMED_RUNS}")
    print(f"Device: {DEVICE}")
    print(f"Decoder CUDA Graph: {USE_CUDAGRAPH}")
    print()

    # Load model
    decoder = load_model_and_decoder()
    print("[OK] Model and decoder loaded\n")

    # Find codes
    code_dumps = find_code_dumps()
    print(f"[OK] Found {len(code_dumps)} code dumps: {list(code_dumps.keys())}\n")

    results = []

    for text_id, (tag, codes) in code_dumps.items():
        print(f"--- {text_id} ({tag}): {codes.shape[0]} frames ---")

        # 1) Baseline (eager)
        print(f"  [1/4] Baseline (eager)...")
        r_baseline = benchmark_decode(decoder, codes, f"{text_id}_eager", use_graph=False)
        print(f"        RTF={r_baseline['rtf_p50']:.4f} (P50) step_avg={r_baseline['step_ms_avg']:.2f}ms hash={r_baseline['pcm_hash']}")

        # Profile launches (eager)
        print(f"  [2/4] Profiling launches (eager)...")
        l_baseline = profile_launches(decoder, codes, use_graph=False)
        print(f"        launches/step={l_baseline['launches_per_step_avg']}")
        r_baseline["launches"] = l_baseline
        results.append(r_baseline)

        # 3) New proper CUDA Graph (conv path with in-place state)
        print(f"  [3/4] NEW Graph (proper state mgmt)...")
        r_new_graph = benchmark_decode(decoder, codes, f"{text_id}_new_graph", use_graph=False, use_new_graph=True)
        print(f"        RTF={r_new_graph['rtf_p50']:.4f} (P50) step_avg={r_new_graph['step_ms_avg']:.2f}ms hash={r_new_graph['pcm_hash']}")

        print(f"  [4/4] Profiling launches (new graph)...")
        l_new_graph = profile_launches_new_graph(decoder, codes)
        print(f"        launches/step={l_new_graph['launches_per_step_avg']}")
        r_new_graph["launches"] = l_new_graph
        results.append(r_new_graph)

        # Bit-exact check
        if r_baseline["pcm_hash"] == r_new_graph["pcm_hash"]:
            print(f"  ✅ Bit-exact: eager == new_graph (hash={r_baseline['pcm_hash']})")
        else:
            print(f"  ❌ NOT bit-exact: eager={r_baseline['pcm_hash']} new_graph={r_new_graph['pcm_hash']}")

        # Speedup
        speedup = r_baseline["rtf_p50"] / r_new_graph["rtf_p50"] if r_new_graph["rtf_p50"] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x (RTF {r_baseline['rtf_p50']:.4f} → {r_new_graph['rtf_p50']:.4f})")
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Label':<25} {'RTF_P50':>8} {'RTF_P95':>8} {'step_avg':>9} {'step_p95':>9} {'launches':>9} {'hash':>14} {'exact':>5}")
    print("-" * 100)
    for r in results:
        l = r.get("launches", {})
        print(
            f"{r['label']:<25} {r['rtf_p50']:>8.4f} {r['rtf_p95']:>8.4f} "
            f"{r['step_ms_avg']:>8.2f}ms {r['step_ms_p95']:>8.2f}ms "
            f"{l.get('launches_per_step_avg', -1):>9.0f} "
            f"{r['pcm_hash']:>14} {'✅' if r['bit_exact'] else '❌':>5}"
        )

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "output", "decoder_microbench.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

