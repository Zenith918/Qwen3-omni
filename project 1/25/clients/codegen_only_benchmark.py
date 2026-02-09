#!/usr/bin/env python3
"""
P3 Step 3: codegen-only end-to-end benchmark for CUDA Graph evaluation.

Runs 4 groups:
  1) baseline (both graph flags 0)
  2) talker=1, cp=0
  3) talker=0, cp=1
  4) talker=1, cp=1

For each group: RTF_codegen_only, cudaLaunchKernel count, graph_used_rate, bit-exactness.
"""

import sys, os, time, json, gc, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/vllm-omni")

import torch

# ── Config ────────────────────────────────────────────────────────────
TEXT = "如果系统在长句播放时出现越来越慢的情况，请先记录当时的时间戳、首包时间以及整体音频时长，并把日志打包。随后尝试相同文本重复三次，观察 RTF 是否呈线性上升，这能帮助我们定位是解码还是拼接导致的问题，同时也便于对比修复效果。"
RUNS = 3
WARMUP = 1
DEVICE = "cuda:0"

GEN_KWARGS = {
    "do_sample": False, "top_p": 1.0, "repetition_penalty": 1.0,
    "subtalker_dosample": False, "subtalker_top_p": 1.0,
    "subtalker_top_k": 0, "subtalker_temperature": 1.0,
}

GROUPS = [
    {"name": "baseline",       "talker": False, "cp": False},
    {"name": "talker=1,cp=0",  "talker": True,  "cp": False},
    {"name": "talker=0,cp=1",  "talker": False, "cp": True},
    {"name": "talker=1,cp=1",  "talker": True,  "cp": True},
]


def count_launches(prof_events):
    total = 0
    for e in prof_events:
        if "cudaLaunchKernel" in e.key or "cudaLaunch" in e.key:
            total += e.count
    return total


def code_hash(codes_list):
    h = hashlib.md5()
    for c in codes_list:
        if isinstance(c, torch.Tensor):
            h.update(c.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class _Collector:
    def __init__(self):
        self.codes = []
    def put(self, c):
        self.codes.append(c)
    def close(self):
        pass
    def end(self):
        pass


def run_codegen(model):
    """Run one codegen pass, return (wall_s, n_frames, codes_hash)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    collector = _Collector()
    torch.cuda.synchronize()
    t0 = time.time()
    model.generate_custom_voice_codes(
        text=TEXT, speaker="serena", language="Chinese",
        instruct="", codec_streamer=collector,
        max_new_tokens=2048, non_streaming_mode=False, **GEN_KWARGS,
    )
    torch.cuda.synchronize()
    wall = time.time() - t0
    n_frames = len(collector.codes)
    ch = code_hash(collector.codes)
    return wall, n_frames, ch


def load_model():
    """Load model once."""
    os.environ["TTS_DEEP_STREAM_ENABLE"] = "1"
    os.environ["TTS_DEEP_STREAM_PROCESS"] = "0"
    os.environ["TTS_DEEP_STREAM_DETERMINISTIC"] = "1"
    os.environ["TTS_DEEP_STREAM_DETERMINISTIC_POLICY"] = "greedy"
    os.environ["TTS_DEEP_STREAM_SEED_MODE"] = "fixed"
    os.environ["TTS_DEEP_STREAM_SEED"] = "42"
    os.environ["TTS_CODE_DUMP_ENABLE"] = "0"
    model_dir = os.environ.get("TTS_DEEP_STREAM_MODEL_DIR",
                                "/workspace/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    os.environ["TTS_DEEP_STREAM_MODEL_DIR"] = model_dir

    import tts_server as ts
    if ts._deep_model is None:
        ts._init_deep_stream_backend()
    return ts._deep_model


def install_graphs(model, talker_flag, cp_flag):
    """Install CUDA Graph patches. Returns (stats, uninstall_fn)."""
    from codegen_cudagraph import install_cudagraph_accelerator
    talker = model.model.talker

    # Save originals for restore
    orig_cp_generate = talker.code_predictor.generate
    orig_talker_forward = talker.model.forward

    stats = install_cudagraph_accelerator(model, talker_flag, cp_flag)

    def uninstall():
        talker.code_predictor.generate = orig_cp_generate
        talker.model.forward = orig_talker_forward

    return stats, uninstall


def run_group(model, group, baseline_hash=None):
    name = group["name"]
    print(f"\n{'='*60}")
    print(f"Group: {name}")
    print(f"{'='*60}")

    # Warmup
    for _ in range(WARMUP):
        run_codegen(model)

    # Timed runs
    walls = []
    hashes = []
    n_frames_ref = 0
    for i in range(RUNS):
        wall, n_frames, ch = run_codegen(model)
        walls.append(wall)
        hashes.append(ch)
        n_frames_ref = n_frames
        print(f"  Run {i+1}: {wall:.3f}s  frames={n_frames}  hash={ch[:12]}")

    # Stats
    p50_wall = sorted(walls)[len(walls) // 2]
    audio_dur = n_frames_ref / 12.5
    rtf = p50_wall / audio_dur if audio_dur > 0 else -1

    # Determinism
    unique_hashes = len(set(hashes))
    bit_exact = unique_hashes == 1
    hash_match_baseline = hashes[0] == baseline_hash if baseline_hash else None

    # Profile launches using SHORT text (1 run)
    SHORT_TEXT = "你好，测试一下。"  # very short → ~10-30 frames
    print("  Profiling launches (short text)...")

    def run_codegen_short(m):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        c = _Collector()
        m.generate_custom_voice_codes(
            text=SHORT_TEXT, speaker="serena", language="Chinese",
            instruct="", codec_streamer=c,
            max_new_tokens=512, non_streaming_mode=False, **GEN_KWARGS,
        )
        torch.cuda.synchronize()
        return len(c.codes)

    # Warmup short
    n_short = run_codegen_short(model)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        n_short = run_codegen_short(model)
    launches_short = count_launches(prof.key_averages())
    launches_per_frame = launches_short / max(1, n_short)
    launches = int(launches_per_frame * n_frames_ref)  # extrapolate
    print(f"    short_frames={n_short}, launches_short={launches_short}, l/f={launches_per_frame:.0f}")

    # Graph stats
    try:
        from codegen_cudagraph import get_cudagraph_stats, reset_cudagraph_stats
        gs = get_cudagraph_stats()
        reset_cudagraph_stats()
    except Exception:
        gs = {}

    graph_used = gs.get("cudagraph_talker_used", 0) + gs.get("cudagraph_cp_used", 0)
    fallbacks = gs.get("fallback_count", 0)
    prefill_eager = gs.get("prefill_eager_steps", 0)
    total_steps = graph_used + fallbacks + prefill_eager
    graph_used_rate = graph_used / max(1, total_steps)

    result = {
        "group": name,
        "talker_flag": group["talker"],
        "cp_flag": group["cp"],
        "p50_wall_s": round(p50_wall, 3),
        "n_frames": n_frames_ref,
        "audio_dur_s": round(audio_dur, 2),
        "rtf_codegen_only": round(rtf, 4),
        "launches": launches,
        "launches_per_frame": round(launches_per_frame, 0),
        "bit_exact": bit_exact,
        "hash_unique": unique_hashes,
        "hash": hashes[0],
        "hash_match_baseline": hash_match_baseline,
        "graph_used_rate": round(graph_used_rate, 4),
        "cudagraph_stats": gs,
    }

    print(f"\n  RTF_codegen_only  = {rtf:.4f}")
    print(f"  Launches          = {launches} ({launches_per_frame:.0f}/frame)")
    print(f"  Bit-exact         = {bit_exact} ({unique_hashes} unique)")
    print(f"  Graph used rate   = {graph_used_rate:.1%}")
    if hash_match_baseline is not None:
        print(f"  Hash vs baseline  = {'✅' if hash_match_baseline else '❌'}")

    return result


def main():
    OUT_DIR = "/workspace/project 1/25/output/p3_codegen_benchmark"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("P3 Step 3: Codegen-Only End-to-End Benchmark")
    print(f"Text: {TEXT[:40]}... ({len(TEXT)} chars)")
    print(f"Runs: {RUNS} (warmup: {WARMUP})")
    print("=" * 60)

    model = load_model()
    results = []
    baseline_hash = None

    for group in GROUPS:
        uninstall_fn = None

        # Install graph patches if needed
        if group["talker"] or group["cp"]:
            _, uninstall_fn = install_graphs(model, group["talker"], group["cp"])

        result = run_group(model, group, baseline_hash=baseline_hash)
        results.append(result)

        if group["name"] == "baseline":
            baseline_hash = result["hash"]

        # Restore model to clean state
        if uninstall_fn is not None:
            uninstall_fn()

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Group':<20} {'RTF':>8} {'Launches':>10} {'L/Frame':>8} {'UsedRate':>10} {'BitExact':>10} {'vsBase':>8}")
    print("-" * 90)
    for r in results:
        vs = "—"
        if r["hash_match_baseline"] is True:
            vs = "✅"
        elif r["hash_match_baseline"] is False:
            vs = "❌"
        print(f"{r['group']:<20} {r['rtf_codegen_only']:>8.4f} {r['launches']:>10} "
              f"{r['launches_per_frame']:>8.0f} "
              f"{r['graph_used_rate']:>9.1%} "
              f"{'✅' if r['bit_exact'] else '❌':>10} {vs:>8}")

    # Speedup summary
    base_rtf = results[0]["rtf_codegen_only"]
    print(f"\nSpeedup vs baseline (RTF {base_rtf:.4f}):")
    for r in results[1:]:
        speedup = base_rtf / r["rtf_codegen_only"] if r["rtf_codegen_only"] > 0 else 0
        print(f"  {r['group']:<20} RTF={r['rtf_codegen_only']:.4f}  speedup={speedup:.2f}x")

    # Go/No-Go
    g4 = results[3] if len(results) >= 4 else None
    print("\n── Go/No-Go ──")
    if g4:
        rtf4 = g4["rtf_codegen_only"]
        rate4 = g4["graph_used_rate"]
        exact4 = g4["hash_match_baseline"]
        if rtf4 <= 0.70 and rate4 >= 0.90 and exact4:
            print(f"  ✅ Group 4 (all): RTF={rtf4:.4f} ≤0.70, rate={rate4:.1%} ≥90%, bit-exact=✅ → PROCEED")
        else:
            print(f"  ⚠️  Group 4 (all): RTF={rtf4:.4f}, rate={rate4:.1%}, bit-exact={'✅' if exact4 else '❌'}")
            if rtf4 > 0.70:
                print(f"     RTF {rtf4:.4f} > 0.70 target")
            if rate4 < 0.90:
                print(f"     Graph used rate {rate4:.1%} < 90%")
            if not exact4:
                print(f"     NOT bit-exact vs baseline")

    # Save
    out_path = os.path.join(OUT_DIR, "p3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
