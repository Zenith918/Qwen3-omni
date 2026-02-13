#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from common import load_wav_mono_int16


def _log_mel_distance(wav_a: str, wav_b: str, sr: int = 24000, n_mels: int = 40) -> float:
    """计算两个 wav 之间的 log-mel spectrogram 距离（失真指标）"""
    try:
        sa, sra = load_wav_mono_int16(wav_a)
        sb, srb = load_wav_mono_int16(wav_b)
        if len(sa) == 0 or len(sb) == 0:
            return -1.0
        xa = sa.astype(np.float32) / 32768.0
        xb = sb.astype(np.float32) / 32768.0
        # 截取到相同长度
        min_len = min(len(xa), len(xb))
        xa, xb = xa[:min_len], xb[:min_len]

        # 简易 mel spectrogram（不依赖 librosa）
        def _mel_spec(x, sr_val, n_fft=1024, hop=512):
            # STFT
            n_frames = (len(x) - n_fft) // hop + 1
            if n_frames <= 0:
                return np.zeros((n_mels, 1))
            frames = np.lib.stride_tricks.as_strided(
                x, shape=(n_frames, n_fft),
                strides=(x.strides[0] * hop, x.strides[0]))
            window = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
            # Mel filterbank
            fmin, fmax = 0.0, sr_val / 2.0
            mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
            mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
            mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
            hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
            bins = np.floor((n_fft + 1) * hz_pts / sr_val).astype(int)
            fb = np.zeros((n_mels, n_fft // 2 + 1))
            for m in range(n_mels):
                for k in range(bins[m], bins[m + 1]):
                    if bins[m + 1] > bins[m]:
                        fb[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
                for k in range(bins[m + 1], bins[m + 2]):
                    if bins[m + 2] > bins[m + 1]:
                        fb[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])
            mel = np.dot(spec, fb.T).T  # (n_mels, n_frames)
            return np.log(mel + 1e-10)

        ma = _mel_spec(xa, sra or sr)
        mb = _mel_spec(xb, srb or sr)
        min_f = min(ma.shape[1], mb.shape[1])
        if min_f == 0:
            return -1.0
        dist = float(np.mean((ma[:, :min_f] - mb[:, :min_f]) ** 2) ** 0.5)
        return round(dist, 4)
    except Exception:
        return -1.0


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _pct(values: list[float], p: float):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def _audio_quality_metrics(wav_path: str) -> dict:
    samples, sr = load_wav_mono_int16(wav_path)
    if len(samples) == 0 or sr <= 0:
        return {
            "dropout_count": 0,
            "max_dropout_ms": 0.0,
            "clipping_ratio": 0.0,
            "rms": 0.0,
            "duration_s": 0.0,
        }

    x = samples.astype(np.float32) / 32768.0
    # 忽略头尾静音，避免固定录制窗口导致 dropout 误报
    voiced_idx = np.where(np.abs(x) >= 0.01)[0]
    if len(voiced_idx) > 0:
        x_voiced = x[int(voiced_idx[0]): int(voiced_idx[-1]) + 1]
    else:
        x_voiced = x
    abs_x = np.abs(x_voiced)

    clipping_ratio = float(np.mean(abs_x > 0.99))
    rms = float(math.sqrt(float(np.mean(x_voiced * x_voiced)))) if len(x_voiced) > 0 else 0.0

    # ── D7 P0-2: 两层 dropout 检测 ──────────────────────────
    # 用 20ms 窗口滑动计算能量，检测 gap
    frame_ms = 20
    frame_samples = int(sr * frame_ms / 1000)
    energy_threshold = 0.005  # 低于此为"静音帧"
    gap_frames = []  # 连续静音帧数
    current_gap = 0
    for i in range(0, len(x_voiced) - frame_samples, frame_samples):
        frame = x_voiced[i:i + frame_samples]
        frame_rms = float(np.sqrt(np.mean(frame * frame)))
        if frame_rms < energy_threshold:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_frames.append(current_gap)
            current_gap = 0
    if current_gap > 0:
        gap_frames.append(current_gap)

    gap_ms_list = [g * frame_ms for g in gap_frames]
    micro_gaps = [g for g in gap_ms_list if 40 <= g < 200]
    audible_gaps = [g for g in gap_ms_list if g >= 200]
    # 也算 120ms+ 连续出现 >=2 次
    mid_gaps = [g for g in gap_ms_list if 120 <= g < 200]
    if len(mid_gaps) >= 2:
        audible_gaps.extend(mid_gaps)

    max_gap_ms = max(gap_ms_list) if gap_ms_list else 0.0

    return {
        "micro_gap_count": len(micro_gaps),
        "audible_dropout_count": len(audible_gaps),
        "max_gap_ms": max_gap_ms,
        "dropout_count": len(micro_gaps) + len(audible_gaps),  # 兼容旧字段
        "max_dropout_ms": max_gap_ms,
        "clipping_ratio": clipping_ratio,
        "rms": rms,
        "duration_s": float(len(samples)) / float(sr),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute AutoRTC audio metrics + gate report.")
    p.add_argument("--run_summary", required=True, help="run summary.json from run_suite")
    p.add_argument("--autortc_traces", required=True, help="output/autortc/traces.jsonl")
    p.add_argument("--agent_traces", default="output/day5_e2e_traces.jsonl", help="agent trace jsonl")
    p.add_argument("--output_dir", required=True, help="run output dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = _read_json(args.run_summary)
    autortc_rows = _read_jsonl(args.autortc_traces)
    agent_rows = _read_jsonl(args.agent_traces)

    # 只保留当前 run_id
    run_id = summary.get("run_id", "")
    autortc_rows = [r for r in autortc_rows if r.get("run_id") == run_id]

    # trace_id -> agent row
    agent_map = {}
    for row in agent_rows:
        tid = row.get("trace_id")
        if tid:
            agent_map[tid] = row

    metrics_rows = []
    eot_first_audio = []
    tts_first_publish = []
    fast_lane_ttft = []
    inter_arrival_p95 = []
    dropout_counts = []
    clipping_ratios = []
    rms_values = []

    autortc_map = {r.get("trace_id"): r for r in autortc_rows if r.get("trace_id")}

    for case in summary.get("cases", []):
        trace_id = case.get("trace_id")
        probe = case.get("probe_result", {})
        wav = case.get("probe_wav")
        aq = _audio_quality_metrics(wav) if wav and os.path.exists(wav) else _audio_quality_metrics("")

        ar = autortc_map.get(trace_id, {})
        agent = agent_map.get(trace_id, {})
        latency_ms = agent.get("latency_ms", {}) if isinstance(agent, dict) else {}

        t_user_end = ar.get("t_user_send_end")
        t_probe_first = ar.get("t_probe_first_audio_after_user_end") or ar.get("t_probe_first_audio_recv")
        eot_fa = (t_probe_first - t_user_end) * 1000.0 if t_user_end and t_probe_first else None

        inter_p95 = probe.get("inter_arrival_p95_ms")
        frame_ts = probe.get("frame_timestamps", []) if isinstance(probe, dict) else []
        inter_gaps = []
        if frame_ts and len(frame_ts) > 1:
            for i in range(1, len(frame_ts)):
                inter_gaps.append((float(frame_ts[i]) - float(frame_ts[i - 1])) * 1000.0)
        gap_dropouts = [g for g in inter_gaps if g > 40.0]
        dropout_count = len(gap_dropouts)
        max_dropout_ms = max(gap_dropouts) if gap_dropouts else 0.0
        tts_pub = latency_ms.get("tts_first_to_publish")
        llm_first = latency_ms.get("stt_done_to_llm_first")

        # D7: Ring1 pre_rtc vs Ring2 post_rtc 失真
        case_id = case.get("case_id", "")
        run_dir_path = Path(args.output_dir)
        pre_rtc_wav = str(run_dir_path / case_id / "pre_rtc.wav")
        mel_dist = _log_mel_distance(pre_rtc_wav, wav) if os.path.exists(pre_rtc_wav) and wav and os.path.exists(wav) else -1.0

        if eot_fa is not None:
            eot_first_audio.append(eot_fa)
        if tts_pub is not None:
            tts_first_publish.append(float(tts_pub))
        if llm_first is not None and float(llm_first) >= 0:
            fast_lane_ttft.append(float(llm_first))
        if inter_p95 is not None:
            inter_arrival_p95.append(float(inter_p95))

        dropout_counts.append(dropout_count)
        clipping_ratios.append(aq["clipping_ratio"])
        rms_values.append(aq["rms"])

        metrics_rows.append({
            "case_id": case.get("case_id"),
            "trace_id": trace_id,
            "ok": case.get("ok"),
            "eot_to_first_audio_ms": round(eot_fa, 1) if eot_fa is not None else "",
            "tts_first_to_publish_ms": tts_pub if tts_pub is not None else "",
            "fast_lane_ttft_ms": llm_first if llm_first is not None else "",
            "inter_arrival_p95_ms": round(inter_p95, 1) if inter_p95 is not None else "",
            "dropout_count": dropout_count,
            "max_dropout_ms": round(max_dropout_ms, 1),
            "clipping_ratio": aq["clipping_ratio"],
            "rms": aq["rms"],
            "mel_distance": mel_dist,
            "micro_gap_count": aq.get("micro_gap_count", 0),
            "audible_dropout_count": aq.get("audible_dropout_count", 0),
            "duration_s": round(aq["duration_s"], 3),
            "probe_wav": wav or "",
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()) if metrics_rows else [
            "case_id", "trace_id", "ok", "eot_to_first_audio_ms", "tts_first_to_publish_ms",
            "fast_lane_ttft_ms", "inter_arrival_p95_ms", "dropout_count", "max_dropout_ms",
            "clipping_ratio", "rms", "duration_s", "probe_wav",
        ])
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    fast_lane_non_negative = all(v >= 0 for v in fast_lane_ttft) if fast_lane_ttft else False
    min_rms = min(rms_values) if rms_values else 0.0

    # D7 P0-2: 合理化 dropout — 收集 audible_dropout 而非所有 gap
    total_audible_dropouts = sum(row.get("audible_dropout_count", 0) for row in
        [_audio_quality_metrics(c.get("probe_wav", "")) for c in summary.get("cases", [])
         if c.get("probe_wav") and os.path.exists(c.get("probe_wav", ""))]
    ) if metrics_rows else 0
    max_gap_all = max((row.get("max_gap_ms", 0) for row in
        [_audio_quality_metrics(c.get("probe_wav", "")) for c in summary.get("cases", [])
         if c.get("probe_wav") and os.path.exists(c.get("probe_wav", ""))]
    ), default=0)

    gates = {
        "EoT->FirstAudio P95 <= 650ms": (_pct(eot_first_audio, 95) or float("inf")) <= 650.0,
        "tts_first->publish P95 <= 120ms": (_pct(tts_first_publish, 95) or float("inf")) <= 120.0,
        "audible_dropout_count == 0": total_audible_dropouts == 0,
        "max_gap_ms < 200ms": max_gap_all < 200,
        "clipping_ratio < 0.1%": (max(clipping_ratios) if clipping_ratios else 1.0) < 0.001,
        "fast lane TTFT P95 <= 80ms": fast_lane_non_negative and ((_pct(fast_lane_ttft, 95) or float("inf")) <= 80.0),
        "all cases have audio (min_rms >= 0.01)": min_rms >= 0.01,
    }
    # 仅在存在数据时判断 jitter
    if inter_arrival_p95:
        gates["inter_arrival_p95 <= 30ms"] = (_pct(inter_arrival_p95, 95) or float("inf")) <= 30.0

    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AutoRTC Report\n\n")
        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- total_cases: `{summary.get('total_cases', 0)}`\n")
        f.write(f"- ok_cases: `{summary.get('ok_cases', 0)}`\n\n")
        f.write("## Aggregates\n\n")
        f.write(f"- EoT->FirstAudio P95: `{_pct(eot_first_audio, 95)}` ms\n")
        f.write(f"- tts_first->publish P95: `{_pct(tts_first_publish, 95)}` ms\n")
        f.write(f"- fast lane TTFT P95: `{_pct(fast_lane_ttft, 95)}` ms\n")
        f.write(f"- inter_arrival P95: `{_pct(inter_arrival_p95, 95)}` ms\n")
        f.write(f"- max clipping_ratio: `{max(clipping_ratios) if clipping_ratios else None}`\n")
        f.write(f"- total dropout_count: `{sum(dropout_counts)}`\n")
        f.write("\n## Gates\n\n")
        for name, ok in gates.items():
            f.write(f"- {'PASS' if ok else 'FAIL'}: {name}\n")

    breakdown_path = out_dir / "latency_breakdown.md"
    with open(breakdown_path, "w", encoding="utf-8") as f:
        f.write("# Latency Breakdown\n\n")
        f.write(f"- EoT->FirstAudio P95: `{_pct(eot_first_audio, 95)}` ms\n")
        f.write(f"- tts_first->publish P95: `{_pct(tts_first_publish, 95)}` ms\n")
        f.write(f"- fast lane TTFT P95: `{_pct(fast_lane_ttft, 95)}` ms\n")

    print(f"metrics_csv={csv_path}")
    print(f"report_md={report_path}")
    print(f"breakdown_md={breakdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

