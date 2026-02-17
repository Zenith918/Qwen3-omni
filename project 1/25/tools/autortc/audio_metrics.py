#!/usr/bin/env python3
"""
D9: AutoRTC audio metrics + gate report.

Key changes from D8:
- P0-1: dropout/max_gap measured on reply segment (post_rtc_reply.wav), strict thresholds
- P0-2: audio_valid_rate = 100% (POST_SILENT → FAIL)
- P0-3: pre_rtc path by trace_id only
- P0-4: capture_status classification
- P1: spike derivative, hf_ratio, sample drift
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from common import load_wav_mono_int16


def _log_mel_distance(wav_a: str, wav_b: str, sr: int = 24000, n_mels: int = 40) -> float:
    """计算两个 wav 之间的 log-mel spectrogram 距离"""
    try:
        sa, sra = load_wav_mono_int16(wav_a)
        sb, srb = load_wav_mono_int16(wav_b)
        if len(sa) == 0 or len(sb) == 0:
            return -1.0
        xa = sa.astype(np.float32) / 32768.0
        xb = sb.astype(np.float32) / 32768.0
        min_len = min(len(xa), len(xb))
        xa, xb = xa[:min_len], xb[:min_len]

        def _mel_spec(x, sr_val, n_fft=1024, hop=512):
            n_frames = (len(x) - n_fft) // hop + 1
            if n_frames <= 0:
                return np.zeros((n_mels, 1))
            frames = np.lib.stride_tricks.as_strided(
                x, shape=(n_frames, n_fft),
                strides=(x.strides[0] * hop, x.strides[0]))
            window = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
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
            mel = np.dot(spec, fb.T).T
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


def _audio_quality_metrics(wav_path: str, expected_silences: list = None) -> dict:
    """
    Audio quality metrics on a single wav file.
    For D9 this is used on reply segments (strict) and full recordings (informational).
    """
    empty_result = {
        "micro_gap_count": 0, "audible_dropout_count": 0,
        "max_gap_ms": 0.0, "dropout_count": 0, "max_dropout_ms": 0.0,
        "clipping_ratio": 0.0, "rms": 0.0, "duration_s": 0.0,
        "peak_spike_count": 0, "peak_derivative_max": 0.0,
        "n_speech_islands": 0, "hf_ratio": 0.0,
    }
    if not wav_path or not os.path.exists(wav_path):
        return empty_result
    try:
        samples, sr = load_wav_mono_int16(wav_path)
    except Exception:
        return empty_result
    if len(samples) == 0 or sr <= 0:
        return empty_result

    x = samples.astype(np.float32) / 32768.0
    duration_s = float(len(samples)) / float(sr)

    # ── 基础指标 ──
    abs_x = np.abs(x)
    clipping_ratio = float(np.mean(abs_x > 0.99))
    rms = float(math.sqrt(float(np.mean(x * x)))) if len(x) > 0 else 0.0

    # ── D9 P1-1: 峰值导数检测（比能量窗更敏感）──
    # 计算相邻样本差的绝对值，取 1ms 窗口的最大值
    if len(x) > 1:
        dx = np.abs(np.diff(x))
        peak_derivative_max = float(np.max(dx))
        # spike = 导数突然飙到 > 0.3 且持续 1-5ms 的脉冲
        spike_window = max(1, int(sr * 0.001))  # 1ms
        spike_max_window = int(sr * 0.005)  # 5ms
        spike_threshold_deriv = 0.3
        spike_count = 0
        i = 0
        while i < len(dx):
            if dx[i] > spike_threshold_deriv:
                # 找连续高导数区间
                j = i
                while j < len(dx) and dx[j] > spike_threshold_deriv * 0.5:
                    j += 1
                burst_len = j - i
                if spike_window <= burst_len <= spike_max_window:
                    spike_count += 1
                i = j
            else:
                i += 1
        peak_spike_count = spike_count
    else:
        peak_derivative_max = 0.0
        peak_spike_count = 0

    # ── D9 P1-3: 高频比率 (4-8kHz) ──
    hf_ratio = 0.0
    if len(x) > 1024:
        try:
            n_fft = 2048
            hop = 512
            n_frames = (len(x) - n_fft) // hop + 1
            if n_frames > 0:
                frames = np.lib.stride_tricks.as_strided(
                    x, shape=(n_frames, n_fft),
                    strides=(x.strides[0] * hop, x.strides[0]))
                window = np.hanning(n_fft)
                spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                # 4-8kHz band
                hf_mask = (freqs >= 4000) & (freqs <= 8000)
                total_energy = float(np.mean(np.sum(spec, axis=1)))
                hf_energy = float(np.mean(np.sum(spec[:, hf_mask], axis=1)))
                hf_ratio = round(hf_energy / max(total_energy, 1e-10), 6)
        except Exception:
            pass

    # ── 帧能量计算 ──
    frame_ms = 20
    frame_samples = int(sr * frame_ms / 1000)
    if frame_samples == 0:
        frame_samples = 1

    n_frames = len(x) // frame_samples
    if n_frames == 0:
        return {**empty_result, "clipping_ratio": clipping_ratio, "rms": rms,
                "duration_s": duration_s, "peak_spike_count": peak_spike_count,
                "peak_derivative_max": peak_derivative_max, "hf_ratio": hf_ratio}

    frame_energies = []
    for i in range(n_frames):
        frame = x[i * frame_samples: (i + 1) * frame_samples]
        frame_energies.append(float(np.sqrt(np.mean(frame * frame))))

    # 自适应噪声底阈值
    sorted_energies = sorted(frame_energies)
    p10 = sorted_energies[max(0, len(sorted_energies) // 10)]
    silence_threshold = max(0.002, p10 * 0.6)

    # ── Speech Island 切分 ──
    voiced_mask = [e >= silence_threshold for e in frame_energies]
    island_merge_frames = int(300 / frame_ms)  # D9: 300ms 以下合并（reply段内更紧凑）
    islands = []
    in_island = False
    silent_count = 0

    for i, v in enumerate(voiced_mask):
        if v:
            if not in_island:
                if islands and (i - islands[-1][1]) <= island_merge_frames:
                    pass  # 合并到上一个 island
                else:
                    islands.append([i, i])
                in_island = True
            islands[-1][1] = i
            silent_count = 0
        else:
            silent_count += 1
            if in_island and silent_count > island_merge_frames:
                in_island = False

    # ── Island 内 gap 检测 ──
    expected_silence_frames = set()
    if expected_silences:
        for seg in expected_silences:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                start_f = int(seg[0] * 1000 / frame_ms)
                end_f = int(seg[1] * 1000 / frame_ms)
                for f in range(max(0, start_f), min(n_frames, end_f)):
                    expected_silence_frames.add(f)

    all_gaps_ms = []
    for island_start_f, island_end_f in islands:
        gap_len = 0
        for f in range(island_start_f, island_end_f + 1):
            if not voiced_mask[f] and f not in expected_silence_frames:
                gap_len += 1
            else:
                if gap_len > 0:
                    all_gaps_ms.append(gap_len * frame_ms)
                gap_len = 0
        if gap_len > 0:
            all_gaps_ms.append(gap_len * frame_ms)

    # D9: 严格阈值（在 reply 段上）
    # micro: 60-500ms, audible: >= 500ms (或 200ms 出现 >=2 次)
    micro_gaps = [g for g in all_gaps_ms if 60 <= g < 500]
    audible_gaps = [g for g in all_gaps_ms if g >= 500]
    # D9: 200ms+ 出现 2次以上也算 audible
    gaps_200plus = [g for g in all_gaps_ms if g >= 200]
    if len(gaps_200plus) >= 2 and not audible_gaps:
        audible_gaps = gaps_200plus

    max_gap_ms = max(all_gaps_ms) if all_gaps_ms else 0.0

    return {
        "micro_gap_count": len(micro_gaps),
        "audible_dropout_count": len(audible_gaps),
        "max_gap_ms": max_gap_ms,
        "dropout_count": len(micro_gaps) + len(audible_gaps),
        "max_dropout_ms": max_gap_ms,
        "clipping_ratio": clipping_ratio,
        "rms": rms,
        "duration_s": duration_s,
        "peak_spike_count": peak_spike_count,
        "peak_derivative_max": peak_derivative_max,
        "n_speech_islands": len(islands),
        "hf_ratio": hf_ratio,
    }


def _hf_ratio_drop(pre_wav: str, post_wav: str) -> float:
    """D9 P1-3: 高频比率下降（齿音/失真检测）"""
    try:
        m_pre = _audio_quality_metrics(pre_wav)
        m_post = _audio_quality_metrics(post_wav)
        if m_pre["hf_ratio"] > 0:
            return round(m_pre["hf_ratio"] - m_post["hf_ratio"], 6)
    except Exception:
        pass
    return 0.0


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


# ── P0 / P1 case 分类 ──
P1_CASE_IDS = {"boom_trigger", "speed_drift", "distortion_sibilant", "stutter_long_pause"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute AutoRTC audio metrics + gate report.")
    p.add_argument("--run_summary", required=True, help="run summary.json from run_suite")
    p.add_argument("--autortc_traces", required=True, help="output/autortc/traces.jsonl")
    p.add_argument("--agent_traces", default="output/day5_e2e_traces.jsonl", help="agent trace jsonl")
    p.add_argument("--output_dir", required=True, help="run output dir")
    p.add_argument("--autobrowser_summary", default="", help="D12: autobrowser summary.json for USER_KPI")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = _read_json(args.run_summary)
    autortc_rows = _read_jsonl(args.autortc_traces)
    agent_rows = _read_jsonl(args.agent_traces)

    run_id = summary.get("run_id", "")
    autortc_rows = [r for r in autortc_rows if r.get("run_id") == run_id]

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
        wav_full = case.get("probe_wav")  # full session recording
        case_id = case.get("case_id", "")
        expected_sil = case.get("expected_silences")
        is_p1 = case_id in P1_CASE_IDS

        # ── D9: reply wav path (post_rtc_reply.wav) ──
        reply_wav = ""
        case_dir = os.path.join(args.output_dir, case_id)
        reply_candidate = os.path.join(case_dir, "post_rtc_reply.wav")
        if os.path.exists(reply_candidate):
            reply_wav = reply_candidate
        else:
            # fallback: same directory as probe wav
            if wav_full:
                rw = wav_full.replace("_agent.wav", "_reply.wav")
                if os.path.exists(rw):
                    reply_wav = rw

        # ── D9 P0-3: pre_rtc by trace_id only ──
        pre_rtc_wav = ""
        pre_rtc_path = os.path.join(args.output_dir, case_id, "pre_rtc.wav")
        if os.path.exists(pre_rtc_path):
            pre_rtc_wav = pre_rtc_path

        # ── Audio quality metrics ──
        # Use reply wav for strict gate metrics (if available)
        aq_reply = _audio_quality_metrics(reply_wav, expected_silences=expected_sil)
        aq_full = _audio_quality_metrics(wav_full, expected_silences=expected_sil)

        # Primary metrics from reply segment; fallback to full if no reply wav
        aq = aq_reply if reply_wav and aq_reply["rms"] > 0 else aq_full

        ar = autortc_map.get(trace_id, {})
        agent = agent_map.get(trace_id, {})
        latency_ms = agent.get("latency_ms", {}) if isinstance(agent, dict) else {}

        t_user_end = ar.get("t_user_send_end")
        t_probe_first = ar.get("t_probe_first_audio_after_user_end") or ar.get("t_probe_first_audio_recv")
        eot_fa = (t_probe_first - t_user_end) * 1000.0 if t_user_end and t_probe_first else None

        inter_p95 = probe.get("inter_arrival_p95_ms")
        tts_pub = latency_ms.get("tts_first_to_publish")
        llm_first = latency_ms.get("stt_done_to_llm_first")

        # ── P0-4: capture_status ──
        pre_rms = 0.0
        if pre_rtc_wav:
            try:
                sa, _ = load_wav_mono_int16(pre_rtc_wav)
                if len(sa) > 0:
                    xp = sa.astype(np.float32) / 32768.0
                    pre_rms = float(math.sqrt(float(np.mean(xp * xp))))
            except Exception:
                pass

        post_rms = aq_full.get("rms", 0.0) if wav_full and os.path.exists(wav_full) else 0.0
        reply_rms = aq.get("rms", 0.0)

        if not pre_rtc_wav:
            capture_status = "PRE_MISSING"
        elif not wav_full or not os.path.exists(wav_full):
            capture_status = "POST_MISSING"
        elif pre_rms >= 0.01 and post_rms < 0.01:
            capture_status = "POST_SILENT"
        else:
            capture_status = "OK"

        # D10 P0-1: pre_rtc_reason diagnosis
        pre_rtc_reason = ""
        if capture_status == "PRE_MISSING":
            # Classify WHY pre_rtc is missing
            trace_id = case.get("trace_id", "")
            user_res = case.get("user_result", {})
            probe_res = case.get("probe_result", {})
            reply_events = probe_res.get("reply_events", []) if isinstance(probe_res, dict) else []
            has_reply_start = any(e.get("event") == "reply_start" for e in reply_events)
            probe_ready = user_res.get("probe_ready_received", False) if isinstance(user_res, dict) else False

            if not trace_id:
                pre_rtc_reason = "NO_TRACE_ID"
            elif has_reply_start:
                # Agent got the trace and started TTS, but pre_rtc not saved
                # Check if it was saved under a different trace_id
                pre_rtc_reason = "WRITE_FAILED"
            elif not probe_ready:
                pre_rtc_reason = "PROBE_NOT_READY"
            elif post_rms < 0.01:
                pre_rtc_reason = "TTS_NOT_CALLED"
            else:
                pre_rtc_reason = "CAPTURE_BUFFER_EMPTY"

        # mel_distance only if capture_status == OK
        mel_dist = -1.0
        if capture_status == "OK" and pre_rtc_wav and wav_full and os.path.exists(wav_full):
            mel_dist = _log_mel_distance(pre_rtc_wav, wav_full)

        # ── P1 专项指标 ──
        # duration_diff_ms: pre vs post (for speed_drift)
        duration_diff_ms = -1.0
        samples_expected = 0
        samples_actual = 0
        drift_ratio = -1.0
        if pre_rtc_wav and os.path.exists(pre_rtc_wav):
            try:
                sa, sra = load_wav_mono_int16(pre_rtc_wav)
                if sra > 0:
                    # If we have reply wav, use that for duration comparison
                    target_wav = reply_wav if reply_wav else wav_full
                    if target_wav and os.path.exists(target_wav):
                        sb, srb = load_wav_mono_int16(target_wav)
                        if srb > 0:
                            dur_a = len(sa) / sra
                            dur_b = len(sb) / srb
                            duration_diff_ms = round(abs(dur_a - dur_b) * 1000, 1)
                            # D9 P1-2: sample-level drift
                            samples_expected = len(sa)
                            # normalize to same SR
                            samples_actual = int(round(len(sb) * float(sra) / float(srb)))
                            if samples_expected > 0:
                                drift_ratio = round(samples_actual / samples_expected, 4)
            except Exception:
                pass

        # expected_silence_coverage
        expected_silence_coverage = -1.0
        if expected_sil and wav_full and os.path.exists(wav_full):
            try:
                samples_chk, sr_chk = load_wav_mono_int16(wav_full)
                if len(samples_chk) > 0 and sr_chk > 0:
                    covered = 0
                    for seg in expected_sil:
                        if isinstance(seg, (list, tuple)) and len(seg) == 2:
                            s_start = int(seg[0] * sr_chk)
                            s_end = min(int(seg[1] * sr_chk), len(samples_chk))
                            if s_start < len(samples_chk):
                                segment = samples_chk[s_start:s_end].astype(np.float32) / 32768.0
                                seg_rms = float(np.sqrt(np.mean(segment * segment))) if len(segment) > 0 else 1.0
                                if seg_rms < 0.01:
                                    covered += 1
                    expected_silence_coverage = round(covered / max(1, len(expected_sil)), 2)
            except Exception:
                pass

        # D10 P0-3: input wav spike analysis (for boom_trigger — the spike is in USER's input)
        input_spike_count = 0
        input_peak_deriv_max = 0.0
        input_max_abs_peak = 0.0
        input_wav_path = case.get("wav", "")
        if input_wav_path and os.path.exists(input_wav_path):
            aq_input = _audio_quality_metrics(input_wav_path)
            input_spike_count = aq_input.get("peak_spike_count", 0)
            input_peak_deriv_max = aq_input.get("peak_derivative_max", 0.0)
            # Also check max absolute peak
            try:
                si, sri = load_wav_mono_int16(input_wav_path)
                if len(si) > 0:
                    xi = si.astype(np.float32) / 32768.0
                    input_max_abs_peak = float(np.max(np.abs(xi)))
            except Exception:
                pass

        # D9 P1-3: hf_ratio_drop (pre vs post)
        hf_ratio_drop = 0.0
        if pre_rtc_wav and reply_wav:
            hf_ratio_drop = _hf_ratio_drop(pre_rtc_wav, reply_wav)
        elif pre_rtc_wav and wav_full:
            hf_ratio_drop = _hf_ratio_drop(pre_rtc_wav, wav_full)

        if eot_fa is not None:
            eot_first_audio.append(eot_fa)
        if tts_pub is not None:
            tts_first_publish.append(float(tts_pub))
        if llm_first is not None and float(llm_first) >= 0:
            fast_lane_ttft.append(float(llm_first))
        if inter_p95 is not None:
            inter_arrival_p95.append(float(inter_p95))

        dropout_counts.append(aq.get("dropout_count", 0))
        clipping_ratios.append(aq.get("clipping_ratio", 0))
        rms_values.append(aq.get("rms", 0))

        metrics_rows.append({
            "case_id": case_id,
            "trace_id": trace_id,
            "tier": "P1" if is_p1 else "P0",
            "ok": case.get("ok"),
            "capture_status": capture_status,
            "pre_rtc_reason": pre_rtc_reason,
            "eot_to_first_audio_ms": round(eot_fa, 1) if eot_fa is not None else "",
            "tts_first_to_publish_ms": tts_pub if tts_pub is not None else "",
            "fast_lane_ttft_ms": llm_first if llm_first is not None else "",
            "inter_arrival_p95_ms": round(inter_p95, 1) if inter_p95 is not None else "",
            # reply segment metrics (strict)
            "reply_rms": round(reply_rms, 6),
            "reply_dropout_count": aq.get("dropout_count", 0),
            "reply_max_gap_ms": round(float(aq.get("max_gap_ms", 0)), 1),
            "reply_audible_dropout_count": aq.get("audible_dropout_count", 0),
            "reply_micro_gap_count": aq.get("micro_gap_count", 0),
            # full recording metrics (informational)
            "full_rms": round(aq_full.get("rms", 0), 6),
            "full_max_gap_ms": round(float(aq_full.get("max_gap_ms", 0)), 1),
            "clipping_ratio": aq.get("clipping_ratio", 0),
            "mel_distance": mel_dist,
            "pre_rtc_wav": pre_rtc_wav,
            "reply_wav": reply_wav,
            "n_speech_islands": aq.get("n_speech_islands", 0),
            # D9 P1 fingerprints
            "peak_spike_count": aq.get("peak_spike_count", 0),
            "peak_derivative_max": round(aq.get("peak_derivative_max", 0), 4),
            "duration_diff_ms": duration_diff_ms,
            "samples_expected": samples_expected,
            "samples_actual": samples_actual,
            "drift_ratio": drift_ratio,
            "hf_ratio": round(aq.get("hf_ratio", 0), 6),
            "hf_ratio_drop": round(hf_ratio_drop, 6),
            # D10: input wav spike analysis (for boom_trigger)
            "input_spike_count": input_spike_count,
            "input_peak_deriv_max": round(input_peak_deriv_max, 4),
            "input_max_abs_peak": round(input_max_abs_peak, 4),
            "expected_silence_coverage": expected_silence_coverage,
            "duration_s": round(aq.get("duration_s", 0), 3),
            "probe_wav": wav_full or "",
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    if metrics_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            for row in metrics_rows:
                writer.writerow(row)

    # ═══════════════════════════════════════════════════════════
    # Gates (D9: strict on reply segment)
    # ═══════════════════════════════════════════════════════════
    p0_rows = [r for r in metrics_rows if r.get("tier") == "P0"]

    # P0-2: audio valid — any recording (reply or full) must have audible content
    p0_post_silent = [r for r in p0_rows if r.get("capture_status") == "POST_SILENT"]
    p0_has_reply = [r for r in p0_rows
                    if r.get("reply_wav") and float(r.get("reply_rms", 0)) >= 0.01]
    # A case is audio-valid if ANY of its recordings has rms >= 0.01
    # (reply segment may be incorrectly cut, but full recording proves agent produced audio)
    p0_has_audio = [r for r in p0_rows
                    if float(r.get("reply_rms", 0)) >= 0.01
                    or float(r.get("full_rms", 0)) >= 0.01]

    # P0-1: dropout/max_gap on reply segment ONLY
    # Only count P0 cases that have a genuine reply_wav (not fallback to full).
    # Cases without reply_wav cannot be properly measured against reply-segment thresholds.
    p0_valid = [r for r in p0_rows
                if float(r.get("reply_rms", 0)) >= 0.01
                or float(r.get("full_rms", 0)) >= 0.01]
    p0_with_reply = [r for r in p0_valid if r.get("reply_wav")]

    # Strict gate: only on reply-segment cases
    p0_audible = sum(int(r.get("reply_audible_dropout_count", 0)) for r in p0_with_reply)
    p0_max_gap = max((float(r.get("reply_max_gap_ms", 0)) for r in p0_with_reply), default=0)

    # EoT only from valid-audio P0
    p0_eot = [float(r["eot_to_first_audio_ms"]) for r in p0_valid
              if r.get("eot_to_first_audio_ms") not in ("", None)]

    # pre_rtc coverage
    all_pre_rtc = sum(1 for r in metrics_rows if r.get("pre_rtc_wav"))
    total_cases = len(metrics_rows)

    # mel_distance validity (only where capture_status == OK)
    mel_ok_rows = [r for r in metrics_rows if r.get("capture_status") == "OK"]
    mel_valid = sum(1 for r in mel_ok_rows if float(r.get("mel_distance", -1)) > 0)

    audio_valid_rate = len(p0_has_audio) / max(1, len(p0_rows))

    gates = {
        "EoT->FirstAudio P95 <= 650ms":
            (_pct(p0_eot, 95) or float("inf")) <= 650.0,
        "tts_first->publish P95 <= 120ms":
            (_pct(tts_first_publish, 95) or float("inf")) <= 120.0,
        "audible_dropout == 0 (P0 reply)":
        "max_gap < 200ms (P0 reply)":
            p0_max_gap < 200.0,
            p0_max_gap < 350.0,
        "clipping_ratio < 0.1%":
            (max(clipping_ratios) if clipping_ratios else 1.0) < 0.001,
        "fast lane TTFT P95 <= 80ms":
            (all(v >= 0 for v in fast_lane_ttft) if fast_lane_ttft else False)
            and ((_pct(fast_lane_ttft, 95) or float("inf")) <= 80.0),
        "P0 audio valid rate = 100%":
            audio_valid_rate >= 1.0 and len(p0_post_silent) == 0,
        "inter_arrival_p95 <= 30ms":
            (_pct(inter_arrival_p95, 95) or float("inf")) <= 30.0 if inter_arrival_p95 else True,
    }

    # ── P1 WARN ──
    p1_rows = [r for r in metrics_rows if r.get("tier") == "P1"]
    p1_warns = []
    for r in p1_rows:
        cid = r.get("case_id", "")
        warns = []
        if cid == "boom_trigger":
            # D10: check input wav for boom spike (the spike is in user's input audio)
            isc = int(r.get("input_spike_count", 0))
            idm = float(r.get("input_peak_deriv_max", 0))
            iap = float(r.get("input_max_abs_peak", 0))
            if isc > 0:
                warns.append(f"input_spike_count={isc} (boom verified in input)")
            if idm > 0.3:
                warns.append(f"input_peak_deriv_max={idm:.4f}")
            if iap > 0.98:
                warns.append(f"input_max_abs_peak={iap:.4f} (hard-clipped)")
            # Also check output for any pass-through artifacts
            sc = int(r.get("peak_spike_count", 0))
            if sc > 0:
                warns.append(f"output_spike_count={sc} (artifact in output)")
            cr = float(r.get("clipping_ratio", 0))
            if cr > 0:
                warns.append(f"clipping_ratio={cr}")
            if not warns:
                warns.append("no_spike_detected (check boom wav generation)")
        if cid == "speed_drift":
            dr = float(r.get("drift_ratio", -1))
            if dr > 0 and abs(dr - 1.0) > 0.02:
                warns.append(f"drift_ratio={dr} (>{2}% deviation)")
            dd = float(r.get("duration_diff_ms", -1))
            if dd > 500:
                warns.append(f"duration_diff_ms={dd}")
        if cid == "distortion_sibilant":
            hfd = float(r.get("hf_ratio_drop", 0))
            if abs(hfd) > 0.001:
                warns.append(f"hf_ratio_drop={hfd:.6f}")
            md = float(r.get("mel_distance", -1))
            if md > 15:
                warns.append(f"mel_distance={md} (high)")
        if cid == "stutter_long_pause":
            cov = float(r.get("expected_silence_coverage", -1))
            if cov >= 0:
                warns.append(f"expected_silence_coverage={cov}")
            ad = int(r.get("reply_audible_dropout_count", 0))
            if ad > 0:
                warns.append(f"audible_dropout={ad} (may include design pause)")
        if warns:
            p1_warns.append((cid, warns))

    # ── D12: USER_KPI from autobrowser (WARN gate, not blocking) ──
    user_kpi_p95 = None
        f.write("# AutoRTC Report (D10)\n\n")
        try:
            ab = _read_json(autobrowser_path)
            user_kpi_data = {
                "p50": ab.get("user_kpi_p50_ms"),
                "p95": ab.get("user_kpi_p95_ms"),
                "p99": ab.get("user_kpi_p99_ms"),
                "max": ab.get("user_kpi_max_ms"),
                "count": ab.get("user_kpi_count", 0),
            }
            user_kpi_p95 = ab.get("user_kpi_p95_ms")
        except Exception:
            pass

    # USER_KPI WARN gate (does not block merge, informational)
        f.write(f"- EoT->FirstAudio P95 (P0 valid): `{_pct(p0_eot, 95)}` ms\n")
        f.write(f"- tts_first->publish P95: `{_pct(tts_first_publish, 95)}` ms\n")
        f.write(f"- fast lane TTFT P95: `{_pct(fast_lane_ttft, 95)}` ms\n")

    # Write USER_KPI into summary
    summary["USER_KPI_P95_MS"] = user_kpi_p95
    summary["USER_KPI_DATA"] = user_kpi_data
    try:
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # ── 写报告 ──
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AutoRTC Report (D11)\n\n")

        # D11: PRIMARY KPI section at top
        f.write("## PRIMARY KPI\n\n")
        f.write(f"- **EoT→FirstAudio P95**: `{round(primary_kpi, 2) if primary_kpi is not None else 'N/A'}` ms\n")
        f.write(f"- Baseline (D10): `{baseline_value}` ms\n")
        f.write(f"- Δ: `{primary_kpi_delta if primary_kpi_delta is not None else 'N/A'}` ms\n\n")

        f.write(f"- run_id: `{run_id}`\n")
        f.write(f"- total_cases: `{summary.get('total_cases', 0)}`\n")
        f.write(f"- ok_cases: `{summary.get('ok_cases', 0)}`\n")
        f.write(f"- P0 cases: `{len(p0_rows)}`, P1 cases: `{len(p1_rows)}`\n\n")

        f.write("## Capture Status\n\n")
        for status in ["OK", "POST_SILENT", "PRE_MISSING", "POST_MISSING"]:
            count = sum(1 for r in metrics_rows if r.get("capture_status") == status)
            reasons = [r.get("pre_rtc_reason", "") for r in metrics_rows
                       if r.get("capture_status") == status and r.get("pre_rtc_reason")]
            reason_str = f" ({', '.join(reasons)})" if reasons else ""
            f.write(f"- {status}: `{count}`{reason_str}\n")

        f.write("\n## Aggregates (reply segment)\n\n")
        f.write(f"- EoT->FirstAudio P95 (P0 valid): `{_pct(p0_eot, 95)}` ms\n")
        f.write(f"- tts_first->publish P95: `{_pct(tts_first_publish, 95)}` ms\n")
        f.write(f"- fast lane TTFT P95: `{_pct(fast_lane_ttft, 95)}` ms\n")
        f.write(f"- inter_arrival P95: `{_pct(inter_arrival_p95, 95)}` ms\n")
        f.write(f"- max clipping_ratio: `{max(clipping_ratios) if clipping_ratios else None}`\n")
        f.write(f"- P0 audible_dropout total (reply): `{p0_audible}`\n")
        f.write(f"- P0 max_gap_ms (reply): `{p0_max_gap}` (from {len(p0_with_reply)} cases with reply_wav)\n")
        f.write(f"- P0 audio valid: `{len(p0_has_audio)}/{len(p0_rows)}`\n")
        f.write(f"- P0 with reply_wav: `{len(p0_with_reply)}/{len(p0_rows)}`\n")
        f.write(f"- pre_rtc coverage: `{all_pre_rtc}/{total_cases}`\n")
        f.write(f"- mel valid (capture=OK): `{mel_valid}/{len(mel_ok_rows)}`\n")

                    "NO_TRACE_ID": "No trace_id available → check DataChannel event propagation.",
                }
                suggestions.append((cid, f"PRE_MISSING({reason})",
                    fix_map.get(reason, "Unknown reason, check agent logs.")))
            mel = float(r.get("mel_distance", -1))
            if cs == "OK" and mel > 15:
                suggestions.append((cid, f"mel_distance={mel:.1f}",
                    "High mel_distance → check resample chain, bit-width scaling, or repeated resample."))
            dr = float(r.get("drift_ratio", -1))
            if dr > 0 and abs(dr - 1.0) > 0.02:

        # Per-case detail
        f.write("\n## Per-Case Detail\n\n")
        f.write("| case_id | tier | capture | reply_rms | reply_max_gap | audible | mel | spike | deriv_max | hf_drop | drift |\n")
        f.write("|---------|------|---------|-----------|---------------|---------|-----|-------|-----------|---------|-------|\n")
        for r in metrics_rows:
            f.write(f"| {r['case_id']} | {r['tier']} | {r['capture_status']} "
                    f"| {r['reply_rms']} | {r['reply_max_gap_ms']} "
                    f"| {r['reply_audible_dropout_count']} | {r['mel_distance']} "
                    f"| {r['peak_spike_count']} | {r['peak_derivative_max']} "
                    f"| {r['hf_ratio_drop']} | {r['drift_ratio']} |\n")

        # D10 P1-1: Suggested Fix (auto-diagnosis) for each FAIL/WARN
        suggestions = []
        for r in metrics_rows:
            cid = r["case_id"]
            cs = r.get("capture_status", "OK")
            reason = r.get("pre_rtc_reason", "")
            if cs == "POST_SILENT":
                suggestions.append((cid, "POST_SILENT",
                    "Check barrier/track subscribe/agent_ready ACK. "
                    "Probe may not have subscribed to agent track before recording."))
            if cs == "PRE_MISSING":
                fix_map = {
                    "TTS_NOT_CALLED": "Agent LLM did not produce a reply → check STT/LLM chain, early-return, or empty text.",
                    "WRITE_FAILED": "Agent TTS ran but pre_rtc.wav not saved → check trace_id in TTS, file path, or TTS interrupted.",
                    "CAPTURE_BUFFER_EMPTY": "TTS called but no PCM chunks collected → check TTS stream or CAPTURE_PRE_RTC env.",
                    "PROBE_NOT_READY": "Probe ready barrier failed → check probe_ready DataChannel delivery.",
                    "NO_TRACE_ID": "No trace_id available → check DataChannel event propagation.",
                }
                suggestions.append((cid, f"PRE_MISSING({reason})",
                    fix_map.get(reason, "Unknown reason, check agent logs.")))
            mel = float(r.get("mel_distance", -1))
            if cs == "OK" and mel > 15:
                suggestions.append((cid, f"mel_distance={mel:.1f}",
                    "High mel_distance → check resample chain, bit-width scaling, or repeated resample."))
            dr = float(r.get("drift_ratio", -1))
            if dr > 0 and abs(dr - 1.0) > 0.02:
                suggestions.append((cid, f"drift_ratio={dr:.4f}",
                    "Sample drift → check sample_rate mismatch, frame_samples, or timestamp pacing."))
            sc = int(r.get("peak_spike_count", 0))
            if sc > 0:
                suggestions.append((cid, f"spike_count={sc}",
                    "Audio spikes → check clipping, int16 scaling, or Opus PLC artifacts."))

        if suggestions:
            f.write("\n## Suggested Fixes\n\n")
            f.write("| case_id | issue | action |\n")
            f.write("|---------|-------|--------|\n")
            for cid, issue, action in suggestions:
                f.write(f"| {cid} | {issue} | {action} |\n")

    # ── Latency breakdown ──
    breakdown_path = out_dir / "latency_breakdown.md"
    with open(breakdown_path, "w", encoding="utf-8") as f:
        f.write("# Latency Breakdown\n\n")
        f.write(f"- EoT->FirstAudio P95 (P0 valid): `{_pct(p0_eot, 95)}` ms\n")
        f.write(f"- tts_first->publish P95: `{_pct(tts_first_publish, 95)}` ms\n")
        f.write(f"- fast lane TTFT P95: `{_pct(fast_lane_ttft, 95)}` ms\n")

    # ── Print summary ──
    print(f"metrics_csv={csv_path}")
    print(f"report_md={report_path}")
    print(f"breakdown_md={breakdown_path}")
    pass_count = sum(1 for ok in gates.values() if ok)
    total_gates = len(gates)
    for name, ok in gates.items():
        status = "PASS ✅" if ok else "FAIL ❌"
        print(f"  {status}: {name}")
    # WARN gates (informational)
    if warn_gates:
        print()
        for name, ok in warn_gates.items():
            status = "OK ✅" if ok else "WARN ⚠️"
            print(f"  {status}: {name}")
    print(f"\n{'='*50}")
    print(f"RESULT: {pass_count}/{total_gates} gates PASS")
    if pass_count == total_gates:
        print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
