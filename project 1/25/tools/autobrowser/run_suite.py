#!/usr/bin/env python3
"""
D12: AutoBrowser Run Suite — Real Browser WYSIWYG regression.

Uses Playwright to launch Chromium with fake media (--use-file-for-fake-audio-capture),
open webrtc_test.html, and collect browser-side traces + recorded audio.

Usage:
    python3 tools/autobrowser/run_suite.py --mode fast --cases_json tools/autortc/cases/all_cases.json
    python3 tools/autobrowser/run_suite.py --mode fast --cases_json tools/autortc/cases/mini_cases.json --net wifi_good
"""
import argparse
import asyncio
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import requests

# ── Reuse common utilities from autortc ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTORTC_DIR = os.path.join(SCRIPT_DIR, "..", "autortc")
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, AUTORTC_DIR)
from common import ensure_parent, wav_duration_s, write_json

# ── Net profile configs (P0-4) ──
NET_PROFILES = {
    "wifi_good": {
        "description": "Baseline WiFi: no impairment",
        "delay_ms": 0,
        "jitter_ms": 0,
        "loss_pct": 0,
    },
    "4g_ok": {
        "description": "4G: 60ms RTT + 20ms jitter + 0.5% loss",
        "delay_ms": 30,   # one-way → RTT ~60ms
        "jitter_ms": 20,
        "loss_pct": 0.5,
    },
    "bad_wifi": {
        "description": "Bad WiFi: 100ms RTT + 40ms jitter + 2% loss",
        "delay_ms": 50,
        "jitter_ms": 40,
        "loss_pct": 2.0,
    },
}


def _ts_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _fetch_token(token_api: str, room: str, identity: str) -> tuple:
    """Fetch LiveKit token from token server."""
    r = requests.get(token_api, params={"room": room, "identity": identity}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["token"], data["url"]


def _load_cases_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for idx, item in enumerate(data.get("cases", [])):
        case_id = item.get("case_id") or f"case_{idx:03d}"
        wav = item.get("wav", "")
        tier = item.get("tier", "P0")
        if wav:
            cases.append({"case_id": case_id, "wav": wav, "tier": tier, **item})
    return cases


def _apply_netem(interface: str, profile_name: str) -> bool:
    """Apply tc netem profile. Returns True if applied."""
    profile = NET_PROFILES.get(profile_name)
    if not profile or profile_name == "wifi_good":
        return False  # baseline = no impairment
    try:
        # Clear existing rules
        subprocess.run(["tc", "qdisc", "del", "dev", interface, "root"],
                       capture_output=True, check=False)
        # Apply netem
        cmd = [
            "tc", "qdisc", "add", "dev", interface, "root", "netem",
            "delay", f"{profile['delay_ms']}ms", f"{profile['jitter_ms']}ms",
            "loss", f"{profile['loss_pct']}%",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[netem] Applied {profile_name}: {profile['description']}")
        return True
    except Exception as e:
        print(f"[netem] WARNING: Failed to apply {profile_name}: {e}")
        return False


def _clear_netem(interface: str):
    """Remove tc netem rules."""
    try:
        subprocess.run(["tc", "qdisc", "del", "dev", interface, "root"],
                       capture_output=True, check=False)
    except Exception:
        pass


SILENCE_PAD_S = 10  # seconds of silence appended after speech

GT_EOT_THRESHOLD = 0.01   # RMS threshold for speech vs silence in ground-truth analysis
GT_EOT_SILENCE_MS = 300   # minimum silence duration (ms) to declare ground-truth EoT


def _analyze_wav_gt_eot(wav_path: str, threshold: float = GT_EOT_THRESHOLD,
                         min_silence_ms: float = GT_EOT_SILENCE_MS) -> float:
    """
    D14 P0-2: Offline analysis of WAV to find ground-truth speech end time.
    Returns gt_speech_end_ms (ms from start of audio to last speech energy above threshold).
    """
    import numpy as np
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)[:, 0]

    frame_len = int(sr * 0.01)  # 10ms frames
    last_speech_frame = 0
    for i in range(0, len(audio) - frame_len, frame_len):
        frame = audio[i:i + frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > threshold:
            last_speech_frame = i + frame_len

    gt_speech_end_ms = (last_speech_frame / sr) * 1000.0
    return gt_speech_end_ms


def _prepare_chromium_wav(src_wav: str, dst_wav: str, silence_pad_s: float = SILENCE_PAD_S) -> str:
    """
    D13: Generate chromium_input.wav = original_speech + long_silence (48k/16bit).

    Chromium's --use-file-for-fake-audio-capture loops the file, so we append
    enough silence that the recording window ends *within* the silence portion,
    preventing loop-back. This lets monitorMic naturally detect EoT via energy
    drop without needing programmatic mic mute.
    """
    try:
        import numpy as np
        with wave.open(src_wav, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

        target_sr = 48000
        if sr != target_sr:
            n_dst = int(round(len(audio) * target_sr / sr))
            xp = np.arange(len(audio), dtype=np.float64)
            xnew = np.linspace(0, len(audio) - 1, n_dst, dtype=np.float64)
            audio = np.clip(np.interp(xnew, xp, audio.astype(np.float32)),
                            -32768, 32767).astype(np.int16)
            sr = target_sr

        # Append silence so Chromium won't loop back into speech
        silence_samples = int(silence_pad_s * sr)
        padded = np.concatenate([audio, np.zeros(silence_samples, dtype=np.int16)])

        ensure_parent(dst_wav)
        with wave.open(dst_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(padded.tobytes())

        speech_dur = len(audio) / sr
        total_dur = len(padded) / sr
        return dst_wav
    except Exception as e:
        print(f"  WARN: WAV preparation failed: {e}, using original")
        return src_wav


async def _run_browser_case(
    page_url: str,
    case_wav_48k: str,
    token: str,
    lk_url: str,
    room: str,
    identity: str,
    trace_id: str,
    case_id: str,
    turn_id: str,
    record_seconds: int,
    case_dir: str,
    headless: bool = True,
    gt_speech_end_ms: float = 0.0,
) -> dict:
    """
    Launch Chromium via Playwright, open webrtc_test.html with auto params,
    wait for traces, collect recording + browser_trace.json.
    """
    from playwright.async_api import async_playwright

    result = {
        "ok": False,
        "joined": False,
        "has_reply_audio": False,
        "browser_traces": [],
        "error": None,
    }

    # Build URL with auto params
    from urllib.parse import urlencode, quote
    params = {
        "auto": "1",
        "lk_token": token,
        "lk_url": lk_url,
        "room": room,
        "identity": identity,
        "trace_id": trace_id,
        "case_id": case_id,
        "turn_id": turn_id,
        "auto_disconnect_s": str(record_seconds),
        "gt_speech_end_ms": str(gt_speech_end_ms),
    }
    full_url = f"{page_url}?{urlencode(params)}"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={case_wav_48k}",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies",
            ],
        )
        context = await browser.new_context(
            permissions=["microphone"],
            ignore_https_errors=True,
        )
        page = await context.new_page()

        # Collect console logs
        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

        try:
            await page.goto(full_url, wait_until="domcontentloaded", timeout=15000)

            # Wait for join
            try:
                await page.wait_for_function(
                    "window.__autobrowser_joined === true",
                    timeout=20000,
                )
                result["joined"] = True
            except Exception as e:
                result["error"] = f"Join timeout: {e}"
                return result

            # D13: No more mic mute + resetForMeasurement.
            # The input WAV now has 10s silence appended, so Chromium won't
            # loop back into speech within the recording window. monitorMic
            # naturally detects EoT via energy drop. This gives us honest
            # USER_KPI values (including negative = talk-over).

            # Wait for auto-disconnect (remaining recording period)
            try:
                await page.wait_for_function(
                    "window.__autobrowser_done === true",
                    timeout=(record_seconds + 15) * 1000,
                )
            except Exception:
                pass

            # Collect browser traces
            traces = await page.evaluate("window.__autobrowser_traces || []")
            result["browser_traces"] = traces

            # Check if we got reply audio
            has_blob = await page.evaluate("!!window.__autobrowser_recording_blob")
            result["has_reply_audio"] = has_blob

            # Save recording via page-side download
            if has_blob:
                try:
                    b64_data = await page.evaluate("""
                        async () => {
                            const blob = window.__autobrowser_recording_blob;
                            if (!blob) return null;
                            const buffer = await blob.arrayBuffer();
                            const bytes = new Uint8Array(buffer);
                            let binary = '';
                            for (let i = 0; i < bytes.length; i++) {
                                binary += String.fromCharCode(bytes[i]);
                            }
                            return btoa(binary);
                        }
                    """)
                    if b64_data:
                        raw = base64.b64decode(b64_data)
                        webm_path = os.path.join(case_dir, "post_browser_reply.webm")
                        ensure_parent(webm_path)
                        with open(webm_path, "wb") as f:
                            f.write(raw)
                        result["recording_webm"] = webm_path
                        result["recording_bytes"] = len(raw)

                        # Convert webm -> wav using ffmpeg if available
                        wav_path = os.path.join(case_dir, "post_browser_reply.wav")
                        try:
                            subprocess.run(
                                ["ffmpeg", "-y", "-i", webm_path, "-ar", "48000",
                                 "-ac", "1", "-sample_fmt", "s16", wav_path],
                                capture_output=True, timeout=30, check=True,
                            )
                            result["recording_wav"] = wav_path
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            pass
                except Exception as e:
                    result["error"] = f"Recording save error: {e}"

            # Check for errors
            error = await page.evaluate("window.__autobrowser_error")
            if error:
                result["error"] = error

            result["ok"] = result["joined"] and (len(traces) > 0 or result["has_reply_audio"])

        except Exception as e:
            result["error"] = str(e)
        finally:
            # Save console logs
            log_path = os.path.join(case_dir, "browser_console.log")
            ensure_parent(log_path)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(console_logs))

            await context.close()
            await browser.close()

    return result


def _delete_room(room: str, lk_url: str):
    """Delete LiveKit room after case completes."""
    try:
        import asyncio as _aio
        from livekit import api as _lk_api
        def _do_delete():
            async def _inner():
                lk = _lk_api.LiveKitAPI(
                    url=lk_url,
                    api_key=os.environ.get("LIVEKIT_API_KEY", "API7fj35wGLumtc"),
                    api_secret=os.environ.get("LIVEKIT_API_SECRET", "WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B"),
                )
                await lk.room.delete_room(_lk_api.DeleteRoomRequest(room=room))
                await lk.aclose()
            _aio.run(_inner())
        _do_delete()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D12 AutoBrowser: real-browser WYSIWYG regression")
    p.add_argument("--mode", default="fast", choices=["fast", "nightly"],
                   help="fast=per-case room, nightly=multi-turn")
    p.add_argument("--cases_json", required=True, help="path to cases JSON")
    p.add_argument("--page_url", default="", help="webrtc_test.html URL (auto-detect from token_server)")
    p.add_argument("--token_api", default="http://127.0.0.1:9090/api/token",
                   help="token server API")
    p.add_argument("--room_prefix", default="autobrowser", help="room name prefix")
    p.add_argument("--identity", default="browser-user-1", help="browser user identity")
    p.add_argument("--run_id", default="", help="optional run_id")
    p.add_argument("--output_root", default="output/autobrowser", help="output directory")
    p.add_argument("--record_s", type=int, default=25,
                   help="per-case recording window (seconds)")
    p.add_argument("--headless", type=int, default=1, help="1=headless, 0=visible browser")
    p.add_argument("--net", default="wifi_good", choices=list(NET_PROFILES.keys()),
                   help="network profile for tc netem")
    p.add_argument("--net_interface", default="eth0", help="network interface for netem")
    p.add_argument("--with_metrics", type=int, default=1, help="1=run audio_metrics after suite")
    p.add_argument("--baseline_summary", default="", help="D11: golden baseline summary.json")
    p.add_argument("--inter_case_wait_s", type=int, default=20,
                   help="wait between cases for agent pool recovery")
    p.add_argument("--p0_only", type=int, default=0, help="1=only run P0 cases")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or _ts_run_id()
    net_suffix = f"_{args.net}" if args.net != "wifi_good" else ""
    run_dir = os.path.join(args.output_root, f"{run_id}{net_suffix}")
    os.makedirs(run_dir, exist_ok=True)

    # Auto-detect page URL from token_server
    page_url = args.page_url
    if not page_url:
        base = args.token_api.rsplit("/api/", 1)[0]
        page_url = f"{base}/webrtc_test.html"

    # Load cases
    cases = _load_cases_json(args.cases_json)
    if args.p0_only:
        cases = [c for c in cases if c.get("tier", "P0") == "P0"]
    if not cases:
        print("ERROR: No cases found", file=sys.stderr)
        return 1

    print(f"[AutoBrowser] run_id={run_id} mode={args.mode} cases={len(cases)} net={args.net}")
    print(f"[AutoBrowser] page_url={page_url}")
    print(f"[AutoBrowser] output={run_dir}")

    # Prepare WAV cache dir for 48k conversions
    wav_cache_dir = os.path.join(run_dir, ".wav_cache")
    os.makedirs(wav_cache_dir, exist_ok=True)

    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "net_profile": args.net,
        "page_url": page_url,
        "cases": [],
    }

    # Apply network profile
    netem_applied = False
    if args.net != "wifi_good":
        netem_applied = _apply_netem(args.net_interface, args.net)
        if not netem_applied:
            print(f"[netem] NOTE: Running with label '{args.net}' but netem not applied "
                  f"(need NET_ADMIN or --cap-add=NET_ADMIN)")
        summary["netem_actually_applied"] = netem_applied

    for idx, case in enumerate(cases):
        case_id = case["case_id"]
        wav_path = case["wav"]
        tier = case.get("tier", "P0")

        # Per-case room
        case_room = f"{args.room_prefix}-{case_id}-{run_id[-6:]}"
        trace_id = f"{run_id}-{case_id}-turn-{idx:03d}"

        case_dir = os.path.join(run_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        # Convert wav to 48k for Chromium fake capture
        wav_48k = os.path.join(wav_cache_dir, f"{case_id}_48k.wav")
        wav_48k = _prepare_chromium_wav(wav_path, wav_48k)

        # D14 P0-2: Offline ground-truth EoT from original WAV
        gt_speech_end_ms = _analyze_wav_gt_eot(wav_path)

        # Get token
        try:
            token, lk_url = _fetch_token(args.token_api, case_room, args.identity)
        except Exception as e:
            print(f"[{idx+1}/{len(cases)}] {case_id} TOKEN FAILED: {e}")
            summary["cases"].append({
                "case_id": case_id, "tier": tier, "ok": False,
                "error": f"token_fetch: {e}", "trace_id": trace_id,
            })
            continue

        # Calculate recording time: wav duration + padding
        wav_dur = wav_duration_s(wav_path)
        rec_s = max(args.record_s, int(wav_dur) + 15)

        print(f"[{idx+1}/{len(cases)}] {case_id} room={case_room} rec={rec_s}s tier={tier}")
        t_start = time.time()

        # Run browser case
        browser_result = asyncio.run(_run_browser_case(
            page_url=page_url,
            case_wav_48k=wav_48k,
            token=token,
            lk_url=lk_url,
            room=case_room,
            identity=args.identity,
            trace_id=trace_id,
            case_id=case_id,
            turn_id=str(idx),
            record_seconds=rec_s,
            case_dir=case_dir,
            headless=bool(args.headless),
            gt_speech_end_ms=gt_speech_end_ms,
        ))

        t_end = time.time()
        elapsed = t_end - t_start

        # Save browser_trace.json
        trace_path = os.path.join(case_dir, "browser_trace.json")
        write_json(trace_path, {
            "trace_id": trace_id,
            "case_id": case_id,
            "turn_id": str(idx),
            "tier": tier,
            "browser_traces": browser_result.get("browser_traces", []),
            "joined": browser_result.get("joined", False),
            "has_reply_audio": browser_result.get("has_reply_audio", False),
            "recording_wav": browser_result.get("recording_wav", ""),
            "recording_webm": browser_result.get("recording_webm", ""),
            "recording_bytes": browser_result.get("recording_bytes", 0),
            "error": browser_result.get("error"),
            "elapsed_s": elapsed,
            "net_profile": args.net,
        })

        # D13: Extract USER_KPI — prefer last non-talk-over trace (full utterance)
        traces = browser_result.get("browser_traces", [])
        user_kpi_raw_ms = None
        user_kpi_clamped_ms = None
        is_talk_over = False
        best = None
        if traces:
            for t in traces:
                if t.get("user_kpi_raw_ms") is not None:
                    if not t.get("is_talk_over", False):
                        best = t
                    elif best is None:
                        best = t
            if best:
                user_kpi_raw_ms = best["user_kpi_raw_ms"]
                user_kpi_clamped_ms = best.get("user_kpi_ms")
                is_talk_over = best.get("is_talk_over", False)

        # D14 P0-2: Extract GT EoT fields from the SAME best trace used for USER_KPI
        browser_eot_lag_ms = None
        is_talk_over_gt = None
        # D15 P0-1: GT-based USER_KPI (primary metric)
        user_kpi_gt_raw_ms = None
        user_kpi_gt_clamped_ms = None
        if best:
            browser_eot_lag_ms = best.get("browser_eot_lag_ms")
            is_talk_over_gt = best.get("is_talk_over_gt")
            user_kpi_gt_raw_ms = best.get("user_kpi_gt_raw_ms")
            user_kpi_gt_clamped_ms = best.get("user_kpi_gt_clamped_ms")

        status = "PASS" if browser_result["ok"] else "FAIL"
        join_s = "Y" if browser_result["joined"] else "N"
        audio_s = "Y" if browser_result["has_reply_audio"] else "N"
        if user_kpi_raw_ms is not None:
            to_flag = " TALK-OVER" if is_talk_over else ""
            kpi_str = f"raw={user_kpi_raw_ms}ms{to_flag}"
        else:
            kpi_str = "N/A"
        print(f"  {status} join={join_s} audio={audio_s} {kpi_str} ({elapsed:.1f}s)")
        if browser_result.get("error"):
            print(f"  ERROR: {browser_result['error']}")

        case_result = {
            "case_id": case_id,
            "tier": tier,
            "ok": browser_result["ok"],
            "joined": browser_result["joined"],
            "has_reply_audio": browser_result["has_reply_audio"],
            "user_kpi_raw_ms": user_kpi_raw_ms,
            "user_kpi_ms": user_kpi_clamped_ms,
            "is_talk_over": is_talk_over,
            "user_kpi_gt_raw_ms": user_kpi_gt_raw_ms,
            "user_kpi_gt_clamped_ms": user_kpi_gt_clamped_ms,
            "is_talk_over_gt": is_talk_over_gt,
            "trace_id": trace_id,
            "room": case_room,
            "elapsed_s": elapsed,
            "error": browser_result.get("error"),
            "recording_wav": browser_result.get("recording_wav", ""),
            "browser_trace_count": len(traces),
            "gt_speech_end_ms": gt_speech_end_ms,
            "browser_eot_lag_ms": browser_eot_lag_ms,
        }
        summary["cases"].append(case_result)

        # Clean up room
        _delete_room(case_room, lk_url)

        # Inter-case wait
        if idx < len(cases) - 1:
            time.sleep(args.inter_case_wait_s)

    # Clear netem
    if netem_applied:
        _clear_netem(args.net_interface)

    # Clean up wav cache
    shutil.rmtree(wav_cache_dir, ignore_errors=True)

    # Summary stats
    total = len(summary["cases"])
    ok = sum(1 for c in summary["cases"] if c.get("ok"))
    joined = sum(1 for c in summary["cases"] if c.get("joined"))
    has_audio = sum(1 for c in summary["cases"] if c.get("has_reply_audio"))

    summary["total_cases"] = total
    summary["ok_cases"] = ok
    summary["joined_cases"] = joined
    summary["has_audio_cases"] = has_audio

    # D13: USER_KPI aggregates — raw (honest) + clamped + talk-over
    import numpy as np
    raw_vals = [c["user_kpi_raw_ms"] for c in summary["cases"]
                if c.get("user_kpi_raw_ms") is not None]
    clamped_vals = [c["user_kpi_ms"] for c in summary["cases"]
                    if c.get("user_kpi_ms") is not None]
    talk_overs = [c for c in summary["cases"] if c.get("is_talk_over")]
    talk_over_raw = [abs(c["user_kpi_raw_ms"]) for c in talk_overs
                     if c.get("user_kpi_raw_ms") is not None]

    if raw_vals:
        arr_raw = np.array(raw_vals, dtype=np.float64)
        arr_clamp = np.array(clamped_vals, dtype=np.float64) if clamped_vals else arr_raw
        # Raw (honest, may include negatives)
        summary["user_kpi_raw_p50_ms"] = float(np.percentile(arr_raw, 50))
        summary["user_kpi_raw_p95_ms"] = float(np.percentile(arr_raw, 95))
        summary["user_kpi_raw_p99_ms"] = float(np.percentile(arr_raw, 99))
        summary["user_kpi_raw_min_ms"] = float(np.min(arr_raw))
        summary["user_kpi_raw_max_ms"] = float(np.max(arr_raw))
        # Clamped (for turn-taking gate)
        summary["user_kpi_p50_ms"] = float(np.percentile(arr_clamp, 50))
        summary["user_kpi_p95_ms"] = float(np.percentile(arr_clamp, 95))
        summary["user_kpi_p99_ms"] = float(np.percentile(arr_clamp, 99))
        summary["user_kpi_max_ms"] = float(np.max(arr_clamp))
        summary["user_kpi_count"] = len(raw_vals)
    # D14 P0-1: Split into Turn-taking subset (is_talk_over=false) and Duplex subset (is_talk_over=true)
    tt_raw = [c["user_kpi_raw_ms"] for c in summary["cases"]
              if c.get("user_kpi_raw_ms") is not None and not c.get("is_talk_over")]
    if tt_raw:
        arr_tt = np.array(tt_raw, dtype=np.float64)
        summary["tt_count"] = len(tt_raw)
        summary["tt_p50_ms"] = float(np.percentile(arr_tt, 50))
        summary["tt_p95_ms"] = float(np.percentile(arr_tt, 95))
        summary["tt_p99_ms"] = float(np.percentile(arr_tt, 99))
        summary["tt_min_ms"] = float(np.min(arr_tt))
        summary["tt_max_ms"] = float(np.max(arr_tt))
    else:
        summary["tt_count"] = 0

    # Talk-over / Duplex stats
    summary["talk_over_count"] = len(talk_overs)
    summary["talk_over_rate"] = len(talk_overs) / len(summary["cases"]) if summary["cases"] else 0
    if talk_over_raw:
        arr_to = np.array(talk_over_raw, dtype=np.float64)
        summary["talk_over_ms_p95"] = float(np.percentile(arr_to, 95))
        summary["duplex_count"] = len(talk_over_raw)
        summary["duplex_abs_p50_ms"] = float(np.percentile(arr_to, 50))
        summary["duplex_abs_p95_ms"] = float(np.percentile(arr_to, 95))
        summary["duplex_abs_max_ms"] = float(np.max(arr_to))
    else:
        summary["talk_over_ms_p95"] = 0
        summary["duplex_count"] = 0

    # D14 P0-2: browser_eot_lag_ms aggregate
    eot_lag_vals = [c["browser_eot_lag_ms"] for c in summary["cases"]
                    if c.get("browser_eot_lag_ms") is not None]
    if eot_lag_vals:
        arr_lag = np.array(eot_lag_vals, dtype=np.float64)
        summary["browser_eot_lag_p50_ms"] = float(np.percentile(arr_lag, 50))
        summary["browser_eot_lag_p95_ms"] = float(np.percentile(arr_lag, 95))
        summary["browser_eot_lag_count"] = len(eot_lag_vals)
    # D14 P0-2: is_talk_over_gt aggregate
    gt_to_count = sum(1 for c in summary["cases"] if c.get("is_talk_over_gt") is True)
    summary["talk_over_gt_count"] = gt_to_count

    # D15 P0-1: GT-based USER_KPI aggregates (primary metric)
    gt_raw_vals = [c["user_kpi_gt_raw_ms"] for c in summary["cases"]
                   if c.get("user_kpi_gt_raw_ms") is not None]
    if gt_raw_vals:
        arr_gt = np.array(gt_raw_vals, dtype=np.float64)
        summary["gt_kpi_raw_p50_ms"] = float(np.percentile(arr_gt, 50))
        summary["gt_kpi_raw_p95_ms"] = float(np.percentile(arr_gt, 95))
        summary["gt_kpi_raw_p99_ms"] = float(np.percentile(arr_gt, 99))
        summary["gt_kpi_raw_min_ms"] = float(np.min(arr_gt))
        summary["gt_kpi_raw_max_ms"] = float(np.max(arr_gt))
        summary["gt_kpi_count"] = len(gt_raw_vals)

    # D15: GT-based turn-taking subset (is_talk_over_gt=false)
    gt_tt_raw = [c["user_kpi_gt_raw_ms"] for c in summary["cases"]
                 if c.get("user_kpi_gt_raw_ms") is not None and not c.get("is_talk_over_gt")]
    if gt_tt_raw:
        arr_gt_tt = np.array(gt_tt_raw, dtype=np.float64)
        summary["gt_tt_count"] = len(gt_tt_raw)
        summary["gt_tt_p50_ms"] = float(np.percentile(arr_gt_tt, 50))
        summary["gt_tt_p95_ms"] = float(np.percentile(arr_gt_tt, 95))
        summary["gt_tt_p99_ms"] = float(np.percentile(arr_gt_tt, 99))
        summary["gt_tt_min_ms"] = float(np.min(arr_gt_tt))
        summary["gt_tt_max_ms"] = float(np.max(arr_gt_tt))
    else:
        summary["gt_tt_count"] = 0

    # D15: GT-based duplex subset (is_talk_over_gt=true, abs values)
    gt_to_raw = [abs(c["user_kpi_gt_raw_ms"]) for c in summary["cases"]
                 if c.get("user_kpi_gt_raw_ms") is not None and c.get("is_talk_over_gt")]
    summary["gt_talk_over_rate"] = gt_to_count / total if total > 0 else 0
    if gt_to_raw:
        arr_gt_to = np.array(gt_to_raw, dtype=np.float64)
        summary["gt_duplex_count"] = len(gt_to_raw)
        summary["gt_duplex_abs_p50_ms"] = float(np.percentile(arr_gt_to, 50))
        summary["gt_duplex_abs_p95_ms"] = float(np.percentile(arr_gt_to, 95))
        summary["gt_duplex_abs_max_ms"] = float(np.max(arr_gt_to))
    else:
        summary["gt_duplex_count"] = 0

    summary_path = os.path.join(run_dir, "summary.json")
    write_json(summary_path, summary)

    # Generate report
    report_path = os.path.join(run_dir, "report.md")
    _write_report(report_path, summary, args.net)

    print(f"\n{'='*60}")
    print(f"[AutoBrowser] RESULT: {ok}/{total} cases OK")
    print(f"[AutoBrowser] Joined: {joined}/{total}")
    print(f"[AutoBrowser] Has Audio: {has_audio}/{total}")
    # D15: GT KPI as primary
    if gt_raw_vals:
        print(f"[AutoBrowser] USER_KPI_GT raw: P50={summary['gt_kpi_raw_p50_ms']:.0f}ms "
              f"P95={summary['gt_kpi_raw_p95_ms']:.0f}ms "
              f"min={summary['gt_kpi_raw_min_ms']:.0f}ms")
    if summary.get("gt_tt_count", 0) > 0:
        print(f"[AutoBrowser] GT Turn-taking: P50={summary['gt_tt_p50_ms']:.0f}ms "
              f"P95={summary['gt_tt_p95_ms']:.0f}ms ({summary['gt_tt_count']} cases)")
    print(f"[AutoBrowser] Talk-over (GT): {gt_to_count}/{total} "
          f"(rate={summary.get('gt_talk_over_rate',0):.1%})")
    # Legacy browser-based (reference)
    if raw_vals:
        print(f"[AutoBrowser] (ref) browser KPI raw: P50={summary['user_kpi_raw_p50_ms']:.0f}ms "
              f"P95={summary['user_kpi_raw_p95_ms']:.0f}ms")
    if eot_lag_vals:
        print(f"[AutoBrowser] EOT_LAG: P50={summary['browser_eot_lag_p50_ms']:.0f}ms "
              f"P95={summary['browser_eot_lag_p95_ms']:.0f}ms (diagnostic)")
    print(f"[AutoBrowser] Net Profile: {args.net}")
    print(f"[AutoBrowser] Summary: {summary_path}")
    print(f"[AutoBrowser] Report: {report_path}")

    return 0 if ok == total else 2


def _write_report(report_path: str, summary: dict, net_profile: str):
    """Generate markdown report (D15: GT KPI as primary)."""
    ensure_parent(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AutoBrowser Report (D15)\n\n")

        # D15 P0-1: PRIMARY — GT-based Turn-taking KPI
        f.write("## USER_KPI_GT Turn-taking (is_talk_over_gt=false)\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        gt_tt_count = summary.get("gt_tt_count", 0)
        if gt_tt_count > 0:
            for lbl, key in [("P50", "gt_tt_p50_ms"), ("P95", "gt_tt_p95_ms"),
                             ("P99", "gt_tt_p99_ms"), ("min", "gt_tt_min_ms"),
                             ("max", "gt_tt_max_ms")]:
                v = summary.get(key)
                if v is not None:
                    f.write(f"| {lbl} | {v:.0f} ms |\n")
        f.write(f"| count | {gt_tt_count} / {summary.get('gt_kpi_count', 0)} |\n")
        f.write(f"| talk_over_gt_rate | {summary.get('gt_talk_over_rate', 0):.1%} |\n")
        if gt_tt_count == 0:
            f.write("| (all cases are talk-over) | -- |\n")
        f.write("\n")

        # D15: GT-based Duplex KPI
        f.write("## USER_KPI_GT Duplex (is_talk_over_gt=true, abs values)\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        gt_dup_count = summary.get("gt_duplex_count", 0)
        f.write(f"| talk_over_gt_count | {summary.get('talk_over_gt_count', 0)} |\n")
        if gt_dup_count > 0:
            for lbl, key in [("abs P50", "gt_duplex_abs_p50_ms"),
                             ("abs P95", "gt_duplex_abs_p95_ms"),
                             ("abs max", "gt_duplex_abs_max_ms")]:
                v = summary.get(key)
                if v is not None:
                    f.write(f"| {lbl} | {v:.0f} ms |\n")
        else:
            f.write("| (no talk-over) | -- |\n")
        f.write("\n")

        # D15: EOT_LAG diagnostic
        if summary.get("browser_eot_lag_count", 0) > 0:
            f.write("## EOT_LAG Diagnostic (browser EoT - GT EoT)\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| eot_lag P50 | {summary['browser_eot_lag_p50_ms']:.0f} ms |\n")
            f.write(f"| eot_lag P95 | {summary['browser_eot_lag_p95_ms']:.0f} ms |\n")
            f.write(f"| samples | {summary['browser_eot_lag_count']} |\n")
            f.write("\n")

        # D15: Legacy browser-based (reference, demoted)
        f.write("## (Reference) Browser-based KPI\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for lbl, key in [("raw P50", "user_kpi_raw_p50_ms"), ("raw P95", "user_kpi_raw_p95_ms"),
                         ("raw P99", "user_kpi_raw_p99_ms"), ("raw min", "user_kpi_raw_min_ms"),
                         ("raw max", "user_kpi_raw_max_ms")]:
            v = summary.get(key)
            if v is not None:
                f.write(f"| {lbl} | {v:.0f} ms |\n")
        f.write(f"| total count | {summary.get('user_kpi_count', 0)} |\n")
        f.write(f"| talk_over (browser) | {summary.get('talk_over_count', 0)} |\n")
        f.write(f"| talk_over (gt) | {summary.get('talk_over_gt_count', 0)} |\n")
        f.write("\n> Browser-based KPI includes SILENCE_TIMEOUT_MS overhead; use GT KPI for gates.\n\n")

        # Run info
        f.write(f"## Run Info\n\n")
        f.write(f"- run_id: `{summary['run_id']}`\n")
        f.write(f"- mode: `{summary['mode']}`\n")
        f.write(f"- net_profile: `{net_profile}` ({NET_PROFILES.get(net_profile, {}).get('description', '')})\n")
        f.write(f"- total: `{summary['total_cases']}`, ok: `{summary['ok_cases']}`, "
                f"joined: `{summary['joined_cases']}`, has_audio: `{summary['has_audio_cases']}`\n\n")

        # Per-case detail with GT columns
        f.write("## Per-Case Detail\n\n")
        f.write("| # | case_id | tier | ok | gt_raw_ms | gt_clamp_ms | TO_gt "
                "| browser_raw_ms | eot_lag_ms | elapsed |\n")
        f.write("|---|---------|------|----|-----------|-------------|-------"
                "|----------------|------------|----------|\n")
        for i, c in enumerate(summary.get("cases", [])):
            ok_s = "PASS" if c.get("ok") else "FAIL"
            gt_raw = f"{c['user_kpi_gt_raw_ms']:.0f}" if c.get('user_kpi_gt_raw_ms') is not None else "--"
            gt_clamp = f"{c['user_kpi_gt_clamped_ms']:.0f}" if c.get('user_kpi_gt_clamped_ms') is not None else "--"
            to_gt = "Y" if c.get("is_talk_over_gt") else ""
            b_raw = f"{c['user_kpi_raw_ms']:.0f}" if c.get('user_kpi_raw_ms') is not None else "--"
            lag = f"{c['browser_eot_lag_ms']:.0f}" if c.get('browser_eot_lag_ms') is not None else "--"
            elapsed = f"{c.get('elapsed_s', 0):.0f}s"
            f.write(f"| {i+1} | {c['case_id']} | {c.get('tier','P0')} | {ok_s} "
                    f"| {gt_raw} | {gt_clamp} | {to_gt} "
                    f"| {b_raw} | {lag} | {elapsed} |\n")

        # Net profile info
        f.write(f"\n## Network Profile: `{net_profile}`\n\n")
        prof = NET_PROFILES.get(net_profile, {})
        if prof:
            f.write(f"- {prof.get('description', 'N/A')}\n")

        # Errors
        errors = [c for c in summary.get("cases", []) if c.get("error")]
        if errors:
            f.write("\n## Errors\n\n")
            for c in errors:
                f.write(f"- `{c['case_id']}`: {c['error']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
