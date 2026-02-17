#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from common import ensure_parent, wav_duration_s, write_json


def _check_wav_silent(wav_path: str, threshold: float = 0.01) -> bool:
    """Return True if wav file is missing or has rms < threshold (silent)."""
    if not wav_path or not os.path.exists(wav_path):
        return True
    try:
        import wave
        import struct
        import math
        with wave.open(wav_path, "rb") as wf:
            n = wf.getnframes()
            if n == 0:
                return True
            raw = wf.readframes(n)
            sw = wf.getsampwidth()
            if sw == 2:
                samples = struct.unpack(f"<{n}h", raw)
                rms = math.sqrt(sum(s * s for s in samples) / n) / 32768.0
            else:
                return True
        return rms < threshold
    except Exception:
        return True


def _ts_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _fetch_token(token_api: str, room: str, identity: str) -> tuple[str, str]:
    r = requests.get(token_api, params={"room": room, "identity": identity}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["token"], data["url"]


def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_cases(wavs: list[str]) -> list[tuple]:
    cases = []
    for w in wavs:
        case_id = Path(w).stem
        cases.append((case_id, w, {}))
    return cases


def _build_nightly_cases(wavs: list[str], turns: int = 20) -> list[tuple]:
    """D8 nightly: 同一个 wav 重复 N 轮（测内存泄漏/延迟漂移）"""
    cases = []
    wav = wavs[0] if wavs else ""
    for i in range(turns):
        cases.append((f"nightly_turn_{i:03d}", wav, {}))
    return cases


def _build_cases_from_json(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for idx, item in enumerate(data.get("cases", [])):
        case_id = item.get("case_id") or f"case_{idx:03d}"
        wav = item.get("wav", "")
        if wav:
            cases.append((case_id, wav, item))  # D8: 传完整 item（含 expected_silences）
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRTC run_suite: three-ring regression.")
    p.add_argument("--mode", default="fast", choices=["fast", "nightly"],
                   help="fast=独立room/case, nightly=同room多轮")
    p.add_argument("--ring0", type=int, default=1, help="1=run TTS core regression (Ring0)")
    p.add_argument("--room", default="voice-agent-test", help="target room")
    p.add_argument("--token_api", default="http://127.0.0.1:3000/api/token", help="token server api")
    p.add_argument("--run_id", default="", help="optional run id")
    p.add_argument("--wavs", nargs="+", default=[], help="wav test cases")
    p.add_argument("--cases_json", default="", help="optional json case file, overrides --wavs")
    p.add_argument("--agent_identity", default="", help="agent participant identity exact match")
    p.add_argument("--agent_identity_prefix", default="agent-", help="preferred agent identity prefix")
    p.add_argument("--user_identity", default="autortc-user", help="user_bot identity")
    p.add_argument("--probe_identity", default="autortc-probe", help="probe_bot identity")
    p.add_argument("--frame_ms", type=int, default=20, help="audio frame ms")
    p.add_argument("--realtime", type=int, default=1, help="user_bot realtime mode")
    p.add_argument("--user_start_delay_s", type=float, default=2.5, help="user waits before first frame")
    p.add_argument("--probe_after_user_s", type=float, default=0.8, help="probe starts after user joins")
    p.add_argument("--record_pad_s", type=float, default=10.0, help="D10: extra record window (6→10s to cover welcome+STT+LLM+TTS)")
    p.add_argument("--output_root", default="output/autortc", help="output root directory")
    p.add_argument("--with_metrics", type=int, default=1, help="1=run audio_metrics after suite")
    p.add_argument("--turns", type=int, default=20, help="nightly mode: number of turns")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or _ts_run_id()
    run_dir = os.path.join(args.output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    user_bot = os.path.join(os.path.dirname(__file__), "user_bot.py")
    probe_bot = os.path.join(os.path.dirname(__file__), "probe_bot.py")
    if args.mode == "nightly":
        cases = _build_nightly_cases(args.wavs or [
            os.path.join(os.path.dirname(__file__), "..", "..", "output", "day1_test_input.wav")
        ], turns=args.turns)
    elif args.cases_json:
        cases = _build_cases_from_json(args.cases_json)
    else:
        cases = _build_cases(args.wavs)
    if not cases:
        print("no cases found: provide --wavs or --cases_json with valid entries", file=sys.stderr)
        return 1

    traces_path = os.path.join(args.output_root, "traces.jsonl")
    ensure_parent(traces_path)
    summary = {"run_id": run_id, "room": args.room, "cases": []}

    # nightly 模式：所有 turn 在同一个 room
    nightly_room = f"{args.room}-nightly-{run_id[-6:]}" if args.mode == "nightly" else ""
    nightly_tokens = {}  # 缓存 nightly room 的 token

    for idx, case_tuple in enumerate(cases):
        case_id, wav_path = case_tuple[0], case_tuple[1]
        case_meta = case_tuple[2] if len(case_tuple) > 2 else {}
        user_identity = args.user_identity
        probe_identity = args.probe_identity

        # D10: 所有模式都用独立 room（nightly 同 room 复用导致 50% retry）
        # agent 是 per-room worker，同 room 复用会导致 stale process state
        case_room = f"{args.room}-{case_id}-{run_id[-6:]}"

        try:
            user_token, livekit_url = _fetch_token(args.token_api, case_room, user_identity)
            probe_token, _ = _fetch_token(args.token_api, case_room, probe_identity)
        except Exception as e:
            summary["cases"].append({"case_id": case_id, "wav": wav_path, "ok": False, "error": str(e)})
            continue

        case_dir = os.path.join(run_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        probe_wav = os.path.join(run_dir, f"{case_id}_agent.wav")
        probe_json = os.path.join(case_dir, "probe_result.json")
        user_json = os.path.join(case_dir, "user_result.json")

        rec_s = wav_duration_s(wav_path) + args.record_pad_s
        trace_id = f"{run_id}-{case_id}-turn-{idx:03d}"

        probe_cmd = [
            sys.executable,
            probe_bot,
            "--room",
            case_room,
            "--url",
            livekit_url,
            "--token",
            probe_token,
            "--target_identity",
            args.agent_identity,
            "--target_identity_prefix",
            args.agent_identity_prefix,
            "--exclude_identity_prefix",
            "autortc-",
            "--frame_ms",
            str(args.frame_ms),
            "--record_seconds",
            str(rec_s),
            "--output_wav",
            probe_wav,
            "--result_json",
            probe_json,
        ]
        # D9: CAPTURE_PRE_RTC=1 (Agent saves to output/pre_rtc/<trace_id>/)
        os.environ["CAPTURE_PRE_RTC"] = "1"

        user_cmd = [
            sys.executable,
            user_bot,
            "--room",
            case_room,
            "--url",
            livekit_url,
            "--token",
            user_token,
            "--wav",
            wav_path,
            "--realtime",
            str(args.realtime),
            "--frame_ms",
            str(args.frame_ms),
            "--start_delay_s",
            str(args.user_start_delay_s),
            "--trace_id",
            trace_id,
            "--case_id",
            case_id,
            "--turn_id",
            str(idx),
            "--result_json",
            user_json,
        ]

        t_case_start = time.time()
        max_attempts = 3  # D10: retry up to 2x if recording is silent (flaky cases like low_volume)
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"  RETRY {attempt}/{max_attempts-1}: {case_id} (previous attempt was silent)")
                # Re-create room/tokens for retry
                case_room = f"{args.room}-{case_id}-{run_id[-6:]}-r{attempt}"
                try:
                    user_token, livekit_url = _fetch_token(args.token_api, case_room, user_identity)
                    probe_token, _ = _fetch_token(args.token_api, case_room, probe_identity)
                except Exception as e:
                    print(f"  RETRY token fetch failed: {e}")
                    break
                # Update tokens in commands
                user_cmd_retry = [x for x in user_cmd]
                probe_cmd_retry = [x for x in probe_cmd]
                for cmd_list in [user_cmd_retry, probe_cmd_retry]:
                    for ci, cv in enumerate(cmd_list):
                        if cv == "--token" and ci + 1 < len(cmd_list):
                            if cmd_list is user_cmd_retry:
                                cmd_list[ci + 1] = user_token
                            else:
                                cmd_list[ci + 1] = probe_token
                        if cv == "--room" and ci + 1 < len(cmd_list):
                            cmd_list[ci + 1] = case_room
                user_cmd = user_cmd_retry
                probe_cmd = probe_cmd_retry
                time.sleep(20)  # extra wait for agent process pool recovery

            print(f"[{idx+1}/{len(cases)}] {case_id} room={case_room}" + (f" (attempt {attempt+1})" if attempt > 0 else ""))
            # D9: user 先加入（触发 Agent JOB），probe 紧跟（订阅 Agent track + 发 probe_ready）
            # user_bot 会等 probe_ready barrier 后才开始推音频
            p_user = subprocess.Popen(user_cmd)
            time.sleep(0.5)  # 让 user 先连上触发 JOB
            p_probe = subprocess.Popen(probe_cmd)

            user_timeout = max(60, int(rec_s) + 35)
            probe_timeout = max(75, int(rec_s) + 40)
            user_rc = -1
            probe_rc = -1
            try:
                user_rc = p_user.wait(timeout=user_timeout)
            except subprocess.TimeoutExpired:
                print(f"  WARN: user_bot timeout ({user_timeout}s), killing")
                p_user.kill()
                p_user.wait()
            try:
                probe_rc = p_probe.wait(timeout=probe_timeout)
            except subprocess.TimeoutExpired:
                print(f"  WARN: probe_bot timeout ({probe_timeout}s), killing")
                p_probe.kill()
                p_probe.wait()
            t_case_end = time.time()

            user_res = _load_json(user_json)
            probe_res = _load_json(probe_json)
            frame_ts = probe_res.get("frame_timestamps", []) if isinstance(probe_res, dict) else []
            t_user_end = user_res.get("t_user_send_end")
            t_probe_after_user_end = None
            if t_user_end and frame_ts:
                for ts in frame_ts:
                    if ts >= t_user_end:
                        t_probe_after_user_end = ts
                        break

            # D9 P0-3: pre_rtc 只用 trace_id 确定性路径，无兜底
            time.sleep(2)  # 让 Agent TTS 完成 + 写文件
            import shutil
            pre_rtc_src = os.path.join("output", "pre_rtc", trace_id, "pre_rtc.wav")
            if os.path.exists(pre_rtc_src):
                pre_dst = os.path.join(case_dir, "pre_rtc.wav")
                try:
                    shutil.copy2(pre_rtc_src, pre_dst)
                except Exception:
                    pass

            # D9: 也拷贝 reply wav
            probe_reply_wav = probe_wav.replace("_agent.wav", "_reply.wav")
            if os.path.exists(probe_reply_wav):
                reply_dst = os.path.join(case_dir, "post_rtc_reply.wav")
                try:
                    shutil.copy2(probe_reply_wav, reply_dst)
                except Exception:
                    pass

            # D10: 所有模式统一删 room + 等回收（nightly 同 room 复用导致 50% retry）
            try:
                import asyncio as _aio
                from livekit import api as _lk_api
                def _delete_room():
                    async def _do():
                        lk = _lk_api.LiveKitAPI(
                            url=livekit_url,
                            api_key=os.environ.get("LIVEKIT_API_KEY", "API7fj35wGLumtc"),
                            api_secret=os.environ.get("LIVEKIT_API_SECRET", "WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B"),
                        )
                        await lk.room.delete_room(_lk_api.DeleteRoomRequest(room=case_room))
                        await lk.aclose()
                    _aio.run(_do())
                _delete_room()
            except Exception:
                pass
            time.sleep(20)  # D10: 统一 20s 回收等待（18s 时 nightly 10% retry）

            # D10: Check recording quality and classify retry reason
            is_silent = _check_wav_silent(probe_wav)
            retry_reason = ""
            if is_silent and attempt < max_attempts - 1:
                # Classify WHY it's silent
                _user_res = _load_json(user_json)
                _probe_res = _load_json(probe_json)
                _probe_ready = _user_res.get("probe_ready_received", False)
                _agent_ready = _user_res.get("agent_ready_received", False)
                _reply_events = _probe_res.get("reply_events", []) if isinstance(_probe_res, dict) else []
                _has_reply_start = any(e.get("event") == "reply_start" for e in _reply_events)

                if not _probe_ready:
                    retry_reason = "PROBE_NOT_READY"
                elif not _agent_ready:
                    retry_reason = "AGENT_TRACK_NOT_PUBLISHED"
                elif not _has_reply_start:
                    retry_reason = "REPLY_EVENTS_MISSING"
                elif is_silent:
                    retry_reason = "POST_RTC_SILENT"
                else:
                    retry_reason = "WORKER_POOL_BUSY"

                print(f"  SILENT detected (rms < 0.01), reason={retry_reason}, will retry...")
                continue  # retry the case
            break  # success or last attempt, proceed

        ok = user_rc == 0 and probe_rc == 0 and bool(probe_res.get("ok"))
        case_result = {
            "case_id": case_id,
            "wav": wav_path,
            "room": case_room,
            "ok": ok,
            "expected_silences": case_meta.get("expected_silences"),
            "user_identity": user_identity,
            "probe_identity": probe_identity,
            "user_rc": user_rc,
            "probe_rc": probe_rc,
            "elapsed_s": t_case_end - t_case_start,
            "user_result": user_res,
            "probe_result": probe_res,
            "probe_wav": probe_wav,
            "trace_id": trace_id,
            "t_probe_first_audio_after_user_end": t_probe_after_user_end,
            # D10: retry tracking
            "attempts": attempt + 1,
            "retry_reason": retry_reason if attempt > 0 else "",
        }
        summary["cases"].append(case_result)

        trace = {
            "run_id": run_id,
            "case_id": case_id,
            "trace_id": trace_id,
            "t_user_send_start": user_res.get("t_user_send_start"),
            "t_user_send_end": user_res.get("t_user_send_end"),
            "t_probe_first_audio_recv": probe_res.get("t_probe_first_audio_recv"),
            "t_probe_first_audio_after_user_end": t_probe_after_user_end,
            "probe_wav": probe_wav,
            "ok": ok,
        }
        with open(traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # nightly 结束后清理 room
    if args.mode == "nightly" and nightly_room:
        try:
            import asyncio as _aio
            from livekit import api as _lk_api
            def _delete_nightly():
                async def _do():
                    lk = _lk_api.LiveKitAPI(
                        url=os.environ.get("LIVEKIT_URL", ""),
                        api_key=os.environ.get("LIVEKIT_API_KEY", ""),
                        api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
                    )
                    await lk.room.delete_room(_lk_api.DeleteRoomRequest(room=nightly_room))
                    await lk.aclose()
                _aio.run(_do())
            _delete_nightly()
        except Exception:
            pass

    summary["ok_cases"] = sum(1 for c in summary["cases"] if c.get("ok"))
    summary["total_cases"] = len(summary["cases"])
    summary["mode"] = args.mode

    # Nightly: 计算 trace join 成功率 + audio valid rate
    if args.mode == "nightly":
        total_turns = len(summary["cases"])
        joined = sum(1 for c in summary["cases"]
                     if c.get("trace_id") and c.get("t_probe_first_audio_after_user_end"))
        audio_valid = sum(1 for c in summary["cases"]
                          if c.get("ok") and c.get("probe_result", {}).get("pcm_bytes", 0) > 1000)
        crashes = sum(1 for c in summary["cases"]
                      if c.get("user_rc", 0) != 0 or c.get("probe_rc", 0) != 0)
        summary["nightly_trace_join_rate"] = round(joined / max(1, total_turns), 3)
        summary["nightly_audio_valid_rate"] = round(audio_valid / max(1, total_turns), 3)
        summary["nightly_crashes"] = crashes
        # D10: retry breakdown
        retried = [c for c in summary["cases"] if c.get("attempts", 1) > 1]
        retry_rate = len(retried) / max(1, total_turns)
        summary["nightly_retry_rate"] = round(retry_rate, 3)
        summary["nightly_retry_count"] = len(retried)

        print(f"[Nightly] Trace join: {joined}/{total_turns} = {summary['nightly_trace_join_rate']:.1%}")
        print(f"[Nightly] Audio valid: {audio_valid}/{total_turns} = {summary['nightly_audio_valid_rate']:.1%}")
        print(f"[Nightly] Crashes: {crashes}")
        print(f"[Nightly] Retry rate: {len(retried)}/{total_turns} = {retry_rate:.1%}")

        # D10: write retry breakdown CSV
        retry_csv = os.path.join(run_dir, "nightly_retry_breakdown.csv")
        import csv as _csv
        with open(retry_csv, "w", newline="", encoding="utf-8") as cf:
            writer = _csv.DictWriter(cf, fieldnames=[
                "case_id", "trace_id", "attempts", "retry_reason",
                "probe_ready", "agent_ready", "final_ok",
            ])
            writer.writeheader()
            for c in summary["cases"]:
                ur = c.get("user_result", {}) or {}
                writer.writerow({
                    "case_id": c.get("case_id", ""),
                    "trace_id": c.get("trace_id", ""),
                    "attempts": c.get("attempts", 1),
                    "retry_reason": c.get("retry_reason", ""),
                    "probe_ready": ur.get("probe_ready_received", ""),
                    "agent_ready": ur.get("agent_ready_received", ""),
                    "final_ok": c.get("ok", False),
                })
        print(f"[Nightly] Retry breakdown: {retry_csv}")

    summary_path = os.path.join(run_dir, "summary.json")
    write_json(summary_path, summary)

    if args.with_metrics == 1:
        metrics_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "audio_metrics.py"),
            "--run_summary",
            summary_path,
            "--autortc_traces",
            traces_path,
            "--agent_traces",
            "output/day5_e2e_traces.jsonl",
            "--output_dir",
            run_dir,
        ]
        subprocess.run(metrics_cmd, check=False)

    # ── D7 Ring0: TTS Core Regression ────────────────────────
    ring0_pass = True
    if args.ring0 == 1:
        print("[Ring0] Running TTS core regression...")
        ring0_script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "run_ci_regression.sh")
        if os.path.exists(ring0_script):
            rc = subprocess.run(["bash", ring0_script, "--mode", "fast"], capture_output=True, text=True)
            ring0_pass = rc.returncode == 0
            ring0_status = "PASS" if ring0_pass else "FAIL"
            print(f"[Ring0] TTS regression: {ring0_status}")
            # 写入 summary
            summary["ring0_tts_regression"] = ring0_status
            write_json(summary_path, summary)
        else:
            print(f"[Ring0] Script not found: {ring0_script}")

    print(f"run_id={run_id}")
    print(f"summary={summary_path}")
    print(f"traces={traces_path}")
    all_ok = summary["ok_cases"] == summary["total_cases"] and ring0_pass
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

