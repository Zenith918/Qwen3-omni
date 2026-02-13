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


def _build_cases(wavs: list[str]) -> list[tuple[str, str]]:
    cases = []
    for w in wavs:
        case_id = Path(w).stem
        cases.append((case_id, w))
    return cases


def _build_cases_from_json(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for idx, item in enumerate(data.get("cases", [])):
        case_id = item.get("case_id") or f"case_{idx:03d}"
        wav = item.get("wav", "")
        if wav:
            cases.append((case_id, wav))
    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRTC run_suite: orchestrate probe_bot + user_bot.")
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
    p.add_argument("--record_pad_s", type=float, default=6.0, help="extra record window over wav duration")
    p.add_argument("--output_root", default="output/autortc", help="output root directory")
    p.add_argument("--with_metrics", type=int, default=1, help="1=run audio_metrics after suite")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or _ts_run_id()
    run_dir = os.path.join(args.output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    user_bot = os.path.join(os.path.dirname(__file__), "user_bot.py")
    probe_bot = os.path.join(os.path.dirname(__file__), "probe_bot.py")
    cases = _build_cases_from_json(args.cases_json) if args.cases_json else _build_cases(args.wavs)
    if not cases:
        print("no cases found: provide --wavs or --cases_json with valid entries", file=sys.stderr)
        return 1

    traces_path = os.path.join(args.output_root, "traces.jsonl")
    ensure_parent(traces_path)
    summary = {"run_id": run_id, "room": args.room, "cases": []}

    for idx, (case_id, wav_path) in enumerate(cases):
        user_identity = args.user_identity
        probe_identity = args.probe_identity

        # F2: 每个 case 用独立 room，确保 Agent 每次都分配新 job
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
        # probe 先加入（确保能订阅到 Agent 的 audio track）
        p_probe = subprocess.Popen(probe_cmd)
        time.sleep(args.probe_after_user_s)
        # user 后加入（触发 Agent JOB）
        p_user = subprocess.Popen(user_cmd)

        user_rc = p_user.wait(timeout=max(30, int(rec_s) + 10))
        probe_rc = p_probe.wait(timeout=max(45, int(rec_s) + 15))
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

        # F2: case 结束后清理 room，让 Agent 子进程正常回收
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
        time.sleep(4)  # 给 Agent 子进程充足的回收时间（默认池 4 个）

        ok = user_rc == 0 and probe_rc == 0 and bool(probe_res.get("ok"))
        case_result = {
            "case_id": case_id,
            "wav": wav_path,
            "room": case_room,
            "ok": ok,
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

    summary["ok_cases"] = sum(1 for c in summary["cases"] if c.get("ok"))
    summary["total_cases"] = len(summary["cases"])
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
    print(f"run_id={run_id}")
    print(f"summary={summary_path}")
    print(f"traces={traces_path}")
    return 0 if summary["ok_cases"] == summary["total_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

