#!/usr/bin/env python3
"""
D15 P0-4: Automated endpointing parameter search.

Sweeps combinations of (ENDPOINTING_MIN_SILENCE_MS, BARGEIN_ACTIVATION_THRESHOLD, BARGEIN_MIN_SPEECH_MS)
across 4 key cases × 3 repeats per config, then outputs a Pareto table and auto-selects
the optimal turn_taking configuration.

Usage:
    python3 tools/autobrowser/run_endpointing_grid.py \
        --cases_json tools/autortc/cases/d14_matrix_cases.json \
        --token_api http://127.0.0.1:9090/api/token \
        --output_root /tmp/d15_grid
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")

GRID = [
    {"ENDPOINTING_MIN_SILENCE_MS": "800",  "BARGEIN_ACTIVATION_THRESHOLD": "0.5", "BARGEIN_MIN_SPEECH_MS": "50"},
    {"ENDPOINTING_MIN_SILENCE_MS": "800",  "BARGEIN_ACTIVATION_THRESHOLD": "0.7", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1000", "BARGEIN_ACTIVATION_THRESHOLD": "0.5", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1000", "BARGEIN_ACTIVATION_THRESHOLD": "0.7", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1200", "BARGEIN_ACTIVATION_THRESHOLD": "0.5", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1200", "BARGEIN_ACTIVATION_THRESHOLD": "0.7", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1200", "BARGEIN_ACTIVATION_THRESHOLD": "0.7", "BARGEIN_MIN_SPEECH_MS": "200"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1500", "BARGEIN_ACTIVATION_THRESHOLD": "0.7", "BARGEIN_MIN_SPEECH_MS": "120"},
    {"ENDPOINTING_MIN_SILENCE_MS": "1500", "BARGEIN_ACTIVATION_THRESHOLD": "0.8", "BARGEIN_MIN_SPEECH_MS": "200"},
]

REPEATS = 3
AGENT_STARTUP_WAIT = 8
AGENT_SHUTDOWN_WAIT = 3
INTER_CASE_WAIT = 10


def parse_args():
    p = argparse.ArgumentParser(description="D15: automated endpointing parameter grid search")
    p.add_argument("--cases_json", required=True)
    p.add_argument("--token_api", default="http://127.0.0.1:9090/api/token")
    p.add_argument("--output_root", default="/tmp/d15_grid")
    p.add_argument("--record_s", type=int, default=25)
    p.add_argument("--repeats", type=int, default=REPEATS)
    return p.parse_args()


def _stop_supervisor_agent():
    subprocess.run(["supervisorctl", "stop", "voice-agent:livekit-agent"],
                   capture_output=True, check=False)
    time.sleep(2)


def _start_agent(env_overrides: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_overrides)
    env.setdefault("MODE", "turn_taking")
    env.setdefault("LIVEKIT_URL", "wss://renshenghehuoren-mpdsjfwe.livekit.cloud")
    env.setdefault("LIVEKIT_API_KEY", "API7fj35wGLumtc")
    env.setdefault("LIVEKIT_API_SECRET", "WK8k8fUhhsHoa2R2qfO076lyuDHgJubwemQuY4nk398B")
    env.setdefault("PYTHONPATH", "/workspace/vllm-omni")

    proc = subprocess.Popen(
        [sys.executable, os.path.join(PROJECT_ROOT, "runtime", "livekit_agent.py"), "start"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(AGENT_STARTUP_WAIT)
    return proc


def _stop_agent(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(AGENT_SHUTDOWN_WAIT)


def _run_suite(cases_json, token_api, output_root, record_s, run_label):
    """Run a single suite and return summary dict or None."""
    cmd = [
        sys.executable, "-u",
        os.path.join(SCRIPT_DIR, "run_suite.py"),
        "--cases_json", cases_json,
        "--token_api", token_api,
        "--record_s", str(record_s),
        "--p0_only", "1",
        "--output_root", output_root,
        "--inter_case_wait_s", str(INTER_CASE_WAIT),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    summaries = glob.glob(os.path.join(output_root, "*/summary.json"))
    if not summaries:
        return None
    summaries.sort(key=os.path.getmtime, reverse=True)
    with open(summaries[0]) as f:
        return json.load(f)


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 70)
    print("D15 P0-4: Endpointing Parameter Grid Search")
    print(f"Grid size: {len(GRID)} configs x {args.repeats} repeats x 4 cases")
    print("=" * 70)

    _stop_supervisor_agent()

    results = []

    for gi, config in enumerate(GRID):
        label = f"ep{config['ENDPOINTING_MIN_SILENCE_MS']}_th{config['BARGEIN_ACTIVATION_THRESHOLD']}_sp{config['BARGEIN_MIN_SPEECH_MS']}"
        print(f"\n[{gi+1}/{len(GRID)}] Config: {label}")

        repeat_data = []
        for rep in range(args.repeats):
            run_dir = os.path.join(args.output_root, f"{label}_r{rep}")
            os.makedirs(run_dir, exist_ok=True)

            proc = _start_agent(config)
            try:
                summary = _run_suite(
                    args.cases_json, args.token_api, run_dir, args.record_s,
                    f"{label}_r{rep}")
                if summary:
                    repeat_data.append(summary)
                    gt_tt_p95 = summary.get("gt_tt_p95_ms", "N/A")
                    gt_to = summary.get("talk_over_gt_count", "?")
                    print(f"  rep {rep}: GT_TT_P95={gt_tt_p95}, TO_GT={gt_to}")
                else:
                    print(f"  rep {rep}: FAILED (no summary)")
            finally:
                _stop_agent(proc)

        if not repeat_data:
            results.append({"config": config, "label": label, "valid": False})
            continue

        import numpy as np
        all_gt_tt_p95 = [s["gt_tt_p95_ms"] for s in repeat_data if s.get("gt_tt_p95_ms") is not None]
        all_to_gt = [s.get("talk_over_gt_count", 0) for s in repeat_data]
        total_to_gt = sum(all_to_gt)
        total_cases = sum(s.get("total_cases", 0) for s in repeat_data)
        to_rate = total_to_gt / total_cases if total_cases > 0 else 1.0

        results.append({
            "config": config,
            "label": label,
            "valid": True,
            "gt_tt_p95_median": float(np.median(all_gt_tt_p95)) if all_gt_tt_p95 else None,
            "gt_tt_p95_max": float(np.max(all_gt_tt_p95)) if all_gt_tt_p95 else None,
            "total_to_gt": total_to_gt,
            "to_rate": to_rate,
            "total_cases": total_cases,
            "repeats": len(repeat_data),
        })

    # Restore supervisor agent
    print("\nRestoring supervisor agent...")
    subprocess.run(["supervisorctl", "start", "voice-agent:livekit-agent"],
                   capture_output=True, check=False)
    time.sleep(5)

    # Print Pareto table
    print("\n" + "=" * 70)
    print("PARETO TABLE")
    print("=" * 70)
    header = f"{'Label':<40} {'GT_TT_P95_med':>14} {'GT_TT_P95_max':>14} {'TO_GT':>6} {'TO_rate':>8} {'Cases':>6}"
    print(header)
    print("-" * len(header))

    valid_results = [r for r in results if r.get("valid")]
    valid_results.sort(key=lambda r: (r.get("total_to_gt", 999), r.get("gt_tt_p95_median") or 99999))

    best = None
    for r in valid_results:
        p95_med = f"{r['gt_tt_p95_median']:.0f}" if r.get('gt_tt_p95_median') is not None else "N/A"
        p95_max = f"{r['gt_tt_p95_max']:.0f}" if r.get('gt_tt_p95_max') is not None else "N/A"
        marker = ""
        if best is None and r.get("total_to_gt", 999) == 0:
            best = r
            marker = " <-- OPTIMAL"
        print(f"{r['label']:<40} {p95_med:>14} {p95_max:>14} {r.get('total_to_gt', '?'):>6} "
              f"{r.get('to_rate', 0):>8.1%} {r.get('total_cases', 0):>6}{marker}")

    if best is None and valid_results:
        best = valid_results[0]
        print(f"\nNo config achieved TO=0. Best compromise: {best['label']}")
    elif best:
        print(f"\nOptimal config: {best['label']} "
              f"(GT_TT_P95={best['gt_tt_p95_median']:.0f}ms, TO={best['total_to_gt']})")

    # Save results
    pareto_path = os.path.join(args.output_root, "pareto_results.json")
    with open(pareto_path, "w") as f:
        json.dump({"grid_results": results, "optimal": best}, f, indent=2)
    print(f"\nResults saved: {pareto_path}")

    # Write optimal config as env file
    if best and best.get("config"):
        env_path = os.path.join(args.output_root, "optimal_env.sh")
        with open(env_path, "w") as f:
            f.write("#!/bin/bash\n# D15 P0-4: Auto-selected optimal endpointing config\n")
            for k, v in best["config"].items():
                f.write(f"export {k}={v}\n")
            f.write("export MODE=turn_taking\n")
        print(f"Optimal env: {env_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
