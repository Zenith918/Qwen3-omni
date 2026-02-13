#!/usr/bin/env python3
"""
Day 3 P0-1: TTS 服务端 cancel + 断连 压力测试

测试场景:
  1. cancel: 正常发请求，读几个 chunk 后调 /tts/cancel
  2. disconnect: 正常发请求，读几个 chunk 后直接关连接（模拟客户端崩溃）
  3. short_text: 发送极短文本（<2 字），验证服务端返回 400 而非 crash
  4. rapid_fire: 快速连续发请求然后立即断开

用法:
  python3 tts_stress_test.py --rounds 1000 --mode cancel
  python3 tts_stress_test.py --rounds 1000 --mode disconnect
  python3 tts_stress_test.py --rounds 1000 --mode mixed
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:9000/tts/stream")
TTS_CANCEL_URL = os.environ.get("TTS_CANCEL_URL", "http://127.0.0.1:9000/tts/cancel")
TTS_HEALTH_URL = os.environ.get("TTS_HEALTH_URL", "http://127.0.0.1:9000/health")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

TEXTS = [
    "今天天气真不错，适合出去走走。",
    "你好，请问最近的地铁站怎么走？",
    "我觉得这个方案可以再优化一下。",
    "好的，一起去吧！",
    "非常感谢你的帮助。",
    "请稍等一下，我马上就来。",
    "明天下午三点我们开个会讨论一下。",
]

SHORT_TEXTS = ["你", "a", "", " "]  # 应该被服务端拒绝


def check_health() -> bool:
    try:
        r = requests.get(TTS_HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def stress_cancel(request_idx: int) -> dict:
    """发请求 → 读几个 chunk → 调 /tts/cancel"""
    text = random.choice(TEXTS)
    request_id = f"stress_{request_idx}_{uuid.uuid4().hex[:6]}"
    result = {"idx": request_idx, "mode": "cancel", "status": "ok",
              "cancel_ms": 0, "chunks_read": 0}

    try:
        payload = {"text": text, "speaker": "serena", "request_id": request_id}
        resp = requests.post(TTS_URL, json=payload, stream=True, timeout=30)
        resp.raise_for_status()

        # Read 1-3 chunks then cancel
        chunks_to_read = random.randint(1, 3)
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                result["chunks_read"] += 1
                if result["chunks_read"] >= chunks_to_read:
                    break

        # Call cancel API
        t_cancel = time.time()
        try:
            requests.post(TTS_CANCEL_URL,
                          json={"request_id": request_id}, timeout=2)
        except Exception:
            pass
        result["cancel_ms"] = round((time.time() - t_cancel) * 1000, 1)

        resp.close()
    except Exception as e:
        result["status"] = f"error:{type(e).__name__}"
    return result


def stress_disconnect(request_idx: int) -> dict:
    """发请求 → 读几个 chunk → 直接关连接（不调 cancel）"""
    text = random.choice(TEXTS)
    result = {"idx": request_idx, "mode": "disconnect", "status": "ok",
              "chunks_read": 0}

    try:
        payload = {"text": text, "speaker": "serena"}
        resp = requests.post(TTS_URL, json=payload, stream=True, timeout=30)
        resp.raise_for_status()

        # Read 0-2 chunks then forcibly close
        chunks_to_read = random.randint(0, 2)
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                result["chunks_read"] += 1
                if result["chunks_read"] >= chunks_to_read:
                    break

        # Force close without cancel
        resp.close()
    except Exception as e:
        result["status"] = f"error:{type(e).__name__}"
    return result


def stress_short_text(request_idx: int) -> dict:
    """发送极短文本，验证服务端返回 400"""
    text = random.choice(SHORT_TEXTS)
    result = {"idx": request_idx, "mode": "short_text", "status": "ok",
              "text": text, "http_code": 0}

    try:
        payload = {"text": text, "speaker": "serena"}
        resp = requests.post(TTS_URL, json=payload, stream=False, timeout=10)
        result["http_code"] = resp.status_code
        if resp.status_code == 400:
            result["status"] = "rejected_as_expected"
        elif resp.status_code == 200:
            result["status"] = "accepted_unexpectedly"
        else:
            result["status"] = f"unexpected_code_{resp.status_code}"
    except Exception as e:
        result["status"] = f"error:{type(e).__name__}"
    return result


def stress_rapid_fire(request_idx: int) -> dict:
    """快速发请求然后立即断开（不等响应体）"""
    text = random.choice(TEXTS)
    result = {"idx": request_idx, "mode": "rapid_fire", "status": "ok"}

    try:
        payload = {"text": text, "speaker": "serena"}
        resp = requests.post(TTS_URL, json=payload, stream=True, timeout=5)
        resp.close()  # 立即关
    except Exception as e:
        result["status"] = f"error:{type(e).__name__}"
    return result


def main():
    parser = argparse.ArgumentParser(description="TTS stress test")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--mode", default="mixed",
                        choices=["cancel", "disconnect", "short_text", "rapid_fire", "mixed"])
    parser.add_argument("--health_check_interval", type=int, default=50,
                        help="Check TTS health every N rounds")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initial health check
    if not check_health():
        print("❌ TTS server not healthy!")
        sys.exit(1)
    print(f"✅ TTS server healthy")

    results = []
    crashes = 0
    health_failures = 0
    cancel_latencies = []

    t_start = time.time()
    for i in range(args.rounds):
        if args.mode == "mixed":
            r = random.random()
            if r < 0.4:
                result = stress_cancel(i)
            elif r < 0.8:
                result = stress_disconnect(i)
            elif r < 0.9:
                result = stress_short_text(i)
            else:
                result = stress_rapid_fire(i)
        elif args.mode == "cancel":
            result = stress_cancel(i)
        elif args.mode == "disconnect":
            result = stress_disconnect(i)
        elif args.mode == "short_text":
            result = stress_short_text(i)
        else:
            result = stress_rapid_fire(i)

        results.append(result)

        if result.get("cancel_ms"):
            cancel_latencies.append(result["cancel_ms"])

        if "error" in result["status"]:
            crashes += 1

        # Periodic progress + health check
        if (i + 1) % args.health_check_interval == 0:
            healthy = check_health()
            if not healthy:
                health_failures += 1
                print(f"\n  ⚠ Health check failed at round {i+1}, waiting for auto-restart...")
                # Wait for auto-restart (up to 60s)
                recovered = False
                for wait_i in range(20):
                    time.sleep(3)
                    if check_health():
                        print(f"  ✅ TTS recovered after {(wait_i+1)*3}s")
                        recovered = True
                        break
                if not recovered:
                    print(f"  ❌ TTS did not recover after 60s, stopping.")
                    break
            pct = (i + 1) / args.rounds * 100
            sys.stdout.write(
                f"\r  [{i+1}/{args.rounds}] {pct:.0f}% "
                f"errors={crashes} health_fails={health_failures}")
            sys.stdout.flush()

        # Small delay between requests to avoid connection pool exhaustion
        time.sleep(0.05)

    elapsed = time.time() - t_start
    print()

    # Final health check
    final_healthy = check_health()

    # Stats
    cancel_results = [r for r in results if r["mode"] == "cancel"]
    disconnect_results = [r for r in results if r["mode"] == "disconnect"]
    short_results = [r for r in results if r["mode"] == "short_text"]
    rapid_results = [r for r in results if r["mode"] == "rapid_fire"]

    report = {
        "total_rounds": len(results),
        "mode": args.mode,
        "elapsed_s": round(elapsed, 1),
        "crashes": crashes,
        "health_check_failures": health_failures,
        "final_healthy": final_healthy,
        "cancel_count": len(cancel_results),
        "cancel_errors": sum(1 for r in cancel_results if "error" in r["status"]),
        "disconnect_count": len(disconnect_results),
        "disconnect_errors": sum(1 for r in disconnect_results if "error" in r["status"]),
        "short_text_count": len(short_results),
        "short_text_rejected": sum(1 for r in short_results if r["status"] == "rejected_as_expected"),
        "short_text_accepted": sum(1 for r in short_results if r["status"] == "accepted_unexpectedly"),
        "rapid_fire_count": len(rapid_results),
        "rapid_fire_errors": sum(1 for r in rapid_results if "error" in r["status"]),
    }

    if cancel_latencies:
        sorted_lat = sorted(cancel_latencies)
        report["cancel_p50_ms"] = sorted_lat[len(sorted_lat) // 2]
        report["cancel_p95_ms"] = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        report["cancel_max_ms"] = max(sorted_lat)

    # Gate checks
    # With auto-restart, the gate is "final healthy + all rounds completed"
    all_rounds_completed = len(results) == args.rounds
    gate_service_available = final_healthy and all_rounds_completed
    gate_cancel_p95 = report.get("cancel_p95_ms", 0) <= 60
    report["gate_service_available"] = gate_service_available
    report["gate_cancel_p95_le_60ms"] = gate_cancel_p95
    report["restarts"] = health_failures
    report["all_rounds_completed"] = all_rounds_completed

    print(f"\n{'='*60}")
    print(f"  TTS Stress Test Report")
    print(f"{'='*60}")
    print(f"  Rounds:     {len(results)}/{args.rounds}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    print(f"  Conn errors:{crashes}")
    print(f"  Restarts:   {health_failures}")
    print(f"  Final:      {'✅ healthy' if final_healthy else '❌ dead'}")
    if cancel_latencies:
        print(f"  Cancel P50: {report['cancel_p50_ms']:.1f}ms")
        print(f"  Cancel P95: {report['cancel_p95_ms']:.1f}ms")
    print(f"  Gate service available: {'✅ PASS' if gate_service_available else '❌ FAIL'}")
    if cancel_latencies:
        print(f"  Gate cancel ≤60ms:     {'✅ PASS' if gate_cancel_p95 else '❌ FAIL'}")

    # Save
    report_path = os.path.join(OUTPUT_DIR, "day3_stress_cancel_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")

    return 0 if gate_service_available else 1


if __name__ == "__main__":
    sys.exit(main())

