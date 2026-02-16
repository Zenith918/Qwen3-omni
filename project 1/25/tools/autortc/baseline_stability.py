#!/usr/bin/env python3
"""
D11: Baseline Stability Analysis

输入：多个 run 的 metrics.csv（mini 或 full）
输出：output/baseline_stability/baseline_stability.md

每个指标计算 min / median / P95 / P99 / max，并给出建议阈值（median + 2σ 或 P95 + 安全余量）。
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


# ── 需要统计的指标 ──
METRICS_SPEC = [
    # (csv_column, display_name, category, unit, higher_is_worse)
    ("eot_to_first_audio_ms", "EoT→FirstAudio", "延迟", "ms", True),
    ("fast_lane_ttft_ms", "Fast Lane TTFT", "延迟", "ms", True),
    ("tts_first_to_publish_ms", "TTS First→Publish", "延迟", "ms", True),
    ("reply_max_gap_ms", "Reply Max Gap", "音质", "ms", True),
    ("reply_audible_dropout_count", "Audible Dropout Count", "音质", "count", True),
    ("clipping_ratio", "Clipping Ratio", "音质", "ratio", True),
    ("mel_distance", "Mel Distance (pre vs post)", "音质", "", True),
    ("hf_ratio_drop", "HF Ratio Drop", "音质", "", True),
    ("reply_rms", "Reply RMS", "可靠性", "", False),
    ("full_rms", "Full RMS", "可靠性", "", False),
]


def _load_metrics_csv(csv_path: str) -> list[dict]:
    """Load a metrics.csv and return list of row dicts."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(val) -> float | None:
    """Convert to float, return None for empty/invalid."""
    if val is None or val == "" or val == "-1" or val == "-1.0":
        return None
    try:
        v = float(val)
        if v < 0:
            return None
        return v
    except (ValueError, TypeError):
        return None


def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def _stats(values: list[float]) -> dict:
    """Compute min/median/mean/std/P95/P99/max for a list of values."""
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "std": None,
                "p95": None, "p99": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": len(arr),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": _pct(arr, 95),
        "p99": _pct(arr, 99),
        "max": float(np.max(arr)),
    }


def _suggest_threshold(stats: dict, higher_is_worse: bool) -> str:
    """建议阈值 = median + 2σ 或 P95 + 安全余量"""
    if stats["n"] == 0 or stats["median"] is None:
        return "N/A"
    median = stats["median"]
    std = stats["std"]
    p95 = stats["p95"]

    if higher_is_worse:
        # 方法1: median + 2σ
        t1 = median + 2 * std if std else median * 1.5
        # 方法2: P95 + 20% safety margin
        t2 = p95 * 1.2
        threshold = max(t1, t2)
        return f"{threshold:.2f}"
    else:
        # lower_is_worse (e.g. rms): use median - 2σ as floor
        t1 = max(0, median - 2 * std) if std else median * 0.5
        return f"{t1:.4f}"


def collect_all_values(all_runs_rows: list[list[dict]], metric_col: str,
                       tier_filter: str = None) -> list[float]:
    """Collect all values for a given metric across all runs."""
    values = []
    for run_rows in all_runs_rows:
        for row in run_rows:
            if tier_filter and row.get("tier", "") != tier_filter:
                continue
            val = _safe_float(row.get(metric_col))
            if val is not None:
                values.append(val)
    return values


def compute_per_run_aggregates(all_runs_rows: list[list[dict]], metric_col: str,
                                tier_filter: str = None) -> list[float]:
    """Compute per-run P95 aggregates for a metric."""
    per_run = []
    for run_rows in all_runs_rows:
        vals = []
        for row in run_rows:
            if tier_filter and row.get("tier", "") != tier_filter:
                continue
            val = _safe_float(row.get(metric_col))
            if val is not None:
                vals.append(val)
        if vals:
            per_run.append(_pct(np.array(vals), 95))
    return per_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D11: Baseline stability analysis.")
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="Directories containing metrics.csv (from mini or full runs)")
    p.add_argument("--output_dir", default="output/baseline_stability",
                   help="Output directory for baseline_stability.md")
    p.add_argument("--output_json", default="", help="Optional: also write stats as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs' metrics
    all_runs_rows = []
    valid_runs = []
    for rd in args.run_dirs:
        csv_path = os.path.join(rd, "metrics.csv")
        rows = _load_metrics_csv(csv_path)
        if rows:
            all_runs_rows.append(rows)
            valid_runs.append(rd)
        else:
            print(f"WARN: no metrics.csv in {rd}, skipping")

    if not all_runs_rows:
        print("ERROR: no valid metrics.csv found in any run directory")
        return 1

    print(f"Loaded {len(all_runs_rows)} runs with metrics data")

    # Compute statistics
    all_stats = {}
    for metric_col, display_name, category, unit, higher_is_worse in METRICS_SPEC:
        values = collect_all_values(all_runs_rows, metric_col, tier_filter="P0")
        s = _stats(values)
        s["display_name"] = display_name
        s["category"] = category
        s["unit"] = unit
        s["suggested_threshold"] = _suggest_threshold(s, higher_is_worse)
        s["higher_is_worse"] = higher_is_worse
        all_stats[metric_col] = s

    # Also compute audio_valid_rate and retry_rate across runs
    audio_valid_rates = []
    retry_rates = []
    for run_rows in all_runs_rows:
        p0_rows = [r for r in run_rows if r.get("tier") == "P0"]
        has_audio = sum(1 for r in p0_rows
                        if _safe_float(r.get("reply_rms")) is not None and float(r.get("reply_rms", 0)) >= 0.01
                        or _safe_float(r.get("full_rms")) is not None and float(r.get("full_rms", 0)) >= 0.01)
        if p0_rows:
            audio_valid_rates.append(has_audio / len(p0_rows))

    # ── Write report ──
    report_path = out_dir / "baseline_stability.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Baseline Stability Report (D11)\n\n")
        f.write(f"- Runs analyzed: **{len(all_runs_rows)}**\n")
        f.write(f"- Run directories:\n")
        for rd in valid_runs:
            f.write(f"  - `{rd}`\n")
        f.write("\n---\n\n")

        # ── Per-category tables ──
        for cat in ["延迟", "音质", "可靠性"]:
            cat_metrics = [(k, v) for k, v in all_stats.items() if v["category"] == cat]
            if not cat_metrics:
                continue
            f.write(f"## {cat}\n\n")
            f.write("| 指标 | N | Min | Median | Mean | Std | P95 | P99 | Max | 建议阈值 |\n")
            f.write("|------|---|-----|--------|------|-----|-----|-----|-----|----------|\n")
            for metric_col, s in cat_metrics:
                if s["n"] == 0:
                    f.write(f"| {s['display_name']} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")
                    continue
                fmt = ".1f" if s.get("unit") == "ms" or s.get("unit") == "count" else ".4f"
                f.write(f"| {s['display_name']} "
                        f"| {s['n']} "
                        f"| {s['min']:{fmt}} "
                        f"| {s['median']:{fmt}} "
                        f"| {s['mean']:{fmt}} "
                        f"| {s['std']:{fmt}} "
                        f"| {s['p95']:{fmt}} "
                        f"| {s['p99']:{fmt}} "
                        f"| {s['max']:{fmt}} "
                        f"| {s['suggested_threshold']} |\n")
            f.write("\n")

        # ── Aggregate rates ──
        f.write("## 可靠性（聚合）\n\n")
        if audio_valid_rates:
            avr_arr = np.array(audio_valid_rates)
            f.write(f"- Audio Valid Rate: min={np.min(avr_arr):.1%}, "
                    f"median={np.median(avr_arr):.1%}, max={np.max(avr_arr):.1%}\n")

        # ── Suggested gate thresholds ──
        f.write("\n---\n\n")
        f.write("## 建议 Gate 阈值\n\n")
        f.write("基于 median + 2σ（或 P95 × 1.2，取较大值）：\n\n")
        f.write("| Gate | 当前阈值 | 建议阈值 | 依据 |\n")
        f.write("|------|---------|---------|------|\n")

        eot_s = all_stats.get("eot_to_first_audio_ms", {})
        f.write(f"| EoT→FirstAudio P95 | ≤ 650ms | ≤ {eot_s.get('suggested_threshold', 'N/A')}ms "
                f"| median={eot_s.get('median', 'N/A')}, σ={eot_s.get('std', 'N/A')} |\n")

        tts_s = all_stats.get("tts_first_to_publish_ms", {})
        f.write(f"| TTS First→Publish P95 | ≤ 120ms | ≤ {tts_s.get('suggested_threshold', 'N/A')}ms "
                f"| median={tts_s.get('median', 'N/A')}, σ={tts_s.get('std', 'N/A')} |\n")

        gap_s = all_stats.get("reply_max_gap_ms", {})
        f.write(f"| Max Gap (reply) | < 200ms | < {gap_s.get('suggested_threshold', 'N/A')}ms "
                f"| median={gap_s.get('median', 'N/A')}, σ={gap_s.get('std', 'N/A')} |\n")

        clip_s = all_stats.get("clipping_ratio", {})
        f.write(f"| Clipping Ratio | < 0.1% | < {clip_s.get('suggested_threshold', 'N/A')} "
                f"| median={clip_s.get('median', 'N/A')}, σ={clip_s.get('std', 'N/A')} |\n")

        fast_s = all_stats.get("fast_lane_ttft_ms", {})
        f.write(f"| Fast Lane TTFT P95 | ≤ 80ms | ≤ {fast_s.get('suggested_threshold', 'N/A')}ms "
                f"| median={fast_s.get('median', 'N/A')}, σ={fast_s.get('std', 'N/A')} |\n")

        f.write("\n> 注：建议阈值仅供参考，需结合实际业务需求和历史数据调整。\n")

    print(f"Report written to: {report_path}")

    # ── Optional JSON output ──
    if args.output_json:
        json_path = args.output_json
    else:
        json_path = str(out_dir / "baseline_stability.json")

    # Convert numpy types for JSON serialization
    json_stats = {}
    for k, v in all_stats.items():
        json_stats[k] = {sk: (float(sv) if isinstance(sv, (np.floating, np.integer)) else sv)
                         for sk, sv in v.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    print(f"JSON stats written to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


