#!/usr/bin/env python3
"""Summarize VERL transfer probe logs into baseline-friendly tables.

Usage:
  python scripts/summarize_transfer_probe.py --log /path/to/transfer_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _iter_events(log_path: Path):
    with log_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            # Expected format:
            # [transfer_probe] {"event":"...", ...}
            if "[transfer_probe]" not in raw:
                continue
            try:
                payload = raw.split("[transfer_probe]", 1)[1].strip()
                yield json.loads(payload)
            except Exception:
                continue


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize(log_path: Path) -> None:
    transfer_stats = defaultdict(
        lambda: {
            "count": 0,
            "dispatch_ms": 0.0,
            "wait_ms": 0.0,
            "collect_ms": 0.0,
            "total_ms": 0.0,
            "send_bytes": 0.0,
            "recv_bytes": 0.0,
        }
    )
    cpu_stats = defaultdict(lambda: {"count": 0, "elapsed_ms": 0.0, "output_bytes": 0.0})
    stage_samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    compress_samples: dict[str, list[dict]] = defaultdict(list)

    for e in _iter_events(log_path):
        event = e.get("event")
        if event == "compress_stats":
            ctx = str(e.get("context", "unknown"))
            compress_samples[ctx].append({
                "mode": e.get("mode", ""),
                "orig_bytes": float(e.get("orig_bytes", 0)),
                "comp_bytes": float(e.get("comp_bytes", 0)),
                "ratio": float(e.get("ratio", 1.0)),
            })
        elif event == "transfer_latency":
            name = str(e.get("method_name", "unknown"))
            s = transfer_stats[name]
            s["count"] += 1
            s["dispatch_ms"] += float(e.get("dispatch_ms", 0.0))
            s["wait_ms"] += float(e.get("wait_ms", 0.0))
            s["collect_ms"] += float(e.get("collect_ms", 0.0))
            s["total_ms"] += float(e.get("total_ms", 0.0))
            s["send_bytes"] += float(e.get("send_bytes", 0.0))
            s["recv_bytes"] += float(e.get("recv_bytes", 0.0))
        elif event == "cpu_overhead":
            name = str(e.get("location", "unknown"))
            s = cpu_stats[name]
            s["count"] += 1
            s["elapsed_ms"] += float(e.get("elapsed_ms", 0.0))
            s["output_bytes"] += float(e.get("output_bytes", 0.0))
        elif event == "critical_path_stage":
            stage = str(e.get("stage", "unknown"))
            wait_ms = float(e.get("wait_ms", 0.0))
            dispatched_at_ms = float(e.get("dispatched_at_ms", 0.0))
            joined_at_ms = float(e.get("joined_at_ms", 0.0))
            in_flight_ms = max(joined_at_ms - dispatched_at_ms, 0.0)
            hidden_ms = max(in_flight_ms - wait_ms, 0.0)
            stage_samples[stage].append(
                {
                    "wait_ms": wait_ms,
                    "in_flight_ms": in_flight_ms,
                    "hidden_ms": hidden_ms,
                    "dispatched_at_ms": dispatched_at_ms,
                    "joined_at_ms": joined_at_ms,
                }
            )

    print("## Transfer Latency Baseline")
    print(
        "| method_name | iters | dispatch_ms/iter | wait_ms/iter | collect_ms/iter | total_ms/iter | send_MB/iter | recv_MB/iter |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in sorted(transfer_stats.items(), key=lambda kv: kv[0]):
        n = s["count"]
        print(
            f"| {name} | {n} | "
            f"{_safe_div(s['dispatch_ms'], n):.3f} | "
            f"{_safe_div(s['wait_ms'], n):.3f} | "
            f"{_safe_div(s['collect_ms'], n):.3f} | "
            f"{_safe_div(s['total_ms'], n):.3f} | "
            f"{_safe_div(s['send_bytes'], n) / (1024**2):.3f} | "
            f"{_safe_div(s['recv_bytes'], n) / (1024**2):.3f} |"
        )

    print("\n## CPU Overhead Baseline")
    print("| location | iters | elapsed_ms/iter | output_MB/iter |")
    print("|---|---:|---:|---:|")
    for name, s in sorted(cpu_stats.items(), key=lambda kv: kv[0]):
        n = s["count"]
        print(
            f"| {name} | {n} | "
            f"{_safe_div(s['elapsed_ms'], n):.3f} | "
            f"{_safe_div(s['output_bytes'], n) / (1024**2):.3f} |"
        )

    if stage_samples:
        print("\n## Critical-Path Stage Overlap")
        print(
            "| stage | iters | wait_ms p50 | wait_ms p90 | wait_ms p99 | "
            "in_flight_ms avg | hidden_ms avg | hidden_frac |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for stage, samples in sorted(stage_samples.items(), key=lambda kv: kv[0]):
            n = len(samples)
            waits = [s["wait_ms"] for s in samples]
            in_flights = [s["in_flight_ms"] for s in samples]
            hiddens = [s["hidden_ms"] for s in samples]
            wait_p50 = _percentile(waits, 50.0)
            wait_p90 = _percentile(waits, 90.0)
            wait_p99 = _percentile(waits, 99.0)
            in_flight_avg = sum(in_flights) / n
            hidden_avg = sum(hiddens) / n
            hidden_frac = _safe_div(hidden_avg, in_flight_avg)
            print(
                f"| {stage} | {n} | "
                f"{wait_p50:.3f} | {wait_p90:.3f} | {wait_p99:.3f} | "
                f"{in_flight_avg:.3f} | {hidden_avg:.3f} | {hidden_frac:.3f} |"
            )
        print(
            "\n_Legend: `wait_ms` = blocking time at the join barrier; "
            "`in_flight_ms` = joined_at_ms - dispatched_at_ms (total time the future was outstanding); "
            "`hidden_ms` = in_flight_ms - wait_ms (time hidden under other compute); "
            "`hidden_frac` = hidden_ms / in_flight_ms._"
        )

    if compress_samples:
        print("\n## Compression Stats (pull dispatch)")
        print("| context | mode | iters | orig_KB/iter | comp_KB/iter | ratio | saved_pct |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for ctx, samples in sorted(compress_samples.items()):
            n = len(samples)
            mode = samples[0]["mode"]
            orig_avg = sum(s["orig_bytes"] for s in samples) / n / 1024
            comp_avg = sum(s["comp_bytes"] for s in samples) / n / 1024
            ratio_avg = sum(s["ratio"] for s in samples) / n
            saved_pct = (1.0 - ratio_avg) * 100
            print(
                f"| {ctx} | {mode} | {n} | "
                f"{orig_avg:.2f} | {comp_avg:.2f} | {ratio_avg:.3f} | {saved_pct:.1f}% |"
            )
        print(
            "\n_Legend: `ratio` = compressed_bytes / original_bytes; "
            "`saved_pct` = percentage reduction in tensor payload size._"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VERL transfer probe logs.")
    parser.add_argument("--log", required=True, help="Path to transfer probe log file.")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log file does not exist: {log_path}")
    summarize(log_path)


if __name__ == "__main__":
    main()
