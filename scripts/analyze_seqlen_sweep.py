#!/usr/bin/env python3
"""Analyse the Rerun-D sequence-length sweep.

Reads the 6 output directories produced by `scripts/run_seqlen_sweep.sh`
and produces:

  (1) a Markdown scaling table suitable for pasting into 7_eval.md §7.4, and
  (2) a 3-panel matplotlib figure saved to `7_seqlen_scaling.png` that
      visualises:
        (a) per-iteration transfer bytes vs response length (push / pull+fp16),
        (b) transfer_ms (dispatch+collect) vs response length,
        (c) mean end-to-end step_time vs response length, with
            Δ step_time annotations per seqlen point.

Usage:
  python scripts/analyze_seqlen_sweep.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESPONSE_LENGTHS = [128, 512, 1024]
VARIANTS = ["base", "lpfp16"]
LABEL = {"base": "Baseline (push, no compress)", "lpfp16": "LP + FP16 (pull, fp16)"}
COLOR = {"base": "#4878CF", "lpfp16": "#D65F5F"}

ROOT = Path("checkpoints/verl_examples")


def _parse_probe(path: Path) -> dict:
    """Return aggregate per-transfer stats for compute_log_prob method only,
    plus total compress_stats event count."""
    dispatch_ms = []
    collect_ms = []
    wait_ms = []
    send_bytes = []
    recv_bytes = []
    compress_orig = []
    compress_comp = []
    compress_count = 0

    if not path.exists():
        return {
            "probe_events": 0,
            "n_xfer": 0,
            "dispatch_ms_mean": float("nan"),
            "collect_ms_mean": float("nan"),
            "wait_ms_mean": float("nan"),
            "send_bytes_mean": float("nan"),
            "recv_bytes_mean": float("nan"),
            "compress_events": 0,
            "orig_bytes_mean": float("nan"),
            "comp_bytes_mean": float("nan"),
            "saved_pct": float("nan"),
        }

    n_events = 0
    for raw in path.read_text(errors="ignore").splitlines():
        if "[transfer_probe]" not in raw:
            continue
        n_events += 1
        try:
            payload = json.loads(raw.split("[transfer_probe]", 1)[1].strip())
        except Exception:
            continue
        ev = payload.get("event", "")
        if ev == "transfer_latency":
            if payload.get("method_name") != "actor_rollout_compute_log_prob":
                continue
            dispatch_ms.append(float(payload.get("dispatch_ms", 0.0)))
            collect_ms.append(float(payload.get("collect_ms", 0.0)))
            wait_ms.append(float(payload.get("wait_ms", 0.0)))
            send_bytes.append(float(payload.get("send_bytes", 0.0)))
            recv_bytes.append(float(payload.get("recv_bytes", 0.0)))
        elif ev == "compress_stats":
            compress_count += 1
            compress_orig.append(float(payload.get("orig_bytes", 0.0)))
            compress_comp.append(float(payload.get("comp_bytes", 0.0)))

    def _avg(xs):
        return statistics.mean(xs) if xs else float("nan")

    comp_saved = (
        (1.0 - _avg(compress_comp) / _avg(compress_orig)) * 100.0
        if compress_orig and compress_comp and _avg(compress_orig) > 0
        else float("nan")
    )

    return {
        "probe_events": n_events,
        "n_xfer": len(dispatch_ms),
        "dispatch_ms_mean": _avg(dispatch_ms),
        "collect_ms_mean": _avg(collect_ms),
        "wait_ms_mean": _avg(wait_ms),
        "send_bytes_mean": _avg(send_bytes),
        "recv_bytes_mean": _avg(recv_bytes),
        "compress_events": compress_count,
        "orig_bytes_mean": _avg(compress_orig),
        "comp_bytes_mean": _avg(compress_comp),
        "saved_pct": comp_saved,
    }


STEP_RE = re.compile(r"\bstep:(\d+)\b")
TIME_RE = re.compile(r"perf/time_per_step:([\d.\-eE+]+)")
THROUGHPUT_RE = re.compile(r"perf/throughput:([\d.\-eE+]+)")


def _parse_train_log(path: Path, skip_first: int = 1) -> dict:
    """Pull per-step wall-clock times (skip vLLM warmup step)."""
    times = []
    throughputs = []
    steps = []
    if not path.exists():
        return {"n": 0, "step_time_mean": float("nan"), "step_time_p50": float("nan"),
                "step_time_p90": float("nan"), "step_time_p99": float("nan"),
                "throughput_mean": float("nan"), "steps_observed": 0}

    for raw in path.read_text(errors="ignore").splitlines():
        if "perf/time_per_step" not in raw:
            continue
        sm = STEP_RE.search(raw)
        tm = TIME_RE.search(raw)
        if not (sm and tm):
            continue
        step = int(sm.group(1))
        if step < 1 + skip_first:
            continue
        steps.append(step)
        times.append(float(tm.group(1)))
        th = THROUGHPUT_RE.search(raw)
        if th:
            throughputs.append(float(th.group(1)))

    def _pct(xs, p):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {
        "n": len(times),
        "step_time_mean": statistics.mean(times) if times else float("nan"),
        "step_time_p50": _pct(times, 50.0),
        "step_time_p90": _pct(times, 90.0),
        "step_time_p99": _pct(times, 99.0),
        "throughput_mean": statistics.mean(throughputs) if throughputs else float("nan"),
        "steps_observed": max(steps) if steps else 0,
    }


def _kb(x: float) -> float:
    return x / 1024.0


def main() -> None:
    rows = []
    missing = []
    for resp_len in RESPONSE_LENGTHS:
        for variant in VARIANTS:
            run_dir = ROOT / f"gsm8k_seqlen_{resp_len}_{variant}"
            log = run_dir / "train_log.txt"
            probe = run_dir / "transfer_probe.jsonl"
            if not log.exists():
                missing.append(str(run_dir))
                continue
            probe_stats = _parse_probe(probe)
            log_stats = _parse_train_log(log)
            rows.append({
                "resp_len": resp_len,
                "variant": variant,
                "run_dir": str(run_dir),
                **probe_stats,
                **log_stats,
            })

    if missing:
        print(f"WARN: missing runs:\n  " + "\n  ".join(missing), file=sys.stderr)

    print("\n## Sequence-Length Scaling (Rerun D; batch=8, 10 steps, step 1 excluded)")
    print()
    print(
        "| resp_len | variant | n_steps | n_xfer | send_KB | recv_KB | orig_KB | "
        "comp_KB | saved% | dispatch_ms | collect_ms | wait_ms | step_time_mean (s) | "
        "step_time_p50 | step_time_p99 |"
    )
    print(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        print(
            f"| {r['resp_len']} | {r['variant']} | {r['n']} | {r['n_xfer']} | "
            f"{_kb(r['send_bytes_mean']):.2f} | {_kb(r['recv_bytes_mean']):.2f} | "
            f"{_kb(r['orig_bytes_mean']):.2f} | {_kb(r['comp_bytes_mean']):.2f} | "
            f"{r['saved_pct']:.1f}% | "
            f"{r['dispatch_ms_mean']:.3f} | {r['collect_ms_mean']:.3f} | "
            f"{r['wait_ms_mean']:.1f} | {r['step_time_mean']:.3f} | "
            f"{r['step_time_p50']:.3f} | {r['step_time_p99']:.3f} |"
        )

    # Paired Δ table (baseline vs lp+fp16 at same seqlen)
    print()
    print("## FP16 speed-up by sequence length (paired deltas)")
    print()
    print("| resp_len | push step_time (s) | LP+FP16 step_time (s) | Δ step_time | "
          "push xfer_ms | LP+FP16 xfer_ms | Δ xfer_ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    by_key = {(r["resp_len"], r["variant"]): r for r in rows}
    for resp_len in RESPONSE_LENGTHS:
        b = by_key.get((resp_len, "base"))
        f = by_key.get((resp_len, "lpfp16"))
        if not (b and f):
            continue
        b_xfer = b["dispatch_ms_mean"] + b["collect_ms_mean"]
        f_xfer = f["dispatch_ms_mean"] + f["collect_ms_mean"]
        delta_step = (f["step_time_mean"] - b["step_time_mean"]) / b["step_time_mean"] * 100.0
        delta_xfer = (f_xfer - b_xfer) / b_xfer * 100.0 if b_xfer else float("nan")
        print(
            f"| {resp_len} | {b['step_time_mean']:.3f} | {f['step_time_mean']:.3f} | "
            f"{delta_step:+.1f}% | {b_xfer:.2f} | {f_xfer:.2f} | {delta_xfer:+.1f}% |"
        )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Sequence-Length Scaling — batch=8, 10 steps, 2×V100 (Rerun D)",
        fontsize=12, fontweight="bold",
    )

    xs = RESPONSE_LENGTHS
    for ax_idx, (metric_key, ylabel, scale_fn) in enumerate([
        ("send_bytes_mean", "send_KB / transfer", _kb),
        ("dispatch_ms_mean", "dispatch_ms + collect_ms / transfer", None),
        ("step_time_mean", "step time (s) — step 1 excluded", None),
    ]):
        ax = axes[ax_idx]
        for variant in VARIANTS:
            ys = []
            for resp_len in RESPONSE_LENGTHS:
                r = by_key.get((resp_len, variant))
                if not r:
                    ys.append(float("nan"))
                    continue
                if metric_key == "send_bytes_mean":
                    y = scale_fn(r["send_bytes_mean"])
                elif metric_key == "dispatch_ms_mean":
                    y = r["dispatch_ms_mean"] + r["collect_ms_mean"]
                else:
                    y = r["step_time_mean"]
                ys.append(y)
            ax.plot(xs, ys, "o-", color=COLOR[variant], label=LABEL[variant], lw=1.8, ms=6)
        ax.set_xlabel("max_response_length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        if ax_idx == 0:
            ax.set_title("(a) transfer payload size")
        elif ax_idx == 1:
            ax.set_title("(b) transfer latency (dispatch+collect)")
        else:
            ax.set_title("(c) end-to-end step time")

    plt.tight_layout()
    out = Path("7_seqlen_scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
