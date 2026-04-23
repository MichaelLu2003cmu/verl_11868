#!/usr/bin/env python3
"""Parse training logs and emit p50/p90/p99 iteration-time jitter + KL/score plots.

Closes two proposal §7.2 evaluation gaps from pre-existing logs (no reruns required):
  (1) p50/p99 iteration latency jitter  — proposal explicitly lists this metric but
      only mean step_time was previously reported.
  (2) KL divergence and task-success-rate stability  — proposal lists these alongside
      reward trajectory but only reward was plotted.

Usage:
  python scripts/jitter_and_kl.py \
      --logs checkpoints/verl_examples/gsm8k_2gpu_baseline_100/train_log.txt \
             checkpoints/verl_examples/gsm8k_2gpu_fp16_100/train_log.txt \
             checkpoints/verl_examples/gsm8k_2gpu_int8_100/train_log.txt \
             checkpoints/verl_examples/gsm8k_2gpu_bf16_100/train_log.txt \
      --labels "Baseline (push)" "+LP+FP16" "+LP+INT8" "+LP+BF16" \
      --plot-out 7_stability_kl_score.png \
      --skip-first 1
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"\bstep:(\d+)\b")
TIME_RE = re.compile(r"perf/time_per_step:([\d.\-eE+]+)")
REWARD_RE = re.compile(r"critic/rewards/mean:([\d.\-eE+]+)")
SCORE_RE = re.compile(r"critic/score/mean:([\d.\-eE+]+)")
KL_RE = re.compile(r"actor/ppo_kl:(?:np\.float64\()?([\-\d.eE+]+)")
THROUGHPUT_RE = re.compile(r"perf/throughput:([\d.\-eE+]+)")

COLORS = ["#4878CF", "#D65F5F", "#B47CC7", "#FFA500", "#6ACC65", "#956CB4"]


def parse_log(path: Path, skip_first: int = 1) -> dict[str, list[float]]:
    """Return dict of per-step series parsed from a verl train_log.txt file."""
    steps: list[int] = []
    times: list[float] = []
    rewards: list[float] = []
    scores: list[float] = []
    kls: list[float] = []
    throughputs: list[float] = []

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
        rm = REWARD_RE.search(raw)
        sc = SCORE_RE.search(raw)
        km = KL_RE.search(raw)
        th = THROUGHPUT_RE.search(raw)
        rewards.append(float(rm.group(1)) if rm else float("nan"))
        scores.append(float(sc.group(1)) if sc else float("nan"))
        kls.append(float(km.group(1)) if km else float("nan"))
        throughputs.append(float(th.group(1)) if th else float("nan"))

    return {
        "steps": steps,
        "time": times,
        "reward": rewards,
        "score": scores,
        "kl": kls,
        "throughput": throughputs,
    }


def percentile(values: list[float], pct: float) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan")
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def rolling(vals: list[float], w: int = 7) -> list[float]:
    out = []
    for i, _ in enumerate(vals):
        lo = max(0, i - w + 1)
        window = [v for v in vals[lo : i + 1] if v == v]
        out.append(sum(window) / len(window) if window else float("nan"))
    return out


def jitter_table(runs: dict[str, dict[str, list[float]]]) -> str:
    lines = []
    lines.append("## Iteration-Time Jitter (p50/p90/p99, mean, max, std)")
    lines.append(
        "| Config | n | mean (s) | p50 (s) | p90 (s) | p99 (s) | max (s) | std (s) | p99/p50 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    base_mean = None
    for label, data in runs.items():
        t = data["time"]
        if not t:
            continue
        mean = statistics.mean(t)
        std = statistics.pstdev(t) if len(t) > 1 else 0.0
        p50 = percentile(t, 50.0)
        p90 = percentile(t, 90.0)
        p99 = percentile(t, 99.0)
        p_max = max(t)
        ratio = p99 / p50 if p50 else float("nan")
        if base_mean is None:
            base_mean = mean
        lines.append(
            f"| {label} | {len(t)} | {mean:.3f} | {p50:.3f} | {p90:.3f} | {p99:.3f} | "
            f"{p_max:.3f} | {std:.3f} | {ratio:.2f} |"
        )
    lines.append(
        "\n_`p99/p50` is a unitless tail-heaviness ratio — values ≈ 1.0 indicate stable "
        "steady-state, larger values indicate tail-latency jitter._"
    )
    return "\n".join(lines)


def stability_table(runs: dict[str, dict[str, list[float]]]) -> str:
    lines = []
    lines.append("\n## Training Stability (reward / score / KL, averaged over steps)")
    lines.append("| Config | n | reward_mean | score_mean | kl_mean (|·|) | kl_std | throughput_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, data in runs.items():
        n = len(data["time"])
        if not n:
            continue
        rewards = [v for v in data["reward"] if v == v]
        scores = [v for v in data["score"] if v == v]
        kls = [v for v in data["kl"] if v == v]
        thr = [v for v in data["throughput"] if v == v]
        rmean = statistics.mean(rewards) if rewards else float("nan")
        smean = statistics.mean(scores) if scores else float("nan")
        kabs = [abs(v) for v in kls]
        kmean = statistics.mean(kabs) if kabs else float("nan")
        kstd = statistics.pstdev(kls) if len(kls) > 1 else 0.0
        tmean = statistics.mean(thr) if thr else float("nan")
        lines.append(
            f"| {label} | {n} | {rmean:.4f} | {smean:.4f} | {kmean:.5f} | {kstd:.5f} | {tmean:.1f} |"
        )
    lines.append(
        "\n_`kl_mean` is averaged `|actor/ppo_kl|` per step (smaller is more on-policy); "
        "`kl_std` is the raw (signed) standard deviation of KL across steps.  "
        "`throughput_mean` is `perf/throughput` (tokens/s) averaged per step._"
    )
    return "\n".join(lines)


def plot_stability(runs: dict[str, dict[str, list[float]]], out: Path, title: str) -> None:
    labels = list(runs.keys())
    palette = {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for label, data in runs.items():
        c = palette[label]
        steps = data["steps"]
        if not steps:
            continue
        axes[0].scatter(steps, data["reward"], color=c, alpha=0.15, s=10)
        axes[0].plot(steps, rolling(data["reward"]), color=c, lw=1.8, label=label)
        axes[1].scatter(steps, data["score"], color=c, alpha=0.15, s=10)
        axes[1].plot(steps, rolling(data["score"]), color=c, lw=1.8, label=label)
        axes[2].scatter(steps, data["kl"], color=c, alpha=0.15, s=10)
        axes[2].plot(steps, rolling(data["kl"]), color=c, lw=1.8, label=label)

    axes[0].set_title("(a) critic/rewards/mean"); axes[0].set_xlabel("step")
    axes[0].set_ylabel("reward"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title("(b) critic/score/mean (GSM8k proxy)"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("score"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[2].set_title("(c) actor/ppo_kl (on-policy drift)"); axes[2].set_xlabel("step")
    axes[2].set_ylabel("ppo_kl"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    axes[2].axhline(0.0, color="gray", lw=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True, help="train_log.txt paths")
    p.add_argument("--labels", nargs="+", required=True, help="human labels (same length as --logs)")
    p.add_argument("--skip-first", type=int, default=1, help="drop N initial warmup steps")
    p.add_argument("--plot-out", default="7_stability_kl_score.png", help="plot destination")
    p.add_argument(
        "--title",
        default="Training Stability (100 steps, batch=32, 2×V100)",
        help="suptitle for the plot",
    )
    args = p.parse_args()

    if len(args.logs) != len(args.labels):
        raise SystemExit("--logs and --labels must have the same length")

    runs: dict[str, dict[str, list[float]]] = {}
    for lab, pth in zip(args.labels, args.logs):
        path = Path(pth)
        if not path.exists():
            print(f"WARN: {path} missing; skipping {lab}", file=sys.stderr)
            continue
        runs[lab] = parse_log(path, skip_first=args.skip_first)

    print(jitter_table(runs))
    print(stability_table(runs))
    plot_stability(runs, Path(args.plot_out), title=args.title)


if __name__ == "__main__":
    main()
