#!/usr/bin/env python3
"""Parse verl console train_log.txt files and plot reward curves.

Usage:
  python scripts/plot_reward_curve.py \
      --logs  checkpoints/.../nocomp/train_log.txt \
               checkpoints/.../fp16/train_log.txt \
      --labels "No Compress" "FP16 Compress" \
      --out    7_reward_curve.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(path: Path) -> dict:
    steps, rewards, scores, step_time = [], [], [], []
    # Match each field independently — order in the log line doesn't matter
    step_re   = re.compile(r"\bstep:(\d+)\b")
    reward_re = re.compile(r"critic/rewards/mean:([\d.\-e]+)")
    score_re  = re.compile(r"critic/score/mean:([\d.\-e]+)")
    time_re   = re.compile(r"timing_s/step:([\d.\-e]+)")
    with path.open() as f:
        for line in f:
            if "critic/rewards/mean" not in line:
                continue
            sm  = step_re.search(line)
            rm  = reward_re.search(line)
            scm = score_re.search(line)
            tm  = time_re.search(line)
            if sm and rm and scm and tm:
                steps.append(int(sm.group(1)))
                rewards.append(float(rm.group(1)))
                scores.append(float(scm.group(1)))
                step_time.append(float(tm.group(1)))
    return {"steps": steps, "rewards": rewards, "scores": scores, "step_time": step_time}


def smooth(values, w=3):
    if len(values) < w:
        return values
    return np.convolve(values, np.ones(w) / w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs",   nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out",    default="7_reward_curve.png")
    args = parser.parse_args()
    assert len(args.logs) == len(args.labels), "need same number of logs and labels"

    colors = ["#4C72B0", "#C44E52", "#55A868", "#DD8452"]
    parsed = [parse_log(Path(p)) for p in args.logs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Training Stability: No Compress vs FP16 Compress\n"
        "(Qwen2.5-1.5B-Instruct · GRPO · 2×V100 · batch=8)",
        fontsize=11, fontweight="bold"
    )

    # ── panel 1: rewards/mean ────────────────────────────────────────────────
    ax = axes[0]
    for d, label, color in zip(parsed, args.labels, colors):
        ax.plot(d["steps"], d["rewards"], "o--", color=color, alpha=0.35,
                markersize=4, linewidth=1)
        sw = 3
        if len(d["rewards"]) >= sw:
            xs = d["steps"][sw - 1:]
            ax.plot(xs, smooth(d["rewards"], sw), "-", color=color,
                    linewidth=2, label=f"{label} (smooth-{sw})")
        else:
            ax.plot(d["steps"], d["rewards"], "-", color=color, linewidth=2, label=label)
    ax.set_xlabel("Training Step"); ax.set_ylabel("critic/rewards/mean")
    ax.set_title("(a) Reward (mean)", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.05, 1.05)

    # ── panel 2: score/mean ──────────────────────────────────────────────────
    ax = axes[1]
    for d, label, color in zip(parsed, args.labels, colors):
        ax.plot(d["steps"], d["scores"], "o--", color=color, alpha=0.35,
                markersize=4, linewidth=1)
        sw = 3
        if len(d["scores"]) >= sw:
            xs = d["steps"][sw - 1:]
            ax.plot(xs, smooth(d["scores"], sw), "-", color=color,
                    linewidth=2, label=f"{label} (smooth-{sw})")
        else:
            ax.plot(d["steps"], d["scores"], "-", color=color, linewidth=2, label=label)
    ax.set_xlabel("Training Step"); ax.set_ylabel("critic/score/mean")
    ax.set_title("(b) Score (mean)", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.05, 1.05)

    # ── panel 3: step time ───────────────────────────────────────────────────
    ax = axes[2]
    for d, label, color in zip(parsed, args.labels, colors):
        steps_skip1 = d["steps"][1:]    # skip step 1 (model load warmup)
        times_skip1 = d["step_time"][1:]
        ax.plot(steps_skip1, times_skip1, "o-", color=color,
                markersize=4, linewidth=1.5, label=label)
        if times_skip1:
            avg = np.mean(times_skip1)
            ax.axhline(avg, color=color, linestyle="--", linewidth=1, alpha=0.6)
            ax.text(steps_skip1[-1] + 0.2, avg, f" {avg:.1f}s", color=color, fontsize=8, va="center")
    ax.set_xlabel("Training Step"); ax.set_ylabel("timing_s/step (s)")
    ax.set_title("(c) Iteration Time (step 2+)", fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

    # ── print summary table ──────────────────────────────────────────────────
    print("\n## Reward Stability Summary")
    print(f"| Config | avg reward | avg score | avg step_time (s) | nonzero steps |")
    print(f"|---|---:|---:|---:|---:|")
    for d, label in zip(parsed, args.labels):
        r = d["rewards"][1:] if len(d["rewards"]) > 1 else d["rewards"]
        s = d["scores"][1:]  if len(d["scores"])  > 1 else d["scores"]
        t = d["step_time"][1:] if len(d["step_time"]) > 1 else d["step_time"]
        nz = sum(1 for x in r if x > 0)
        print(f"| {label} | {np.mean(r):.4f} | {np.mean(s):.4f} | {np.mean(t):.2f} | {nz}/{len(r)} |")


if __name__ == "__main__":
    main()
