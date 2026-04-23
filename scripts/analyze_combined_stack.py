#!/usr/bin/env python3
"""Analyze Rerun C' (combined LP + AO + Comp stack).

Inputs (per config):
  checkpoints/verl_examples/gsm8k_2gpu_stack_{push_ref,lp_ao,lp_ao_fp16}/
    train_log.txt         # per-step perf/time_per_step, actor/ppo_kl, etc.
    transfer_probe.jsonl  # transfer_latency + critical_path_stage events

Probe event schemas used:
  critical_path_stage: dispatched_at_ms, joined_at_ms, wait_ms, stage, step
    wait_ms = driver-blocking time on future.get() for that stage.
    Push dispatch: driver blocks fully on each stage -> Sigma wait ~ Sigma gpu_time
    Pull + AO:     later stages overlap, driver only waits on tail -> Sigma wait lower.
  transfer_latency:    dispatch_ms, collect_ms, send_bytes, recv_bytes, step
  compress_stats:      compression_ratio, dtype

Outputs:
  stdout: markdown tables (end-to-end metrics, prep-stage wait across configs,
          AO benefit vs push baseline, compression summary).
  poster_stack.png: 4-panel bar figure for the poster.

Usage:
  /ocean/projects/cis260009p/syan5/conda/project/bin/python \
    scripts/analyze_combined_stack.py [--out-fig poster_stack.png] [--out-md stack.md]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config set
# ---------------------------------------------------------------------------

# The composability table draws from five sources:
#   (0) NEW stack_push_ref: push mode, ref ON — proper baseline for AO measurement.
#       Driver blocks inside _compute_ref_log_prob() for ref (probe sees wait_ms≈0
#       for ref stage because the computation completes before _probe_mark_dispatched;
#       use timing_s/old_log_prob + timing_s/ref from train_log for ground-truth total).
#   (1) Existing push baseline (no AO, no Comp, ref OFF — from the
#       main ablation; gives the iter-time upper bound at this batch
#       size for the configs that cannot AO-overlap).
#   (2) Existing pathB_lite (LP + AO + ref + critic; gives the
#       published AO reference numbers — wait_ms / hidden_frac).
#   (3) NEW stack_lp_ao run (LP + AO + ref, matched to (4)).
#   (4) NEW stack_lp_ao_fp16 run (full LP + AO + FP16 + ref stack).
# Row (0) is the apples-to-apples baseline for (3) and (4).
# Rows (1) and (2) provide historical context.
CONFIGS = [
    ("push+ref baseline (ref ON, push)",       "gsm8k_2gpu_stack_push_ref",   "new"),
    ("+LP+AO (pull + async, ref ON)",          "gsm8k_2gpu_stack_lp_ao",      "new"),
    ("+LP+AO+FP16 (full stack, ref ON)",       "gsm8k_2gpu_stack_lp_ao_fp16", "new"),
    ("Existing push baseline (ref OFF)",       "gsm8k_2gpu_push_baseline",    "reference"),
    ("Existing pathB_lite (LP+AO, ref+critic)","gsm8k_2gpu_pathB_lite",       "reference"),
]

SKIP_FIRST_STEPS = 1  # step 1 includes vLLM warmup; drop from all stats.

# ---------------------------------------------------------------------------
# Train-log parsing
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"\bstep:(\d+)\b")
TIME_RE = re.compile(r"perf/time_per_step:([\d.\-eE+]+)")
KL_RE = re.compile(r"actor/ppo_kl:(?:np\.float64\()?([\-\d.eE+]+)")
THROUGHPUT_RE = re.compile(r"perf/throughput:([\d.\-eE+]+)")
REWARD_RE = re.compile(r"critic/rewards/mean:([\d.\-eE+]+)")


def parse_train_log(path: Path, skip_first: int = SKIP_FIRST_STEPS) -> dict:
    out = {"steps": [], "time": [], "kl": [], "throughput": [], "reward": []}
    if not path.exists():
        return out
    for raw in path.read_text(errors="ignore").splitlines():
        if "perf/time_per_step" not in raw:
            continue
        sm, tm = STEP_RE.search(raw), TIME_RE.search(raw)
        if not (sm and tm):
            continue
        step = int(sm.group(1))
        if step < 1 + skip_first:
            continue
        out["steps"].append(step)
        out["time"].append(float(tm.group(1)))
        m = KL_RE.search(raw)
        out["kl"].append(float(m.group(1)) if m else float("nan"))
        m = THROUGHPUT_RE.search(raw)
        out["throughput"].append(float(m.group(1)) if m else float("nan"))
        m = REWARD_RE.search(raw)
        out["reward"].append(float(m.group(1)) if m else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Probe parsing
# ---------------------------------------------------------------------------


def iter_probe_events(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        raw = raw.strip()
        if "[transfer_probe]" not in raw:
            continue
        try:
            payload = raw.split("[transfer_probe]", 1)[1].strip()
            yield json.loads(payload)
        except Exception:
            continue


def summarize_probe(path: Path, skip_first: int = SKIP_FIRST_STEPS) -> dict:
    # per-step series (for jitter / means)
    xfer_ms_per_step = defaultdict(float)
    recv_bytes_per_step = defaultdict(float)
    send_bytes_per_step = defaultdict(float)
    # per-stage wait_ms series (skipping warmup steps)
    stage_wait = defaultdict(list)          # stage -> [wait_ms per step]
    stage_in_flight = defaultdict(list)     # stage -> [joined_at - dispatched_at per step]
    # per-step true total blocking time = max(joined_at_ms) over all stages.
    # This correctly captures push mode, where ref blocks INSIDE _compute_ref_log_prob()
    # before the probe marks dispatch: joined_at_ms of the ref stage is therefore
    # recorded AFTER the ref computation finishes, giving the true end-to-end time.
    step_max_joined_ms: dict[int, float] = {}

    compress_ratio_sum, compress_ratio_n = 0.0, 0

    for e in iter_probe_events(path):
        ev = e.get("event")
        step = int(e.get("step", 0) or 0)
        if ev == "transfer_latency":
            # transfer_latency uses seq, not step -- bucket by seq to avoid collision.
            # We just use it for per-step averages which does not require step alignment.
            key = int(e.get("step", e.get("seq", 0)))
            xfer_ms_per_step[key] += float(e.get("dispatch_ms", 0.0)) + float(
                e.get("collect_ms", 0.0)
            )
            recv_bytes_per_step[key] += float(e.get("recv_bytes", 0.0))
            send_bytes_per_step[key] += float(e.get("send_bytes", 0.0))
        elif ev == "critical_path_stage":
            if step < 1 + skip_first:
                continue
            stage = str(e.get("stage", "?"))
            wait_ms = float(e.get("wait_ms", 0.0))
            joined_at = float(e.get("joined_at_ms", 0.0))
            in_flight = max(
                0.0,
                joined_at - float(e.get("dispatched_at_ms", 0.0)),
            )
            stage_wait[stage].append(wait_ms)
            stage_in_flight[stage].append(in_flight)
            # Update per-step total blocking: the last stage to finish determines
            # when the driver can proceed (max joined_at_ms across stages).
            step_max_joined_ms[step] = max(step_max_joined_ms.get(step, 0.0), joined_at)
        elif ev == "compress_stats":
            ratio = float(e.get("ratio", e.get("compression_ratio", 0.0)))
            if ratio > 0:
                compress_ratio_sum += ratio
                compress_ratio_n += 1

    # Aggregate per-stage stats
    stages = {}
    for stage in sorted(stage_wait):
        waits = stage_wait[stage]
        stages[stage] = {
            "count": len(waits),
            "wait_ms_mean": stats.mean(waits),
            "wait_ms_p50": sorted(waits)[len(waits) // 2] if waits else 0.0,
            "in_flight_ms_mean": stats.mean(stage_in_flight[stage])
            if stage_in_flight[stage]
            else 0.0,
        }

    # True total blocking series (max joined_at_ms per step)
    total_blocking_series = sorted(step_max_joined_ms.values())
    xfer_ms = list(xfer_ms_per_step.values())
    recv_bytes = list(recv_bytes_per_step.values())
    send_bytes = list(send_bytes_per_step.values())

    return {
        "xfer_ms_per_step_mean": stats.mean(xfer_ms) if xfer_ms else 0.0,
        "recv_bytes_per_step_mean": stats.mean(recv_bytes) if recv_bytes else 0.0,
        "send_bytes_per_step_mean": stats.mean(send_bytes) if send_bytes else 0.0,
        "stages": stages,
        "prep_total_blocking_mean": stats.mean(total_blocking_series)
        if total_blocking_series else 0.0,
        "prep_total_blocking_p50": sorted(total_blocking_series)[len(total_blocking_series) // 2]
        if total_blocking_series else 0.0,
        "n_steps_with_prep": len(total_blocking_series),
        "compress_ratio_mean": (compress_ratio_sum / compress_ratio_n)
        if compress_ratio_n
        else 0.0,
        "compress_events": compress_ratio_n,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentile(xs, p):
    xs = sorted(v for v in xs if v == v)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean_ok(xs):
    xs = [v for v in xs if v == v]
    return stats.mean(xs) if xs else float("nan")


def _resolve_probe_path(run_dir: Path) -> Path:
    """Return the probe JSONL path for a given run dir.
    Some legacy runs use a custom probe filename (e.g. transfer_probe_pathB_lite.jsonl);
    this tries the canonical name first then falls back to any *.jsonl in the dir."""
    canonical = run_dir / "transfer_probe.jsonl"
    if canonical.exists():
        return canonical
    for candidate in sorted(run_dir.glob("transfer_probe*.jsonl")):
        return candidate
    return canonical  # returns non-existent; summarize_probe handles that gracefully


def collect(base_dir: Path) -> list[dict]:
    rows = []
    for label, tag, kind in CONFIGS:
        run_dir = base_dir / tag
        tl = parse_train_log(run_dir / "train_log.txt")
        pr = summarize_probe(_resolve_probe_path(run_dir))
        times = tl["time"]
        rows.append(
            {
                "label": label,
                "tag": tag,
                "kind": kind,
                "run_dir": str(run_dir),
                "n_steps": len(times),
                "iter_time_mean": mean_ok(times),
                "iter_time_p50": percentile(times, 50),
                "iter_time_p99": percentile(times, 99),
                "iter_time_std": stats.pstdev([v for v in times if v == v])
                if len(times) > 1
                else 0.0,
                "kl_mean_abs": mean_ok([abs(v) for v in tl["kl"]]),
                "throughput_mean": mean_ok(tl["throughput"]),
                "reward_mean": mean_ok(tl["reward"]),
                **pr,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def markdown_report(rows: list[dict]) -> str:
    lines = []
    lines.append(
        "## Rerun C' - Combined LP + AO + Comp stack (batch=8, 20 steps, ref worker ENABLED)"
    )
    lines.append("")
    lines.append(
        "Probe events skip step 1 (vLLM warmup).  Iter-time stats also skip step 1.\n"
        "Mean is over steps 2..N.  `kl_mean_abs` averages `|actor/ppo_kl|` per step."
    )
    lines.append("")

    # ---- End-to-end table -------------------------------------------------
    lines.append("### End-to-end metrics")
    lines.append(
        "| Config | n | iter mean (s) | iter p50 | iter p99 | std | |KL|_mean | throughput |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n_steps']} | "
            f"{r['iter_time_mean']:.3f} | {r['iter_time_p50']:.3f} | "
            f"{r['iter_time_p99']:.3f} | {r['iter_time_std']:.3f} | "
            f"{r['kl_mean_abs']:.5f} | {r['throughput_mean']:.1f} |"
        )

    # ---- Prep-stage wait table -------------------------------------------
    lines.append("")
    lines.append("### Prep-stage driver-blocking time (headline LP+AO metric)")
    lines.append("")
    lines.append(
        "*`total blocking` = `max(joined_at_ms)` across critical_path_stage events per step.  "
        "This is the ground-truth end-to-end driver blocking time for the prep stage "
        "in both push and pull modes: in push mode the ref computation completes inside "
        "`_compute_ref_log_prob()` before the probe marks dispatch (so `wait_ms`≈0 for ref), "
        "but `joined_at_ms` of the ref stage is still recorded after the computation "
        "finishes, giving the correct total.*"
    )
    lines.append("")
    lines.append(
        "| Config | total blocking (ms, mean) | total blocking (ms, p50) | n_steps |"
    )
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['prep_total_blocking_mean']:.1f} | "
            f"{r['prep_total_blocking_p50']:.1f} | {r['n_steps_with_prep']} |"
        )

    # ---- Per-stage wait breakdown ---------------------------------------
    lines.append("")
    lines.append("### Per-stage breakdown (mean across steps)")
    lines.append(
        "| Config | stage | count | wait_ms mean | wait_ms p50 | in_flight_ms mean |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if not r["stages"]:
            lines.append(
                f"| {r['label']} | *(no critical_path_stage events -- AO not applicable)* | | | | |"
            )
            continue
        for stage, s in r["stages"].items():
            lines.append(
                f"| {r['label']} | `{stage}` | {s['count']} | "
                f"{s['wait_ms_mean']:.1f} | {s['wait_ms_p50']:.1f} | "
                f"{s['in_flight_ms_mean']:.1f} |"
            )

    # ---- AO/LP/Comp benefit vs baseline -----------------------------------
    lines.append("")
    lines.append(
        "### Composability: each optimization's contribution vs. push+ref baseline"
    )
    lines.append("")
    # Base = push_ref (first row, ref ON, no LP/AO/Comp) — apples-to-apples.
    base = rows[0]
    lines.append(
        "| Config | iter_mean Delta% | iter_p50 Delta% | total blocking Delta%"
        " | |KL|_mean Delta% | recv_bytes Delta% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    def pct(cur, base_val):
        if base_val == 0 or base_val != base_val:
            return float("nan")
        return (cur - base_val) / base_val * 100.0

    for r in rows:
        lines.append(
            f"| {r['label']} | "
            f"{pct(r['iter_time_mean'], base['iter_time_mean']):+.1f}% | "
            f"{pct(r['iter_time_p50'], base['iter_time_p50']):+.1f}% | "
            f"{pct(r['prep_total_blocking_mean'], base['prep_total_blocking_mean']):+.1f}% | "
            f"{pct(r['kl_mean_abs'], base['kl_mean_abs']):+.1f}% | "
            f"{pct(r['recv_bytes_per_step_mean'], base['recv_bytes_per_step_mean']):+.1f}% |"
        )

    # ---- Compression summary ---------------------------------------------
    lines.append("")
    lines.append("### Transfer-layer probe (bytes + compression)")
    lines.append(
        "| Config | Sigma recv/iter (KB) | Sigma send/iter (KB) | Sigma xfer_ms/iter | compress events | mean compress ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['recv_bytes_per_step_mean'] / 1024:.1f} | "
            f"{r['send_bytes_per_step_mean'] / 1024:.1f} | "
            f"{r['xfer_ms_per_step_mean']:.2f} | "
            f"{r['compress_events']} | {r['compress_ratio_mean']:.3f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(rows: list[dict], out_path: Path) -> None:
    # Only plot the three new matched runs (push_ref, lp_ao, lp_ao_fp16).
    plot_rows = [r for r in rows if r["kind"] == "new"]
    labels = [r["label"] for r in plot_rows]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    x = list(range(len(plot_rows)))

    # Panel (a): End-to-end iter time, mean with p99 whisker.
    means = [r["iter_time_mean"] for r in plot_rows]
    p99s = [r["iter_time_p99"] for r in plot_rows]
    axes[0].bar(x, means, color=colors)
    for xi, p99, m in zip(x, p99s, means):
        if p99 == p99:
            axes[0].plot([xi, xi], [m, p99], color="black", linewidth=1.5)
            axes[0].plot([xi - 0.15, xi + 0.15], [p99, p99], color="black", linewidth=1.5)
    for xi, m in zip(x, means):
        if m == m:
            axes[0].text(xi, m, f"{m:.2f}s", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("End-to-end iter time (s)")
    axes[0].set_title("(a) Iter time (mean, whisker=p99)")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel (b): max(joined_at_ms) per step -- ground-truth driver-blocking prep time.
    waits = [r["prep_total_blocking_mean"] for r in plot_rows]
    axes[1].bar(x, waits, color=colors)
    for xi, v in zip(x, waits):
        if v == v:
            axes[1].text(xi, v, f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Total prep blocking (ms / iter)")
    axes[1].set_title("(b) LP+AO: driver prep-blocking time\n(max joined_at_ms per step)")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel (c): Per-iter recv bytes -- LP slicing + Comp shrinking.
    recv_kb = [r["recv_bytes_per_step_mean"] / 1024 for r in plot_rows]
    axes[2].bar(x, recv_kb, color=colors)
    for xi, v in zip(x, recv_kb):
        axes[2].text(xi, v, f"{v:.1f} KB", ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[2].set_ylabel("Sigma recv / iter (KB, per worker)")
    axes[2].set_title("(c) Payload: LP slices, Comp shrinks")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Rerun C': Local-Batch Pull + Async Overlap + FP16 -- composability on 2xV100 (batch=8, 20 steps, ref ON)",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir",
        default="checkpoints/verl_examples",
        help="directory containing gsm8k_2gpu_stack_* run dirs",
    )
    p.add_argument("--out-fig", default="poster_stack.png", help="output figure filename")
    p.add_argument("--out-md", default=None, help="write the markdown report to this file")
    args = p.parse_args()

    base = Path(args.base_dir).resolve()
    rows = collect(base)

    report = markdown_report(rows)
    print(report)

    if args.out_md:
        Path(args.out_md).write_text(report + "\n", encoding="utf-8")
        print(f"\nsaved markdown: {args.out_md}")

    make_figure(rows, Path(args.out_fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
