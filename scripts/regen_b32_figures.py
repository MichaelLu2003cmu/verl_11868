"""Generate the two batch=32 / 100-step companion figures:
    7_eval_b32.png        — 4-panel end-to-end summary bar chart
    4_3_compress_b32.png  — 3-panel compression trade-off + Pareto

Both figures use the SAME 4 configurations at batch=32 / 100 steps, FSDP OFF,
step 1 excluded:
    Baseline (push, no compress)  -> gsm8k_2gpu_baseline_100
    +LP (pull, no compress)       -> gsm8k_2gpu_lp_only_100        (Rerun A)
    +LP+FP16  (pull, FP16)        -> gsm8k_2gpu_lp_fp16_100
    +LP+BF16  (pull, BF16)        -> gsm8k_2gpu_lp_bf16_100        (Rerun B)

Neither figure shows transfer-layer microbench (recv bytes, dispatch_ms)
because the BF16 run was launched without VERL_TRANSFER_PROBE=1.  The
batch=8 microbench story lives in 7_eval.png and 4_3_compress.png.

All values come directly from scripts/jitter_and_kl.py output over the four
train_log.txt files (4-config run on 2026-04-22).  They match 7_eval.md
Tables 3c'/3d'/3e' exactly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -------- shared inputs --------
CONFIGS = [
    "Baseline\n(push)",
    "+LP\n(pull)",
    "+LP\n+FP16",
    "+LP\n+BF16",
]
COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#CC8800"]

# Values re-extracted from jitter_and_kl.py over the four train_log.txt files
# (re-run on 2026-04-22 after the on_policy stability fix in
# verl/workers/actor/dp_actor.py).  Match 7_eval.md Tables 3c'/3d'/3e'.
#
# Post-fix key findings:
#   - LP alone now learns (reward 0.065 vs pre-fix 0.014) AND is faster
#     than baseline (17.9s vs 21.9s).  Fix eliminates the step-12
#     gradient explosion and subsequent permanent grad_norm=0 deadlock.
#   - LP+FP16 / LP+BF16 still destabilise (reward 0.018 / 0.016).
#     Root cause: fp16/bf16 round-trip quantisation of old_log_probs
#     inflates PPO ratio noise past the clip boundary, causing format
#     collapse after ~25 steps.  Documented as a composability ceiling
#     in the Limitations section.
MEAN_S       = [21.952, 17.933, 22.375, 21.483]
P50_S        = [21.888, 17.658, 22.401, 22.400]
P99_S        = [23.084, 22.192, 23.199, 22.922]
STD_S        = [ 0.657,  0.955,  0.444,  1.825]
KL_MEAN      = [0.01700, 0.02124, 0.00473, 0.02030]
KL_STD       = [0.02329, 0.03456, 0.01334, 0.06712]
REWARD_MEAN  = [0.2251,  0.0650,  0.0183,  0.0158]
THROUGHPUT   = [136.5,   101.2,   160.9,   147.7]

# Deltas vs push baseline (index 0) for annotation convenience.
def pct_delta(xs: list[float]) -> list[float]:
    base = xs[0]
    return [(v - base) / base * 100.0 for v in xs]

BAR_KW = dict(width=0.6, edgecolor="white", linewidth=0.8)


# ---------------------------------------------------------------------------
# Figure 1:  7_eval_b32.png  —  4-panel end-to-end summary
# ---------------------------------------------------------------------------
def make_fig1_eval_b32() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "End-to-End Summary at batch=32 — Qwen2.5-1.5B-Instruct, 2xV100, "
        "100 steps, FSDP OFF, step 1 excluded",
        fontsize=11, fontweight="bold",
    )

    # (a) Mean iter time
    ax = axes[0, 0]
    ax.bar(range(4), MEAN_S, color=COLORS, **BAR_KW)
    ax.axhline(MEAN_S[0], color=COLORS[0], linestyle=":", lw=1.0, alpha=0.6)
    for i, v in enumerate(MEAN_S):
        lbl = f"{v:.2f}s" if i == 0 else f"{v:.2f}s\n({pct_delta(MEAN_S)[i]:+.1f}%)"
        ax.text(i, v + 0.2, lbl, ha="center", va="bottom", fontsize=9)
    ax.set_title("(a) Mean Iter Time (s/step)", fontweight="bold", fontsize=10)
    ax.set_ylabel("seconds / step")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(0, max(MEAN_S) * 1.15)

    # (b) p99 iter time (tail latency)
    ax = axes[0, 1]
    ax.bar(range(4), P99_S, color=COLORS, **BAR_KW)
    ax.axhline(P99_S[0], color=COLORS[0], linestyle=":", lw=1.0, alpha=0.6)
    for i, v in enumerate(P99_S):
        lbl = f"{v:.2f}s" if i == 0 else f"{v:.2f}s\n({pct_delta(P99_S)[i]:+.1f}%)"
        ax.text(i, v + 0.2, lbl, ha="center", va="bottom", fontsize=9)
    ax.set_title("(b) p99 Iter Time (tail latency, s/step)",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("seconds / step")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(0, max(P99_S) * 1.15)

    # (c) Mean |KL| — stability
    ax = axes[1, 0]
    ax.bar(range(4), KL_MEAN, color=COLORS, **BAR_KW)
    ax.axhline(KL_MEAN[0], color=COLORS[0], linestyle=":", lw=1.0, alpha=0.6,
               label=f"baseline |KL| = {KL_MEAN[0]:.4f}")
    for i, v in enumerate(KL_MEAN):
        lbl = f"{v:.4f}" if i == 0 else f"{v:.4f}\n({pct_delta(KL_MEAN)[i]:+.0f}%)"
        ax.text(i, v + 0.0012, lbl, ha="center", va="bottom", fontsize=9)
    ax.set_title("(c) Mean |actor/ppo_kl| per step (on-policy drift)",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("mean |KL|")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(0, max(KL_MEAN) * 1.35)
    ax.legend(fontsize=7.5, loc="upper left")

    # (d) Throughput
    ax = axes[1, 1]
    ax.bar(range(4), THROUGHPUT, color=COLORS, **BAR_KW)
    ax.axhline(THROUGHPUT[0], color=COLORS[0], linestyle=":", lw=1.0, alpha=0.6)
    for i, v in enumerate(THROUGHPUT):
        lbl = f"{v:.1f}" if i == 0 else f"{v:.1f}\n({pct_delta(THROUGHPUT)[i]:+.1f}%)"
        ax.text(i, v + 2, lbl, ha="center", va="bottom", fontsize=9)
    ax.set_title("(d) Throughput (tokens/s, averaged per step)",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("tokens / second")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(0, max(THROUGHPUT) * 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("7_eval_b32.png", dpi=150, bbox_inches="tight")
    print("Saved 7_eval_b32.png")


# ---------------------------------------------------------------------------
# Figure 2:  4_3_compress_b32.png  —  3-panel compression trade-off + Pareto
# ---------------------------------------------------------------------------
def make_fig2_compress_b32() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Compression Trade-off at batch=32 — Qwen2.5-1.5B-Instruct, 2xV100, "
        "100 steps, FSDP OFF, step 1 excluded",
        fontsize=11, fontweight="bold",
    )

    # (a) %Delta mean iter time vs baseline
    ax = axes[0]
    deltas_time = pct_delta(MEAN_S)
    b = ax.bar(range(4), deltas_time, color=COLORS, **BAR_KW)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(deltas_time):
        offset = 1.0 if v >= 0 else -1.0
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + offset, f"{v:+.1f}%", ha="center", va=va,
                fontweight="bold", fontsize=9.5)
    ax.set_title("(a) %Δ Mean Iter Time vs Baseline\n(lower = faster)",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("%Δ vs baseline (push, no compress)")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(min(deltas_time) - 6, max(deltas_time) + 6)

    # (b) %Delta mean |KL| vs baseline
    ax = axes[1]
    deltas_kl = pct_delta(KL_MEAN)
    b = ax.bar(range(4), deltas_kl, color=COLORS, **BAR_KW)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(deltas_kl):
        offset = 8 if v >= 0 else -8
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + offset, f"{v:+.0f}%", ha="center", va=va,
                fontweight="bold", fontsize=9.5)
    ax.set_title("(b) %Δ Mean |KL| vs Baseline\n(lower = more on-policy)",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("%Δ vs baseline (push, no compress)")
    ax.set_xticks(range(4)); ax.set_xticklabels(CONFIGS, fontsize=9)
    ax.set_ylim(min(deltas_kl) - 20, max(deltas_kl) + 30)

    # (c) Pareto scatter:  iter time (lower=better) vs |KL| (lower=better)
    # A config is Pareto-dominated if another config beats it on BOTH axes.
    ax = axes[2]
    for i, label in enumerate(CONFIGS):
        ax.scatter(MEAN_S[i], KL_MEAN[i], color=COLORS[i], s=260,
                   edgecolor="black", linewidth=1.0, zorder=3, label=label.replace("\n", " "))
    # Annotate each point
    for i in range(4):
        dy = 0.0032 if i in (0, 2) else -0.0045
        dx = 0.25 if i != 2 else -0.25
        ha = "left" if dx > 0 else "right"
        ax.annotate(CONFIGS[i].replace("\n", " "),
                    (MEAN_S[i], KL_MEAN[i]),
                    xytext=(MEAN_S[i] + dx, KL_MEAN[i] + dy),
                    fontsize=8.5, ha=ha, color=COLORS[i], fontweight="bold")
    # Compute and draw Pareto frontier.
    # "Better" = (lower iter, lower |KL|).  A config is Pareto-optimal if no
    # other config beats it on both axes.
    pts = list(zip(MEAN_S, KL_MEAN, CONFIGS, COLORS))
    pareto_idx = []
    for i, (t, k, _, _) in enumerate(pts):
        dominated = False
        for j, (tj, kj, _, _) in enumerate(pts):
            if j == i:
                continue
            if tj <= t and kj <= k and (tj < t or kj < k):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    pareto_idx.sort(key=lambda i: pts[i][0])  # sort by iter time
    if len(pareto_idx) >= 2:
        xs = [pts[i][0] for i in pareto_idx]
        ys = [pts[i][1] for i in pareto_idx]
        ax.plot(xs, ys, "--", color="gray", lw=1.2, alpha=0.6, zorder=2,
                label="Pareto frontier")
    ax.set_title("(c) Throughput/Stability Pareto\n(bottom-left is better)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("mean iter time (s/step) — lower = faster")
    ax.set_ylabel("mean |KL| per step — lower = on-policy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    # Breathe a little so the text labels don't clip.
    ax.set_xlim(min(MEAN_S) - 2.0, max(MEAN_S) + 2.0)
    ax.set_ylim(-0.003, max(KL_MEAN) * 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("4_3_compress_b32.png", dpi=150, bbox_inches="tight")
    print("Saved 4_3_compress_b32.png")


if __name__ == "__main__":
    make_fig1_eval_b32()
    make_fig2_compress_b32()
