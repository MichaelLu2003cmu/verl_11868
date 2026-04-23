"""Regenerate 4_3_compress.png.

Three-panel compression microbench figure at batch=8 / 20 steps, 3 configs
(baseline-push / FP16-pull / INT8-pull).  All values sourced from the raw
probe JSONLs checked into `checkpoints/verl_examples/*/transfer_probe*.jsonl`.

BF16 is intentionally NOT shown here because no batch=8 / 20-step BF16 run
exists and the batch=32 / 100-step BF16 run was launched with the
transfer-probe disabled.  BF16's end-to-end stability argument lives in
7_stability_corrected.png and in 7_eval.md Tables 3c'/3d' (batch=32 /
100 steps) instead.

Sources (exact values; re-verified on 2026-04-21):
  Baseline dispatch = 25.318 ms  <- gsm8k_2gpu_push_baseline/transfer_probe.jsonl
  FP16 dispatch     =  8.563 ms  <- gsm8k_2gpu_compress_fp16/transfer_probe_fp16.jsonl
  INT8 dispatch     = 14.507 ms  <- gsm8k_2gpu_compress_int8/transfer_probe_int8.jsonl

  Baseline recv_bytes = 14336 B  -> 14.3 KB (decimal) / 14.0 KiB
  FP16/INT8 recv_bytes =  8192 B ->  8.2 KB (decimal) /  8.0 KiB

  compress_stats (pull_build_handle):
    FP16: orig=110600 B, comp=108552 B  -> saved = 2048 B = 2 KiB   (1.85%)
    INT8: orig=110600 B, comp=107528 B  -> saved = 3072 B = 3 KiB   (2.78%)
    Implied float32 payload = 4 KiB (FP16 halves, INT8 quarters it);
    int64 payload = 110600 - 4096 = 106504 B ~= 104 KiB.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

configs  = ["Baseline\n(no compress,\npush)", "FP16\n(pull)", "INT8\n(pull)"]
colors   = ["#4878CF", "#6ACC65", "#CC4444"]

dispatch = [25.318, 8.563, 14.507]   # dispatch_ms from probes
recv_kb  = [14.3,  8.2,  8.2]        # recv_bytes / 1000 (decimal KB, matches MD tables)

f32_orig = 4.000
f32_kb   = [4.000, 2.000, 1.000]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle(
    "Section 4.3 — Lightweight Compression for Selected Tensors\n"
    "(Qwen2.5-1.5B-Instruct, 2xV100, batch=8, pull-dispatch, 20 steps)",
    fontsize=11, fontweight="bold",
)
bar_kw = dict(width=0.55, edgecolor="white", linewidth=0.8)

# (a) Dispatch overhead
ax = axes[0]
b = ax.bar(range(3), dispatch, color=colors, **bar_kw)
ax.set_title("Dispatch Overhead", fontweight="bold")
ax.set_ylabel("dispatch_ms / iter")
ax.set_xticks(range(3)); ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0, max(dispatch) * 1.3)
for bb, v in zip(b, dispatch):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.3, f"{v:.1f}",
            ha="center", va="bottom", fontweight="bold", fontsize=9.5)
ax.plot(range(3), dispatch, "o--", color="gray", ms=5, lw=1)
ax.axhline(dispatch[1], color=colors[1], linestyle=":", lw=1.2, alpha=0.7)
pct = (dispatch[0] - dispatch[1]) / dispatch[0] * 100
ax.text(1.55, dispatch[1] + 0.4, f"-{pct:.0f}%", color=colors[1], fontsize=8.5)

# (b) Received payload size
ax = axes[1]
b = ax.bar(range(3), recv_kb, color=colors, **bar_kw)
ax.set_title("Received Payload Size", fontweight="bold")
ax.set_ylabel("recv KB / iter")
ax.set_xticks(range(3)); ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0, 20)
for bb, v in zip(b, recv_kb):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.2, f"{v:.1f} KB",
            ha="center", va="bottom", fontweight="bold", fontsize=9.5)

# (c) Float32 compressible portion
ax = axes[2]
b = ax.bar(range(3), f32_kb, color=colors, **bar_kw,
           label="float32 payload (after compress)")
ax.axhline(f32_orig, color="gray", linestyle="--", lw=1.4, alpha=0.7,
           label=f"original float32 ({f32_orig:.3f} KB)")
saves_pct = [(f32_orig - v) / f32_orig * 100 for v in f32_kb]
for i, (v, s) in enumerate(zip(f32_kb, saves_pct)):
    ax.text(i, v + 0.07, f"{v:.3f} KB", ha="center", va="bottom",
            fontweight="bold", fontsize=9.5)
    if s > 0:
        ax.text(i, v - 0.40, f"-{s:.0f}% vs orig", ha="center", va="bottom",
                fontsize=7.5, color=colors[i],
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=colors[i], lw=0.7))
ax.set_title("Compressible Float32 Payload\n(int64 ~= 104 KB, unchanged)",
             fontweight="bold", fontsize=9)
ax.set_ylabel("Float32 portion KB / iter")
ax.set_xticks(range(3)); ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0, 5.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.text(1, -1.15,
        "int64 (token ids, masks) ~= 104 KB — incompressible across all runs",
        ha="center", va="top", fontsize=7, color="gray",
        transform=ax.transData, clip_on=False)

plt.tight_layout()
plt.savefig("4_3_compress.png", dpi=150, bbox_inches="tight")
print("Saved 4_3_compress.png")
