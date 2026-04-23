"""Regenerate 7_eval.png.

All four panels share the same 4-config set at batch=8 / 20 steps,
FSDP OFF, step 1 excluded:
  push_baseline    <- gsm8k_2gpu_push_baseline      (push dispatch, no compress)
  pathA            <- gsm8k_2gpu_nocomp             (pull dispatch, no compress)
  fp16             <- gsm8k_2gpu_compress_fp16      (pull dispatch, fp16)
  int8             <- gsm8k_2gpu_compress_int8      (pull dispatch, int8)

This figure is the transfer-layer microbench story.  The BF16 compression
mode is intentionally absent here because no batch=8 / 20-step BF16 run
exists and the batch=32 / 100-step BF16 run was launched with the
transfer-probe disabled (see what_we_have_done.md §12).  BF16's end-to-end
stability advantage is covered in 7_stability_corrected.png and in
7_eval.md Tables 3c'/3d' at batch=32 / 100 steps, where all four
configurations (push / +LP / +LP+FP16 / +LP+BF16) appear.

----------------------------------------------------------------------
Panel sources (values re-extracted from raw logs)
----------------------------------------------------------------------

(a) Received Transfer Volume (KB/iter, from transfer_probe recv_bytes, per worker):
    push_baseline  14336 B -> 14.3 KB
    pathA (pull)   14336 B -> 14.3 KB
    compress_fp16   8192 B ->  8.2 KB
    compress_int8   8192 B ->  8.2 KB

(b) Dispatch ms/iter (from transfer_probe compute_log_prob):
    push_baseline  25.318
    pathA          18.706
    fp16            8.563
    int8           14.507

(c) DataProto.concat ms/iter (from transfer_probe cpu_overhead):
    push_baseline  0.252
    pathA          0.245
    fp16           0.118
    int8           0.124

(d) End-to-end iter time (s/step, step 1 excluded).  Numbers re-extracted
    on 2026-04-22 from each run's train_log.txt:
    push_baseline  8.939 s   (n=19 steps after warmup)
    pathA (pull)   9.121 s
    compress_fp16  8.546 s
    compress_int8  8.865 s
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
fig.suptitle(
    "Evaluation Summary — 2-GPU, Qwen2.5-1.5B-Instruct, batch=8, 20 steps, FSDP OFF, step 1 excluded\n"
    "Panels (a)-(c): transfer_probe microbench · Panel (d): end-to-end iter time",
    fontsize=11, fontweight="bold",
)

bar_kw = dict(width=0.6, edgecolor="white", linewidth=0.8)

# ---------- microbench panels (a,b,c): 4 configs ----------
mb_configs = ["Baseline\n(push)", "+LP\n(pull)", "+LP\n+FP16", "+LP\n+INT8"]
mb_colors  = ["#4878CF", "#6ACC65", "#D65F5F", "#956CB4"]

# (a) recv KB
ax = axes[0, 0]
recv = [14.3, 14.3, 8.2, 8.2]
b = ax.bar(range(4), recv, color=mb_colors, **bar_kw)
ax.set_title("(a) Received Transfer Volume (KB/iter)", fontweight="bold", fontsize=10)
ax.set_ylabel("KB / iter")
ax.set_xticks(range(4)); ax.set_xticklabels(mb_configs, fontsize=8.5)
ax.set_ylim(0, 20)
for bb, v in zip(b, recv):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.3, f"{v:.1f}",
            ha="center", va="bottom", fontsize=9)

# (b) dispatch ms
ax = axes[0, 1]
disp = [25.318, 18.706, 8.563, 14.507]
b = ax.bar(range(4), disp, color=mb_colors, **bar_kw)
ax.set_title("(b) Host Transfer Overhead (dispatch ms/iter)", fontweight="bold", fontsize=10)
ax.set_ylabel("ms / iter")
ax.set_xticks(range(4)); ax.set_xticklabels(mb_configs, fontsize=8.5)
ax.set_ylim(0, max(disp)*1.2)
for bb, v in zip(b, disp):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.4, f"{v:.1f}",
            ha="center", va="bottom", fontsize=9)

# (c) concat ms
ax = axes[1, 0]
cpu = [0.252, 0.245, 0.118, 0.124]
b = ax.bar(range(4), cpu, color=mb_colors, **bar_kw)
ax.set_title("(c) CPU Aggregation Overhead (concat ms/iter)", fontweight="bold", fontsize=10)
ax.set_ylabel("ms / iter")
ax.set_xticks(range(4)); ax.set_xticklabels(mb_configs, fontsize=8.5)
ax.set_ylim(0, 0.32)
for bb, v in zip(b, cpu):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.004, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

# (d) end-to-end iter time, same 4 configs at batch=8 / 20 steps, FSDP OFF,
# step 1 excluded.  Re-extracted from each run's train_log.txt on 2026-04-22.
ax = axes[1, 1]
e2e = [8.939, 9.121, 8.546, 8.865]
baseline_e2e = e2e[0]
deltas = [v - baseline_e2e for v in e2e]
b = ax.bar(range(4), e2e, color=mb_colors, **bar_kw)
ax.set_title("(d) End-to-End Iter Time (s/step)",
             fontweight="bold", fontsize=10)
ax.set_ylabel("seconds / step")
ax.set_xticks(range(4)); ax.set_xticklabels(mb_configs, fontsize=8.5)
ax.set_ylim(8.0, 9.6)
ax.axhline(baseline_e2e, color=mb_colors[0], linestyle=":", lw=1.0, alpha=0.6)
for i, (bb, v, d) in enumerate(zip(b, e2e, deltas)):
    label = (
        f"{v:.3f}s" if i == 0
        else f"{v:.3f}s\n({100*d/baseline_e2e:+.1f}%)"
    )
    ax.text(bb.get_x()+bb.get_width()/2, v+0.03, label,
            ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig("7_eval.png", dpi=150, bbox_inches="tight")
print("Saved 7_eval.png")
