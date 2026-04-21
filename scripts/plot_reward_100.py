"""Regenerate 7_reward_curve_controlled.png from the 100-step batch=32 runs."""
import re, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

step_re   = re.compile(r"\bstep:(\d+)\b")
reward_re = re.compile(r"critic/rewards/mean:([\d.\-e]+)")
time_re   = re.compile(r"timing_s/step:([\d.\-e]+)")

logs = {
    "Baseline (push)": "checkpoints/verl_examples/gsm8k_2gpu_baseline_100/train_log.txt",
    "+LP+FP16":        "checkpoints/verl_examples/gsm8k_2gpu_fp16_100/train_log.txt",
    "+LP+INT8":        "checkpoints/verl_examples/gsm8k_2gpu_int8_100/train_log.txt",
}
colors = {
    "Baseline (push)": "#4878CF",
    "+LP+FP16":        "#D65F5F",
    "+LP+INT8":        "#B47CC7",
}

def rolling(vals, w=7):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(sum(vals[lo:i+1]) / (i - lo + 1))
    return out

data = {}
for label, path in logs.items():
    steps, rewards, times = [], [], []
    try:
        for line in open(path):
            if "timing_s/step" not in line:
                continue
            sm = step_re.search(line)
            rm = reward_re.search(line)
            tm = time_re.search(line)
            if sm and rm and tm and int(sm.group(1)) >= 2:
                steps.append(int(sm.group(1)))
                rewards.append(float(rm.group(1)))
                times.append(float(tm.group(1)))
    except FileNotFoundError:
        print(f"WARNING: {path} not found, skipping {label}")
        continue
    data[label] = (steps, rewards, times)
    print(f"{label}: n={len(steps)}, mean_reward={statistics.mean(rewards):.4f}, "
          f"mean_time={statistics.mean(times):.3f}s")

if not data:
    print("No data found, exiting.")
    sys.exit(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Training Stability — Controlled Ablation (100 Steps)\n"
    "(2 GPU, Qwen2.5-1.5B-Instruct, batch=32, FSDP offload OFF)",
    fontsize=11, fontweight="bold",
)

for label, (steps, rewards, times) in data.items():
    c = colors[label]
    mu_r = statistics.mean(rewards)
    mu_t = statistics.mean(times)
    ax1.scatter(steps, rewards, color=c, alpha=0.18, s=12)
    ax1.plot(steps, rolling(rewards), color=c, lw=2,
             label=f"{label} (μ={mu_r:.3f})")
    ax2.scatter(steps, times, color=c, alpha=0.18, s=12)
    ax2.plot(steps, rolling(times), color=c, lw=2,
             label=f"{label} (μ={mu_t:.2f}s)")
    ax2.axhline(mu_t, color=c, linestyle="--", lw=1, alpha=0.6)

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Reward (critic/rewards/mean)")
ax1.set_title("(a) Reward Curve")
ax1.legend(fontsize=8.5)
ax1.grid(alpha=0.3)

ax2.set_xlabel("Training Step")
ax2.set_ylabel("Seconds / Step")
ax2.set_title("(b) Iteration Time")
ax2.legend(fontsize=8.5)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = "7_reward_curve_controlled.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")

# Summary table
print("\n── Summary ──")
base_t = statistics.mean(data["Baseline (push)"][2]) if "Baseline (push)" in data else None
for label, (steps, rewards, times) in data.items():
    mu_t = statistics.mean(times)
    delta_str = ""
    if base_t:
        d = (mu_t - base_t) / base_t * 100
        delta_str = f"  ({'+' if d>=0 else ''}{d:.1f}% vs baseline)"
    print(f"  {label:<20}  reward={statistics.mean(rewards):.4f}  time={mu_t:.3f}s{delta_str}")
